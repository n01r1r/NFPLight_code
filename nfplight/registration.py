from __future__ import annotations
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image
from nfplight.geometry import fit_homography_lstsq

BURST_METADATA_FIELDS = (
    "raw_shape",
    "cfa_pattern",
    "visible_cfa_indices",
    "cfa_rgb_indices",
    "color_desc",
    "visible_origin",
    "black_level_per_channel",
    "white_level",
    "camera_whitebalance",
    "color_matrix",
    "orientation_flip",
    "exposure_settings",
)
_EXIF_EXPOSURE_TAGS = {
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso_speed_ratings",
    37386: "focal_length",
}

def _same_metadata_value(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or left.keys() != right.keys():
            return False
        return all(_same_metadata_value(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) != len(right):
            return False
        return all(_same_metadata_value(a, b) for a, b in zip(left, right))
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False

def validate_burst_metadata(reference: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    """Require identical sensor, calibration, and exposure signatures.

    Signatures are assembled by the caller from rawpy metadata and the capture
    ``metadata.json`` settings.  Array-valued entries may be lists or NumPy
    arrays; comparison is exact and does not silently coerce precision.
    """
    if not isinstance(reference, Mapping) or not isinstance(current, Mapping):
        raise TypeError("burst metadata signatures must be mappings")
    for signature in (reference.get("exposure_settings"), current.get("exposure_settings")):
        exif = signature.get("dng_exif") if isinstance(signature, Mapping) else None
        if not isinstance(exif, Mapping) or exif.get("known") is not True:
            raise ValueError("burst metadata has unknown DNG exposure EXIF")
    for field in BURST_METADATA_FIELDS:
        if field not in reference or field not in current:
            raise ValueError(f"burst metadata is missing required field {field!r}")
        if not _same_metadata_value(reference[field], current[field]):
            raise ValueError(f"burst metadata mismatch for {field}")

def _record_geometry(record: Mapping[str, Any], min_tags = 4) -> tuple[dict[int, np.ndarray], np.ndarray]:
    if not isinstance(record, Mapping):
        raise TypeError("marker geometry must be a mapping")
    marker_ids = record.get("marker_ids")
    corner_map = record.get("marker_corners_raw")
    if not isinstance(marker_ids, Sequence) or isinstance(marker_ids, (str, bytes)):
        raise ValueError("marker geometry must contain marker_ids")
    if not isinstance(corner_map, Mapping):
        raise ValueError("marker geometry must contain marker_corners_raw")
    ids = [int(marker_id) for marker_id in marker_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("marker geometry contains duplicate IDs")
    if len(ids) < min_tags:
        raise ValueError(f'at least {min_tags} detected marker tags are required')

    corners: dict[int, np.ndarray] = {}
    for marker_id in ids:
        value = corner_map.get(str(marker_id))
        if value is None:
            raise ValueError(f"marker geometry has no corners for ID {marker_id}")
        if isinstance(value, np.ndarray) and value.dtype != np.dtype(np.float32):
            raise TypeError("marker corners must use float32 coordinates")
        points = np.asarray(value, dtype=np.float32)
        if points.shape != (4, 2) or not np.isfinite(points).all():
            raise ValueError(f"marker {marker_id} corners must be finite float32 [4,2]")
        corners[marker_id] = np.ascontiguousarray(points, dtype=np.float32)

    homography_value = record.get("homography")
    if isinstance(homography_value, np.ndarray) and homography_value.dtype != np.dtype(np.float32):
        raise TypeError("detected homography must use float32")
    homography = np.asarray(homography_value, dtype=np.float32)
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        raise ValueError("detected homography must be finite float32 [3,3]")
    if abs(float(homography[2, 2])) <= np.finfo(np.float32).eps:
        raise ValueError("detected homography has invalid projective scale")
    homography = np.ascontiguousarray(homography / homography[2, 2], dtype=np.float32)
    return corners, homography

def _project_points(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    mapped = homogeneous @ homography.T
    denominator = mapped[:, 2]
    if not np.isfinite(mapped).all() or np.any(np.abs(denominator) <= 1e-6):
        raise ValueError("homography maps a shared marker corner to infinity")
    projected = mapped[:, :2] / denominator[:, None]
    if not np.isfinite(projected).all():
        raise ValueError("homography produced non-finite marker coordinates")
    return np.ascontiguousarray(projected, dtype=np.float32)

def _to_final_pixels(points512: np.ndarray, warp_size: int, crop_margin: int, output_size: int) -> np.ndarray:
    scale = np.float32(output_size) / np.float32(warp_size - 2 * crop_margin)
    return np.ascontiguousarray((points512 - np.float32(crop_margin)) * scale, dtype=np.float32)

def _alignment_record(
    reference_corners: dict[int, np.ndarray],
    reference_homography: np.ndarray,
    current_corners: dict[int, np.ndarray],
    current_homography: np.ndarray,
    *,
    warp_size: int,
    crop_margin: int,
    output_size: int,
    max_rms: float,
) -> dict[str, Any]:
    shared_ids = sorted(reference_corners.keys() & current_corners.keys())
    if len(shared_ids) < 4:
        raise ValueError(f"need at least four common marker tags, found {len(shared_ids)}")
    source = np.concatenate([current_corners[tag_id] for tag_id in shared_ids]).astype(np.float32, copy=False)
    destination = np.concatenate([reference_corners[tag_id] for tag_id in shared_ids]).astype(np.float32, copy=False)
    relative = fit_homography_lstsq(source, destination)
    final = np.ascontiguousarray(reference_homography @ relative, dtype=np.float32)
    if abs(float(final[2, 2])) <= np.finfo(np.float32).eps:
        raise ValueError("composed homography has invalid projective scale")
    final /= final[2, 2]

    current_warp = _project_points(final, source)
    reference_warp = _project_points(reference_homography, destination)
    current_output = _to_final_pixels(current_warp, warp_size, crop_margin, output_size)
    reference_output = _to_final_pixels(reference_warp, warp_size, crop_margin, output_size)
    residuals = np.ascontiguousarray(current_output - reference_output, dtype=np.float32)
    squared_distance = np.sum(residuals * residuals, axis=1, dtype=np.float32)
    residual_rms = np.sqrt(squared_distance.mean(dtype=np.float32))
    residual_lengths = np.sqrt(squared_distance)
    return {
        "shared_marker_ids": shared_ids,
        "shared_corner_count": int(len(source)),
        "homography_relative": relative.tolist(),
        "homography_final": final.tolist(),
        "residuals_px": residuals.tolist(),
        "residual_rms_px": float(residual_rms),
        "residual_max_px": float(residual_lengths.max()),
        "within_rms_limit": bool(residual_rms <= np.float32(max_rms)),
    }

def register_marker_frames_to_first(
    geometries: Sequence[Mapping[str, Any]],
    *,
    warp_size: int = 512,
    crop_margin: int = 46,
    output_size: int = 256,
    max_rms: float = 1.0,
) -> list[dict[str, Any]]:
    """Register every frame's source coordinates to frame zero's marker plane.

    Each geometry record is the output of ``detect_marker_geometry``.  The
    first record supplies H0 and receives an identity alignment.  Later frames
    use all four corners for every shared marker ID in one normalized FP32
    least-squares fit.  Returned ``homography_final`` maps each original
    source image directly to the reference 512 plane; no image is warped here.
    Outliers remain in the fit and are reflected in per-corner/output-pixel
    residuals so the caller can reject the burst without dropping frames.
    """
    if not isinstance(geometries, Sequence) or isinstance(geometries, (str, bytes)) or not geometries:
        raise ValueError("a marker burst must contain a reference frame")
    if warp_size <= 2 * crop_margin or output_size <= 0:
        raise ValueError("invalid output geometry")

    parsed = [_record_geometry(record) for record in geometries]
    reference_corners, reference_homography = parsed[0]
    reference_ids = sorted(reference_corners)
    identity = np.eye(3, dtype=np.float32)
    first = {
        "shared_marker_ids": reference_ids,
        "shared_corner_count": int(sum(len(reference_corners[tag]) for tag in reference_ids)),
        "homography_relative": identity.tolist(),
        "homography_final": reference_homography.tolist(),
        "residuals_px": np.zeros((4 * len(reference_ids), 2), dtype=np.float32).tolist(),
        "residual_rms_px": 0.0,
        "residual_max_px": 0.0,
        "within_rms_limit": True,
    }
    alignments = [first]
    for current_corners, current_homography in parsed[1:]:
        alignment = _alignment_record(
            reference_corners,
            reference_homography,
            current_corners,
            current_homography,
            warp_size=warp_size,
            crop_margin=crop_margin,
            output_size=output_size,
            max_rms=max_rms,
        )
        alignments.append(alignment)
    return alignments

def _json_exif_value(value: Any) -> Any:
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator is not None:
        return {"numerator": int(numerator), "denominator": int(denominator)}
    if isinstance(value, (tuple, list)):
        return [_json_exif_value(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def read_dng_exposure_exif(path: str | Path) -> dict[str, Any]:
    """Read only the EXIF exposure IFD, preserving rational numerator/denominator."""
    with Image.open(path) as image:
        exif = image.getexif()
        ifd = exif.get_ifd(34665) if hasattr(exif, "get_ifd") else {}
    values = {name: _json_exif_value(ifd[tag]) for tag, name in _EXIF_EXPOSURE_TAGS.items() if tag in ifd}
    missing = [name for name in _EXIF_EXPOSURE_TAGS.values() if name not in values]
    return {
        "known": not missing,
        "tags": values,
        "missing": missing,
    }

def _safe_burst_filename(value: Any, side: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value:
        raise ValueError(f"unsafe {side} frame filename: {value!r}")
    path = Path(value)
    if path.name != value or path.is_absolute() or path.suffix.lower() != ".dng":
        raise ValueError(f"unsafe {side} frame filename: {value!r}")
    if not value.lower().startswith(f"{side}_"):
        raise ValueError(f"{side} frame filename has the wrong side prefix: {value!r}")
    return value

def _metadata_signature(raw_metadata: Mapping[str, Any], capture_settings: Any,
                        dng_exif: Mapping[str, Any]) -> dict[str, Any]:
    signature = {field: raw_metadata.get(field) for field in BURST_METADATA_FIELDS if field != "exposure_settings"}
    signature["exposure_settings"] = {
        "capture_settings": capture_settings,
        "dng_exif": dict(dng_exif),
    }
    return signature


def register_target_frame(reference, current, config):
    policy = config['registration']
    minimum = policy['min_shared_tags']
    ref, reference_h = _record_geometry(reference, min_tags = minimum)
    cur, _ = _record_geometry(current, min_tags = minimum)
    ids = sorted(ref.keys() & cur.keys())
    if len(ids) < minimum:
        raise ValueError('insufficient common tags for robust target registration')
    source = np.stack([cur[tag] for tag in ids])
    destination = np.stack([ref[tag] for tag in ids])
    target = _project_points(reference_h, destination.reshape(-1, 2))
    scale = np.float32(config['output_size'] / config['warp_size'])
    threshold = config['max_marker_rms_px']

    def fit(selected):
        relative = (np.eye(3, dtype = np.float32) if np.array_equal(source[selected], destination[selected])
                    else fit_homography_lstsq(source[selected].reshape(-1, 2), destination[selected].reshape(-1, 2)))
        final = np.ascontiguousarray(reference_h @ relative, dtype = np.float32)
        final /= final[2, 2]
        residuals = (_project_points(final, source.reshape(-1, 2)) - target) * scale
        errors = np.sqrt(np.square(residuals).sum(1).reshape(-1, 4).mean(1))
        return relative, final, residuals, errors

    all_tags = np.arange(len(ids))
    initial = fit(all_tags)
    inliers = initial[3] <= threshold
    best = (int(inliers.sum()), -float(initial[3][inliers].mean()) if inliers.any() else -float('inf'), inliers)
    if not inliers.all():
        subsets = list(combinations(range(len(ids)), minimum))
        chosen = np.linspace(0, len(subsets) - 1, min(len(subsets), policy['max_trials']), dtype = int)
        for index in chosen:
            try:
                result = fit(np.asarray(subsets[index]))
            except (ValueError, np.linalg.LinAlgError):
                continue
            accepted = result[3] <= threshold
            score = (int(accepted.sum()), -float(result[3][accepted].mean()) if accepted.any() else -float('inf'))
            if score > best[:2]:
                best = (*score, accepted)
        inliers = best[2]
    fallback = inliers.sum() < minimum or 1 - inliers.mean() > policy['max_rejected_fraction']
    if fallback:
        inliers = np.ones(len(ids), dtype = bool)
    relative, final, residuals, errors = fit(np.flatnonzero(inliers))
    if not fallback and (errors[inliers] > threshold).any():
        fallback = True
        inliers = np.ones(len(ids), dtype = bool)
        relative, final, residuals, errors = initial
    accepted_target = target.reshape(-1, 4, 2)[inliers].reshape(-1, 2)
    spans = np.ptp(accepted_target, axis = 0) / np.float32(config['warp_size'])
    if (spans < policy['min_axis_span']).any():
        raise ValueError('common tags do not span enough of both target axes')
    hull = cv2.convexHull(accepted_target).reshape(-1, 2)
    following = np.roll(hull, -1, axis = 0)
    area = float(np.abs(np.sum(hull[:, 0] * following[:, 1] - hull[:, 1] * following[:, 0], dtype = np.float32)) * .5 / config['warp_size'] ** 2)
    if area < policy['min_axis_span'] ** 2:
        raise ValueError('common tag footprint is too narrow to constrain the target plane')
    accepted_residuals = residuals.reshape(-1, 4, 2)[inliers].reshape(-1, 2)
    rms = float(np.sqrt(np.square(accepted_residuals).sum(1).mean()))
    return {'shared_marker_ids': ids, 'used_marker_ids': [tag for tag, use in zip(ids, inliers) if use],
        'excluded_marker_ids': [tag for tag, use in zip(ids, inliers) if not use],
        'shared_corner_count': len(ids) * 4, 'homography_relative': relative.tolist(),
        'homography_final': final.tolist(), 'residuals_px': residuals.tolist(),
        'tag_rms_px': dict(zip(map(str, ids), map(float, errors))),
        'residual_rms_px': rms, 'residual_max_px': float(np.linalg.norm(accepted_residuals, axis = 1).max()),
        'within_rms_limit': bool(not fallback and rms <= threshold), 'axis_span_fraction': spans.tolist(),
        'marker_hull_area_fraction': area, 'fit_status': 'all_tags_fallback_no_average' if fallback else 'consistent_tags'}
