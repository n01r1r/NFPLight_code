"""FP32 common-marker registration and DNG burst preparation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry import area_resize, detect_marker_geometry, fit_homography_lstsq, warp_perspective
from .raw import libraw_ahd_reference, sha256_file, unpack_dng


MAX_MARKER_RMS_PX = 1.0
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


def _record_geometry(record: Mapping[str, Any]) -> tuple[dict[int, np.ndarray], np.ndarray]:
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
    if len(ids) < 4:
        raise ValueError("at least four detected marker tags are required")

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
        "within_rms_limit": bool(residual_rms <= np.float32(MAX_MARKER_RMS_PX)),
    }


def register_marker_frames_to_first(
    geometries: Sequence[Mapping[str, Any]],
    *,
    warp_size: int = 512,
    crop_margin: int = 46,
    output_size: int = 256,
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
        )
        alignments.append(alignment)
    return alignments


def measure_pair_marker_residual(
    near_geometry: Mapping[str, Any],
    far_geometry: Mapping[str, Any],
    *,
    warp_size: int = 512,
    crop_margin: int = 46,
    output_size: int = 256,
) -> dict[str, Any]:
    """Measure independently detected near/far marker alignment in output pixels.

    Each side's existing detector homography is applied as-is.  This diagnostic
    never changes either homography or registers one side to the other.
    """
    if warp_size <= 2 * crop_margin or output_size <= 0:
        raise ValueError("invalid output geometry")
    near_corners, near_homography = _record_geometry(near_geometry)
    far_corners, far_homography = _record_geometry(far_geometry)
    shared_ids = sorted(near_corners.keys() & far_corners.keys())
    if len(shared_ids) < 4:
        return {
            "available": False,
            "reason": f"need at least four common marker tags, found {len(shared_ids)}",
            "shared_marker_ids": shared_ids,
            "shared_corner_count": 4 * len(shared_ids),
            "residuals_px": None,
            "residual_rms_px": None,
            "residual_max_px": None,
        }
    near_points = np.concatenate([near_corners[tag_id] for tag_id in shared_ids]).astype(np.float32, copy=False)
    far_points = np.concatenate([far_corners[tag_id] for tag_id in shared_ids]).astype(np.float32, copy=False)
    near_output = _to_final_pixels(_project_points(near_homography, near_points), warp_size, crop_margin, output_size)
    far_output = _to_final_pixels(_project_points(far_homography, far_points), warp_size, crop_margin, output_size)
    residuals = np.ascontiguousarray(near_output - far_output, dtype=np.float32)
    squared_distance = np.sum(residuals * residuals, axis=1, dtype=np.float32)
    residual_lengths = np.sqrt(squared_distance)
    return {
        "available": True,
        "reason": None,
        "shared_marker_ids": shared_ids,
        "shared_corner_count": int(len(near_points)),
        "residuals_px": residuals.tolist(),
        "residual_rms_px": float(np.sqrt(squared_distance.mean(dtype=np.float32))),
        "residual_max_px": float(residual_lengths.max()),
    }


_EXIF_EXPOSURE_TAGS = {
    33434: "exposure_time",
    33437: "f_number",
    34855: "iso_speed_ratings",
    37386: "focal_length",
}


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


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _warp_stages(image: np.ndarray, homography: np.ndarray) -> dict[str, Any]:
    warped, valid = warp_perspective(image, homography, (512, 512), return_valid=True)
    crop = warped[46:466, 46:466]
    valid_crop = valid[46:466, 46:466]
    output = area_resize(crop, (256, 256))
    return {
        "warp512": warped,
        "valid512": valid,
        "crop420": crop,
        "valid420": valid_crop,
        "output256": output,
        "geometry": {
            "warp_size": 512,
            "crop_margin": 46,
            "crop_shape": [420, 420],
            "output_size": 256,
            "interpolation": "float32 inverse bilinear warp; exact pixel-area resize",
            "valid_fraction": float(valid_crop.mean()),
        },
    }


def _save_burst_frame_archive(path: Path, capture: Any, stages: dict[str, Any],
                              libraw_demosaic: np.ndarray, libraw_stages: dict[str, Any],
                              difference: np.ndarray) -> None:
    arrays = {
        "bayer_u16": capture.bayer_u16,
        "bayer_selected_dtype": capture.bayer_float,
        "demosaic_rgb_selected_dtype": capture.rgb_float,
        "warp512": stages["warp512"],
        "crop420": stages["crop420"],
        "output256": stages["output256"],
        "valid512": stages["valid512"].astype(np.uint8),
        "valid420": stages["valid420"].astype(np.uint8),
        "libraw_demosaic_rgb_selected_dtype": libraw_demosaic,
        "libraw_output256": libraw_stages["output256"],
        "libraw_difference256": difference,
    }
    for name, value in arrays.items():
        if np.issubdtype(np.asarray(value).dtype, np.floating) and not np.isfinite(value).all():
            raise FloatingPointError(f"non-finite source-frame array {name}: {path}")
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays):
            raise AssertionError(f"source-frame NPZ keys changed: {path}")
        for name, value in arrays.items():
            if not np.array_equal(archive[name], value):
                raise AssertionError(f"source-frame NPZ round-trip mismatch: {path.name}/{name}")


def _save_raw_only_burst_frame(path: Path, capture: Any) -> None:
    arrays = {
        "bayer_u16": capture.bayer_u16,
        "bayer_selected_dtype": capture.bayer_float,
        "demosaic_rgb_selected_dtype": capture.rgb_float,
    }
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(not np.array_equal(archive[key], value) for key, value in arrays.items()):
            raise AssertionError(f"raw-only source-frame NPZ round-trip mismatch: {path.name}")


def _metadata_signature(raw_metadata: Mapping[str, Any], capture_settings: Any,
                        dng_exif: Mapping[str, Any]) -> dict[str, Any]:
    signature = {field: raw_metadata.get(field) for field in BURST_METADATA_FIELDS if field != "exposure_settings"}
    signature["exposure_settings"] = {
        "capture_settings": capture_settings,
        "dng_exif": dict(dng_exif),
    }
    return signature


def prepare_burst(source: str | Path, output: str | Path) -> dict[str, Any]:
    """Prepare every DNG in one near/far burst without averaging or inference.

    Aligned FP32 source archives are written once per frame. The returned
    manifest reports the quantitative averaging gate, but does not create a
    mean: a separate visual review must accept texture registration first.
    """
    source = Path(source).resolve()
    output = Path(output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"capture source directory does not exist: {source}")
    if output == source or source in output.parents:
        raise ValueError("preparation output must not be inside the source capture")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already contains files: {output}")
    try:
        capture_metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"source metadata.json is required: {source / 'metadata.json'}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source metadata JSON: {source / 'metadata.json'}") from exc
    if not isinstance(capture_metadata, dict):
        raise ValueError("capture metadata must be a JSON object")

    paths: dict[str, list[Path]] = {}
    for side in ("near", "far"):
        section = capture_metadata.get(side)
        names = section.get("filenames") if isinstance(section, Mapping) else None
        if not isinstance(names, list) or len(names) != 5:
            raise ValueError(f"metadata must list exactly five {side} DNG filenames")
        if section.get("frameCount") not in (None, 5):
            raise ValueError(f"{side}.frameCount must be five")
        paths[side] = [source / _safe_burst_filename(name, side) for name in names]
        missing = [str(path) for path in paths[side] if not path.is_file()]
        if missing:
            raise FileNotFoundError("capture DNG is missing: " + ", ".join(missing))

    source_metadata_path = source / "metadata.json"
    metadata_hash_before = sha256_file(source_metadata_path)
    hash_before = {
        f"{side}/{path.name}": sha256_file(path)
        for side, frame_paths in paths.items() for path in frame_paths
    }
    core_paths = {
        name: Path(__file__).with_name(name)
        for name in ("raw.py", "demosaic.py", "geometry.py", "burst.py")
    }
    implementation_hash_before = {name: sha256_file(path) for name, path in core_paths.items()}

    output.mkdir(parents=True, exist_ok=True)
    frame_dir = output / "source_frames"
    frame_dir.mkdir(exist_ok=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": str(source),
        "source_name": source.name,
        "capture_metadata": capture_metadata,
        "precision": "float32",
        "frame_count_per_side": 5,
        "spatial_pipeline": "source AHD RGB -> one FP32 Hfinal inverse-bilinear warp512 -> crop46/420 -> exact-area resize256",
        "spatial_reference": "near_00 marker plane for both near and far",
        "status": "preparing",
        "averaging": {
            "quantitative_eligible": False,
            "mean_generated": False,
            "awaiting_visual_review": True,
            "selected_indices": {"near": [0], "far": [0]},
            "reason": None,
        },
        "sides": {},
        "near_far_pair_residuals": {},
        "capture_metadata_sha256": metadata_hash_before,
        "implementation_sha256_before": implementation_hash_before,
        "implementation_sha256_after": None,
        "verification": {
            "input_file_hashes_before": hash_before,
            "input_file_hashes_after": None,
            "source_hashes_unchanged": None,
            "implementation_unchanged": None,
            "archives_round_trip_checked": True,
            "source_metadata_unchanged": None,
        },
    }

    for side in ("near", "far"):
        manifest["sides"][side] = {
            "metadata": [],
            "metadata_compatible": False,
            "metadata_mismatches": [],
            "averaging_eligible": False,
            "selected_indices": [0],
            "fallback_reason": None,
            "frames": [],
        }
    _write_json(output / "preparation.json", manifest)

    first_geometries: dict[str, dict[str, Any] | None] = {"near": None, "far": None}
    signatures: dict[str, list[dict[str, Any] | None]] = {side: [None] * 5 for side in ("near", "far")}
    quant_gate: dict[str, bool] = {"near": True, "far": True}

    for side in ("near", "far"):
        side_record = manifest["sides"][side]
        side_record["metadata"] = [None] * 5
        section = capture_metadata[side]
        capture_settings = section.get("settings")
        for index, path in enumerate(paths[side]):
            archive_rel = Path("source_frames") / f"{side}_{index:02d}.npz"
            frame: dict[str, Any] = {
                "side": side,
                "index": index,
                "filename": path.name,
                "sha256": None,
                "archive": str(archive_rel),
                "archive_sha256": None,
                "geometry": None,
                "alignment": None,
                "processing": None,
                "metadata_compatible": False,
                "metadata_error": None,
                "error": None,
            }
            side_record["frames"].append(frame)
            capture = None
            geometry = None
            alignment = None
            stages = None
            try:
                capture = unpack_dng(path, dtype=np.float32)
                frame["sha256"] = capture.sha256
                black = capture.metadata["black_level_per_channel"]
                white = int(capture.metadata["white_level"])
                if not black or len(set(black)) != 1 or not 0 <= int(black[0]) < white:
                    raise ValueError("DNG black/white levels must be equal-channel with 0 <= B < W")
                exif = read_dng_exposure_exif(path)
                signature = _metadata_signature(capture.metadata, capture_settings, exif)
                signatures[side][index] = signature
                annotated_metadata = dict(capture.metadata)
                annotated_metadata["capture_settings"] = capture_settings
                annotated_metadata["dng_exif"] = exif
                side_record["metadata"][index] = annotated_metadata

                _, geometry = detect_marker_geometry(capture.rgb_float)
                geometry = dict(geometry)
                geometry.update(sha256=capture.sha256, raw_shape=list(capture.bayer_u16.shape))
                frame["geometry"] = geometry
                if index == 0:
                    first_geometries[side] = geometry
                # Both distances must sample the same material coordinates.
                # Independent inner-hole detections can choose different scales.
                reference_geometry = first_geometries["near"]
                if side == "near" and index == 0:
                    alignment = register_marker_frames_to_first([geometry])[0]
                elif reference_geometry is None:
                    raise ValueError("near frame 00 marker geometry is unavailable; no common reference exists")
                else:
                    alignment = register_marker_frames_to_first([reference_geometry, geometry])[1]
                alignment["reference_side"] = "near"
                alignment["reference_index"] = 0
                frame["alignment"] = alignment

                homography = np.asarray(alignment["homography_final"], dtype=np.float32)
                stages = _warp_stages(capture.rgb_float, homography)
                libraw_counts = libraw_ahd_reference(path).astype(np.float32)
                if libraw_counts.shape != capture.rgb_float.shape:
                    raise ValueError("LibRaw comparison shape differs from visible Bayer AHD output")
                libraw_stages = _warp_stages(libraw_counts, homography)
                difference = np.ascontiguousarray(stages["output256"] - libraw_stages["output256"], dtype=np.float32)
                _save_burst_frame_archive(
                    output / archive_rel, capture, stages, libraw_counts, libraw_stages, difference,
                )
                frame["archive_sha256"] = sha256_file(output / archive_rel)
                valid_fraction = float(stages["valid420"].mean())
                frame["processing"] = {
                    **stages["geometry"],
                    "spatial_homography": "homography_final",
                    "source_dtype": str(capture.rgb_float.dtype),
                    "output_dtype": str(stages["output256"].dtype),
                    "valid420_all": bool(stages["valid420"].all()),
                    "valid420_fraction": valid_fraction,
                    "ahd_libraw_mae": float(np.mean(np.abs(capture.rgb_float - libraw_counts), dtype=np.float32)),
                    "ahd_libraw_max_abs": float(np.max(np.abs(capture.rgb_float - libraw_counts))),
                }
                if not alignment["within_rms_limit"]:
                    quant_gate[side] = False
                if not stages["valid420"].all():
                    quant_gate[side] = False
                if not exif["known"]:
                    quant_gate[side] = False
            except Exception as exc:
                frame["error"] = f"{type(exc).__name__}: {exc}"
                quant_gate[side] = False
                # When detection or registration fails but a detected H exists,
                # preserve an independent spatial diagnostic, never an aligned candidate.
                if capture is not None and geometry is not None and frame["processing"] is None:
                    try:
                        independent_h = np.asarray(geometry["homography"], dtype=np.float32)
                        stages = _warp_stages(capture.rgb_float, independent_h)
                        libraw_counts = libraw_ahd_reference(path).astype(np.float32)
                        libraw_stages = _warp_stages(libraw_counts, independent_h)
                        difference = np.ascontiguousarray(stages["output256"] - libraw_stages["output256"], dtype=np.float32)
                        _save_burst_frame_archive(
                            output / archive_rel, capture, stages, libraw_counts, libraw_stages, difference,
                        )
                        frame["archive_sha256"] = sha256_file(output / archive_rel)
                        frame["processing"] = {
                            **stages["geometry"],
                            "spatial_homography": "detected_independent_diagnostic_only",
                            "source_dtype": str(capture.rgb_float.dtype),
                            "output_dtype": str(stages["output256"].dtype),
                            "valid420_all": bool(stages["valid420"].all()),
                            "valid420_fraction": float(stages["valid420"].mean()),
                        }
                    except Exception as diagnostic_error:
                        frame["error"] += f"; diagnostic archive failed: {type(diagnostic_error).__name__}: {diagnostic_error}"
                if capture is not None and not (output / archive_rel).is_file():
                    try:
                        _save_raw_only_burst_frame(output / archive_rel, capture)
                        frame["archive_sha256"] = sha256_file(output / archive_rel)
                    except Exception as archive_error:
                        frame["error"] += f"; raw archive failed: {type(archive_error).__name__}: {archive_error}"
            finally:
                if not (output / archive_rel).is_file():
                    frame["archive"] = None
                    frame["archive_sha256"] = None
                _write_json(output / "preparation.json", manifest)

        reference_signature = signatures[side][0]
        mismatch = []
        for index, signature in enumerate(signatures[side]):
            try:
                if reference_signature is None:
                    raise ValueError("frame 00 metadata is unavailable")
                if signature is None:
                    raise ValueError(f"frame {index:02d} metadata is unavailable")
                validate_burst_metadata(reference_signature, signature)
                side_record["frames"][index]["metadata_compatible"] = True
            except (TypeError, ValueError) as exc:
                mismatch.append({"index": index, "reason": str(exc)})
                side_record["frames"][index]["metadata_error"] = str(exc)
                quant_gate[side] = False
        side_record["metadata_mismatches"] = mismatch
        side_record["metadata_compatible"] = bool(signatures[side]) and not mismatch
        side_record["averaging_eligible"] = bool(quant_gate[side] and len(side_record["frames"]) == 5
                                                  and all(frame["error"] is None for frame in side_record["frames"]))
        if side_record["averaging_eligible"]:
            side_record["selected_indices"] = [0, 1, 2, 3, 4]
            side_record["fallback_reason"] = None
        else:
            first = side_record["frames"][0] if side_record["frames"] else {}
            side_record["selected_indices"] = (
                [0] if first.get("processing") and first["processing"].get("valid420_all")
                and first.get("error") is None and first.get("archive_sha256") else []
            )
            reasons = [item["reason"] for item in mismatch]
            reasons.extend(frame["error"] for frame in side_record["frames"] if frame["error"])
            reasons.extend(
                f"frame {frame['index']} marker RMS {frame['alignment']['residual_rms_px']:.6g}px exceeds {MAX_MARKER_RMS_PX:g}px"
                for frame in side_record["frames"] if frame["alignment"] and not frame["alignment"]["within_rms_limit"]
            )
            reasons.extend(
                f"frame {frame['index']} crop valid fraction {frame['processing']['valid420_fraction']:.6g} is below 1"
                for frame in side_record["frames"] if frame["processing"] and not frame["processing"]["valid420_all"]
            )
            if not reasons:
                reasons.append("one or more frames are missing a valid marker alignment, spatial archive, or known exposure metadata")
            side_record["fallback_reason"] = reasons
        _write_json(output / "preparation.json", manifest)

    for index in range(5):
        near, far = manifest["sides"]["near"]["frames"][index], manifest["sides"]["far"]["frames"][index]
        near_geometry, far_geometry = near["geometry"], far["geometry"]
        if near_geometry is None or far_geometry is None or near["alignment"] is None or far["alignment"] is None:
            manifest["near_far_pair_residuals"][str(index)] = {
                "available": False, "reason": "near or far frame lacks marker geometry/alignment",
                "shared_marker_ids": [], "shared_corner_count": 0,
                "residuals_px": None, "residual_rms_px": None, "residual_max_px": None,
            }
            continue
        near_final, far_final = dict(near_geometry), dict(far_geometry)
        near_final["homography"] = near["alignment"]["homography_final"]
        far_final["homography"] = far["alignment"]["homography_final"]
        manifest["near_far_pair_residuals"][str(index)] = measure_pair_marker_residual(near_final, far_final)

    for side in ("near", "far"):
        manifest["sides"][side]["averaging_eligible"] &= manifest["sides"][side]["metadata_compatible"]
    first_pair_usable = all(
        bool(manifest["sides"][side]["frames"])
        and manifest["sides"][side]["frames"][0]["processing"] is not None
        and manifest["sides"][side]["frames"][0]["processing"]["valid420_all"]
        and manifest["sides"][side]["frames"][0]["error"] is None
        and manifest["sides"][side]["frames"][0]["archive_sha256"] is not None
        for side in ("near", "far")
    )
    both_eligible = first_pair_usable and all(
        manifest["sides"][side]["averaging_eligible"] for side in ("near", "far")
    )
    if not both_eligible:
        for side in ("near", "far"):
            side_record = manifest["sides"][side]
            first = side_record["frames"][0] if side_record["frames"] else {}
            first_usable = (first.get("processing") and first["processing"].get("valid420_all")
                            and first.get("error") is None and first.get("archive_sha256"))
            side_record["selected_indices"] = [0] if first_usable and first_pair_usable else []
            if side_record["averaging_eligible"]:
                side_record["fallback_reason"] = ["the paired near/far burst did not pass both side gates"]
    manifest["averaging"].update(
        quantitative_eligible=bool(both_eligible),
        selected_indices={side: list(manifest["sides"][side]["selected_indices"]) for side in ("near", "far")},
        reason=None if both_eligible else {
            side: manifest["sides"][side]["fallback_reason"]
            for side in ("near", "far") if not manifest["sides"][side]["averaging_eligible"]
        },
    )
    manifest["status"] = "awaiting_visual_review" if both_eligible else "prepared_single_frame_fallback"
    if not first_pair_usable:
        manifest["status"] = "failed_first_frame"
        manifest["averaging"]["selected_indices"] = {"near": [], "far": []}
        manifest["averaging"]["reason"] = "frame00 did not produce a fully supported spatial archive for both sides"
    source_hash_after = {
        f"{side}/{path.name}": sha256_file(path)
        for side, frame_paths in paths.items() for path in frame_paths
    }
    implementation_hash_after = {name: sha256_file(path) for name, path in core_paths.items()}
    verification = manifest["verification"]
    verification["input_file_hashes_after"] = source_hash_after
    verification["source_hashes_unchanged"] = source_hash_after == hash_before
    manifest["implementation_sha256_after"] = implementation_hash_after
    verification["implementation_unchanged"] = implementation_hash_after == implementation_hash_before
    verification["source_metadata_unchanged"] = sha256_file(source_metadata_path) == metadata_hash_before
    if not verification["source_hashes_unchanged"] or not verification["source_metadata_unchanged"]:
        manifest["status"] = "failed_source_changed"
    if not verification["implementation_unchanged"]:
        manifest["status"] = "failed_implementation_changed_during_preparation"
    _write_json(output / "preparation.json", manifest)
    return manifest


__all__ = [
    "BURST_METADATA_FIELDS",
    "MAX_MARKER_RMS_PX",
    "measure_pair_marker_residual",
    "prepare_burst",
    "read_dng_exposure_exif",
    "register_marker_frames_to_first",
    "validate_burst_metadata",
]
