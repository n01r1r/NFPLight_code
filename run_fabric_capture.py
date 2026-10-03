"""Run the canonical float32 fabric DNG capture comparison and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from capture_processing import (
    CONDITIONS,
    apply_compensation,
    apply_linear_color,
    read_color_transform,
)
from capture_processing.burst import prepare_burst
from capture_processing.report import (
    write_capture_index,
    write_capture_report,
    write_prepare_review,
)
from capture_processing.raw import sha256_file
from model.author_real_capture_adapter import (
    AuthorRealCaptureAdapter,
    GEOMETRY_BUFFER_DTYPE,
    GEOMETRY_SETUP_DTYPE,
    FEATURE_VERSION as AUTHOR_FEATURE_VERSION,
    NORMAL_HEAD as AUTHOR_NORMAL_HEAD,
)


AUTHORIZED_CHECKPOINT_SHA256 = {
    "net_g_real.pth": "75f23d38f1298e635d51b98cd82389117d124018cef1c2bae123689198f726b9",
}
CAPTURE_GROUPS = (
    "260818_174845_478",
    "260818_175005_078",
    "260923_141241_844",
    "260923_161825_503",
    "260930_152732_008",
)
DEFAULT_SOURCE = Path("data/260930_152732_008")
DEFAULT_CHECKPOINT = Path("checkpoints/net_g_real.pth")
DEFAULT_OUTPUT = Path("artifacts/fabric_capture_20261001_author_original")
INFERENCE_FORMAT = "nfplight.fabric-capture.v5"
PREPARATION_CORE_FILES = (
    Path("capture_processing/raw.py"),
    Path("capture_processing/demosaic.py"),
    Path("capture_processing/geometry.py"),
    Path("capture_processing/burst.py"),
)
PREPARED_FRAME_KEYS = {
    "bayer_u16", "bayer_selected_dtype", "demosaic_rgb_selected_dtype",
    "warp512", "crop420", "output256", "valid512", "valid420",
    "libraw_demosaic_rgb_selected_dtype", "libraw_output256", "libraw_difference256",
}


def _safe_frame_filename(value: Any, side: str) -> str:
    """Validate a metadata-provided DNG name before joining it to ``source``."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{side} frame filename must be a non-empty string")
    # Path.name alone is insufficient on Windows when metadata contains a
    # POSIX separator, so reject both separator forms explicitly.
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ValueError(f"unsafe {side} frame filename: {value!r}")
    path = Path(value)
    if path.name != value or path.is_absolute() or path.suffix.lower() != ".dng":
        raise ValueError(f"unsafe {side} frame filename: {value!r}")
    if not value.lower().startswith(f"{side.lower()}_"):
        raise ValueError(f"{side} frame filename has the wrong side prefix: {value!r}")
    return value


def select_source_frames(source: Path, frame_index: int = 0) -> tuple[dict[str, Path], dict[str, Any]]:
    """Resolve one metadata-declared near/far DNG pair."""
    source = Path(source)
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError("frame_index must be a non-negative integer")

    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"source metadata.json is required to select paired DNG frames: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid source metadata JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("source metadata must be a JSON object")

    filenames: dict[str, list[str]] = {}
    for side in ("near", "far"):
        section = metadata.get(side)
        names = section.get("filenames") if isinstance(section, dict) else None
        if not isinstance(names, list) or not names:
            raise ValueError(f"source metadata must provide a non-empty {side}.filenames list")
        declared_count = section.get("frameCount")
        if declared_count is not None and declared_count != len(names):
            raise ValueError(f"{side}.frameCount does not match filenames length")
        filenames[side] = [_safe_frame_filename(name, side) for name in names]

    if len(filenames["near"]) != len(filenames["far"]):
        raise ValueError("near/far burst filename lists must have the same length")
    if frame_index >= len(filenames["near"]):
        raise IndexError(f"frame_index {frame_index} is outside burst length {len(filenames['near'])}")

    selected = {side: filenames[side][frame_index] for side in ("near", "far")}
    paths = {side: source / selected[side] for side in ("near", "far")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("selected source DNG does not exist: " + ", ".join(missing))

    record = {
        "mode": "burst_single",
        "frame_index": frame_index,
        "frame_count": 1,
        "available_frame_count": len(filenames["near"]),
        "averaged": False,
        "near_filenames": [selected["near"]],
        "far_filenames": [selected["far"]],
        "near_filename": selected["near"],
        "far_filename": selected["far"],
        "near_paths": [str(paths["near"])],
        "far_paths": [str(paths["far"])],
    }
    return paths, record


def _stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise FloatingPointError("non-finite array in manifest")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.astype(np.float32, copy=False).mean(dtype=np.float32)),
        "std": float(array.astype(np.float32, copy=False).std(dtype=np.float32)),
        "p01": float(np.float32(np.percentile(array.astype(np.float32, copy=False), np.float32(1)))),
        "p99": float(np.float32(np.percentile(array.astype(np.float32, copy=False), np.float32(99)))),
        "below_zero": float(np.mean(array < 0)),
        "above_one": float(np.mean(array > 1)),
    }


def _clipping_stats(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    removed = before[before != after]
    return {"lower_fraction": float(np.mean(before < 0)), "upper_fraction": float(np.mean(before > 1)),
            "changed_fraction": float(np.mean(before != after)),
            "removed_min": float(removed.min()) if removed.size else None,
            "removed_max": float(removed.max()) if removed.size else None,
            "max_abs_delta": float(np.max(np.abs(after - before)))}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _hwc(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu().numpy()
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"HWC export expects batch size 1, got {value.shape[0]}")
        value = value[0]
    if value.ndim == 3:
        value = value.transpose(1, 2, 0)
    return np.ascontiguousarray(value)


def _core_hashes() -> dict[str, str]:
    return {path.as_posix(): sha256_file(path) for path in PREPARATION_CORE_FILES}


def _hash_records_match(expected: Any, current: dict[str, str], label: str) -> None:
    if not isinstance(expected, dict):
        raise ValueError(f"prepared manifest is missing {label}")
    normalized = {str(key).replace("\\", "/"): str(value) for key, value in expected.items()}
    for name, digest in current.items():
        base = Path(name).name
        matches = [value for key, value in normalized.items()
                   if key == name or key.endswith("/" + name) or Path(key).name == base]
        if len(matches) != 1 or matches[0] != digest:
            raise ValueError(f"prepared {label} mismatch for {name}")


def _validate_author_checkpoint_identity(name: str, digest: str, role: str) -> None:
    """Allow only the exact authors' original checkpoint bytes for each role."""
    allowed = {
        "estimator": {"net_g_real.pth"},
    }
    if role not in allowed or name not in allowed[role]:
        raise ValueError(f"{role} must use an allowlisted original checkpoint, got {name!r}")
    expected = AUTHORIZED_CHECKPOINT_SHA256[name]
    if digest != expected:
        raise ValueError(f"original checkpoint SHA256 mismatch for {name}: {digest}")


def _verified_author_checkpoint(path: Path, role: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"original {role} checkpoint does not exist: {path}")
    digest = sha256_file(path)
    _validate_author_checkpoint_identity(path.name, digest, role)
    return digest


def _source_file_for_record(source: Path, record: str) -> Path:
    parts = Path(record.replace("\\", "/")).parts
    if len(parts) == 2 and parts[0] in {"near", "far"}:
        name = _safe_frame_filename(parts[1], parts[0])
        return source / name
    if len(parts) != 1:
        raise ValueError(f"unsafe prepared source-file key: {record!r}")
    return source / parts[0]


def _archive_path(prepared_dir: Path, record: dict[str, Any]) -> Path:
    value = record.get("archive")
    if not isinstance(value, str) or not value:
        raise ValueError("prepared frame archive path is missing")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe prepared archive path: {value!r}")
    path = (prepared_dir / relative).resolve()
    if prepared_dir.resolve() not in path.parents:
        raise ValueError(f"prepared archive escapes output directory: {value!r}")
    return path


def _validate_preparation(prepared_dir: Path, *, reject_unusable: bool = True) -> tuple[dict[str, Any], Path, dict[str, str]]:
    manifest_path = prepared_dir / "preparation.json"
    try:
        prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"fresh preparation manifest is required: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid preparation manifest: {manifest_path}") from exc
    if not isinstance(prepared, dict) or prepared.get("schema_version") != 1:
        raise ValueError("unsupported prepared-burst manifest schema")
    if prepared.get("precision") != "float32" or prepared.get("frame_count_per_side") != 5:
        raise ValueError("prepared input must contain five FP32 frames per side")
    if reject_unusable and prepared.get("status") in {"failed_first_frame", "failed_source_changed", "failed_review_assets"}:
        raise ValueError(f"prepared capture group is unusable: {prepared.get('status')}")

    source = Path(prepared.get("source", "")).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"prepared source directory is missing: {source}")
    metadata_path = source / "metadata.json"
    metadata_digest = prepared.get("capture_metadata_sha256")
    if not isinstance(metadata_digest, str) or sha256_file(metadata_path) != metadata_digest:
        raise ValueError("source metadata.json changed after preparation")

    verification = prepared.get("verification")
    if not isinstance(verification, dict):
        raise ValueError("prepared manifest has no verification record")
    for flag in ("source_hashes_unchanged", "source_metadata_unchanged",
                 "implementation_unchanged", "archives_round_trip_checked"):
        if verification.get(flag) is not True:
            raise ValueError(f"preparation verification did not pass: {flag}")
    before = verification.get("input_file_hashes_before")
    after = verification.get("input_file_hashes_after")
    if not isinstance(before, dict) or before != after:
        raise ValueError("prepared source hashes differ before and after preparation")
    current_source_hashes: dict[str, str] = {}
    for key, expected in before.items():
        path = _source_file_for_record(source, key)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source file changed after preparation: {path.name}")
        current_source_hashes[str(key)] = actual

    expected_core = prepared.get("implementation_sha256_before")
    after_core = prepared.get("implementation_sha256_after")
    if expected_core != after_core:
        raise ValueError("preparation implementation changed during frame processing")
    _hash_records_match(expected_core, _core_hashes(), "preparation core hash")

    sides = prepared.get("sides")
    if not isinstance(sides, dict):
        raise ValueError("prepared manifest has no side records")
    for side in ("near", "far"):
        side_record = sides.get(side)
        frames = side_record.get("frames") if isinstance(side_record, dict) else None
        metadata = side_record.get("metadata") if isinstance(side_record, dict) else None
        if not isinstance(frames, list) or len(frames) != 5 or not isinstance(metadata, list) or len(metadata) != 5:
            raise ValueError(f"prepared {side} side must retain five frame records and metadata entries")
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or frame.get("index") != index:
                raise ValueError(f"prepared {side} frame order is invalid at index {index}")
            filename = _safe_frame_filename(frame.get("filename"), side)
            source_path = source / filename
            if frame.get("sha256") != sha256_file(source_path):
                raise ValueError(f"source DNG changed after preparation: {filename}")
            archive_path = _archive_path(prepared_dir, frame)
            if not archive_path.is_file() or frame.get("archive_sha256") != sha256_file(archive_path):
                raise ValueError(f"prepared frame archive changed or is missing: {frame.get('archive')}")
        if reject_unusable:
            first = frames[0]
            if (first.get("error") is not None
                    or (first.get("processing") or {}).get("valid420_all") is not True):
                raise ValueError(f"{side}_00 cannot serve as a valid full-crop fallback")
    assets = prepared.get("review_assets")
    if isinstance(assets, dict):
        for filename, digest in assets.get("sha256", {}).items():
            path = Path(filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe preparation review asset path: {filename!r}")
            asset = prepared_dir / path
            if not asset.is_file() or sha256_file(asset) != digest:
                raise ValueError(f"preparation review asset changed or is missing: {filename}")
    return prepared, source, current_source_hashes


def _load_root_review(prepared_dir: Path) -> dict[str, Any]:
    path = prepared_dir / "root_review.json"
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"root visual review is required before inference: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid root visual review JSON: {path}") from exc
    if (not isinstance(review, dict) or review.get("status") != "reviewed"
            or type(review.get("texture_registration_passed")) is not bool
            or not isinstance(review.get("note"), str) or not review["note"].strip()):
        raise ValueError("root_review.json must contain reviewed status, texture decision, and a non-empty note")
    inspected = review.get("inspected_assets")
    if inspected is not None:
        if not isinstance(inspected, list):
            raise ValueError("inspected_assets must be a list when provided")
        for item in inspected:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                raise ValueError("each inspected_assets entry must contain path and sha256")
            asset = Path(item["path"])
            if asset.is_absolute() or ".." in asset.parts:
                raise ValueError(f"unsafe inspected asset path: {item['path']!r}")
            asset_path = (prepared_dir / asset).resolve()
            if prepared_dir.resolve() not in asset_path.parents or not asset_path.is_file():
                raise ValueError(f"inspected asset is missing or outside prepared output: {item['path']!r}")
            if sha256_file(asset_path) != item["sha256"]:
                raise ValueError(f"inspected asset changed after visual review: {item['path']!r}")
    return review


def _machine_average_gate(prepared: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    averaging = prepared.get("averaging", {})
    if not isinstance(averaging, dict) or averaging.get("quantitative_eligible") is not True:
        reasons.append(str(averaging.get("reason") or "quantitative burst gate failed"))
    for side in ("near", "far"):
        side_record = prepared["sides"][side]
        if side_record.get("metadata_compatible") is not True or side_record.get("averaging_eligible") is not True:
            reasons.append(f"{side} metadata/alignment/crop gate failed")
        for frame in side_record["frames"]:
            alignment = frame.get("alignment") or {}
            processing = frame.get("processing") or {}
            if alignment.get("within_rms_limit") is not True:
                reasons.append(f"{side}_{frame['index']:02d} marker RMS gate failed")
            if processing.get("valid420_all") is not True:
                reasons.append(f"{side}_{frame['index']:02d} full valid crop gate failed")
            if frame.get("error") is not None or frame.get("metadata_error") is not None:
                reasons.append(f"{side}_{frame['index']:02d} preparation error")
    return not reasons, reasons


def _select_prepared_indices(prepared: dict[str, Any], review: dict[str, Any]) -> tuple[dict[str, list[int]], bool, bool, list[str]]:
    machine_gate, reasons = _machine_average_gate(prepared)
    texture_passed = review.get("texture_registration_passed") is True
    averaged = machine_gate and texture_passed
    if machine_gate and not texture_passed:
        reasons.append("root texture registration review did not pass")
    for side in ("near", "far"):
        frame00 = prepared["sides"][side]["frames"][0]
        processing = frame00.get("processing") or {}
        if processing.get("valid420_all") is not True or frame00.get("error") is not None:
            raise ValueError("frame00 is not a valid full-crop fallback for both near and far")
    indices = {side: ([0, 1, 2, 3, 4] if averaged else [0]) for side in ("near", "far")}
    return indices, averaged, machine_gate, reasons


def _load_prepared_side(prepared_dir: Path, prepared: dict[str, Any], side: str,
                        indices: list[int]) -> dict[str, Any]:
    side_record = prepared["sides"][side]
    records = side_record["frames"]
    stage_names = ("warp512", "crop420", "output256", "libraw_output256")
    sums: dict[str, np.ndarray] = {}
    valid: np.ndarray | None = None
    for index in indices:
        record = records[index]
        archive_path = _archive_path(prepared_dir, record)
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != PREPARED_FRAME_KEYS:
                raise ValueError(f"prepared frame archive keys differ from contract: {record['archive']}")
            for name in stage_names:
                value = np.asarray(archive[name])
                if value.dtype != np.dtype(np.float32) or not np.isfinite(value).all():
                    raise TypeError(f"prepared {side}_{index:02d} {name} must be finite float32")
                expected_shape = {"warp512": (512, 512, 3), "crop420": (420, 420, 3),
                                  "output256": (256, 256, 3), "libraw_output256": (256, 256, 3)}[name]
                if value.shape != expected_shape:
                    raise ValueError(f"prepared {side}_{index:02d} {name} has shape {value.shape}, expected {expected_shape}")
                if name not in sums:
                    sums[name] = value.copy()
                else:
                    np.add(sums[name], value, out=sums[name])
            crop_valid = np.asarray(archive["valid420"])
            if crop_valid.shape != (420, 420) or crop_valid.dtype != np.uint8 or not np.all(crop_valid == 1):
                raise ValueError(f"selected prepared {side}_{index:02d} crop is not fully valid")
            valid = crop_valid.astype(bool) if valid is None else np.logical_and(valid, crop_valid)
    for value in sums.values():
        value /= np.float32(len(indices))
    combined = sums
    metadata = side_record["metadata"][0]
    black_levels = metadata.get("black_level_per_channel")
    if not isinstance(black_levels, list) or not black_levels or len(set(black_levels)) != 1:
        raise ValueError(f"prepared {side} metadata requires equal black levels")
    black, white = float(black_levels[0]), float(metadata["white_level"])
    if not np.isfinite((black, white)).all() or not 0 <= black < white:
        raise ValueError(f"prepared {side} metadata has invalid black/white levels")
    frame_records = [{key: frame.get(key) for key in
                      ("index", "filename", "sha256", "archive", "geometry", "alignment", "processing", "error")}
                     for frame in records]
    return {
        "black": black, "white": white,
        "counts": combined["output256"], "warp512": combined["warp512"],
        "crop420": combined["crop420"], "ref_counts": combined["libraw_output256"],
        "valid": valid, "frames": frame_records, "metadata": side_record["metadata"],
        "frame_count": len(indices),
    }


def _review_asset_paths(assets: dict[str, Any]) -> list[str]:
    names = [assets.get("html"), *assets.get("contact_sheets", []),
             *assets.get("alignment_differences", []), *assets.get("mean_candidates", [])]
    display_gamma = assets.get("display_gamma")
    gamma_hashes = display_gamma.get("asset_sha256", {}) if isinstance(display_gamma, dict) else {}
    if isinstance(gamma_hashes, dict):
        names.extend(gamma_hashes)
    result: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name:
            continue
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe preparation review asset path: {name!r}")
        if name not in result:
            result.append(name)
    return result


def _update_review_asset_hashes(output: Path, assets: dict[str, Any]) -> dict[str, Any]:
    names = _review_asset_paths(assets)
    missing = [name for name in names if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError("preparation review assets are missing: " + ", ".join(missing))
    assets["sha256"] = {name: sha256_file(output / name) for name in names}
    return assets


def _prepare_group(source: Path, output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output already contains files: {output}")
    prepared = prepare_burst(source, output)
    try:
        assets = write_prepare_review(output, prepared, group_label=source.name)
        prepared["review_assets"] = _update_review_asset_hashes(output, assets)
    except Exception as exc:
        prepared["review_asset_error"] = f"{type(exc).__name__}: {exc}"
        if prepared.get("averaging", {}).get("quantitative_eligible") is True:
            prepared["status"] = "failed_review_assets"
    _write_json(output / "preparation.json", prepared)
    _validate_preparation(output, reject_unusable=False)
    return prepared


def _ensure_review_assets(prepared_dir: Path, prepared: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "html": "prepare_review.html",
        "contact_sheets": ["prepare_near_contact_sheet.png", "prepare_far_contact_sheet.png"],
        "alignment_differences": ["prepare_near_alignment_difference.png", "prepare_far_alignment_difference.png"],
        "mean_candidates": (["prepare_near_frame00_vs_mean_display_only.png",
                             "prepare_far_frame00_vs_mean_display_only.png"]
                            if prepared.get("averaging", {}).get("quantitative_eligible") is True else []),
    }
    assets = prepared.get("review_assets")
    if not isinstance(assets, dict):
        assets = {}
    for key, value in expected.items():
        assets.setdefault(key, value)
    names = _review_asset_paths(assets)
    if any(not (prepared_dir / name).is_file() for name in names):
        assets = write_prepare_review(prepared_dir, prepared, group_label=prepared.get("source_name"))
    prepared["review_assets"] = _update_review_asset_hashes(prepared_dir, assets)
    prepared.pop("review_asset_error", None)
    _write_json(prepared_dir / "preparation.json", prepared)
    return prepared


def run(source: Path = DEFAULT_SOURCE, checkpoint: Path = DEFAULT_CHECKPOINT,
        output: Path | None = None, device: str = "cuda",
        precision: str = "float32", frame_index: int = 0, *,
        prepare_only: bool = False, prepared: Path | None = None) -> dict[str, Any]:
    source, checkpoint = Path(source), Path(checkpoint)
    prepared = Path(prepared) if prepared is not None else None
    output = Path(output) if output is not None else (prepared or DEFAULT_OUTPUT / source.name)
    precision = {"fp32": "float32"}.get(precision.lower(), precision.lower())
    if set(CONDITIONS) != {"both"}:
        raise RuntimeError("active capture runner requires the single 'both' photometry condition")
    if precision != "float32":
        raise ValueError("capture runner supports only fp32")
    work_dtype = np.float32
    if frame_index != 0:
        raise ValueError("the prepared five-frame capture workflow always uses frame 00 as its reference")
    if prepared is None:
        if not prepare_only:
            raise ValueError("run preparation with --prepare-only, inspect its review assets, then use --prepared")
        if not source.is_dir():
            raise FileNotFoundError(f"source capture directory does not exist: {source}")
        prep = _prepare_group(source, output)
        return prep
    if prepare_only:
        raise ValueError("--prepared and --prepare-only select different workflow phases")
    if output.resolve() != prepared.resolve():
        raise ValueError("prepared inference writes into the preparation directory; output must match --prepared")
    output = prepared
    prep, source, before_hashes = _validate_preparation(prepared)
    prep = _ensure_review_assets(prepared, prep)
    root_review = _load_root_review(prepared)
    root_review["sha256"] = sha256_file(prepared / "root_review.json")
    if prep.get("averaging", {}).get("quantitative_eligible") is True:
        assets = prep.get("review_assets", {})
        if (not isinstance(assets, dict) or len(assets.get("contact_sheets", [])) != 2
                or len(assets.get("mean_candidates", [])) != 2
                or prep.get("review_asset_error")):
            raise ValueError("quantitative burst passed but review images are unavailable; refusing inference")
        for filename in [assets.get("html"), *assets.get("contact_sheets", []),
                         *assets.get("alignment_differences", []), *assets.get("mean_candidates", [])]:
            if not filename or not (prepared / filename).is_file():
                raise FileNotFoundError(f"prepared visual review asset is missing: {filename}")
    selected_indices, averaged, machine_gate, machine_reasons = _select_prepared_indices(prep, root_review)
    frame_selection = {
        "mode": "burst_mean" if averaged else "burst_single_fallback",
        "available_frame_count": 5,
        "frame_count": 5 if averaged else 1,
        "selected_indices": selected_indices,
        "selected_filenames": {side: [prep["sides"][side]["frames"][index]["filename"] for index in indices]
                               for side, indices in selected_indices.items()},
        "selected_paths": {side: [str(source / prep["sides"][side]["frames"][index]["filename"])
                                   for index in indices] for side, indices in selected_indices.items()},
        "averaged": averaged,
        "averaging_gate": {"quantitative_passed": machine_gate,
                           "texture_review_status": "reviewed" if machine_gate else "not_applicable",
                           "texture_registration_passed": root_review["texture_registration_passed"] if machine_gate else None,
                           "reasons": machine_reasons},
        "root_review": root_review,
    }
    selected_paths_by_side = {side: source / prep["sides"][side]["frames"][0]["filename"]
                              for side in ("near", "far")}
    near_result = _load_prepared_side(prepared, prep, "near", selected_indices["near"])
    far_result = _load_prepared_side(prepared, prep, "far", selected_indices["far"])
    near_black, near_white = near_result["black"], near_result["white"]
    far_black, far_white = far_result["black"], far_result["white"]
    near_counts, far_counts = near_result["counts"], far_result["counts"]
    near_ref_counts, far_ref_counts = near_result["ref_counts"], far_result["ref_counts"]
    near_valid, far_valid = near_result["valid"], far_result["valid"]
    near_stages = {"output256": near_counts, "warp512": near_result["warp512"], "crop420": near_result["crop420"]}
    far_stages = {"output256": far_counts, "warp512": far_result["warp512"], "crop420": far_result["crop420"]}
    near_processing = {"frame_count": near_result["frame_count"], "valid_intersection_fraction": float(near_valid.mean()),
                       "per_frame": [item["processing"] for item in near_result["frames"]]}
    far_processing = {"frame_count": far_result["frame_count"], "valid_intersection_fraction": float(far_valid.mean()),
                      "per_frame": [item["processing"] for item in far_result["frames"]]}
    capture_metadata = prep.get("capture_metadata")
    if not isinstance(capture_metadata, dict):
        raise ValueError("prepared manifest has no capture metadata")
    expected_hash = _verified_author_checkpoint(checkpoint, "estimator")
    inference_code_paths = [
        Path(__file__), Path("capture_processing/__init__.py"),
        Path("capture_processing/raw.py"), Path("capture_processing/photometry.py"),
        Path("model/author_real_capture_adapter.py"), Path("network/base_net.py"),
        Path("network/nfplight_net.py"),
    ]
    code_hashes_before = {str(path): sha256_file(path) for path in inference_code_paths}
    report_code_path = Path("capture_processing/report.py")
    report_code_hash_before = sha256_file(report_code_path)
    output.mkdir(parents=True, exist_ok=True)
    occupied = [path.name for path in (output / "manifest.json", output / "verification.json",
                                       output / "source_arrays.npz") if path.exists()]
    occupied.extend(condition for condition in CONDITIONS if (output / condition).exists())
    if occupied:
        raise FileExistsError("prepared output already contains inference artifacts: " + ", ".join(occupied))
    before_hashes_by_side = {
        side: [prep["sides"][side]["frames"][index]["sha256"] for index in selected_indices[side]]
        for side in ("near", "far")
    }
    transforms = {
        "near": read_color_transform(selected_paths_by_side["near"]),
        "far": read_color_transform(selected_paths_by_side["far"]),
    }
    geometry = {
        "provenance": "fresh FP32 burst preparation; aligned per-frame archives",
        "averaging": averaged,
        "near": near_result["frames"][0]["geometry"],
        "far": far_result["frames"][0]["geometry"],
        "per_frame": {"near": near_result["frames"], "far": far_result["frames"]},
        "root_review": root_review,
    }
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for author-original inference but is unavailable")
    model = AuthorRealCaptureAdapter(checkpoint, device=device, image_size=256)
    if model.compute_dtype != torch.float32:
        raise AssertionError("model precision does not match requested pipeline precision")
    if model.feature_version != AUTHOR_FEATURE_VERSION or model.normal_head != AUTHOR_NORMAL_HEAD:
        raise AssertionError("capture runner requires the author legacy33/tanh output contract")
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.net_g.parameters()})
    buffer_dtypes = sorted({str(buffer.dtype) for buffer in model.net_g.buffers()})
    expected_torch_dtype = str(model.compute_dtype)
    if parameter_dtypes != [expected_torch_dtype] or buffer_dtypes not in ([], [expected_torch_dtype]):
        raise AssertionError(f"model dtype audit failed: parameters={parameter_dtypes}, buffers={buffer_dtypes}")
    constant_dtypes = {name: str(value.dtype) for name, value in vars(model).items()
                       if isinstance(value, torch.Tensor) and value.is_floating_point()}
    if any(dtype != expected_torch_dtype for dtype in constant_dtypes.values()):
        raise AssertionError(f"model constants lost float32 dtype: {constant_dtypes}")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    condition_records: dict[str, Any] = {}
    condition_arrays: dict[str, dict[str, np.ndarray]] = {}
    all_display: dict[str, list[np.ndarray]] = {}
    common_arrays = {
        "near_counts": near_counts,
        "far_counts": far_counts,
        "near_warp512": near_stages["warp512"],
        "far_warp512": far_stages["warp512"],
        "near_crop420": near_stages["crop420"],
        "far_crop420": far_stages["crop420"],
        "near_libraw_counts": near_ref_counts,
        "far_libraw_counts": far_ref_counts,
        "near_ahd_rectified_difference": near_counts - near_ref_counts,
        "far_ahd_rectified_difference": far_counts - far_ref_counts,
        "near_valid": near_valid.astype(np.uint8),
        "far_valid": far_valid.astype(np.uint8),
    }
    np.savez_compressed(output / "source_arrays.npz", **common_arrays)
    with np.load(output / "source_arrays.npz", allow_pickle=False) as archive:
        for key, value in common_arrays.items():
            if not np.array_equal(archive[key], value):
                raise AssertionError(f"source NPZ round-trip mismatch: {key}")
    for condition, (formula, black_used, white_used) in CONDITIONS.items():
        folder = output / condition
        (folder / "display").mkdir(parents=True, exist_ok=False)
        near_comp = apply_compensation(near_counts, near_black, near_white, condition, dtype=work_dtype)
        far_comp = apply_compensation(far_counts, far_black, far_white, condition, dtype=work_dtype)
        near_wb, near_linear = apply_linear_color(near_comp, transforms["near"])
        far_wb, far_linear = apply_linear_color(far_comp, transforms["far"])
        near_wb, near_linear = near_wb.astype(work_dtype), near_linear.astype(work_dtype)
        far_wb, far_linear = far_wb.astype(work_dtype), far_linear.astype(work_dtype)
        packed = np.concatenate((near_linear, far_linear), axis=-1)
        input_unclipped = packed.copy()
        input_clipped = np.clip(input_unclipped, 0.0, 1.0)
        input_clip_delta = input_clipped - input_unclipped
        tensor = torch.from_numpy(input_clipped.transpose(2, 0, 1)[None].copy()).to(
            device=model.device, dtype=model.compute_dtype)
        prediction, trace = model.infer(tensor)
        trace_matches_model = torch.equal(trace["features_legacy33"], model.last_estimator_input)
        if prediction.dtype != model.compute_dtype or not torch.isfinite(prediction).all():
            raise FloatingPointError("model prediction has invalid dtype or values")
        # Comparison-only replay through the same geometry, photometry, and
        # model.  The LibRaw uint16 AHD path never supplies authoritative input.
        ref_near_comp = apply_compensation(near_ref_counts, near_black, near_white, condition, dtype=work_dtype)
        ref_far_comp = apply_compensation(far_ref_counts, far_black, far_white, condition, dtype=work_dtype)
        _, ref_near_linear = apply_linear_color(ref_near_comp, transforms["near"])
        _, ref_far_linear = apply_linear_color(ref_far_comp, transforms["far"])
        ref_input = np.concatenate((ref_near_linear, ref_far_linear), axis=-1)
        ref_tensor = torch.from_numpy(np.clip(ref_input, 0.0, 1.0).transpose(2, 0, 1)[None].copy()).to(
            device=model.device, dtype=model.compute_dtype)
        reference_prediction, ref_trace = model.infer(ref_tensor)
        if not torch.isfinite(reference_prediction).all():
            raise FloatingPointError("comparison LibRaw prediction is invalid")
        arrays: dict[str, np.ndarray] = {
            "near_compensated": near_comp,
            "far_compensated": far_comp,
            "near_whitebalanced": near_wb,
            "far_whitebalanced": far_wb,
            "near_linear_srgb": near_linear,
            "far_linear_srgb": far_linear,
            "near_input_unclipped": input_unclipped[..., :3],
            "far_input_unclipped": input_unclipped[..., 3:],
            "near_input_clipped": input_clipped[..., :3],
            "far_input_clipped": input_clipped[..., 3:],
            "near_input_clip_delta": input_clip_delta[..., :3],
            "far_input_clip_delta": input_clip_delta[..., 3:],
            "prediction_nchw": prediction.cpu().numpy(),
            "prediction": _hwc(prediction),
            "prediction_display": _hwc((prediction + 1) / 2),
            "libraw_relation_difference": _hwc(trace["relation"] - ref_trace["relation"]),
            "libraw_log_relation_difference": _hwc(trace["relation_log"] - ref_trace["relation_log"]),
            "libraw_prediction_difference": _hwc(prediction - reference_prediction),
        }
        arrays.update({key: _hwc(value) for key, value in trace.items()})
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"non-finite saved array: {key}")
        np.savez_compressed(folder / "arrays.npz", **arrays)
        with np.load(folder / "arrays.npz", allow_pickle=False) as archive:
            for key, value in arrays.items():
                if not np.array_equal(archive[key], value):
                    raise AssertionError(f"NPZ round-trip mismatch: {condition}/{key}")
        condition_arrays[condition] = arrays
        for key, value in arrays.items():
            if value.ndim == 2:
                all_display.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
            elif value.ndim == 3:
                channels = value.shape[-1]
                if channels > 3:
                    for channel in range(channels):
                        all_display.setdefault(f"{key}_c{channel}", []).append(
                            np.asarray(value[..., channel], dtype=np.float32))
                else:
                    all_display.setdefault(key, []).append(np.asarray(value, dtype=np.float32))
        condition_records[condition] = {
            "formula": formula,
            "black_compensation": black_used,
            "white_compensation": white_used,
            "input_clipped_fraction": float(np.mean(input_clip_delta != 0)),
            "input_lower_clip_fraction": float(np.mean(input_unclipped < 0)),
            "input_upper_clip_fraction": float(np.mean(input_unclipped > 1)),
            "input_removed_min": _clipping_stats(input_unclipped, input_clipped)["removed_min"],
            "input_removed_max": _clipping_stats(input_unclipped, input_clipped)["removed_max"],
            "clipping": {"near_input": _clipping_stats(arrays["near_input_unclipped"], arrays["near_input_clipped"]),
                         "far_input": _clipping_stats(arrays["far_input_unclipped"], arrays["far_input_clipped"]),
                         "far_gain": _clipping_stats(arrays["original_copy_far_gained_unclipped"],
                                                     arrays["original_copy_far_gained_clipped"]),
                         "far_time": _clipping_stats(arrays["far_time_unclipped"], arrays["far_time_clipped"])},
            "libraw_difference_metrics": {key: {"mae": float(np.mean(np.abs(value))), "max_abs": float(np.max(np.abs(value)))}
                                         for key, value in arrays.items() if key.startswith("libraw_") and key.endswith("difference")},
            "arrays": {key: _stats(value) for key, value in arrays.items()},
            "trace_matches_model_exactly": trace_matches_model,
            "trace_method": "legacy33 feature tensor compared to the exact tensor passed to estimator forward",
            "npz_roundtrip_exact": True,
        }
    display_ranges: dict[str, list[float]] = {}
    for key, values in all_display.items():
        low = min(float(np.float32(np.percentile(value.astype(np.float32, copy=False), np.float32(1)))) for value in values)
        high = max(float(np.float32(np.percentile(value.astype(np.float32, copy=False), np.float32(99)))) for value in values)
        display_ranges[key] = [low, high]
    after_hashes = {
        key: sha256_file(_source_file_for_record(source, key))
        for key in before_hashes
    }
    if after_hashes != before_hashes:
        raise AssertionError("one or more source DNGs changed during inference")
    if sha256_file(source / "metadata.json") != prep["capture_metadata_sha256"]:
        raise AssertionError("source metadata.json changed during inference")
    prep_after, _, verified_after_hashes = _validate_preparation(prepared)
    if verified_after_hashes != before_hashes or prep_after.get("capture_metadata_sha256") != prep["capture_metadata_sha256"]:
        raise AssertionError("prepared DNG or frame archive changed during inference")
    after_hashes_by_side = {
        side: [sha256_file(source / prep["sides"][side]["frames"][index]["filename"])
               for index in selected_indices[side]]
        for side in ("near", "far")
    }
    frame_selection["sha256_before"] = before_hashes_by_side
    frame_selection["sha256_after"] = after_hashes_by_side
    checkpoint_hash_after = sha256_file(checkpoint)
    if checkpoint_hash_after != expected_hash:
        raise AssertionError("checkpoint changed during inference")
    code_hashes_after = {str(path): sha256_file(path) for path in inference_code_paths}
    if code_hashes_after != code_hashes_before:
        raise AssertionError("inference implementation changed during the run")
    manifest = {
        "format": INFERENCE_FORMAT,
        "status": "ok",
        "source": str(source.resolve()),
        "metadata_sha256": prep["capture_metadata_sha256"],
        "preparation_manifest_sha256": sha256_file(prepared / "preparation.json"),
        "source_dng_sha256_before": before_hashes_by_side,
        "source_dng_sha256_after": after_hashes_by_side,
        "preparation_source_hashes": {
            "all_dng_sha256_before": before_hashes,
            "all_dng_sha256_after": after_hashes,
            "capture_metadata_sha256": prep["capture_metadata_sha256"],
        },
        "selected_frames": frame_selection,
        "frame_average": {
            "enabled": averaged,
            "equal_weight": True,
            "mean_generated": averaged,
            "stage": "aligned output256 FP32 RGB counts",
            "quantitative_gate_passed": machine_gate,
            "texture_review_status": "reviewed" if machine_gate else "not_applicable",
            "texture_registration_passed": root_review["texture_registration_passed"] if machine_gate else None,
            "selected_indices": selected_indices,
            "fallback_reason": machine_reasons if not averaged else None,
            "near_count": near_result["frame_count"],
            "far_count": far_result["frame_count"],
        },
        "burst_preparation": {
            "status": prep["status"],
            "spatial_reference": prep.get("spatial_reference"),
            "quantitative_eligible": prep.get("averaging", {}).get("quantitative_eligible"),
            "selected_indices": selected_indices,
            "reason": prep.get("averaging", {}).get("reason"),
            "near_far_pair_residuals": prep.get("near_far_pair_residuals"),
            "mean_generated_by_preparation": prep.get("averaging", {}).get("mean_generated"),
        },
        "source_frame_archives": {
            "near": [record["archive"] for record in prep["sides"]["near"]["frames"]],
            "far": [record["archive"] for record in prep["sides"]["far"]["frames"]],
            "roundtrip_exact": True,
        },
        "near_far_pair_residuals": prep.get("near_far_pair_residuals"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": expected_hash,
        "report_implementation_sha256": {str(report_code_path): report_code_hash_before},
        "denoiser": {
            "used": False,
            "mode": "excluded_identity_copy",
            "network_instantiated": False,
            "checkpoint_read": False,
            "weights_loaded": False,
            "forward_called": False,
            "legacy_slots_source": "original linear-RGB input pair copied into historical original_copy feature slots",
            "provenance": "user-approved exclusion of the original optional no_denoise branch from commit cc79c9b",
        },
        "root_visual_review": root_review,
        "preparation_review_assets": prep.get("review_assets"),
        "geometry": {"path": None, "content": geometry,
                      "near_processing": near_processing, "far_processing": far_processing,
                      "selected_dtype": precision},
        "source_arrays": {key: _stats(value) for key, value in common_arrays.items()},
        "capture_metadata": capture_metadata,
        "raw_captures": {"near": near_result["metadata"], "far": far_result["metadata"]},
        "libraw_ahd_comparison": {
            "near": _stats(common_arrays["near_ahd_rectified_difference"]),
            "far": _stats(common_arrays["far_ahd_rectified_difference"]),
            "path": "source_frames/*; selected/combined RGB in source_arrays.npz; comparison-only uint16 LibRaw AHD",
            "rectified_difference_metrics": {side: {"mae": float(np.mean(np.abs(common_arrays[f"{side}_ahd_rectified_difference"]), dtype=np.float32)),
                                                    "max_abs": float(np.max(np.abs(common_arrays[f"{side}_ahd_rectified_difference"])))} for side in ("near", "far")},
        },
        "color_transforms": transforms,
        "conditions": condition_records,
        "pipeline": ["each DNG unpacked once during fresh preparation", "uint16 Bayer -> FP32 exact cast", "continuous AHD demosaic",
                      "fresh marker geometry per frame", prep.get("spatial_reference", "within-side marker alignment to frame00"), "512 warp", "crop 46 -> 420", "exact area resize -> 256",
                      "conditional FP32 mean of aligned output256 counts or near_00/far_00 fallback",
                      "black/white condition", "camera WB", "camera -> linear sRGB matrix",
                      "preserve unclipped", "model input clip", "pass the already-linear FP32 RGB pair directly to the author adapter; do not apply x^2.2 again",
                      "denoiser excluded: copy the clipped original pair into legacy original_copy feature slots; no DenoiseNet object or checkpoint is created/read",
                      "author center-patch near/far gain and legacy 33-channel feature assembly", "author TwoBranchRealNet inference"],
        "display_ranges": display_ranges,
        "precision": {"requested": precision, "raw": f"uint16 -> {precision} exact",
                       "demosaic": f"{precision} AHD", "spatial": f"{precision} bilinear + exact area",
                       "photometry": precision, "model": precision, "estimator": precision,
                       "geometry_setup": f"{GEOMETRY_SETUP_DTYPE} CPU NumPy calculation; buffers stored as {GEOMETRY_BUFFER_DTYPE}",
                       "denoiser": "excluded; no load or forward", "amp": False, "tf32": False},
        "model_dtype_audit": {"parameters": parameter_dtypes, "buffers": buffer_dtypes,
                              "expected": expected_torch_dtype, "constants": constant_dtypes,
                              "geometry_setup_dtype": model.geometry_setup_dtype,
                              "geometry_buffer_dtype": model.geometry_buffer_dtype,
                              "model_tensor_inference_dtype": expected_torch_dtype},
        "model_geometry": {"near_distance": model.near_distance, "far_distance": model.far_distance,
                           "setup_dtype": model.geometry_setup_dtype,
                           "setup_device": "CPU NumPy only; no FP64 model tensors or inference",
                           "buffer_dtype": model.geometry_buffer_dtype,
                           "distance_map": "rho=sqrt((x-127.5)^2+(y-127.5)^2)/128; theta=atan(rho/distance)",
                           "coefficient": "((cos(theta_far)-cos(theta_near))/cos(theta_near))/global_max",
                           "coefficient_denominator_epsilon": 1e-5,
                           "time_co_map": "cos(theta_near)/cos(theta_far)",
                           "far_gain": "mean(original_copy_near[:, :, 118:138, 118:138]) / clamp(mean(original_copy_far[:, :, 118:138, 118:138]), min=1e-5)",
                           "far_gain_patch": [118, 138], "log_normalization_epsilon": 1e-2,
                           "saturation_threshold": 0.95},
        "model_contract": {
            "estimator_family": "author_legacy33",
            "estimator_architecture": "author TwoBranchRealNet; width=32, each branch encoder=[2,2,4,8], middle=12, decoder=[2,2,2,2]; NAFBlock DW_Expand=2, FFN_Expand=2, dropout=0",
            "layer_norm": "author LayerNorm2d; per-pixel channel normalization, eps=1e-6",
            "feature_version": model.feature_version,
            "normal_head": model.normal_head,
            "denoiser": {"used": False, "mode": "excluded_identity_copy",
                          "network_instantiated": False, "checkpoint_read": False,
                          "forward_called": False,
                          "legacy_denoised_slot_source": "original linear-RGB pair identity-copied into explicit original_copy feature slots"},
            "input_shape": [1, 6, 256, 256], "input_channel_order": ["near_r", "near_g", "near_b", "far_r", "far_g", "far_b"],
            "input_range": {"linear_rgb": [0.0, 1.0], "network_tensor": [-1.0, 1.0]},
            "input_gamma_applied": False,
            "feature_channels": 33,
            "feature_tensor_layout": "NCHW estimator input; trace NPZ arrays are HWC",
            "prediction_storage_layout": {"prediction_nchw": "NCHW", "prediction": "HWC"},
            "feature_channel_order": [
                "original_near_rgb", "original_far_rgb", "original_copy_near_rgb", "original_copy_far_rgb",
                "original_copy_gain_scaled_far_rgb", "log_original_6_same_order", "log_original_copy_6_same_order",
                "log_original_copy_gain_scaled_far_rgb", "relation", "log_relation", "non_saturated_mask",
            ],
            "network_input_branch_split": {"intro_albedo": "channels[0:30]", "intro_specular": "channels[30:33]"},
            "prediction_channels": 10,
            "raw_prediction_range": [-1.0, 1.0],
            "prediction_channel_order": ["normal_x", "normal_y", "normal_z", "diffuse_r", "diffuse_g", "diffuse_b", "roughness", "specular_r", "specular_g", "specular_b"],
            "prediction_decode": "raw tanh channels; display preview only uses (raw + 1) / 2; no normal renormalization or head conversion",
        },
        "limitations": ["DNG unpack exposes decoded Bayer samples; no claim of recovering pre-ADC signals.",
                        "Only the exact allowlisted author-original net_g_real checkpoint bytes are used; no denoiser checkpoint was read or used.",
                        "The DNG input is already linear sRGB; the author PNG x^2.2 decode is not applied a second time.",
                        "Marker residual threshold does not guarantee texture registration; averaging also requires root visual review.",
                        "No GT; this is checkpoint/input sensitivity evidence.",
                        "RGB preview gamma is display-only and does not alter the model input or saved raw prediction."],
        "source_dng_unchanged": True,
    }
    manifest["verification"] = {
        "status": "passed", "source_dng_unchanged": True,
        "source_metadata_unchanged": True,
        "all_prepared_archives_hash_valid": True,
        "preparation_core_unchanged": True,
        "checkpoint_unchanged": checkpoint_hash_after == expected_hash,
        "denoiser_excluded": manifest["denoiser"]["used"] is False,
        "denoiser_checkpoint_read": manifest["denoiser"]["checkpoint_read"] is False,
        "denoiser_forward_called": manifest["denoiser"]["forward_called"] is False,
        "inference_code_unchanged": code_hashes_after == code_hashes_before,
        "source_npz_roundtrip_exact": True,
        "condition_trace_matches_actual_features": all(v["trace_matches_model_exactly"] for v in condition_records.values()),
        "condition_npz_roundtrip_exact": all(v["npz_roundtrip_exact"] for v in condition_records.values()),
        "active_condition": "both",
        "single_active_condition": True,
        "model_dtype_audit": manifest["model_dtype_audit"],
    }
    import platform
    import rawpy
    import scipy
    manifest["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "geometry_lapack": "sgels/sgesv/sgetrf/sgetri", "torch": torch.__version__, "rawpy": rawpy.__version__, "libraw": rawpy.libraw_version}
    manifest["implementation_sha256"] = code_hashes_before
    manifest["limitations"].append("AHD is a continuous floating adaptation of LibRaw 0.21.4; direction decisions can differ, not only quantization.")
    manifest["ahd_source"] = "https://raw.githubusercontent.com/LibRaw/LibRaw/0.21.4/src/demosaic/ahd_demosaic.cpp"
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "verification.json", manifest["verification"])
    write_capture_report(output, display_ranges, condition_records, sorted(all_display))
    if sha256_file(report_code_path) != report_code_hash_before:
        raise AssertionError("report implementation changed while generating the capture report")
    manifest["verification"]["report_implementation_unchanged"] = True
    _write_json(output / "manifest.json", manifest)
    _write_json(output / "verification.json", manifest["verification"])
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def _batch_record(group: str, source: Path, output: Path,
                  result: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
    prep_path = output / "preparation.json"
    prepared = {}
    if prep_path.is_file():
        try:
            prepared = json.loads(prep_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            prepared = {}
    is_inference = isinstance(result, dict) and result.get("format") == INFERENCE_FORMAT
    report_name = "report.html"
    review = output / (report_name if is_inference else "prepare_review.html")
    average = result.get("frame_average", {}) if is_inference else {
        "enabled": False,
        "fallback_reason": prepared.get("averaging", {}).get("reason"),
    }
    gate = prepared.get("averaging", {})
    if error:
        status = "failed"
    elif is_inference:
        status = "complete" if result.get("status") == "ok" else str(result.get("status", "failed"))
    elif prepared.get("status", "").startswith("failed"):
        status = "failed"
    elif gate.get("quantitative_eligible") is True:
        status = "pending_root_review"
    else:
        status = "pending_root_review_fallback"
    conditions = result.get("conditions", {}) if is_inference else {}
    clipping = {name: record.get("clipping", {}) for name, record in conditions.items()}
    return {
        "group": group, "source": str(source), "output": group,
        "status": status, "error": error,
        "report_html": f"{group}/{review.name}" if review.is_file() else None,
        "frame_average": average,
        "prepared": {
            "averaging_eligible": gate.get("quantitative_eligible"),
            "selected_frames": result.get("selected_frames") if is_inference else gate.get("selected_indices"),
            "fallback_reason": gate.get("reason"),
        },
        "conditions": conditions,
        "clipping": clipping,
    }


def _run_all_groups(source_root: Path, output_root: Path, checkpoint: Path, device: str,
                   *, prepare_only: bool, prepared_phase: bool, precision: str,
                   frame_index: int) -> list[dict[str, Any]]:
    records = []
    for group in CAPTURE_GROUPS:
        source, output = source_root / group, output_root / group
        result = None
        error = None
        try:
            if prepared_phase:
                result = run(source=source, checkpoint=checkpoint, output=output, device=device,
                             precision=precision, frame_index=frame_index, prepared=output)
            else:
                result = run(source=source, checkpoint=checkpoint, output=output, device=device,
                             precision=precision, frame_index=frame_index, prepare_only=prepare_only)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        records.append(_batch_record(group, source, output, result=result, error=error))
    write_capture_index(output_root, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", "--output", dest="out", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32",), default="fp32")
    parser.add_argument("--all-groups", action="store_true", help="process all five new capture folders independently")
    phase = parser.add_mutually_exclusive_group()
    phase.add_argument("--prepare-only", action="store_true", help="write fresh aligned per-frame archives and visual review assets")
    phase.add_argument("--prepared", type=Path, nargs="?", const=DEFAULT_OUTPUT, metavar="DIR",
                       help="infer from a fresh prepared directory after root_review.json is recorded")
    parser.add_argument("--frame-index", type=int, default=0,
                        help="compatibility argument; the fixed burst workflow uses frame 00 as its reference")
    args = parser.parse_args()
    if not args.prepare_only and args.prepared is None:
        parser.error("choose --prepare-only or --prepared after root visual review")
    if args.all_groups:
        prepared_phase = args.prepared is not None
        output_root = args.out or args.prepared or DEFAULT_OUTPUT
        if prepared_phase and args.out is not None and args.out.resolve() != args.prepared.resolve():
            parser.error("batch --out and --prepared must name the same artifact root")
        records = _run_all_groups(args.source.parent, output_root, args.checkpoint, args.device,
                                  prepare_only=args.prepare_only, prepared_phase=prepared_phase,
                                  precision=args.precision, frame_index=args.frame_index)
        failures = [item for item in records if item["status"] == "failed"]
        print(f"CAPTURE_INDEX {output_root / 'index.html'} groups={len(records)} failed={len(failures)}", flush=True)
        if failures:
            raise SystemExit(1)
        return

    if args.prepared is not None:
        prepared_dir = args.prepared
        if prepared_dir == DEFAULT_OUTPUT and args.out is None:
            prepared_dir = DEFAULT_OUTPUT / args.source.name
        output = args.out or prepared_dir
        result = run(source=args.source, checkpoint=args.checkpoint, output=output,
                     device=args.device, precision=args.precision, frame_index=args.frame_index,
                     prepared=prepared_dir)
    else:
        output = args.out or DEFAULT_OUTPUT / args.source.name
        result = run(source=args.source, checkpoint=args.checkpoint, output=output,
                     device=args.device, precision=args.precision, frame_index=args.frame_index,
                     prepare_only=True)
    print(json.dumps({"status": result.get("status"), "source": result.get("source"),
                      "output": str(output), "averaging": result.get("frame_average", result.get("averaging"))},
                     ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
