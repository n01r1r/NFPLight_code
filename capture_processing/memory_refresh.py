"""Refresh capture reports from preserved FP32 camera RGB without saving NPZ."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .burst import _warp_stages, register_marker_frames_to_first, validate_burst_metadata
from .geometry import detect_marker_geometry
from .photometry import apply_compensation, apply_linear_color
from .report import (_display_configs, _figure_gallery, _make_html,
                     _make_range, _render_array, _save_display_previews,
                     _save_prediction_component_previews, _stats, write_capture_index)
from model.author_real_capture_adapter import AuthorRealCaptureAdapter


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def publish_memory_records(output: Path, manifest: dict) -> None:
    """Publish current root metadata, preserving the preceding JSON records."""
    parent = output.parent
    history = parent / "history_before_refresh_20261006"
    names = ("manifest.json", "preparation.json", "root_review.json", "verification.json",
             "report_html_generation.json")
    for name in names:
        previous = parent / name
        if previous.is_file():
            history.mkdir(exist_ok=True)
            if not (history / name).exists():
                shutil.copyfile(previous, history / name)
    _json(parent / "manifest.json", manifest)
    _json(parent / "preparation.json", json.loads((output / "registration.json").read_text(encoding="utf-8")))
    _json(parent / "root_review.json", manifest["root_visual_review"])
    _json(parent / "verification.json", manifest["verification"])
    _json(parent / "report_html_generation.json", {
        "format": "nfplight.memory-refresh-publication.v1", "current_artifact": output.name,
        "npz_saved": False, "sha256_executed": False, "inference_executed": True,
        "previous_json_records": history.name})


def _linear(counts: np.ndarray, metadata: dict, transform: dict) -> tuple[np.ndarray, ...]:
    black = metadata["black_level_per_channel"]
    if not black or len(set(black)) != 1:
        raise ValueError("equal-channel black levels required")
    compensated = apply_compensation(counts, black[0], metadata["white_level"], "both")
    wb, linear = apply_linear_color(compensated, transform)
    return compensated, wb, linear, np.clip(linear, 0, 1)


def _contact(output: Path, previews: dict[str, list[np.ndarray]]) -> None:
    sheet = Image.new("RGB", (256 * 5, 280 * 2), "#222222")
    draw = ImageDraw.Draw(sheet)
    for row, side in enumerate(("near", "far")):
        for index, rgb in enumerate(previews[side]):
            sheet.paste(Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)),
                        (256 * index, 280 * row + 24))
            draw.text((256 * index + 4, 280 * row + 4), f"{side}_{index:02d} | [0,1]", fill="white")
    sheet.save(output / "registration_contact.png")


def prepare_memory_group(archive: Path, destination: Path) -> dict:
    """Use inset0 -> warp512 -> crop46/420 -> area256; retain FP32 frames in RAM."""
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
    previous = json.loads((archive / "preparation.json").read_text(encoding="utf-8"))
    counts = {side: [] for side in ("near", "far")}
    previews = {side: [] for side in counts}
    records = {side: [] for side in counts}
    metadata = previous["sides"]
    reference = None
    failures = []
    protected = {}
    for side in counts:
        signature = None
        for index in range(5):
            path = archive / "source_frames" / f"{side}_{index:02d}.npz"
            protected[str(path)] = (path.stat().st_size, path.stat().st_mtime_ns)
            with np.load(path, allow_pickle=False) as saved:
                rgb = saved["demosaic_rgb_selected_dtype"]
            if rgb.dtype != np.float32 or not np.isfinite(rgb).all():
                raise ValueError(f"invalid preserved camera RGB: {path}")
            _, geometry = detect_marker_geometry(rgb, inset=0.0)
            if reference is None:
                reference = geometry
                alignment = register_marker_frames_to_first([reference], crop_margin=46)[0]
            else:
                alignment = register_marker_frames_to_first([reference, geometry], crop_margin=46)[1]
            stages = _warp_stages(rgb, np.asarray(alignment["homography_final"], dtype=np.float32))
            current = metadata[side]["metadata"][index]
            if signature is None:
                signature = current
            try:
                from .burst import _metadata_signature
                validate_burst_metadata(
                    _metadata_signature(signature, signature["capture_settings"], signature["dng_exif"]),
                    _metadata_signature(current, current["capture_settings"], current["dng_exif"]))
            except ValueError as exc:
                failures.append(f"{side}_{index:02d}: {exc}")
            supported = bool(stages["valid420"].all())
            if index == 0 and not supported:
                raise ValueError(f"{side}_00 has incomplete crop support")
            if not supported or not alignment["within_rms_limit"]:
                failures.append(f"{side}_{index:02d}: RMS={alignment['residual_rms_px']:.6g}px; crop={supported}")
            records[side].append({"index": index, "geometry": geometry, "alignment": alignment,
                                  "full_crop_support": supported})
            counts[side].append(stages["output256"])
            previews[side].append(_linear(stages["output256"], current, manifest["color_transforms"][side])[3])
            del rgb, stages
            print(f"FRAME {archive.name} {side}_{index:02d} RMS={alignment['residual_rms_px']:.4f}", flush=True)
    _contact(destination, previews)
    for index in range(5):
        near, far = previews["near"][index], previews["far"][index]
        yy, xx = np.indices(near.shape[:2])
        checker = np.where(((yy // 32 + xx // 32) % 2)[..., None], near, far)
        _render_array(checker, "linear_srgb", destination / f"registration_checker_{index:02d}.png", (0, 1), "gray")
    review = {"status": "pending", "texture_registration_passed": None,
              "quantitative_passed": not failures, "failures": failures,
              "display_range": [0, 1], "source": str(archive.resolve()),
              "npz_saved": False, "sha256_executed": False}
    _json(destination / "root_review.json", review)
    _json(destination / "registration.json", {"reference": "near_00", "frames": records,
                                              "spatial_policy": {"version": "sample_crop_only_v1", "sample_side_cm": 5.0, "sample_side_basis": "user-confirmed sample ROI after paper exclusion", "inset": 0.0, "crop_margin": 46, "warp_size": 512, "output_size": 256}, "rms_limit_px": 1, "failures": failures})
    return {"archive": archive, "output": destination, "manifest": manifest,
            "counts": counts, "previews": previews, "records": records,
            "metadata": metadata, "failures": failures, "protected": protected}


def infer_memory_group(group: dict, model: AuthorRealCaptureAdapter) -> dict:
    """Infer once from the reviewed population and render in-memory traces."""
    output = group["output"]
    review = json.loads((output / "root_review.json").read_text(encoding="utf-8"))
    if review.get("status") != "reviewed" or not isinstance(review.get("texture_registration_passed"), bool):
        raise ValueError("record an actual visual review before memory inference")
    averaged = not group["failures"] and review["texture_registration_passed"]
    chosen = list(range(5)) if averaged else [0]
    arrays = {}
    common = {}
    for side in ("near", "far"):
        stack = np.stack([group["counts"][side][index] for index in chosen])
        counts = np.mean(stack, axis=0, dtype=np.float32)
        comp, wb, linear, clipped = _linear(counts, group["metadata"][side]["metadata"][0],
                                           group["manifest"]["color_transforms"][side])
        arrays.update({side + "_compensated": comp, side + "_whitebalanced": wb,
                       side + "_linear_srgb": linear, side + "_input_unclipped": linear,
                       side + "_input_clipped": clipped, side + "_input_clip_delta": clipped - linear})
        for index, image in enumerate(group["previews"][side]):
            common[f"frame_{side}_{index:02d}_linear_srgb"] = image
    packed = np.concatenate([arrays["near_input_clipped"], arrays["far_input_clipped"]], axis=-1)
    tensor = torch.from_numpy(packed.transpose(2, 0, 1)[None].copy()).to(model.device)
    prediction, trace = model.infer(tensor)
    if not torch.equal(trace["features_legacy33"], model.last_estimator_input):
        raise AssertionError("feature trace does not match estimator input")
    from run_fabric_capture import _hwc, _clipping_stats
    arrays.update({key: _hwc(value) for key, value in trace.items()})
    arrays.update(prediction=_hwc(prediction), prediction_display=_hwc((prediction + 1) / 2))
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError("non-finite or non-FP32 memory result")
    display_keys = []
    for key, value in arrays.items():
        if value.ndim in (2, 3):
            display_keys.extend([f"{key}_c{i}" for i in range(value.shape[-1])]
                                if value.ndim == 3 and value.shape[-1] > 3 else [key])
    conditions = {"both": {"formula": "(D-B)/(W-B)", "black_compensation": True,
                            "white_compensation": True, "arrays": {k: _stats(v) for k, v in arrays.items()},
                            "trace_matches_model_exactly": True, "npz_saved": False,
                            "clipping": {side: _clipping_stats(arrays[side + "_input_unclipped"],
                                                               arrays[side + "_input_clipped"])
                                         for side in ("near", "far")}}}
    prior = group["manifest"]
    manifest = {key: prior[key] for key in ("source", "capture_metadata", "color_transforms", "model_contract", "model_geometry")}
    manifest["spatial_policy"] = {"version": "sample_crop_only_v1", "inset": 0.0,
                                  "crop_margin": 46, "warp_size": 512, "output_size": 256,
                                  "sample_side_cm": 5.0,
                                  "sample_side_basis": "user-confirmed sample ROI after paper exclusion"}
    manifest.update(format="nfplight.fabric-capture.memory.v1", status="complete", conditions=conditions,
                    checkpoint="checkpoints/net_g_real.pth", checkpoint_identity_verification="not rerun; original path only",
                    precision={"processing": "float32", "inference": "float32", "AMP": False, "TF32": False},
                    denoiser={"used": False, "network_instantiated": False, "forward_called": False},
                    selected_frames={"averaged": averaged, "mode": "burst_mean" if averaged else "burst_single_fallback",
                                     "selected_indices": {side: chosen for side in ("near", "far")},
                                     "averaging_gate": {"quantitative_passed": not group["failures"],
                                                        "texture_registration_passed": review["texture_registration_passed"],
                                                        "reasons": group["failures"]}},
                    frame_average={"enabled": averaged, "near_count": len(chosen), "far_count": len(chosen),
                                   "fallback_reason": group["failures"] or (None if averaged else "texture review failed")},
                    root_visual_review=review, numeric_storage="memory only; NPZ saving prohibited",
                    source_arrays={"provenance": str(group["archive"].resolve()), "read_only": True},
                    limitations=["Marker RMS is not fiber-level accuracy; far focus and resolution differ."],
                    verification={"npz_saved": False, "sha256_executed": False, "trace_matches_estimator": True})
    manifest["model_dtype_audit"] = {"parameters": sorted({str(p.dtype) for p in model.net_g.parameters()})}
    ranges = {key: list(_make_range(key, {"both": arrays}, {}, manifest)) for key in display_keys}
    _save_display_previews(output, ranges, conditions, display_keys, {"both": arrays}, manifest)
    common_stats = {key: _stats(value) for key, value in common.items()}
    for key, value in common.items():
        ranges[key] = [0, 1]
        for gamma in (False, True):
            folder = "common_display_gamma22" if gamma else "common_display"
            _render_array(value, key, output / folder / f"{key}.png", (0, 1), "gray", gamma22=gamma)
    galleries = _figure_gallery(output, {"both": arrays}, manifest["model_contract"])
    _save_prediction_component_previews(output, {"both": arrays})
    manifest.update(display_ranges=ranges, common_array_stats=common_stats,
                    report={"static_figures": galleries, "display_only": True, "numeric_storage": "memory only"})
    _json(output / "manifest.json", manifest)
    _json(output / "verification.json", manifest["verification"])
    configs = _display_configs(display_keys, ranges, {"both": arrays}, common_stats)
    page = _make_html(manifest, display_keys, ranges, configs, common_stats, galleries)
    # This mode has no saved numerical arrays: do not reuse the disk-mode claim.
    page = page.replace("저장 NPZ 값은", "메모리의 FP32 값은")
    page = page.replace("raw 10 channels는 NPZ에 그대로 남습니다.", "raw 10채널은 추론 중 메모리에서만 유지하며 저장하지 않습니다.")
    page = page.replace("NPZ", "메모리 배열(저장 없음)")
    notice = '<p style="padding:10px;background:#263b27">2026-10-06 새 정합·FP32 재추론. NPZ 저장 없음. 모든 선형 RGB 표시 [0,1]. <a href="registration.json">정합 수치</a> · <a href="registration_contact.png">10장 정합 검토</a></p>'
    page = page.replace("<main>", "<main>" + notice, 1)
    (output / "report.html").write_text(page, encoding="utf-8")
    wrapper = page.replace('<meta charset="utf-8">', '<meta charset="utf-8"><base href="' + output.name + '/">', 1)
    (output.parent / "report.html").write_text(wrapper, encoding="utf-8")
    publish_memory_records(output, manifest)
    for path, identity in group["protected"].items():
        p = Path(path)
        if identity != (p.stat().st_size, p.stat().st_mtime_ns):
            raise AssertionError(f"source archive changed: {p}")
    if list(output.rglob("*.npz")):
        raise AssertionError("memory refresh must never save NPZ")
    return {"group": output.parent.name, "source": manifest["source"], "status": "complete",
            "report_html": output.parent.name + "/report.html", "frame_average": manifest["frame_average"],
            "selected_frames": manifest["selected_frames"], "npz_saved": False}


def refresh_capture_memory(output: Path, archive_root: Path, groups: tuple[str, ...], checkpoint: Path,
                           *, device: str = "cuda") -> list[dict]:
    """Prepare all groups, await recorded reviews, and refresh their reports."""
    if checkpoint.resolve() != Path("checkpoints/net_g_real.pth").resolve():
        raise ValueError("memory refresh accepts only the original repository net_g_real.pth")
    prepared = [prepare_memory_group(archive_root / group, output / group / "refresh_20261006")
                for group in groups]
    print("REVIEW_READY all contact/checkerboard PNGs written; record root_review.json for every group", flush=True)
    deadline = time.monotonic() + 900
    while any(json.loads((group["output"] / "root_review.json").read_text(encoding="utf-8")).get("status") != "reviewed"
              for group in prepared):
        if time.monotonic() > deadline:
            raise TimeoutError("visual review was not recorded; inference has not run")
        time.sleep(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = AuthorRealCaptureAdapter(checkpoint, device=device)
    records = []
    for group in prepared:
        records.append(infer_memory_group(group, model))
        group["counts"].clear()
        group["previews"].clear()
        print("REFRESH_COMPLETE " + group["output"].parent.name, flush=True)
    write_capture_index(output, records)
    return records
