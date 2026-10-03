"""Independent float32 relation and feature trace for the fixed estimator."""

from __future__ import annotations

from typing import Any

import torch


def assemble_feature_trace(model: Any, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Recompute model features term by term and verify exact equivalence.

    The implementation mirrors the fixed checkpoint contract in
    ``NFPLightModel.build_features`` while exposing every intermediate needed
    by the report.  It intentionally requires a floating tensor whose dtype
    matches the model's configured compute dtype.
    """

    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (6, 256, 256):
        raise ValueError(f"expected [B,6,256,256], got {tuple(inputs.shape)}")
    if not inputs.is_floating_point() or not torch.isfinite(inputs).all():
        raise ValueError("inputs must be finite floating-point values")
    dtype = getattr(model, "compute_dtype", inputs.dtype)
    if inputs.dtype != dtype:
        raise TypeError(f"trace dtype {inputs.dtype} does not match model dtype {dtype}")
    near, far = inputs.chunk(2, dim=1)
    if getattr(model, "feature_version", "sample_v1") == "raw_calibrated_v2":
        gain_scalar = (model.far_distance / model.near_distance) ** 2
        gain = near.new_full((len(near), 1, 1, 1), gain_scalar)
    else:
        gain = (near[:, :, 127, 127].mean(1) /
                far[:, :, 127, 127].mean(1).clamp(min=1e-5))[:, None, None, None]
    far_gained = far * gain
    far_time = model.time_co_map * far_gained
    far_time_clipped = far_time.clamp(0, 1)
    signed_diff = far_time_clipped - near
    abs_diff = signed_diff.abs()
    mean_abs_diff = abs_diff.mean(1, keepdim=True)
    coefficient = model.coefficient.expand(len(near), -1, -1, -1)
    denominator = coefficient.clamp(min=1e-5)
    relation_raw = mean_abs_diff / denominator
    relation_floored = relation_raw.clamp(min=1e-5)
    relation_log_raw = relation_floored.log()
    relation_normalized = model.ClipToOne(relation_raw)
    relation_log_normalized = model.ClipToOne(relation_log_raw)
    far_clipped = far_gained.clamp(0, 1)
    mask = model.maskExtract(near, far_clipped)
    appearance = torch.cat((near, far, far_clipped), dim=1)
    appearance_log = model.log_normalization(appearance)
    relation = model.indenty - relation_normalized
    relation_log = model.indenty - relation_log_normalized
    features = torch.cat((appearance, appearance_log, relation, relation_log, mask), dim=1) * 2 - 1
    result = {
        "near": near,
        "far": far,
        "far_gain_map": gain.expand_as(near[:, :1]),
        "far_gained_unclipped": far_gained,
        "far_gained_clipped": far_clipped,
        "far_gain_clip_delta": far_clipped - far_gained,
        "time_co_map": model.time_co_map.expand_as(near[:, :1]),
        "far_time_unclipped": far_time,
        "far_time_clipped": far_time_clipped,
        "time_clip_delta": far_time_clipped - far_time,
        "signed_diff": signed_diff,
        "abs_diff": abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "coefficient": coefficient,
        "coefficient_raw": getattr(model, "coefficient_raw", coefficient),
        "coefficient_normalization_max": getattr(
            model, "coefficient_normalization_max", coefficient.new_tensor(1.0)
        ).expand_as(coefficient),
        "denominator": denominator,
        "relation_raw": relation_raw,
        "relation_floored": relation_floored,
        "relation_log_raw": relation_log_raw,
        "relation_normalized": relation_normalized,
        "relation_log_normalized": relation_log_normalized,
        "relation": relation,
        "relation_log": relation_log,
        "valid_mask": mask,
        "log_near": appearance_log[:, :3],
        "log_far": appearance_log[:, 3:6],
        "log_far_gained": appearance_log[:, 6:9],
        "features": features,
    }
    if hasattr(model, "coefficient_raw"):
        result["coefficient_raw"] = model.coefficient_raw.expand_as(near[:, :1])
        result["coefficient_normalization_max"] = model.coefficient_normalization_max.expand_as(near[:, :1])
    expected, expected_log, expected_far = model.build_features(inputs)
    for label, observed, actual in (
        ("features", features, expected),
        ("log_relation", relation_log, expected_log),
        ("aligned_far", far_clipped, expected_far),
    ):
        if observed.dtype != inputs.dtype or actual.dtype != inputs.dtype:
            raise AssertionError(f"{label} lost float32 dtype")
        if not torch.equal(observed, actual):
            maximum = (observed - actual).abs().max().item()
            raise AssertionError(f"independent {label} trace differs from model: {maximum}")
    for label, value in result.items():
        if value.dtype != inputs.dtype:
            raise AssertionError(f"trace {label} has dtype {value.dtype}, expected {inputs.dtype}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"non-finite feature trace: {label}")
    return result


__all__ = ["assemble_feature_trace"]
