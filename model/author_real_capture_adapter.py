"""Strict FP32 adapter for the authors' original real-capture checkpoint.

Source contract: upstream ``cd16bab`` files ``real.py``,
``model/nfplight_real_model.py``, and ``network/nfplight_net.py``. This
user-approved capture path excludes denoising and identity-copies the original
input into the historical copy-feature slots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from network.nfplight_net import TwoBranchNet
from .paper_equations import paper_relation


IMAGE_SIZE = 256
FEATURE_VERSION = "legacy_batch_v0"
NORMAL_HEAD = "author_legacy_tanh"
LOG_EPSILON = 1e-2
SATURATION_THRESHOLD = 0.95
GAIN_PATCH = (slice(118, 138), slice(118, 138))
GEOMETRY_SETUP_DTYPE = "numpy.float64"
GEOMETRY_BUFFER_DTYPE = "torch.float32"


class AuthorTwoBranchRealNet(TwoBranchNet):
    """Original 30+3 input network, reusing the unchanged shared blocks."""

    def __init__(self, width=32, middle_blk_num=12,
                 enc_blk_nums=(2, 2, 4, 8), dec_blk_nums=(2, 2, 2, 2)):
        super().__init__(width, middle_blk_num, list(enc_blk_nums), list(dec_blk_nums))
        self.intro_albedo = nn.Conv2d(
            in_channels=30, out_channels=width, kernel_size=3, padding=1,
            stride=1, groups=1, bias=True,
        )

    def forward(self, inp):
        if inp.ndim != 4 or inp.shape[1] != 33:
            raise ValueError(f"author real estimator expects [B,33,H,W], got {tuple(inp.shape)}")
        _, _, height, width = inp.shape
        inp_albedo, inp_specular = torch.split(inp, [30, 3], dim=1)

        x = self.intro_albedo(inp_albedo)
        encs_albedo = []
        for encoder, down in zip(self.encoders_albedo, self.downs_albedo):
            x = encoder(x)
            encs_albedo.append(x)
            x = down(x)
        feature_albedo = self.middle_blks_albedo(x)

        x = self.intro_specular(inp_specular)
        encs_specular = []
        for encoder, down in zip(self.encoders_specular, self.downs_specular):
            x = encoder(x)
            encs_specular.append(x)
            x = down(x)
        feature_specular = self.middle_blks_specular(x)

        x = feature_albedo
        for decoder, up, enc_skip in zip(
            self.decoders_albedo, self.ups_albedo, encs_albedo[::-1]
        ):
            x = decoder(up(x) + enc_skip)
        albedo_map = self.albedoMap(x)

        x = torch.cat([feature_specular, feature_albedo], dim=1)
        for decoder, up, enc_skip_albedo, enc_skip_specular in zip(
            self.decoders_specular, self.ups_specular,
            encs_albedo[::-1], encs_specular[::-1],
        ):
            skip = torch.cat([enc_skip_albedo, enc_skip_specular], dim=1)
            x = decoder(up(x) + skip)
        specular_map = self.specularMap(x)

        diffuse, specular = torch.split(albedo_map, [3, 3], dim=1)
        normal, roughness = torch.split(specular_map, [3, 1], dim=1)
        prediction = torch.cat([normal, diffuse, roughness, specular], dim=1)
        return prediction[:, :, :height, :width]


def log_normalization(image: torch.Tensor, eps: float = LOG_EPSILON) -> torch.Tensor:
    """Exact scalar-tensor form used by the original CUDA real model."""
    one = torch.ones((1,), dtype=image.dtype, device=image.device)
    epsilon = one * eps
    return (torch.log(image + eps) - torch.log(epsilon)) / (
        torch.log(one + epsilon) - torch.log(epsilon)
    )


def legacy_geometry(image_size: int, device: torch.device | str) -> tuple[torch.Tensor, ...]:
    """Original NumPy-float64 geometry formulas stored in FP32 torch buffers."""
    y, x = np.ogrid[:image_size, :image_size]
    center_y = center_x = (image_size - 1) / 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / (image_size / 2.0)
    angle_near = np.arctan(distance / 4.0)
    angle_far = np.arctan(distance / 12.0)
    coefficient_raw = (np.cos(angle_far) - np.cos(angle_near)) / np.cos(angle_near)
    coefficient_max = np.max(coefficient_raw)
    coefficient = coefficient_raw / coefficient_max
    time_coefficient = np.cos(angle_near) / np.cos(angle_far)
    tensor_args = {"dtype": torch.float32, "device": device}
    coefficient_tensor = torch.as_tensor(coefficient, **tensor_args)[None, None]
    time_tensor = torch.as_tensor(time_coefficient, **tensor_args)[None, None]
    max_tensor = torch.as_tensor(coefficient_max, **tensor_args)
    return coefficient_tensor, time_tensor, max_tensor


def clip_to_one(feature: torch.Tensor) -> torch.Tensor:
    low, high = feature.amin(), feature.amax()
    return (feature - low) / (high - low).clamp(min=1e-5)


def assemble_legacy33_features(
    original: torch.Tensor,
    original_copy: torch.Tensor,
    coefficient: torch.Tensor,
    time_coefficient: torch.Tensor,
    coefficient_max: torch.Tensor,
    *, equation_mode: str = "author_code",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build and expose the authors' exact 33 channels from one FP32 pair."""
    if original.shape != (1, 6, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"expected original [1,6,256,256], got {tuple(original.shape)}")
    if original_copy.shape != original.shape:
        raise ValueError(f"expected original-copy pair {tuple(original.shape)}, got {tuple(original_copy.shape)}")
    if original.dtype != torch.float32 or original_copy.dtype != torch.float32:
        raise TypeError("author legacy33 original and original-copy input must remain FP32")
    if not torch.isfinite(original).all() or not torch.isfinite(original_copy).all():
        raise FloatingPointError("author legacy33 features received non-finite values")

    if equation_mode not in {"author_code", "paper_equations"}:
        raise ValueError(f"unknown equation mode: {equation_mode}")
    near, far = original_copy.split(3, dim=1)
    paper = paper_relation(original_copy) if equation_mode == "paper_equations" else None
    if paper is not None:
        gain = paper["paper_gain"]
        coefficient = paper["paper_denominator"]
        coefficient_max = coefficient.amax()
        time_coefficient = paper["paper_time_map"]
    else:
        near_mean = near[:, :, GAIN_PATCH[0], GAIN_PATCH[1]].mean()
        far_mean = far[:, :, GAIN_PATCH[0], GAIN_PATCH[1]].mean().clamp(min=1e-5)
        gain = (near_mean / far_mean).reshape(1, 1, 1, 1)
    far_gained_unclipped = far * gain
    far_time_unclipped = time_coefficient * far_gained_unclipped
    far_time_clipped = far_time_unclipped.clamp(0, 1)
    original_copy_near = near
    # Paper RM uses scalar means before subtraction/abs, not RGB diagnostics.
    signed_diff = paper["paper_signed_difference"] if paper is not None else far_time_clipped - original_copy_near
    abs_diff = signed_diff.abs()
    mean_abs_diff = paper["paper_numerator"] if paper is not None else abs_diff.mean(dim=1, keepdim=True)
    denominator = coefficient if paper is not None else coefficient.clamp(min=1e-5)
    relation_raw = paper["paper_relation_raw"] if paper is not None else mean_abs_diff / denominator
    relation_floored = relation_raw.clamp(min=1e-5)
    relation_log_raw = torch.log(relation_floored)
    relation = 1.0 - clip_to_one(relation_raw)
    relation_log = 1.0 - clip_to_one(relation_log_raw)
    far_gained_clipped = far_gained_unclipped.clamp(0, 1)

    original_near, original_far = original.split(3, dim=1)
    saturated = (
        (original_near.amax(dim=1, keepdim=True) > SATURATION_THRESHOLD)
        | (original_far.amax(dim=1, keepdim=True) > SATURATION_THRESHOLD)
        | (near.amax(dim=1, keepdim=True) > SATURATION_THRESHOLD)
        | (far_gained_clipped.amax(dim=1, keepdim=True) > SATURATION_THRESHOLD)
    )
    non_saturated_mask = (~saturated).to(dtype=torch.float32)

    appearance = torch.cat([original, original_copy, far_gained_clipped], dim=1)
    log_appearance = log_normalization(appearance)
    unscaled_features = torch.cat(
        [appearance, log_appearance, relation, relation_log, non_saturated_mask], dim=1
    )
    features = unscaled_features * 2 - 1
    if features.shape[1] != 33:
        raise AssertionError(f"legacy estimator feature count changed: {features.shape[1]}")

    trace = {
        "original_rgb_6": original,
        "original_copy_pair_6": original_copy,
        "far_gain_scalar": gain,
        "original_copy_far_gained_unclipped": far_gained_unclipped,
        "original_copy_far_gained_clipped": far_gained_clipped,
        "time_co_map": time_coefficient.expand_as(far_gained_clipped[:, :1]),
        "far_time_unclipped": far_time_unclipped,
        "far_time_clipped": far_time_clipped,
        "signed_diff": signed_diff,
        "abs_diff": abs_diff,
        "mean_abs_diff_rgb": mean_abs_diff,
        "coefficient": coefficient.expand_as(mean_abs_diff),
        "coefficient_normalization_max": coefficient_max.expand_as(mean_abs_diff),
        "denominator": denominator.expand_as(mean_abs_diff),
        "relation_raw": relation_raw,
        "relation_floored": relation_floored,
        "relation_log_raw": relation_log_raw,
        "relation_normalized": clip_to_one(relation_raw),
        "relation_log_normalized": clip_to_one(relation_log_raw),
        "relation": relation,
        "relation_log": relation_log,
        "non_saturated_mask": non_saturated_mask,
        "log_original_6": log_appearance[:, :6],
        "log_original_copy_6": log_appearance[:, 6:12],
        "log_original_copy_far_gained_3": log_appearance[:, 12:15],
        "features_unscaled_33": unscaled_features,
        "features_legacy33": features,
    }
    if paper is not None:
        trace.update(paper)
    for name, value in trace.items():
        if value.dtype != torch.float32:
            raise AssertionError(f"legacy trace {name} has dtype {value.dtype}, expected float32")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"legacy trace {name} contains non-finite values")
    return features, trace


class AuthorRealCaptureAdapter:
    """Strict-load the original real estimator without constructing a denoiser."""

    def __init__(self, estimator_checkpoint: Path, device: str = "cuda",
                 image_size: int = IMAGE_SIZE):
        if image_size != IMAGE_SIZE:
            raise ValueError("author legacy33 inference requires native 256x256 input")
        self.device = torch.device(device)
        self.compute_dtype = torch.float32
        self.feature_version = FEATURE_VERSION
        self.normal_head = NORMAL_HEAD
        self.geometry_setup_dtype = GEOMETRY_SETUP_DTYPE
        self.geometry_buffer_dtype = GEOMETRY_BUFFER_DTYPE
        self.near_distance = 4.0
        self.far_distance = 12.0
        (self.coefficient, self.time_co_map,
         self.coefficient_normalization_max) = legacy_geometry(image_size, self.device)
        self.net_g = AuthorTwoBranchRealNet().to(device=self.device, dtype=torch.float32)
        self._load_strict_fp32(self.net_g, estimator_checkpoint)
        self.net_g.eval()

    @staticmethod
    def _load_strict_fp32(model: nn.Module, checkpoint_path: Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("params") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise ValueError(f"original checkpoint must contain a params mapping: {checkpoint_path}")
        for key, value in state.items():
            if not torch.is_tensor(value) or value.dtype != torch.float32:
                raise TypeError(f"checkpoint tensor {key!r} must be FP32: {checkpoint_path}")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"checkpoint tensor {key!r} is non-finite: {checkpoint_path}")
        model.load_state_dict(state, strict=True)
        del payload, state

    @torch.no_grad()
    def infer(self, inputs: torch.Tensor, *, equation_mode: str = "author_code") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if tuple(inputs.shape) != (1, 6, IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"author inference expects [1,6,256,256], got {tuple(inputs.shape)}")
        if inputs.dtype != torch.float32 or not inputs.is_floating_point():
            raise TypeError("author inference input must be FP32 linear RGB")
        if not torch.isfinite(inputs).all() or inputs.amin() < 0 or inputs.amax() > 1:
            raise ValueError("author inference input must be finite linear RGB in [0,1]")
        inputs = inputs.to(device=self.device, dtype=torch.float32)
        # The approved no-denoiser path identity-copies the original pair into
        # the historical six copy-feature slots; no denoiser is constructed.
        original_copy = inputs.clamp(0, 1)
        features, trace = assemble_legacy33_features(
            inputs, original_copy, self.coefficient, self.time_co_map,
            self.coefficient_normalization_max,
            equation_mode=equation_mode,
        )
        self.last_estimator_input = features
        prediction = self.net_g(features)
        if prediction.dtype != torch.float32 or not torch.isfinite(prediction).all():
            raise FloatingPointError("author estimator produced invalid FP32 output")
        trace["prediction_raw_10"] = prediction
        trace["prediction_display_10"] = (prediction + 1) / 2
        return prediction, trace


__all__ = [
    "AuthorRealCaptureAdapter", "AuthorTwoBranchRealNet",
    "assemble_legacy33_features", "legacy_geometry", "log_normalization",
    "FEATURE_VERSION", "NORMAL_HEAD", "IMAGE_SIZE",
    "GEOMETRY_SETUP_DTYPE", "GEOMETRY_BUFFER_DTYPE",
]
