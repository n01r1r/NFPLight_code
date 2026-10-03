"""Printed NFPLight equations (4)–(5), before checkpoint feature adaptation."""

from __future__ import annotations

import torch


def paper_relation(inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Evaluate scalar RGB-mean radiances on the fixed 4/12, 256px geometry.

    No gained-far clipping, denominator normalization, or denominator epsilon.
    The RGB mean precedes subtraction and absolute value. The central sample
    is [128,128], following the printed [H/2,W/2] index. All math is FP32.
    """
    if inputs.shape != (1, 6, 256, 256) or inputs.dtype != torch.float32:
        raise ValueError("paper equations require FP32 [1,6,256,256]")
    if not torch.isfinite(inputs).all() or inputs.amin() < 0 or inputs.amax() > 1:
        raise ValueError("paper equations require finite common linear RGB in [0,1]")
    near_rgb, far_rgb = inputs.split(3, dim=1)
    near = near_rgb.mean(dim=1, keepdim=True)
    far = far_rgb.mean(dim=1, keepdim=True)
    near_center, far_center = near[:, :, 128:129, 128:129], far[:, :, 128:129, 128:129]
    if far_center.item() <= 0:
        raise ValueError("Eq. (4) is undefined for nonpositive far center intensity")
    gain = near_center / far_center

    axis = (torch.arange(256, device=inputs.device, dtype=torch.float32) - 127.5) / 128.0
    rho_squared = (axis[:, None].square() + axis[None, :].square())[None, None]
    near_length = torch.sqrt(1.0 + rho_squared / 16.0)
    far_length = torch.sqrt(1.0 + rho_squared / 144.0)
    cos_near, cos_far = near_length.reciprocal(), far_length.reciprocal()
    # Conjugate form of cos_far - cos_near avoids subtracting almost equal
    # FP32 cosines close to the central axis. It is algebraically identical.
    cosine_difference = (rho_squared * (1.0 / 16.0 - 1.0 / 144.0)
                         / (near_length * far_length * (near_length + far_length)))
    denominator = cos_near * cosine_difference
    if not torch.isfinite(denominator).all() or torch.any(denominator <= 0):
        raise ValueError("Eq. (5) requires finite positive non-axis denominator")
    time_map = cos_near / cos_far
    far_scaled = gain * time_map * far
    signed_difference = near - far_scaled
    numerator = signed_difference.abs()
    relation = numerator / denominator
    values = {
        "paper_near_intensity": near,
        "paper_far_intensity": far,
        "paper_near_center": near_center,
        "paper_far_center": far_center,
        "paper_gain": gain,
        "paper_cos_near": cos_near,
        "paper_cos_far": cos_far,
        "paper_time_map": time_map,
        "paper_denominator": denominator,
        "paper_far_scaled_intensity": far_scaled,
        "paper_signed_difference": signed_difference,
        "paper_numerator": numerator,
        "paper_relation_raw": relation,
    }
    if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values.values()):
        raise FloatingPointError("printed paper equations produced invalid FP32 values")
    return values
