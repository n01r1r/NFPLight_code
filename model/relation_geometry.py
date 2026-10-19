"""Shared center-normalized inverse-square relation geometry (FP32)."""
import math
import torch

RELATION_VERSION = "spatial_inverse_square_v1"

def spatial_attenuation_factor(distance_squared_near, distance_squared_far):
    """FP32 center-normalized r_far^2/r_near^2 from the s34/s15 report.

    Distances must be positive [B,1,H,W] tensors in common world units.
    The lower central pixel matches the report's [127,127] at size 256.
    The center brightness gain is already in aligned_far and is NOT
    multiplied again here. This factor corrects distance, not normals.
    """
    if (distance_squared_near.ndim != 4
            or distance_squared_near.shape[1] != 1
            or distance_squared_near.shape != distance_squared_far.shape
            or min(distance_squared_near.shape) <= 0):
        raise ValueError("squared distances require matching [B,1,H,W] shapes")
    if distance_squared_near.device != distance_squared_far.device:
        raise ValueError("squared distances must share a device")
    for distance in (distance_squared_near, distance_squared_far):
        if distance.dtype != torch.float32:
            raise TypeError("spatial attenuation requires FP32 squared distances")
        if not torch.isfinite(distance).all() or torch.any(distance <= 0):
            raise ValueError("squared distances must be finite and positive")
    ratio = distance_squared_far / distance_squared_near
    cy, cx = [(size - 1) // 2 for size in ratio.shape[-2:]]
    factor = ratio / ratio[:, :, cy:cy+1, cx:cx+1]
    if not torch.isfinite(factor).all() or torch.any(factor <= 0):
        raise ValueError("spatial attenuation factor must be finite and positive")
    return factor


def planar_squared_distances(size, near, far, device):
    if size <= 0 or not (math.isfinite(near) and math.isfinite(far) and 0 < near < far):
        raise ValueError("require positive size and finite 0 < near < far")
    axis = (torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2) / (size / 2.0)
    rho_squared = (axis[:, None].square() + axis[None, :].square())[None, None]
    return rho_squared + near ** 2, rho_squared + far ** 2


def planar_spatial_factor(size, near, far, device):
    return spatial_attenuation_factor(*planar_squared_distances(size, near, far, device))


def spatial_far(aligned_far, angular, factor):
    # Preserve FP32 multiplication order and clip only after all corrections.
    return angular * (aligned_far * factor)
