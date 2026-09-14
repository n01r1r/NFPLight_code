"""Shared fixed decoders for the Fabric estimator normal output.

The estimator always stores ten normalized output channels.  A Fabric
training contract may select a fixed decoder for the first three channels;
the decoder has no parameters, so it must be applied identically during
training and inference.
"""

import math

import torch


POLE_EPS = 1e-6
NORMAL_HEADS = ("xyz", "phi_theta")


def apply_normal_head(prediction, *, normal_head="xyz"):
    """Decode a normal head while preserving the ``[B,10,H,W]`` contract.

    ``xyz`` is the established identity path.  ``phi_theta`` interprets the
    first three network channels as ``(cos(phi), sin(phi), theta_code)`` and
    reconstructs a unit normal on the nonnegative-z hemisphere.  At an
    undefined azimuth, phi is fixed to zero while theta remains supervised.
    """
    if normal_head == "xyz":
        return prediction
    if normal_head != "phi_theta":
        raise ValueError(f"unknown normal head: {normal_head}")
    if prediction.ndim != 4 or prediction.shape[1] != 10:
        raise ValueError("normal head expects [B,10,H,W] prediction")
    if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
        raise FloatingPointError(
            "normal head received a non-finite or non-floating prediction"
        )

    raw = prediction[:, :3]
    theta = (raw[:, 2:3].clamp(-1, 1) + 1) * (math.pi / 4)
    xy_radius = torch.linalg.vector_norm(raw[:, :2], dim=1, keepdim=True)
    defined = xy_radius >= POLE_EPS
    denominator = xy_radius.clamp_min(POLE_EPS)
    cos_phi = torch.where(
        defined, raw[:, 0:1] / denominator, torch.ones_like(denominator)
    )
    sin_phi = torch.where(
        defined, raw[:, 1:2] / denominator, torch.zeros_like(denominator)
    )
    normal = torch.cat(
        (theta.sin() * cos_phi, theta.sin() * sin_phi, theta.cos()), dim=1
    )
    return torch.cat((normal, prediction[:, 3:]), dim=1)


def checkpoint_normal_head(checkpoint):
    """Resolve the decoder recorded in a checkpoint.

    Weight-only checkpoints without a contract are legacy XYZ checkpoints.
    When both contract locations are present they must agree; malformed or
    unsupported values are rejected instead of silently guessing a decoder.
    """
    if not isinstance(checkpoint, dict):
        raise ValueError("network checkpoint must be a mapping")
    contract = checkpoint.get("contract")
    if contract is None:
        return "xyz"
    if not isinstance(contract, dict):
        raise ValueError("checkpoint contract must be a mapping")

    values = []
    for section_name, section_key in (
        ("settings", "normal_head"),
        ("normal_objective", "head"),
    ):
        section = contract.get(section_name)
        if section is None:
            continue
        if not isinstance(section, dict):
            raise ValueError(f"checkpoint contract {section_name} must be a mapping")
        if section_key not in section:
            continue
        value = section[section_key]
        if not isinstance(value, str) or value not in NORMAL_HEADS:
            raise ValueError(
                f"checkpoint contract {section_name}.{section_key} has unsupported "
                f"normal head: {value!r}"
            )
        values.append((f"{section_name}.{section_key}", value))

    if not values:
        # Older contracts did not record a head and therefore used the
        # established XYZ output semantics.
        return "xyz"
    selected = values[0][1]
    conflicts = [(name, value) for name, value in values if value != selected]
    if conflicts:
        details = ", ".join(f"{name}={value!r}" for name, value in values)
        raise ValueError(f"checkpoint normal-head contract conflicts: {details}")
    return selected
