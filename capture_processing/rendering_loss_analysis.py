"""Decode saved FP32 material maps; completed optimization code is archived."""

import torch


def _validate_prediction(prediction: torch.Tensor) -> None:
    if prediction.ndim != 4 or prediction.shape[1] != 10:
        raise ValueError(f"prediction must have shape [B,10,H,W], got {tuple(prediction.shape)}")
    if prediction.dtype != torch.float32:
        raise TypeError(f"prediction must be FP32, got {prediction.dtype}")
    if not torch.isfinite(prediction).all():
        raise ValueError("prediction contains non-finite values")
    normal_norm = torch.linalg.vector_norm(prediction[:, :3], dim=1)
    if (normal_norm <= 1e-8).any():
        raise ValueError("prediction contains a zero normal")
    for name, value in (
        ("diffuse", prediction[:, 3:6]),
        ("roughness", prediction[:, 6:7]),
        ("specular", prediction[:, 7:10]),
    ):
        if value.amin() < -1.00001 or value.amax() > 1.00001:
            raise ValueError(f"{name} raw values must be in [-1,1]")


def physical_maps(material: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decode renderer physical maps, including the renderer's roughness floor."""
    _validate_prediction(material)
    normal = material[:, :3] / torch.linalg.vector_norm(material[:, :3], dim=1, keepdim=True)
    return {
        "normal": normal,
        "diffuse": (material[:, 3:6] + 1.0) * 0.5,
        "roughness": ((material[:, 6:7] + 1.0) * 0.5).clamp(min=0.05, max=1.0),
        "specular": (material[:, 7:10] + 1.0) * 0.5,
    }
