"""Independent paper-equation checks retained for numerical diagnostics."""

from __future__ import annotations


import numpy as np



def verify_paper_equations(inputs: np.ndarray, paper: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute Eq. (4)/(5) independently using NumPy FP32 trig identities."""
    if inputs.shape != (256, 256, 6) or inputs.dtype != np.float32:
        raise ValueError("independent paper check requires HWC FP32 common input")
    if not np.isfinite(inputs).all() or inputs.min() < 0 or inputs.max() > 1:
        raise ValueError("independent paper check requires finite linear RGB in [0,1]")
    near, far = inputs[..., :3].mean(-1, keepdims=True), inputs[..., 3:].mean(-1, keepdims=True)
    if far[128, 128, 0] <= 0:
        raise ValueError("Eq. (4) is undefined for nonpositive far center intensity")
    gain = near[128, 128, 0] / far[128, 128, 0]
    axis = (np.arange(256, dtype=np.float32) - np.float32(127.5)) / np.float32(128)
    rho = np.sqrt(axis[:, None] ** 2 + axis[None, :] ** 2)[..., None]
    tn, tf = np.arctan(rho / np.float32(4)), np.arctan(rho / np.float32(12))
    cn, cf = np.cos(tn), np.cos(tf)
    # Independent trig form avoids cancellation without the production
    # reciprocal-length/conjugate formula.
    cm = cn * np.float32(2) * np.sin((tn + tf) / np.float32(2)) * np.sin((tn - tf) / np.float32(2))
    far_scaled = gain * (cn / cf) * far
    signed = near - far_scaled
    numerator = np.abs(signed)
    expected = {"paper_gain": np.asarray(gain, dtype=np.float32),
                "paper_near_intensity": near, "paper_far_intensity": far,
                "paper_time_map": cn / cf, "paper_far_scaled_intensity": far_scaled,
                "paper_signed_difference": signed,
                "paper_denominator": cm, "paper_numerator": numerator,
                "paper_relation_raw": numerator / cm}
    errors = {}
    for key, value in expected.items():
        actual = np.asarray(paper[key])
        if actual.dtype != np.float32 or not np.isfinite(actual).all():
            raise AssertionError(f"invalid FP32 paper array: {key}")
        actual = actual.reshape(value.shape)
        delta = np.abs(actual - value)
        # Bound FP32 radiance cancellation before division: near-axis RM is
        # ill-conditioned. Do not use a denominator floor to hide that fact.
        if key == "paper_relation_raw":
            tolerance = np.float32(4e-7) / cm + np.float32(5e-5) * np.abs(value)
        elif key == "paper_denominator":
            tolerance = np.float32(5e-12) + np.float32(5e-6) * np.abs(value)
        else:
            tolerance = np.float32(4e-7) + np.float32(5e-5) * np.abs(value)
        if not np.all(delta <= tolerance):
            raise AssertionError(f"independent paper formula mismatch: {key}, max error {delta.max()}")
        errors[key] = {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())}
    return {"passed": True, "arithmetic": "numpy.float32", "errors": errors,
            "radiance_absolute_error_bound": 4e-7,
            "relation_error_bound": "4e-7 / C_M + 5e-5 * abs(RM); no raw denominator floor",
            "near_axis_is_ill_conditioned": True}


def _difference(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    delta = np.asarray(b - a, dtype=np.float32)
    return {"mae": float(np.abs(delta).mean(dtype=np.float32)),
            "rmse": float(np.sqrt(np.square(delta).mean(dtype=np.float32))),
            "max_abs": float(np.abs(delta).max())}
