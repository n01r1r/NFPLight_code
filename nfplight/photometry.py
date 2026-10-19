"""Float32 black/white compensation and linear camera colour operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rawpy
from scipy.linalg.lapack import get_lapack_funcs


CONDITIONS = {
    "both": ("(D - B) / (W - B)", True, True),
}


def _single_matrix_rank(matrix: np.ndarray) -> int:
    """Compute rank with single precision LAPACK ``sgesdd``."""
    values = np.array(matrix, dtype=np.float32, order="F", copy=True)
    gesdd = get_lapack_funcs("gesdd", (values,))
    if gesdd.typecode != "s":
        raise RuntimeError("float32 color matrices require sgesdd LAPACK")
    _, singular_values, _, info = gesdd(values, compute_uv=0, full_matrices=0, overwrite_a=1)
    if info < 0:
        raise ValueError(f"single matrix-rank SVD rejected argument {-info}")
    if info > 0:
        raise ValueError("single matrix-rank SVD failed to converge")
    if singular_values.size == 0:
        return 0
    threshold = np.finfo(np.float32).eps * max(values.shape) * singular_values[0]
    return int(np.count_nonzero(singular_values > threshold))


def apply_compensation(counts: np.ndarray, black: float, white: float, condition: str,
                       *, dtype: np.dtype | type | None = None) -> np.ndarray:
    """Apply one specified condition to HWC counts, retaining signed values."""

    if condition not in CONDITIONS:
        raise ValueError(
            f"unsupported compensation condition {condition!r}; only 'both' is active "
            "(D - B) / (W - B); 'white_only' and 'neither' are retired"
        )
    if not (np.isfinite(black) and np.isfinite(white) and 0 <= black < white):
        raise ValueError("expected finite 0 <= black < white")
    values = np.asarray(counts)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError("counts must be numeric")
    if np.issubdtype(values.dtype, np.floating) and values.dtype != np.dtype(np.float32):
        raise TypeError(f"capture compensation requires float32 counts, got {values.dtype}")
    precision = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
    if precision != np.dtype(np.float32):
        raise TypeError("capture compensation requires dtype float32")
    values = values.astype(precision, copy=False)
    result = (values - float(black)) / float(white - black)
    if not np.isfinite(result).all():
        raise FloatingPointError("compensation produced non-finite values")
    return np.ascontiguousarray(result, dtype=precision)


def read_color_transform(path: str | Path) -> dict[str, Any]:
    """Read camera WB and the embedded rawpy camera matrix without conversion."""

    with rawpy.imread(str(path)) as raw:
        wb4 = np.asarray(raw.camera_whitebalance, dtype=np.float32).copy()
        matrix4 = np.asarray(raw.color_matrix, dtype=np.float32).copy()
        color_desc = raw.color_desc.decode("ascii") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
    if color_desc != "RGBG" or wb4.size < 3 or np.any(wb4[:3] <= 0) or not np.isfinite(wb4).all():
        raise ValueError("expected finite positive RGBG camera white balance")
    if matrix4.ndim != 2 or matrix4.shape[0] < 3 or matrix4.shape[1] < 3:
        raise ValueError(f"invalid camera color matrix shape: {matrix4.shape}")
    matrix = matrix4[:, :3]
    if not np.isfinite(matrix).all() or _single_matrix_rank(matrix) != 3:
        raise ValueError("camera color matrix must be finite and rank 3")
    wb = wb4[:3] / wb4[:3].min()
    return {
        "camera_wb": wb4.tolist(),
        "normalized_rgb_wb": wb.tolist(),
        "camera_to_linear_srgb": matrix.tolist(),
        "matrix_source": "rawpy.color_matrix / LibRaw DNG cmatrix",
        "wb_normalization": "positive RGB minimum = 1",
        "gamma_applied": False,
        "icc_profile_transform": False,
        "arithmetic": "float32",
    }


def apply_linear_color(sensor_rgb: np.ndarray, transform: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return WB camera RGB and unclipped linear-sRGB RGB in input precision."""

    values = np.asarray(sensor_rgb)
    if values.dtype != np.dtype(np.float32):
        raise TypeError(f"linear color transform requires float32 input, got {values.dtype}")
    precision = values.dtype
    wb = np.asarray(transform["normalized_rgb_wb"], dtype=precision)
    matrix = np.asarray(transform["camera_to_linear_srgb"], dtype=precision)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"expected HWC RGB, got {values.shape}")
    balanced = values * wb
    linear = balanced @ matrix.T
    if not np.isfinite(linear).all():
        raise FloatingPointError("linear color transform produced non-finite values")
    return np.ascontiguousarray(balanced), np.ascontiguousarray(linear)


__all__ = ["CONDITIONS", "apply_compensation", "read_color_transform", "apply_linear_color"]
