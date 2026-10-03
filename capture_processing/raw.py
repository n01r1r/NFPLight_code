"""Rawpy DNG unpacking with an explicit float32 capture contract."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rawpy

from .demosaic import demosaic_ahd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RawCapture:
    """Decoded DNG contents and copied metadata.

    ``bayer_u16`` is the visible decoded Bayer sample exactly as exposed by
    rawpy.  ``bayer_float`` and ``rgb_float`` use the capture computational
    dtype.
    """

    path: str
    sha256: str
    bayer_u16: np.ndarray
    bayer_float: np.ndarray
    cfa: np.ndarray
    colors_visible: np.ndarray
    rgb_float: np.ndarray
    metadata: dict[str, Any]


def unpack_dng(path: str | Path, *, dtype: np.dtype | type = np.float32) -> RawCapture:
    """Unpack a 2x2 Bayer DNG and demosaic it in float32.

    No ``postprocess`` call is made: rawpy's integer RGB output would violate
    the precision contract.  The function intentionally fails if the AHD
    implementation is unavailable instead of falling back to another method.
    """

    source = Path(path)
    precision = np.dtype(dtype)
    if precision != np.dtype(np.float32):
        raise TypeError("capture unpack requires dtype float32")
    digest = sha256_file(source)
    with rawpy.imread(str(source)) as raw:
        bayer_u16 = np.asarray(raw.raw_image_visible).copy()
        if bayer_u16.ndim != 2 or bayer_u16.dtype != np.uint16:
            raise TypeError(f"expected visible uint16 Bayer data, got {bayer_u16.dtype} {bayer_u16.shape}")
        cfa = np.asarray(raw.raw_pattern).copy()
        colors_visible = np.asarray(raw.raw_colors_visible).copy()
        if cfa.shape != (2, 2):
            raise ValueError(f"only 2x2 Bayer CFA is supported, got {cfa.shape}")
        if colors_visible.shape != bayer_u16.shape:
            raise ValueError("raw CFA mask and visible Bayer shape differ")
        color_desc = raw.color_desc.decode("ascii") if isinstance(raw.color_desc, bytes) else str(raw.color_desc)
        labels = np.frombuffer(color_desc.encode("ascii"), dtype=np.uint8)
        if colors_visible.shape[0] < 2 or colors_visible.shape[1] < 2:
            raise ValueError("visible Bayer area is too small for a 2x2 CFA")
        visible_indices = np.asarray(colors_visible[:2, :2], dtype=np.int64)
        # raw_pattern is defined in the full raw coordinate origin; visible
        # margins may shift the phase, so demosaic from the visible mask.
        cfa_rgb = labels[visible_indices]
        cfa_rgb = np.where(cfa_rgb == ord("R"), 0, np.where(cfa_rgb == ord("G"), 1, 2)).astype(np.int64)
        expected = np.tile(cfa_rgb, (bayer_u16.shape[0] // 2 + 1, bayer_u16.shape[1] // 2 + 1))[:bayer_u16.shape[0], :bayer_u16.shape[1]]
        observed = np.where(np.isin(labels[colors_visible.astype(np.int64)], [ord("R"), ord("G"), ord("B")]), labels[colors_visible.astype(np.int64)], 0)
        observed = np.where(observed == ord("R"), 0, np.where(observed == ord("G"), 1, 2)).astype(np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError("visible CFA mask is not periodic 2x2")
        metadata: dict[str, Any] = {
            "path": str(source.resolve()),
            "sha256": digest,
            "raw_shape": list(bayer_u16.shape),
            "raw_dtype": str(bayer_u16.dtype),
            "raw_min": int(bayer_u16.min()),
            "raw_max": int(bayer_u16.max()),
            "cfa_pattern": cfa.tolist(),
            "cfa_rgb_indices": cfa_rgb.tolist(),
            "visible_cfa_indices": visible_indices.tolist(),
            "visible_origin": [int(raw.sizes.top_margin), int(raw.sizes.left_margin)],
            "color_desc": color_desc,
            "black_level_per_channel": [int(v) for v in raw.black_level_per_channel],
            "white_level": int(raw.white_level),
            "camera_whitebalance": np.asarray(raw.camera_whitebalance, dtype=np.float32).tolist(),
            "color_matrix": np.asarray(raw.color_matrix, dtype=np.float32).tolist(),
            "orientation_flip": int(raw.sizes.flip),
            "orientation_applied": False,
        }
    bayer_float = bayer_u16.astype(precision)
    if not np.array_equal(bayer_float, bayer_u16):
        raise AssertionError("uint16 to float32 conversion changed a Bayer sample")
    camera_matrix = np.asarray(metadata["color_matrix"], dtype=precision)
    camera_matrix = camera_matrix[:, :3] if camera_matrix.ndim == 2 else None
    if camera_matrix is not None and camera_matrix.shape != (3, 3):
        camera_matrix = None
    rgb_float = demosaic_ahd(
        bayer_float,
        cfa_rgb,
        camera_to_linear_srgb=camera_matrix,
        dtype=precision,
    )
    if not isinstance(rgb_float, np.ndarray) or rgb_float.dtype != precision:
        raise TypeError(f"AHD must return {precision}, got {type(rgb_float).__name__} {getattr(rgb_float, 'dtype', None)}")
    if rgb_float.shape != (*bayer_u16.shape, 3):
        raise ValueError(f"AHD must return HWC RGB, got {rgb_float.shape}")
    if not np.isfinite(rgb_float).all():
        raise FloatingPointError("FP32 AHD returned non-finite values")
    metadata["demosaic"] = {
        "method": "AHD",
        "arithmetic": str(precision),
        "camera_matrix_for_lab": camera_matrix.tolist() if camera_matrix is not None else None,
        "native_sample_policy": "measured CFA samples retained by the implementation",
    }
    return RawCapture(str(source.resolve()), digest, bayer_u16, bayer_float,
                      cfa, colors_visible, rgb_float, metadata)


def libraw_ahd_reference(path: str | Path) -> np.ndarray:
    """Return the uint16 LibRaw AHD result for a comparison-only audit.

    This function is intentionally outside the authoritative float32 input
    path.  Its integer output is used only to quantify implementation and
    rounding differences against :func:`unpack_dng`.
    """

    params = dict(
        gamma=(1, 1), no_auto_bright=True, no_auto_scale=True,
        use_camera_wb=False, use_auto_wb=False, output_color=rawpy.ColorSpace.raw,
        output_bps=16, user_flip=0, user_black=0,
        user_cblack=[0, 0, 0, 0], adjust_maximum_thr=0,
        demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
    )
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(**params)
    if rgb.dtype != np.uint16 or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise TypeError(f"LibRaw AHD comparison expected uint16 HWC RGB, got {rgb.dtype} {rgb.shape}")
    return np.ascontiguousarray(rgb)


__all__ = ["RawCapture", "sha256_file", "unpack_dng", "libraw_ahd_reference"]
