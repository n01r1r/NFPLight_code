"""Marker geometry and deterministic float32 spatial resampling."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.linalg import LinAlgError
from scipy.linalg.lapack import get_lapack_funcs


def _single_inverse(matrix: np.ndarray) -> np.ndarray:
    """Invert with the single precision LAPACK ``sgetrf``/``sgetri`` path."""
    values = np.array(matrix, dtype=np.float32, order="F", copy=True)
    getrf = get_lapack_funcs("getrf", (values,))
    getri = get_lapack_funcs("getri", (values,))
    if getrf.typecode != "s" or getri.typecode != "s":
        raise RuntimeError("float32 geometry requires sgetrf/sgetri LAPACK")
    lu, pivots, info = getrf(values, overwrite_a=1)
    if info < 0:
        raise ValueError(f"single LU factorization rejected argument {-info}")
    if info > 0:
        raise LinAlgError("homography is singular")
    inverse, info = getri(lu, pivots, overwrite_lu=1)
    if info < 0:
        raise ValueError(f"single LU inverse rejected argument {-info}")
    if info > 0:
        raise LinAlgError("homography is singular")
    return np.ascontiguousarray(inverse, dtype=np.float32)


def _single_solve(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve with the single precision LAPACK ``sgesv`` path."""
    lhs = np.array(matrix, dtype=np.float32, order="F", copy=True)
    rhs = np.array(values, dtype=np.float32, order="F", copy=True)
    gesv = get_lapack_funcs("gesv", (lhs, rhs))
    if gesv.typecode != "s":
        raise RuntimeError("float32 geometry requires sgesv LAPACK")
    _, _, solution, info = gesv(lhs, rhs, overwrite_a=1, overwrite_b=1)
    if info < 0:
        raise ValueError(f"single linear solve rejected argument {-info}")
    if info > 0:
        raise LinAlgError("homography solve is singular")
    return np.ascontiguousarray(solution, dtype=np.float32)


def _single_least_squares(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Solve an overdetermined system with the single precision ``sgels`` path."""
    lhs = np.array(matrix, dtype=np.float32, order="F", copy=True)
    rhs = np.array(values, dtype=np.float32, order="F", copy=True)
    if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] < lhs.shape[1]:
        raise ValueError("least-squares system must be 2D and overdetermined")
    if rhs.shape[0] != lhs.shape[0]:
        raise ValueError("least-squares right-hand side row count differs")
    gels = get_lapack_funcs("gels", (lhs, rhs))
    if gels.typecode != "s":
        raise RuntimeError("float32 geometry requires sgels LAPACK")
    _, solution, info = gels(lhs, rhs, overwrite_a=1, overwrite_b=1)
    if info < 0:
        raise ValueError(f"single least-squares solve rejected argument {-info}")
    if info > 0:
        raise LinAlgError("least-squares system is rank deficient")
    return np.ascontiguousarray(solution[:lhs.shape[1]], dtype=np.float32)


def _validate_hwc(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim not in (2, 3) or min(values.shape[:2]) < 2:
        raise ValueError(f"expected 2D or HWC image, got {values.shape}")
    if values.dtype != np.dtype(np.float32):
        raise TypeError(f"capture spatial processing requires float32 input, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("spatial input contains non-finite values")
    if values.ndim == 3 and values.shape[2] < 1:
        raise ValueError("HWC image has no channels")
    return np.ascontiguousarray(values)


def warp_perspective(image: np.ndarray, homography: np.ndarray,
                         output_shape: tuple[int, int] = (512, 512),
                         *, return_valid: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Inverse-map a source image with explicit FP32 bilinear weights.

    ``homography`` maps source pixel coordinates to destination pixel
    coordinates, matching ``cv2.getPerspectiveTransform``.  Output and source
    coordinates use integer pixel centers; outside samples are zero and marked
    invalid.  OpenCV is used only for matrix validation, never interpolation.
    """

    source = _validate_hwc(image)
    h_out, w_out = (int(output_shape[0]), int(output_shape[1]))
    if h_out <= 0 or w_out <= 0:
        raise ValueError("output_shape must be positive")
    precision = np.dtype(np.float32)
    matrix = np.asarray(homography, dtype=precision)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-15:
        raise ValueError("homography must be finite 3x3 with nonzero scale")
    matrix = matrix / matrix[2, 2]
    try:
        inverse = _single_inverse(matrix)
    except LinAlgError as exc:
        raise ValueError("homography is singular") from exc
    yy, xx = np.indices((h_out, w_out), dtype=precision)
    destination = np.stack((xx.ravel(), yy.ravel(), np.ones(xx.size, dtype=precision)), axis=0)
    mapped = inverse @ destination
    denominator = mapped[2]
    valid_projective = np.isfinite(mapped).all(axis=0) & (np.abs(denominator) > 1e-15)
    sx = np.full(xx.size, np.nan, dtype=precision)
    sy = np.full(xx.size, np.nan, dtype=precision)
    sx[valid_projective] = mapped[0, valid_projective] / denominator[valid_projective]
    sy[valid_projective] = mapped[1, valid_projective] / denominator[valid_projective]
    h_src, w_src = source.shape[:2]
    inside = valid_projective & (sx >= 0.0) & (sx <= w_src - 1.0) & (sy >= 0.0) & (sy <= h_src - 1.0)
    # Clamp only after checking support, so an exact edge coordinate is safe.
    cast = precision.type
    sx_safe = np.clip(np.nan_to_num(sx, nan=cast(0.0)), cast(0.0), cast(max(w_src - 1.0, 0.0)))
    sy_safe = np.clip(np.nan_to_num(sy, nan=cast(0.0)), cast(0.0), cast(max(h_src - 1.0, 0.0)))
    x0 = np.floor(sx_safe).astype(np.int64)
    y0 = np.floor(sy_safe).astype(np.int64)
    x1 = np.minimum(x0 + 1, w_src - 1)
    y1 = np.minimum(y0 + 1, h_src - 1)
    wx = sx_safe - x0.astype(precision)
    wy = sy_safe - y0.astype(precision)
    if source.ndim == 2:
        v00, v01 = source[y0, x0], source[y0, x1]
        v10, v11 = source[y1, x0], source[y1, x1]
        values = ((1.0 - wy) * ((1.0 - wx) * v00 + wx * v01)
                  + wy * ((1.0 - wx) * v10 + wx * v11))
        values[~inside] = 0.0
        result = values.reshape(h_out, w_out)
    else:
        v00, v01 = source[y0, x0, :], source[y0, x1, :]
        v10, v11 = source[y1, x0, :], source[y1, x1, :]
        values = ((1.0 - wy)[:, None] * ((1.0 - wx)[:, None] * v00 + wx[:, None] * v01)
                  + wy[:, None] * ((1.0 - wx)[:, None] * v10 + wx[:, None] * v11))
        values[~inside, :] = 0.0
        result = values.reshape(h_out, w_out, source.shape[2])
    valid = inside.reshape(h_out, w_out)
    result = np.ascontiguousarray(result, dtype=precision)
    return (result, valid) if return_valid else result


def _area_weights(source_size: int, output_size: int, dtype: np.dtype) -> list[tuple[np.ndarray, np.ndarray]]:
    if source_size <= 0 or output_size <= 0:
        raise ValueError("resize dimensions must be positive")
    scale = np.asarray(source_size, dtype=dtype) / np.asarray(output_size, dtype=dtype)
    weights: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(output_size):
        start, end = np.asarray(index, dtype=dtype) * scale, np.asarray(index + 1, dtype=dtype) * scale
        first, last = int(np.floor(start)), int(np.ceil(end))
        positions = np.arange(first, last, dtype=np.int64)
        positions_float = positions.astype(dtype)
        overlap = np.minimum(positions_float + np.asarray(1.0, dtype=dtype), end) - np.maximum(positions_float, start)
        keep = overlap > 0.0
        weights.append((positions[keep], (overlap[keep] / np.asarray(scale, dtype=dtype)).astype(dtype)))
    return weights


def _perspective_from_quad(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Solve the 8-parameter projective mapping in float32."""

    matrix = []
    values = []
    for (x, y), (u, v) in zip(source, destination):
        matrix.extend(((x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y),
                       (0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)))
        values.extend((u, v))
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise TypeError("marker geometry coordinates must be float32")
    precision = np.dtype(np.float32)
    solution = _single_solve(np.asarray(matrix, dtype=precision), np.asarray(values, dtype=precision))
    return np.asarray((*solution, precision.type(1.0)), dtype=precision).reshape(3, 3)


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley-normalize finite float32 points to mean radius sqrt(2)."""
    center = points.mean(axis=0, dtype=np.float32)
    delta = points - center
    distances = np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32))
    mean_distance = distances.mean(dtype=np.float32)
    if not np.isfinite(mean_distance) or mean_distance <= np.finfo(np.float32).eps:
        raise ValueError("homography points have no spatial extent")
    scale = np.sqrt(np.float32(2.0)) / mean_distance
    transform = np.asarray(
        ((scale, 0.0, -scale * center[0]),
         (0.0, scale, -scale * center[1]),
         (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    normalized = delta * scale
    return np.ascontiguousarray(normalized, dtype=np.float32), transform


def fit_homography_lstsq(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    """Fit source-to-destination projective coordinates from all point pairs.

    The fit Hartley-normalizes both point sets, then solves the 8-parameter
    homography by float32 LAPACK least squares with ``h[2, 2] = 1``.  It does
    not reject or downweight outliers; callers can use the returned residuals
    to fail a registration against their fixed acceptance threshold.
    """
    source = np.asarray(source)
    destination = np.asarray(destination)
    if source.dtype != np.dtype(np.float32) or destination.dtype != np.dtype(np.float32):
        raise TypeError("homography least squares requires float32 coordinates")
    if source.ndim != 2 or destination.ndim != 2 or source.shape != destination.shape or source.shape[1:] != (2,):
        raise ValueError("homography coordinates must be matching [N,2] arrays")
    if source.shape[0] < 4:
        raise ValueError("homography least squares requires at least four points")
    if not np.isfinite(source).all() or not np.isfinite(destination).all():
        raise ValueError("homography coordinates must be finite")

    source_normalized, source_transform = _normalize_points(source)
    destination_normalized, destination_transform = _normalize_points(destination)
    x, y = source_normalized.T
    u, v = destination_normalized.T
    count = source.shape[0]
    design = np.zeros((2 * count, 8), dtype=np.float32)
    rhs = np.empty((2 * count, 1), dtype=np.float32)
    one = np.ones(count, dtype=np.float32)
    design[0::2] = np.stack((x, y, one, np.zeros_like(x), np.zeros_like(x),
                             np.zeros_like(x), -u * x, -u * y), axis=1)
    design[1::2] = np.stack((np.zeros_like(x), np.zeros_like(x), np.zeros_like(x),
                             x, y, one, -v * x, -v * y), axis=1)
    rhs[0::2, 0], rhs[1::2, 0] = u, v
    parameters = _single_least_squares(design, rhs)[:, 0]
    normalized_homography = np.asarray(
        ((parameters[0], parameters[1], parameters[2]),
         (parameters[3], parameters[4], parameters[5]),
         (parameters[6], parameters[7], 1.0)),
        dtype=np.float32,
    )

    scale = np.float32(destination_transform[0, 0])
    inverse_destination_transform = np.asarray(
        ((1.0 / scale, 0.0, -destination_transform[0, 2] / scale),
         (0.0, 1.0 / scale, -destination_transform[1, 2] / scale),
         (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    homography = inverse_destination_transform @ normalized_homography @ source_transform
    if not np.isfinite(homography).all() or abs(float(homography[2, 2])) <= np.finfo(np.float32).eps:
        raise ValueError("fitted homography has invalid projective scale")
    homography /= homography[2, 2]
    return np.ascontiguousarray(homography, dtype=np.float32)


def area_resize(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Exact separable pixel-area resize with float32 weights and accumulation."""

    source = _validate_hwc(image)
    out_h, out_w = (int(output_shape[0]), int(output_shape[1]))
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_shape must be positive")
    h_src, w_src = source.shape[:2]
    if out_h > h_src or out_w > w_src:
        raise ValueError("area_resize only supports downsampling")
    precision = source.dtype
    y_weights, x_weights = _area_weights(h_src, out_h, precision), _area_weights(w_src, out_w, precision)
    channels = 1 if source.ndim == 2 else source.shape[2]
    intermediate = np.empty((out_h, w_src, channels), dtype=precision)
    source_hwc = source[..., None] if source.ndim == 2 else source
    for oy, (indices, weights) in enumerate(y_weights):
        intermediate[oy] = np.tensordot(weights, source_hwc[indices], axes=(0, 0))
    result = np.empty((out_h, out_w, channels), dtype=precision)
    for ox, (indices, weights) in enumerate(x_weights):
        result[:, ox] = np.tensordot(weights, intermediate[:, indices], axes=(0, 1))
    if source.ndim == 2:
        result = result[..., 0]
    return np.ascontiguousarray(result, dtype=precision)


def rectify_stages(rgb_float: np.ndarray, homography: np.ndarray,
                       *, warp_size: int = 512, crop_margin: int = 46,
                       output_size: int = 256) -> dict[str, Any]:
    """Return warp, crop, final image, and support for the fixed geometry."""

    if warp_size != 512 or crop_margin != 46 or output_size != 256:
        raise ValueError("this experiment fixes 512 -> crop46 -> 420 -> 256")
    warped, valid = warp_perspective(rgb_float, homography, (warp_size, warp_size), return_valid=True)
    cropped = warped[crop_margin:warp_size - crop_margin, crop_margin:warp_size - crop_margin]
    valid_crop = valid[crop_margin:warp_size - crop_margin, crop_margin:warp_size - crop_margin]
    if not valid_crop.all():
        raise ValueError("marker crop contains source coordinates outside the DNG")
    output = area_resize(cropped, (output_size, output_size))
    geometry = {
        "warp_size": warp_size,
        "crop_margin": crop_margin,
        "crop_shape": [warp_size - 2 * crop_margin] * 2,
        "output_size": output_size,
        "interpolation": f"{rgb_float.dtype} inverse bilinear warp; exact pixel-area resize",
        "valid_fraction": float(valid_crop.mean()),
    }
    return {"warp512": warped, "valid512": valid,
            "crop420": cropped, "valid420": valid_crop,
            "output256": output, "geometry": geometry}


def marker_rectify(rgb_float: np.ndarray, homography: np.ndarray,
                       *, warp_size: int = 512, crop_margin: int = 46,
                       output_size: int = 256) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Warp to 512, crop 46 pixels, then area-resize to 256."""

    stages = rectify_stages(rgb_float, homography, warp_size=warp_size,
                                crop_margin=crop_margin, output_size=output_size)
    return stages["output256"], stages["valid420"], stages["geometry"]


def detect_marker_geometry(rgb_float: np.ndarray, *, min_tags: int = 4,
                           inset: float = 0.035) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect AprilTags on a display-only copy and return a promoted homography.

    Detection uses uint8 only for coordinates.  The returned quad and
    homography are computed as float32 and are subsequently used by the
    float32 resampler. No display pixels are used as measurements.
    """

    source = _validate_hwc(rgb_float)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("marker detection expects HWC RGB")
    h, w = source.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    small_shape = (max(1, round(w * scale)), max(1, round(h * scale)))
    if scale < 1.0:
        # This is a detector-only display copy; no computation uses it.
        small = cv2.resize(source, small_shape, interpolation=cv2.INTER_AREA)
    else:
        small = source
    low, high = np.percentile(small, np.asarray([1, 99], dtype=np.float32))
    display = np.uint8(np.clip((small - low) / max(float(high - low), 1.0), 0, 1) * 255.0 + 0.5)
    gray = cv2.cvtColor(display, cv2.COLOR_RGB2GRAY)
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        cv2.aruco.DetectorParameters(),
    )
    corners, ids, _ = detector.detectMarkers(gray)
    count = 0 if ids is None else int(len(ids))
    if corners is None or count < min_tags:
        raise ValueError(f"AprilTag detection found {count} tags, need {min_tags}")
    points = np.concatenate([np.asarray(c, dtype=np.float32).reshape(-1, 2) for c in corners], axis=0)
    # Use the source inner-hole construction: choose the innermost detected
    # corner on each board-frame sector.  A bounding box around the tag ring
    # would describe the outer board and crop the wrong material region.
    precision = source.dtype
    tags = [np.asarray(c, dtype=precision).reshape(4, 2) for c in corners]
    all_corners = np.concatenate(tags, axis=0)
    centre = all_corners.mean(axis=0, dtype=precision)
    edges = np.asarray([tag[1] - tag[0] for tag in tags], dtype=precision)
    theta = np.median(np.arctan2(edges[:, 1], edges[:, 0]))
    cosine, sine = np.cos(theta), np.sin(theta)
    rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=precision)
    board = (all_corners - centre) @ rotation.T
    horizontal = np.abs(board[:, 0]) > np.abs(board[:, 1])
    sectors = (horizontal & (board[:, 0] < 0), horizontal & (board[:, 0] > 0),
               ~horizontal & (board[:, 1] < 0), ~horizontal & (board[:, 1] > 0))
    if not all(mask.any() for mask in sectors):
        raise ValueError("AprilTag detections do not bound all four board sectors")
    x_left = board[sectors[0], 0].max()
    x_right = board[sectors[1], 0].min()
    y_top = board[sectors[2], 1].max()
    y_bottom = board[sectors[3], 1].min()
    quad_board = np.asarray([[x_left, y_top], [x_right, y_top],
                             [x_right, y_bottom], [x_left, y_bottom]], dtype=precision)
    quad_board = quad_board.mean(axis=0) + (quad_board - quad_board.mean(axis=0)) * (1.0 - float(inset))
    quad = quad_board @ rotation + centre
    quad[:, 0] /= scale
    quad[:, 1] /= scale
    destination = np.asarray([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=precision)
    homography = _perspective_from_quad(quad, destination)
    return homography, {
        "tags": count,
        "marker_ids": np.asarray(ids).reshape(-1).astype(int).tolist(),
        "marker_corners_raw": {str(int(tag_id)): (tag / scale).tolist() for tag_id, tag in zip(np.asarray(ids).reshape(-1), tags)},
        "board_axis_angle_radians": float(theta),
        "quad": quad.tolist(),
        "homography": homography.tolist(),
        "detector_scale": float(scale),
        "detector_display_percentiles": [float(low), float(high)],
        "detector_input": f"{precision} AHD RGB -> display-only linear percentile uint8",
        "coordinate_precision": f"detector uint8 coordinates promoted to {precision}",
    }


__all__ = ["warp_perspective", "area_resize", "rectify_stages",
           "marker_rectify", "detect_marker_geometry", "fit_homography_lstsq"]
