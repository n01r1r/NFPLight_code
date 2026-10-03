"""FP32 Adaptive Homogeneity-Directed (AHD) Bayer demosaicing.

The interpolation and homogeneity stages follow Dave Coffin's ``dcraw``
``ahd_interpolate`` implementation (dcraw 9.28-8, as carried by LibRaw):

* green is estimated independently in horizontal and vertical directions;
* red/blue values are completed from colour differences; and
* CIELab local homogeneity selects or averages the two candidates.

Reference sources (accessed 2026-09-30):
https://sources.debian.org/src/dcraw/9.28-8/dcraw.c (``ahd_interpolate``,
lines 15271-15373; ``cielab``, lines 15023-15050; and
``border_interpolate(5)``, lines 14777-14795) and
https://raw.githubusercontent.com/LibRaw/LibRaw/0.21.4/src/demosaic/ahd_demosaic.cpp
(``cielab`` lines 23-68, green candidates 69-94, colour completion and Lab
95-154, shared homogeneity thresholds 166-224, combination 226-279, and
orchestration including ``border_interpolate(5)`` 280-330) and
https://raw.githubusercontent.com/LibRaw/LibRaw/0.21.4/src/tables/colorconst.cpp
(``xyz_rgb`` and ``d65_white`` lines 15-20) and
https://github.com/LibRaw/LibRaw/blob/master/doc/API-datastruct.html
(``user_qual == 3`` is AHD).  LibRaw's production path stores ``ushort``
pixels and uses integer shifts and clipping.  This module is a source-based
FP32 adaptation, not a bit-exact LibRaw port: interpolation arithmetic is
performed in float32 precision, source R/B candidate
values retain LibRaw's [0, 65535] clip while integer storage and rounding are
removed, and an optional camera matrix is applied before the fixed
linear-sRGB to D65 XYZ matrix below for the internal Lab criterion.  The two
candidates are evaluated in vectorised NumPy
tiles (512-pixel cores with an 8-pixel halo), avoiding a Python loop per
pixel.  Outer-image boundaries use edge-extended Bayer samples; this supplies
the neighbourhood required by the source's ``border_interpolate(5)`` but is
not numerically identical to that source border pass.  This boundary choice is
an explicit adaptation; preserving the input sample at every CFA location is
still enforced.

The returned RGB values are sensor-count-like values in the selected floating
precision.  This function
does not apply black/white correction, white balance, gamma, tone mapping, or
display conversion.
"""

from __future__ import annotations

from typing import Final

import numpy as np


_TILE: Final[int] = 512
_HALO: Final[int] = 8
_D65_XYZ_FROM_LINEAR_SRGB: Final[np.ndarray] = np.array(
    (
        (0.4124564, 0.3575761, 0.1804375),
        (0.2126729, 0.7151522, 0.0721750),
        (0.0193339, 0.1191920, 0.9503041),
    ),
    dtype=np.float32,
)
_D65_WHITE: Final[np.ndarray] = np.array(
    (0.95047, 1.0, 1.08883),
    dtype=np.float32,
)


def _validate_inputs(
    bayer: np.ndarray,
    cfa: np.ndarray,
    dtype: np.dtype | type,
    camera_to_linear_srgb: np.ndarray | None,
) -> tuple[int, int, np.ndarray, np.dtype, np.ndarray]:
    """Validate the public contract and return dimensions, CFA and precision."""

    if not isinstance(bayer, np.ndarray):
        raise TypeError("bayer must be a numpy.ndarray with dtype float32")
    if bayer.ndim != 2:
        raise ValueError(f"bayer must be a 2-D HxW array, got ndim={bayer.ndim}")
    precision = np.dtype(dtype)
    if precision != np.dtype(np.float32):
        raise TypeError(f"capture AHD requires dtype float32, got {precision}")
    if bayer.dtype != precision:
        raise TypeError(f"bayer must have dtype {precision}, got {bayer.dtype}")
    if not np.all(np.isfinite(bayer)):
        raise ValueError("bayer must contain only finite values")
    height, width = bayer.shape
    if height < 2 or width < 2:
        raise ValueError("bayer must be at least 2x2")

    cfa_array = np.asarray(cfa)
    if cfa_array.shape != (2, 2):
        raise ValueError(f"cfa must have shape (2, 2), got {cfa_array.shape}")
    if not np.issubdtype(cfa_array.dtype, np.integer):
        raise TypeError(f"cfa must contain integer RGB channel indices, got {cfa_array.dtype}")
    cfa_int = np.array(cfa_array, dtype=np.int64, copy=True)
    if not np.array_equal(np.sort(cfa_int.reshape(-1)), np.array((0, 1, 1, 2), dtype=np.int64)):
        raise ValueError("cfa must contain exactly one 0 (R), two 1 (G), and one 2 (B)")
    if camera_to_linear_srgb is None:
        matrix = np.eye(3, dtype=precision)
    else:
        matrix = np.asarray(camera_to_linear_srgb)
        if matrix.shape != (3, 3):
            raise ValueError(
                f"camera_to_linear_srgb must have shape (3, 3), got {matrix.shape}"
            )
        if matrix.dtype != precision:
            raise TypeError(
                f"camera_to_linear_srgb must have dtype {precision}, got {matrix.dtype}"
            )
        matrix = np.array(matrix, dtype=precision, copy=True)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("camera_to_linear_srgb must contain only finite values")
    return height, width, cfa_int, precision, matrix


def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Return ``array[y + dy, x + dx]`` without wraparound.

    The caller only consumes positions at least eight pixels from this tile's
    edge.  Zero at the unused edge keeps the helper allocation-free apart from
    its result and avoids ``np.roll``'s incorrect wraparound semantics.
    """

    result = np.zeros_like(array)
    height, width = array.shape[:2]
    src_y0 = max(dy, 0)
    src_y1 = min(height, height + dy)
    dst_y0 = max(-dy, 0)
    dst_y1 = min(height, height - dy)
    src_x0 = max(dx, 0)
    src_x1 = min(width, width + dx)
    dst_x0 = max(-dx, 0)
    dst_x1 = min(width, width - dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return result


def _shift_batch(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift an array whose first axis is a candidate batch."""

    result = np.zeros_like(array)
    height, width = array.shape[1:3]
    src_y0 = max(dy, 0)
    src_y1 = min(height, height + dy)
    dst_y0 = max(-dy, 0)
    dst_y1 = min(height, height - dy)
    src_x0 = max(dx, 0)
    src_x1 = min(width, width + dx)
    dst_x0 = max(-dx, 0)
    dst_x1 = min(width, width - dx)
    if src_y0 < src_y1 and src_x0 < src_x1:
        result[:, dst_y0:dst_y1, dst_x0:dst_x1, ...] = array[
            :, src_y0:src_y1, src_x0:src_x1, ...
        ]
    return result


def _take_channel(array: np.ndarray, channel: np.ndarray) -> np.ndarray:
    """Gather one channel at each pixel from an HxWx3 array."""

    return np.take_along_axis(array, channel[..., None], axis=-1)[..., 0]


def _source_rgb_with_border(
    bayer: np.ndarray,
    cfa: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build measured RGB planes and apply dcraw's border_interpolate(5)."""

    height, width = bayer.shape
    rows = np.arange(height, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    colour = cfa[np.mod(rows, 2), np.mod(cols, 2)]
    channels = np.arange(3, dtype=np.int64)
    measured = colour[..., None] == channels
    source = bayer[..., None] * measured

    border = (
        (rows < 5)
        | (rows >= height - 5)
        | (cols < 5)
        | (cols >= width - 5)
    )
    for channel in channels:
        plane = source[..., channel]
        present = (colour == channel).astype(bayer.dtype, copy=False)
        neighbour_sum = np.zeros_like(plane)
        neighbour_count = np.zeros_like(plane)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                neighbour_sum += _shift(plane, dy, dx)
                neighbour_count += _shift(present, dy, dx)
        average = np.divide(
            neighbour_sum,
            neighbour_count,
            out=np.zeros_like(plane),
            where=neighbour_count != 0,
        )
        missing = border & (colour != channel) & (neighbour_count != 0)
        source[..., channel] = np.where(missing, average, plane)
    return source, colour


def _cielab(rgb: np.ndarray, camera_to_linear_srgb: np.ndarray) -> np.ndarray:
    """Convert candidate RGB counts to the scaled Lab coordinates used by AHD."""

    # dcraw's cielab() adds 0.5 before its 16-bit lookup and clamps the XYZ
    # lookup index to [0, 65535].  Preserve the criterion's scale while doing
    # the arithmetic directly in the selected precision.  The clamp is
    # internal to Lab only; the returned image remains signed and unclipped.
    precision = rgb.dtype
    matrix = np.asarray(camera_to_linear_srgb, dtype=precision)
    srgb = np.matmul(rgb, matrix.T)
    xyz_matrix = np.asarray(_D65_XYZ_FROM_LINEAR_SRGB, dtype=precision)
    d65_white = np.asarray(_D65_WHITE, dtype=precision)
    xyz = np.matmul(srgb, xyz_matrix.T)
    xyz = xyz / (np.asarray(65535.0, dtype=precision) * d65_white)
    # The source indexes a 16-bit Lab lookup with CLIP((int)xyz).  Here the
    # lookup/truncation is continuous, while its [0, 65535] range is retained.
    xyz = np.clip(xyz, 0.0, 1.0)
    pivot = 0.008856
    f_xyz = np.where(
        xyz > pivot,
        np.cbrt(xyz),
        np.asarray(7.787, dtype=precision) * xyz
        + np.asarray(16.0 / 116.0, dtype=precision),
    )
    lab = np.empty_like(rgb)
    lab[..., 0] = np.asarray(64.0, dtype=precision) * (
        np.asarray(116.0, dtype=precision) * f_xyz[..., 1]
        - np.asarray(16.0, dtype=precision)
    )
    lab[..., 1] = np.asarray(64.0 * 500.0, dtype=precision) * (
        f_xyz[..., 0] - f_xyz[..., 1]
    )
    lab[..., 2] = np.asarray(64.0 * 200.0, dtype=precision) * (
        f_xyz[..., 1] - f_xyz[..., 2]
    )
    return lab


def _candidate_pair(
    bayer: np.ndarray,
    colour: np.ndarray,
    source_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """Build the horizontal and vertical AHD RGB candidates for one tile."""

    # A Bayer sample exists in exactly one channel.  Keep the other channels
    # at zero until the corresponding AHD interpolation formula fills them.
    channels = np.arange(3, dtype=np.int64)
    measured = colour[..., None] == channels
    base = (
        bayer[..., None] * measured
        if source_rgb is None
        else np.asarray(source_rgb, dtype=bayer.dtype)
    )
    candidates = np.empty((2,) + base.shape, dtype=bayer.dtype)

    green_site = colour == 1
    non_green = ~green_site
    left = _shift(base, 0, -1)
    right = _shift(base, 0, 1)
    up = _shift(base, -1, 0)
    down = _shift(base, 1, 0)
    left2 = _shift(base, 0, -2)
    right2 = _shift(base, 0, 2)
    up2 = _shift(base, -2, 0)
    down2 = _shift(base, 2, 0)

    centre_sample = _take_channel(base, colour)
    horizontal_green = (
        2.0 * (left[..., 1] + centre_sample + right[..., 1])
        - _take_channel(left2, colour)
        - _take_channel(right2, colour)
    ) / 4.0
    vertical_green = (
        2.0 * (up[..., 1] + centre_sample + down[..., 1])
        - _take_channel(up2, colour)
        - _take_channel(down2, colour)
    ) / 4.0
    horizontal_green = np.clip(horizontal_green, np.minimum(left[..., 1], right[..., 1]), np.maximum(left[..., 1], right[..., 1]))
    vertical_green = np.clip(vertical_green, np.minimum(up[..., 1], down[..., 1]), np.maximum(up[..., 1], down[..., 1]))

    for direction, green in enumerate((horizontal_green, vertical_green)):
        rgb = base.copy()
        rgb[..., 1] = np.where(non_green, green, base[..., 1])

        # Complete red/blue at green sites with colour-difference interpolation.
        colour_below = _shift(colour, 1, 0)
        horizontal_colour = 2 - colour_below
        vertical_colour = colour_below
        horizontal_value = base[..., 1] + 0.5 * (
            _take_channel(left, horizontal_colour)
            + _take_channel(right, horizontal_colour)
            - _shift(rgb, 0, -1)[..., 1]
            - _shift(rgb, 0, 1)[..., 1]
        )
        vertical_value = base[..., 1] + 0.5 * (
            _take_channel(up, vertical_colour)
            + _take_channel(down, vertical_colour)
            - _shift(rgb, -1, 0)[..., 1]
            - _shift(rgb, 1, 0)[..., 1]
        )
        horizontal_value = np.clip(horizontal_value, 0.0, 65535.0)
        vertical_value = np.clip(vertical_value, 0.0, 65535.0)
        for channel in (0, 2):
            rgb[..., channel] = np.where(
                green_site & (horizontal_colour == channel),
                horizontal_value,
                rgb[..., channel],
            )
            rgb[..., channel] = np.where(
                green_site & (vertical_colour == channel),
                vertical_value,
                rgb[..., channel],
            )

        # Complete the missing colour at red/blue sites from the four diagonal
        # samples, as in dcraw's ((sum + 1) >> 2) formula without integer bias.
        diagonal_source = (
            _shift(base, -1, -1)
            + _shift(base, -1, 1)
            + _shift(base, 1, -1)
            + _shift(base, 1, 1)
        )
        diagonal_green = (
            _shift(rgb, -1, -1)[..., 1]
            + _shift(rgb, -1, 1)[..., 1]
            + _shift(rgb, 1, -1)[..., 1]
            + _shift(rgb, 1, 1)[..., 1]
        )
        missing_colour = 2 - colour
        missing_value = rgb[..., 1] + 0.25 * (
            _take_channel(diagonal_source, missing_colour) - diagonal_green
        )
        missing_value = np.clip(missing_value, 0.0, 65535.0)
        for channel in (0, 2):
            rgb[..., channel] = np.where(
                non_green & (missing_colour == channel),
                missing_value,
                rgb[..., channel],
            )

        # A measured value is copied exactly into every candidate.  This also
        # makes the sample-preservation invariant explicit before homogeneity
        # selection combines candidates at output pixels.
        rgb = np.where(measured, base, rgb)
        candidates[direction] = rgb

    return candidates


def _demosaic_tile(
    bayer: np.ndarray,
    cfa: np.ndarray,
    camera_to_linear_srgb: np.ndarray,
    source_rgb: np.ndarray,
) -> np.ndarray:
    """Demosaic one haloed Bayer tile, returning its full haloed candidate."""

    height, width = bayer.shape
    rows = np.arange(-_HALO, height - _HALO, dtype=np.int64)[:, None]
    cols = np.arange(-_HALO, width - _HALO, dtype=np.int64)[None, :]
    colour = cfa[np.mod(rows, 2), np.mod(cols, 2)]
    candidates = _candidate_pair(bayer, colour, source_rgb)
    lab = _cielab(candidates, camera_to_linear_srgb)

    # Directions are W, E, N, S, matching dcraw's dir[] ordering.
    ldiff = np.empty((2, 4, height, width), dtype=bayer.dtype)
    abdiff = np.empty_like(ldiff)
    directions = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for direction, (dy, dx) in enumerate(directions):
        neighbour = _shift_batch(lab, dy, dx)
        ldiff[:, direction] = np.abs(lab[..., 0] - neighbour[..., 0])
        abdiff[:, direction] = np.sum((lab[..., 1:] - neighbour[..., 1:]) ** 2, axis=-1)

    leps = np.minimum(
        np.maximum(ldiff[0, 0], ldiff[0, 1]),
        np.maximum(ldiff[1, 2], ldiff[1, 3]),
    )
    abeps = np.minimum(
        np.maximum(abdiff[0, 0], abdiff[0, 1]),
        np.maximum(abdiff[1, 2], abdiff[1, 3]),
    )
    homogeneous = (ldiff <= leps[None, None]) & (abdiff <= abeps[None, None])
    homo = homogeneous.sum(axis=1, dtype=np.int16)

    # Sum each candidate's homogeneity map over the 3x3 neighbourhood.
    local_homogeneity = np.zeros_like(homo, dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            local_homogeneity += _shift_batch(homo, dy, dx)

    chosen = np.where(
        (local_homogeneity[1] > local_homogeneity[0])[..., None],
        candidates[1],
        candidates[0],
    )
    tie = (local_homogeneity[0] == local_homogeneity[1])[..., None]
    chosen = np.where(tie, 0.5 * (candidates[0] + candidates[1]), chosen)
    return chosen


def demosaic_ahd(
    bayer: np.ndarray,
    cfa: np.ndarray,
    *,
    camera_to_linear_srgb: np.ndarray | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Demosaic a finite Bayer array with Adaptive Homogeneity Direction.

    Parameters
    ----------
    bayer:
        A two-dimensional, finite ``numpy.float32`` array
        containing one raw sample per pixel.  Values are kept as signed and
        unclipped in the selected precision; the default is float32.
    cfa:
        A 2x2 integer array containing channel indices ``0=R, 1=G, 2=B``;
        exactly one red, two green, and one blue entry are required.
    camera_to_linear_srgb:
        Optional finite 3x3 matrix mapping camera RGB to linear sRGB before
        the internal D65 Lab criterion, using the same row-vector convention
        as ``balanced @ matrix.T`` (the first three columns of rawpy's
        ``color_matrix``/LibRaw ``rgb_cam`` for this RGBG capture).  It must
        have the selected dtype.  An identity matrix is used when camera
        metadata is unavailable.
    dtype:
        Explicit computational precision, fixed to ``numpy.float32``. The
        Bayer input and optional matrix must have this exact dtype.

    Returns
    -------
    numpy.ndarray
        An ``H x W x 3`` contiguous array with the selected dtype.  All
        original CFA samples are copied exactly at their measured output
        channel.
    """

    height, width, cfa_int, precision, matrix = _validate_inputs(
        bayer, cfa, dtype, camera_to_linear_srgb
    )
    source_rgb, colour = _source_rgb_with_border(bayer, cfa_int)
    padded = np.pad(bayer, ((_HALO, _HALO), (_HALO, _HALO)), mode="edge")
    output = np.empty((height, width, 3), dtype=precision)

    for row0 in range(0, height, _TILE):
        row1 = min(row0 + _TILE, height)
        for col0 in range(0, width, _TILE):
            col1 = min(col0 + _TILE, width)
            tile = padded[row0 : row1 + 2 * _HALO, col0 : col1 + 2 * _HALO]
            tile_rows = np.clip(
                np.arange(row0 - _HALO, row1 + _HALO, dtype=np.int64),
                0,
                height - 1,
            )
            tile_cols = np.clip(
                np.arange(col0 - _HALO, col1 + _HALO, dtype=np.int64),
                0,
                width - 1,
            )
            source_tile = source_rgb[np.ix_(tile_rows, tile_cols)]
            demosaiced = _demosaic_tile(tile, cfa_int, matrix, source_tile)
            output[row0:row1, col0:col1] = demosaiced[
                _HALO : _HALO + row1 - row0,
                _HALO : _HALO + col1 - col0,
            ]

    # AHD candidate construction already carries measurements through, but
    # writing them once more from the source makes this contract independent of
    # any future candidate optimisation and prevents accidental tie averaging
    # from changing a CFA sample.
    rows = np.arange(height, dtype=np.int64)[:, None]
    cols = np.arange(width, dtype=np.int64)[None, :]
    border = (
        (rows < 5)
        | (rows >= height - 5)
        | (cols < 5)
        | (cols >= width - 5)
    )
    output[border] = source_rgb[border]
    for channel in range(3):
        output[..., channel] = np.where(colour == channel, bayer, output[..., channel])

    if not np.all(np.isfinite(output)):
        raise FloatingPointError("AHD produced non-finite output")
    return np.ascontiguousarray(output, dtype=precision)


__all__ = ["demosaic_ahd"]
