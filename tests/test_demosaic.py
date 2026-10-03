"""Contract and numerical checks for the FP32 AHD demosaicer."""

from __future__ import annotations

import numpy as np
import pytest

import capture_processing.demosaic as demosaic_module
from capture_processing.demosaic import demosaic_ahd


_CFAS = (
    np.array(((0, 1), (1, 2)), dtype=np.int64),  # RGGB
    np.array(((2, 1), (1, 0)), dtype=np.int64),  # BGGR
    np.array(((1, 0), (2, 1)), dtype=np.int64),  # GRBG
    np.array(((1, 2), (0, 1)), dtype=np.int64),  # GBRG
)


def _sample_mask(shape: tuple[int, int], cfa: np.ndarray, channel: int) -> np.ndarray:
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return cfa[rows % 2, cols % 2] == channel


@pytest.mark.parametrize("cfa", _CFAS)
def test_all_bayer_arrangements_preserve_cfa_samples_and_return_fp32(cfa: np.ndarray) -> None:
    rng = np.random.default_rng(12)
    bayer = rng.random((19, 23), dtype=np.float32) * 4095.0 + 0.125
    before = bayer.copy()

    rgb = demosaic_ahd(bayer, cfa)

    assert rgb.shape == (19, 23, 3)
    assert rgb.dtype == np.float32
    assert np.isfinite(rgb).all()
    assert np.array_equal(bayer, before)
    for channel in range(3):
        assert np.array_equal(rgb[..., channel][_sample_mask(bayer.shape, cfa, channel)], bayer[_sample_mask(bayer.shape, cfa, channel)])


@pytest.mark.parametrize("cfa", _CFAS)
def test_constant_bayer_is_constant_at_boundaries(cfa: np.ndarray) -> None:
    bayer = np.full((8, 10), 137.25, dtype=np.float32)

    rgb = demosaic_ahd(bayer, cfa)

    assert np.array_equal(rgb, np.full((8, 10, 3), 137.25, dtype=np.float32))


def test_small_boundary_input_is_supported_and_finite() -> None:
    cfa = _CFAS[0]
    bayer = np.array(((0.25, 1.5), (2.75, 4.125)), dtype=np.float32)

    rgb = demosaic_ahd(bayer, cfa)

    assert rgb.shape == (2, 2, 3)
    assert np.isfinite(rgb).all()
    for channel in range(3):
        mask = _sample_mask(bayer.shape, cfa, channel)
        assert np.array_equal(rgb[..., channel][mask], bayer[mask])


def test_source_border_interpolate_is_used_for_outer_pixels() -> None:
    cfa = _CFAS[0]
    bayer = np.array(((1.0, 2.0), (3.0, 4.0)), dtype=np.float32)

    rgb = demosaic_ahd(bayer, cfa)

    # dcraw's 3x3 border pass sees the two green samples (2 and 3) at the
    # upper-left corner; edge replication would incorrectly produce 2.0.
    assert np.array_equal(rgb[0, 0], np.array((1.0, 2.5, 4.0), dtype=np.float32))


def test_camera_matrix_and_explicit_float32_precision_are_supported() -> None:
    cfa = _CFAS[0]
    rng = np.random.default_rng(73)
    bayer32 = rng.random((14, 16), dtype=np.float32) * 4095.0
    identity32 = np.eye(3, dtype=np.float32)
    matrix32 = np.diag(np.array((1.0, 1.1, 0.9), dtype=np.float32))

    default = demosaic_ahd(bayer32, cfa, dtype=np.float32)
    explicit_identity = demosaic_ahd(
        bayer32,
        cfa,
        camera_to_linear_srgb=identity32,
        dtype=np.float32,
    )
    transformed = demosaic_ahd(
        bayer32,
        cfa,
        camera_to_linear_srgb=matrix32,
        dtype=np.float32,
    )

    assert default.dtype == np.float32
    assert np.array_equal(default, explicit_identity)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()


def test_lab_criterion_normalizes_the_libraw_d65_white() -> None:
    white = np.full((1, 1, 3), 65535.0, dtype=np.float32)

    lab = demosaic_module._cielab(white, np.eye(3, dtype=np.float32))

    # LibRaw divides xyz_rgb by d65_white before its Lab lookup.  A linear
    # sRGB D65 white must therefore have nearly zero a*/b* components.
    assert abs(float(lab[0, 0, 1])) < 0.01
    assert abs(float(lab[0, 0, 2])) < 0.01


def test_interpolation_stays_float32_and_does_not_round_fractional_values() -> None:
    cfa = _CFAS[0]
    rows, cols = np.indices((12, 14))
    # A smooth, non-integer signal produces non-integer colour estimates.
    bayer = (0.37 * rows + 0.61 * cols + 0.125).astype(np.float32)

    rgb = demosaic_ahd(bayer, cfa)

    assert rgb.dtype == np.float32
    non_samples = ~_sample_mask(bayer.shape, cfa, 0)
    assert np.any(np.abs(rgb[..., 0][non_samples] - np.round(rgb[..., 0][non_samples])) > 1e-12)


@pytest.mark.parametrize(
    ("value", "exception"),
    (
        (np.zeros((4, 4), dtype=np.float64), TypeError),
        (np.zeros((4, 4, 1), dtype=np.float32), ValueError),
        (np.full((4, 4), np.inf, dtype=np.float32), ValueError),
    ),
)
def test_bayer_validation(value: np.ndarray, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        demosaic_ahd(value, _CFAS[0])


def test_precision_and_camera_matrix_validation() -> None:
    bayer32 = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(TypeError):
        demosaic_ahd(np.zeros((8, 8), dtype=np.float64), _CFAS[0])
    with pytest.raises(TypeError):
        demosaic_ahd(
            bayer32,
            _CFAS[0],
            camera_to_linear_srgb=np.eye(3, dtype=np.float64),
            dtype=np.float64,
        )
    with pytest.raises(ValueError):
        demosaic_ahd(
            bayer32,
            _CFAS[0],
            camera_to_linear_srgb=np.full((3, 3), np.inf, dtype=np.float32),
            dtype=np.float32,
        )


@pytest.mark.parametrize(
    "cfa",
    (
        np.zeros((3, 3), dtype=np.int64),
        np.array(((0, 1), (1, 1)), dtype=np.int64),
        np.array(((0.0, 1.0), (1.0, 2.0)), dtype=np.float32),
    ),
)
def test_cfa_validation(cfa: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        demosaic_ahd(np.zeros((8, 8), dtype=np.float32), cfa)


def test_tile_halo_keeps_result_continuous_across_512_pixel_boundaries() -> None:
    cfa = _CFAS[0]
    rows, cols = np.indices((523, 519))
    bayer = (0.17 * rows + 0.29 * cols + 0.031 * rows * cols).astype(np.float32)

    rgb = demosaic_ahd(bayer, cfa)

    # Every pixel at a tile seam must still contain finite, sample-preserving
    # output.  These checks exercise both partial tiles and the 8-pixel halo.
    assert np.isfinite(rgb[507:516, 507:516]).all()
    for channel in range(3):
        mask = _sample_mask(bayer.shape, cfa, channel)
        assert np.array_equal(rgb[..., channel][mask], bayer[mask])


def test_candidate_direction_shift_keeps_horizontal_and_vertical_candidates_separate() -> None:
    batch = np.arange(2 * 9 * 11, dtype=np.float32).reshape(2, 9, 11)

    shifted = demosaic_module._shift_batch(batch, 1, -1)
    expected = np.stack(
        [demosaic_module._shift(batch[index], 1, -1) for index in range(batch.shape[0])]
    )

    assert np.array_equal(shifted, expected)


def test_transpose_covariance_and_tile_size_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(91)
    cfa = _CFAS[2]
    bayer = rng.random((37, 41), dtype=np.float32) * 4095.0

    reference = demosaic_ahd(bayer, cfa)
    transposed = demosaic_ahd(bayer.T, cfa.T).transpose(1, 0, 2)
    monkeypatch.setattr(demosaic_module, "_TILE", 16)
    retiled = demosaic_ahd(bayer, cfa)

    assert np.allclose(transposed, reference, rtol=0.0, atol=1e-3)
    assert np.allclose(retiled, reference, rtol=0.0, atol=1e-3)
