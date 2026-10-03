"""Deterministic contracts for common-marker burst registration."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

import capture_processing.burst as burst
from capture_processing.burst import (
    BURST_METADATA_FIELDS,
    measure_pair_marker_residual,
    prepare_burst,
    register_marker_frames_to_first,
    validate_burst_metadata,
)
from capture_processing.geometry import fit_homography_lstsq


def _project(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate((points, np.ones((len(points), 1), dtype=np.float32)), axis=1)
    mapped = homogeneous @ homography.T
    return (mapped[:, :2] / mapped[:, 2:3]).astype(np.float32)


def _geometry(corners: dict[int, np.ndarray], homography: np.ndarray | None = None) -> dict:
    return {
        "marker_ids": sorted(corners),
        "marker_corners_raw": {str(tag): points.tolist() for tag, points in corners.items()},
        "homography": (np.eye(3, dtype=np.float32) if homography is None else homography).tolist(),
    }


def _marker_corners() -> dict[int, np.ndarray]:
    centers = ((120, 130), (800, 115), (850, 820), (135, 850), (470, 490))
    result = {}
    for tag, (x, y) in enumerate(centers):
        result[tag] = np.asarray(
            ((x - 12, y - 12), (x + 12, y - 12), (x + 12, y + 12), (x - 12, y + 12)),
            dtype=np.float32,
        )
    return result


def test_hartley_normalized_fp32_fit_recovers_subpixel_homography() -> None:
    source = np.asarray(
        ((0, 0), (150, 0), (150, 90), (0, 90), (35, 21), (115, 70), (81, 12), (14, 78)),
        dtype=np.float32,
    )
    expected = np.asarray(
        ((1.003, 0.014, 2.35), (-0.009, 0.997, -1.75), (0.00003, -0.00002, 1)),
        dtype=np.float32,
    )
    destination = _project(expected, source)

    actual = fit_homography_lstsq(source, destination)

    assert actual.dtype == np.float32
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(_project(actual, source), destination, rtol=0.0, atol=2e-4)


def test_first_frame_reference_composition_and_output_pixel_residuals() -> None:
    reference_corners = _marker_corners()
    h0 = np.asarray(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 1)), dtype=np.float32)
    h_relative = np.asarray(
        ((1.001, 0.008, 2.3), (-0.006, 0.999, -1.7), (0.00001, -0.000008, 1)),
        dtype=np.float32,
    )
    inverse_relative = np.linalg.inv(h_relative).astype(np.float32)
    current_corners = {
        tag: _project(inverse_relative, points) for tag, points in reference_corners.items()
    }

    alignments = register_marker_frames_to_first(
        [_geometry(reference_corners, h0), _geometry(current_corners, h0)]
    )

    assert len(alignments) == 2
    first, aligned = alignments
    np.testing.assert_array_equal(np.asarray(first["homography_relative"], dtype=np.float32), np.eye(3, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(first["homography_final"], dtype=np.float32), h0)
    assert first["residual_rms_px"] == 0.0
    assert aligned["shared_marker_ids"] == [0, 1, 2, 3, 4]
    assert aligned["shared_corner_count"] == 20
    expected_final = h0 @ np.asarray(aligned["homography_relative"], dtype=np.float32)
    np.testing.assert_allclose(aligned["homography_final"], expected_final, rtol=0.0, atol=1e-6)
    assert aligned["residual_rms_px"] < 1e-3
    assert aligned["residual_max_px"] < 1e-3
    assert aligned["within_rms_limit"] is True


def test_all_shared_corners_remain_in_fit_and_outlier_fails_rms_gate() -> None:
    reference_corners = _marker_corners()
    h0 = np.asarray(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 1)), dtype=np.float32)
    current_corners = {tag: points.copy() for tag, points in reference_corners.items()}
    current_corners[4] += np.asarray((100.0, -80.0), dtype=np.float32)

    alignment = register_marker_frames_to_first(
        [_geometry(reference_corners, h0), _geometry(current_corners, h0)]
    )[1]

    assert alignment["shared_marker_ids"] == [0, 1, 2, 3, 4]
    assert alignment["shared_corner_count"] == 20
    assert len(alignment["residuals_px"]) == 20
    assert alignment["residual_rms_px"] > 1.0
    assert alignment["within_rms_limit"] is False


def test_burst_metadata_requires_equal_sensor_and_exposure_signature() -> None:
    reference = {
        "raw_shape": [3024, 4032],
        "cfa_pattern": [[0, 1], [3, 2]],
        "visible_cfa_indices": [[0, 1], [1, 2]],
        "cfa_rgb_indices": [[0, 1], [1, 2]],
        "color_desc": "RGBG",
        "visible_origin": [0, 0],
        "black_level_per_channel": [528, 528, 528, 528],
        "white_level": 4095,
        "camera_whitebalance": [2.0, 1.0, 1.8, 0.0],
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "orientation_flip": 6,
        "exposure_settings": {
            "capture_settings": {"iso": 100, "shutter_seconds": 0.05},
            "dng_exif": {"known": True, "tags": {"exposure_time": {"numerator": 1, "denominator": 20}}},
        },
    }
    assert set(reference) == set(BURST_METADATA_FIELDS)
    matching = copy.deepcopy(reference)

    validate_burst_metadata(reference, matching)

    changed = copy.deepcopy(reference)
    changed["camera_whitebalance"][0] += 0.001
    with pytest.raises(ValueError, match="camera_whitebalance"):
        validate_burst_metadata(reference, changed)

    unknown = copy.deepcopy(reference)
    unknown["exposure_settings"]["dng_exif"]["known"] = False
    with pytest.raises(ValueError, match="unknown DNG exposure EXIF"):
        validate_burst_metadata(reference, unknown)


def test_independent_near_far_residual_preserves_detected_homographies() -> None:
    corners = _marker_corners()
    near_h = np.asarray(((0.5, 0, 0), (0, 0.5, 0), (0, 0, 1)), dtype=np.float32)
    far_h = near_h.copy()
    near = _geometry(corners, near_h)
    far = _geometry(corners, far_h)

    residual = measure_pair_marker_residual(near, far)

    assert residual["available"] is True
    assert residual["shared_marker_ids"] == [0, 1, 2, 3, 4]
    assert residual["shared_corner_count"] == 20
    assert residual["residual_rms_px"] == 0.0
    assert near["homography"] == near_h.tolist()
    assert far["homography"] == far_h.tolist()


def test_independent_near_far_residual_is_unavailable_without_four_common_tags() -> None:
    near_corners = _marker_corners()
    far_corners = {tag + 10: points.copy() for tag, points in near_corners.items()}
    h = np.eye(3, dtype=np.float32)

    residual = measure_pair_marker_residual(_geometry(near_corners, h), _geometry(far_corners, h))

    assert residual["available"] is False
    assert residual["shared_marker_ids"] == []
    assert residual["residual_rms_px"] is None
    assert residual["residual_max_px"] is None


def test_homography_fit_rejects_float64_coordinates() -> None:
    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)

    with pytest.raises(TypeError, match="float32"):
        fit_homography_lstsq(points, points.copy())


def _fake_capture(path, dtype=np.float32):
    raw = np.full((8, 8), 1000, dtype=np.uint16)
    rgb = np.full((8, 8, 3), 1000, dtype=np.float32)
    metadata = {
        "sha256": "unused",
        "raw_shape": [8, 8],
        "cfa_pattern": [[0, 1], [3, 2]],
        "visible_cfa_indices": [[0, 1], [1, 2]],
        "cfa_rgb_indices": [[0, 1], [1, 2]],
        "color_desc": "RGBG",
        "visible_origin": [0, 0],
        "black_level_per_channel": [528, 528, 528, 528],
        "white_level": 4095,
        "camera_whitebalance": [2.0, 1.0, 1.8, 0.0],
        "color_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "orientation_flip": 6,
    }
    return SimpleNamespace(
        path=str(path), sha256="unused", bayer_u16=raw, bayer_float=raw.astype(np.float32),
        rgb_float=rgb, metadata=metadata,
    )


def _fake_spatial_stages(image, homography):
    warp = np.ones((512, 512, 3), dtype=np.float32)
    valid = np.ones((512, 512), dtype=bool)
    crop = warp[46:466, 46:466]
    valid_crop = valid[46:466, 46:466]
    output = np.ones((256, 256, 3), dtype=np.float32)
    return {
        "warp512": warp, "valid512": valid, "crop420": crop,
        "valid420": valid_crop, "output256": output,
        "geometry": {"warp_size": 512, "crop_margin": 46, "crop_shape": [420, 420],
                     "output_size": 256, "interpolation": "test", "valid_fraction": 1.0},
    }


def _make_fake_burst(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    capture_metadata = {}
    for side in ("near", "far"):
        names = [f"{side}_{index:02d}.dng" for index in range(5)]
        for name in names:
            (source / name).write_bytes(name.encode("ascii"))
        capture_metadata[side] = {"frameCount": 5, "filenames": names,
                                  "settings": {"appliedISO": 100, "appliedShutterSeconds": 0.05}}
    (source / "metadata.json").write_text(json.dumps(capture_metadata), encoding="utf-8")
    return source


def _install_fake_burst_inputs(monkeypatch, *, changed_exif=None):
    corners = _marker_corners()
    geometry = _geometry(corners)
    monkeypatch.setattr(burst, "unpack_dng", lambda path, dtype=np.float32: _fake_capture(path, dtype))
    monkeypatch.setattr(burst, "detect_marker_geometry", lambda image: (np.eye(3, dtype=np.float32), copy.deepcopy(geometry)))
    monkeypatch.setattr(burst, "libraw_ahd_reference", lambda path: np.full((8, 8, 3), 1000, dtype=np.uint16))
    monkeypatch.setattr(burst, "_warp_stages", _fake_spatial_stages)

    def read_exif(path):
        name = str(path)
        numerator = 1
        if changed_exif is not None and changed_exif in name:
            numerator = 2
        return {"known": True, "tags": {"exposure_time": {"numerator": numerator, "denominator": 20}}, "missing": []}

    monkeypatch.setattr(burst, "read_dng_exposure_exif", read_exif)


def test_prepare_burst_writes_roundtrip_frames_but_does_not_average(tmp_path, monkeypatch) -> None:
    source = _make_fake_burst(tmp_path)
    output = tmp_path / "output"
    _install_fake_burst_inputs(monkeypatch)

    manifest = prepare_burst(source, output)

    assert manifest["status"] == "awaiting_visual_review"
    assert manifest["averaging"]["quantitative_eligible"] is True
    assert manifest["averaging"]["mean_generated"] is False
    assert manifest["averaging"]["selected_indices"] == {"near": [0, 1, 2, 3, 4], "far": [0, 1, 2, 3, 4]}
    assert not (output / "source_arrays.npz").exists()
    with np.load(output / "source_frames" / "near_02.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "bayer_u16", "bayer_selected_dtype", "demosaic_rgb_selected_dtype", "warp512",
            "crop420", "output256", "valid512", "valid420", "libraw_demosaic_rgb_selected_dtype",
            "libraw_output256", "libraw_difference256",
        }
        assert archive["output256"].dtype == np.float32
        assert archive["output256"].shape == (256, 256, 3)


def test_prepare_burst_metadata_mismatch_selects_only_pair_zero(tmp_path, monkeypatch) -> None:
    source = _make_fake_burst(tmp_path)
    output = tmp_path / "output"
    _install_fake_burst_inputs(monkeypatch, changed_exif="near_03.dng")

    manifest = prepare_burst(source, output)

    assert manifest["status"] == "prepared_single_frame_fallback"
    assert manifest["averaging"]["quantitative_eligible"] is False
    assert manifest["averaging"]["selected_indices"] == {"near": [0], "far": [0]}
    assert manifest["sides"]["near"]["metadata_compatible"] is False
    assert any(item["index"] == 3 and "exposure_settings" in item["reason"]
               for item in manifest["sides"]["near"]["metadata_mismatches"])


def test_prepare_burst_aligns_far_scale_to_near_zero_marker_plane(tmp_path, monkeypatch) -> None:
    source = _make_fake_burst(tmp_path)
    _install_fake_burst_inputs(monkeypatch)
    near_corners = _marker_corners()
    far_corners = {tag: points / np.float32(3) for tag, points in near_corners.items()}
    detected = iter([_geometry(near_corners)] * 5 + [_geometry(far_corners)] * 5)
    monkeypatch.setattr(burst, "detect_marker_geometry", lambda image: (None, next(detected)))

    manifest = prepare_burst(source, tmp_path / "output")

    assert manifest["averaging"]["quantitative_eligible"] is True
    for frame in manifest["sides"]["far"]["frames"]:
        alignment = frame["alignment"]
        assert alignment["reference_side"] == "near"
        assert alignment["reference_index"] == 0
        np.testing.assert_allclose(
            alignment["homography_final"], np.diag(np.float32([3, 3, 1])), atol=3e-4,
        )
    assert all(pair["residual_rms_px"] < 1e-3
               for pair in manifest["near_far_pair_residuals"].values())


def test_prepare_burst_rejects_frame00_without_full_crop_support(tmp_path, monkeypatch) -> None:
    source = _make_fake_burst(tmp_path)
    output = tmp_path / "output"
    _install_fake_burst_inputs(monkeypatch)

    def invalid_support(image, homography):
        stages = _fake_spatial_stages(image, homography)
        stages["valid420"] = np.zeros((420, 420), dtype=bool)
        stages["geometry"]["valid_fraction"] = 0.0
        return stages

    monkeypatch.setattr(burst, "_warp_stages", invalid_support)

    manifest = prepare_burst(source, output)

    assert manifest["status"] == "failed_first_frame"
    assert manifest["averaging"]["selected_indices"] == {"near": [], "far": []}
    assert manifest["sides"]["near"]["frames"][0]["processing"]["valid420_all"] is False
    assert manifest["sides"]["far"]["frames"][0]["processing"]["valid420_all"] is False
