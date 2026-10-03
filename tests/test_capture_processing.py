"""Small deterministic contracts for the FP32 capture primitives."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from capture_processing import area_resize, apply_compensation, marker_rectify, warp_perspective
from capture_processing.demosaic import demosaic_ahd
from run_fabric_capture import (
    PREPARED_FRAME_KEYS,
    _core_hashes,
    _ensure_review_assets,
    _load_prepared_side,
    _load_root_review,
    _select_prepared_indices,
    _validate_preparation,
    select_source_frames,
)


class CaptureProcessingTests(unittest.TestCase):
    def test_compensation_retains_signed_fp32_values(self):
        from capture_processing.photometry import CONDITIONS

        self.assertEqual(set(CONDITIONS), {"both"})
        counts = np.asarray([0, 528, 4095, 4096], dtype=np.uint16)
        result = apply_compensation(counts, 528, 4095, "both")
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result[1:3], [0.0, 1.0])
        self.assertLess(result[0], 0.0)
        self.assertGreater(result[-1], 1.0)

    def test_retired_photometry_conditions_are_rejected(self):
        counts = np.ones((2, 2, 3), dtype=np.float32)
        for retired in ("white_only", "neither"):
            with self.subTest(condition=retired), self.assertRaisesRegex(ValueError, "only 'both' is active"):
                apply_compensation(counts, 0, 100, retired)

    def test_area_resize_constant_and_dtype(self):
        source = np.full((420, 420, 3), 3.25, dtype=np.float32)
        result = area_resize(source, (256, 256))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (256, 256, 3))
        np.testing.assert_allclose(result, 3.25, rtol=0.0, atol=1e-6)

    def test_identity_warp_preserves_values(self):
        source = np.arange(12 * 9 * 3, dtype=np.float32).reshape(12, 9, 3)
        result, valid = warp_perspective(source, np.eye(3, dtype=np.float32), (12, 9), return_valid=True)
        np.testing.assert_array_equal(result, source)
        self.assertTrue(valid.all())

    def test_marker_rectify_fixed_shape_and_finite(self):
        source = np.linspace(0.0, 1.0, 600 * 600 * 3, dtype=np.float32).reshape(600, 600, 3)
        result, valid, metadata = marker_rectify(source, np.eye(3, dtype=np.float32))
        self.assertEqual(result.shape, (256, 256, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(valid.all())
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(metadata["crop_shape"], [420, 420])

    def test_ahd_preserves_native_cfa_samples(self):
        cfa = np.asarray([[0, 1], [1, 2]], dtype=np.int64)
        bayer = np.arange(16 * 18, dtype=np.float32).reshape(16, 18)
        rgb = demosaic_ahd(bayer, cfa)
        self.assertEqual(rgb.dtype, np.float32)
        rows = np.arange(bayer.shape[0])[:, None]
        cols = np.arange(bayer.shape[1])[None, :]
        for channel in range(3):
            mask = cfa[rows % 2, cols % 2] == channel
            np.testing.assert_array_equal(rgb[..., channel][mask], bayer[mask])

    def test_explicit_fp32_spatial_photometry_and_model_coefficients_stay_fp32(self):
        source = np.full((600, 600, 3), 0.5, dtype=np.float32)
        warped = warp_perspective(source, np.eye(3, dtype=np.float32), (512, 512))
        resized = area_resize(source[:420, :420], (256, 256))
        self.assertEqual(warped.dtype, np.float32)
        self.assertEqual(resized.dtype, np.float32)
        compensated = apply_compensation(source, 528, 4095, "both")
        self.assertEqual(compensated.dtype, np.float32)

        from model.nfplight_model import NFPLightModel
        model = NFPLightModel.__new__(NFPLightModel)
        model.device = torch.device("cpu")
        model.compute_dtype = torch.float32
        model.capture_precision = True
        model.near_distance, model.far_distance = 2.414, 10.0
        coefficient, time_map = model.coefficientGeneration()
        self.assertEqual(coefficient.dtype, torch.float32)
        self.assertEqual(time_map.dtype, torch.float32)
        torch.testing.assert_close(
            coefficient,
            model.coefficient_raw / model.coefficient_normalization_max,
            rtol=0.0,
            atol=0.0,
        )

    def test_capture_primitives_reject_float64(self):
        source64 = np.ones((8, 8, 3), dtype=np.float64)
        with self.assertRaises(TypeError):
            warp_perspective(source64, np.eye(3, dtype=np.float64), (8, 8))
        with self.assertRaises(TypeError):
            area_resize(source64, (4, 4))
        with self.assertRaises(TypeError):
            apply_compensation(source64, 528, 4095, "both")
        with self.assertRaises(TypeError):
            demosaic_ahd(np.ones((8, 8), dtype=np.float64), np.asarray([[0, 1], [1, 2]]))

    def test_runner_is_float32_only_and_defaults_to_float32(self):
        import inspect
        from run_fabric_capture import run
        self.assertEqual(inspect.signature(run).parameters["precision"].default, "float32")
        with self.assertRaises(ValueError):
            run(precision="fp64")

    def test_float32_linear_algebra_uses_single_lapack_backends(self):
        from unittest.mock import patch
        import capture_processing.geometry as geometry
        import capture_processing.photometry as photometry

        calls = []
        real_get_lapack = geometry.get_lapack_funcs

        def spy(name, arrays):
            function = real_get_lapack(name, arrays)
            calls.append(function.__name__)
            return function

        identity = np.eye(3, dtype=np.float32)
        with patch.object(geometry, "get_lapack_funcs", side_effect=spy):
            np.testing.assert_allclose(geometry._single_inverse(identity), identity, atol=1e-6)
            np.testing.assert_allclose(geometry._single_solve(identity, np.ones(3, dtype=np.float32)), 1.0, atol=1e-6)
        self.assertIn("function sgetrf", calls)
        self.assertIn("function sgetri", calls)
        self.assertIn("function sgesv", calls)

        with patch.object(photometry, "get_lapack_funcs", side_effect=spy):
            self.assertEqual(photometry._single_matrix_rank(identity), 3)
        self.assertIn("function sgesdd", calls)

    def test_marker_homography_uses_selected_precision_and_records_ids(self):
        from unittest.mock import patch
        from capture_processing.geometry import detect_marker_geometry
        import numpy.ma  # Load NumPy lazy modules before spying on clip.
        boxes = [(5, 45), (108, 45), (45, 5), (45, 108)]
        corners = [np.asarray([[[x, y], [x+15, y], [x+15, y+15], [x, y+15]]], dtype=np.float32)
                   for x, y in boxes]
        ids = np.asarray([[0], [1], [2], [3]], dtype=np.int32)
        for dtype in (np.float32,):
            with patch("cv2.aruco.ArucoDetector") as detector, patch("capture_processing.geometry.np.clip", wraps=np.clip) as clipper:
                detector.return_value.detectMarkers.return_value = (corners, ids, [])
                matrix, record = detect_marker_geometry(np.ones((128, 128, 3), dtype=dtype))
            self.assertEqual(matrix.dtype, np.dtype(dtype))
            self.assertEqual(clipper.call_args_list[0].args[0].dtype, np.dtype(np.float32))
            self.assertEqual(record["marker_ids"], [0, 1, 2, 3])
            self.assertEqual(set(record["marker_corners_raw"]), {"0", "1", "2", "3"})

    def test_model_execution_rejects_double_precision(self):
        from types import SimpleNamespace
        from model.nfplight_model import NFPLightModel
        with self.assertRaisesRegex(ValueError, "only float32"):
            NFPLightModel(SimpleNamespace(precision="float64"))

    def test_clipping_stats_report_only_removed_values(self):
        from run_fabric_capture import _clipping_stats
        before = np.asarray([-2.0, 0.3, 0.8, 4.0], dtype=np.float32)
        result = _clipping_stats(before, np.clip(before, 0, 1))
        self.assertEqual(result["removed_min"], -2.0)
        self.assertEqual(result["removed_max"], 4.0)
        self.assertEqual(result["changed_fraction"], 0.5)
        clean = before[1:3]
        self.assertIsNone(_clipping_stats(clean, clean)["removed_min"])

    @staticmethod
    def _write_burst(root: Path, *, metadata: bool = True, count: int = 5) -> None:
        for index in range(count):
            (root / f"near_{index:02d}.dng").touch()
            (root / f"far_{index:02d}.dng").touch()
        if metadata:
            (root / "metadata.json").write_text(json.dumps({
                "near": {"filenames": [f"near_{i:02d}.dng" for i in range(count)]},
                "far": {"filenames": [f"far_{i:02d}.dng" for i in range(count)]},
            }), encoding="utf-8")

    def test_default_selects_only_first_metadata_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_burst(root, count=5)
            paths, record = select_source_frames(root)
            self.assertEqual(paths["near"].name, "near_00.dng")
            self.assertEqual(paths["far"].name, "far_00.dng")
            self.assertEqual(record["mode"], "burst_single")
            self.assertEqual(record["frame_index"], 0)
            self.assertEqual(record["frame_count"], 1)
            self.assertEqual(record["available_frame_count"], 5)
            self.assertFalse(record["averaged"])
            self.assertEqual(record["near_filenames"], ["near_00.dng"])
            self.assertEqual(record["far_filenames"], ["far_00.dng"])
            self.assertEqual(len(record["near_paths"]), 1)
            self.assertEqual(len(record["far_paths"]), 1)

    def test_explicit_frame_index_selects_one_metadata_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_burst(root, count=5)
            paths, record = select_source_frames(root, frame_index=3)
            self.assertEqual(paths["near"].name, "near_03.dng")
            self.assertEqual(paths["far"].name, "far_03.dng")
            self.assertEqual(record["frame_index"], 3)
            self.assertEqual(record["frame_count"], 1)
            self.assertFalse(record["averaged"])

    def test_burst_explicit_frame_index_checks_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_burst(root, count=2)
            with self.assertRaisesRegex(ValueError, "non-negative"):
                select_source_frames(root, frame_index=-1)
            with self.assertRaisesRegex(IndexError, "outside burst length"):
                select_source_frames(root, frame_index=2)

    def test_burst_rejects_unsafe_metadata_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({
                "near": {"filenames": ["../near_00.dng"]},
                "far": {"filenames": ["far_00.dng"]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe near frame filename"):
                select_source_frames(root, frame_index=0)

    def test_numbered_files_without_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_burst(root, metadata=False, count=2)
            with self.assertRaisesRegex(FileNotFoundError, "metadata.json is required"):
                select_source_frames(root)

    def test_none_and_legacy_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_burst(root, count=1)
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                select_source_frames(root, frame_index=None)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "near.dng").touch()
            (root / "far.dng").touch()
            with self.assertRaisesRegex(FileNotFoundError, "metadata.json is required"):
                select_source_frames(root)

    @staticmethod
    def _prepared_gate(*, fail_side: str | None = None, fail_index: int | None = None):
        sides = {}
        for side in ("near", "far"):
            frames = []
            for index in range(5):
                passed = not (side == fail_side and index == fail_index)
                frames.append({
                    "index": index,
                    "alignment": {"within_rms_limit": passed},
                    "processing": {"valid420_all": True},
                    "error": None,
                    "metadata_error": None,
                })
            sides[side] = {"metadata_compatible": True, "averaging_eligible": fail_side != side,
                           "frames": frames}
        return {"averaging": {"quantitative_eligible": fail_side is None, "reason": "synthetic failure" if fail_side else None},
                "sides": sides}

    def test_runner_averages_only_when_quantitative_and_visual_gates_pass(self):
        prepared = self._prepared_gate()
        review = {"texture_registration_passed": True}
        indices, averaged, quantitative, reasons = _select_prepared_indices(prepared, review)
        self.assertTrue(quantitative)
        self.assertTrue(averaged)
        self.assertEqual(indices, {"near": [0, 1, 2, 3, 4], "far": [0, 1, 2, 3, 4]})
        self.assertEqual(reasons, [])

    def test_runner_uses_first_pair_when_marker_gate_fails(self):
        prepared = self._prepared_gate(fail_side="far", fail_index=3)
        review = {"texture_registration_passed": False}
        indices, averaged, quantitative, reasons = _select_prepared_indices(prepared, review)
        self.assertFalse(quantitative)
        self.assertFalse(averaged)
        self.assertEqual(indices, {"near": [0], "far": [0]})
        self.assertTrue(reasons)
        self.assertNotIn("root texture registration review did not pass", reasons)

    def test_runner_uses_first_pair_when_texture_review_fails(self):
        prepared = self._prepared_gate()
        review = {"texture_registration_passed": False}
        indices, averaged, quantitative, reasons = _select_prepared_indices(prepared, review)
        self.assertTrue(quantitative)
        self.assertFalse(averaged)
        self.assertEqual(indices, {"near": [0], "far": [0]})
        self.assertIn("root texture registration review did not pass", reasons)

    def test_runner_rejects_first_pair_without_full_crop_support(self):
        prepared = self._prepared_gate()
        prepared["sides"]["near"]["frames"][0]["processing"]["valid420_all"] = False
        with self.assertRaisesRegex(ValueError, "frame00 is not a valid full-crop fallback"):
            _select_prepared_indices(prepared, {"texture_registration_passed": True})

    def test_runner_means_only_saved_aligned_fp32_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_dir = root / "source_frames"
            frame_dir.mkdir()
            frames = []
            for index in range(5):
                name = f"near_{index:02d}.npz"
                value = np.float32(index)
                arrays = {
                    "bayer_u16": np.full((2, 2), index, dtype=np.uint16),
                    "bayer_selected_dtype": np.full((2, 2), value, dtype=np.float32),
                    "demosaic_rgb_selected_dtype": np.full((2, 2, 3), value, dtype=np.float32),
                    "warp512": np.full((512, 512, 3), value, dtype=np.float32),
                    "crop420": np.full((420, 420, 3), value, dtype=np.float32),
                    "output256": np.full((256, 256, 3), value, dtype=np.float32),
                    "valid512": np.ones((512, 512), dtype=np.uint8),
                    "valid420": np.ones((420, 420), dtype=np.uint8),
                    "libraw_demosaic_rgb_selected_dtype": np.full((2, 2, 3), value, dtype=np.float32),
                    "libraw_output256": np.full((256, 256, 3), value + 10, dtype=np.float32),
                    "libraw_difference256": np.zeros((256, 256, 3), dtype=np.float32),
                }
                self.assertEqual(set(arrays), PREPARED_FRAME_KEYS)
                np.savez_compressed(frame_dir / name, **arrays)
                frames.append({"index": index, "archive": f"source_frames/{name}"})
            prepared = {"sides": {"near": {
                "frames": frames,
                "metadata": [{"black_level_per_channel": [528] * 4, "white_level": 4095}] * 5,
            }}}
            side = _load_prepared_side(root, prepared, "near", [0, 1, 2, 3, 4])
            self.assertEqual(side["counts"].dtype, np.float32)
            self.assertEqual(side["frame_count"], 5)
            np.testing.assert_array_equal(side["counts"], np.full((256, 256, 3), 2.0, dtype=np.float32))
            np.testing.assert_array_equal(side["ref_counts"], np.full((256, 256, 3), 12.0, dtype=np.float32))
            single = _load_prepared_side(root, prepared, "near", [0])
            self.assertEqual(single["frame_count"], 1)
            np.testing.assert_array_equal(single["counts"], np.zeros((256, 256, 3), dtype=np.float32))

    def test_runner_records_hashes_for_existing_review_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names = ["prepare_review.html", "prepare_near_contact_sheet.png", "prepare_far_contact_sheet.png",
                     "prepare_near_alignment_difference.png", "prepare_far_alignment_difference.png"]
            for name in names:
                (output / name).write_bytes(name.encode())
            gamma_name = "prepare_near_contact_sheet_gamma22.png"
            (output / gamma_name).write_bytes(gamma_name.encode())
            prepared = {
                "source_name": "capture",
                "averaging": {"quantitative_eligible": False},
                "review_assets": {"display_gamma": {"asset_sha256": {gamma_name: "old-hash"}}},
            }
            result = _ensure_review_assets(output, prepared)
            self.assertEqual(result["review_assets"]["sha256"], {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in [*names, gamma_name]
            })
            self.assertTrue((output / "preparation.json").is_file())

    def test_runner_rejects_pending_root_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "root_review.json"
            path.write_text(json.dumps({"status": "pending"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root_review.json must contain reviewed status"):
                _load_root_review(path.parent)

    def test_runner_rejects_changed_prepared_archive_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "capture"
            output = root / "prepared"
            source.mkdir()
            (source / "metadata.json").write_text("{}", encoding="utf-8")
            (output / "source_frames").mkdir(parents=True)
            before = {}
            sides = {}
            for side in ("near", "far"):
                frames = []
                metadata = []
                for index in range(5):
                    filename = f"{side}_{index:02d}.dng"
                    dng = source / filename
                    dng.write_bytes(f"{side}:{index}".encode())
                    archive = output / "source_frames" / f"{side}_{index:02d}.npz"
                    archive.write_bytes(f"archive:{side}:{index}".encode())
                    before[f"{side}/{filename}"] = hashlib.sha256(dng.read_bytes()).hexdigest()
                    frames.append({"index": index, "filename": filename,
                                   "sha256": before[f"{side}/{filename}"],
                                   "archive": f"source_frames/{side}_{index:02d}.npz",
                                   "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                   "processing": {"valid420_all": True}, "error": None})
                    metadata.append({})
                sides[side] = {"frames": frames, "metadata": metadata}
            core = _core_hashes()
            verification = {
                "input_file_hashes_before": before, "input_file_hashes_after": before,
                "source_hashes_unchanged": True, "source_metadata_unchanged": True,
                "implementation_unchanged": True, "archives_round_trip_checked": True,
            }
            prep = {
                "schema_version": 1, "source": str(source), "precision": "float32",
                "frame_count_per_side": 5, "capture_metadata_sha256": hashlib.sha256((source / "metadata.json").read_bytes()).hexdigest(),
                "implementation_sha256_before": core, "implementation_sha256_after": core,
                "verification": verification, "sides": sides, "status": "awaiting_visual_review",
            }
            (output / "preparation.json").write_text(json.dumps(prep), encoding="utf-8")
            _validate_preparation(output)
            changed = output / "source_frames" / "far_04.npz"
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "prepared frame archive changed"):
                _validate_preparation(output)


if __name__ == "__main__":
    unittest.main()
