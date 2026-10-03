"""Deterministic contracts for capture HTML, preview scales, and review sheets."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from capture_processing.raw import sha256_file

from capture_processing.report import (
    _feature_groups,
    _identity_copy_features_match,
    _make_html,
    _figure_gallery,
    _write_checkpoint_comparison,
    _display_rgb,
    _gamma22,
    _render_array,
    _display_contract,
    _stage_label,
    _array_for_key,
    _stats,
    _saved_common_shapes,
    _display_configs_from_manifest,
    regenerate_capture_gallery_and_html,
    regenerate_capture_html_only,
    regenerate_capture_index_html_only,
    regenerate_feature_gallery_only,
    regenerate_capture_report,
    regenerate_prepare_html_only,
    write_capture_index,
    write_capture_report,
    write_prepare_review,
)


class CaptureReportTests(unittest.TestCase):
    def test_author_difference_display_has_paper_sign_and_matching_stats(self):
        signed = np.array([[[-0.3, 0.1, 0.2]]], dtype=np.float32)
        archive = {"signed_diff": signed}
        np.testing.assert_array_equal(_array_for_key(archive, "near_minus_scaled_far"), -signed)
        np.testing.assert_array_equal(_array_for_key(archive, "near_minus_scaled_far_c1"), -signed[..., 1])
        manifest = {"model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
                    "conditions": {"both": {"arrays": {"signed_diff": _stats(signed)}}}}
        original = json.dumps(manifest)
        page = _make_html(manifest, ["signed_diff"], {"signed_diff": [-0.4, 0.4]},
                          {"signed_diff": {"range": [-0.4, 0.4], "rgb": False}}, {}, [])
        self.assertIn('value="near_minus_scaled_far"', page)
        self.assertNotIn('value="signed_diff"', page)
        self.assertIn('both/display/near_minus_scaled_far.png', page)
        self.assertIn('I_N − clip(K I_F,0,1)', page)
        self.assertEqual(json.dumps(manifest), original)
        np.testing.assert_array_equal(archive["signed_diff"], signed)

    def test_source_headers_distinguish_rgb_previews_from_scalar_maps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "source_frames"
            frames.mkdir()
            path = frames / "near_00.npz"
            np.savez(path, bayer_u16=np.zeros((2, 2), dtype=np.uint16),
                     demosaic_rgb_selected_dtype=np.zeros((2, 2, 3), dtype=np.float32),
                     valid512=np.ones((2, 2), dtype=np.float32))
            before = sha256_file(path)
            stats = _saved_common_shapes(root, {})
            configs = _display_configs_from_manifest({}, {}, stats)
            self.assertTrue(configs["frame_near_00_demosaic_rgb_selected_dtype"]["rgb"])
            self.assertTrue(configs["frame_near_00_bayer_u16"]["rgb"])
            self.assertFalse(configs["frame_near_00_valid512"]["rgb"])
            self.assertEqual(sha256_file(path), before)

    def test_paper_stage_labels_and_grayscale_endpoints(self):
        self.assertIn("K_θ", _stage_label("time_co_map"))
        self.assertIn("C_M", _stage_label("denominator"))
        self.assertIn("촬영 1", _stage_label("frame_near_00_output256"))
        self.assertIn("신경망 해상도", _stage_label("frame_near_00_output256"))
        self.assertEqual(_display_contract("features_legacy33_c30", [-0.8, 0.7]), (-1.0, 1.0, "gray"))
        self.assertEqual(_display_contract("frame_near_00_valid420", [1, 1]), (0.0, 1.0, "gray"))
        values = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
        before = values.copy()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signed.png"
            _render_array(values, "signed_diff", path, (-1, 1), "gray")
            shown = np.asarray(Image.open(path))
            np.testing.assert_array_equal(shown[0, :, 0], [0, 128, 255])
            np.testing.assert_array_equal(shown[..., 0], shown[..., 1])
            np.testing.assert_array_equal(shown[..., 1], shown[..., 2])
        np.testing.assert_array_equal(values, before)

    @staticmethod
    def _frame_archives(root: Path, *, gate_ok: bool = True):
        records = {
            "near": {"frames": [], "metadata_compatible": True, "averaging_eligible": True, "metadata": []},
            "far": {"frames": [], "metadata_compatible": True, "averaging_eligible": True, "metadata": []},
        }
        y, x = np.mgrid[:256, :256]
        for side_index, side in enumerate(("near", "far")):
            for index in range(5):
                metadata = {
                    "camera_whitebalance": [2.0, 1.0, 1.5, 0.0],
                    "color_matrix": [[1.0, 0.25, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                }
                records[side]["metadata"].append(metadata)
                rgb = np.zeros((24, 32, 3), dtype=np.float32)
                rgb[..., 0] = np.float32(100 + side_index * 20 + index)
                rgb[..., 1] = np.float32(200 + x[:24, :32] * 0.1)
                rgb[..., 2] = np.float32(300 + y[:24, :32] * 0.2)
                crop = np.stack((x / 255, y / 255, np.full_like(x, 0.5)), axis=-1).astype(np.float32)
                path = Path("source_frames") / f"{side}_{index:02d}.npz"
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(root / path, demosaic_rgb_selected_dtype=rgb, output256=crop)
                records[side]["frames"].append({
                    "side": side, "index": index, "filename": f"{side}_{index:02d}.dng",
                    "archive": str(path), "raw_shape": [24, 32],
                    "geometry": {
                        "marker_ids": [0, 1, 2, 3],
                        "marker_corners_raw": {str(marker): [[2, 2], [5, 2], [5, 5], [2, 5]] for marker in range(4)},
                        "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                        "raw_shape": [24, 32],
                    },
                    "alignment": {"homography_final": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                  "residual_rms_px": 0.0, "within_rms_limit": gate_ok or index == 0},
                    "metadata_compatible": True,
                    "processing": {"valid_fraction": 1.0, "valid420_all": True},
                })
        return records

    def test_prepare_review_keeps_native_crops_and_gates_mean_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._frame_archives(root, gate_ok=True)
            result = write_prepare_review(root, {"sides": records}, group_label="synthetic-group")
            self.assertTrue(result["marker_metadata_crop_gate_passed"])
            self.assertEqual(len(result["mean_candidates"]), 2)
            with Image.open(root / "prepare_near_contact_sheet.png") as near:
                self.assertEqual(near.size, (5 * 256, 2 * (256 + 28)))
            with Image.open(root / "prepare_near_frame00_vs_mean_display_only.png") as mean_candidate:
                self.assertEqual(mean_candidate.size, (512, 284))
            page = (root / "prepare_review.html").read_text(encoding="utf-8")
            self.assertIn("texture review pending", page)
            self.assertIn("camera→linear-sRGB matrix", page)
            self.assertIn("diagonal (3×3)", page)
            self.assertIn("gamma x^(1/2.2)", page)
            self.assertIn("background:#111;color:#ddd", page)
            self.assertIn("data-gamma22", page)
            self.assertIn("marker_metadata_crop_gate_passed", json.dumps(result))
            self.assertTrue((root / "prepare_near_contact_sheet_gamma22.png").is_file())
            with Image.open(root / "prepare_near_contact_sheet.png") as linear:
                linear_pixel = np.asarray(linear)[420, 100].astype(np.int16)
            with Image.open(root / "prepare_near_contact_sheet_gamma22.png") as gamma:
                gamma_pixel = np.asarray(gamma)[420, 100].astype(np.int16)
            self.assertGreater(int(np.abs(gamma_pixel - linear_pixel).max()), 0)

            # Simulate an older prepared page: refresh only its HTML and recorded HTML hash.
            stale_page = page.replace("diagonal (3×3)", "diagonal placeholder")
            (root / "prepare_review.html").write_text(stale_page, encoding="utf-8")
            preparation = {"sides": records, "review_assets": result}
            preparation["review_assets"]["sha256"] = {"prepare_review.html": "old-html-hash"}
            (root / "preparation.json").write_text(json.dumps(preparation), encoding="utf-8")
            png_hashes = {path.name: sha256_file(path) for path in root.glob("*.png")}
            refreshed_prepare = regenerate_prepare_html_only(root)
            self.assertEqual(refreshed_prepare, root / "prepare_review.html")
            self.assertIn("WB diagonal and camera matrix", refreshed_prepare.read_text(encoding="utf-8"))
            updated_preparation = json.loads((root / "preparation.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_preparation["review_assets"]["sha256"]["prepare_review.html"],
                             sha256_file(refreshed_prepare))
            self.assertEqual(png_hashes, {path.name: sha256_file(path) for path in root.glob("*.png")})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._frame_archives(root, gate_ok=False)
            result = write_prepare_review(root, {"sides": records})
            self.assertFalse(result["marker_metadata_crop_gate_passed"])
            self.assertEqual(result["mean_candidates"], [])

    def test_prepare_review_keeps_both_sides_when_frame_archive_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = self._frame_archives(root, gate_ok=True)
            records["near"]["frames"][1]["archive"] = "source_frames/missing.npz"
            result = write_prepare_review(root, {"sides": records})
            self.assertFalse(result["marker_metadata_crop_gate_passed"])
            self.assertEqual(result["mean_candidates"], [])
            self.assertEqual(len(result["failed_frames"]), 1)
            for side in ("near", "far"):
                with Image.open(root / f"prepare_{side}_contact_sheet.png") as sheet:
                    self.assertEqual(sheet.width, 5 * 256)
            page = (root / "prepare_review.html").read_text(encoding="utf-8")
            self.assertIn("frame archive missing", page)

    def test_combined_index_accepts_runner_group_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_capture_index(root, [
                {"label": "260818_174845_478", "source": "data/260818_174845_478",
                 "status": "prepared", "report": "260818_174845_478/prepare_review.html",
                 "average_used": False, "averaging_eligible": False,
                 "fallback_reason": "marker RMS gate failed", "clipping": {"white_only": {"obsolete": True},
                     "neither": {"obsolete": True}, "both": {
                     "near_input": {"lower_fraction": 0.001, "upper_fraction": 0.02},
                     "far_input": {"lower_fraction": 0.0, "upper_fraction": 0.03}}}},
            ])
            self.assertTrue(path.is_file())
            page = path.read_text(encoding="utf-8")
            self.assertIn("260818_174845_478/prepare_review.html", page)
            self.assertIn("near_00/far_00 fallback", page)
            self.assertIn("RMS and clipping details", page)
            self.assertIn("background:#111;color:#ddd", page)
            self.assertNotIn("No GT", page)
            self.assertNotIn("white_only", page)
            self.assertNotIn("neither", page)
            index_path = root / "index.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["groups"][0]["clipping"] = {"white_only": {"obsolete": True},
                                                 "neither": {"obsolete": True},
                                                 "both": {"near_input": {"upper_fraction": 0.02}}}
            index_path.write_text(json.dumps(index), encoding="utf-8")
            index_sha = sha256_file(root / "index.json")
            refreshed = regenerate_capture_index_html_only(root)
            self.assertEqual(refreshed, root / "index.html")
            self.assertEqual(sha256_file(root / "index.json"), index_sha)
            refreshed_page = refreshed.read_text(encoding="utf-8")
            self.assertIn("both", refreshed_page)
            self.assertNotIn("white_only", refreshed_page)
            self.assertNotIn("neither", refreshed_page)
            refresh_record = json.loads((root / "index_html_generation.json").read_text(encoding="utf-8"))
            self.assertFalse(refresh_record["inference_executed"])
            index = json.loads((root / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["groups"][0]["fallback_reason"], "marker RMS gate failed")

    def test_final_report_scales_relation_and_raw_arrays_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source_frames").mkdir()
            h, w = 256, 256
            yy, xx = np.mgrid[:h, :w]
            base_features = np.stack([np.full((h, w), (channel / 20) * 2 - 1, dtype=np.float32)
                                      for channel in range(21)], axis=-1)
            base_prediction = np.stack([np.full((h, w), (channel / 9) * 2 - 1, dtype=np.float32)
                                        for channel in range(10)], axis=-1)
            conditions = {}
            original = {}
            for offset, mode in enumerate(("both",)):
                relation = np.asarray((xx + offset) / 255, dtype=np.float32)
                relation_log = np.asarray((yy + offset) / 255, dtype=np.float32)
                arrays = {
                    "near_input_clipped": np.full((h, w, 3), 0.2 + offset * 0.1, dtype=np.float32),
                    "far_input_clipped": np.full((h, w, 3), 0.4 + offset * 0.1, dtype=np.float32),
                    "far_gained_clipped": np.full((h, w, 3), 0.6, dtype=np.float32),
                    "far_time_clipped": np.full((h, w, 3), 0.5, dtype=np.float32),
                    "relation_raw": relation + 0.5,
                    "relation_log_raw": relation_log - 0.5,
                    "relation": relation,
                    "relation_log": relation_log,
                    "relation_normalized": relation,
                    "relation_log_normalized": relation_log,
                    "signed_diff": np.stack((relation - 0.5, 0.5 - relation, relation - 0.4), axis=-1),
                    "abs_diff": np.stack((relation, relation * 0.7, relation * 0.3), axis=-1),
                    "features": base_features.copy(),
                    "prediction": base_prediction.copy(),
                }
                original[mode] = {key: value.copy() for key, value in arrays.items()}
                (root / mode).mkdir()
                np.savez_compressed(root / mode / "arrays.npz", **arrays)
                conditions[mode] = {"formula": "D / W", "arrays": {}}
            np.savez_compressed(root / "source_arrays.npz", near_valid=np.ones((h, w), dtype=np.uint8))
            bayer = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
            demosaic = np.stack((bayer, bayer + 1, bayer + 2), axis=-1).astype(np.float32)
            np.savez_compressed(root / "source_frames" / "near_00.npz", bayer_u16=bayer,
                                bayer_selected_dtype=bayer.astype(np.float32),
                                demosaic_rgb_selected_dtype=demosaic)
            manifest = {
                "source": "synthetic", "group": "synthetic-group", "precision": {"requested": "float32"},
                "conditions": conditions,
                "frame_average": {"enabled": False, "quantitative_gate_passed": False,
                                  "texture_review_status": "not_applicable", "fallback_reason": ["frame 1 RMS >1 px"],
                                  "near_count": 1, "far_count": 1},
                "selected_frames": {"averaged": False,
                                    "selected_filenames": {"near": ["near_00.dng"], "far": ["far_00.dng"]}},
                "root_visual_review": {"status": "reviewed", "note": "Fallback confirmed from numeric gate."},
                "model_geometry": {"near_distance": 2.414, "far_distance": 10.0, "far_gain": (10/2.414)**2},
                "model_contract": {"feature_version": "raw_calibrated_v2", "normal_head": "xyz",
                                   "output_channel_order": ["normal_x", "normal_y", "normal_z", "diffuse_r", "diffuse_g", "diffuse_b", "roughness", "specular_r", "specular_g", "specular_b"]},
                "color_transforms": {side: {
                    "normalized_rgb_wb": [2.0, 1.0, 1.5],
                    "camera_to_linear_srgb": [[1.0, 0.25, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                } for side in ("near", "far")},
                "pipeline": ["synthetic report fixture"], "environment": {"libraw": "0.22.0"},
                "ahd_source": "continuous adaptation source", "limitations": ["No GT."],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "verification.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            (root / "prepare_review.html").write_text("review", encoding="utf-8")
            (root / "independent_formula_verification.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            display_keys = ["near_input_clipped", "far_input_clipped", "far_gained_clipped", "far_time_clipped",
                            "relation_raw", "relation_log_raw", "relation", "relation_log", "signed_diff",
                            "abs_diff", "features_c0", "prediction_c9"]
            ranges = {key: [-1.0, 1.0] for key in display_keys}
            write_capture_report(root, ranges, conditions, display_keys)

            with np.load(root / "both" / "arrays.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["features"], original["both"]["features"])
                np.testing.assert_array_equal(archive["prediction"], original["both"]["prediction"])
            with Image.open(root / "both/display/relation.png") as relation_preview:
                self.assertEqual(relation_preview.size, (256, 256))
            with Image.open(root / "both/display/signed_diff.png") as signed_preview:
                self.assertEqual(signed_preview.size, (768, 256))
            saved_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["display_ranges"]["relation"], [0.0, 1.0])
            self.assertEqual(saved_manifest["display_ranges"]["relation_log"], [0.0, 1.0])
            self.assertEqual(saved_manifest["display_ranges"]["features_c0"], [-1.0, 1.0])
            self.assertTrue(saved_manifest["report"]["raw_prediction_arrays_unchanged"])
            self.assertTrue((root / "features_gallery.png").is_file())
            self.assertTrue((root / "prediction_semantic_gallery.png").is_file())
            self.assertTrue((root / "prediction_semantic_gallery_gamma22.png").is_file())
            self.assertFalse((root / "prediction_raw_gallery.png").exists())
            self.assertFalse((root / "prediction_physical_gallery.png").exists())
            self.assertTrue((root / "both/display/prediction_normal_rgb.png").is_file())
            self.assertFalse((root / "both/display_gamma22/prediction_normal_rgb.png").exists())
            self.assertTrue((root / "both/display_gamma22/prediction_diffuse_rgb.png").is_file())
            self.assertFalse((root / "both/display_gamma22/prediction_roughness.png").exists())
            self.assertTrue((root / "both/display_gamma22/prediction_specular_rgb.png").is_file())
            self.assertTrue((root / "common_display_gamma22" / "frame_near_00_bayer_u16.png").is_file())
            with Image.open(root / "common_display" / "frame_near_00_bayer_u16.png") as fallback:
                self.assertEqual(fallback.size, (32, 24))
                self.assertEqual(fallback.mode, "RGB")
            report = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("relation_log_raw=ln(relation_floor)", report)
            self.assertIn("valid_mask = 1 − sat", report)
            self.assertIn("normal_x", report)
            self.assertIn("component encoding", report)
            self.assertNotIn("Runtime LibRaw version: 0.22.0", report)
            self.assertIn("synthetic-group", report)
            self.assertIn("원본 near_00/far_00 단일 쌍 fallback", report)
            self.assertIn("frame 1 RMS &gt;1 px", report)
            self.assertIn("prepare_review.html", report)
            self.assertIn("independent_formula_verification.json", report)
            self.assertIn("both_only_capture_contract_snapshot.md", report)
            self.assertNotIn("capture_requirements_snapshot.md", report)
            self.assertNotIn("dng_pipeline_audit_snapshot.md", report)
            self.assertTrue((root / "capture_requirements_snapshot.md").is_file())
            self.assertTrue((root / "dng_pipeline_audit_snapshot.md").is_file())
            self.assertIn("감마 2.2", report)
            self.assertIn("toggleGammaDisplay", report)
            self.assertIn("camera-to-linear-sRGB matrix", report)
            self.assertIn("same-frame WB-demosaic RGB", report)
            self.assertIn("camera→linear RGB matrix", report)
            self.assertTrue((root / "both/display_gamma22/near_input_clipped.png").is_file())
            self.assertFalse((root / "both/display_gamma22/relation.png").exists())
            self.assertIn("WB diagonal matrix (3×3)", report)
            self.assertIn("n · 표면이 향하는 방향", report)
            self.assertIn("r · 표면 거칠기", report)
            self.assertNotIn('<option value="prediction_c0">', report)
            self.assertNotIn('<option value="prediction_c9">', report)
            self.assertNotIn('<option value="prediction_display_c0">', report)
            self.assertIn("data-gamma-src=\"both/display_gamma22/prediction_diffuse_rgb.png\"", report)
            self.assertIn("stage-nav", report)
            self.assertIn("aria-pressed", report)
            self.assertIn("@media(max-width:1050px)", report)
            self.assertIn("화면 밝기 감마 2.2 (기본 꺼짐)", report)
            self.assertIn("background:#111;color:#ddd", report)
            self.assertIn('class="case-row"', report)
            self.assertNotIn("prediction-grid", report)
            self.assertNotIn("comparison-grid", report)
            self.assertIn('class="case-label">Prediction</div>', report)
            self.assertIn('class="case-label">Inputs</div>', report)
            self.assertIn('class="case-tile"><img class="case-image" src="both/display/near_input_clipped.png" data-linear-src=', report)
            self.assertIn('data-gamma-src="both/display_gamma22/near_input_clipped.png"', report)
            self.assertNotIn('<pre id="audit">', report)
            self.assertNotIn("white_only", report)
            self.assertNotIn("neither", report)
            with Image.open(root / "prediction_semantic_gallery.png") as semantic_figure:
                self.assertGreater(semantic_figure.width, 1500)
                self.assertGreater(semantic_figure.height, 700)
                self.assertTrue(290 <= semantic_figure.info["dpi"][0] <= 310)
            with Image.open(root / "both/display/prediction_normal_rgb.png") as normal_preview:
                np.testing.assert_array_equal(np.asarray(normal_preview)[0, 0], [0, 28, 57])
            with Image.open(root / "both/display/prediction_diffuse_rgb.png") as diffuse_preview:
                base = np.asarray(diffuse_preview)[0, 0].astype(np.int16)
            with Image.open(root / "both/display_gamma22/prediction_diffuse_rgb.png") as diffuse_gamma:
                gamma = np.asarray(diffuse_gamma)[0, 0].astype(np.int16)
            self.assertGreater(int(np.max(np.abs(gamma - base))), 0)

            # The report-only refresh must rebuild display assets without inference.
            previous_prediction = original["both"]["prediction"].copy()
            refreshed = regenerate_capture_report(root)
            self.assertEqual(refreshed, root / "report.html")
            with np.load(root / "both" / "arrays.npz", allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["prediction"], previous_prediction)
            (root / "preparation.json").write_text(json.dumps({
                "review_assets": {"sha256": {"prepare_review.html": "latest-prepare-html-sha"}}
            }), encoding="utf-8")
            current_prepare_manifest_sha = sha256_file(root / "preparation.json")
            manifest_sha = sha256_file(root / "manifest.json")
            png_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*.png")}
            from unittest.mock import patch
            with patch("numpy.load", side_effect=AssertionError("HTML-only helper read an NPZ")):
                html_only_path = regenerate_capture_html_only(root)
            self.assertEqual(html_only_path, root / "report.html")
            self.assertEqual(sha256_file(root / "manifest.json"), manifest_sha)
            self.assertEqual(png_hashes, {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*.png")})
            generation = json.loads((root / "report_html_generation.json").read_text(encoding="utf-8"))
            self.assertEqual(generation["html_sha256"], sha256_file(root / "report.html"))
            self.assertEqual(generation["manifest_sha256_before"], manifest_sha)
            self.assertEqual(generation["manifest_sha256_after"], manifest_sha)
            self.assertEqual(generation["preparation_manifest_sha256_current"], current_prepare_manifest_sha)
            self.assertEqual(generation["preparation_review_assets_sha256_current"],
                             {"prepare_review.html": "latest-prepare-html-sha"})
            self.assertFalse(generation["png_assets_regenerated"])
            self.assertFalse(generation["numeric_arrays_read"])
            refreshed_html = (root / "report.html").read_text(encoding="utf-8")
            self.assertIn("report_html_generation.json", refreshed_html)
            self.assertIn(current_prepare_manifest_sha, refreshed_html)

            npz_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in root.rglob("*.npz")}
            refreshed_gallery = regenerate_capture_gallery_and_html(root)
            self.assertEqual(refreshed_gallery, root / "report.html")
            gallery_record = json.loads((root / "report_html_generation.json").read_text(encoding="utf-8"))
            self.assertEqual(gallery_record["active_condition"], "both")
            self.assertFalse(gallery_record["inference_executed"])
            self.assertTrue(gallery_record["numeric_archives_unchanged"])
            self.assertEqual(gallery_record["numeric_npz_sha256"]["both/arrays.npz"], npz_hashes["both/arrays.npz"])
            self.assertEqual(gallery_record["html_sha256"], sha256_file(refreshed_gallery))
            updated_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(updated_manifest["conditions"].keys(), {"both"})
            self.assertTrue((root / "features_gallery_both_only.png").is_file())
            self.assertTrue((root / "prediction_semantic_gallery_both_only.png").is_file())
            self.assertTrue((root / "prediction_semantic_gallery_both_only_gamma22.png").is_file())

    def test_feature_gallery_layout_follows_recorded_channel_contract(self):
        rgb21, scalar21, range21 = _feature_groups(21, {"feature_version": "raw_calibrated_v2"})
        self.assertEqual(len(rgb21), 6)
        self.assertEqual(len(scalar21), 3)
        self.assertEqual(rgb21[0], ("N", (0, 1, 2)))
        self.assertEqual(scalar21, [("relation", 18), ("log relation", 19), ("valid mask", 20)])
        self.assertEqual(range21, (-1.0, 1.0))

        rgb33, scalar33, range33 = _feature_groups(33, {"feature_version": "legacy_batch_v0"})
        self.assertEqual(len(rgb33), 10)
        self.assertEqual(len(scalar33), 3)
        self.assertEqual(rgb33[0], ("original near", (0, 1, 2)))
        self.assertEqual(rgb33[2], ("copy of near", (6, 7, 8)))
        self.assertEqual(rgb33[4], ("gained far", (12, 13, 14)))
        self.assertEqual(rgb33[5], ("log original near", (15, 16, 17)))
        self.assertEqual(scalar33, [("relation", 30), ("log relation", 31), ("valid mask", 32)])
        self.assertEqual(range33, (-1.0, 1.0))

    def test_author_real33_identity_slots_are_verified_and_gallery_uses_named_archive(self):
        feature = np.zeros((12, 14, 33), dtype=np.float32)
        feature[..., 0:3] = np.float32(-0.2)
        feature[..., 3:6] = np.float32(0.15)
        feature[..., 6:9] = feature[..., 0:3]
        feature[..., 9:12] = feature[..., 3:6]
        feature[..., 15:18] = np.float32(-0.4)
        feature[..., 18:21] = np.float32(0.3)
        feature[..., 21:24] = feature[..., 15:18]
        feature[..., 24:27] = feature[..., 18:21]
        self.assertTrue(_identity_copy_features_match({"both": feature}))
        changed = feature.copy()
        changed[0, 0, 6] = 1.0
        self.assertFalse(_identity_copy_features_match({"both": changed}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            produced = _figure_gallery(
                root,
                {mode: {"features_legacy33": feature} for mode in ("white_only", "neither", "both")},
                {"estimator_family": "author_legacy33", "feature_channels": 33},
            )
            self.assertIn("features_gallery.png", produced)

    def test_author_real_formula_and_channel_details_exclude_denoiser_stages(self):
        contract = {"estimator_family": "author_legacy33", "feature_channels": 33,
                    "normal_head": "author_legacy_tanh",
                    "prediction_channel_order": ["normal_x", "normal_y", "normal_z", "diffuse_r", "diffuse_g", "diffuse_b", "roughness", "specular_r", "specular_g", "specular_b"]}
        manifest = {"group": "synthetic-author", "checkpoint_sha256": "author-sha",
                    "model_contract": contract, "conditions": {mode: {} for mode in ("white_only", "neither", "both")},
                    "report": {}, "display_visual_review": {}}
        page = _make_html(manifest, ["near_input_clipped", "features_legacy33_c0",
                                     "log_original_6_near_rgb", "log_original_6_far_rgb"],
                          {"near_input_clipped": [0, 1], "features_legacy33_c0": [-1, 1],
                           "log_original_6_near_rgb": [0, 1], "log_original_6_far_rgb": [0, 1]},
                          {}, {}, [])
        self.assertIn("저자 real · denoiser 제외", page)
        self.assertIn("신경망 입력 33개 채널", page)
        self.assertIn("가까운 영상 미리보기", page)
        self.assertIn("log_original_6_near_rgb", page)
        self.assertIn("log_original_6_far_rgb", page)
        self.assertIn("전체 feature gallery", page)
        self.assertIn("feature-gallery-details", page)
        self.assertIn("scrollIntoView({block:'start'})", page)
        self.assertIn("identity copy of original near", page)
        self.assertIn("K_L", page)
        self.assertIn("[I_N, I_F]", page)
        self.assertNotIn("denoised near", page)
        self.assertNotIn("denoiser input", page)
        self.assertNotIn("white_only", page)
        self.assertNotIn("neither", page)

    def test_case_results_align_input_and_prediction_rows_for_author_and_best_only(self):
        modes = ("white_only", "neither", "both")
        component_keys = ("prediction_normal_rgb", "prediction_diffuse_rgb",
                          "prediction_roughness", "prediction_specular_rgb")
        conditions = {
            mode: {"arrays": {key: {} for key in (
                "near_input_clipped", "far_input_clipped", "relation", "relation_log", "prediction")}}
            for mode in modes
        }
        display_configs = {
            "near_input_clipped": {"rgb": True},
            "far_input_clipped": {"rgb": True},
        }
        assets = {}
        for mode in modes:
            assets[mode] = {}
            for key in component_keys:
                gammaable = key in {"prediction_diffuse_rgb", "prediction_specular_rgb"}
                assets[mode][key] = {
                    "author_real_no_denoiser": f"comparison/{mode}/author_{key}.png",
                    "best_render_preserved": f"comparison/{mode}/best_{key}.png",
                }
                if gammaable:
                    assets[mode][key]["author_real_no_denoiser_gamma22"] = f"comparison/{mode}/author_{key}_gamma22.png"
                    assets[mode][key]["best_render_preserved_gamma22"] = f"comparison/{mode}/best_{key}_gamma22.png"
        author_manifest = {
            "group": "author-group", "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
            "conditions": conditions,
            "report": {"checkpoint_comparison": {"status": "ready", "assets": assets,
                         "author_label": "저자 real · denoiser 제외", "reference_label": "best_render · 보존 결과",
                         "interpretation": "same-input comparison"}},
        }
        author_html = _make_html(author_manifest,
                                 ["near_input_clipped", "far_input_clipped", "relation", "relation_log"],
                                 {key: [0, 1] for key in ("near_input_clipped", "far_input_clipped", "relation", "relation_log")},
                                 display_configs, {}, [])
        self.assertEqual(author_html.count('<section class="case">'), 1)
        self.assertEqual(author_html.count('<div class="case-row">'), 3)
        self.assertIn('data-gamma-src="both/display_gamma22/near_input_clipped.png"', author_html)
        self.assertIn("comparison/both/author_prediction_normal_rgb.png", author_html)
        self.assertIn("comparison/both/best_prediction_normal_rgb.png", author_html)
        self.assertIn("comparison/both/author_prediction_diffuse_rgb_gamma22.png", author_html)
        self.assertIn("comparison/both/best_prediction_specular_rgb_gamma22.png", author_html)
        self.assertNotIn("white_only", author_html)
        self.assertNotIn("neither", author_html)
        self.assertNotIn("prediction-grid", author_html)

        best_manifest = {
            "group": "best-group", "checkpoint_sha256": "6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd",
            "conditions": conditions, "model_contract": {"feature_channels": 21},
        }
        best_html = _make_html(best_manifest,
                               ["near_input_clipped", "far_input_clipped", "relation", "relation_log"],
                               {key: [0, 1] for key in ("near_input_clipped", "far_input_clipped", "relation", "relation_log")},
                               display_configs, {}, [])
        self.assertEqual(best_html.count('<section class="case">'), 1)
        self.assertEqual(best_html.count('<div class="case-row">'), 2)
        self.assertNotIn("white_only", best_html)
        self.assertNotIn("neither", best_html)
        self.assertIn('class="case-label">best_render · 보존 결과</div>', best_html)
        self.assertNotIn('class="case-label">저자 real</div>', best_html)

    def test_author_html_only_refresh_links_audit_prepare_and_generation_without_common_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("white_only", "neither", "both"):
                (root / mode).mkdir()
            (root / "prepare_review.html").write_text("review", encoding="utf-8")
            (root / "independent_formula_verification.json").write_text("{}", encoding="utf-8")
            comparison = {
                "status": "ready", "author_label": "저자 real · denoiser 제외",
                "reference_label": "best_render · 보존 결과", "assets": {
                    "white_only": {"prediction_normal_rgb": {
                        "author_real_no_denoiser": "comparison/white_only/author_normal.png",
                        "best_render_preserved": "comparison/white_only/best_normal.png"}},
                    "both": {"prediction_normal_rgb": {
                        "author_real_no_denoiser": "comparison/both/author_normal.png",
                        "best_render_preserved": "comparison/both/best_normal.png"}},
                },
            }
            (root / "checkpoint_comparison_display.json").write_text(
                json.dumps(comparison, ensure_ascii=False), encoding="utf-8")
            manifest = {
                "group": "author-test",
                "conditions": {mode: {"arrays": {}} for mode in ("white_only", "neither", "both")},
                "display_ranges": {"near_input_clipped": [0.0, 1.0]},
                "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            before_manifest = sha256_file(root / "manifest.json")
            page_path = regenerate_capture_html_only(root)
            page = page_path.read_text(encoding="utf-8")
            self.assertIn("author_real_pipeline_snapshot.md", page)
            self.assertIn("prepare_review.html", page)
            self.assertIn("independent_formula_verification.json", page)
            self.assertIn("report_html_generation.json", page)
            self.assertIn("both_only_capture_contract_snapshot.md", page)
            self.assertIn("저자 real · denoiser 제외 / best_render · 보존 결과", page)
            self.assertIn("comparison/both/author_normal.png", page)
            self.assertNotIn("white_only", page)
            self.assertNotIn("neither", page)
            self.assertEqual(sha256_file(root / "manifest.json"), before_manifest)
            record = json.loads((root / "report_html_generation.json").read_text(encoding="utf-8"))
            self.assertFalse(record["inference_executed"])
            self.assertFalse(record["execution_manifest_changed"])
            self.assertEqual(record["checkpoint_comparison_display_sha256_current"],
                             sha256_file(root / "checkpoint_comparison_display.json"))

    def test_author_feature_gallery_adds_only_semantic_display_alias_pngs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("white_only", "neither", "both"):
                (root / mode).mkdir()
            features = np.zeros((12, 14, 33), dtype=np.float32)
            features[..., 6:9] = features[..., 0:3]
            features[..., 9:12] = features[..., 3:6]
            features[..., 21:24] = features[..., 15:18]
            features[..., 24:27] = features[..., 18:21]
            np.savez_compressed(root / "both" / "arrays.npz", features_legacy33=features)
            conditions = {mode: {"arrays": {"features_legacy33": {"shape": [12, 14, 33]}}}
                          for mode in ("white_only", "neither", "both")}
            manifest = {"conditions": conditions,
                        "display_ranges": {"features_legacy33": [-1.0, 1.0]},
                        "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            legacy_gallery = root / "features_gallery.png"
            legacy_gallery.write_bytes(b"immutable historical gallery")
            manifest_sha = sha256_file(root / "manifest.json")
            numeric_sha = sha256_file(root / "both" / "arrays.npz")
            legacy_sha = sha256_file(legacy_gallery)
            gallery = regenerate_feature_gallery_only(root)
            self.assertEqual(gallery, root / "features_gallery_both_only.png")
            self.assertTrue(gallery.is_file())
            self.assertEqual(sha256_file(legacy_gallery), legacy_sha)
            self.assertEqual(sha256_file(root / "manifest.json"), manifest_sha)
            self.assertEqual(sha256_file(root / "both" / "arrays.npz"), numeric_sha)
            generation = json.loads((root / "both_only_gallery_generation.json").read_text(encoding="utf-8"))
            self.assertTrue(generation["numeric_archives_unchanged"])
            self.assertTrue(generation["preexisting_pngs_unchanged"])
            self.assertFalse(generation["inference_executed"])
            self.assertEqual(generation["active_condition"], "both")

    def test_author_prediction_comparison_is_read_only_and_uses_shared_semantic_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            author = root / "fabric_capture_20261001_author_original" / "group-a"
            reference = root / "fabric_capture_20261001_clean" / "group-a"
            author.mkdir(parents=True)
            reference.mkdir(parents=True)
            reference_manifest = {
                "checkpoint": "checkpoints/best_render.pth",
                "checkpoint_sha256": "6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd",
            }
            (reference / "manifest.json").write_text(json.dumps(reference_manifest), encoding="utf-8")
            (reference / "report.html").write_text("preserved report", encoding="utf-8")
            old_arrays = {}
            new_arrays = {}
            yy, xx = np.mgrid[:4, :5]
            prediction = np.zeros((4, 5, 10), dtype=np.float32)
            prediction[..., 0] = 1.0
            prediction[..., 1] = -1.0
            prediction[..., 2] = 0.0
            prediction[..., 3:6] = 0.0
            prediction[..., 6] = (xx + yy) / 8.0 * 2.0 - 1.0
            prediction[..., 7:10] = 0.0
            for mode in ("white_only", "neither", "both"):
                (reference / mode).mkdir()
                np.savez_compressed(reference / mode / "arrays.npz", prediction=prediction + np.float32(0.0))
                old_arrays[mode] = {"prediction": prediction.copy()}
                new_arrays[mode] = {"prediction_nchw": prediction[None].transpose(0, 3, 1, 2).copy()}
            before = {mode: sha256_file(reference / mode / "arrays.npz")
                      for mode in ("white_only", "neither", "both")}
            comparison, assets = _write_checkpoint_comparison(
                author, {"group": "group-a", "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33}},
                new_arrays)
            self.assertEqual(comparison["status"], "ready")
            self.assertTrue(comparison["reference_numeric_archives_unchanged"])
            self.assertEqual(before, {mode: sha256_file(reference / mode / "arrays.npz")
                                      for mode in ("white_only", "neither", "both")})
            self.assertGreater(len(assets), 0)
            self.assertEqual(comparison["author_label"], "저자 real · denoiser 제외")
            self.assertEqual(comparison["reference_label"], "best_render · 보존 결과")
            diffuse = author / comparison["assets"]["both"]["prediction_diffuse_rgb"]["author_real_no_denoiser"]
            diffuse_gamma = author / comparison["assets"]["both"]["prediction_diffuse_rgb"]["author_real_no_denoiser_gamma22"]
            with Image.open(diffuse) as image:
                self.assertEqual(np.asarray(image)[0, 0].tolist(), [128, 128, 128])
            with Image.open(diffuse_gamma) as image:
                self.assertTrue(np.all(np.abs(np.asarray(image)[0, 0].astype(int) - 186) <= 1))
            self.assertNotIn("author_real_no_denoiser_gamma22",
                             comparison["assets"]["both"]["prediction_normal_rgb"])
            self.assertNotIn("author_real_no_denoiser_gamma22",
                             comparison["assets"]["both"]["prediction_roughness"])
            comparison_page = _make_html(
                {"group": "group-a", "model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
                 "conditions": {mode: {} for mode in ("white_only", "neither", "both")},
                 "report": {"checkpoint_comparison": comparison}},
                [], {}, {}, {}, [])
            self.assertIn("best_render · 보존 결과", comparison_page)
            self.assertIn("not a weight-only experiment", comparison_page)
            self.assertIn("case-image[data-gamma-src]", comparison_page)

    def test_rgb_display_transform_order_and_gamma_pixels(self):
        raw = np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        original = raw.copy()
        manifest = {"color_transforms": {"near": {
            "normalized_rgb_wb": [2.0, 1.0, 3.0],
            "camera_to_linear_srgb": [[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]],
        }}}
        camera_preview = _display_rgb(raw, "near_counts", manifest)
        np.testing.assert_allclose(camera_preview, [[[3.0, 2.0, 18.0]]], rtol=0, atol=0)
        # Already-whitebalanced RGB receives the matrix once and skips WB.
        whitebalanced_preview = _display_rgb(raw, "near_whitebalanced", manifest)
        np.testing.assert_allclose(whitebalanced_preview, [[[2.0, 2.0, 6.0]]], rtol=0, atol=0)
        np.testing.assert_array_equal(raw, original)

        with tempfile.TemporaryDirectory() as directory:
            linear_path, gamma_path = Path(directory) / "linear.png", Path(directory) / "gamma.png"
            pixel = np.full((1, 1, 3), 0.25, dtype=np.float32)
            _render_array(pixel, "near_linear_srgb", linear_path, (0.0, 1.0), "cividis")
            _render_array(pixel, "near_linear_srgb", gamma_path, (0.0, 1.0), "cividis", gamma22=True)
            with Image.open(linear_path) as image:
                linear_rgb = np.asarray(image)[0, 0]
            with Image.open(gamma_path) as image:
                gamma_rgb = np.asarray(image)[0, 0]
        self.assertTrue(np.all(np.abs(linear_rgb.astype(int) - 64) <= 1))
        expected = int(round(float(_gamma22(np.asarray([0.25], dtype=np.float32))[0]) * 255))
        self.assertTrue(np.all(np.abs(gamma_rgb.astype(int) - expected) <= 1))


if __name__ == "__main__":
    unittest.main()
