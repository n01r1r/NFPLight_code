"""CPU invariants for the RAW feature, target support and float-map contracts."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from data.fabric import canonical_target
from model.nfplight_model import NFPLightModel, checkpoint_feature_version


def feature_model():
    model = NFPLightModel.__new__(NFPLightModel)
    model.feature_version = 'raw_calibrated_v2'
    model.near_distance, model.far_distance = 2.414, 10.0
    model.coefficient = torch.linspace(.001, 1, 256 * 256).reshape(1, 1, 256, 256)
    model.time_co_map = torch.ones_like(model.coefficient) * .9
    model.indenty = torch.ones_like(model.coefficient)
    return model


class RawCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_calibrated_scale_survives_zero_or_saturated_center(self):
        model = feature_model()
        pair = torch.full((1, 6, 256, 256), .01)
        pair[:, :3] *= 10
        expected = .01 * (10 / 2.414)**2
        for center in (0., 1.):
            pair[:, :, 127, 127] = center
            feature, _, aligned = model.build_features(pair)
            self.assertAlmostEqual(float(aligned[0, 0, 0, 0]), expected, places=6)
            self.assertTrue(torch.isfinite(feature).all())
            self.assertLessEqual(float(feature.amax()), 1.)
            self.assertGreaterEqual(float(feature.amin()), -1.)
        self.assertEqual(checkpoint_feature_version({}), 'legacy_batch_v0')

    def test_invalid_pixels_cannot_change_valid_features(self):
        model = feature_model()
        pair = torch.rand((1, 6, 256, 256), generator=torch.Generator().manual_seed(8)) * .01
        valid = torch.ones((1, 1, 256, 256), dtype=torch.bool)
        valid[:, :, :8, :8] = False
        expected = model.build_features(pair, valid)[0]
        pair[:, :, :8, :8] = 1
        actual = model.build_features(pair, valid)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue((actual[:, :, :8, :8] == -1).all())
        with self.assertRaisesRegex(ValueError, 'valid pixels'):
            model.build_features(pair, torch.zeros_like(valid))

    def test_target_masks_without_flipping_or_changing_valid_colors(self):
        target = torch.zeros(1, 10, 4, 4)
        target[:, 2] = 1
        target[:, 6] = -1
        target[:, 2, 0, 0] = -1
        safe, valid = canonical_target(target)
        self.assertFalse(valid[0, 0, 0, 0])
        self.assertEqual(float(target[0, 2, 0, 0]), -1)
        self.assertAlmostEqual(float((safe[0, 6, 1, 1] + 1) / 2), .05, places=7)
        torch.testing.assert_close(safe[:, 3:6, 1:, 1:], target[:, 3:6, 1:, 1:])
        changed = target.clone()
        changed[:, 3:, 0, 0] = .9
        torch.testing.assert_close(canonical_target(changed)[0], safe, rtol=0, atol=0)
        target[:, 2] = -1
        with self.assertRaisesRegex(ValueError, 'no valid'):
            canonical_target(target)

    def test_float_output_preserves_linear_colors_and_unit_normal(self):
        model = feature_model()
        prediction = torch.zeros(1, 10, 4, 4)
        prediction[:, 2] = .8
        prediction[:, 6] = -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prediction'
            model.save_svbrdf_maps(prediction, path, metadata={'test': True})
            with np.load(str(path) + '.npz', allow_pickle=False) as maps:
                self.assertEqual(maps['normal'].dtype, np.float32)
                np.testing.assert_allclose(np.linalg.norm(maps['normal'], axis=-1), 1)
                np.testing.assert_array_equal(maps['diffuse'], np.full((4, 4, 3), .5, np.float32))
                np.testing.assert_allclose(maps['roughness'], .05)
            metadata = json.loads(Path(str(path) + '.json').read_text())
            self.assertEqual(metadata['working_space'], 'linear_srgb')
            self.assertEqual(metadata['feature_version'], 'raw_calibrated_v2')
            prediction[:, :3] = 0
            with self.assertRaisesRegex(ValueError, 'zero predicted normal'):
                model.save_svbrdf_maps(prediction, path)


if __name__ == '__main__':
    unittest.main()
