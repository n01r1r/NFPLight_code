"""CPU contracts for RAW capture loading and float-map inference export."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from data.dataset import RealDataset
from model.nfplight_matsynth_capture_model import MatSynthCaptureModel
from real import resolve_input_mode


def raw_metadata():
    return {
        'working_space': 'linear_srgb',
        'near_distance': 2.414,
        'far_distance': 10.0,
        'reference_exposure': 'equal',
        'reference_flux': 'equal',
        'preprocessing_done_externally': True,
    }


def write_metadata(capture):
    (capture / 'metadata.json').write_text(
        json.dumps(raw_metadata()), encoding='utf-8'
    )


class RawCaptureTests(unittest.TestCase):
    def make_capture(self, root, near, far, *, metadata=True, ext='npy'):
        capture = Path(root) / 'capture'
        capture.mkdir()
        if ext == 'npy':
            np.save(capture / 'near.npy', near)
            np.save(capture / 'far.npy', far)
        else:
            # cv2 writes BGR, while RealDataset converts back to RGB.
            cv2.imwrite(str(capture / 'near.png'), near[:, :, ::-1])
            cv2.imwrite(str(capture / 'far.png'), far[:, :, ::-1])
        if metadata:
            write_metadata(capture)
        return capture

    def test_float32_linear_values_and_hashes_survive(self):
        with tempfile.TemporaryDirectory() as root:
            near = np.full((2, 3, 3), 0.5, dtype=np.float32)
            far = np.full((2, 3, 3), 0.25, dtype=np.float32)
            self.make_capture(root, near, far)
            sample = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]
            self.assertEqual(sample['inputs'].dtype, torch.float32)
            self.assertEqual(tuple(sample['inputs'].shape), (6, 2, 3))
            self.assertTrue(torch.equal(sample['inputs'][:3], torch.full((3, 2, 3), 0.5)))
            self.assertTrue(torch.equal(sample['inputs'][3:], torch.full((3, 2, 3), 0.25)))
            self.assertEqual(set(sample['input_hashes']), {'near', 'far', 'metadata'})
            with self.assertRaisesRegex(ValueError, 'square patch'):
                RealDataset({'root': root, 'input_mode': 'linear_rgb', 'imageSize': 256})[0]

    def test_uint16_linear_png_uses_65535_without_gamma(self):
        with tempfile.TemporaryDirectory() as root:
            near_u16 = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
            far_u16 = np.array([[[65535, 32768, 0]]], dtype=np.uint16)
            self.make_capture(root, near_u16, far_u16, ext='png')
            sample = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]
            expected = torch.tensor([0.0, 32768 / 65535.0, 1.0])
            torch.testing.assert_close(sample['inputs'][:3, 0, 0], expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(sample['inputs'][3:, 0, 0], expected.flip(0), rtol=0, atol=1e-7)

    def test_gamma22_decodes_before_resizing_checkerboard(self):
        with tempfile.TemporaryDirectory() as root:
            # Mean linear intensity is 0.5. Averaging encoded codes first would
            # produce (0.5 ** 2.2), proving the order of operations.
            checker = np.array([[0, 255], [255, 0]], dtype=np.uint8)
            checker = np.repeat(checker[:, :, None], 3, axis=2)
            self.make_capture(root, checker, checker, metadata=False, ext='png')
            sample = RealDataset({'root': root, 'input_mode': 'gamma22', 'imageSize': 1})[0]
            self.assertAlmostEqual(float(sample['inputs'][0, 0, 0]), 0.5, places=6)
            self.assertAlmostEqual(float(sample['inputs'][3, 0, 0]), 0.5, places=6)

    def test_invalid_raw_inputs_and_metadata_are_rejected(self):
        cases = (
            ('dtype', np.full((2, 2, 3), 0.5, dtype=np.float64), np.full((2, 2, 3), 0.5, dtype=np.float32), TypeError),
            ('range', np.full((2, 2, 3), 1.1, dtype=np.float32), np.full((2, 2, 3), 0.5, dtype=np.float32), ValueError),
            ('finite', np.full((2, 2, 3), np.nan, dtype=np.float32), np.full((2, 2, 3), 0.5, dtype=np.float32), ValueError),
            ('shape', np.full((2, 2, 4), 0.5, dtype=np.float32), np.full((2, 2, 3), 0.5, dtype=np.float32), ValueError),
        )
        for label, near, far, error in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as root:
                self.make_capture(root, near, far)
                with self.assertRaises(error):
                    _ = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]
        with tempfile.TemporaryDirectory() as root:
            near = np.full((2, 2, 3), 0.5, dtype=np.float32)
            far = np.full((3, 2, 3), 0.5, dtype=np.float32)
            self.make_capture(root, near, far)
            with self.assertRaisesRegex(ValueError, 'height mismatch'):
                _ = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]
        with tempfile.TemporaryDirectory() as root:
            near = np.full((2, 2, 3), 0.5, dtype=np.float32)
            far = np.full((2, 2, 3), 0.5, dtype=np.float32)
            self.make_capture(root, near, far, metadata=False)
            with self.assertRaisesRegex(FileNotFoundError, 'metadata'):
                _ = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]
        for field, value in (('reference_scale', 2.0), ('reference_scale', float('nan')),
                             ('reference_flux', 'unequal')):
            with self.subTest(metadata=field), tempfile.TemporaryDirectory() as root:
                near = np.full((2, 2, 3), 0.5, dtype=np.float32)
                far = np.full((2, 2, 3), 0.5, dtype=np.float32)
                capture = self.make_capture(root, near, far)
                document = raw_metadata()
                document['reference_scale'] = 1.0
                document[field] = value
                (capture / 'metadata.json').write_text(json.dumps(document), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'reference'):
                    _ = RealDataset({'root': root, 'input_mode': 'linear_rgb'})[0]

    def test_raw_checkpoint_rejects_gamma_mode(self):
        with self.assertRaisesRegex(ValueError, 'incompatible'):
            resolve_input_mode('gamma22', 'matsynth21', 'raw_calibrated_v2')
        self.assertEqual(
            resolve_input_mode('auto', 'matsynth21', 'raw_calibrated_v2'), 'linear_rgb'
        )

    def test_dataloader_metadata_collation_and_float_export(self):
        with tempfile.TemporaryDirectory() as root:
            near = np.full((256, 256, 3), 0.5, dtype=np.float32)
            far = np.full((256, 256, 3), 0.25, dtype=np.float32)
            self.make_capture(root, near, far)
            dataset = RealDataset({'root': root, 'input_mode': 'linear_rgb'})
            loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
            batch = next(iter(loader))
            model = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
            model.device = torch.device('cpu')
            model.args = SimpleNamespace(image_size=256, input_mode='linear_rgb')
            model.feed_data(batch)
            self.assertEqual(model.inputs.dtype, torch.float32)
            self.assertEqual(model.capture_metadata['working_space'], 'linear_srgb')
            self.assertEqual(len(model.input_hashes['near']), 64)
            with self.assertRaisesRegex(TypeError, 'floating-point'):
                model.feed_data({'inputs': torch.ones((1, 6, 256, 256), dtype=torch.uint8)})

            prediction = torch.zeros((1, 10, 256, 256), dtype=torch.float32)
            prediction[:, 2] = 1.0
            with tempfile.TemporaryDirectory() as output:
                model.save_svbrdf_maps(prediction, str(Path(output) / 'capture'), metadata={
                    'input_hashes': model.input_hashes,
                    'capture_metadata': model.capture_metadata,
                })
                with np.load(Path(output) / 'capture.npz') as archive:
                    self.assertEqual(archive['normal'].dtype, np.float32)
                    self.assertEqual(archive['normal'].shape, (256, 256, 3))
                document = json.loads((Path(output) / 'capture.json').read_text())
                self.assertEqual(document['working_space'], 'linear_srgb')
                self.assertEqual(document['provenance']['capture_metadata']['near_distance'], 2.414)


if __name__ == '__main__':
    unittest.main()
