"""CPU contracts for the fixed XYZ and spherical normal output heads."""
import math
import tempfile
import unittest
from pathlib import Path

import torch

from train_fabric import apply_normal_head, build_contract, parse_args
from model.normal_head import checkpoint_normal_head


class NormalHeadTests(unittest.TestCase):
    def test_xyz_head_is_an_identity(self):
        prediction = torch.randn(2, 10, 3, 4)
        actual = apply_normal_head(prediction, normal_head="xyz")
        self.assertIs(actual, prediction)

    def test_phi_theta_head_reconstructs_unit_upper_hemisphere_normals(self):
        raw = torch.zeros(1, 10, 1, 4)
        # The fixed head reads raw [z] as theta and raw [x,y] as the
        # (cos(phi), sin(phi)) direction.  Tanh bounds map raw z to [0, pi/2].
        raw[:, 2, 0, 0] = -1.0  # north pole; azimuth is immaterial
        raw[:, 0, 0, 1] = 1.0
        raw[:, 2, 0, 1] = 1.0  # theta=pi/2, phi=0 -> +x
        raw[:, 1, 0, 2] = 1.0
        raw[:, 2, 0, 2] = 1.0  # theta=pi/2, phi=pi/2 -> +y
        raw[:, 0, 0, 3] = -1.0
        raw[:, 2, 0, 3] = 0.0  # theta=pi/4, phi=pi -> (-sqrt(.5), 0, sqrt(.5))
        raw[:, 3:, 0, :] = 0.37

        actual = apply_normal_head(raw, normal_head="phi_theta")
        normals = actual[:, :3]
        torch.testing.assert_close(normals[:, :, 0, 0], torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(normals[:, :, 0, 1], torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-6, rtol=0)
        torch.testing.assert_close(normals[:, :, 0, 2], torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-6, rtol=0)
        expected = torch.tensor([[-math.sqrt(0.5), 0.0, math.sqrt(0.5)]])
        torch.testing.assert_close(normals[:, :, 0, 3], expected, atol=1e-6, rtol=0)
        torch.testing.assert_close(normals.norm(dim=1), torch.ones(1, 1, 4), atol=1e-6, rtol=0)
        self.assertGreaterEqual(float(normals[:, 2].min()), -1e-7)
        torch.testing.assert_close(actual[:, 3:], raw[:, 3:])

    def test_phi_theta_head_has_finite_backward_at_undefined_azimuth(self):
        raw = torch.zeros(1, 10, 2, 2)
        raw[:, 2].fill_(0.2)
        raw.requires_grad_()
        value = apply_normal_head(raw, normal_head="phi_theta")[:, 2].sum()
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_cli_and_contract_identify_head_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            xyz = build_contract(parse_args(["--out", "unused"]), manifest)
            spherical = build_contract(parse_args(["--out", "unused", "--normal-head", "phi_theta"]), manifest)
            xyz_phi_theta = build_contract(parse_args([
                "--out", "unused", "--normal-head", "xyz", "--normal-loss", "phi_theta"
            ]), manifest)
        self.assertEqual(xyz["settings"]["normal_head"], "xyz")
        self.assertIn("model/normal_head.py", xyz["source_sha256"])
        self.assertEqual(spherical["settings"]["normal_head"], "phi_theta")
        self.assertEqual(spherical["normal_objective"]["head"], "phi_theta")
        self.assertNotEqual(xyz["normal_objective"]["head_formula"], spherical["normal_objective"]["head_formula"])
        self.assertEqual(xyz_phi_theta["settings"]["normal_head"], "xyz")
        self.assertEqual(xyz_phi_theta["normal_objective"]["head"], "xyz")
        self.assertEqual(xyz_phi_theta["normal_objective"]["kind"], "phi_theta")
        self.assertEqual(xyz_phi_theta["normal_objective"]["head_formula"], "identity(raw_xyz)")

    def test_checkpoint_head_metadata_defaults_to_legacy_xyz(self):
        self.assertEqual(checkpoint_normal_head({"params": {}}), "xyz")
        self.assertEqual(
            checkpoint_normal_head({
                "params": {},
                "contract": {"settings": {"normal_head": "phi_theta"},
                              "normal_objective": {"head": "phi_theta"}},
            }),
            "phi_theta",
        )

    def test_checkpoint_head_metadata_rejects_conflicts_and_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            checkpoint_normal_head({
                "contract": {"settings": {"normal_head": "xyz"},
                              "normal_objective": {"head": "phi_theta"}},
            })
        with self.assertRaisesRegex(ValueError, "unsupported"):
            checkpoint_normal_head({
                "contract": {"settings": {"normal_head": "other"}},
            })


if __name__ == "__main__":
    unittest.main()
