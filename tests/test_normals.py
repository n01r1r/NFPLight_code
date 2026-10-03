"""CPU invariants for checkpoint normal heads and normal objectives."""
import math
import tempfile
import unittest
from pathlib import Path
import torch
from model.normal_head import checkpoint_normal_head
from train_fabric import apply_normal_head, build_contract, parse_args
from types import SimpleNamespace
from torch import nn
from train_fabric import (
    FEATURE_VERSION,
    build_contract,
    load_checkpoint,
    map_loss,
    normal_loss,
    parse_args,
    save_checkpoint,
)


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


def vectors(*values):
    """Pack N vectors as [1,3,1,N] float32 tensors."""
    return torch.tensor(values, dtype=torch.float32).transpose(0, 1).reshape(1, 3, 1, -1)


def full_mask(count):
    return torch.ones(1, 1, 1, count, dtype=torch.bool)


class NormalLossTests(unittest.TestCase):
    def test_normal_loss_defaults_to_established_l1(self):
        pred = vectors((0.3, 0.4, 0.8660254))
        target = vectors((0.2, 0.5, 0.8426149))
        mask = full_mask(1)
        torch.testing.assert_close(normal_loss(pred, target, mask),
                                   normal_loss(pred, target, mask, kind="l1"))

    def test_geodesic_is_normalized_shortest_arc(self):
        same = vectors((0, 0, 1))
        x = vectors((1, 0, 0))
        y = vectors((0, 1, 0))
        opposite = vectors((0, 0, -1))
        mask = full_mask(1)
        self.assertAlmostEqual(float(normal_loss(same, same, mask, kind="geodesic")), 0.0, places=7)
        self.assertAlmostEqual(float(normal_loss(same, x, mask, kind="geodesic")), 0.5, places=6)
        self.assertAlmostEqual(float(normal_loss(x, y, mask, kind="geodesic")), 0.5, places=6)
        self.assertAlmostEqual(float(normal_loss(same, opposite, mask, kind="geodesic")), 1.0, places=6)

    def test_known_angles_cosine_and_phi_theta(self):
        same = vectors((0, 0, 1))
        x = vectors((1, 0, 0))
        y = vectors((0, 1, 0))
        minus_x = vectors((-1, 0, 0))
        mask = full_mask(1)
        self.assertAlmostEqual(float(normal_loss(same, same, mask, kind="cosine")), 0.0, places=7)
        self.assertAlmostEqual(float(normal_loss(same, x, mask, kind="cosine")), 1.0, places=7)
        self.assertAlmostEqual(float(normal_loss(same, -same, mask, kind="cosine")), 2.0, places=7)
        # A pole has no defined azimuth, so only its pi/2 theta error remains.
        self.assertAlmostEqual(float(normal_loss(same, x, mask, kind="phi_theta")), 0.5, places=6)
        self.assertAlmostEqual(float(normal_loss(same, y, mask, kind="phi_theta")), 0.5, places=6)
        self.assertAlmostEqual(float(normal_loss(x, y, mask, kind="phi_theta")), 0.5, places=6)
        self.assertAlmostEqual(float(normal_loss(x, minus_x, mask, kind="phi_theta")), 1.0, places=6)

    def test_phi_periodic_seam_wraps_to_short_arc(self):
        eps = 1e-3
        pred = vectors((-math.cos(eps), math.sin(eps), 0.0))
        target = vectors((-math.cos(eps), -math.sin(eps), 0.0))
        value = normal_loss(pred, target, full_mask(1), kind="phi_theta")
        self.assertAlmostEqual(float(value), 2 * eps / math.pi, places=6)
        geodesic = normal_loss(pred, target, full_mask(1), kind="geodesic")
        self.assertAlmostEqual(float(geodesic), 2 * eps / math.pi, places=6)

    def test_direction_losses_are_positive_scale_invariant(self):
        pred = vectors((0.3, 0.4, 0.8660254))
        scaled = pred * 17.0
        target = vectors((0.2, 0.5, 0.8426149))
        mask = full_mask(1)
        for kind in ("cosine", "phi_theta", "geodesic"):
            torch.testing.assert_close(normal_loss(pred, target, mask, kind=kind),
                                       normal_loss(scaled, target, mask, kind=kind),
                                       rtol=1e-5, atol=1e-6)

    def test_finite_useful_gradients_away_from_singularities(self):
        target = vectors((0.2, 0.5, 0.8426149))
        mask = full_mask(1)
        for kind in ("cosine", "phi_theta", "geodesic"):
            pred = vectors((0.3, 0.4, 0.8660254)).requires_grad_()
            normal_loss(pred, target, mask, kind=kind).backward()
            self.assertTrue(torch.isfinite(pred.grad).all())
            self.assertGreater(float(pred.grad.norm()), 1e-7)

    def test_exact_and_near_poles_have_finite_backward(self):
        target = vectors((0.2, 0.5, 0.8426149))
        mask = full_mask(1)
        for values in ((0.0, 0.0, 1.0), (1e-8, -2e-8, 1.0)):
            for kind in ("cosine", "phi_theta", "geodesic"):
                pred = vectors(values).requires_grad_()
                value = normal_loss(pred, target, mask, kind=kind)
                value.backward()
                self.assertTrue(torch.isfinite(value))
                self.assertTrue(torch.isfinite(pred.grad).all(), msg=kind)

    def test_geodesic_endpoints_have_finite_backward(self):
        pred = vectors((0.0, 0.0, 1.0)).requires_grad_()
        for target_values in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)):
            pred.grad = None
            target = vectors(target_values)
            value = normal_loss(pred, target, full_mask(1), kind="geodesic")
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_invalid_mask_is_invariant_and_has_zero_invalid_gradient(self):
        base_pred = vectors((0.3, 0.4, 0.8660254), (3.0, -4.0, 5.0))
        target = vectors((0.2, 0.5, 0.8426149), (0.1, 0.2, 0.3))
        mask = torch.tensor([[[[True, False]]]])
        corrupted = target.clone()
        corrupted[:, :, :, 1] = torch.tensor([100.0, -100.0, 50.0]).view(1, 3, 1)
        for kind in ("cosine", "phi_theta", "geodesic"):
            pred = base_pred.clone().requires_grad_()
            first = normal_loss(pred, target, mask, kind=kind)
            second = normal_loss(pred.detach(), corrupted, mask, kind=kind)
            torch.testing.assert_close(first.detach(), second)
            first.backward()
            self.assertTrue(torch.equal(pred.grad[:, :, :, 1], torch.zeros_like(pred.grad[:, :, :, 1])))

    def test_rejections_for_zero_nonfinite_shape_and_empty_support(self):
        target = vectors((0.2, 0.5, 0.8426149))
        mask = full_mask(1)
        with self.assertRaisesRegex(FloatingPointError, "zero or near-zero"):
            normal_loss(torch.zeros_like(target), target, mask, kind="cosine")
        nonfinite = target.clone()
        nonfinite[:, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            normal_loss(nonfinite, target, mask, kind="phi_theta")
        with self.assertRaisesRegex(ValueError, "equal"):
            normal_loss(target, torch.cat((target, target), dim=-1), mask, kind="cosine")
        with self.assertRaisesRegex(ValueError, "zero valid support"):
            normal_loss(target, target, torch.zeros_like(mask), kind="cosine")
        with self.assertRaisesRegex(ValueError, "unknown"):
            normal_loss(target, target, mask, kind="quadratic")

    def test_map_l1_formula_and_other_components_are_unchanged(self):
        pred = torch.zeros(1, 10, 1, 2)
        target = torch.zeros_like(pred)
        pred[:, 0:3, 0, 0] = 0.25
        target[:, 0:3, 0, 0] = -0.25
        pred[:, 3:, 0, 0] = 0.5
        target[:, 3:, 0, 0] = 0.0
        valid = torch.tensor([[[[True, False]]]])
        l1_total, l1_parts = map_loss(pred, target, valid, normal_kind="l1")
        expected_normal = (pred[:, :3, :, :] - target[:, :3, :, :]).abs()[:, :, :, 0].mean()
        self.assertAlmostEqual(float(l1_parts["normal"]), float(expected_normal), places=7)
        self.assertAlmostEqual(float(l1_total), float(sum(l1_parts.values())), places=7)
        for kind in ("cosine", "phi_theta", "geodesic"):
            _, parts = map_loss(pred, target, valid, normal_kind=kind)
            for name in ("diffuse", "roughness", "specular"):
                torch.testing.assert_close(parts[name], l1_parts[name])

    def test_cli_and_contract_identify_each_objective(self):
        contracts = {}
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            for kind in ("l1", "cosine", "phi_theta", "geodesic"):
                args = parse_args(["--out", "unused", "--normal-loss", kind])
                self.assertEqual(args.normal_loss, kind)
                contract = build_contract(args, manifest)
                contracts[kind] = contract["normal_objective"]
                self.assertEqual(contract["input"]["feature_version"], FEATURE_VERSION)
                self.assertEqual(contract["normal_objective"]["kind"], kind)
                self.assertEqual(contract["normal_objective"]["weight"], 1.0)
                self.assertIn("selection_metrics", contract["normal_objective"])
        self.assertNotEqual(contracts["l1"]["formula"], contracts["cosine"]["formula"])
        self.assertNotEqual(contracts["cosine"]["formula"], contracts["phi_theta"]["formula"])

    def test_checkpoint_rejects_changed_normal_objective_contract(self):
        net = nn.Conv2d(1, 1, 1)
        model = SimpleNamespace(net_g=net, feature_version=FEATURE_VERSION)
        other = SimpleNamespace(net_g=nn.Conv2d(1, 1, 1), feature_version=FEATURE_VERSION)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            l1 = build_contract(parse_args(["--out", "unused", "--normal-loss", "l1"]), manifest)
            cosine = build_contract(parse_args(["--out", "unused", "--normal-loss", "cosine"]), manifest)
            checkpoint = Path(temporary) / "normal.pt"
            save_checkpoint(checkpoint, model, l1, {"stage": "map", "step": 0, "total_step": 0})
            load_checkpoint(checkpoint, other, l1)
            with self.assertRaisesRegex(ValueError, "contract differs"):
                load_checkpoint(checkpoint, other, cosine)


if __name__ == '__main__':
    unittest.main()
