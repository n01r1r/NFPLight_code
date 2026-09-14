"""Focused smoke checks for real-capture estimator family wiring.

These checks avoid allocating a second full estimator while a training job is
running.  The CPU harness still exercises the inherited synthetic ``test``
assembly with a shape-compatible stand-in network.
"""

import hashlib
import json
import tempfile
from types import SimpleNamespace

import torch

from real import (
    MatSynthCaptureModel,
    detect_estimator_family,
    resolve_estimator_family,
    write_inference_manifest,
)
from model.nfplight_matsynth_capture_model import normalize_checkpoint_state_dict
from test import validate_synthetic_checkpoint


class DummyEstimator(torch.nn.Module):
    def forward(self, inputs):
        assert inputs.shape[1] == 21, inputs.shape
        return torch.zeros(
            inputs.shape[0], 10, inputs.shape[2], inputs.shape[3],
            device=inputs.device,
        )


def check_family_detection():
    with tempfile.TemporaryDirectory() as output_dir:
        matsynth_path = f"{output_dir}/matsynth21.pth"
        legacy_path = f"{output_dir}/legacy33.pth"
        torch.save({
            "params": {
                "intro_albedo.weight": torch.zeros(32, 18, 3, 3),
            }
        }, matsynth_path)
        torch.save({
            "params": {
                "intro_albedo.weight": torch.zeros(32, 30, 3, 3),
            }
        }, legacy_path)
        assert detect_estimator_family(matsynth_path) == "matsynth21"
        assert detect_estimator_family(legacy_path) == "legacy33"
        try:
            resolve_estimator_family("legacy33", matsynth_path)
        except ValueError as error:
            assert "conflicts" in str(error)
        else:
            raise AssertionError("family mismatch was accepted")

    normalized = normalize_checkpoint_state_dict({
        "state_dict": {
            "module._orig_mod.intro_albedo.weight": torch.zeros(32, 18, 3, 3),
            "module._orig_mod.intro_albedo.bias": torch.zeros(32),
        }
    })
    assert "intro_albedo.weight" in normalized
    assert "intro_albedo.bias" in normalized


def check_matsynth_capture_shape_and_no_denoise():
    assert "test" in MatSynthCaptureModel.__dict__
    assert "init_distance" not in MatSynthCaptureModel.__dict__
    model = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
    model.device = torch.device("cpu")
    model.args = SimpleNamespace(image_size=256)
    model.coefficient = torch.ones(1, 1, 256, 256)
    model.time_co_map = torch.ones(1, 1, 256, 256)
    model.indenty = torch.ones(1, 1, 256, 256)
    model.net_g = DummyEstimator()
    model.normal_head = "phi_theta"
    model.log_normalization = lambda image, eps=1e-2: (
        torch.log(image + eps) / torch.log(torch.tensor(1.0 + eps))
    )
    model.feed_data({
        "inputs": torch.rand(1, 6, 256, 256) * 0.8 + 0.1,
        "name": ["capture"],
    })
    model.test()
    assert model.pred_svbrdf.shape == (1, 10, 256, 256)
    assert torch.allclose(
        model.pred_svbrdf[:, :3].norm(dim=1),
        torch.ones(1, 256, 256),
        atol=1e-6,
    )
    assert float(model.pred_svbrdf[:, 2].min()) >= -1e-7
    assert not hasattr(model, "net_denoise")

    for invalid_inputs, expected_message in (
        (torch.rand(6, 256, 256), "rank-4"),
        (torch.full((1, 6, 256, 256), 1.1), "linear inputs"),
        (torch.full((1, 6, 256, 256), float("nan")), "non-finite"),
    ):
        try:
            model.feed_data({"inputs": invalid_inputs})
        except ValueError as error:
            assert expected_message in str(error)
        else:
            raise AssertionError(f"invalid input was accepted: {expected_message}")


def check_checkpoint_head_metadata_load():
    class TinyEstimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_path = f"{output_dir}/phi.pth"
        torch.save({
            "feature_version": "raw_calibrated_v2",
            "params": TinyEstimator().state_dict(),
            "contract": {
                "settings": {"normal_head": "phi_theta"},
                "normal_objective": {"head": "phi_theta"},
            },
        }, checkpoint_path)
        model = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
        model.args = SimpleNamespace(input_mode="linear_rgb")
        model.feature_version = "legacy_batch_v0"
        net = TinyEstimator()
        model.load_network(net, checkpoint_path)
        assert model.feature_version == "raw_calibrated_v2"
        assert model.normal_head == "phi_theta"

        legacy_path = f"{output_dir}/legacy.pth"
        torch.save({"params": TinyEstimator().state_dict()}, legacy_path)
        model.load_network(net, legacy_path)
        assert model.normal_head == "xyz"


def check_synthetic_path_rejects_fabric_contracts():
    with tempfile.TemporaryDirectory() as output_dir:
        raw_path = f"{output_dir}/raw.pth"
        torch.save({"feature_version": "raw_calibrated_v2", "params": {}}, raw_path)
        try:
            validate_synthetic_checkpoint(raw_path)
        except ValueError as error:
            assert "real.py" in str(error)
        else:
            raise AssertionError("RAW Fabric checkpoint entered test.py")

        phi_path = f"{output_dir}/phi.pth"
        torch.save({
            "params": {},
            "contract": {"settings": {"normal_head": "phi_theta"}},
        }, phi_path)
        try:
            validate_synthetic_checkpoint(phi_path)
        except ValueError as error:
            assert "real.py" in str(error)
        else:
            raise AssertionError("phi_theta checkpoint entered test.py")

        legacy_path = f"{output_dir}/legacy.pth"
        torch.save({"params": {}}, legacy_path)
        assert validate_synthetic_checkpoint(legacy_path) == ("legacy_batch_v0", "xyz")


def check_inference_manifest():
    with tempfile.TemporaryDirectory() as output_dir:
        checkpoint_path = f"{output_dir}/matsynth21.pth"
        torch.save({
            "params": {
                "intro_albedo.weight": torch.zeros(32, 18, 3, 3),
            }
        }, checkpoint_path)
        args = SimpleNamespace(
            save_root=output_dir,
            loadpath_network_g=checkpoint_path,
            image_size=256,
        )
        model = SimpleNamespace(near_distance=2.414, far_distance=10.0)
        manifest_path = write_inference_manifest(args, "matsynth21", model)
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        with open(checkpoint_path, "rb") as checkpoint_handle:
            digest = hashlib.sha256(checkpoint_handle.read()).hexdigest()
        assert manifest["estimator_family"] == "matsynth21"
        assert manifest["network"] == "TwoBranchNet"
        assert manifest["checkpoint_sha256"] == digest
        assert manifest["checkpoint_intro_albedo_input_channels"] == 18
        assert manifest["input"]["range_tolerance"] == 1e-5
        assert manifest["output"]["shape"] == "[B,10,256,256]"
        assert manifest["feature_geometry"] == {"near_distance": 2.414, "far_distance": 10.0}
        assert manifest["normal_head"] == "xyz"
        assert manifest["denoiser"]["used"] is False


def main():
    check_family_detection()
    check_matsynth_capture_shape_and_no_denoise()
    check_checkpoint_head_metadata_load()
    check_synthetic_path_rejects_fabric_contracts()
    check_inference_manifest()
    print("REAL_ESTIMATOR_SMOKE_OK")


if __name__ == "__main__":
    main()
