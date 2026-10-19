"""CPU tests for the maintained 21-channel capture model and strict checkpoint loading."""
import tempfile
from types import SimpleNamespace
import torch
from model.nfplight_matsynth_capture_model import MatSynthCaptureModel, normalize_checkpoint_state_dict

class DummyEstimator(torch.nn.Module):
    def forward(self, inputs):
        assert inputs.shape[1] == 21, inputs.shape
        return torch.zeros(
            inputs.shape[0], 10, inputs.shape[2], inputs.shape[3],
            device=inputs.device,
        )

def test_capture_shape_and_input_validation():
    assert "test" in MatSynthCaptureModel.__dict__
    assert "init_distance" not in MatSynthCaptureModel.__dict__
    model = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
    model.device = torch.device("cpu")
    model.args = SimpleNamespace(image_size=256)
    model.coefficient = torch.ones(1, 1, 256, 256)
    model.time_co_map = torch.ones(1, 1, 256, 256)
    model.spatial_attenuation_factor = torch.ones_like(model.time_co_map)
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

def test_checkpoint_head_metadata_load():
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

def test_checkpoint_prefix_normalization():
    normalized = normalize_checkpoint_state_dict({"state_dict": {"module._orig_mod.weight": torch.ones(1)}})
    assert set(normalized) == {"weight"}
    try:
        normalize_checkpoint_state_dict({"params": {"weight": torch.ones(1), "module.weight": torch.ones(1)}})
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("ambiguous checkpoint prefixes were accepted")


def test_gain9_exports_and_suite_checkpoints_keep_unit_normal_and_raw_specular(tmp_path):
    class TinyEstimator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.ones(1))
        def forward(self,inputs):
            value=inputs.new_full((inputs.shape[0],10,256,256),0.3)
            value[:,:3]=2
            return value
    payloads=[
        {"feature_version":"raw_calibrated_gain9_v3","normal_decoder":"unit XYZ"},
        {"package":"full_dataset_suite","identity":{"scientific_settings":{"rendering":{"feature_version":"raw_calibrated_gain9_v3"}}}},
        {"feature_version":"raw_calibrated_gain9_v3","normal_decoder":"unit normalization before loss/inference","contract":{"f0_input_and_both_render_branches":0.04}},
    ]
    for i,payload in enumerate(payloads):
        model=MatSynthCaptureModel.__new__(MatSynthCaptureModel)
        model.args=SimpleNamespace(image_size=256,input_mode="linear_rgb")
        model.device=torch.device("cpu")
        model.compute_dtype=torch.float32
        model.capture_precision=True
        model.net_g=TinyEstimator()
        path=tmp_path / f"export_{i}.pth"
        torch.save(dict(payload,net_g=model.net_g.state_dict()),path)
        model.load_network(model.net_g,path)
        assert model.feature_version=="raw_calibrated_gain9_v3" and model.unit_normal
        model.build_features=lambda inputs,**kwargs:(inputs.new_zeros((1,21,256,256)),None,None)
        model.feed_data({"inputs":torch.full((1,6,256,256),0.1)})
        model.test()
        torch.testing.assert_close(model.pred_svbrdf[:,:3].norm(dim=1),torch.ones((1,256,256)))
        assert torch.equal(model.pred_svbrdf[:,7:10],torch.full((1,3,256,256),0.3))


def test_capture_rejects_training_gain_mode():
    import pytest
    model = MatSynthCaptureModel.__new__(MatSynthCaptureModel)
    model.args = SimpleNamespace(capture_gain_mode='checkpoint')
    model.net_g = DummyEstimator()
    with pytest.raises(ValueError, match='training gain is separate'):
        model.test()
