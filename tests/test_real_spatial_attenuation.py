"""FP32 physics and call-path checks; no checkpoints or denoiser required."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import model.nfplight_real_model as real
from utils import svBRDF


def geometry_model():
    model = real.NFPLightModel.__new__(real.NFPLightModel)
    model.args = SimpleNamespace(image_size=256)
    model.device = torch.device('cpu')
    model.init_distance()
    model.init_rendering()
    return model


def test_report_geometry_cancels_flat_lambertian_distance_residual():
    renderer = svBRDF()
    surface = renderer.surface(256, 1, dtype=torch.float32)
    observations, distances, cosines = [], [], []
    # Same point lights / grid as 20261014_s34_normal_s15_spatial_attenuation.
    for z in (2.414, 10.0):
        point = torch.tensor([[0., 0., z]], dtype=torch.float32)
        light, _, distance, _ = renderer.torch_generate(point, point, pos=surface)
        albedo = torch.linspace(.1, .5, 256).view(1, 1, 1, 256)
        observations.append(albedo * light[:, 2:3] / distance)
        distances.append(distance)
        cosines.append(light[:, 2:3])
    near, far = observations
    gain = near[:, :, 127:128, 127:128] / far[:, :, 127:128, 127:128]
    baseline = far * gain * cosines[0] / cosines[1]
    factor = real.NFPLightModel.spatialAttenuationFactor(*distances)
    corrected = baseline * factor
    assert factor.dtype == torch.float32
    assert factor[0, 0, 127, 127] == 1
    assert (baseline - near).abs().mean() > 1e-4
    assert (corrected - near).abs().max() < 2e-7


def test_original_geometry_and_appearance_slots_preserved():
    model = geometry_model()
    coefficient, angular = model.coefficientGeneration()
    assert torch.equal(model.coefficient, coefficient)
    assert torch.equal(model.time_co_map, angular)
    assert model.spatial_attenuation_factor[0, 0, 0, 0] < 1
    near = torch.full((1, 3, 256, 256), .2)
    far_gained = torch.full_like(near, .3)
    expected = model.relationmapExtraction(
        near, far_gained * model.spatial_attenuation_factor)
    actual = model.relationmapExtractionSpatialAttenuation(near, far_gained)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    with pytest.raises(TypeError, match='FP32'):
        model.relationmapExtractionSpatialAttenuation(near.half(), far_gained)


@pytest.mark.parametrize('bad', [0., -1., float('nan'), float('inf')])
def test_invalid_squared_distances_rejected(bad):
    near = torch.ones(1, 1, 2, 2)
    near[0, 0, 0, 0] = bad
    with pytest.raises(ValueError):
        real.NFPLightModel.spatialAttenuationFactor(near, torch.ones_like(near))


def test_fp64_distance_setup_rejected():
    with pytest.raises(TypeError, match='FP32'):
        real.NFPLightModel.spatialAttenuationFactor(
            torch.ones(1, 1, 2, 2, dtype=torch.float64), torch.ones(1, 1, 2, 2))


def test_active_test_calls_updated_relation_without_denoiser(monkeypatch):
    class Estimator(nn.Module):
        def forward(self, features):
            self.features = features
            return features[:, :10]

    monkeypatch.setattr(real, 'TwoBranchRealNet', Estimator)
    monkeypatch.setattr(real.NFPLightModel, 'load_network', lambda *args: None)
    model = real.NFPLightModel(SimpleNamespace(
        image_size=256, device='cpu', loadpath_network_g='unused'))
    assert not hasattr(model, 'net_denoise')
    model.feed_data({'inputs': torch.full((1, 6, 256, 256), .2), 'name': ['test']})
    calls = []
    updated = model.relationmapExtractionSpatialAttenuation

    def record(*args):
        calls.append(True)
        return updated(*args)

    model.relationmapExtractionSpatialAttenuation = record
    model.test()
    assert calls == [True]
    assert model.net_g.features.shape == (1, 33, 256, 256)
    assert model.net_g.features.dtype == torch.float32
    assert torch.isfinite(model.net_g.features).all()
    # The first 30 appearance/log-appearance channels retain the old layout.
    expected_rgb = torch.cat([model.inputs, model.inputs, model.inputs[:, 3:]], 1)
    torch.testing.assert_close(model.net_g.features[:, :15], expected_rgb * 2 - 1)


def test_rejects_non_native_size_and_multi_sample_gain():
    with pytest.raises(ValueError, match='256x256'):
        real.NFPLightModel(SimpleNamespace(image_size=16))
    model = geometry_model()
    with pytest.raises(ValueError, match='inputs require'):
        model.feed_data({'inputs': torch.full((2, 6, 256, 256), .2),
                         'name': ['a', 'b']})


def test_validation_saves_png_with_declared_dependencies(tmp_path):
    from PIL import Image
    model = geometry_model()
    model.args.save_root = str(tmp_path)
    model.test = lambda: setattr(model, 'pred_svbrdf', torch.zeros(1, 10, 256, 256))
    model.validation([{'inputs': torch.full((1, 6, 256, 256), .2), 'name': ['sample']}])
    with Image.open(tmp_path / 'sample.png') as saved:
        assert saved.size == (256 * 6, 256)
        assert saved.mode == 'RGB'
