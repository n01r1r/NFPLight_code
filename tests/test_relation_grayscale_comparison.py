import pytest
import torch
from model.nfplight_model import NFPLightModel
from experiments.relation_grayscale_comparison import relation_numerator, weighted_grayscale_relation


def test_grayscale_precedes_abs_and_uses_declared_channel_weights():
    delta = torch.tensor([[[[.3, .3, 0., 0.]], [[-.3, 0., .3, 0.]], [[0., 0., 0., .3]]]])
    rgb = relation_numerator(delta, 'rgb_mean')
    gray = relation_numerator(delta, 'weighted_grayscale')
    assert torch.allclose(rgb, torch.tensor([[[[.2, .1, .1, .1]]]]))
    assert torch.allclose(gray, torch.tensor([[[[.0864, .0897, .1761, .0342]]]]))
    assert gray[0, 0, 0, 0] != pytest.approx(.2658)  # weighted abs(RGB) is different
    with pytest.raises(ValueError):
        relation_numerator(delta, 'invalid')
    with pytest.raises(ValueError):
        relation_numerator(delta * float('nan'), 'rgb_mean')


def test_override_changes_only_relation_features_and_restores_on_error():
    m = NFPLightModel.__new__(NFPLightModel)
    m.feature_version = 'raw_calibrated_gain9_v3'
    m.far_distance, m.near_distance = 12., 4.
    m.time_co_map = torch.ones(1, 1, 256, 256)
    m.spatial_attenuation_factor = torch.ones_like(m.time_co_map)
    m.coefficient = torch.ones_like(m.time_co_map)
    m.indenty = torch.ones_like(m.time_co_map)
    x = torch.full((1, 6, 256, 256), .2)
    x[:, 3, :30] = .4
    x[:, 4, 30:60] = .4
    original = m.build_capture_features(x)[0]
    with pytest.raises(RuntimeError):
        with weighted_grayscale_relation(m):
            altered = m.build_capture_features(x)[0]
            assert torch.equal(original[:, :18], altered[:, :18])
            assert torch.equal(original[:, 20:], altered[:, 20:])
            assert not torch.equal(original[:, 18:20], altered[:, 18:20])
            raise RuntimeError('fixture')
    assert 'relationmapExtraction' not in m.__dict__
    assert torch.equal(original, m.build_capture_features(x)[0])
