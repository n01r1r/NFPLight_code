"""Deterministic checks for the approved gain9 input contract."""
from types import SimpleNamespace
import torch
from model.nfplight_model import NFPLightModel, CURRENT_RAW_FEATURE_VERSION
from model.nfplight_matsynth_capture_model import normalize_checkpoint_state_dict
from train_fabric import FEATURE_VERSION, RENDER_NEAR, RENDER_FAR
from utils import svBRDF


def model(version):
    m = NFPLightModel.__new__(NFPLightModel)
    m.feature_version = version
    m.device = torch.device("cpu")
    m.capture_precision = True
    m.renderer = svBRDF()
    m.init_distance()
    m.init_rendering()
    return m


def test_gain9_geometry_and_features_ignore_center_ratio():
    torch.set_num_threads(2)
    m = model(CURRENT_RAW_FEATURE_VERSION)
    assert FEATURE_VERSION == CURRENT_RAW_FEATURE_VERSION
    assert (m.near_distance, m.far_distance) == (RENDER_NEAR, RENDER_FAR) == (4., 12.)
    inputs = torch.full((1, 6, 256, 256), .02)
    inputs[:, :3] = .18
    for center in (0., 1.):
        inputs[:, :, 127:129, 127:129] = center
        features, relation, far = m.build_features(inputs)
        assert torch.equal(far[:, :, 0, 0], inputs[:, 3:, 0, 0] * 9.0)
        assert features.dtype == relation.dtype == far.dtype == torch.float32
        assert torch.isfinite(features).all()


def test_legacy_checkpoint_geometry_is_preserved():
    m = model("raw_calibrated_v2")
    assert (m.near_distance, m.far_distance) == (2.414, 10.)
    assert (m.far_distance / m.near_distance) ** 2 != 9.


def test_new_training_checkpoint_wrapper():
    weight = torch.ones(1)
    assert normalize_checkpoint_state_dict({"net_g": {"module.weight": weight}})["weight"] is weight


def test_capture_patch20_gain_uses_area_rgb_mean_per_sample():
    m = model("raw_calibrated_v2")
    x = torch.full((2, 6, 256, 256), .02)
    x[:, :3] = .08
    x[1, :3] = .12
    x[:, 3:, 127, 127] = .5
    n = x[:, :3, 118:138, 118:138].mean((1, 2, 3))
    f = x[:, 3:, 118:138, 118:138].mean((1, 2, 3))
    features, relation, far = m.build_features(x, gain_mode="author_patch20")
    expected = (x[:, 3:] * (n / f)[:, None, None, None]).clamp(0, 1)
    assert torch.equal(far, expected)
    assert torch.allclose((features[:, 6:9] + 1) / 2, expected, atol=1e-7)
    assert torch.allclose((f * (n / f)), n, atol=1e-7)
    assert torch.isfinite(relation).all()
    _, _, default_far = m.build_features(x)
    assert torch.equal(default_far, (x[:, 3:] * (10 / 2.414)**2).clamp(0, 1))
    x[:, 3:] = 0
    features, relation, far = m.build_features(x, gain_mode="author_patch20")
    assert torch.isfinite(features).all() and torch.isfinite(relation).all()
    assert not far.any()


def test_capture_patch20_rejects_invalid_support_and_mode():
    import pytest
    m = model("raw_calibrated_v2")
    x = torch.full((1, 6, 256, 256), .02)
    support = torch.ones((1, 1, 256, 256), dtype=torch.bool)
    support[:, :, 118, 118] = False
    with pytest.raises(ValueError, match="full valid support"):
        m.build_features(x, support, gain_mode="author_patch20")
    with pytest.raises(ValueError, match="unknown capture gain"):
        m.build_features(x, gain_mode="invalid")


def test_measured_gain_entrypoint_is_separate_from_training_gain9():
    m = model(CURRENT_RAW_FEATURE_VERSION)
    x = torch.full((2, 6, 256, 256), .02)
    x[0, :3] = .08
    x[1, :3] = .12
    original = x.clone()
    _, _, training_far = m.build_features(x)
    features, _, measured_far = m.build_capture_features(x)
    assert torch.equal(x, original)
    assert torch.allclose(training_far, x[:, 3:] * 9)
    assert torch.allclose(measured_far[0], x[0, 3:] * 4)
    assert torch.allclose(measured_far[1], x[1, 3:] * 6)
    assert torch.allclose((features[:, 6:9]+1)/2, measured_far, atol=1e-7)
    assert not torch.equal(training_far, measured_far)


def test_relation_averages_absolute_rgb_differences_equally():
    m = NFPLightModel.__new__(NFPLightModel)
    m.time_co_map = torch.ones(1, 1, 2, 2)
    m.spatial_attenuation_factor = torch.ones_like(m.time_co_map)
    m.coefficient = torch.ones(1, 1, 2, 2)
    m.indenty = torch.ones(1, 1, 2, 2)
    near = torch.full((1, 3, 2, 2), 0.5)
    delta = torch.tensor([[[[.3, 0.], [0., .3]],
                           [[0., .3], [0., -.3]],
                           [[0., 0.], [.3, 0.]]]])
    far = near + delta
    actual_delta = far.clamp(0, 1) - near
    expected_q = actual_delta.abs().sum(1, keepdim=True) / 3
    relation, log_relation = m.relationmapExtraction(near, far)
    assert torch.allclose(relation, 1 - m.ClipToOne(expected_q), atol=1e-6)
    assert torch.allclose(log_relation, 1 - m.ClipToOne(expected_q.log()), atol=1e-6)
    assert torch.allclose(relation[0, 0].flatten()[:3], torch.ones(3), atol=1e-6)
    weights = torch.tensor([.299, .587, .114]).reshape(1, 3, 1, 1)
    weighted_q = (actual_delta.abs() * weights).sum(1, keepdim=True)
    assert not torch.allclose(relation, 1 - m.ClipToOne(weighted_q))
    assert not torch.allclose(expected_q, actual_delta.mean(1, keepdim=True).abs())
