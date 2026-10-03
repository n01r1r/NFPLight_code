"""Algebraic invariants for printed NFPLight equations, independent of weights."""

import pytest
import torch

from model.paper_equations import paper_relation
from model.author_real_capture_adapter import assemble_legacy33_features, legacy_geometry


def test_equation5_diffuse_cancellation_and_specular_identity():
    seed = torch.full((1, 6, 256, 256), 0.1, dtype=torch.float32)
    geometry = paper_relation(seed)
    cn, cf = geometry["paper_cos_near"], geometry["paper_cos_far"]
    diffuse = torch.full_like(cn, 0.08)
    sn, sf = torch.full_like(cn, 0.03), torch.full_like(cn, 0.01)
    # Eq. (4)'s central equal-BRDF premise is part of the synthetic setup.
    sf[:, :, 128, 128] = sn[:, :, 128, 128]
    near = 2.0 * (diffuse + sn) * cn
    far = (diffuse + sf) * cf
    # Force the central ray to the stipulated intensity ratio exactly.
    far[:, :, 128, 128] = near[:, :, 128, 128] / 2.0
    original = torch.cat([near.expand(-1, 3, -1, -1), far.expand(-1, 3, -1, -1)], 1)
    result = paper_relation(original)
    expected = 2.0 * (sn - sf).abs() / (cf - cn).abs().clamp_min(1e-7)
    # Outer noncentral points avoid cancellation in the independent RHS.
    outer = geometry["paper_denominator"] > 0.01
    # The independently subtracted FP32 cosines lose several low bits.
    torch.testing.assert_close(result["paper_relation_raw"][outer], expected[outer], rtol=3e-5, atol=1e-5)
    diffuse_only = torch.cat([(diffuse * cn).expand(-1, 3, -1, -1),
                              (diffuse * cf).expand(-1, 3, -1, -1)], 1)
    # Set central intensities equal to fix K_L=1 without a sampled-ray bias.
    diffuse_only[:, 3:, 128, 128] = diffuse_only[:, :3, 128, 128]
    cancelled = paper_relation(diffuse_only)
    assert cancelled["paper_numerator"].max() < 2e-7


def test_rgb_mean_precedes_absolute_value_and_scaled_far_is_not_clipped():
    x = torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32)
    x[:, :3, 128, 128] = 0.8
    x[:, 3:, 128, 128] = 0.2
    x[:, :3, 0, 0] = torch.tensor([0.2, 0.4, 0.6])
    x[:, 3:, 0, 0] = torch.tensor([0.3, 0.4, 0.5])
    result = paper_relation(x)
    assert result["paper_gain"].item() == pytest.approx(4.0)
    assert result["paper_far_scaled_intensity"][0, 0, 0, 0] > 1
    expected = abs(0.4 - 4.0 * result["paper_time_map"][0, 0, 0, 0].item() * 0.4)
    assert result["paper_numerator"][0, 0, 0, 0].item() == pytest.approx(expected)
    # Opposing chromatic changes with equal RGB means cancel in printed math.
    x[:, :3, 128, 128] = 0.2
    x[:, 3:, 128, 128] = 0.2
    result = paper_relation(x)
    channelwise = (x[:, :3] - result["paper_time_map"] * x[:, 3:]).abs().mean(1, keepdim=True)
    assert channelwise[0, 0, 0, 0] > result["paper_numerator"][0, 0, 0, 0]


def test_raw_denominator_has_no_floor_and_checkpoint_adapter_is_explicit():
    x = torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32)
    coefficient, time_map, maximum = legacy_geometry(256, "cpu")
    features, trace = assemble_legacy33_features(x, x, coefficient, time_map, maximum,
                                                equation_mode="paper_equations")
    assert 0 < trace["denominator"].min() < 1e-5
    torch.testing.assert_close(trace["relation_raw"], trace["paper_relation_raw"], rtol=0, atol=0)
    assert features.shape == (1, 33, 256, 256)
    assert features.dtype == torch.float32 and torch.isfinite(features).all()
    assert features.min() >= -1 and features.max() <= 1
    assert trace["signed_diff"].shape == (1, 1, 256, 256)
    torch.testing.assert_close(trace["signed_diff"], trace["paper_signed_difference"], rtol=0, atol=0)
    torch.testing.assert_close(trace["abs_diff"], trace["signed_diff"].abs(), rtol=0, atol=0)
    torch.testing.assert_close(trace["abs_diff"], trace["mean_abs_diff_rgb"], rtol=0, atol=0)
    torch.testing.assert_close(trace["relation_raw"], trace["abs_diff"] / trace["denominator"], rtol=0, atol=0)


def test_undefined_gain_and_invalid_precision_are_rejected():
    x = torch.zeros((1, 6, 256, 256), dtype=torch.float32)
    with pytest.raises(ValueError, match="far center"):
        paper_relation(x)
    with pytest.raises(ValueError, match="FP32"):
        paper_relation(x.double())
    x.fill_(0.2)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        paper_relation(x)


def test_independent_check_detects_a_hidden_denominator_floor():
    import numpy as np
    from capture_processing.paper_comparison import verify_paper_equations

    x = torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32)
    result = paper_relation(x)
    arrays = {key: value.numpy()[0].transpose(1, 2, 0) for key, value in result.items()}
    inputs = x.numpy()[0].transpose(1, 2, 0)
    assert verify_paper_equations(inputs, arrays)["passed"]
    arrays["paper_denominator"] = np.maximum(arrays["paper_denominator"], np.float32(1e-5))
    with pytest.raises(AssertionError, match="paper_denominator"):
        verify_paper_equations(inputs, arrays)


def test_independent_check_rejects_reversed_sign_even_when_absolute_value_matches():
    import numpy as np
    from capture_processing.paper_comparison import verify_paper_equations

    x = torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32)
    x[:, :3, 0, 0] = 0.5
    result = paper_relation(x)
    arrays = {key: value.numpy()[0].transpose(1, 2, 0) for key, value in result.items()}
    inputs = x.numpy()[0].transpose(1, 2, 0)
    assert verify_paper_equations(inputs, arrays)["passed"]
    arrays["paper_signed_difference"] = -arrays["paper_signed_difference"]
    with pytest.raises(AssertionError, match="paper_signed_difference"):
        verify_paper_equations(inputs, arrays)
