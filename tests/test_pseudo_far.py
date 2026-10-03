"""Deterministic CPU checks of copied-near inverse-square experiment meaning."""
import numpy as np
import pytest
import torch

from capture_processing.pseudo_far import pseudo_far_input
from model.paper_equations import paper_relation


def test_pseudo_far_preserves_near_and_recalculates_gain():
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0.05, 0.8, (256, 256, 6)).astype(np.float32)
    saved = inputs.copy()
    pseudo = pseudo_far_input(inputs)
    assert pseudo.dtype == np.float32 and pseudo.shape == inputs.shape
    assert np.array_equal(saved, inputs)
    assert np.array_equal(pseudo[..., :3], inputs[..., :3])
    assert np.array_equal(pseudo[..., 3:], inputs[..., :3] * np.float32(1 / 9))
    assert pseudo.min() >= 0 and pseudo.max() <= 1
    tensor = torch.from_numpy(pseudo.transpose(2, 0, 1)[None].copy())
    result = paper_relation(tensor)
    assert result["paper_gain"].item() == pytest.approx(9, rel=2e-7)
    expected = result["paper_near_intensity"] * (1 - result["paper_time_map"])
    torch.testing.assert_close(result["paper_signed_difference"], expected, rtol=0, atol=2e-7)
    torch.testing.assert_close(result["paper_numerator"], expected.abs(), rtol=0, atol=2e-7)
    expected_raw = result["paper_near_intensity"] / (result["paper_cos_near"] * result["paper_cos_far"])
    bound = 4e-7 / result["paper_denominator"] + 5e-5 * expected_raw.abs()
    assert torch.all((result["paper_relation_raw"] - expected_raw).abs() <= bound)
    # Global attenuation is removed by the independently recomputed gain.
    unattenuated = torch.cat([tensor[:, :3], tensor[:, :3]], dim=1)
    second = paper_relation(unattenuated)
    torch.testing.assert_close(result["paper_numerator"], second["paper_numerator"], rtol=0, atol=2e-7)
    assert second["paper_gain"].item() == 1


def test_pseudo_far_rejects_invalid_input():
    inputs = np.full((256, 256, 6), 0.2, np.float32)
    with pytest.raises(ValueError, match="FP32"):
        pseudo_far_input(inputs.astype(np.float64))
    inputs[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pseudo_far_input(inputs)
    inputs.fill(0)
    with pytest.raises(ValueError, match="positive near center"):
        pseudo_far_input(inputs)


def test_pseudo_paper_stage_ui_preserves_scalar_sign_and_component_navigation():
    from capture_processing.pseudo_far import _paper_formulas
    from capture_processing.report import _make_html, _stats
    formulas = _paper_formulas()
    keys = [row[-1] for row in formulas]
    signed = np.array([[[-0.2], [0.3]]], dtype=np.float32)
    manifest = {"model_contract": {"estimator_family": "author_legacy33", "feature_channels": 33},
                "conditions": {"both": {"arrays": {"paper_signed_difference": _stats(signed)}}}}
    page = _make_html(manifest, keys, {key: [-0.3, 0.3] for key in keys},
                      {key: {"range": [-0.3, 0.3], "rgb": False} for key in keys}, {}, [],
                      formula_records=formulas, notation_note="paper scalar order")
    assert keys.index("paper_signed_difference") < keys.index("paper_numerator") < keys.index("paper_denominator")
    assert 'value="paper_signed_difference"' in page
    assert 'value="signed_diff"' not in page and 'value="near_minus_scaled_far"' not in page
    assert 'I_N − K_L K_θ I_F' in page and 'paper scalar order' in page
    assert '中央 20' not in page and '중앙 20×20' not in page
    for key in ("prediction_normal_rgb", "prediction_diffuse_rgb", "prediction_roughness", "prediction_specular_rgb"):
        assert f'value="{key}"' in page
    assert manifest["conditions"]["both"]["arrays"]["paper_signed_difference"]["min"] == float(signed.min())
