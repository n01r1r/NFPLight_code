"""CPU contract tests for the authors' original real-capture inference path."""

from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import model.author_real_capture_adapter as adapter_module
from model.author_real_capture_adapter import (
    AuthorRealCaptureAdapter,
    GAIN_PATCH,
    assemble_legacy33_features,
    legacy_geometry,
    log_normalization,
)
from run_fabric_capture import (
    AUTHORIZED_CHECKPOINT_SHA256,
    _hwc,
    _validate_author_checkpoint_identity,
)


def test_original_checkpoint_allowlist_is_role_and_sha_bound():
    _validate_author_checkpoint_identity(
        "net_g_real.pth", AUTHORIZED_CHECKPOINT_SHA256["net_g_real.pth"], "estimator"
    )
    with pytest.raises(ValueError, match="allowlisted original checkpoint"):
        _validate_author_checkpoint_identity(
            "best_render.pth", "6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd",
            "estimator",
        )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _validate_author_checkpoint_identity("net_g_real.pth", "0" * 64, "estimator")
    with pytest.raises(ValueError, match="allowlisted original checkpoint"):
        _validate_author_checkpoint_identity(
            "net_g_real.pth", AUTHORIZED_CHECKPOINT_SHA256["net_g_real.pth"], "denoiser"
        )


def test_original_loader_uses_fp32_and_strict_state_matching():
    model = nn.Conv2d(2, 3, kernel_size=1)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "synthetic-original.pth"
        torch.save({"params": model.state_dict()}, path)
        target = nn.Conv2d(2, 3, kernel_size=1)
        AuthorRealCaptureAdapter._load_strict_fp32(target, path)
        assert all(parameter.dtype == torch.float32 for parameter in target.parameters())
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["params"].pop("bias")
        torch.save(payload, path)
        with pytest.raises(RuntimeError):
            AuthorRealCaptureAdapter._load_strict_fp32(target, path)


def test_adapter_never_reads_or_calls_a_denoiser(tmp_path, monkeypatch):
    class FakeEstimator(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.forward_calls = 0

        def forward(self, features):
            self.forward_calls += 1
            return self.value.expand(1, 10, 256, 256)

    estimator_path = tmp_path / "net_g_real.pth"
    torch.save({"params": FakeEstimator().state_dict()}, estimator_path)
    monkeypatch.setattr(adapter_module, "AuthorTwoBranchRealNet", FakeEstimator)
    original_torch_load = torch.load
    loaded_names = []

    def guarded_torch_load(path, *args, **kwargs):
        name = Path(path).name
        if "denois" in name.lower():
            raise AssertionError("the denoiser checkpoint must never be read")
        loaded_names.append(name)
        return original_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(adapter_module.torch, "load", guarded_torch_load)
    adapter = AuthorRealCaptureAdapter(estimator_path, device="cpu")
    prediction, trace = adapter.infer(torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32))

    assert loaded_names == ["net_g_real.pth"]
    assert not hasattr(adapter, "net_denoise")
    assert adapter.net_g.forward_calls == 1
    assert prediction.shape == (1, 10, 256, 256)
    torch.testing.assert_close(trace["original_copy_pair_6"], trace["original_rgb_6"], rtol=0, atol=0)
    torch.testing.assert_close(trace["features_legacy33"], adapter.last_estimator_input, rtol=0, atol=0)
    assert not any("denois" in key.lower() for key in trace)
    assert adapter.geometry_setup_dtype == "numpy.float64"
    assert adapter.geometry_buffer_dtype == "torch.float32"


def test_original_log_operator_and_geometry_are_fp32_native_256():
    image = torch.tensor([0.0, 1.0], dtype=torch.float32).reshape(1, 1, 1, 2)
    logged = log_normalization(image)
    assert logged.dtype == torch.float32
    assert logged[0, 0, 0, 0].item() == pytest.approx(0.0, abs=1e-7)
    assert logged[0, 0, 0, 1].item() == pytest.approx(1.0, abs=1e-6)
    coefficient, time_map, maximum = legacy_geometry(256, "cpu")
    assert coefficient.shape == (1, 1, 256, 256)
    assert time_map.shape == coefficient.shape
    assert coefficient.dtype == time_map.dtype == maximum.dtype == torch.float32
    assert 0.0 < coefficient[0, 0, 127, 127].item() < 1e-4
    assert time_map[0, 0, 127, 127].item() == pytest.approx(1.0, abs=1e-6)
    assert coefficient.amax().item() == pytest.approx(1.0, abs=1e-7)


def test_geometry_buffers_are_bit_exact_to_original_numpy_setup():
    image_size = 256
    y, x = np.ogrid[:image_size, :image_size]
    center_y, center_x = (image_size - 1) / 2, (image_size - 1) / 2
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / (image_size / 2.0)
    angle_near = np.arctan(distance / 4.0)
    angle_far = np.arctan(distance / 12.0)
    coefficient = (np.cos(angle_far) - np.cos(angle_near)) / np.cos(angle_near)
    coefficient_max = np.max(coefficient)
    coefficient = coefficient / coefficient_max
    time_coefficient = np.cos(angle_near) / np.cos(angle_far)
    expected_coefficient = torch.Tensor(coefficient, device="cpu").unsqueeze(0).unsqueeze(0)
    expected_time = torch.Tensor(time_coefficient, device="cpu").unsqueeze(0).unsqueeze(0)
    expected_max = torch.as_tensor(coefficient_max, dtype=torch.float32)

    actual_coefficient, actual_time, actual_max = legacy_geometry(image_size, "cpu")
    for actual, expected in (
        (actual_coefficient, expected_coefficient),
        (actual_time, expected_time),
        (actual_max, expected_max),
    ):
        assert actual.dtype == torch.float32
        assert np.array_equal(actual.numpy().view(np.uint32), expected.numpy().view(np.uint32))


def test_hwc_export_keeps_all_33_feature_channels_last():
    source = torch.arange(33 * 2 * 4, dtype=torch.float32).reshape(1, 33, 2, 4)
    exported = _hwc(source)
    assert exported.shape == (2, 4, 33)
    for y, x, channel in ((0, 0, 0), (1, 3, 32), (0, 2, 17)):
        assert exported[y, x, channel] == source[0, channel, y, x].item()
    with pytest.raises(ValueError, match="batch size 1"):
        _hwc(source.repeat(2, 1, 1, 1))


def test_legacy33_channels_center_gain_relation_and_saturation_mask():
    original = torch.zeros((1, 6, 256, 256), dtype=torch.float32)
    for channel in range(6):
        original[:, channel] = 0.01 * (channel + 1)
    original[0, 3, 0, 0] = 0.96

    original_copy = torch.zeros_like(original)
    original_copy[:, 0] = 0.3
    original_copy[:, 1] = 0.4
    original_copy[:, 2] = 0.5
    original_copy[:, 3] = 0.1
    original_copy[:, 4] = 0.2
    original_copy[:, 5] = 0.3
    coefficient, time_map, maximum = legacy_geometry(256, "cpu")
    features, trace = assemble_legacy33_features(
        original, original_copy, coefficient, time_map, maximum
    )

    assert features.shape == (1, 33, 256, 256)
    assert features.dtype == torch.float32
    assert trace["far_gain_scalar"].item() == pytest.approx(2.0)
    assert trace["original_copy_far_gained_clipped"][0, 0, 128, 128].item() == pytest.approx(0.2)
    assert trace["features_unscaled_33"][0, 0, 128, 128].item() == pytest.approx(0.01)
    assert trace["features_unscaled_33"][0, 6, 128, 128].item() == pytest.approx(0.3)
    assert trace["features_unscaled_33"][0, 12, 128, 128].item() == pytest.approx(0.2)
    assert trace["features_unscaled_33"][0, 32, 128, 128].item() == 1.0
    assert trace["features_unscaled_33"][0, 32, 0, 0].item() == 0.0
    torch.testing.assert_close(features, trace["features_unscaled_33"] * 2 - 1, rtol=0, atol=0)
    assert GAIN_PATCH == (slice(118, 138), slice(118, 138))
    assert np.isfinite(features.cpu().numpy()).all()


def test_legacy33_identity_fallback_duplicates_original_rgb_slots():
    original = torch.full((1, 6, 256, 256), 0.2, dtype=torch.float32)
    original[:, 3:] = 0.1
    coefficient, time_map, maximum = legacy_geometry(256, "cpu")
    features, trace = assemble_legacy33_features(
        original, original.clamp(0, 1), coefficient, time_map, maximum
    )
    torch.testing.assert_close(trace["original_copy_pair_6"], original, rtol=0, atol=0)
    torch.testing.assert_close(features[:, 0:6], features[:, 6:12], rtol=0, atol=0)
    assert features.shape == (1, 33, 256, 256)
