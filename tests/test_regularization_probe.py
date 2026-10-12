from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from data.fabric import canonical_target, pack_svbrdf
from experiments.regularization import config as config_module
from experiments.regularization import run as runner
from experiments.regularization.manage import main as manage_main
from experiments.regularization import report as report_module
from experiments.regularization.report import _effects, build_report
from model.nfplight_model import NFPLightModel
from network.nfplight_net import NAFBlock, TwoBranchNet


@pytest.fixture
def historical_reference(tmp_path, monkeypatch):
    """Use a small frozen result instead of checkpoints deleted by approved cleanup."""
    result=tmp_path / "historical_result.json"
    result.write_text('{"complete":true}',encoding="utf8")
    record=config_module.file_identity(result)
    monkeypatch.setattr(config_module,"_historical_identity",lambda *a,**k:{
        "directory":str(tmp_path),"identity_manifest":record,"files":[record]})


def _settings_for_model():
    config, _ = config_module.load_config()
    return config["scientific_settings"]


def _packed_target(size=8):
    yy, xx = np.mgrid[:size, :size]
    nx = np.full((size, size), 0.2, dtype=np.float32)
    ny = np.full((size, size), -0.3, dtype=np.float32)
    nz = np.sqrt(1 - nx * nx - ny * ny)
    nz[1:3, 2:5] *= -1
    normal = np.stack((nx, ny, nz), axis=-1)
    diffuse = np.stack((
        0.2 + 0.4 * xx / max(size - 1, 1),
        0.25 + 0.3 * yy / max(size - 1, 1),
        np.full((size, size), 0.35, dtype=np.float32),
    ), axis=-1).astype(np.float32)
    rough = (0.15 + 0.4 * xx / max(size - 1, 1))[..., None].astype(np.float32)
    specular = np.full_like(diffuse, 0.04)
    return pack_svbrdf(normal, diffuse, rough, specular)


def test_local_config_freezes_2x2_contract_and_rejects_mutation():
    config, _ = config_module.load_config()
    assert [row["id"] for row in config["design"]["conditions"]] == [
        "baseline", "d4_only", "dropout_only", "d4_dropout"
    ]
    assert config["scientific_settings"]["baseline_parameters"] == 7609834
    assert config["training"]["seeds"] == [7, 19, 31]

    changed = copy.deepcopy(config)
    changed["design"]["dropout_probability"] = 0.2
    with pytest.raises(ValueError, match="0.1"):
        config_module.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["training"]["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown unexpected"):
        config_module.validate_config(changed)
    changed = copy.deepcopy(config)
    changed["scientific_settings"]["backbone"]["encoder_blocks"] = [2, 2, 3]
    with pytest.raises(ValueError, match="scientific settings"):
        config_module.validate_config(changed)


def test_d4_transforms_raw_normal_vectors_and_backface_mask_together():
    raw = _packed_target()
    raw_batch = raw.unsqueeze(0)
    _, valid = canonical_target(raw_batch)
    original_normal = raw[:3].permute(1, 2, 0).numpy()
    for transform in range(8):
        rotated = np.rot90(original_normal, transform % 4, axes=(0, 1)).copy()
        nx, ny = rotated[..., 0].copy(), rotated[..., 1].copy()
        if transform % 4 == 1:
            rotated[..., 0], rotated[..., 1] = -ny, nx
        elif transform % 4 == 2:
            rotated[..., 0], rotated[..., 1] = -nx, -ny
        elif transform % 4 == 3:
            rotated[..., 0], rotated[..., 1] = ny, -nx
        expected_mask = np.rot90(valid[0, 0].numpy(), transform % 4)
        if transform >= 4:
            rotated = rotated[:, ::-1].copy()
            rotated[..., 0] *= -1
            expected_mask = expected_mask[:, ::-1]
        actual = runner.transform_target_d4(raw, transform)
        _, actual_valid = canonical_target(actual[None])
        np.testing.assert_allclose(actual[:3].permute(1, 2, 0).numpy(), rotated, atol=1e-7, rtol=0)
        assert torch.equal(actual_valid[0, 0], torch.from_numpy(np.ascontiguousarray(expected_mask)))
        assert torch.equal(actual_valid.sum(), valid.sum())


class _FeatureOnlyModel(NFPLightModel):
    def init_network(self):
        self.net_g = None


def test_d4_rebuilds_renderer_features_after_transform_on_cpu():
    model = _FeatureOnlyModel(SimpleNamespace(
        device="cpu", precision="float32",
        feature_version="raw_calibrated_gain9_v3", capture_precision=True,
    ))
    raw = _packed_target(256)[None]
    before_safe, before_valid, before_features = runner.prepare_features(model, raw, 0.04)
    transformed = runner.augment_targets([raw[0]], [1])
    after_safe, after_valid, after_features = runner.prepare_features(model, transformed, 0.04)
    assert torch.equal(after_valid, torch.rot90(before_valid, 1, dims=(-2, -1)))
    torch.testing.assert_close(after_features, torch.rot90(before_features, 1, dims=(-2, -1)), atol=2e-5, rtol=2e-5)
    assert torch.equal(after_safe[:, 6:7].amin(), before_safe[:, 6:7].amin())


def test_dropout_factory_preserves_baseline_init_params_and_eval_function():
    scientific = _settings_for_model()
    backbone = scientific["backbone"]
    torch.set_num_threads(1)
    torch.manual_seed(441)
    baseline = TwoBranchNet(
        width=backbone["width"], middle_blk_num=backbone["middle_blocks"],
        enc_blk_nums=backbone["encoder_blocks"], dec_blk_nums=backbone["decoder_blocks"],
    )
    torch.manual_seed(441)
    p0 = runner.create_network(0.0, backbone)
    torch.manual_seed(441)
    dropout = runner.create_network(0.1, backbone)
    assert runner.parameter_count(baseline) == 7609834
    assert runner.parameter_count(p0) == runner.parameter_count(dropout) == 7609834
    assert list(baseline.state_dict()) == list(dropout.state_dict())
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, dropout.state_dict()[key])
    assert all(isinstance(block.dropout1, nn.Identity) and isinstance(block.dropout2, nn.Identity)
               for block in p0.modules() if isinstance(block, NAFBlock))
    blocks = [block for block in dropout.modules() if isinstance(block, NAFBlock)]
    assert len(blocks) == 36
    assert sum(isinstance(layer, nn.Dropout) for block in blocks for layer in (block.dropout1, block.dropout2)) == 72
    assert all(layer.p == 0.1 for block in blocks for layer in (block.dropout1, block.dropout2))

    sample = torch.randn(1, 21, 32, 32)
    baseline.eval()
    p0.eval()
    dropout.eval()
    expected = baseline(sample)
    assert torch.equal(expected, p0(sample))
    assert torch.equal(expected, dropout(sample))

    with torch.no_grad():
        for block in blocks:
            block.beta.fill_(0.15)
            block.gamma.fill_(0.15)
    dropout.train()
    torch.manual_seed(17)
    initial_rng = torch.random.get_rng_state()
    first = dropout(sample)
    after_first = torch.random.get_rng_state()
    torch.random.set_rng_state(initial_rng)
    restored = dropout(sample)
    assert torch.equal(first, restored)
    torch.random.set_rng_state(after_first)
    second = dropout(sample)
    assert not torch.equal(first, second)
    dropout.eval()
    assert torch.equal(dropout(sample), dropout(sample))


def test_sample_schedule_does_not_depend_on_d4_random_stream():
    config = _settings_for_model()
    settings = {
        "dataset": config["dataset"],
        "sampling": config["sampling"],
        "training": {"map_updates": 3, "render_updates": 2},
    }
    first = runner.sample_order(19, settings)
    d4_rng = random.Random(19 + 20000)
    [d4_rng.randrange(8) for _ in range(4 * len(first))]
    second = runner.sample_order(19, settings)
    assert first == second
    assert all(len(row) == 4 and len(set(row)) == 4 for row in first)


def test_prepare_from_foreign_cwd_is_cpu_only_and_idempotent(tmp_path, monkeypatch, historical_reference):
    imported_before={name for name in ("train_fabric","benchmark_fabric_leather") if name in sys.modules}
    target = tmp_path / "prepared"
    monkeypatch.chdir(tmp_path)
    before = torch.cuda.is_initialized()
    result = manage_main(["prepare", "--out", str(target)])
    assert result == target.resolve()
    execution = config_module.load_execution_config(target)
    assert execution["experiment_settings"]["scientific_settings"]["baseline_parameters"] == 7609834
    assert json.loads((target / "status.json").read_text(encoding="utf-8"))["training_started"] is False
    first_mtime = (target / "execution_config.json").stat().st_mtime_ns
    manage_main(["prepare", "--out", str(target)])
    assert (target / "execution_config.json").stat().st_mtime_ns == first_mtime
    assert torch.cuda.is_initialized() is before
    assert {name for name in ("train_fabric","benchmark_fabric_leather") if name in sys.modules} == imported_before

class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 10, 1)

    def forward(self, value):
        return self.conv(value)


class _ToyModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.net_g = _TinyNet()


def _toy_settings(map_updates=2, render_updates=1, patience=100):
    return {
        "dataset": {
            "categories": ["Fabric", "Leather"],
            "materials_per_category": 4,
            "train_crops": [0, 8],
        },
        "sampling": {"batch_size": 4, "sample_order_seed_offset": 10000},
        "training": {
            "map_updates": map_updates, "render_updates": render_updates,
            "map_learning_rate": 5e-4, "render_learning_rate": 1e-4,
            "adam_betas": (0.9, 0.999), "gradient_clip_norm": 1.0,
            "render_loss_weight": 1.0, "consecutive_intervals": patience,
            "relative_improvement_threshold": 0.005,
        },
        "scientific": {"evaluation": {"evaluation_interval": 1, "checkpoint_interval": 1}},
        "experiment": {"design": {"dropout_probability": 0.1}},
        "rendering": {},
    }


def _toy_execution():
    return {"experiment_settings": {"training": {"d4_seed_offset": 20000}}}


def _toy_condition():
    return {"id": "d4_only", "label": "D4 only", "d4": True, "dropout": False}


def _toy_evaluator(model, total, stage, local):
    means = {
        "ndr_mean": 0.5, "map_normal": 0.2, "map_diffuse": 0.2,
        "map_roughness": 0.2, "normal_degrees": 10.0,
        "render_l1": 0.8, "near_l1": 0.7,
    }
    material_rows = [
        {"material_id": "f", "category": "Fabric", "crop": 4, **means},
        {"material_id": "l", "category": "Leather", "crop": 4, **means},
    ]
    population = {"mean": means, "materials": material_rows}
    return {
        "total_step": total, "stage": stage, "step": local,
        "populations": {
            "train": population, "unseen_crop": population, "heldout": population,
        },
    }


def _toy_batch_provider(target, features, records, *, fail_on=None):
    calls = {"count": 0}

    def provide(indices, rng, stage, model):
        calls["count"] += 1
        if fail_on is not None and calls["count"] == fail_on:
            raise RuntimeError("injected interruption after last committed update")
        choice = rng.randrange(8)
        scale = torch.rand(())
        records.append({
            "indices": list(indices), "stage": stage,
            "choice": choice, "scale": float(scale),
        })
        selected = torch.tensor([index % len(target) for index in indices])
        return target[selected], torch.ones_like(target[selected, :1], dtype=torch.bool), features[selected] * (1 + scale), [choice] * len(indices)

    return provide, calls


def _toy_targets():
    target = torch.zeros(4, 10, 8, 8)
    target[:, 2] = 1.0
    target[:, 3:6] = -0.25
    target[:, 6:10] = -0.5
    features = torch.linspace(-1, 1, 4 * 8 * 8).reshape(4, 1, 8, 8)
    return target, features


def _invoke_toy_condition(run_dir, model, settings, *, resume, provider, identity=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir
    return runner._run_condition(
        output_dir=output,
        execution=_toy_execution(),
        settings=settings,
        invocation={"pid": os.getpid(), "process_identity": "cpu-test"},
        seed=7,
        condition=_toy_condition(),
        train_population=[],
        raw_train=None,
        populations={},
        train_lights=None,
        novel_lights=None,
        model=model,
        resume=resume,
        identity=identity or {"contract": "toy-cpu"},
        batch_provider=provider,
        evaluator=_toy_evaluator,
    )


def test_cpu_toy_resume_restores_adam_sample_and_random_prefix_and_rolls_back_log(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "render_error", lambda model, pred, *args: pred.new_zeros(()))
    settings = _toy_settings()
    target, features = _toy_targets()
    identity = {"contract": "toy-cpu-v1"}

    torch.manual_seed(99)
    uninterrupted_model = _ToyModel()
    uninterrupted_trace = []
    uninterrupted_provider, _ = _toy_batch_provider(target, features, uninterrupted_trace)
    uninterrupted_dir = tmp_path / "uninterrupted"
    _invoke_toy_condition(
        uninterrupted_dir, uninterrupted_model, settings, resume=False,
        provider=uninterrupted_provider, identity=identity,
    )

    resumed_dir = tmp_path / "resumed"
    torch.manual_seed(99)
    interrupted_model = _ToyModel()
    interrupted_trace = []
    failing_provider, _ = _toy_batch_provider(
        target, features, interrupted_trace, fail_on=2
    )
    with pytest.raises(RuntimeError, match="injected interruption"):
        _invoke_toy_condition(
            resumed_dir, interrupted_model, settings, resume=False,
            provider=failing_provider, identity=identity,
        )
    condition_dir = resumed_dir / "seed7_d4_only"
    committed = torch.load(condition_dir / "last.pt", map_location="cpu", weights_only=False)
    assert committed["map_updates"] == 1 and committed["render_updates"] == 0
    optimizer_steps = {int(value["step"].item()) for value in committed["optimizer"]["state"].values()}
    assert optimizer_steps == {1}

    log_path = condition_dir / "train.jsonl"
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"total_step": 2, "interrupted_suffix": True}) + "\n")
    torch.manual_seed(1234)
    resumed_model = _ToyModel()
    resumed_trace = []
    resumed_provider, _ = _toy_batch_provider(target, features, resumed_trace)
    _invoke_toy_condition(
        resumed_dir, resumed_model, settings, resume=True,
        provider=resumed_provider, identity=identity,
    )

    expected_trace = uninterrupted_trace
    observed_trace = interrupted_trace + resumed_trace
    assert observed_trace == expected_trace
    expected_state = torch.load(
        uninterrupted_dir / "seed7_d4_only" / "last.pt", map_location="cpu", weights_only=False
    )["net_g"]
    actual_state = torch.load(condition_dir / "last.pt", map_location="cpu", weights_only=False)["net_g"]
    for key in expected_state:
        assert torch.equal(expected_state[key], actual_state[key])
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["total_step"] for row in rows] == [1, 2, 3]
    assert (condition_dir / "status.json").is_file()


def test_cpu_toy_early_stop_keeps_dynamic_render_axis_and_plateau_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "render_error", lambda model, pred, *args: pred.new_zeros(()))
    settings = _toy_settings(map_updates=3, render_updates=4, patience=1)
    target, features = _toy_targets()
    model = _ToyModel()
    torch.manual_seed(13)
    trace = []
    provider, _ = _toy_batch_provider(target, features, trace)
    run_dir = tmp_path / "early-stop"
    complete = _invoke_toy_condition(
        run_dir, model, settings, resume=False, provider=provider,
    )
    assert complete["map_updates"] == 1
    assert complete["render_updates"] == 2
    assert complete["map_stop_reason"] == "plateau"
    assert complete["render_stop_reason"] == "plateau"
    condition_dir = run_dir / "seed7_d4_only"
    validation = json.loads((condition_dir / "validation.json").read_text(encoding="utf-8"))["history"]
    assert [(row["stage"], row["step"], row["total_step"]) for row in validation] == [
        ("map", 0, 0), ("map", 1, 1), ("render", 1, 2), ("render", 2, 3)
    ]
    status = json.loads((condition_dir / "status.json").read_text(encoding="utf-8"))
    assert (status["stage"], status["step"], status["total_step"]) == ("render", 2, 3)
    assert complete["render_plateau"]["streak"] == 1


def test_resume_rejects_changed_identity_before_training(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "render_error", lambda model, pred, *args: pred.new_zeros(()))
    settings = _toy_settings()
    target, features = _toy_targets()
    model = _ToyModel()
    records = []
    provider, _ = _toy_batch_provider(target, features, records, fail_on=2)
    run_dir = tmp_path / "identity"
    with pytest.raises(RuntimeError, match="injected interruption"):
        _invoke_toy_condition(
            run_dir, model, settings, resume=False, provider=provider,
            identity={"contract": "v1"},
        )
    resumed_model = _ToyModel()
    valid_provider, _ = _toy_batch_provider(target, features, [])
    with pytest.raises(ValueError, match="identity mismatch"):
        _invoke_toy_condition(
            run_dir, resumed_model, settings, resume=True, provider=valid_provider,
            identity={"contract": "v2"},
        )


def test_report_effects_use_paired_error_contrasts_only_when_full():
    seeds = [7, 19, 31]
    rows = {}
    for seed in seeds:
        values = {"baseline": 10.0, "d4_only": 8.0, "dropout_only": 9.0, "d4_dropout": 6.0}
        for condition, value in values.items():
            rows[seed, condition] = {
                "final": {"heldout": {"mean": {"ndr_mean": value, "render_l1": value / 10}}}
            }
    effects = _effects(rows, seeds)
    assert effects["per_seed"]["7"]["ndr_mean"]["d4_main_effect"] == -2.5
    assert effects["per_seed"]["7"]["ndr_mean"]["dropout_main_effect"] == -1.5
    assert effects["per_seed"]["7"]["ndr_mean"]["interaction"] == -1.0
    assert effects["three_seed_mean"]["render_l1"]["interaction"] == pytest.approx(-0.1)


def test_partial_report_has_no_effect_and_preserves_actual_completion_count(tmp_path, historical_reference):
    run_dir = config_module.prepare_run(output_dir=tmp_path / "partial")
    result = build_report(run_dir)
    assert result["status"] == "partial"
    assert result["effects"] is None
    assert result["completed_condition_seed_pairs"] in (0, 3)
    assert result["expected_condition_seed_pairs"] == 12



def test_baseline_extension_policy_requires_exact_caps_and_plateau_contract():
    policy = {
        "arm_id": "baseline", "seeds": [7, 19, 31],
        "map_total_limit": 10000, "render_total_limit": 3000,
        "evaluation_interval": 100, "checkpoint_interval": 100,
        "early_stop": {
            "consecutive_intervals": 10,
            "relative_improvement_threshold": 0.005,
            "relative_improvement_formula": "(previous-current)/max(abs(previous),1e-12)",
            "map_metric": "heldout.ndr_mean", "render_metric": "heldout.render_l1",
        },
    }
    config_module._validate_baseline_extension_policy(policy, [7, 19, 31])
    changed = copy.deepcopy(policy)
    changed["early_stop"]["render_metric"] = "heldout.ndr_mean"
    with pytest.raises(ValueError, match="plateau policy"):
        config_module._validate_baseline_extension_policy(changed, [7, 19, 31])


def test_report_uses_actual_nested_baseline_steps(tmp_path, monkeypatch, historical_reference):
    run_dir = config_module.prepare_run(output_dir=tmp_path / "baseline-report")
    final = {"heldout": {"mean": {"ndr_mean": 0.2, "render_l1": 0.03}}}
    snapshot = {"seeds": {}}
    for seed in (7, 19, 31):
        snapshot["seeds"][str(seed)] = {
            "complete": {
                "final": final,
                "extension": {
                    "actual_map_updates": 10000, "render_updates": 2700,
                    "total_updates": 12700,
                },
                "actual_map_updates": 1, "render_updates": 2, "updates": 3,
            },
            "status": {"actual_map_updates": 1000, "render_updates": 300},
        }
    (run_dir / "baseline_reference.json").write_text(json.dumps(snapshot), encoding="utf-8")
    monkeypatch.setattr(report_module, "verify_baseline_extension", lambda *args, **kwargs: snapshot)
    result = build_report(run_dir)
    row = next(row for row in result["rows"] if row["seed"] == 7 and row["condition"] == "baseline")
    assert (row["map_updates"], row["render_updates"], row["total_updates"]) == (10000, 2700, 12700)
    assert result["completed_condition_seed_pairs"] == 3
    assert result["effects"] is None


def test_baseline_completion_gate_runs_before_runtime_or_cuda(tmp_path, monkeypatch, historical_reference):
    output = tmp_path / "blocked-before-runtime"
    before = torch.cuda.is_initialized()
    calls = []

    def incomplete(*args, **kwargs):
        calls.append("baseline-gate")
        raise RuntimeError("baseline extension is not complete")

    import train_fabric
    monkeypatch.setattr(train_fabric, "require_scheduler_ready", lambda **kwargs: None)
    monkeypatch.setattr(runner, "verify_baseline_extension", incomplete)
    monkeypatch.setattr(runner, "_runtime_environment", lambda *args: pytest.fail("runtime reached before baseline completion"))
    with pytest.raises(RuntimeError, match="not complete"):
        runner.run_experiment(output_dir=output)
    assert calls == ["baseline-gate"]
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["status"] == "prepared"
    assert not (output / "runtime.json").exists()
    assert torch.cuda.is_initialized() is before


def test_completed_extension_uses_frozen_parent_config_and_rejects_changes(tmp_path):
    parent = tmp_path / "parent"
    reference = tmp_path / "extension"
    output = tmp_path / "regularization"
    parent.mkdir()
    reference.mkdir()
    output.mkdir()
    expected = {"execution_config_version": 1, "config_source": {}, "file_identities": [], "settings": {}}
    parent_config = parent / "config.json"
    parent_config.write_text(json.dumps(expected), encoding="utf-8")
    manifest = {
        "parent_directory": str(parent), "parent_execution_identity": expected,
        "parent_files": {"root": {"config.json": config_module.file_identity(parent_config)}},
    }
    (reference / "extension_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (reference / "status.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    policy = json.loads((Path(config_module.REPO_ROOT) / "experiments/bottleneck_fc/extension_policy.json").read_text())
    (reference / "extension_policy.json").write_text(json.dumps(policy), encoding="utf-8")
    for seed in (7, 19, 31):
        run = reference / f"seed{seed}_baseline"
        run.mkdir()
        (run / "complete.json").write_text(json.dumps({"arm_id": "baseline", "parameters": 7609834}), encoding="utf-8")
        (run / "status.json").write_text(json.dumps({"status": "complete", "seed": seed}), encoding="utf-8")
        for name in ("last.pt", "train.jsonl", "evaluation.json", "validation.json"):
            (run / name).write_text("fixture", encoding="utf-8")
    execution = {"baseline_extension_reference": str(reference), "baseline_execution_identity": expected,
                 "experiment_settings": {"training": {"seeds": [7, 19, 31]}}}
    snapshot = config_module.verify_baseline_extension(execution, output, record=False)
    assert not (reference / "config.json").exists()
    assert any(row["path"] == str(parent_config.resolve()) for row in snapshot["file_identities"])
    assert not (output / "baseline_reference.json").exists()
    parent_config.write_text(json.dumps({**expected, "changed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="parent config identity changed"):
        config_module.verify_baseline_extension(execution, output, record=False)
