from __future__ import annotations
import copy
import json
import random
import numpy as np
import pytest
import torch
from torch import nn

from experiments.robustness_suite import config
from experiments.robustness_suite.engine import Engine, capture_rng, load_checkpoint, neutral_evaluation
from experiments.robustness_suite.run import create_network, memory_is_low, process_lock
from experiments.regularization.run import create_network as old_network, parameter_count
def equal(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right





def build(out, arm="baseline7m_p01"):
    torch.manual_seed(19)
    np.random.seed(19)
    random.seed(19)
    policy = config.policy_for_arm(arm)
    policy.update(map_lr=.01, map_min_lr=.0025, render_lr=.01, render_min_lr=.0025,
                  patience=0, cooldown=0, floor_observations=2, evaluation_interval=1, checkpoint_interval=1)
    network = nn.Sequential(nn.Linear(2, 1), nn.Dropout(.1))
    def objective(net, indices, transforms, stage):
        target = .8 + engine.augmentation_rng.random() / 100 + float(torch.rand((), generator=engine.noise_generator)) / 100
        features = torch.tensor([[sum(indices) / 20, sum(transforms) / 32]])
        return (net(features) - target).square().mean(), {"target": target}
    def evaluate(net, stage, step):
        assert not net.training and not torch.is_grad_enabled()
        torch.rand(3); random.random(); np.random.rand()
        return {"heldout": {"mean": {"ndr_mean": 1 if step == 0 else .9 if step == 1 else .95,
                                      "render_l1": 1 if stage == "map" or step == 0 else .5 if step == 1 else .51}}}
    engine = Engine(out, network, policy, {"test": "fixed"}, objective, evaluate, sample_count=8)
    return engine


@pytest.mark.parametrize("arm", ["baseline7m_p01", "ema7m_p01", "adamw7m_p01"])
@pytest.mark.parametrize("cut", [3, 5, 8])
def test_exact_resume_with_independent_rng_ema_and_adamw(tmp_path, arm, cut):
    caps = {"map_cap": 30, "render_cap": 30}
    continuous = build(tmp_path / "continuous", arm)
    continuous.run(caps)
    interrupted = build(tmp_path / "resumed", arm)
    interrupted.run(caps, interrupt_after=cut)
    resumed = build(tmp_path / "resumed", arm)
    resumed.run(caps, resume=True)
    equal(continuous.snapshot(), resumed.snapshot())
    rows = [[{k: v for k, v in json.loads(line).items() if k != "seconds"}
             for line in (tmp_path / name / "train.jsonl").read_text().splitlines()]
             for name in ("continuous", "resumed")]
    assert rows[0] == rows[1]
    checkpoint = load_checkpoint(tmp_path / "resumed/last.pt")
    assert checkpoint["package"] == "robustness_suite"
    assert checkpoint["augmentation_rng"] and checkpoint["noise_rng"].dtype == torch.uint8
    assert len(list((tmp_path / "resumed").glob("*.pt"))) == 7


@pytest.mark.parametrize("arm,expected,p", [("baseline7m_p01", 7609834, .1),
                                            ("capacity15m_p01", 15085034, .1),
                                            ("dropout7m_p05", 7609834, .5)])
def test_cpu_architecture_depth_count_dropout_and_baseline_identity(arm, expected, p):
    backbone = {"width": 32, "encoder_blocks": [2, 2, 4], "middle_blocks": 12 if arm == "capacity15m_p01" else 4,
                "decoder_blocks": [2, 2, 2]}
    torch.manual_seed(19)
    net = create_network(config.policy_for_arm(arm), backbone)
    assert parameter_count(net) == expected
    assert all(module.p == p for module in net.modules() if isinstance(module, nn.Dropout))
    if arm == "baseline7m_p01":
        torch.manual_seed(19)
        old = old_network(.1, backbone)
        equal(net.state_dict(), old.state_dict())


def test_ema_eval_swap_restores_online_weights_modes_rng_and_selects_shadow(tmp_path):
    engine = build(tmp_path, "ema7m_p01")
    before = copy.deepcopy(engine.net.state_dict())
    engine.ema = {k: v + 1 for k, v in engine.ema.items()}
    rng = capture_rng()
    modes = [m.training for m in engine.net.modules()]
    engine.evaluate()
    equal(engine.net.state_dict(), before)
    equal(capture_rng(), rng)
    assert [m.training for m in engine.net.modules()] == modes
    equal(engine.state["best_map"]["net_g"], engine.ema)
    assert "online_populations" in engine.state["history"][-1]


def test_ema_transition_preserves_online_full_state_and_starts_selected_map(tmp_path):
    engine = build(tmp_path, "ema7m_p01")
    engine.run({"map_cap": 30, "render_cap": 30})
    stage_end = load_checkpoint(tmp_path / "map_stage_end.pt")
    assert stage_end["optimizer"]["state"] and stage_end["ema"]
    assert stage_end["state"]["stage"] == "map"
    assert stage_end["state"]["best_map"]["evaluation_weights"] == "ema"


def test_ema_updates_once_per_optimizer_step_and_adamw_explicit(tmp_path):
    engine = build(tmp_path / "ema", "ema7m_p01")
    shadow = copy.deepcopy(engine.ema)
    with torch.no_grad():
        for parameter in engine.net.parameters():
            parameter.add_(2)
    online = engine.net.state_dict()
    engine.update_ema()
    for key in shadow:
        assert torch.allclose(engine.ema[key], shadow[key] * .999 + online[key] * .001)
        assert engine.ema[key].device.type == "cpu"
    assert isinstance(build(tmp_path / "adamw", "adamw7m_p01").optimizer, torch.optim.AdamW)
    assert build(tmp_path / "adamw2", "adamw7m_p01").optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert build(tmp_path / "adam", "baseline7m_p01").optimizer.param_groups[0]["weight_decay"] == 0


def test_memory_guard_rolls_back_custom_rng_shadow_progress(tmp_path):
    engine = build(tmp_path, "ema7m_p01")
    engine.run({"map_cap": 2, "render_cap": 30})
    checkpoint = load_checkpoint(tmp_path / "last.pt")
    engine.memory_guard = lambda: True
    engine.run({"map_cap": 30, "render_cap": 30}, resume=True)
    assert engine.state["status"] == "memory_paused" and engine.state["total_step"] == 2
    equal(engine.ema, checkpoint["ema"])
    equal(engine.augmentation_rng.getstate(), checkpoint["augmentation_rng"])
    assert memory_is_low({"available_ram_bytes": 1})
    assert memory_is_low({"cuda_free_bytes": 1, "cuda_reserved_bytes": 0, "cuda_allocated_bytes": 0})
    assert not memory_is_low({"available_ram_bytes": 2**30, "cuda_free_bytes": 2**30})


def test_new_package_rejects_historical_checkpoint_and_duplicate_lock(tmp_path):
    torch.save({"checkpoint_version": 1, "package": "plateau_followup"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="only new"):
        load_checkpoint(tmp_path / "old.pt")
    with process_lock(tmp_path, resume=False):
        with pytest.raises(RuntimeError, match="active"):
            with process_lock(tmp_path, resume=True):
                pass


def test_policy_validation_prevents_scientific_drift():
    policy = config.policy_for_arm("baseline7m_p01")
    scientific = {"rendering": {"feature_version": "raw_calibrated_gain9_v3", "precision": "float32", "amp": False,
                               "tf32": False, "near_distance": 4., "far_distance": 12., "gain": 9., "f0": .04},
                  "sampling": {"batch_size": 4, "sample_order_seed_offset": 10000},
                  "backbone": {"width": 32, "encoder_blocks": [2, 2, 4], "middle_blocks": 4, "decoder_blocks": [2, 2, 2]},
                  "optimization": {"adam_betas": [.9, .999], "gradient_clip_norm": 1., "render_loss_weight": 1.}}
    config.validate_policy(policy, scientific)
    policy["seed"] = 31
    with pytest.raises(ValueError, match="approved"):
        config.validate_policy(policy, scientific)
    policy = config.policy_for_arm("baseline7m_p01")
    scientific["rendering"]["amp"] = True
    with pytest.raises(ValueError, match="FP32"):
        config.validate_policy(policy, scientific)


def test_baseline_training_prefix_matches_historical_engine(tmp_path):
    from experiments.plateau_followup.engine import Engine as OldEngine
    def make(kind, out):
        torch.manual_seed(19); np.random.seed(19); random.seed(19)
        net = nn.Sequential(nn.Linear(2, 1), nn.Dropout(.1))
        def objective(network, indices, transforms, stage):
            features = torch.tensor([[sum(indices) / 20, sum(transforms) / 32]])
            target = .8 + .01 * (random.random() + np.random.rand())
            return (network(features) - target).square().mean(), {"target": target}
        def evaluator(network, stage, step):
            return {"heldout": {"mean": {"ndr_mean": 1., "render_l1": 1.}}}
        policy = config.policy_for_arm("baseline7m_p01")
        policy.update(evaluation_interval=1, checkpoint_interval=1)
        return kind(out, net, policy, {"test": "fixed"}, objective, evaluator, sample_count=8)
    old = make(OldEngine, tmp_path / "old")
    old.run({"map_cap": 3, "render_cap": 30})
    new = make(Engine, tmp_path / "new")
    new.run({"map_cap": 3, "render_cap": 30})
    for key in ("net_g", "optimizer", "scheduler", "rng", "sample_rng", "d4_rng"):
        equal(old.snapshot()[key], new.snapshot()[key])
    logs = [[{k: v for k, v in json.loads(line).items() if k != "seconds"}
             for line in (tmp_path / name / "train.jsonl").read_text().splitlines()] for name in ("old", "new")]
    assert logs[0] == logs[1]


def test_preparation_source_snapshot_and_identity_immutability(tmp_path, monkeypatch):
    source = tmp_path / "source.py"
    source.write_text("source = 1")
    scientific = {"rendering": {"feature_version": "raw_calibrated_gain9_v3", "precision": "float32", "amp": False,
                               "tf32": False, "near_distance": 4., "far_distance": 12., "gain": 9., "f0": .04},
                  "sampling": {"batch_size": 4, "sample_order_seed_offset": 10000},
                  "backbone": {"width": 32, "encoder_blocks": [2, 2, 4], "middle_blocks": 4, "decoder_blocks": [2, 2, 2]},
                  "optimization": {"adam_betas": [.9, .999], "gradient_clip_norm": 1., "render_loss_weight": 1.}, "dataset": {}}
    inputs = {"train": [{"crop": 0}] * 16}
    def base():
        return {"scientific_settings": copy.deepcopy(scientific), "resolved_inputs": copy.deepcopy(inputs),
                "file_identities": [config.file_identity(source)], "control": "historical"}
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "base_identity", base)
    import experiments.robustness_suite.data as data
    monkeypatch.setattr(data, "audit_inputs", lambda execution: {"eligible_extra_crop": False, "safe_train_slots": inputs["train"], "evidence_files": []})
    out = tmp_path / "run"
    _, execution = config.prepare(out)
    assert (out / "source_snapshot/source.py").read_text() == source.read_text()
    assert execution["control_roles"]["fresh_matched_control"] == "baseline7m_p01"
    config.prepare(out)
    source.write_text("changed = 123")
    with pytest.raises(ValueError, match="identity"):
        config.prepare(out)
    blocked = tmp_path / "blocked"
    with pytest.raises(ValueError, match="extra_crop"):
        config.prepare(blocked, arm="extra_crop7m_p01")
    assert not blocked.exists()


def test_reject_smoke_output_before_cuda(tmp_path, monkeypatch):
    from experiments.robustness_suite import run as runner
    (tmp_path / "smoke.json").write_text("{}")
    monkeypatch.setattr(runner, "prepare", lambda *args, **kwargs: (tmp_path, {"policy": config.policy_for_arm("baseline7m_p01")}))
    monkeypatch.setattr(runner, "runtime_environment", lambda: pytest.fail("CUDA boundary reached"))
    with pytest.raises(ValueError, match="separate"):
        runner.run(tmp_path)
