from __future__ import annotations
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn

from experiments.plateau_followup import config
from experiments.plateau_followup.engine import Engine, capture_rng, convergence_update, load_checkpoint, metric_values, neutral_evaluation
from experiments.plateau_followup.run import fork_checkpoint, process_lock


def tiny_policy():
    policy = copy.deepcopy(config.POLICY)
    policy.update(map_lr=0.01, map_min_lr=0.0025, render_lr=0.01, render_min_lr=0.0025,
                  patience=0, cooldown=0, floor_observations=2, evaluation_interval=1, checkpoint_interval=1)
    return policy


def build(output, *, evaluator=None, policy=None):
    torch.manual_seed(19)
    np.random.seed(19)
    random.seed(19)
    net = nn.Sequential(nn.Linear(2, 1), nn.Dropout(0.1))
    render_starts = []
    def objective(network, indices, transforms, stage):
        if stage == "render":
            render_starts.append(copy.deepcopy(network.state_dict()))
        value = torch.tensor([[sum(indices)/20, sum(transforms)/32]], dtype=torch.float32)
        desired = 0.8 + 0.01 * (random.random() + np.random.rand())
        loss = (network(value) - desired).square().mean()
        return loss, {"target": desired}
    def evaluate(network, stage, step):
        assert not network.training and not torch.is_grad_enabled()
        torch.rand(3)
        np.random.rand()
        random.random()
        if stage == "map":
            ndr = 1.0 if step == 0 else 0.9 if step == 1 else 0.95
            rendering = 1.0
        else:
            ndr = 0.9
            rendering = 1.0 if step == 0 else 0.5 if step == 1 else 0.51
        return {"heldout": {"mean": {"ndr_mean": ndr, "render_l1": rendering}, "materials": []}}
    engine = Engine(output, net, policy or tiny_policy(), {"test": "fixed"}, objective, evaluator or evaluate,
                    sample_count=8, batch_size=4)
    engine.render_starts = render_starts
    return engine


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


@pytest.mark.parametrize("cut", [3, 5, 8])
def test_exact_resume_at_lr_drop_and_map_render_boundary(tmp_path, cut):
    caps = {"map_cap": 30, "render_cap": 30}
    uninterrupted = build(tmp_path / "continuous")
    uninterrupted.run(caps)
    interrupted = build(tmp_path / "resumed")
    interrupted.run(caps, interrupt_after=cut)
    checkpoint = load_checkpoint(tmp_path / "resumed/last.pt")
    assert checkpoint["state"]["status"] == "interrupted"
    resumed = build(tmp_path / "resumed")
    resumed.run(caps, resume=True)
    equal(uninterrupted.snapshot(), resumed.snapshot())
    logs = [[{key: value for key, value in json.loads(line).items() if key != "seconds"} for line in (tmp_path / name / "train.jsonl").read_text().splitlines()] for name in ("continuous", "resumed")]
    assert logs[0] == logs[1]
    assert uninterrupted.state["map_updates"] == 5
    assert uninterrupted.state["best_map"]["step"] == 1
    equal(uninterrupted.render_starts[0], uninterrupted.state["best_map"]["net_g"])
    assert uninterrupted.state["render_start_sample_offset"] == 5
    assert uninterrupted.state["best_render"]["step"] == 1
    end = load_checkpoint(tmp_path / "continuous/map_stage_end.pt")
    assert end["state"]["stage"] == "map" and end["state"]["stage_step"] == 5
    assert end["optimizer"]["state"]
    assert end["scheduler"]["_last_lr"] == [0.0025]
    assert len(list((tmp_path / "continuous").glob("*.pt"))) == 7
    assert load_checkpoint(tmp_path / "continuous/selected.pt")["step"] == 1
    assert load_checkpoint(tmp_path / "continuous/last.pt")["state"]["stage_step"] == 5


@pytest.mark.parametrize("raises", [False, True])
def test_evaluation_restores_all_rng_and_modes_even_on_error(raises):
    network = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.1))
    network.train()
    network[0].eval()
    modes = [module.training for module in network.modules()]
    before = capture_rng()
    try:
        with neutral_evaluation(network):
            assert all(not module.training for module in network.modules())
            assert not torch.is_grad_enabled()
            x = torch.ones(8, 2)
            assert torch.equal(network(x), network(x))
            random.random()
            np.random.rand()
            torch.rand(2)
            if raises:
                raise RuntimeError("evaluation failure")
    except RuntimeError:
        assert raises
    equal(before, capture_rng())
    assert [module.training for module in network.modules()] == modes


def test_floor_arrival_and_cumulative_meaningful_improvement():
    record = Engine._new_convergence()
    convergence_update(record, 1.0, lr_before=0.01, lr_after=0.01, floor=0.0025, threshold=0.005)
    convergence_update(record, 0.999, lr_before=0.005, lr_after=0.0025, floor=0.0025, threshold=0.005)
    assert record["floor_bad_observations"] == 0
    convergence_update(record, 0.997, lr_before=0.0025, lr_after=0.0025, floor=0.0025, threshold=0.005)
    assert record["floor_bad_observations"] == 1 and record["meaningful_best"] == 1.0
    convergence_update(record, 0.994, lr_before=0.0025, lr_after=0.0025, floor=0.0025, threshold=0.005)
    assert record["floor_bad_observations"] == 0 and record["meaningful_best"] == 0.994


def test_raw_minimum_distinct_from_threshold_and_adam_preserved(tmp_path):
    def evaluation(net, stage, step):
        return {"heldout": {"mean": {"ndr_mean": 1 - 0.001 * min(step, 2), "render_l1": 1.0}}}
    engine = build(tmp_path, evaluator=evaluation)
    engine.run({"map_cap": 2, "render_cap": 30})
    assert engine.state["status"] == "budget_paused"
    assert engine.state["best_map"]["step"] == 2
    assert engine.state["convergence"]["map"]["meaningful_best"] == 1.0
    assert engine.scheduler.best == 1.0
    assert engine.optimizer.param_groups[0]["lr"] == 0.0025
    assert all(float(value["step"]) == 2 for value in engine.optimizer.state.values())


@pytest.mark.parametrize("initial_eligible", [True, False])
def test_render_guard_and_baseline_fallback(tmp_path, initial_eligible):
    def evaluation(net, stage, step):
        ndr = 1.0 if stage == "map" else (1.0 if step == 0 and initial_eligible else 2.0)
        render = 0.5 if stage == "map" or step == 0 else 0.7
        return {"heldout": {"mean": {"ndr_mean": ndr, "render_l1": render}}}
    engine = build(tmp_path, evaluator=evaluation)
    engine.run({"map_cap": 30, "render_cap": 30})
    assert engine.state["selected"] == ("best_render" if initial_eligible else "best_map")
    selected = engine.state[engine.state["selected"]]
    equal(selected["net_g"], engine.state["best_map"]["net_g"])
    assert selected["render_l1"] == 0.5


def test_eligible_but_worse_render_keeps_stage0_bestmap(tmp_path):
    def evaluation(net, stage, step):
        return {"heldout": {"mean": {"ndr_mean": 1.0, "render_l1": 0.5 if stage == "map" or step == 0 else 0.7}}}
    engine = build(tmp_path, evaluator=evaluation)
    engine.run({"map_cap": 30, "render_cap": 30})
    assert engine.state["best_render"]["step"] == 0
    equal(engine.state["best_render"]["net_g"], engine.state["best_map"]["net_g"])


def test_cap_pauses_map_and_explicit_extension_matches_continuous(tmp_path):
    uninterrupted = build(tmp_path / "full")
    uninterrupted.run({"map_cap": 30, "render_cap": 30})
    first = build(tmp_path / "extended")
    first.run({"map_cap": 2, "render_cap": 30})
    assert first.state["stage"] == "map" and first.state["status"] == "budget_paused"
    assert not (tmp_path / "extended/map_stage_end.pt").exists()
    resumed = build(tmp_path / "extended")
    resumed.run({"map_cap": 30, "render_cap": 30}, resume=True)
    equal(uninterrupted.snapshot(), resumed.snapshot())


def test_budget_monotonic_immutable_base(tmp_path):
    path = tmp_path / "budget_ledger.json"
    config.atomic_write_json(path, [{"map_cap": 50000, "render_cap": 15000}])
    original_policy = copy.deepcopy(config.POLICY)
    for value in (40000, 50000, 50101, True):
        with pytest.raises(ValueError):
            config.budget(tmp_path, resume=True, extend_map_to=value)
    with pytest.raises(ValueError, match="resume"):
        config.budget(tmp_path, extend_map_to=60000)
    assert config.budget(tmp_path, resume=True, extend_map_to=60000)["map_cap"] == 60000
    ledger = config.read_json(path)
    assert len(ledger) == 2 and ledger[0]["map_cap"] == 50000
    assert config.POLICY == original_policy


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_rejects_nonfinite_and_negative_metrics(value):
    with pytest.raises(FloatingPointError):
        metric_values({"heldout": {"mean": {"ndr_mean": value, "render_l1": 1.0}}})
    with pytest.raises(FloatingPointError):
        metric_values({"heldout": {"mean": {"ndr_mean": 1.0, "render_l1": 1.0}, "materials": [{"ndr_mean": value}]}})


def test_strict_checkpoint_and_duplicate_process_lock(tmp_path):
    engine = build(tmp_path)
    engine.run({"map_cap": 2, "render_cap": 30})
    checkpoint = load_checkpoint(tmp_path / "last.pt")
    checkpoint["identity"] = {"changed": True}
    with pytest.raises(ValueError, match="identity"):
        engine.restore(checkpoint)
    torch.save({"checkpoint_version": 1, "package": "historical"}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="only new"):
        load_checkpoint(tmp_path / "old.pt")
    with process_lock(tmp_path, resume=False):
        with pytest.raises(RuntimeError, match="active"):
            with process_lock(tmp_path, resume=True):
                pass
    assert not (tmp_path / "process_lock.json").exists()


def test_resume_rolls_back_uncommitted_log_tail(tmp_path):
    engine = build(tmp_path)
    engine.run({"map_cap": 2, "render_cap": 30})
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write(json.dumps({"total_step": 3}) + "\n")
        stream.write('{"total_step":4')
    resumed = build(tmp_path)
    resumed.run({"map_cap": 2, "render_cap": 30}, resume=True)
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 2



def test_sample_stream_prefix_matches_legacy_without_cap_sized_order(tmp_path):
    engine = build(tmp_path)
    engine.run({"map_cap": 2, "render_cap": 30})
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    sample_rng = random.Random(10019)
    d4_rng = random.Random(20019)
    assert [row["indices"] for row in rows] == [sample_rng.sample(range(8), 4) for _ in rows]
    assert [row["d4_transforms"] for row in rows] == [[d4_rng.randrange(8) for _ in range(4)] for _ in rows]
    assert not hasattr(engine, "sample_order")


def test_proactive_memory_pause_rolls_back_to_commit(tmp_path):
    policy = tiny_policy()
    policy["checkpoint_interval"] = 2
    engine = build(tmp_path, policy=policy)
    engine.memory_guard = lambda: engine.state["total_step"] == 3
    engine.run({"map_cap": 30, "render_cap": 30})
    assert engine.state["status"] == "memory_paused" and engine.state["total_step"] == 2
    assert len((tmp_path / "train.jsonl").read_text().splitlines()) == 2
    assert load_checkpoint(tmp_path / "last.pt")["state"]["total_step"] == 2


def test_invalid_population_and_objective_fail_explicitly(tmp_path):
    with pytest.raises(ValueError, match="batch"):
        Engine(tmp_path, nn.Linear(1, 1), tiny_policy(), {}, None, None, sample_count=2, batch_size=4)
    engine = build(tmp_path)
    engine.objective = lambda *args: (torch.tensor(float("nan"), requires_grad=True), {})
    with pytest.raises(FloatingPointError, match="objective"):
        engine.run({"map_cap": 30, "render_cap": 30})
    assert config.read_json(tmp_path / "status.json")["status"] == "failed"
    assert load_checkpoint(tmp_path / "last.pt")["state"]["total_step"] == 0


def test_smoke_cli_and_both_objectives_with_cpu_adapter(tmp_path, monkeypatch):
    from experiments.plateau_followup import run as runner
    fake = build(tmp_path)
    class FakePipeline:
        def __init__(self, execution):
            from types import SimpleNamespace
            self.model = SimpleNamespace(net_g=fake.net)
            self.objective = fake.objective
            self.evaluate = fake.evaluator
    monkeypatch.setattr(runner, "prepare", lambda output: (Path(output), {}))
    monkeypatch.setattr(runner, "runtime_environment", lambda: {"test": "CPU"})
    monkeypatch.setattr(runner, "Pipeline", FakePipeline)
    result = runner.smoke(tmp_path)
    assert set(result["stages"]) == {"map", "render"}
    assert result["status"] == "smoke_complete"
    assert (tmp_path / "smoke.json").exists()
    with pytest.raises(SystemExit):
        runner.main(["status", "--out", str(tmp_path), "--resume"])


@pytest.mark.parametrize("stage", ["map", "render"])
def test_fork_preserves_last_state_resets_floor_and_never_edits_parent(tmp_path, monkeypatch, stage):
    from experiments.plateau_followup import run as runner
    policy = tiny_policy()
    monkeypatch.setattr(runner, "POLICY", policy)
    parent = build(tmp_path / "parent", policy=policy)
    parent.run({"map_cap": 30, "render_cap": 30})
    source = tmp_path / "parent" / (stage + "_stage_end.pt")
    before = source.read_bytes()
    child = tmp_path / "child"
    child.mkdir()
    runner.fork_checkpoint(child, {"test": "fixed"}, source, stage)
    saved = load_checkpoint(child / "last.pt")
    original = load_checkpoint(source)
    equal(saved["net_g"], original["net_g"])
    equal(saved["optimizer"], original["optimizer"])
    equal(saved["scheduler"], original["scheduler"])
    equal(saved["rng"], original["rng"])
    assert saved["state"]["convergence"][stage]["floor_bad_observations"] == 0
    assert saved["state"]["status"] == "interrupted"
    assert source.read_bytes() == before
    resumed = build(child, policy=policy)
    resumed.run({"map_cap": 30, "render_cap": 30}, resume=True)
    assert resumed.state["status"] == "complete"
    assert resumed.state[stage + "_updates"] == original["state"][stage + "_updates"] + 2
    assert source.read_bytes() == before


def test_rejected_completed_extension_does_not_change_ledger(tmp_path, monkeypatch):
    from experiments.plateau_followup import run as runner
    engine = build(tmp_path)
    engine.run({"map_cap": 30, "render_cap": 30})
    config.atomic_write_json(tmp_path / "budget_ledger.json", [{"map_cap": 50000, "render_cap": 15000}])
    before = (tmp_path / "budget_ledger.json").read_bytes()
    monkeypatch.setattr(runner, "prepare", lambda output: (Path(output), {"test": "fixed"}))
    monkeypatch.setattr(runner, "runtime_environment", lambda: {"test": "CPU"})
    with pytest.raises(ValueError, match="completed-stage"):
        runner.run(tmp_path, resume=True, extend_render_to=20000)
    assert (tmp_path / "budget_ledger.json").read_bytes() == before


def test_memory_guard_accounts_for_unused_cuda_reservation():
    from experiments.plateau_followup.run import memory_is_low
    assert memory_is_low({"available_ram_bytes": 100})
    assert memory_is_low({"cuda_free_bytes": 100})
    assert not memory_is_low({"cuda_free_bytes": 100, "cuda_reserved_bytes": 2 * 1024**3, "cuda_allocated_bytes": 1024**3})


def test_complete_resume_repairs_missing_outputs_without_training_or_rng_change(tmp_path):
    engine = build(tmp_path)
    engine.run({"map_cap": 30, "render_cap": 30})
    original = engine.snapshot()
    (tmp_path / "complete.json").unlink()
    (tmp_path / "render_stage_end.pt").unlink()
    (tmp_path / "selected.pt").unlink()
    resumed = build(tmp_path)
    resumed.objective = lambda *args: pytest.fail("completed resume trained again")
    resumed.evaluator = lambda *args: pytest.fail("completed resume evaluated again")
    resumed.run({"map_cap": 30, "render_cap": 30}, resume=True)
    equal(original, resumed.snapshot())
    assert (tmp_path / "complete.json").exists()
    assert (tmp_path / "render_stage_end.pt").exists()
    assert (tmp_path / "selected.pt").exists()
    equal(original, load_checkpoint(tmp_path / "render_stage_end.pt"))


def test_fork_requires_matching_parent_runtime(tmp_path):
    from experiments.plateau_followup.run import validate_fork_environment
    source = tmp_path / "map_stage_end.pt"
    with pytest.raises(FileNotFoundError, match="runtime"):
        validate_fork_environment(source, {"torch": "current"})
    config.atomic_write_json(tmp_path / "runtime.json", {"environment": {"torch": "original"}})
    with pytest.raises(ValueError, match="environment"):
        validate_fork_environment(source, {"torch": "current"})
    validate_fork_environment(source, {"torch": "original"})
