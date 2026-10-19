import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.bottleneck_fc import extend as extension
from experiments.bottleneck_fc import run as probe_run
from experiments.bottleneck_fc.config import load_config
from experiments.bottleneck_fc.extend import (
    _checkpoint_payload,
    _copy_parent_prefix,
    _load_extension_checkpoint,
    _initial_stage_position,
    _root_status,
    _seed_status,
    _stage_total_limit,
    _validated_train_log,
    _verify_extension_log,
    _write_history,
    build_sample_order,
    load_policy,
    plateau_decision,
    _read_evaluation_history,
    stage_local_step,
)
from experiments.bottleneck_fc.extension_controller import _parent_ready


def _metric_row(step, stage, value):
    key = "ndr_mean" if stage == "map" else "render_l1"
    return {
        "total_step": step,
        "stage": stage,
        "populations": {"heldout": {"mean": {key: value}}},
    }


def test_policy_and_plateau_rule_use_exact_metric_and_ten_intervals():
    policy = load_policy()
    assert policy["map_total_limit"] == 10000
    assert policy["render_total_limit"] == 3000
    assert policy["early_stop"]["consecutive_intervals"] == 10
    assert policy["early_stop"]["relative_improvement_threshold"] == 0.005

    values = [1 - index * 0.001 for index in range(11)]
    history = [_metric_row(index * 100, "map", value) for index, value in enumerate(values)]
    assert plateau_decision(history[:-1], "map", policy)["status"] == "observing"
    decision = plateau_decision(history, "map", policy)
    assert decision["status"] == "plateau"
    assert decision["metric"] == "heldout.ndr_mean"
    assert decision["streak"] == 10

    history[-1] = _metric_row(1000, "map", 0.98)
    assert plateau_decision(history, "map", policy)["streak"] == 0

    history[-1] = _metric_row(1000, "map", -0.01)
    with pytest.raises(FloatingPointError, match="heldout map metric"):
        plateau_decision(history, "map", policy)


def test_initial_extension_status_uses_frozen_stage_boundaries():
    config, _ = load_config()
    invocation = {"pid": 12, "process_identity": "test"}
    policy = load_policy()

    map_position = _initial_stage_position(config, "extend_map_then_fresh_render")
    render_position = _initial_stage_position(config, "continue_pilot_render")
    assert map_position == ("map", config["training"]["map_updates"], config["training"]["map_updates"])
    assert render_position == (
        "render",
        config["training"]["render_updates"],
        config["training"]["map_updates"] + config["training"]["render_updates"],
    )
    status = _root_status(
        status="running", output_dir=Path("extension"), invocation=invocation,
        policy=policy, seed=7, stage=render_position[0], step=render_position[1],
        total_step=render_position[2], committed=render_position[2],
    )
    assert (status["step"], status["total_step"]) == (300, 1300)


def test_parent_evaluation_history_uses_canonical_json_list(tmp_path):
    path = tmp_path / "evaluation.json"
    history = [_metric_row(100, "map", 0.4)]
    path.write_text(json.dumps(history), encoding="utf-8")
    assert _read_evaluation_history(path) == history

    path.write_text(json.dumps({"history": history}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON list"):
        _read_evaluation_history(path)


def test_render_after_early_map_stop_keeps_stage_local_and_global_axes(tmp_path):
    config, _ = load_config()
    actual_map = 2500
    total_step = actual_map + 1
    assert stage_local_step(total_step, "render", actual_map) == 1
    policy = load_policy()
    assert _stage_total_limit(policy, "render", actual_map) == 5500

    history = [
        _metric_row(actual_map, "map", 0.4),
        _metric_row(total_step, "render", 0.3),
    ]
    history[0]["populations"]["heldout"]["mean"]["render_l1"] = 0.4
    history[1]["populations"]["heldout"]["mean"]["ndr_mean"] = 0.3
    materials = [
        {"category": category, "ndr_mean": 0.3, "render_l1": 0.3}
        for category in config["dataset"]["categories"]
    ]
    for row in history:
        row["populations"]["heldout"]["materials"] = materials

    _write_history(tmp_path, history, config, actual_map)
    evaluation = json.loads((tmp_path / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation == history
    validation = json.loads((tmp_path / "validation.json").read_text(encoding="utf-8"))["history"]
    assert [(row["step"], row["total_step"]) for row in validation] == [
        (actual_map, actual_map), (1, total_step)
    ]
    status = _seed_status(
        status="running", run_dir=tmp_path, seed=7, stage="render",
        total_step=total_step, committed=total_step, map_total=actual_map,
        render_updates=1, total_limit=5500, map_limit=10000,
        invocation={"pid": 123, "process_identity": "test"},
    )
    assert (status["step"], status["total_step"], status["actual_map_updates"]) == (
        1, total_step, actual_map
    )


def test_parent_log_prefix_copy_accepts_validated_suffix_without_weakening_full_check(tmp_path):
    source = tmp_path / "parent_train.jsonl"
    destination = tmp_path / "extension_train.jsonl"
    lines = []
    for step in range(1, 1301):
        stage = "map" if step <= 1000 else "render"
        local_step = step if stage == "map" else step - 1000
        lines.append(json.dumps({
            "total_step": step, "step": local_step, "stage": stage, "loss": float(step),
        }).encode("utf-8") + b"\n")
    source.write_bytes(b"".join(lines))

    assert len(_validated_train_log(source, 1300, 1000)) == 1300
    with pytest.raises(ValueError, match="incomplete parent training log"):
        _validated_train_log(source, 1000, 1000)

    _copy_parent_prefix(source, destination, 1000, 1000)
    assert destination.read_bytes() == b"".join(lines[:1000])
    _verify_extension_log(destination, 1000, 1000)


class _ToyMixer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False))
        torch.nn.init.constant_(self.fc[0].weight, 0.25)
        self.gate = torch.nn.Parameter(torch.tensor(0.1))
        self.last_relative_residual = 0.25

    def forward(self, value):
        return self.fc(value) + self.gate * value


def _training_eval_row(step, stage, settings, ndr_value, render_value):
    means = {"ndr_mean": ndr_value, "render_l1": render_value}
    materials = [
        {"category": category, **means}
        for category in settings["dataset"]["categories"]
    ]
    return {
        "total_step": step,
        "stage": stage,
        "populations": {"heldout": {"mean": means, "materials": materials}},
    }


def test_run_seed_warm_starts_map_then_uses_fresh_render_adam_cpu(tmp_path, monkeypatch):
    assert not torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings, _ = load_config()
    settings = copy.deepcopy(settings)
    settings["runtime"]["device"] = "cpu"
    execution = {
        "execution_config_version": 1,
        "config_source": {"path": "synthetic-config"},
        "file_identities": [],
        "settings": settings,
        "resolved_inputs": {"train": [], "unseen_crop": [], "heldout": []},
    }
    environment = {"device": "CPU", "runtime": "synthetic"}
    parent = tmp_path / "parent"
    seed_dir = parent / "seed7_baseline"
    seed_dir.mkdir(parents=True)
    output_dir = tmp_path / "extension"
    output_dir.mkdir()

    parent_model = _ToyMixer()
    parent_optimizer = probe_run._optimizer(
        SimpleNamespace(net_g=parent_model), settings, "map"
    )
    parent_optimizer.zero_grad(set_to_none=True)
    parent_model(torch.ones(1, 1)).sum().backward()
    parent_optimizer.step()
    parent_model_state = {
        name: value.detach().clone()
        for name, value in parent_model.state_dict().items()
    }
    for state in parent_optimizer.state.values():
        state["step"] = torch.tensor(1000.0)
    history = [_training_eval_row(1000, "map", settings, 1.0, 1.0)]
    identity = {
        "execution_config_version": execution["execution_config_version"],
        "config_source": execution["config_source"],
        "file_identities": execution["file_identities"],
        "settings": settings,
    }
    probe_run._atomic_torch_save(seed_dir / "map_stage_end.pt", {
        "relation_version": "spatial_inverse_square_v1", "checkpoint_version": 1, "seed": 7, "arm_id": "baseline", "step": 1000,
        "net_g": parent_model.state_dict(), "optimizer": parent_optimizer.state_dict(),
        "history": history, "execution_identity": identity, "environment": environment,
    })
    (seed_dir / "train.jsonl").write_text(
        "".join(
            json.dumps({"total_step": step, "step": step, "stage": "map"}) + "\n"
            for step in range(1, 1001)
        ),
        encoding="utf-8",
    )
    policy = {
        "seeds": [7],
        "evaluation_interval": 1, "checkpoint_interval": 1,
        "map_total_limit": 1001, "render_total_limit": 2,
        "early_stop": {
            "relative_improvement_threshold": 0.005, "consecutive_intervals": 1,
        },
    }
    snapshot = {
        "parent_directory": str(parent), "parent_runtime": environment,
        "plans": {"7": {
            "route": "extend_map_then_fresh_render",
            "initial_map_plateau": {"status": "observing"},
            "initial_render_plateau": {"status": "observing"},
        }},
    }

    loaded_states, optimizers, saved_steps = [], [], []
    real_optimizer = probe_run._optimizer
    real_atomic_save = probe_run._atomic_torch_save

    def cache_slots(model, bank, rows, f0):
        if not loaded_states:
            loaded_states.append({
                name: value.detach().clone()
                for name, value in model.net_g.state_dict().items()
            })
        return [None]

    def tracked_optimizer(model, configured, stage):
        optimizer = real_optimizer(model, configured, stage)
        optimizers.append((stage, optimizer))
        return optimizer

    def tracked_atomic_save(path, state):
        saved_steps.append((state["step"], state["stage"], state["optimizer"] is None))
        return real_atomic_save(path, state)

    def evaluation_entry(step, stage, model, populations, novel_lights, configured):
        ndr, render = (0.99, 0.9) if stage == "map" else (0.98, 0.89)
        return _training_eval_row(step, stage, configured, ndr, render)

    monkeypatch.setattr(extension, "parameter_count", lambda net: settings["model"]["expected_parameters"]["baseline"])
    monkeypatch.setattr(extension, "cache_slots", cache_slots)
    monkeypatch.setattr(extension, "make_lights", lambda *args: None)
    monkeypatch.setattr(extension, "tensors", lambda population, indices: (
        torch.full((1, 1), 0.5),
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones((1, 1)),
    ))
    monkeypatch.setattr(extension, "unit_prediction", lambda value: value)

    def ndr_loss(prediction, target, valid):
        loss = (prediction - target).square().mean()
        return loss, (loss, loss, loss)

    monkeypatch.setattr(extension, "ndr_loss", ndr_loss)
    monkeypatch.setattr(
        extension, "render_error",
        lambda model, prediction, *args: (prediction - 0.2).square().mean(),
    )
    monkeypatch.setattr(extension, "mixers", lambda net: [net])
    monkeypatch.setattr(
        probe_run, "ProbeModel",
        lambda configured, arm_id, device: SimpleNamespace(
            net_g=_ToyMixer(), device=torch.device("cpu")
        ),
    )
    monkeypatch.setattr(probe_run, "_optimizer", tracked_optimizer)
    monkeypatch.setattr(probe_run, "_atomic_torch_save", tracked_atomic_save)
    monkeypatch.setattr(probe_run, "_evaluate_entry", evaluation_entry)
    monkeypatch.setattr(
        probe_run, "evaluate", lambda *args, **kwargs: {"heldout": {"mean": {}}},
    )

    extension._run_seed(
        execution=execution, snapshot=snapshot, manifest_identity="synthetic-manifest",
        invocation={"pid": 22, "process_identity": "cpu-test"}, policy=policy, seed=7,
        output_dir=output_dir, train_bank=object(), heldout_bank=object(),
    )

    assert all(torch.equal(loaded_states[0][key], parent_model_state[key]) for key in parent_model_state)
    assert [stage for stage, _ in optimizers] == ["map", "render"]
    assert {int(state["step"].item()) for state in optimizers[0][1].state.values()} == {1001}
    assert {int(state["step"].item()) for state in optimizers[1][1].state.values()} == {2}

    seed_output = output_dir / "seed7_baseline"
    rows = [json.loads(line) for line in (seed_output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1003
    assert (rows[-3]["stage"], rows[-3]["step"], rows[-3]["total_step"]) == ("map", 1001, 1001)
    assert (rows[-2]["stage"], rows[-2]["step"], rows[-2]["total_step"]) == ("render", 1, 1002)
    assert (rows[-1]["stage"], rows[-1]["step"], rows[-1]["total_step"]) == ("render", 2, 1003)
    _verify_extension_log(seed_output / "train.jsonl", 1003, 1001)
    assert saved_steps == [
        (1001, "map", False), (1001, "render", True),
        (1002, "render", False), (1003, "render", False),
    ]
    checkpoint = torch.load(seed_output / "last.pt", map_location="cpu", weights_only=False)
    assert (checkpoint["step"], checkpoint["stage"], checkpoint["actual_map_updates"]) == (1003, "render", 1001)
    assert {int(state["step"].item()) for state in checkpoint["optimizer"]["state"].values()} == {2}
    assert not (seed_output / "last.pt.tmp").exists()

    validation = json.loads((seed_output / "validation.json").read_text(encoding="utf-8"))["history"]
    assert (validation[-1]["stage"], validation[-1]["step"], validation[-1]["total_step"]) == ("render", 2, 1003)
    status = json.loads((seed_output / "status.json").read_text(encoding="utf-8"))
    assert (status["status"], status["stage"], status["step"], status["total_step"]) == (
        "complete", "render", 2, 1003,
    )
    complete = json.loads((seed_output / "complete.json").read_text(encoding="utf-8"))
    assert complete["extension"]["stop_reasons"]["map"]["reason"] == "update_cap"
    assert complete["extension"]["stop_reasons"]["render"]["reason"] == "plateau"
    root_status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert (root_status["stage"], root_status["step"], root_status["total_step"]) == (
        "render", 2, 1003,
    )
    assert not torch.cuda.is_initialized()


def test_resume_checkpoint_keeps_adam_and_sample_order_prefix(tmp_path):
    config, _ = load_config()
    prefix = build_sample_order(7, config, 1300)
    extended = build_sample_order(7, config, 13000)
    assert extended[:len(prefix)] == prefix

    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.ones(2, 2)).square().mean().backward()
    optimizer.step()
    execution = {
        "execution_config_version": 1,
        "config_source": {"path": "fixture"},
        "file_identities": [],
        "settings": {"probe": True},
    }
    environment = {"device": "CPU"}
    manifest_identity = {"path": "manifest", "size": 1, "mtime_ns": 1}
    checkpoint = _checkpoint_payload(
        seed=7, step=1001, stage="map", actual_map_updates=1001,
        model=SimpleNamespace(net_g=net), optimizer=optimizer, history=[],
        execution=execution, environment=environment,
        manifest_identity=manifest_identity,
    )
    path = tmp_path / "last.pt"
    probe_run._atomic_torch_save(path, checkpoint)
    restored = _load_extension_checkpoint(
        path, seed=7, execution=execution, environment=environment,
        manifest_identity=manifest_identity,
    )

    restored_net = torch.nn.Linear(2, 1)
    restored_net.load_state_dict(restored["net_g"])
    restored_optimizer = torch.optim.Adam(restored_net.parameters(), lr=1e-3)
    restored_optimizer.load_state_dict(restored["optimizer"])
    saved_step = optimizer.state[next(iter(net.parameters()))]["step"]
    resumed_step = restored_optimizer.state[next(iter(restored_net.parameters()))]["step"]
    assert torch.equal(saved_step, resumed_step)
    assert torch.equal(
        optimizer.state[next(iter(net.parameters()))]["exp_avg"],
        restored_optimizer.state[next(iter(restored_net.parameters()))]["exp_avg"],
    )


def test_resume_truncates_rows_after_last_committed_checkpoint(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"total_step": step, "step": step, "stage": "map", "loss": float(step)}
        for step in range(1, 5)
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + '{"partial":', encoding="utf-8")
    probe_run.truncate_log_to_checkpoint(path, 2)
    _verify_extension_log(path, 2, 2)
    saved = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["total_step"] for row in saved] == [1, 2]


def test_controller_waits_for_parent_completion_and_checkpoint_archive(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    status_path = parent / "status.json"
    archive_path = parent / "stage_archive_status.json"
    monkeypatch.setattr(
        "experiments.bottleneck_fc.extension_controller.process_identity_matches",
        lambda pid, identity: True,
    )
    status_path.write_text(json.dumps({
        "status": "running", "pid": 10, "process_identity": "live"
    }), encoding="utf-8")
    assert _parent_ready(parent) == (False, "parent_training_active")

    monkeypatch.setattr(
        "experiments.bottleneck_fc.extension_controller.process_identity_matches",
        lambda pid, identity: False,
    )
    status_path.write_text(json.dumps({
        "status": "complete", "pid": 10, "process_identity": "finished"
    }), encoding="utf-8")
    archive_path.write_text(json.dumps({"status": "watching"}), encoding="utf-8")
    assert _parent_ready(parent) == (False, "waiting_for_map_checkpoint_archive")
    archive_path.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    assert _parent_ready(parent) == (True, "ready")