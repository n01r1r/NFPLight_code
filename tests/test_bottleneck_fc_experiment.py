import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from experiments.bottleneck_fc.config import (
    REPO_ROOT,
    load_config,
    prepare_run,
    process_identity,
    resolve_repo_path,
    validate_config,
)
from experiments.bottleneck_fc.manage import _status
from experiments.bottleneck_fc.models import BottleneckMixer, create_network, parameter_count


def test_frozen_labels_parameter_counts_and_identical_initial_functions():
    config, _ = load_config()
    arms = {}
    for arm in config["model"]["arms"]:
        torch.manual_seed(7)
        arms[arm["id"]] = create_network(arm["id"], config["model"]).eval()

    baseline = arms["baseline"].state_dict()
    for arm_id in ("bottleneck_fc", "pixel_mlp"):
        assert all(torch.equal(value, arms[arm_id].state_dict()[key])
                   for key, value in baseline.items())
    assert {
        arm_id: parameter_count(network)
        for arm_id, network in arms.items()
    } == config["model"]["expected_parameters"]
    assert config["model"]["expected_parameters"] == {
        "baseline": 7609834,
        "bottleneck_fc": 7741932,
        "pixel_mlp": 7741932,
    }

    torch.set_num_threads(4)
    sample = torch.randn(1, 21, 256, 256)
    with torch.no_grad():
        outputs = [network(sample) for network in arms.values()]
    assert outputs[0].shape == (1, 10, 256, 256)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    ("operator", "is_global"),
    [
        ("per_channel_shared_spatial_fc", True),
        ("per_pixel_channel_mlp", False),
    ],
)
def test_global_spatial_connectivity_and_gradient(operator, is_global):
    config, _ = load_config()
    mixer = BottleneckMixer(operator, config["model"]["bottleneck"])
    mixer.gate.data.fill_(0.1)
    original = torch.randn(1, 256, 32, 32)
    changed = original.clone()
    changed[0, 0, 0, 0] += 1
    delta = mixer(changed) - mixer(original)
    assert (delta[:, :, 20, 20].abs().sum() > 0).item() is is_global
    mixer(original).square().mean().backward()
    assert mixer.fc[0].weight.grad.abs().sum() > 0
    mixer.enabled = False
    assert torch.equal(mixer(original), original)


def test_config_rejects_historical_ids_unknown_fields_bad_shapes_and_ranges():
    config, _ = load_config()
    assert [row["id"] for row in config["model"]["arms"]] == [
        "baseline", "bottleneck_fc", "pixel_mlp"
    ]
    assert [row["label"] for row in config["model"]["arms"]] == [
        "baseline", "baseline+Bottleneck_FC", "baseline+Pixel_MLP"
    ]

    invalid = copy.deepcopy(config)
    invalid["model"]["arms"][1]["id"] = "spatial_fc"
    with pytest.raises(ValueError, match="frozen"):
        validate_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["unlisted"] = True
    with pytest.raises(ValueError, match="unknown"):
        validate_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["model"]["bottleneck"]["side"] = 31
    with pytest.raises(ValueError, match="side"):
        validate_config(invalid)

    invalid = copy.deepcopy(config)
    invalid["rendering"]["novel_lights"]["theta_degrees"] = [12, 91]
    with pytest.raises(ValueError, match="lighting grid is frozen"):
        validate_config(invalid)


def _write_fake_manifest(tmp_path, config):
    manifest = tmp_path / "manifest.jsonl"
    records = []
    for split in (config["dataset"]["train_split"], config["dataset"]["heldout_split"]):
        for category in config["dataset"]["categories"]:
            for index in range(config["dataset"]["materials_per_category"]):
                shard = tmp_path / f"{split}_{category}_{index}.npz"
                shard.write_bytes(b"identity fixture")
                records.append({
                    "split": split,
                    "category": category,
                    "material_id": f"{split}_{category}_{index}",
                    "shard_path": str(shard),
                })
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    config["dataset"]["manifest"] = str(manifest)
    config_path = tmp_path / "smoke_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


def test_prepare_copies_immutable_config_and_uses_repo_root_paths(tmp_path, monkeypatch):
    config, _ = load_config()
    config_path = _write_fake_manifest(tmp_path, copy.deepcopy(config))
    out = tmp_path / "prepared"

    monkeypatch.chdir(tmp_path)
    prepared = prepare_run(config_path, out)
    saved = json.loads((prepared / "config.json").read_text(encoding="utf-8"))
    assert saved["settings"]["experiment_id"] == "bottleneck_fc_probe"
    assert len(saved["resolved_inputs"]["train"]) == 16
    assert len(saved["resolved_inputs"]["unseen_crop"]) == 8
    assert len(saved["resolved_inputs"]["heldout"]) == 8
    assert all("sha256" not in identity for identity in saved["file_identities"])
    assert (prepared / "metadata.json").is_file()
    assert json.loads((prepared / "status.json").read_text(encoding="utf-8"))["status"] == "prepared"
    assert not (prepared / "runtime.json").exists()
    assert resolve_repo_path("artifacts/cwd-independent") == REPO_ROOT / "artifacts" / "cwd-independent"

    original = (prepared / "config.json").read_text(encoding="utf-8")
    modified, _ = load_config(config_path)
    modified["training"]["map_updates"] = 2
    config_path.write_text(json.dumps(modified, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable execution config mismatch"):
        prepare_run(config_path, out)
    assert (prepared / "config.json").read_text(encoding="utf-8") == original


def test_atomic_json_retry_and_checkpoint_log_rollback(tmp_path, monkeypatch):
    from experiments.bottleneck_fc import config as config_module
    from experiments.bottleneck_fc.run import truncate_log_to_checkpoint

    real_replace = config_module.os.replace
    calls = 0

    def busy_twice(source, target):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("simulated Windows reader lock")
        return real_replace(source, target)

    monkeypatch.setattr(config_module.os, "replace", busy_twice)
    monkeypatch.setattr(config_module.time, "sleep", lambda _: None)
    atomic_path = tmp_path / "status.json"
    config_module.atomic_write_json(atomic_path, {"status": "saved"})
    assert json.loads(atomic_path.read_text(encoding="utf-8")) == {"status": "saved"}
    assert calls == 3

    log_path = tmp_path / "train.jsonl"
    rows = [{"total_step": step, "loss": float(step)} for step in (1, 2, 3)]
    log_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + '{"partial":',
        encoding="utf-8",
    )
    monkeypatch.setattr(config_module.os, "replace", real_replace)
    truncate_log_to_checkpoint(log_path, 2)
    actual = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["total_step"] for row in actual] == [1, 2]


def test_status_detects_a_terminated_process_even_if_windows_can_query_its_handle(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    child = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(1.0)"
    ])
    identity = process_identity(child.pid)
    assert identity is not None
    status_path = output_dir / "status.json"
    status_path.write_text(json.dumps({
        "status": "running",
        "pid": child.pid,
        "process_identity": identity,
    }), encoding="utf-8")
    assert _status(output_dir)["status"] == "running"
    child.wait(timeout=5)
    status = _status(output_dir)
    assert status["recorded_status"] == "running"
    assert status["status"] == "interrupted"


def test_modules_import_without_cuda_or_legacy_entrypoint_imports():
    script = r'''
import importlib
import sys
import torch

def forbidden():
    raise AssertionError("import touched CUDA")

torch.cuda.is_available = forbidden
for name in (
    "experiments.bottleneck_fc.manage",
    "experiments.bottleneck_fc.report",
    "experiments.bottleneck_fc.run",
):
    importlib.import_module(name)
assert "train_fabric" not in sys.modules
assert "benchmark_fabric_leather" not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_absolute_manage_script_runs_from_unrelated_working_directory(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = REPO_ROOT / "experiments" / "bottleneck_fc" / "manage.py"
    result = subprocess.run(
        [sys.executable, str(script), "status", "--out", str(tmp_path / "missing-run")],
        cwd=elsewhere,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "unprepared"



def test_per_arm_status_and_evaluation_match_live_viewer_schema(tmp_path):
    from live_training_plot import LogReader
    from experiments.bottleneck_fc.run import _status_document

    config, _ = load_config()
    settings = copy.deepcopy(config)
    settings["training"]["map_updates"] = 2
    settings["training"]["render_updates"] = 1
    run_dir = tmp_path / "seed7_bottleneck_fc"
    run_dir.mkdir()
    status = _status_document(
        settings=settings,
        status="complete",
        output_dir=run_dir,
        invocation={"pid": 12, "process_identity": "process-start"},
        seed=7,
        arm_id="bottleneck_fc",
        stage="render",
        total_step=3,
        last_committed_step=3,
        total_updates=3,
    )
    assert status["step"] == 1
    assert status["total_step"] == 3
    assert status["stage"] == "render"
    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
    (run_dir / "train.jsonl").write_text(
        json.dumps({"total_step": 3, "stage": "render", "loss": 0.2}) + "\n",
        encoding="utf-8",
    )
    (run_dir / "validation.json").write_text(json.dumps({"history": [{
        "total_step": 3,
        "step": 1,
        "stage": "render",
        "map_normal": 0.1,
        "map_diffuse": 0.2,
        "map_roughness": 0.3,
        "render_l1": 0.4,
        "normal_degrees": 5.0,
        "category": {},
    }]}), encoding="utf-8")
    snapshot = LogReader(run_dir).snapshot()
    assert snapshot["latest"]["total_step"] == 3
    assert snapshot["status"]["step"] == 1
    assert snapshot["status"]["total_step"] == 3
    assert snapshot["val"]["render:render_l1"] == [[3, 0.4]]



def test_factory_rejects_historical_ids_and_keeps_spatial_fc_dimensions():
    config, _ = load_config()
    for old_id in ("none", "spatial_fc", "local_mlp"):
        with pytest.raises(ValueError, match="unknown arm id"):
            create_network(old_id, config["model"])
    mixer = BottleneckMixer(
        "per_channel_shared_spatial_fc", config["model"]["bottleneck"]
    )
    assert (mixer.fc[0].in_features, mixer.fc[0].out_features) == (1024, 32)
    assert (mixer.fc[2].in_features, mixer.fc[2].out_features) == (32, 1024)
    assert mixer.fc[0].bias is None and mixer.fc[2].bias is None



def test_resume_restores_adam_order_and_discards_uncommitted_rows(tmp_path, monkeypatch):
    import random
    from experiments.bottleneck_fc import run as run_module

    config, _ = load_config()
    settings = copy.deepcopy(config)
    settings["training"].update(
        seeds=[7], map_updates=2, render_updates=1,
        checkpoint_interval=1, evaluation_interval=1,
    )
    settings["model"]["expected_parameters"]["baseline"] = 1
    output_dir = tmp_path / "resume-run"
    arm_dir = output_dir / "seed7_baseline"
    arm_dir.mkdir(parents=True)
    environment = {"device": "cpu-test"}
    (output_dir / "runtime.json").write_text(
        json.dumps({"environment": environment}), encoding="utf-8"
    )
    failed_status = {"status": "failed", "last_committed_step": 1}
    assert run_module._validate_run_state(output_dir, failed_status, True) == "failed"
    with pytest.raises(RuntimeError, match="pass --resume"):
        run_module._validate_run_state(output_dir, failed_status, False)

    class TinyNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.4))

        def forward(self, features):
            return self.weight * features

    class TinyModel:
        def __init__(self, *_args):
            self.net_g = TinyNetwork()
            self.device = torch.device("cpu")

    seed_model = TinyNetwork()
    seed_optimizer = torch.optim.Adam(
        seed_model.parameters(),
        lr=settings["training"]["map_learning_rate"],
        betas=tuple(settings["training"]["adam_betas"]),
    )
    (seed_model(torch.ones(1, 1, 1, 1)) ** 2).mean().backward()
    seed_optimizer.step()
    optimizer_checkpoint = seed_optimizer.state_dict()

    execution = {
        "execution_config_version": 1,
        "config_source": {},
        "file_identities": [],
        "settings": settings,
        "resolved_inputs": {
            "manifest": "synthetic-manifest",
            "train": [
                {"material_id": f"train{i}", "category": "Fabric", "crop": i}
                for i in range(4)
            ],
            "unseen_crop": [
                {"material_id": "unseen", "category": "Fabric", "crop": 4}
            ],
            "heldout": [
                {"material_id": "heldout-f", "category": "Fabric", "crop": 4},
                {"material_id": "heldout-l", "category": "Leather", "crop": 4},
            ],
        },
    }
    fake_rows = lambda slots: [
        {
            "safe": torch.zeros(1, 1, 1, 1),
            "valid": torch.ones(1, 1, 1, 1, dtype=torch.bool),
            "features": torch.ones(1, 1, 1, 1),
            **slot,
        }
        for slot in slots
    ]
    sampled_batches = []

    def fake_tensors(rows, indices):
        sampled_batches.append(list(indices))
        return [torch.cat([rows[index][key] for index in indices])
                for key in ("safe", "valid", "features")]

    metric_names = (
        "ndr_mean", "map_normal", "map_diffuse", "map_roughness",
        "normal_degrees", "render_l1", "near_l1",
    )

    def fake_evaluate(_model, populations, _lights, _settings, save_dir=None):
        result = {}
        for population, rows in populations.items():
            materials = []
            for row in rows:
                materials.append({
                    "material_id": row["material_id"],
                    "category": row["category"],
                    **{key: 0.1 for key in metric_names},
                })
            result[population] = {
                "mean": {key: 0.1 for key in metric_names},
                "materials": materials,
            }
        return result

    monkeypatch.setattr(run_module, "ProbeModel", TinyModel)
    monkeypatch.setattr(run_module, "parameter_count", lambda _network: 1)
    monkeypatch.setattr(run_module, "cache_slots", lambda _model, _bank, slots, _f0: fake_rows(slots))
    monkeypatch.setattr(run_module, "make_lights", lambda _config, _device: torch.zeros(1, 3))
    monkeypatch.setattr(run_module, "tensors", fake_tensors)
    monkeypatch.setattr(run_module, "unit_prediction", lambda prediction: prediction)
    monkeypatch.setattr(
        run_module, "ndr_loss",
        lambda prediction, _target, _valid: (prediction.square().mean(), [prediction.square().mean()] * 3),
    )
    monkeypatch.setattr(run_module, "render_error", lambda _model, pred, *_args: pred.new_tensor(0.01))
    monkeypatch.setattr(run_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(run_module, "mixers", lambda _network: [])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    initial_populations = {
        "train": fake_rows(execution["resolved_inputs"]["train"]),
        "unseen_crop": fake_rows(execution["resolved_inputs"]["unseen_crop"]),
        "heldout": fake_rows(execution["resolved_inputs"]["heldout"]),
    }
    history = run_module._evaluate_entry(
        0, "map", TinyModel(), initial_populations, torch.zeros(1, 3), settings,
    )
    state = {
        "relation_version": "spatial_inverse_square_v1", "checkpoint_version": 1,
        "seed": 7,
        "arm_id": "baseline",
        "step": 1,
        "net_g": seed_model.state_dict(),
        "optimizer": optimizer_checkpoint,
        "history": [history],
        "execution_identity": {
            "execution_config_version": 1,
            "config_source": {},
            "file_identities": [],
            "settings": settings,
        },
        "environment": environment,
    }
    torch.save(state, arm_dir / "last.pt")
    old_rows = [
        {"total_step": 1, "loss": 0.4},
        {"total_step": 2, "loss": 0.3},
    ]
    (arm_dir / "train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in old_rows) + '{"partial":',
        encoding="utf-8",
    )

    saved_checkpoints = []
    real_save = run_module._atomic_torch_save

    def capture_checkpoint(path, checkpoint):
        saved_checkpoints.append(checkpoint)
        real_save(path, checkpoint)

    monkeypatch.setattr(run_module, "_atomic_torch_save", capture_checkpoint)
    invocation = {"pid": 12, "process_identity": "test-process"}
    run_module._run_arm(
        output_dir=output_dir,
        settings=settings,
        execution=execution,
        invocation=invocation,
        seed=7,
        arm=settings["model"]["arms"][0],
        train_bank=object(),
        heldout_bank=object(),
        resume=True,
        global_status={},
    )

    rng = random.Random(7 + settings["sampling"]["sample_order_seed_offset"])
    expected_order = [rng.sample(range(4), 4) for _ in range(3)]
    assert sampled_batches == expected_order[1:]
    assert [json.loads(line)["total_step"] for line in (arm_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()] == [1, 2, 3]
    map_checkpoint = next(item for item in saved_checkpoints if item["step"] == 2)
    map_parameter_state = next(iter(map_checkpoint["optimizer"]["state"].values()))
    assert float(map_parameter_state["step"]) == 2.0
    render_checkpoint = next(item for item in saved_checkpoints if item["step"] == 3)
    render_parameter_state = next(iter(render_checkpoint["optimizer"]["state"].values()))
    assert float(render_parameter_state["step"]) == 1.0
    assert json.loads((arm_dir / "status.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_run_selection_is_immutable_and_status_reports_scope(tmp_path):
    from experiments.bottleneck_fc.run import _ensure_run_selection, _status_document

    config, _ = load_config()
    output_dir = tmp_path / "selected-run"
    output_dir.mkdir()
    selection = _ensure_run_selection(
        output_dir, config, "baseline", resume=False
    )
    assert selection == {
        "selection_version": 1,
        "scope": "selected",
        "selected_arm_ids": ["baseline"],
    }
    saved_text = (output_dir / "run_selection.json").read_text(encoding="utf-8")
    assert _ensure_run_selection(
        output_dir, config, "baseline", resume=True
    ) == selection
    with pytest.raises(ValueError, match="immutable run selection mismatch"):
        _ensure_run_selection(output_dir, config, "pixel_mlp", resume=True)
    assert (output_dir / "run_selection.json").read_text(encoding="utf-8") == saved_text

    status = _status_document(
        settings=config,
        status="running",
        output_dir=output_dir,
        invocation={"pid": 12, "process_identity": "test-process"},
        run_selection=selection,
    )
    assert status["run_scope"] == "selected"
    assert status["selected_arm_ids"] == ["baseline"]
    assert status["scheduled_runs"] == 3
    assert status["comparison_expected_runs"] == 9

    all_output = tmp_path / "all-run"
    all_output.mkdir()
    all_selection = _ensure_run_selection(all_output, config, None, resume=False)
    assert all_selection["scope"] == "all"
    assert all_selection["selected_arm_ids"] == [
        arm["id"] for arm in config["model"]["arms"]
    ]


def test_run_and_manage_cli_forward_baseline_arm(monkeypatch):
    from experiments.bottleneck_fc import run as run_module
    from experiments.bottleneck_fc.manage import main as manage_main

    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(run_module, "run_experiment", capture)
    run_module.main(["--arm", "baseline", "--resume"])
    manage_main(["run", "--arm", "baseline", "--resume"])

    assert len(calls) == 2
    assert all(call[1] == {"resume": True, "arm_id": "baseline"} for call in calls)


def test_run_experiment_schedules_only_selected_arm_and_marks_root_scope(tmp_path, monkeypatch):
    from experiments.bottleneck_fc import run as run_module

    import train_fabric
    monkeypatch.setattr(train_fabric, "require_scheduler_ready", lambda **kwargs: None)
    config, _ = load_config()
    output_dir = tmp_path / "baseline-only"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({"status": "prepared"}), encoding="utf-8"
    )
    execution = {
        "settings": config,
        "resolved_inputs": {"manifest": "synthetic-manifest"},
    }
    monkeypatch.setattr(run_module, "load_config", lambda _path: (config, tmp_path / "config.json"))
    monkeypatch.setattr(run_module, "prepare_run", lambda *_args: output_dir)
    monkeypatch.setattr(run_module, "load_execution_config", lambda _out: execution)
    monkeypatch.setattr(run_module, "build_execution_config", lambda *_args: execution)
    monkeypatch.setattr(run_module, "same_execution_config", lambda *_args: True)
    monkeypatch.setattr(run_module, "_verify_low_level_contract", lambda _settings: None)
    monkeypatch.setattr(run_module, "_runtime_environment", lambda _settings: {"device": "cpu-test"})
    monkeypatch.setattr(run_module, "_record_invocation", lambda *_args: {
        "pid": 12, "process_identity": "test-process"
    })
    monkeypatch.setattr(run_module, "_set_deterministic_settings", lambda _settings: None)
    monkeypatch.setattr(run_module, "Bank", lambda *_args: object())
    scheduled = []
    monkeypatch.setattr(
        run_module,
        "_run_arm",
        lambda **kwargs: scheduled.append((kwargs["seed"], kwargs["arm"]["id"])),
    )

    run_module.run_experiment(arm_id="baseline")

    assert scheduled == [(7, "baseline"), (19, "baseline"), (31, "baseline")]
    assert json.loads((output_dir / "run_selection.json").read_text(encoding="utf-8")) == {
        "selection_version": 1,
        "scope": "selected",
        "selected_arm_ids": ["baseline"],
    }
    status = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "complete"
    assert status["selected_arm_ids"] == ["baseline"]
    assert status["scheduled_runs"] == 3
    assert status["comparison_expected_runs"] == 9
