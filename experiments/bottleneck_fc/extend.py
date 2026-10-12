"""Strict, resumable baseline-only extension runner."""
from __future__ import annotations
import json
import math
import os
import random
import sys
import time
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from experiments.bottleneck_fc import run as probe_run
from experiments.bottleneck_fc.config import (
    atomic_write_json, build_execution_config, file_identity,
    load_config, load_execution_config, process_identity_matches,
    replace_with_retry, resolve_repo_path, same_execution_config,
)
from experiments.bottleneck_fc.models import mixers, parameter_count
from experiments.bottleneck_fc.pipeline import (
    Bank, cache_slots, make_lights, ndr_loss, render_error, tensors, unit_prediction,
)
POLICY_PATH = Path(__file__).with_name("extension_policy.json")
EXTENSION_VERSION = 1


def load_policy(path=POLICY_PATH):
    policy = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    fields = {
        "policy_version", "parent_run_directory", "output_directory", "arm_id", "seeds",
        "map_total_limit", "render_total_limit", "evaluation_interval", "checkpoint_interval",
        "sample_order_seed_offset", "early_stop", "checkpoint_archive", "stage_order",
        "identity_method", "original_results_preserved", "other_arms_authorized", "approved_by_user",
    }
    if not isinstance(policy, dict) or set(policy) != fields:
        raise ValueError(f"extension policy fields must be exactly {sorted(fields)}")
    early = policy["early_stop"]
    if not isinstance(early, dict) or set(early) != {
        "consecutive_intervals", "relative_improvement_threshold",
        "relative_improvement_formula", "map_metric", "render_metric",
    }:
        raise ValueError("invalid extension early-stop policy fields")
    if policy["policy_version"] != 1 or policy["arm_id"] != "baseline":
        raise ValueError("only extension policy v1 for baseline is supported")
    if policy["seeds"] != [7, 19, 31] or policy["other_arms_authorized"] is not False:
        raise ValueError("extension is restricted to baseline seeds 7, 19, and 31")
    frozen = {
        "map_total_limit": 10000, "render_total_limit": 3000,
        "evaluation_interval": 100, "checkpoint_interval": 100,
        "sample_order_seed_offset": 10000,
        "checkpoint_archive": "seed{seed}_baseline/map_stage_end.pt",
        "stage_order": (
            "continue_map_from_archived_map1000_adam_if_map_not_plateaued; "
            "then fresh_adam_render; otherwise continue_original_render300_adam"
        ),
    }
    if any(policy.get(key) != value for key, value in frozen.items()):
        raise ValueError("extension caps, intervals, sample order, and stage route are frozen")
    for key in ("map_total_limit", "render_total_limit", "evaluation_interval", "checkpoint_interval"):
        value = policy[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"extension policy {key} must be a positive integer")
    if any(policy[k] % policy["evaluation_interval"] for k in ("map_total_limit", "render_total_limit")):
        raise ValueError("extension stage caps must align to validation intervals")
    if policy["checkpoint_interval"] != policy["evaluation_interval"]:
        raise ValueError("extension checkpoints and validation must share the approved interval")
    threshold, patience = early["relative_improvement_threshold"], early["consecutive_intervals"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold <= 0
            or isinstance(patience, bool) or not isinstance(patience, int) or patience < 1):
        raise ValueError("invalid extension plateau threshold or patience")
    if early["relative_improvement_formula"] != "(previous-current)/max(abs(previous),1e-12)":
        raise ValueError("unsupported relative-improvement formula")
    if (early["map_metric"] != "heldout.ndr_mean"
            or early["render_metric"] != "heldout.render_l1"
            or patience != 10 or threshold != 0.005):
        raise ValueError("extension plateau policy is frozen at 10 intervals and relative improvement 0.005")
    if (policy["identity_method"] != "resolved path, file size, and mtime_ns; no content hashes"
            or policy["original_results_preserved"] is not True or policy["approved_by_user"] is not True):
        raise ValueError("invalid extension authorization or identity policy")
    return policy


def plateau_decision(history, stage, policy):
    if stage not in ("map", "render"):
        raise ValueError(f"unknown extension stage {stage!r}")
    metric = "ndr_mean" if stage == "map" else "render_l1"
    values, prior_step = [], -1
    for row in history:
        if row.get("stage") != stage:
            continue
        step = row.get("total_step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= prior_step:
            raise ValueError(f"invalid {stage} validation update order")
        prior_step = step
        try:
            value = row["populations"]["heldout"]["mean"][metric]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing heldout {stage} metric {metric}") from exc
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise FloatingPointError(f"invalid heldout {stage} metric {metric}")
        values.append(float(value))
    if not values:
        return {
            "status": "observing", "metric": f"heldout.{metric}",
            "observations": 0, "streak": 0, "latest_relative_improvement": None,
            "threshold": policy["early_stop"]["relative_improvement_threshold"],
            "required_intervals": policy["early_stop"]["consecutive_intervals"],
        }
    improvements = [(a - b) / max(abs(a), 1e-12) for a, b in zip(values, values[1:])]
    streak, threshold = 0, policy["early_stop"]["relative_improvement_threshold"]
    for value in reversed(improvements):
        if value >= threshold:
            break
        streak += 1
    patience = policy["early_stop"]["consecutive_intervals"]
    return {
        "status": "plateau" if streak >= patience else "observing",
        "metric": f"heldout.{metric}", "observations": len(values), "streak": streak,
        "latest_relative_improvement": improvements[-1] if improvements else None,
        "threshold": threshold, "required_intervals": patience,
    }


def stage_local_step(total_step, stage, map_total_updates):
    if isinstance(total_step, bool) or not isinstance(total_step, int) or total_step < 0:
        raise ValueError("total_step must be a nonnegative integer")
    if stage == "map":
        return total_step
    if stage == "render":
        if isinstance(map_total_updates, bool) or not isinstance(map_total_updates, int):
            raise ValueError("render step requires the actual map boundary")
        local = total_step - map_total_updates
        if local < 0:
            raise ValueError("render total_step precedes actual map boundary")
        return local
    raise ValueError(f"unknown extension stage {stage!r}")




def _initial_stage_position(settings, route):
    training = settings["training"]
    if route == "extend_map_then_fresh_render":
        stage, total_step = "map", training["map_updates"]
    elif route in ("continue_pilot_render", "already_converged"):
        stage = "render"
        total_step = training["map_updates"] + training["render_updates"]
    else:
        raise ValueError(f"unknown extension route {route!r}")
    return stage, stage_local_step(total_step, stage, training["map_updates"]), total_step
def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_evaluation_history(path):
    history = _json(path)
    if not isinstance(history, list):
        raise ValueError(f"evaluation history must be a JSON list: {path}")
    return history


def _execution_identity(execution):
    return {key: execution[key] for key in (
        "execution_config_version", "config_source", "file_identities", "settings",
    )}


def _source_identities():
    package = Path(__file__).resolve().parent
    return [file_identity(package / name) for name in (
        "extend.py", "extension_controller.py", "extension_policy.json",
    )]


def _load_parent_checkpoint(path, seed, step, execution, environment):
    before = file_identity(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if before != file_identity(path):
        raise ValueError(f"parent checkpoint changed while reading: {path}")
    if (state.get("seed") != seed or state.get("arm_id") != "baseline"
            or state.get("step") != step or state.get("environment") != environment
            or not probe_run._identity_matches(state.get("execution_identity", {}), execution)):
        raise ValueError(f"parent checkpoint identity mismatch: {path}")
    return state, before


def _validated_train_log(path, endpoint, map_updates, *, allow_suffix=False):
    lines = Path(path).read_bytes().splitlines(keepends=True)
    wrong_length = len(lines) < endpoint if allow_suffix else len(lines) != endpoint
    if wrong_length:
        raise ValueError(f"incomplete parent training log: {path}")
    prefix = lines[:endpoint]
    if any(not row.endswith(b"\n") for row in prefix):
        raise ValueError(f"incomplete parent training log: {path}")
    for index, line in enumerate(prefix, 1):
        row = json.loads(line)
        stage = "map" if index <= map_updates else "render"
        local = index if stage == "map" else index - map_updates
        if (row.get("total_step"), row.get("step"), row.get("stage")) != (index, local, stage):
            raise ValueError(f"parent log has invalid step {index}: {path}")
    return prefix

def _build_parent_snapshot(parent_dir, policy):
    parent_dir = Path(parent_dir).resolve(strict=True)
    parent_status = _json(parent_dir / "status.json")
    if parent_status.get("status") != "complete":
        raise RuntimeError("parent baseline run is not complete")
    if parent_status.get("process_identity") is None or process_identity_matches(
            parent_status.get("pid"), parent_status.get("process_identity")):
        raise RuntimeError("parent baseline process is still live or cannot be verified")
    selection = _json(parent_dir / "run_selection.json")
    expected_selection = {"selection_version": 1, "scope": "selected", "selected_arm_ids": ["baseline"]}
    if selection != expected_selection:
        raise ValueError("parent run must be the explicitly selected baseline-only run")
    execution = load_execution_config(parent_dir)
    settings = execution["settings"]
    training = settings["training"]
    if (training["seeds"] != policy["seeds"]
            or training["map_updates"] != 1000
            or training["render_updates"] != 300
            or training["evaluation_interval"] != policy["evaluation_interval"]
            or training["checkpoint_interval"] != policy["checkpoint_interval"]
            or settings["sampling"]["sample_order_seed_offset"] != policy["sample_order_seed_offset"]
            or "baseline" not in settings["model"]["expected_parameters"]):
        raise ValueError("parent settings disagree with the approved extension policy")
    source_config, source_path = load_config(execution["config_source"]["path"])
    candidate = build_execution_config(source_config, source_path, parent_dir)
    if not same_execution_config(execution, candidate):
        raise ValueError("parent source or data identity changed since baseline training")
    environment = _json(parent_dir / "runtime.json").get("environment")
    if not isinstance(environment, dict) or not environment:
        raise ValueError("parent runtime identity is missing")
    archive_status = _json(parent_dir / "stage_archive_status.json")
    if archive_status.get("status") != "complete":
        raise RuntimeError("parent map-stage checkpoint archive is not complete")
    base_map, base_render = training["map_updates"], training["render_updates"]
    base_total = base_map + base_render
    saved_archives = {str(Path(item).resolve()) for item in archive_status.get("saved", [])}
    parent_files = {
        "root": {name: file_identity(parent_dir / name) for name in (
            "config.json", "runtime.json", "run_selection.json", "stage_archive_status.json",
        )},
        "seeds": {},
    }
    plans = {}
    for seed in policy["seeds"]:
        run_dir = parent_dir / f"seed{seed}_baseline"
        seed_status = _json(run_dir / "status.json")
        complete = _json(run_dir / "complete.json")
        history = _read_evaluation_history(run_dir / "evaluation.json")
        if (seed_status.get("status") != "complete" or seed_status.get("arm") != "baseline"
                or complete.get("arm_id") != "baseline" or complete.get("updates") != base_total
                or not isinstance(history, list)):
            raise ValueError(f"baseline seed {seed} is not complete")
        map_plateau = plateau_decision(history, "map", policy)
        render_plateau = plateau_decision(history, "render", policy)
        if map_plateau["observations"] < 1 or render_plateau["observations"] < 1:
            raise ValueError(f"baseline seed {seed} lacks completed stage evaluations")
        archive = run_dir / "map_stage_end.pt"
        last = run_dir / "last.pt"
        if str(archive.resolve()) not in saved_archives:
            raise ValueError(f"stage archive status omits {archive}")
        map_state, map_identity = _load_parent_checkpoint(
            archive, seed, base_map, execution, environment
        )
        final_state, final_identity = _load_parent_checkpoint(
            last, seed, base_total, execution, environment
        )
        map_history = [row for row in history if row.get("stage") == "map"]
        if map_state.get("history") != map_history or final_state.get("history") != history:
            raise ValueError(f"checkpoint/evaluation history mismatch for seed {seed}")
        _validated_train_log(run_dir / "train.jsonl", base_total, base_map)
        route = (
            "already_converged" if map_plateau["status"] == "plateau" and render_plateau["status"] == "plateau"
            else "continue_pilot_render" if map_plateau["status"] == "plateau"
            else "extend_map_then_fresh_render"
        )
        start = last if route != "extend_map_then_fresh_render" else archive
        start_identity = final_identity if start == last else map_identity
        plans[str(seed)] = {
            "seed": seed, "arm_id": "baseline", "route": route,
            "initial_map_plateau": map_plateau,
            "initial_render_plateau": render_plateau,
            "starting_checkpoint": start_identity,
            "starting_checkpoint_role": "pilot_final_render" if start == last else "pilot_map_boundary",
        }
        parent_files["seeds"][str(seed)] = {
            "run_directory": str(run_dir.resolve()),
            "files": {
                name: file_identity(run_dir / name) for name in (
                    "status.json", "complete.json", "evaluation.json", "validation.json", "train.jsonl",
                )
            } | {"last.pt": final_identity, "map_stage_end.pt": map_identity},
        }
        del map_state, final_state
    snapshot = {
        "extension_version": EXTENSION_VERSION,
        "parent_directory": str(parent_dir),
        "parent_execution_identity": _execution_identity(execution),
        "parent_runtime": environment,
        "parent_selection": selection,
        "parent_files": parent_files,
        "source_identities": _source_identities(),
        "policy": policy,
        "plans": plans,
    }
    return execution, snapshot


def _ensure_manifest(parent_dir, output_dir, policy):
    execution, snapshot = _build_parent_snapshot(parent_dir, policy)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "extension_manifest.json"
    policy_copy = output_dir / "extension_policy.json"
    if manifest_path.is_file():
        if _json(manifest_path) != snapshot or _json(policy_copy) != policy:
            raise ValueError("immutable extension manifest or policy mismatch")
        return execution, snapshot, file_identity(manifest_path)
    allowed_logs = {
        f"viewer_seed{seed}.{kind}.log"
        for seed in policy["seeds"] for kind in ("stdout", "stderr")
    } | {"controller.stdout.log", "controller.stderr.log"}
    seed_dirs = {f"seed{seed}_baseline" for seed in policy["seeds"]}
    for item in output_dir.iterdir():
        if item.name in ({"controller_status.json", "launch.json"} | allowed_logs):
            continue
        if item.is_dir() and item.name in seed_dirs and not any(item.iterdir()):
            continue
        raise FileExistsError(f"extension output already contains unowned data: {item}")
    if policy_copy.exists() and _json(policy_copy) != policy:
        raise ValueError("existing extension policy copy differs from checked-in policy")
    atomic_write_json(policy_copy, policy)
    atomic_write_json(manifest_path, snapshot)
    return execution, snapshot, file_identity(manifest_path)


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def _write_history(run_dir, history, settings, actual_map_updates):
    atomic_write_json(run_dir / "evaluation.json", history)
    viewer_settings = dict(settings)
    viewer_settings["training"] = dict(settings["training"])
    viewer_settings["training"]["map_updates"] = actual_map_updates
    atomic_write_json(run_dir / "validation.json", probe_run._validation_history(history, viewer_settings))



def build_sample_order(seed, settings, updates):
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("updates must be a nonnegative integer")
    rng = random.Random(seed + settings["sampling"]["sample_order_seed_offset"])
    count = len(settings["dataset"]["categories"]) * settings["dataset"]["materials_per_category"] * len(settings["dataset"]["train_crops"])
    batch_size = settings["sampling"]["batch_size"]
    return [rng.sample(range(count), batch_size) for _ in range(updates)]


def _copy_parent_prefix(parent_log, destination, endpoint, map_updates):
    rows = _validated_train_log(parent_log, endpoint, map_updates, allow_suffix=True)
    _atomic_bytes(destination, b"".join(rows))

def _checkpoint_payload(*, seed, step, stage, actual_map_updates, model, optimizer,
                        history, execution, environment, manifest_identity):
    return {
        "extension_checkpoint_version": 1,
        "seed": seed, "arm_id": "baseline", "step": step, "stage": stage,
        "actual_map_updates": actual_map_updates,
        "net_g": model.net_g.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "history": history,
        "execution_identity": _execution_identity(execution),
        "environment": environment,
        "extension_manifest_identity": manifest_identity,
        "rng_state": {
            "python": random.getstate(),
            "numpy": __import__("numpy").random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _restore_rng_state(state):
    import numpy as np
    rng = state.get("rng_state")
    if not isinstance(rng, dict):
        raise ValueError("extension checkpoint has no random-generator state")
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint requires CUDA random-generator state")
        torch.cuda.set_rng_state_all(rng["cuda"])


def _save_checkpoint(run_dir, **values):
    payload = _checkpoint_payload(**values)
    probe_run._atomic_torch_save(Path(run_dir) / "last.pt", payload)
    return payload


def _load_extension_checkpoint(path, *, seed, execution, environment, manifest_identity):
    before = file_identity(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if before != file_identity(path):
        raise ValueError(f"extension checkpoint changed while reading: {path}")
    if (state.get("extension_checkpoint_version") != 1 or state.get("seed") != seed
            or state.get("arm_id") != "baseline" or state.get("step", -1) < 0
            or state.get("environment") != environment
            or state.get("extension_manifest_identity") != manifest_identity
            or not probe_run._identity_matches(state.get("execution_identity", {}), execution)):
        raise ValueError(f"extension checkpoint identity mismatch: {path}")
    return state


def _seed_status(*, status, run_dir, seed, stage, total_step, committed,
                 map_total, render_updates, total_limit, map_limit, invocation, error=None):
    local = stage_local_step(total_step, stage, map_total)
    result = {
        "status": status, "pid": invocation["pid"],
        "process_identity": invocation["process_identity"],
        "seed": seed, "arm": "baseline", "label": "baseline", "stage": stage,
        "step": local, "total_step": total_step, "last_committed_step": committed,
        "total_updates": total_limit, "output_directory": str(run_dir),
        "population": "8 fixed heldout materials; probe, not full training",
        "extension": True, "run_scope": "baseline_extension",
        "selected_arm_ids": ["baseline"], "map_total_limit": map_limit,
        "actual_map_updates": map_total, "render_updates": render_updates,
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result

def _verify_extension_log(path, committed_step, map_boundary):
    if not Path(path).is_file():
        raise ValueError(f"missing extension training log: {path}")
    rows = Path(path).read_bytes().splitlines(keepends=True)
    if len(rows) < committed_step:
        raise ValueError(f"training log is shorter than checkpoint: {path}")
    for index, line in enumerate(rows[:committed_step], 1):
        if not line.endswith(b"\n"):
            raise ValueError(f"incomplete committed train row {index}: {path}")
        row = json.loads(line)
        stage = "map" if index <= map_boundary else "render"
        local = index if stage == "map" else index - map_boundary
        if (row.get("total_step"), row.get("step"), row.get("stage")) != (index, local, stage):
            raise ValueError(f"invalid train row {index}: {path}")


def _copy_json(source, destination):
    _atomic_bytes(destination, Path(source).read_bytes())


def _record_noop(*, plan, parent_run, run_dir, invocation, manifest_identity,
                 policy, map_updates, render_updates):
    source = Path(parent_run) / f"seed{plan['seed']}_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "evaluation.json", "validation.json"):
        _copy_json(source / name, run_dir / name)
    original = _json(source / "complete.json")
    complete = dict(original)
    complete["extension"] = {
        "route": "already_converged", "added_map_updates": 0,
        "added_render_updates": 0, "extension_manifest_identity": manifest_identity,
        "reason": "both pilot stages already met the approved plateau rule",
    }
    atomic_write_json(run_dir / "complete.json", complete)
    atomic_write_json(run_dir / "status.json", _seed_status(
        status="complete", run_dir=run_dir, seed=plan["seed"], stage="render",
        total_step=map_updates + render_updates, committed=map_updates + render_updates,
        map_total=map_updates, render_updates=render_updates,
        total_limit=map_updates + policy["render_total_limit"],
        map_limit=policy["map_total_limit"], invocation=invocation,
    ))


def _run_seed(*, execution, snapshot, manifest_identity, invocation, policy, seed,
              output_dir, train_bank, heldout_bank):
    settings = execution["settings"]
    parent_dir = Path(snapshot["parent_directory"])
    plan = snapshot["plans"][str(seed)]
    run_dir = Path(output_dir) / f"seed{seed}_baseline"
    run_dir.mkdir(parents=True, exist_ok=True)
    complete_path = run_dir / "complete.json"
    checkpoint_path = run_dir / "last.pt"
    train_path = run_dir / "train.jsonl"
    if complete_path.is_file():
        complete = _json(complete_path)
        status = _json(run_dir / "status.json")
        if (complete.get("extension", {}).get("extension_manifest_identity") != manifest_identity
                or status.get("status") != "complete"):
            raise ValueError(f"completed extension identity mismatch: {run_dir}")
        return
    if plan["route"] == "already_converged":
        _record_noop(
            plan=plan, parent_run=parent_dir, run_dir=run_dir, invocation=invocation,
            manifest_identity=manifest_identity, policy=policy,
            map_updates=settings["training"]["map_updates"],
            render_updates=settings["training"]["render_updates"],
        )
        return

    seed_dir = parent_dir / f"seed{seed}_baseline"
    base_map = settings["training"]["map_updates"]
    base_render = settings["training"]["render_updates"]
    base_total = base_map + base_render
    state = None
    if checkpoint_path.is_file():
        state = _load_extension_checkpoint(
            checkpoint_path, seed=seed, execution=execution,
            environment=snapshot["parent_runtime"], manifest_identity=manifest_identity,
        )
        start_step = state["step"]
        stage = state["stage"]
        map_total = state["actual_map_updates"]
        history = state["history"]
        probe_run.truncate_log_to_checkpoint(train_path, start_step)
        _verify_extension_log(train_path, start_step, map_total)
    else:
        if plan["route"] == "extend_map_then_fresh_render":
            initial = _load_parent_checkpoint(
                seed_dir / "map_stage_end.pt", seed, base_map, execution,
                snapshot["parent_runtime"],
            )[0]
            start_step, stage, map_total = base_map, "map", base_map
            history = [row for row in initial["history"] if row["stage"] == "map"]
            parent_endpoint = base_map
        else:
            initial = _load_parent_checkpoint(
                seed_dir / "last.pt", seed, base_total, execution,
                snapshot["parent_runtime"],
            )[0]
            start_step, stage, map_total = base_total, "render", base_map
            history = initial["history"]
            parent_endpoint = base_total
        if train_path.exists():
            probe_run.truncate_log_to_checkpoint(train_path, parent_endpoint)
            _verify_extension_log(train_path, parent_endpoint, base_map)
        else:
            _copy_parent_prefix(seed_dir / "train.jsonl", train_path, parent_endpoint, base_map)
        state = initial

    if not history:
        raise ValueError(f"no checkpoint validation history for seed {seed}")
    _write_history(run_dir, history, settings, map_total)
    probe_run._seed(seed)
    model = probe_run.ProbeModel(settings, "baseline", settings["runtime"]["device"])
    count = parameter_count(model.net_g)
    if count != settings["model"]["expected_parameters"]["baseline"]:
        raise ValueError(f"baseline parameter count changed: {count}")
    model.net_g.load_state_dict(state["net_g"])
    train_rows = execution["resolved_inputs"]["train"]
    heldout_rows = execution["resolved_inputs"]["heldout"]
    populations = {
        "train": cache_slots(model, train_bank, train_rows, settings["rendering"]["f0"]),
        "unseen_crop": cache_slots(
            model, train_bank, execution["resolved_inputs"]["unseen_crop"], settings["rendering"]["f0"]
        ),
        "heldout": cache_slots(model, heldout_bank, heldout_rows, settings["rendering"]["f0"]),
    }
    sample_order = build_sample_order(
        seed, settings, _stage_total_limit(policy, stage, map_total)
    )
    train_lights = make_lights(settings["rendering"]["training_lights"], model.device)
    novel_lights = make_lights(settings["rendering"]["novel_lights"], model.device)
    optimizer = None
    if stage == "map" or (stage == "render" and state.get("optimizer") is not None):
        optimizer = probe_run._optimizer(model, settings, stage)
        optimizer.load_state_dict(state["optimizer"])
    else:
        optimizer = probe_run._optimizer(model, settings, "render")
    if state.get("extension_checkpoint_version"):
        _restore_rng_state(state)

    training = settings["training"]
    interval = policy["evaluation_interval"]
    committed = start_step
    stop_reasons = {}
    started = time.perf_counter()
    atomic_write_json(run_dir / "status.json", _seed_status(
        status="running", run_dir=run_dir, seed=seed, stage=stage,
        total_step=start_step, committed=committed, map_total=map_total,
        render_updates=max(0, start_step - map_total),
        total_limit=_stage_total_limit(policy, stage, map_total),
        map_limit=policy["map_total_limit"], invocation=invocation,
    ))
    atomic_write_json(Path(output_dir) / "status.json", _root_status(
        status="running", output_dir=Path(output_dir), invocation=invocation,
        policy=policy, seed=seed, stage=stage,
        step=stage_local_step(start_step, stage, map_total),
        total_step=start_step, committed=committed,
    ))
    try:
        while True:
            if stage == "map":
                decision = plateau_decision(history, "map", policy)
                if decision["status"] == "plateau":
                    stop_reasons["map"] = {"reason": "plateau", **decision}
                    map_total = start_step
                    stage = "render"
                    optimizer = probe_run._optimizer(model, settings, "render")
                    _save_checkpoint(
                        run_dir, seed=seed, step=start_step, stage=stage,
                        actual_map_updates=map_total, model=model, optimizer=None,
                        history=history, execution=execution,
                        environment=snapshot["parent_runtime"], manifest_identity=manifest_identity,
                    )
                    atomic_write_json(run_dir / "status.json", _seed_status(
                        status="running", run_dir=run_dir, seed=seed, stage=stage,
                        total_step=start_step, committed=committed, map_total=map_total,
                        render_updates=0, total_limit=_stage_total_limit(policy, stage, map_total),
                        map_limit=policy["map_total_limit"], invocation=invocation,
                    ))
                    atomic_write_json(Path(output_dir) / "status.json", _root_status(
                        status="running", output_dir=Path(output_dir), invocation=invocation,
                        policy=policy, seed=seed, stage=stage, step=0,
                        total_step=start_step, committed=committed,
                    ))
                    continue
                if start_step >= policy["map_total_limit"]:
                    stop_reasons["map"] = {"reason": "update_cap", **decision}
                    map_total = start_step
                    stage = "render"
                    optimizer = probe_run._optimizer(model, settings, "render")
                    _save_checkpoint(
                        run_dir, seed=seed, step=start_step, stage=stage,
                        actual_map_updates=map_total, model=model, optimizer=None,
                        history=history, execution=execution,
                        environment=snapshot["parent_runtime"], manifest_identity=manifest_identity,
                    )
                    atomic_write_json(run_dir / "status.json", _seed_status(
                        status="running", run_dir=run_dir, seed=seed, stage=stage,
                        total_step=start_step, committed=committed, map_total=map_total,
                        render_updates=0, total_limit=_stage_total_limit(policy, stage, map_total),
                        map_limit=policy["map_total_limit"], invocation=invocation,
                    ))
                    atomic_write_json(Path(output_dir) / "status.json", _root_status(
                        status="running", output_dir=Path(output_dir), invocation=invocation,
                        policy=policy, seed=seed, stage=stage, step=0,
                        total_step=start_step, committed=committed,
                    ))
                    continue
            else:
                decision = plateau_decision(history, "render", policy)
                render_step = start_step - map_total
                if decision["status"] == "plateau":
                    stop_reasons["render"] = {"reason": "plateau", **decision}
                    break
                if render_step >= policy["render_total_limit"]:
                    stop_reasons["render"] = {"reason": "update_cap", **decision}
                    break
            if stage == "map" and optimizer is None:
                optimizer = probe_run._optimizer(model, settings, "map")
            safe, valid, features = tensors(populations["train"], sample_order[start_step])
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            pred = unit_prediction(model.net_g(features))
            map_loss, parts = ndr_loss(pred, safe, valid)
            render_loss = (
                render_error(model, pred, safe, valid, train_lights, settings["rendering"])
                if stage == "render" else map_loss.new_zeros(())
            )
            loss = map_loss + training["render_loss_weight"] * render_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("invalid extension objective")
            loss.backward()
            diagnostics = [
                {"gate": float(mixer.gate.detach()),
                 "relative_residual": float(mixer.last_relative_residual),
                 "fc_grad_norm": float(mixer.fc[0].weight.grad.norm())}
                for mixer in mixers(model.net_g)
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.net_g.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError("zero or nonfinite extension gradient")
            optimizer.step()
            total_step = start_step + 1
            map_total = total_step if stage == "map" else map_total
            local_step = total_step if stage == "map" else total_step - map_total
            row = {
                "total_step": total_step, "step": local_step, "stage": stage,
                "map_loss": float(map_loss.detach()), "normal_loss": float(parts[0].detach()),
                "diffuse_loss": float(parts[1].detach()), "roughness_loss": float(parts[2].detach()),
                "render_loss": float(render_loss.detach()), "loss": float(loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "preclip_grad_norm": float(gradient_norm), "mixers": diagnostics,
                "seconds": time.perf_counter() - tick,
            }
            probe_run._append_train_row(train_path, row)
            at_cap = (
                total_step == policy["map_total_limit"] if stage == "map"
                else total_step - map_total == policy["render_total_limit"]
            )
            evaluate_now = total_step % interval == 0 or at_cap
            if evaluate_now:
                history.append(probe_run._evaluate_entry(
                    total_step, stage, model, populations, novel_lights, settings
                ))
                _write_history(run_dir, history, settings, map_total)
                metric = history[-1]["populations"]["heldout"]["mean"]
                print(seed, stage, total_step, metric["ndr_mean"], metric["render_l1"], flush=True)
            checkpoint_due = total_step % policy["checkpoint_interval"] == 0 or at_cap
            if checkpoint_due:
                payload = _save_checkpoint(
                    run_dir, seed=seed, step=total_step, stage=stage,
                    actual_map_updates=map_total, model=model, optimizer=optimizer,
                    history=history, execution=execution,
                    environment=snapshot["parent_runtime"], manifest_identity=manifest_identity,
                )
                committed = total_step
            else:
                payload = None
            start_step = total_step
            atomic_write_json(run_dir / "status.json", _seed_status(
                status="running", run_dir=run_dir, seed=seed, stage=stage,
                total_step=total_step, committed=committed, map_total=map_total,
                render_updates=max(0, total_step - map_total),
                total_limit=_stage_total_limit(policy, stage, map_total),
                map_limit=policy["map_total_limit"], invocation=invocation,
            ))
            atomic_write_json(Path(output_dir) / "status.json", _root_status(
                status="running", output_dir=Path(output_dir), invocation=invocation,
                policy=policy, seed=seed, stage=stage, step=local_step, total_step=total_step, committed=committed,
            ))
            if evaluate_now:
                decision = plateau_decision(history, stage, policy)
            if payload is not None:
                del payload

        arrays_dir = run_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        final = probe_run.evaluate(model, populations, novel_lights, settings, save_dir=arrays_dir)
        atomic_write_json(complete_path, {
            "arm_id": "baseline", "label": "baseline", "parameters": count,
            "final": final, "mixer_disabled": None, "updates": start_step,
            "seconds": time.perf_counter() - started,
            "extension": {
                "route": plan["route"], "actual_map_updates": map_total,
                "render_updates": start_step - map_total, "stop_reasons": stop_reasons,
                "initial_map_plateau": plan["initial_map_plateau"],
                "initial_render_plateau": plan["initial_render_plateau"],
                "extension_manifest_identity": manifest_identity,
            },
        })
        atomic_write_json(run_dir / "status.json", _seed_status(
            status="complete", run_dir=run_dir, seed=seed, stage="render",
            total_step=start_step, committed=committed, map_total=map_total,
            render_updates=start_step - map_total,
            total_limit=_stage_total_limit(policy, stage, map_total),
            map_limit=policy["map_total_limit"], invocation=invocation,
        ))
    except BaseException as exc:
        failed_stage = stage
        try:
            atomic_write_json(run_dir / "status.json", _seed_status(
                status="failed", run_dir=run_dir, seed=seed, stage=failed_stage,
                total_step=start_step, committed=committed, map_total=map_total,
                render_updates=max(0, start_step - map_total),
                total_limit=_stage_total_limit(policy, stage, map_total),
                map_limit=policy["map_total_limit"], invocation=invocation, error=exc,
            ))
        finally:
            raise
    finally:
        del model, populations, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _assert_parent_snapshot(snapshot):
    parent_status = _json(Path(snapshot["parent_directory"]) / "status.json")
    if (parent_status.get("status") != "complete"
            or parent_status.get("process_identity") is None
            or process_identity_matches(parent_status.get("pid"), parent_status.get("process_identity"))):
        raise RuntimeError("parent baseline is not quiescent")
    execution = load_execution_config(snapshot["parent_directory"])
    config, path = load_config(execution["config_source"]["path"])
    candidate = build_execution_config(config, path, snapshot["parent_directory"])
    if not same_execution_config(execution, candidate):
        raise ValueError("parent source or data identity changed during extension")
    if _source_identities() != snapshot["source_identities"]:
        raise ValueError("extension source identity changed during extension")
    for name, identity in snapshot["parent_files"]["root"].items():
        if file_identity(Path(snapshot["parent_directory"]) / name) != identity:
            raise ValueError(f"parent run metadata changed during extension: {name}")
    for seed_row in snapshot["parent_files"]["seeds"].values():
        run_dir = Path(seed_row["run_directory"])
        for name, identity in seed_row["files"].items():
            if file_identity(run_dir / name) != identity:
                raise ValueError(f"parent seed artifact changed during extension: {run_dir / name}")



def _root_status(*, status, output_dir, invocation, policy, stage=None, seed=None,
                 step=0, total_step=0, committed=0, error=None):
    result = {
        "status": status, "pid": invocation["pid"] if invocation else None,
        "process_identity": invocation["process_identity"] if invocation else None,
        "stage": stage, "seed": seed, "step": step, "total_step": total_step,
        "last_committed_step": committed, "output_directory": str(output_dir),
        "extension": True, "run_scope": "baseline_extension",
        "selected_arm_ids": ["baseline"], "scheduled_runs": len(policy["seeds"]),
        "comparison_expected_runs": len(policy["seeds"]) * 3,
        "comparison_complete": False, "comparison_status": "partial_baseline_extension",
        "map_total_limit": policy["map_total_limit"],
        "render_total_limit": policy["render_total_limit"],
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result
def run_extension(parent_dir=None, output_dir=None, policy_path=POLICY_PATH):
    policy = load_policy(policy_path)
    parent_dir = resolve_repo_path(parent_dir or policy["parent_run_directory"])
    output_dir = resolve_repo_path(output_dir or policy["output_directory"])
    execution, snapshot, manifest_identity = _ensure_manifest(parent_dir, output_dir, policy)
    _assert_parent_snapshot(snapshot)
    settings = execution["settings"]
    environment = probe_run._runtime_environment(settings)
    if environment != snapshot["parent_runtime"]:
        raise ValueError("runtime environment differs from the completed baseline run")
    probe_run._set_deterministic_settings(settings)
    status_path = output_dir / "status.json"
    if status_path.is_file():
        previous = _json(status_path)
        if (previous.get("status") == "running"
                and process_identity_matches(previous.get("pid"), previous.get("process_identity"))):
            raise RuntimeError(f"baseline extension is already active as PID {previous.get('pid')}")
    invocation = probe_run._record_invocation(output_dir, environment)
    atomic_write_json(output_dir / "status.json", _root_status(
        status="running", output_dir=output_dir, invocation=invocation, policy=policy,
    ))
    train_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
    heldout_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
    try:
        for seed in policy["seeds"]:
            _assert_parent_snapshot(snapshot)
            route = snapshot["plans"][str(seed)]["route"]
            stage, local_step, initial_step = _initial_stage_position(settings, route)
            atomic_write_json(output_dir / "status.json", _root_status(
                status="running", output_dir=output_dir, invocation=invocation,
                policy=policy, seed=seed, stage=stage, step=local_step,
                total_step=initial_step, committed=initial_step,
            ))
            _run_seed(
                execution=execution, snapshot=snapshot,
                manifest_identity=manifest_identity, invocation=invocation,
                policy=policy, seed=seed, output_dir=output_dir,
                train_bank=train_bank, heldout_bank=heldout_bank,
            )
        atomic_write_json(output_dir / "status.json", _root_status(
            status="complete", output_dir=output_dir, invocation=invocation, policy=policy,
        ))
    except BaseException as exc:
        current = _json(output_dir / "status.json")
        seed = current.get("seed")
        if seed in policy["seeds"]:
            seed_dir = output_dir / f"seed{seed}_baseline"
            seed_status_path = seed_dir / "status.json"
            if seed_status_path.is_file():
                seed_status = _json(seed_status_path)
                if seed_status.get("status") == "running":
                    seed_status["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    seed_status["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    atomic_write_json(seed_status_path, seed_status)
        atomic_write_json(output_dir / "status.json", _root_status(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            output_dir=output_dir, invocation=invocation, policy=policy,
            seed=seed, stage=current.get("stage"), step=current.get("step", 0),
            total_step=current.get("total_step", 0),
            committed=current.get("last_committed_step", 0), error=exc,
        ))
        raise
    finally:
        del train_bank, heldout_bank
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output_dir

def _stage_total_limit(policy, stage, map_total):
    map_budget = policy["map_total_limit"] if stage == "map" else map_total
    return map_budget + policy["render_total_limit"]