"""Paired FP32 bottleneck probe runner.

Run with python -m experiments.bottleneck_fc.run or by absolute script path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
import platform
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bottleneck_fc.config import (
    atomic_write_json,
    build_execution_config,
    load_config,
    load_execution_config,
    prepare_run,
    process_identity,
    process_identity_matches,
    replace_with_retry,
    same_execution_config,
)
from experiments.bottleneck_fc.models import create_network, mixers, parameter_count
from experiments.bottleneck_fc.pipeline import (
    Bank,
    cache_slots,
    fixed_f0,
    make_lights,
    ndr_loss,
    render_error,
    render_input_images_batched,
    tensors,
    unit_prediction,
)

import numpy as np
import torch
from model.relation_geometry import RELATION_VERSION

from model.nfplight_model import (
    CURRENT_RAW_FEATURE_VERSION,
    INPUT_FAR,
    INPUT_GAIN,
    INPUT_NEAR,
    NFPLightModel,
)


class ProbeModel(NFPLightModel):
    """Fresh TwoBranchNet with the selected probe operator."""

    def __init__(self, settings, arm_id, device):
        self.model_config = settings["model"]
        self.arm_id = arm_id
        rendering = settings["rendering"]
        args = SimpleNamespace(
            device=device,
            precision=rendering["precision"],
            feature_version=rendering["feature_version"],
            capture_precision=True,
        )
        super().__init__(args)

    def init_network(self):
        self.net_g = self.model_to_device(create_network(self.arm_id, self.model_config))


def _verify_low_level_contract(settings):
    rendering = settings["rendering"]
    observed = (
        rendering["feature_version"],
        rendering["near_distance"],
        rendering["far_distance"],
        rendering["gain"],
    )
    expected = (CURRENT_RAW_FEATURE_VERSION, INPUT_NEAR, INPUT_FAR, INPUT_GAIN)
    if observed != expected:
        raise ValueError(
            f"configured feature/lighting inputs disagree with NFPLightModel: {observed!r} != {expected!r}"
        )


def _set_deterministic_settings(settings):
    runtime = settings["runtime"]
    torch.set_num_threads(runtime["torch_num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = runtime["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = runtime["cudnn_deterministic"]


def _runtime_environment(settings):
    runtime = settings["runtime"]
    if runtime["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required by this execution config")
        device_name = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
    else:
        device_name, capability = "CPU", None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": runtime["device"],
        "device_name": device_name,
        "device_capability": capability,
        "precision": settings["rendering"]["precision"],
        "amp": settings["rendering"]["amp"],
        "tf32": settings["rendering"]["tf32"],
    }


def _record_invocation(output_dir, environment):
    path = output_dir / "runtime.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("environment") != environment:
            raise ValueError("runtime environment mismatch; strict resume is not allowed")
    else:
        saved = {"runtime_version": 1, "environment": environment, "invocations": []}
    invocation = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
    }
    saved["invocations"].append(invocation)
    atomic_write_json(path, saved)
    return invocation


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _identity_matches(saved, execution_config):
    return (
        saved.get("execution_config_version") == execution_config["execution_config_version"]
        and saved.get("file_identities") == execution_config["file_identities"]
        and saved.get("config_source") == execution_config["config_source"]
        and saved.get("settings") == execution_config["settings"]
    )


def _atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, path)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    replace_with_retry(temporary, path)


def _append_train_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def truncate_log_to_checkpoint(path, total_step):
    """Keep only complete, ordered rows through the last committed checkpoint."""
    path = Path(path)
    if not path.exists():
        return
    retained = []
    last_step = 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        try:
            row = json.loads(line)
            step = row["total_step"]
            if isinstance(step, bool) or not isinstance(step, int) or step <= last_step:
                break
        except (ValueError, KeyError, TypeError):
            break
        if step > total_step:
            break
        retained.append(line)
        last_step = step
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(retained), encoding="utf-8", newline="")
    replace_with_retry(temporary, path)


def _validation_history(history, settings):
    converted = []
    heldout_categories = settings["dataset"]["categories"]
    for entry in history:
        population = entry["populations"]["heldout"]
        rows = population["materials"]
        categories = {}
        for category in heldout_categories:
            subset = [row for row in rows if row["category"] == category]
            if not subset:
                raise ValueError(f"heldout evaluation lacks category {category}")
            categories[category] = {
                "materials": len(subset),
                **{
                    key: float(np.mean([row[key] for row in subset]))
                    for key in population["mean"]
                },
            }
        total_step = entry["total_step"]
        stage = entry["stage"]
        converted.append({
            "total_step": total_step,
            "step": total_step if stage == "map" else max(0, total_step - settings["training"]["map_updates"]),
            "stage": stage,
            **population["mean"],
            "category": categories,
            "materials": rows,
            "population": "8 fixed heldout materials; probe, not full validation",
        })
    return {"history": converted}


def _write_history(run_dir, history, settings):
    atomic_write_json(run_dir / "evaluation.json", history)
    atomic_write_json(
        run_dir / "validation.json",
        _validation_history(history, settings),
    )


@torch.no_grad()
def evaluate(model, populations, novel_lights, settings, *, save_dir=None):
    model.net_g.eval()
    results = {}
    metric_names = (
        "ndr_mean", "map_normal", "map_diffuse", "map_roughness",
        "normal_degrees", "render_l1", "near_l1",
    )
    for label, rows in populations.items():
        metrics = []
        for row in rows:
            target, valid, features = (row[key] for key in ("safe", "valid", "features"))
            pred = unit_prediction(model.net_g(features))
            map_loss, parts = ndr_loss(pred, target, valid)
            normal_angle = torch.rad2deg(torch.acos(
                (pred[:, :3] * target[:, :3]).sum(1, keepdim=True).clamp(-1, 1)
            )).masked_select(valid).mean()
            relighting = render_error(
                model, pred, target, valid, novel_lights, settings["rendering"]
            )
            near_prediction = render_input_images_batched(model, fixed_f0(pred, settings["rendering"]["f0"]))[:, :3]
            near_ground_truth = render_input_images_batched(model, target)[:, :3]
            near_l1 = (
                (near_prediction - near_ground_truth).abs()
                .masked_select(valid.expand_as(near_prediction)).mean()
            )
            record = {
                "material_id": row["material_id"],
                "category": row["category"],
                "crop": row["crop"],
                "ndr_mean": float(map_loss),
                "map_normal": float(parts[0]),
                "map_diffuse": float(parts[1]),
                "map_roughness": float(parts[2]),
                "normal_degrees": float(normal_angle),
                "render_l1": float(relighting),
                "near_l1": float(near_l1),
            }
            metrics.append(record)
            if save_dir is not None:
                safe_material = "".join(
                    char if char.isalnum() or char in "-_." else "_"
                    for char in str(row["material_id"])
                )
                _atomic_npz(
                    save_dir / f"{label}_{safe_material}_{row['crop']}.npz",
                    prediction=pred.cpu().numpy(),
                    target=target.cpu().numpy(),
                    valid=valid.cpu().numpy(),
                    near_prediction=near_prediction.cpu().numpy(),
                    near_gt=near_ground_truth.cpu().numpy(),
                )
        means = {
            key: float(np.mean([row[key] for row in metrics]))
            for key in metric_names
        }
        if not np.isfinite(list(means.values())).all():
            raise FloatingPointError(f"invalid evaluation for {label}")
        results[label] = {"mean": means, "materials": metrics}
    model.net_g.train()
    return results


def _status_document(
    *, settings, status, output_dir, invocation, seed=None, arm_id=None,
    stage=None, total_step=0, last_committed_step=0, total_updates=0, error=None,
    run_selection=None,
):
    local_step = total_step
    if stage == "render":
        local_step = max(0, total_step - settings["training"]["map_updates"])
    result = {
        "status": status,
        "pid": invocation["pid"] if invocation else None,
        "process_identity": invocation["process_identity"] if invocation else None,
        "seed": seed,
        "arm": arm_id,
        "label": None,
        "stage": stage,
        "step": local_step,
        "total_step": total_step,
        "last_committed_step": last_committed_step,
        "total_updates": total_updates,
        "output_directory": str(output_dir),
        "population": "8 fixed heldout materials; probe, not full training",
    }
    if arm_id is not None:
        result["label"] = next(
            arm["label"] for arm in settings["model"]["arms"] if arm["id"] == arm_id
        )
    if run_selection is not None:
        result.update(
            run_scope=run_selection["scope"],
            selected_arm_ids=list(run_selection["selected_arm_ids"]),
            scheduled_runs=(
                len(settings["training"]["seeds"])
                * len(run_selection["selected_arm_ids"])
            ),
            comparison_expected_runs=(
                len(settings["training"]["seeds"])
                * len(settings["model"]["arms"])
            ),
        )
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _ensure_run_selection(output_dir, settings, arm_id, *, resume):
    arm_ids = [arm["id"] for arm in settings["model"]["arms"]]
    if arm_id is not None and arm_id not in arm_ids:
        raise ValueError(f"unknown arm selection {arm_id!r}; expected one of {arm_ids}")
    selection = {
        "selection_version": 1,
        "scope": "all" if arm_id is None else "selected",
        "selected_arm_ids": arm_ids if arm_id is None else [arm_id],
    }
    path = output_dir / "run_selection.json"
    if path.is_file():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid immutable run selection: {path}") from exc
        if saved != selection:
            raise ValueError(
                f"immutable run selection mismatch in {path}: "
                f"saved={saved!r}, requested={selection!r}"
            )
        return saved
    if resume:
        raise ValueError(f"immutable run selection is missing; cannot resume: {path}")
    atomic_write_json(path, selection)
    return selection


def _optimizer(model, settings, stage):
    training = settings["training"]
    learning_rate = (
        training["map_learning_rate"] if stage == "map"
        else training["render_learning_rate"]
    )
    return torch.optim.Adam(
        model.net_g.parameters(),
        lr=learning_rate,
        betas=tuple(training["adam_betas"]),
    )


def _evaluate_entry(step, stage, model, populations, novel_lights, settings):
    return {
        "total_step": step,
        "stage": stage,
        "populations": evaluate(model, populations, novel_lights, settings),
    }


def _run_arm(
    *, output_dir, settings, execution, invocation, seed, arm, train_bank,
    heldout_bank, resume, global_status, run_selection=None,
):
    training = settings["training"]
    sampling = settings["sampling"]
    arm_id = arm["id"]
    total_updates = training["map_updates"] + training["render_updates"]
    run_dir = output_dir / f"seed{seed}_{arm_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    train_path = run_dir / "train.jsonl"
    evaluation_path = run_dir / "evaluation.json"
    complete_path = run_dir / "complete.json"
    if complete_path.exists():
        atomic_write_json(run_dir / "status.json", _status_document(
            settings=settings,
            status="complete",
            output_dir=run_dir,
            invocation=invocation,
            seed=seed,
            arm_id=arm_id,
            stage="render",
            total_step=total_updates,
            last_committed_step=total_updates,
            total_updates=total_updates,
        ))
        return

    if not resume and any(path.exists() for path in (checkpoint_path, train_path, evaluation_path)):
        raise FileExistsError(f"partial run exists; pass --resume: {run_dir}")

    _seed(seed)
    model = ProbeModel(settings, arm_id, settings["runtime"]["device"])
    count = parameter_count(model.net_g)
    expected = settings["model"]["expected_parameters"][arm_id]
    if count != expected:
        raise ValueError(f"{arm_id} parameter count {count} != configured {expected}")
    train_rows = execution["resolved_inputs"]["train"]
    unseen_rows = execution["resolved_inputs"]["unseen_crop"]
    heldout_rows = execution["resolved_inputs"]["heldout"]
    populations = {
        "train": cache_slots(model, train_bank, train_rows, settings["rendering"]["f0"]),
        "unseen_crop": cache_slots(model, train_bank, unseen_rows, settings["rendering"]["f0"]),
        "heldout": cache_slots(model, heldout_bank, heldout_rows, settings["rendering"]["f0"]),
    }
    order_rng = random.Random(seed + sampling["sample_order_seed_offset"])
    sample_order = [
        order_rng.sample(range(len(populations["train"])), sampling["batch_size"])
        for _ in range(total_updates)
    ]
    train_lights = make_lights(settings["rendering"]["training_lights"], model.device)
    novel_lights = make_lights(settings["rendering"]["novel_lights"], model.device)

    state = None
    start_step = 0
    history = []
    if resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("relation_version") != RELATION_VERSION:
            raise ValueError("checkpoint relation_version differs; start a new training lineage")
        if (
            state.get("seed") != seed
            or state.get("arm_id") != arm_id
            or state.get("step") < 0
            or state.get("step") > total_updates
            or state.get("environment") != json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))["environment"]
            or not _identity_matches(state.get("execution_identity", {}), execution)
        ):
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        model.net_g.load_state_dict(state["net_g"])
        start_step = state["step"]
        history = state["history"]
        truncate_log_to_checkpoint(train_path, start_step)
        _write_history(run_dir, history, settings)
    elif resume:
        truncate_log_to_checkpoint(train_path, 0)
        history = []

    if not history:
        history.append(_evaluate_entry(0, "map", model, populations, novel_lights, settings))
        _write_history(run_dir, history, settings)

    optimizer = None
    last_committed_step = start_step
    started = time.perf_counter()
    atomic_write_json(run_dir / "status.json", _status_document(
        settings=settings,
        status="running",
        output_dir=run_dir,
        invocation=invocation,
        seed=seed,
        arm_id=arm_id,
        stage="map" if start_step < training["map_updates"] else "render",
        total_step=start_step,
        last_committed_step=last_committed_step,
        total_updates=total_updates,
    ))
    try:
        for update_index in range(start_step, total_updates):
            stage = "map" if update_index < training["map_updates"] else "render"
            if optimizer is None or update_index == training["map_updates"]:
                optimizer = _optimizer(model, settings, stage)
                if (
                    state is not None
                    and update_index == start_step
                    and update_index != training["map_updates"]
                ):
                    optimizer.load_state_dict(state["optimizer"])
            safe, valid, features = tensors(
                populations["train"], sample_order[update_index]
            )
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
                raise FloatingPointError("invalid objective")
            loss.backward()
            diagnostics = [
                {
                    "gate": float(mixer.gate.detach()),
                    "relative_residual": float(mixer.last_relative_residual),
                    "fc_grad_norm": float(mixer.fc[0].weight.grad.norm()),
                }
                for mixer in mixers(model.net_g)
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.net_g.parameters(),
                training["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError("zero or nonfinite parameter gradient")
            optimizer.step()
            total_step = update_index + 1
            row = {
                "total_step": total_step,
                "step": total_step if stage == "map" else total_step - training["map_updates"],
                "stage": stage,
                "map_loss": float(map_loss.detach()),
                "normal_loss": float(parts[0].detach()),
                "diffuse_loss": float(parts[1].detach()),
                "roughness_loss": float(parts[2].detach()),
                "render_loss": float(render_loss.detach()),
                "loss": float(loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "preclip_grad_norm": float(gradient_norm),
                "mixers": diagnostics,
                "seconds": time.perf_counter() - tick,
            }
            _append_train_row(train_path, row)

            stage_boundary = total_step in (training["map_updates"], total_updates)
            evaluation_due = (
                total_step % training["evaluation_interval"] == 0 or stage_boundary
            )
            if evaluation_due:
                history.append(_evaluate_entry(
                    total_step, stage, model, populations, novel_lights, settings
                ))
                _write_history(run_dir, history, settings)

            checkpoint_due = (
                total_step % training["checkpoint_interval"] == 0 or stage_boundary
            )
            if checkpoint_due:
                checkpoint = {
                    "relation_version": RELATION_VERSION, "checkpoint_version": 1,
                    "seed": seed,
                    "arm_id": arm_id,
                    "step": total_step,
                    "net_g": model.net_g.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "execution_identity": {
                        "execution_config_version": execution["execution_config_version"],
                        "config_source": execution["config_source"],
                        "file_identities": execution["file_identities"],
                        "settings": execution["settings"],
                    },
                    "environment": json.loads(
                        (output_dir / "runtime.json").read_text(encoding="utf-8")
                    )["environment"],
                }
                _atomic_torch_save(checkpoint_path, checkpoint)
                last_committed_step = total_step

            arm_status = _status_document(
                settings=settings,
                status="running",
                output_dir=run_dir,
                invocation=invocation,
                seed=seed,
                arm_id=arm_id,
                stage=stage,
                total_step=total_step,
                last_committed_step=last_committed_step,
                total_updates=total_updates,
            )
            atomic_write_json(run_dir / "status.json", arm_status)
            status_document = _status_document(
                settings=settings,
                status="running",
                output_dir=output_dir,
                invocation=invocation,
                seed=seed,
                arm_id=arm_id,
                stage=stage,
                total_step=total_step,
                last_committed_step=last_committed_step,
                total_updates=total_updates,
                run_selection=run_selection,
            )
            global_status.update(
                seed=seed, arm=arm_id, stage=stage, step=total_step,
                last_committed_step=last_committed_step,
            )
            atomic_write_json(output_dir / "status.json", status_document)
            global_status.update(status_document)

            if evaluation_due:
                print(
                    seed,
                    arm_id,
                    total_step,
                    "train",
                    round(history[-1]["populations"]["train"]["mean"]["ndr_mean"], 6),
                    "heldout",
                    round(history[-1]["populations"]["heldout"]["mean"]["ndr_mean"], 6),
                    flush=True,
                )

        arrays_dir = run_dir / "arrays"
        arrays_dir.mkdir(exist_ok=True)
        final = evaluate(
            model, populations, novel_lights, settings, save_dir=arrays_dir
        )
        ablated = None
        if arm_id != "baseline":
            for mixer in mixers(model.net_g):
                mixer.enabled = False
            ablated = evaluate(model, populations, novel_lights, settings)
        atomic_write_json(complete_path, {
            "arm_id": arm_id,
            "label": arm["label"],
            "parameters": count,
            "final": final,
            "mixer_disabled": ablated,
            "updates": total_updates,
            "seconds": time.perf_counter() - started,
        })
        atomic_write_json(run_dir / "status.json", _status_document(
            settings=settings,
            status="complete",
            output_dir=run_dir,
            invocation=invocation,
            seed=seed,
            arm_id=arm_id,
            stage="render",
            total_step=total_updates,
            last_committed_step=total_updates,
            total_updates=total_updates,
        ))
    finally:
        del model, populations, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _validate_run_state(output_dir, status, resume):
    state = status.get("status", "prepared")
    if state == "complete":
        if resume:
            return "complete"
        raise RuntimeError(f"run is complete: {output_dir}")
    if state == "running":
        if process_identity_matches(status.get("pid"), status.get("process_identity")):
            raise RuntimeError(
                f"run is active as PID {status.get('pid')}; refusing concurrent training"
            )
        if not resume:
            raise RuntimeError(f"stale running status; resume explicitly: {output_dir}")
    elif state in ("failed", "interrupted"):
        if not resume:
            raise RuntimeError(f"run status is {state}; pass --resume: {output_dir}")
    elif state == "prepared":
        if resume:
            raise RuntimeError("cannot resume a prepared run that has not started")
    else:
        raise ValueError(f"unknown run status {state!r} in {output_dir / 'status.json'}")
    return state


def run_experiment(config_path=None, output_dir=None, *, resume=False, arm_id=None):
    from train_fabric import require_scheduler_ready
    require_scheduler_ready(historical_resume=resume)
    config, resolved_config_path = load_config(config_path)
    output_dir = prepare_run(resolved_config_path, output_dir)
    execution = load_execution_config(output_dir)
    candidate = build_execution_config(config, resolved_config_path, output_dir)
    if not same_execution_config(execution, candidate):
        raise ValueError("immutable execution config or source/data identities changed")
    settings = execution["settings"]
    _verify_low_level_contract(settings)
    current_status_path = output_dir / "status.json"
    current_status = json.loads(current_status_path.read_text(encoding="utf-8"))
    state = _validate_run_state(output_dir, current_status, resume)
    run_selection = _ensure_run_selection(output_dir, settings, arm_id, resume=resume)
    if state == "complete":
        print(f"already complete: {output_dir}")
        return output_dir
    total_updates = settings["training"]["map_updates"] + settings["training"]["render_updates"]
    selected_arms = [
        arm for arm in settings["model"]["arms"]
        if arm["id"] in run_selection["selected_arm_ids"]
    ]

    invocation = None
    active = {}
    try:
        environment = _runtime_environment(settings)
        invocation = _record_invocation(output_dir, environment)
        _set_deterministic_settings(settings)
        if resume:
            runtime = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))
            if runtime["environment"] != environment:
                raise ValueError("runtime environment mismatch; strict resume is not allowed")
        atomic_write_json(output_dir / "status.json", _status_document(
            settings=settings,
            status="running",
            output_dir=output_dir,
            invocation=invocation,
            total_updates=total_updates,
            run_selection=run_selection,
        ))
        train_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
        heldout_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
        for seed in settings["training"]["seeds"]:
            for arm in selected_arms:
                active.update(seed=seed, arm=arm["id"], stage="map", step=0, last_committed_step=0)
                atomic_write_json(output_dir / "status.json", _status_document(
                    settings=settings,
                    status="running",
                    output_dir=output_dir,
                    invocation=invocation,
                    seed=seed,
                    arm_id=arm["id"],
                    stage="map",
                    total_step=0,
                    last_committed_step=0,
                    total_updates=total_updates,
                    run_selection=run_selection,
                ))
                _run_arm(
                    output_dir=output_dir,
                    settings=settings,
                    execution=execution,
                    invocation=invocation,
                    seed=seed,
                    arm=arm,
                    train_bank=train_bank,
                    heldout_bank=heldout_bank,
                    resume=resume,
                    global_status=active,
                    run_selection=run_selection,
                )
        atomic_write_json(output_dir / "status.json", _status_document(
            settings=settings,
            status="complete",
            output_dir=output_dir,
            invocation=invocation,
            total_step=total_updates,
            last_committed_step=total_updates,
            total_updates=total_updates,
            run_selection=run_selection,
        ))
    except BaseException as exc:
        if invocation is not None and active.get("arm") is not None:
            failed_run_dir = output_dir / f"seed{active['seed']}_{active['arm']}"
            failed_arm = _status_document(
                settings=settings,
                status="failed",
                output_dir=failed_run_dir,
                invocation=invocation,
                seed=active.get("seed"),
                arm_id=active.get("arm"),
                stage=active.get("stage"),
                total_step=active.get("step", 0),
                last_committed_step=active.get("last_committed_step", 0),
                total_updates=settings["training"]["map_updates"] + settings["training"]["render_updates"],
                error=exc,
            )
            try:
                if failed_run_dir.is_dir():
                    atomic_write_json(failed_run_dir / "status.json", failed_arm)
            except OSError:
                pass
        failed = _status_document(
            settings=settings,
            status="failed",
            output_dir=output_dir,
            invocation=invocation,
            seed=active.get("seed"),
            arm_id=active.get("arm"),
            stage=None,
            total_step=active.get("step", 0),
            last_committed_step=active.get("last_committed_step", 0),
            total_updates=(
                settings["training"]["map_updates"] + settings["training"]["render_updates"]
                if "settings" in locals() else 0
            ),
            error=exc,
            run_selection=run_selection if "run_selection" in locals() else None,
        )
        try:
            atomic_write_json(output_dir / "status.json", failed)
        except OSError:
            pass
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="experiment JSON config")
    parser.add_argument("--out", type=Path, help="run directory; relative paths use repository root")
    parser.add_argument("--resume", action="store_true", help="resume a strict matching run")
    parser.add_argument("--arm", help="run one configured arm; omit to run all three")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.out, resume=args.resume, arm_id=args.arm)


if __name__ == "__main__":
    main()
