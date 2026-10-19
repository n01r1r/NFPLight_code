"""Runner for the bounded D4/dropout probe; importing starts no work."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
from types import SimpleNamespace

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from model.relation_geometry import RELATION_VERSION
import torch.nn as nn

from data.fabric import pack_svbrdf
from experiments.bottleneck_fc.pipeline import (
    Bank, cache_slots, fixed_f0, make_lights, ndr_loss, prepare_features,
    render_error, render_input_images_batched, tensors, unit_prediction,
)
from experiments.regularization.config import (
    atomic_write_json, file_identity, load_config, load_execution_config,
    prepare_run, verify_baseline_extension,
)
from experiments.regularization import config as experiment_config
from model.nfplight_model import NFPLightModel
from network.nfplight_net import NAFBlock, TwoBranchNet


def effective_settings(execution):
    """Build runtime settings from this package's frozen local config."""
    experiment = execution["experiment_settings"]
    scientific = experiment["scientific_settings"]
    training = experiment["training"]
    return {
        "dataset": scientific["dataset"],
        "model": {
            "backbone": scientific["backbone"],
            "parameters": scientific["baseline_parameters"],
        },
        "sampling": scientific["sampling"],
        "training": {
            **scientific["optimization"],
            **scientific["evaluation"],
            "seeds": training["seeds"],
            "map_updates": training["map_cap"],
            "render_updates": training["render_cap"],
            "consecutive_intervals": training["consecutive_intervals"],
            "relative_improvement_threshold": training["relative_improvement_threshold"],
        },
        "loss": scientific["loss"],
        "rendering": scientific["rendering"],
        "runtime": scientific["runtime"],
        "experiment": experiment,
        "scientific": scientific,
    }


def create_network(dropout_probability, backbone):
    if dropout_probability not in (0.0, 0.1):
        raise ValueError("regularization probe only supports dropout p=0 or p=0.1")
    network = TwoBranchNet(
        width=backbone["width"],
        middle_blk_num=backbone["middle_blocks"],
        enc_blk_nums=list(backbone["encoder_blocks"]),
        dec_blk_nums=list(backbone["decoder_blocks"]),
    )
    if dropout_probability:
        for block in network.modules():
            if isinstance(block, NAFBlock):
                block.dropout1 = nn.Dropout(dropout_probability)
                block.dropout2 = nn.Dropout(dropout_probability)
    return network


class RegularizationModel(NFPLightModel):
    def __init__(self, settings, dropout_probability, device):
        self.model_settings = settings["model"]
        self.dropout_probability = dropout_probability
        rendering = settings["rendering"]
        super().__init__(SimpleNamespace(
            device=device,
            precision=rendering["precision"],
            feature_version=rendering["feature_version"],
            capture_precision=True,
        ))

    def init_network(self):
        self.net_g = self.model_to_device(create_network(
            self.dropout_probability, self.model_settings["backbone"]
        ))


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def transform_target_d4(target, transform):
    """Transform packed raw GT before canonicalization/rendering; normal XY uses pack_svbrdf."""
    if isinstance(transform, bool) or not isinstance(transform, int) or not 0 <= transform < 8:
        raise ValueError("D4 transform index must be an integer in [0,7]")
    was_batch = target.ndim == 4
    values = target if was_batch else target.unsqueeze(0)
    if values.ndim != 4 or values.shape[1] != 10:
        raise ValueError("D4 targets must have shape [10,H,W] or [B,10,H,W]")
    rotation, hflip = transform % 4, transform >= 4
    transformed = []
    for row in values:
        image = row.detach().to(device="cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
        packed = pack_svbrdf(
            image[..., :3].copy(),
            ((image[..., 3:6] + 1.0) * 0.5).clip(0, 1),
            ((image[..., 6:7] + 1.0) * 0.5).clip(0, 1),
            ((image[..., 7:10] + 1.0) * 0.5).clip(0, 1),
            rot=rotation,
            hflip=hflip,
        )
        transformed.append(packed)
    result = torch.stack(transformed)
    return result if was_batch else result[0]


def augment_targets(targets, transforms):
    if len(targets) != len(transforms):
        raise ValueError("one D4 transform is required for each target")
    return torch.stack([
        transform_target_d4(target, transform)
        for target, transform in zip(targets, transforms)
    ])


def sample_order(seed, settings):
    sampling = settings["sampling"]
    total = settings["training"]["map_updates"] + settings["training"]["render_updates"]
    rng = random.Random(seed + sampling["sample_order_seed_offset"])
    data = settings["dataset"]
    sample_count = len(data["categories"]) * data["materials_per_category"] * len(data["train_crops"])
    return [rng.sample(range(sample_count), sampling["batch_size"]) for _ in range(total)]


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _set_deterministic_settings(settings):
    torch.set_num_threads(settings["runtime"]["torch_num_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = settings["runtime"]["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = settings["runtime"]["cudnn_deterministic"]


def _runtime_environment(settings):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required only when run is explicitly invoked")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "device": settings["runtime"]["device"],
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "precision": settings["rendering"]["precision"],
        "amp": settings["rendering"]["amp"],
        "tf32": settings["rendering"]["tf32"],
    }


def _record_invocation(output_dir, environment):
    from experiments.bottleneck_fc.config import atomic_write_json as write_json
    from experiments.bottleneck_fc.config import process_identity

    path = output_dir / "runtime.json"
    if path.is_file():
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
    write_json(path, saved)
    return invocation


def _optimizer(model, settings, stage):
    training = settings["training"]
    lr = training["map_learning_rate"] if stage == "map" else training["render_learning_rate"]
    return torch.optim.Adam(
        model.net_g.parameters(), lr=lr, betas=tuple(training["adam_betas"])
    )


def _append_train_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _truncate_log(path, committed_step):
    path = Path(path)
    if not path.exists():
        return
    keep, last = [], 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            break
        try:
            row = json.loads(line)
            step = row["total_step"]
            if isinstance(step, bool) or not isinstance(step, int) or step <= last or step > committed_step:
                break
        except (ValueError, KeyError, TypeError):
            break
        keep.append(line)
        last = step
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(keep), encoding="utf-8", newline="")
    experiment_config.replace_with_retry(temporary, path)


def _atomic_torch_save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    experiment_config.replace_with_retry(temporary, path)


def _evaluation(model, populations, novel_lights, settings):
    model.net_g.eval()
    result = {}
    for label, rows in populations.items():
        material_metrics = []
        for row in rows:
            target, valid, features = (row[key] for key in ("safe", "valid", "features"))
            pred = unit_prediction(model.net_g(features))
            map_loss, parts = ndr_loss(pred, target, valid)
            angle = torch.rad2deg(torch.acos(
                (pred[:, :3] * target[:, :3]).sum(1, keepdim=True).clamp(-1, 1)
            )).masked_select(valid).mean()
            relighting = render_error(model, pred, target, valid, novel_lights, settings["rendering"])
            near_pred = render_input_images_batched(model, fixed_f0(pred, settings["rendering"]["f0"]))[:, :3]
            near_gt = render_input_images_batched(model, target)[:, :3]
            near_l1 = (near_pred - near_gt).abs().masked_select(valid.expand_as(near_pred)).mean()
            material_metrics.append({
                "material_id": row["material_id"], "category": row["category"], "crop": row["crop"],
                "ndr_mean": float(map_loss), "map_normal": float(parts[0]),
                "map_diffuse": float(parts[1]), "map_roughness": float(parts[2]),
                "normal_degrees": float(angle), "render_l1": float(relighting),
                "near_l1": float(near_l1),
            })
        keys = ("ndr_mean", "map_normal", "map_diffuse", "map_roughness",
                "normal_degrees", "render_l1", "near_l1")
        means = {key: float(np.mean([row[key] for row in material_metrics])) for key in keys}
        if not np.isfinite(list(means.values())).all():
            raise FloatingPointError(f"non-finite evaluation for {label}")
        result[label] = {"mean": means, "materials": material_metrics}
    model.net_g.train()
    return result


def _history_entry(total_step, stage, stage_step, model, populations, novel_lights, settings):
    return {
        "total_step": total_step, "stage": stage, "step": stage_step,
        "populations": _evaluation(model, populations, novel_lights, settings),
    }


def _write_history(run_dir, history, settings):
    atomic_write_json(run_dir / "evaluation.json", history)
    categories = settings["dataset"]["categories"]
    converted = []
    for entry in history:
        population = entry["populations"]["heldout"]
        rows = population["materials"]
        by_category = {}
        for category in categories:
            subset = [row for row in rows if row["category"] == category]
            if not subset:
                raise ValueError(f"heldout evaluation lacks category {category}")
            by_category[category] = {
                "materials": len(subset),
                **{key: float(np.mean([row[key] for row in subset]))
                   for key in population["mean"]},
            }
        converted.append({
            "total_step": entry["total_step"], "step": entry["step"], "stage": entry["stage"],
            **population["mean"], "category": by_category, "materials": rows,
            "population": "8 fixed heldout materials; diagnostic probe",
        })
    atomic_write_json(run_dir / "validation.json", {"history": converted})

def _plateau_update(state, value, settings):
    if not math.isfinite(value) or value < 0:
        raise FloatingPointError("plateau metric must be finite and nonnegative")
    previous = state.get("previous")
    if previous is None:
        state.update(previous=value, streak=0, best=value)
        return False
    improvement = (previous - value) / max(abs(previous), 1e-12)
    streak = state["streak"] + 1 if improvement < settings["training"]["relative_improvement_threshold"] else 0
    state.update(
        previous=value, streak=streak, best=min(state.get("best", value), value),
        latest_relative_improvement=improvement,
    )
    return streak >= settings["training"]["consecutive_intervals"]


def _status_document(*, status, run_dir, invocation, seed, condition, stage,
                     stage_step, total_step, last_committed_step, settings, error=None):
    row = {
        "status": status,
        "pid": invocation["pid"] if invocation else None,
        "process_identity": invocation["process_identity"] if invocation else None,
        "seed": seed, "arm": condition["id"], "condition": condition["id"],
        "label": condition["label"], "d4": condition["d4"],
        "dropout_probability": settings["experiment"]["design"]["dropout_probability"] if condition["dropout"] else 0.0,
        "stage": stage, "step": stage_step, "total_step": total_step,
        "last_committed_step": last_committed_step,
        "total_updates": settings["training"]["map_updates"] + settings["training"]["render_updates"],
        "output_directory": str(run_dir),
        "run_scope": "three new conditions; baseline extension is reference-only",
    }
    if error is not None:
        row["error"] = {"type": type(error).__name__, "message": str(error)}
    return row


def _root_status(output_dir, *, status, invocation=None, seed=None, condition=None,
                 stage=None, step=0, total_step=0, last_committed_step=0,
                 comparison_complete=False, error=None):
    result = {
        "status": status,
        "pid": invocation["pid"] if invocation else None,
        "process_identity": invocation["process_identity"] if invocation else None,
        "seed": seed, "condition": condition, "stage": stage, "step": step,
        "total_step": total_step, "last_committed_step": last_committed_step,
        "output_directory": str(output_dir),
        "run_scope": "d4_only,dropout_only,d4_dropout; baseline extension reference-only",
        "selected_condition_ids": ["d4_only", "dropout_only", "d4_dropout"],
        "scheduled_training_runs": 9, "comparison_expected_runs": 12,
        "comparison_complete": comparison_complete,
        "training_started": status in ("running", "failed", "complete"),
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _write_run_status(run_dir, output_dir, invocation, seed, condition, settings, *,
                      status, stage, stage_step, total_step, last_committed_step, error=None):
    row = _status_document(
        status=status, run_dir=run_dir, invocation=invocation, seed=seed, condition=condition,
        stage=stage, stage_step=stage_step, total_step=total_step,
        last_committed_step=last_committed_step, settings=settings, error=error,
    )
    atomic_write_json(run_dir / "status.json", row)
    atomic_write_json(output_dir / "status.json", {
        **row, "output_directory": str(output_dir),
        "scheduled_training_runs": 9, "comparison_expected_runs": 12,
        "comparison_complete": False,
        "selected_condition_ids": ["d4_only", "dropout_only", "d4_dropout"],
    })
    return row


def _execution_identity(execution, baseline_reference):
    return {
        "execution_config_version": execution["execution_config_version"],
        "file_identities": execution["file_identities"],
        "experiment_config_source": execution["experiment_config_source"],
        "baseline_execution_identity": execution["baseline_execution_identity"],
        "baseline_reference_file_identities": baseline_reference["file_identities"],
        "settings": execution["experiment_settings"],
    }


def _heldout_metric(entry, stage):
    mean = entry["populations"]["heldout"]["mean"]
    return mean["ndr_mean"] if stage == "map" else mean["render_l1"]


def _condition_batch(model, condition, indices, train_population, raw_train, rng, settings):
    if condition["d4"]:
        choices = [rng.randrange(8) for _ in indices]
        targets = [raw_train[index] for index in indices]
        batch = augment_targets(targets, choices)
        safe, valid, features = prepare_features(model, batch, settings["rendering"]["f0"])
        return safe, valid, features, choices
    safe, valid, features = tensors(train_population, indices)
    return safe, valid, features, [0 for _ in indices]


def _run_condition(*, output_dir, execution, settings, invocation, seed, condition,
                   train_population, raw_train, populations, train_lights, novel_lights,
                   model, resume, identity, batch_provider=None, evaluator=None):
    """One resumable arm: Adam, sample prefix, augmentation RNG and torch RNG are committed."""
    run_dir = output_dir / f"seed{seed}_{condition['id']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "last.pt"
    train_path = run_dir / "train.jsonl"
    complete_path = run_dir / "complete.json"
    training = settings["training"]
    interval = settings["scientific"]["evaluation"]["evaluation_interval"]
    checkpoint_interval = settings["scientific"]["evaluation"]["checkpoint_interval"]
    if complete_path.is_file():
        if not resume:
            raise FileExistsError(f"condition already completed; pass --resume: {run_dir}")
        return json.loads(complete_path.read_text(encoding="utf-8"))
    if not resume and any(path.exists() for path in (
        checkpoint_path, train_path, run_dir / "evaluation.json"
    )):
        raise FileExistsError(f"partial condition exists; pass --resume: {run_dir}")

    order = sample_order(seed, settings)
    aug_rng = random.Random(seed + execution["experiment_settings"]["training"]["d4_seed_offset"])
    state = None
    history = []
    map_updates = render_updates = 0
    map_stopped = False
    map_plateau, render_plateau, stop_reasons = {}, {}, {}
    optimizer = None
    last_committed_step = 0
    if resume and checkpoint_path.is_file():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if state.get("relation_version") != RELATION_VERSION:
            raise ValueError("checkpoint relation_version differs; start a new training lineage")
        if state.get("condition") != condition["id"] or state.get("seed") != seed or state.get("identity") != identity:
            raise ValueError(f"checkpoint identity mismatch: {checkpoint_path}")
        model.net_g.load_state_dict(state["net_g"])
        map_updates, render_updates = state["map_updates"], state["render_updates"]
        map_stopped = state["map_stopped"]
        map_plateau, render_plateau = state["map_plateau"], state["render_plateau"]
        stop_reasons, history = state["stop_reasons"], state["history"]
        aug_rng.setstate(state["augmentation_rng"])
        torch.random.set_rng_state(state["torch_rng"])
        if model.device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        last_committed_step = map_updates + render_updates
        if state.get("optimizer") is not None:
            optimizer = _optimizer(model, settings, state["stage"])
            optimizer.load_state_dict(state["optimizer"])
        _truncate_log(train_path, last_committed_step)
        _write_history(run_dir, history, settings)
    elif resume:
        _truncate_log(train_path, 0)

    evaluator = evaluator or (
        lambda current, total, stage, local: _history_entry(
            total, stage, local, current, populations, novel_lights, settings
        )
    )
    if not history:
        initial = evaluator(model, 0, "map", 0)
        history.append(initial)
        first_metric = _heldout_metric(initial, "map")
        map_plateau = {"previous": first_metric, "streak": 0, "best": first_metric}
        _write_history(run_dir, history, settings)
    if map_stopped and not render_plateau:
        render_plateau = {"previous": None, "streak": 0}

    total_step = map_updates + render_updates
    _write_run_status(
        run_dir, output_dir, invocation, seed, condition, settings, status="running",
        stage="render" if map_stopped else "map",
        stage_step=render_updates if map_stopped else map_updates,
        total_step=total_step, last_committed_step=last_committed_step,
    )
    started = time.perf_counter()
    try:
        while render_updates < training["render_updates"] and "render" not in stop_reasons:
            stage = "map" if not map_stopped and map_updates < training["map_updates"] else "render"
            if stage == "render" and not map_stopped:
                map_stopped = True
                stop_reasons["map"] = "update_cap"
                render_plateau = {"previous": None, "streak": 0}
            desired_lr = training["map_learning_rate"] if stage == "map" else training["render_learning_rate"]
            if optimizer is None or optimizer.param_groups[0]["lr"] != desired_lr:
                optimizer = _optimizer(model, settings, stage)

            indices = order[map_updates + render_updates]
            if batch_provider is None:
                safe, valid, features, transform_ids = _condition_batch(
                    model, condition, indices, train_population, raw_train, aug_rng, settings
                )
            else:
                safe, valid, features, transform_ids = batch_provider(indices, aug_rng, stage, model)
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction = unit_prediction(model.net_g(features))
            map_loss, parts = ndr_loss(prediction, safe, valid)
            render_loss = (
                render_error(model, prediction, safe, valid, train_lights, settings["rendering"])
                if stage == "render" else map_loss.new_zeros(())
            )
            loss = map_loss + training["render_loss_weight"] * render_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite objective")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.net_g.parameters(), training["gradient_clip_norm"], error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
                raise FloatingPointError("zero or non-finite parameter gradient")
            optimizer.step()

            if stage == "map":
                map_updates += 1
                local_step = map_updates
            else:
                render_updates += 1
                local_step = render_updates
            total_step = map_updates + render_updates
            _append_train_row(train_path, {
                "total_step": total_step, "step": local_step, "stage": stage,
                "map_loss": float(map_loss.detach()), "normal_loss": float(parts[0].detach()),
                "diffuse_loss": float(parts[1].detach()), "roughness_loss": float(parts[2].detach()),
                "render_loss": float(render_loss.detach()), "loss": float(loss.detach()),
                "lr": optimizer.param_groups[0]["lr"],
                "preclip_grad_norm": float(gradient_norm),
                "d4_transforms": transform_ids,
                "dropout_probability": execution["experiment_settings"]["design"]["dropout_probability"] if condition["dropout"] else 0.0,
                "seconds": time.perf_counter() - tick,
            })

            cap = training["map_updates"] if stage == "map" else training["render_updates"]
            stage_finished = local_step >= cap
            evaluation_due = local_step % interval == 0 or stage_finished
            plateau_stopped = False
            if evaluation_due:
                entry = evaluator(model, total_step, stage, local_step)
                history.append(entry)
                _write_history(run_dir, history, settings)
                metric = _heldout_metric(entry, stage)
                plateau = map_plateau if stage == "map" else render_plateau
                plateau_stopped = _plateau_update(plateau, metric, settings)
                if plateau_stopped:
                    stop_reasons[stage] = "plateau"
                elif stage_finished:
                    stop_reasons[stage] = "update_cap"
            stage_finished = stage_finished or plateau_stopped
            if stage == "map" and stage_finished:
                map_stopped = True
                optimizer = None
                render_plateau = {"previous": None, "streak": 0}

            checkpoint_due = local_step % checkpoint_interval == 0 or stage_finished
            if checkpoint_due:
                checkpoint_stage = "render" if stage == "map" and stage_finished else stage
                device = getattr(model, "device", torch.device("cpu"))
                committed = {
                    "relation_version": RELATION_VERSION, "checkpoint_version": 1, "condition": condition["id"], "seed": seed,
                    "stage": checkpoint_stage, "map_updates": map_updates,
                    "render_updates": render_updates, "map_stopped": map_stopped,
                    "map_plateau": map_plateau, "render_plateau": render_plateau,
                    "stop_reasons": stop_reasons, "net_g": model.net_g.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "history": history, "augmentation_rng": aug_rng.getstate(),
                    "torch_rng": torch.random.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "identity": identity,
                }
                _atomic_torch_save(checkpoint_path, committed)
                last_committed_step = total_step

            _write_run_status(
                run_dir, output_dir, invocation, seed, condition, settings, status="running",
                stage="render" if stage == "map" and stage_finished else stage,
                stage_step=0 if stage == "map" and stage_finished else local_step,
                total_step=total_step, last_committed_step=last_committed_step,
            )
            if stage == "render" and stage_finished:
                break

        final_step = map_updates + render_updates
        final_entry = evaluator(model, final_step, "render", render_updates)
        complete = {
            "condition": condition["id"], "label": condition["label"], "seed": seed,
            "parameters": parameter_count(model.net_g), "map_updates": map_updates,
            "render_updates": render_updates, "total_updates": final_step,
            "map_stop_reason": stop_reasons.get("map", "unknown"),
            "render_stop_reason": stop_reasons.get("render", "unknown"),
            "map_plateau": map_plateau, "render_plateau": render_plateau,
            "final": final_entry["populations"],
            "elapsed_seconds": time.perf_counter() - started,
            "comparison_claim": "same fixed material bank diagnostic; not generalization evidence",
        }
        atomic_write_json(complete_path, complete)
        _write_run_status(
            run_dir, output_dir, invocation, seed, condition, settings, status="complete",
            stage="render", stage_step=render_updates, total_step=final_step,
            last_committed_step=last_committed_step,
        )
        return complete
    except BaseException as error:
        total_step = map_updates + render_updates
        stage = "map" if not map_stopped else "render"
        local_step = map_updates if stage == "map" else render_updates
        try:
            _write_run_status(
                run_dir, output_dir, invocation, seed, condition, settings, status="failed",
                stage=stage, stage_step=local_step, total_step=total_step,
                last_committed_step=last_committed_step, error=error,
            )
        except OSError:
            pass
        raise

def _condition_contract(experiment, condition_id):
    for row in experiment["design"]["conditions"]:
        if row["id"] == condition_id:
            if row["source"] != "train":
                raise ValueError("baseline is a reference condition and is never trained here")
            labels = {
                "d4_only": "D4 only",
                "dropout_only": "Dropout p=0.1 only",
                "d4_dropout": "D4 + dropout p=0.1",
            }
            return {**row, "label": labels[row["id"]]}
    raise ValueError(f"unknown condition: {condition_id}")


def _validate_run_state(output_dir, resume):
    path = output_dir / "status.json"
    status = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"status": "prepared"}
    state = status.get("status")
    if state == "complete":
        if resume:
            return status, "complete"
        raise RuntimeError(f"run is complete: {output_dir}")
    if state == "running":
        from experiments.bottleneck_fc.config import process_identity_matches
        if process_identity_matches(status.get("pid"), status.get("process_identity")):
            raise RuntimeError(f"run is active as PID {status.get('pid')}")
        if not resume:
            raise RuntimeError("stale running status; pass --resume explicitly")
    elif state in ("failed", "interrupted"):
        if not resume:
            raise RuntimeError(f"run status is {state}; pass --resume")
    elif state == "prepared":
        if resume:
            raise RuntimeError("cannot resume a prepared run that has not started")
    else:
        raise ValueError(f"unknown run status {state!r}")
    return status, state


def run_experiment(config_path=None, output_dir=None, *, resume=False):
    from train_fabric import require_scheduler_ready
    require_scheduler_ready(historical_resume=resume)
    config, config_path = load_config(config_path)
    output_dir = prepare_run(config_path, output_dir)
    execution = load_execution_config(output_dir)
    candidate = experiment_config.build_execution_config(config, config_path, output_dir)
    if not experiment_config.same_execution_config(execution, candidate):
        raise ValueError("immutable local config/source/data identity changed")

    # This gate runs before CUDA checks, model construction, or any training output.
    reference = verify_baseline_extension(execution, output_dir)
    current, state = _validate_run_state(output_dir, resume)
    if state == "complete":
        return output_dir

    settings = effective_settings(execution)
    environment = _runtime_environment(settings)
    invocation = _record_invocation(output_dir, environment)
    _set_deterministic_settings(settings)
    if resume:
        saved_environment = json.loads((output_dir / "runtime.json").read_text(encoding="utf-8"))
        if saved_environment.get("environment") != environment:
            raise ValueError("runtime environment changed; strict resume is not allowed")

    identity = _execution_identity(execution, reference)
    atomic_write_json(output_dir / "status.json", _root_status(
        output_dir, status="running", invocation=invocation,
        total_step=current.get("total_step", 0),
        last_committed_step=current.get("last_committed_step", 0),
    ))
    train_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
    heldout_bank = Bank(execution["resolved_inputs"]["manifest"], settings["dataset"])
    active = {"seed": None, "condition": None, "stage": None, "step": 0, "committed": 0}
    final_total = 0
    try:
        for seed in settings["training"]["seeds"]:
            for condition_id in ("d4_only", "dropout_only", "d4_dropout"):
                condition = _condition_contract(config, condition_id)
                active.update(seed=seed, condition=condition_id, stage="map", step=0, committed=0)
                run_dir = output_dir / f"seed{seed}_{condition_id}"
                if (run_dir / "complete.json").is_file() and resume:
                    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
                    final_total = complete["total_updates"]
                    continue

                _seed(seed)
                dropout = config["design"]["dropout_probability"] if condition["dropout"] else 0.0
                model = RegularizationModel(settings, dropout, settings["runtime"]["device"])
                if parameter_count(model.net_g) != settings["model"]["parameters"]:
                    raise ValueError("condition changed the frozen 7,609,834 parameter baseline")
                populations = {
                    "train": cache_slots(model, train_bank, execution["resolved_inputs"]["train"], settings["rendering"]["f0"]),
                    "unseen_crop": cache_slots(model, train_bank, execution["resolved_inputs"]["unseen_crop"], settings["rendering"]["f0"]),
                    "heldout": cache_slots(model, heldout_bank, execution["resolved_inputs"]["heldout"], settings["rendering"]["f0"]),
                }
                raw_train = (
                    [train_bank.crop(slot) for slot in execution["resolved_inputs"]["train"]]
                    if condition["d4"] else None
                )
                train_lights = make_lights(settings["rendering"]["training_lights"], model.device)
                novel_lights = make_lights(settings["rendering"]["novel_lights"], model.device)
                try:
                    result = _run_condition(
                        output_dir=output_dir, execution=execution, settings=settings,
                        invocation=invocation, seed=seed, condition=condition,
                        train_population=populations["train"], raw_train=raw_train,
                        populations=populations, train_lights=train_lights,
                        novel_lights=novel_lights, model=model, resume=resume,
                        identity=identity,
                    )
                    final_total = result["total_updates"]
                    active.update(
                        stage="render", step=result["render_updates"],
                        committed=result["total_updates"],
                    )
                finally:
                    del model, populations, raw_train
                    if settings["runtime"]["device"].startswith("cuda"):
                        torch.cuda.empty_cache()

        atomic_write_json(output_dir / "status.json", _root_status(
            output_dir, status="complete", invocation=invocation,
            seed=settings["training"]["seeds"][-1], step=final_total,
            total_step=final_total, last_committed_step=final_total,
            comparison_complete=True,
        ))
    except BaseException as error:
        try:
            atomic_write_json(output_dir / "status.json", _root_status(
                output_dir, status="failed", invocation=invocation,
                seed=active["seed"], condition=active["condition"], stage=active["stage"],
                step=active["step"], total_step=active["committed"],
                last_committed_step=active["committed"], error=error,
            ))
        except OSError:
            pass
        raise
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="regularization package config")
    parser.add_argument("--out", type=Path, help="output path; relative paths use repository root")
    parser.add_argument("--resume", action="store_true", help="resume strict identity-matched checkpoints")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.out, resume=args.resume)


if __name__ == "__main__":
    main()
