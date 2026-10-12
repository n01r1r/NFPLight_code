"""Bounded-memory, injectable training state machine with safe exact resume."""
from __future__ import annotations
from contextlib import contextmanager
import copy
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.regularization.config import atomic_write_json, replace_with_retry
from experiments.regularization.run import _append_train_row, _truncate_log
from experiments.full_dataset_suite.reporting import write_history as _write_history
from experiments.full_dataset_suite.storage import require_disk_reserve, DiskReserveError


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    return value


def capture_rng():
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": [numpy[0], numpy[1].tolist(), *numpy[2:]],
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(saved):
    random.setstate(saved["python"])
    numpy = saved["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.random.set_rng_state(saved["torch"])
    if saved["cuda"]:
        torch.cuda.set_rng_state_all(saved["cuda"])


@contextmanager
def neutral_evaluation(network):
    saved = capture_rng()
    modes = [(module, module.training) for module in network.modules()]
    network.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(saved)


def atomic_save(path, value, *, rotate=False):
    path = Path(path)
    require_disk_reserve(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(cpu_tree(value), stream)
        stream.flush()
        os.fsync(stream.fileno())
    if rotate and path.exists():
        replace_with_retry(path, path.with_name("previous.pt"))
    replace_with_retry(temporary, path)


def load_checkpoint(path):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("checkpoint_version") != 1 or state.get("package") != "full_dataset_suite":
        raise ValueError("only new full_dataset_suite checkpoints are accepted")
    return state


def metric_values(populations):
    # Validity applies to every population/material, not only the monitor mean.
    def check(value):
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, (float, int)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                raise FloatingPointError("evaluation metrics must be finite and nonnegative")
    check(populations)
    mean = populations["heldout"]["mean"]
    return float(mean["ndrs_sum"]), float(mean["render_l1"])


def convergence_update(record, metric, *, lr_before, lr_after, floor, threshold):
    if not math.isfinite(metric) or metric < 0:
        raise FloatingPointError("invalid plateau metric")
    meaningful = record["meaningful_best"]
    improved = meaningful is None or metric < meaningful * (1 - threshold)
    if improved:
        record["meaningful_best"] = metric
        record["floor_bad_observations"] = 0
    at_floor = lr_after <= floor + 1e-15
    if not at_floor:
        record["floor_bad_observations"] = 0
    elif not record["floor_reached"]:
        record["floor_bad_observations"] = 0
    elif lr_before <= floor + 1e-15 and not improved:
        record["floor_bad_observations"] += 1
    record["floor_reached"] = at_floor
    return improved


class Engine:
    """Callbacks return scalar objective/metadata and full-population NDRS-sum evaluation."""
    def __init__(self, output_dir, network, policy, identity, objective, evaluator,
                 *, sample_count=16, batch_size=4, adam_betas=(0.9, 0.999), clip_norm=1.0, augmentation_rng=None, noise_generator=None):
        self.output = Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.net = network
        self.policy = copy.deepcopy(policy)
        self.identity = identity
        self.objective = objective
        self.evaluator = evaluator
        if not 1 <= batch_size <= sample_count:
            raise ValueError("batch size must be within sample population")
        self.sample_count, self.batch_size = sample_count, batch_size
        self.betas, self.clip_norm = adam_betas, clip_norm
        self.samples = random.Random(policy["seed"] + 10000)
        self.d4 = random.Random(policy["seed"] + 20000)
        self.augmentation_rng = augmentation_rng or random.Random(policy["seed"] + 30000)
        self.noise_generator = noise_generator or torch.Generator(device="cpu").manual_seed(policy["seed"] + 40000)
        self.ema = cpu_tree(network.state_dict()) if policy.get("ema_decay") is not None else None
        self.after_step = None
        self.state = {"seen_slots": [], "stage": "map", "stage_step": 0, "total_step": 0,
                      "map_updates": 0, "render_updates": 0, "status": "running",
                      "stop_reasons": {}, "best_map": None, "best_render": None,
                      "history": [], "evaluations": {"map": 0, "render": 0},
                      "convergence": {"map": self._new_convergence(), "render": self._new_convergence()}}
        self.seen_slots = set()
        self.memory_guard = None
        self.started = time.perf_counter()
        self._new_optimizer("map")

    @staticmethod
    def _new_convergence():
        return {"meaningful_best": None, "floor_bad_observations": 0, "floor_reached": False}

    def _new_optimizer(self, stage):
        optimizer_class = torch.optim.AdamW if self.policy.get("optimizer") == "AdamW" else torch.optim.Adam
        self.optimizer = optimizer_class(self.net.parameters(), lr=self.policy[stage + "_lr"], betas=self.betas, weight_decay=self.policy.get("weight_decay", 0.0))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=self.policy["mode"], factor=self.policy["factor"],
            patience=self.policy["patience"], threshold=self.policy["threshold"],
            threshold_mode=self.policy["threshold_mode"], cooldown=self.policy["cooldown"],
            min_lr=self.policy[stage + "_min_lr"], eps=self.policy["eps"])

    def snapshot(self):
        self.state["seen_slots"] = sorted(self.seen_slots)
        return cpu_tree({"checkpoint_version": 1, "package": "full_dataset_suite",
                         "identity": self.identity, "policy": self.policy,
                         "state": self.state, "net_g": self.net.state_dict(),
                         "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                         "rng": capture_rng(), "sample_rng": self.samples.getstate(), "d4_rng": self.d4.getstate(),
                         "augmentation_rng": self.augmentation_rng.getstate(), "noise_rng": self.noise_generator.get_state(), "ema": self.ema})

    def restore(self, checkpoint):
        if checkpoint["identity"] != self.identity or checkpoint["policy"] != self.policy:
            raise ValueError("checkpoint scientific/source/input identity mismatch")
        self.state = copy.deepcopy(checkpoint["state"])
        self.seen_slots = set(self.state["seen_slots"])
        self.net.load_state_dict(checkpoint["net_g"])
        self._new_optimizer(self.state["stage"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.samples.setstate(checkpoint["sample_rng"])
        self.d4.setstate(checkpoint["d4_rng"])
        self.augmentation_rng.setstate(checkpoint["augmentation_rng"])
        self.noise_generator.set_state(checkpoint["noise_rng"])
        self.ema = cpu_tree(checkpoint["ema"])
        restore_rng(checkpoint["rng"])

    def _write_status(self, status=None, error=None):
        if status is not None:
            self.state["status"] = status
        row = {key: self.state[key] for key in ("status", "stage", "stage_step", "total_step", "map_updates", "render_updates", "stop_reasons", "evaluations", "convergence")}
        row.update(pid=os.getpid(), sample_count=self.sample_count, unique_train_slots_seen=len(self.seen_slots),
                   population_exposure=self.state["total_step"] * self.batch_size / self.sample_count,
                   step=self.state["stage_step"], last_committed_step=self.committed_step,
                   lr=self.optimizer.param_groups[0]["lr"], arm=self.policy.get("arm"), seed=self.policy["seed"], parameters=self.policy["parameters"], elapsed_seconds=time.perf_counter() - self.started, elapsed_seconds_this_invocation=time.perf_counter() - self.started)
        if error is not None:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
        atomic_write_json(self.output / "status.json", row)

    def _write_views(self):
        atomic_write_json(self.output / "evaluation.json", self.state["history"])
        if "scientific_settings" in self.identity:
            _write_history(self.output, self.state["history"], {"dataset": self.identity["scientific_settings"]["dataset"]})
        for key in ("best_map", "best_render"):
            best = self.state[key]
            if best is not None:
                atomic_save(self.output / (key + ".pt"), {"checkpoint_version": 1, "package": "full_dataset_suite",
                            "identity": self.identity, **best})
        selected = self.state.get("selected")
        if selected:
            best = self.state[selected]
            atomic_save(self.output / "selected.pt", {"checkpoint_version": 1, "package": "full_dataset_suite",
                        "identity": self.identity, "selection": selected, **best})

    def commit(self):
        atomic_save(self.output / "last.pt", self.snapshot(), rotate=True)
        self.committed_step = self.state["total_step"]
        self._write_views()
        self._write_status()

    def update_ema(self):
        if self.ema is not None:
            decay = self.policy["ema_decay"]
            with torch.no_grad():
                for key, value in self.net.state_dict().items():
                    current = value.detach().cpu()
                    if current.is_floating_point():
                        self.ema[key].mul_(decay).add_(current, alpha=1 - decay)
                    else:
                        self.ema[key].copy_(current)

    def evaluate(self, *, select=True):
        stage = self.state["stage"]
        online = cpu_tree(self.net.state_dict()) if self.ema is not None else None
        with neutral_evaluation(self.net):
            online_populations = self.evaluator(self.net, stage, self.state["stage_step"])
            metric_values(online_populations)
            if self.ema is None:
                populations = online_populations
            else:
                try:
                    self.net.load_state_dict(self.ema)
                    populations = self.evaluator(self.net, stage, self.state["stage_step"])
                finally:
                    self.net.load_state_dict(online)
        ndr, render = metric_values(populations)
        selected_weights = self.ema if self.ema is not None else self.net.state_dict()
        entry = {"stage": stage, "step": self.state["stage_step"], "total_step": self.state["total_step"], "populations": populations,
                 "evaluation_weights": "ema" if self.ema is not None else "online"}
        if self.ema is not None:
            entry["online_populations"] = online_populations
        if select:
            key = "best_map" if stage == "map" else "best_render"
            value = ndr if stage == "map" else render
            eligible = stage == "map" or ndr <= self.policy["render_ndr_guard"] * self.state["best_map"]["ndrs_sum"]
            entry["eligible_for_selection"] = eligible
            best = self.state[key]
            if eligible and (best is None or value < best["metric"]):
                self.state[key] = {"net_g": cpu_tree(selected_weights), "metric": value,
                                   "ndrs_sum": ndr, "render_l1": render,
                                   "stage": stage, "step": self.state["stage_step"], "total_step": self.state["total_step"],
                                   "evaluation_weights": "ema" if self.ema is not None else "online"}
        metric = ndr if stage == "map" else render
        lr_before = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step(metric)
        lr_after = self.optimizer.param_groups[0]["lr"]
        convergence = self.state["convergence"][stage]
        meaningful = convergence_update(convergence, metric, lr_before=lr_before, lr_after=lr_after,
            floor=self.policy[stage + "_min_lr"], threshold=self.policy["threshold"])
        self.state["evaluations"][stage] += 1
        entry.update(monitor=metric, meaningful_improvement=meaningful, lr_before=lr_before, lr=lr_after,
                     scheduler={key: copy.deepcopy(value) for key, value in self.scheduler.state_dict().items() if key != "mode_worse"}, convergence=copy.deepcopy(convergence))
        self.state["history"].append(entry)
        return convergence["floor_bad_observations"] >= self.policy["floor_observations"]

    def transition(self):
        # Preserve the actual last map optimizer/RNG before restoring selected weights.
        self.state["stop_reasons"]["map"] = "floor_no_meaningful_improvement"
        atomic_save(self.output / "map_stage_end.pt", self.snapshot())
        self.net.load_state_dict(self.state["best_map"]["net_g"])
        if self.ema is not None:
            self.ema = cpu_tree(self.net.state_dict())
        self.state.update(stage="render", stage_step=0)
        self.state["render_start_sample_offset"] = self.state["map_updates"]
        self._new_optimizer("render")
        self.evaluate(select=True)
        self.commit()

    def finish(self):
        self.state["stop_reasons"]["render"] = "floor_no_meaningful_improvement"
        self.state["selected"] = "best_render" if self.state["best_render"] is not None else "best_map"
        self.state["status"] = "complete"
        # last retains resumable model/Adam, selected.pt contains validation-selected weights.
        self.commit()
        self._write_completed_artifacts()

    def _write_completed_artifacts(self):
        atomic_save(self.output / "render_stage_end.pt", self.snapshot())
        selected = self.state[self.state["selected"]]
        atomic_write_json(self.output / "complete.json", {
            "arm": self.policy.get("arm"), "seed": self.policy["seed"], "parameters": self.policy["parameters"],
            "evaluation_weights": selected.get("evaluation_weights", "online"),
            "selected": self.state["selected"], "selection_step": selected["step"],
            "selection_stage": selected["stage"], "initial_map_baseline_selected": self.state["selected"] == "best_map" or selected["step"] == 0, "ndrs_sum": selected["ndrs_sum"],
            "render_l1": selected["render_l1"], "map_updates": self.state["map_updates"],
            "render_updates": self.state["render_updates"], "stop_reasons": self.state["stop_reasons"],
            "claim_scope": "full train566/val164; single-seed screen; operational stopping rule, not convergence proof"})

    def run(self, caps, *, resume=False, interrupt_after=None):
        self.committed_step = 0
        last_path = self.output / "last.pt"
        previous = self.output / "previous.pt"
        if resume:
            self.restore(load_checkpoint(last_path if last_path.exists() else previous))
            self.committed_step = self.state["total_step"]
            _truncate_log(self.output / "train.jsonl", self.committed_step)
            self._write_views()
            if self.state["status"] == "complete":
                self._write_completed_artifacts()
                self._write_status()
                return self.state
            self.state["status"] = "running"
        elif last_path.exists() or previous.exists():
            raise FileExistsError("existing checkpoint requires --resume")
        else:
            self.evaluate()
            self.commit()
        try:
            while True:
                stage = self.state["stage"]
                cap = caps[stage + "_cap"]
                if self.state["stage_step"] >= cap:
                    self._write_status("budget_paused")
                    self.commit()
                    return self.state
                if self.memory_guard is not None and self.memory_guard():
                    self.restore(load_checkpoint(last_path if last_path.exists() else previous))
                    self.committed_step = self.state["total_step"]
                    _truncate_log(self.output / "train.jsonl", self.committed_step)
                    self._write_views()
                    self._write_status("memory_paused")
                    return self.state
                self.net.train()
                indices = self.samples.sample(range(self.sample_count), self.batch_size)
                transforms = [self.d4.randrange(8) for _ in indices]
                self.optimizer.zero_grad(set_to_none=True)
                tick = time.perf_counter()
                loss, metrics = self.objective(self.net, indices, transforms, stage)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise FloatingPointError("objective must be a finite scalar")
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_norm, error_if_nonfinite=True)
                if not torch.isfinite(gradient) or gradient <= 0:
                    raise FloatingPointError("gradient must be finite and nonzero")
                self.optimizer.step()
                self.seen_slots.update(indices)
                self.update_ema()
                if self.after_step is not None:
                    self.after_step(self)
                self.state["stage_step"] += 1
                self.state["total_step"] += 1
                self.state[stage + "_updates"] += 1
                row = {"stage": stage, "step": self.state["stage_step"], "total_step": self.state["total_step"],
                       "indices": indices, "d4_transforms": transforms, "lr": self.optimizer.param_groups[0]["lr"],
                       "loss": float(loss.detach()), "gradient_norm": float(gradient), "preclip_grad_norm": float(gradient), "seconds": time.perf_counter() - tick, **metrics}
                # Serialize here so a callback cannot retain live GPU tensors in logs/history.
                json.dumps(row, allow_nan=False)
                _append_train_row(self.output / "train.jsonl", row)
                del loss
                evaluated = self.state["stage_step"] % self.policy["evaluation_interval"] == 0
                stopped = self.evaluate() if evaluated else False
                if stopped:
                    if stage == "map":
                        self.transition()
                    else:
                        self.finish()
                        return self.state
                elif self.state["stage_step"] % self.policy["checkpoint_interval"] == 0 or self.state["stage_step"] >= cap:
                    self.commit()
                self._write_status()
                if interrupt_after is not None and self.state["total_step"] >= interrupt_after:
                    self._write_status("interrupted")
                    self.commit()
                    return self.state
        except DiskReserveError as error:
            if last_path.exists():
                self.restore(load_checkpoint(last_path))
                self.committed_step = self.state["total_step"]
                _truncate_log(self.output / "train.jsonl", self.committed_step)
            self._write_status("disk_paused", error)
            return self.state
        except BaseException as error:
            self._write_status("failed", error)
            raise
