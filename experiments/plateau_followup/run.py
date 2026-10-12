"""CLI and FP32 adapter for the approved seed19 7.61M fresh follow-up."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import gc
import traceback
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from experiments.bottleneck_fc.config import process_identity, process_identity_matches
from experiments.bottleneck_fc.pipeline import Bank, make_lights, ndr_loss, prepare_features, render_error, unit_prediction
from experiments.regularization.config import atomic_write_json, file_identity
from experiments.regularization.run import RegularizationModel, augment_targets, parameter_count, _evaluation, _seed, _set_deterministic_settings
from experiments.plateau_followup.config import DEFAULT_OUT, POLICY, budget, prepare, read_json
from experiments.plateau_followup.engine import Engine, atomic_save, load_checkpoint, neutral_evaluation


def settings_from_execution(execution):
    scientific = execution["scientific_settings"]
    return {"dataset": scientific["dataset"], "model": {"backbone": scientific["backbone"], "parameters": POLICY["parameters"]},
            "sampling": scientific["sampling"], "training": scientific["optimization"],
            "rendering": scientific["rendering"], "runtime": scientific["runtime"]}


def memory_observation():
    result = {}
    try:
        import psutil
        memory = psutil.virtual_memory()
        result.update(rss_bytes=psutil.Process().memory_info().rss, available_ram_bytes=memory.available)
    except ImportError:
        result["ram_observation"] = "psutil unavailable"
    if torch.cuda.is_initialized():
        free, total = torch.cuda.mem_get_info()
        result.update(cuda_free_bytes=free, cuda_total_bytes=total,
                      cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved(),
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    return result


def memory_is_low(observation):
    if observation.get("available_ram_bytes", 2**63) < 1024**3:
        return True
    cuda_available = observation.get("cuda_free_bytes", 2**63) + observation.get("cuda_reserved_bytes", 0) - observation.get("cuda_allocated_bytes", 0)
    return cuda_available < 512 * 1024**2


@contextmanager
def process_lock(output, *, resume):
    path = Path(output) / "process_lock.json"
    identity = {"pid": os.getpid(), "process_identity": process_identity(os.getpid())}
    if identity["process_identity"] is None:
        raise RuntimeError("cannot obtain local process creation identity")
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            previous = read_json(path)
            if process_identity_matches(previous.get("pid"), previous.get("process_identity")):
                raise RuntimeError(f"run already active as PID {previous.get('pid')}")
            if not resume:
                raise RuntimeError("stale process lock requires --resume")
            path.unlink()
    else:
        raise RuntimeError("cannot acquire process lock")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(identity, stream)
            stream.flush()
            os.fsync(stream.fileno())
        yield identity
    finally:
        if path.exists() and read_json(path) == identity:
            path.unlink()


class DeviceRows:
    """Only one evaluation slot resides on the accelerator at a time."""
    def __init__(self, rows, device):
        self.rows, self.device = rows, device

    def __iter__(self):
        for row in self.rows:
            yield {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in row.items()}


class Pipeline:
    def __init__(self, execution):
        self.settings = settings_from_execution(execution)
        _set_deterministic_settings(self.settings)
        _seed(POLICY["seed"])
        self.model = RegularizationModel(self.settings, POLICY["dropout_probability"], self.settings["runtime"]["device"])
        if parameter_count(self.model.net_g) != POLICY["parameters"]:
            raise ValueError("frozen baseline parameter count differs")
        self.raw_train = []
        self.populations = {}
        bank = Bank(execution["resolved_inputs"]["manifest"], self.settings["dataset"])
        with neutral_evaluation(self.model.net_g):
            for label in ("train", "unseen_crop", "heldout"):
                rows = []
                for slot in execution["resolved_inputs"][label]:
                    raw = bank.crop(slot)
                    if label == "train":
                        self.raw_train.append(raw.cpu())
                    safe, valid, features = prepare_features(self.model, raw[None], self.settings["rendering"]["f0"])
                    rows.append({"safe": safe.cpu(), "valid": valid.cpu(), "features": features.cpu(),
                                 **{key: slot[key] for key in ("material_id", "category", "crop")}})
                    del safe, valid, features
                self.populations[label] = rows
        bank.cache.clear()
        self.train_lights = make_lights(self.settings["rendering"]["training_lights"], self.model.device)
        self.novel_lights = make_lights(self.settings["rendering"]["novel_lights"], self.model.device)
        if len(self.train_lights) != 40 or len(self.novel_lights) != 12:
            raise ValueError("frozen light populations differ")

    def objective(self, network, indices, transforms, stage):
        target = augment_targets([self.raw_train[index] for index in indices], transforms)
        safe, valid, features = prepare_features(self.model, target, self.settings["rendering"]["f0"])
        prediction = unit_prediction(network(features))
        map_loss, parts = ndr_loss(prediction, safe, valid)
        render_loss = render_error(self.model, prediction, safe, valid, self.train_lights, self.settings["rendering"]) if stage == "render" else map_loss.new_zeros(())
        loss = map_loss + self.settings["training"]["render_loss_weight"] * render_loss
        return loss, {"map_loss": float(map_loss.detach()), "render_loss": float(render_loss.detach()),
                      "normal_loss": float(parts[0].detach()), "diffuse_loss": float(parts[1].detach()), "roughness_loss": float(parts[2].detach())}

    def evaluate(self, network, stage, step):
        populations = {name: DeviceRows(rows, self.model.device) for name, rows in self.populations.items()}
        return _evaluation(self.model, populations, self.novel_lights, self.settings)


def runtime_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("real run/smoke requires CUDA")
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "device_name": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)), "precision": "float32", "amp": False, "tf32": False}


def record_runtime(output, environment, process):
    path = output / "runtime.json"
    saved = read_json(path) if path.exists() else {"environment": environment, "invocations": []}
    if saved["environment"] != environment:
        raise ValueError("strict resume runtime environment changed")
    saved["invocations"].append({**process, "started_at_utc": datetime.now(timezone.utc).isoformat(), "memory": memory_observation()})
    atomic_write_json(path, saved)


def validate_fork_environment(source, environment):
    parent_runtime = Path(source).resolve().parent / "runtime.json"
    if not parent_runtime.is_file():
        raise FileNotFoundError("fork parent runtime identity is missing")
    if read_json(parent_runtime).get("environment") != environment:
        raise ValueError("fork parent runtime environment changed")


def fork_checkpoint(output, execution, source, stage):
    source = Path(source).resolve()
    if source.parent == output:
        raise ValueError("fork requires a distinct new output lineage")
    if (output / "last.pt").exists():
        raise FileExistsError("fork output already contains a checkpoint")
    checkpoint = load_checkpoint(source)
    if checkpoint["identity"] != execution or checkpoint["policy"] != POLICY:
        raise ValueError("fork scientific/source/input identity mismatch")
    state = checkpoint["state"]
    if state["stage"] != stage or state["stage_step"] <= 0:
        raise ValueError("fork must use the requested stage's last state")
    if stage == "map" and source.name != "map_stage_end.pt":
        raise ValueError("map fork requires preserved map_stage_end.pt")
    if stage == "render" and source.name not in ("render_stage_end.pt", "last.pt"):
        raise ValueError("render fork requires render_stage_end.pt or last.pt")
    state["status"] = "interrupted"
    state.pop("selected", None)
    state["stop_reasons"].pop(stage, None)
    state["convergence"][stage]["floor_bad_observations"] = 0
    if stage == "map":
        state["best_render"] = None
        state["render_updates"] = 0
        state["evaluations"]["render"] = 0
        state["convergence"]["render"] = Engine._new_convergence()
        state["stop_reasons"].pop("render", None)
    atomic_write_json(output / "lineage.json", {"fork_source": file_identity(source), "stage": stage,
        "floor_counter_reset": 0, "additional_floor_observations": POLICY["floor_observations"],
        "retained": "last model/Adam/scheduler/RNG and raw/meaningful best; map fork renders fresh after new stop"})
    atomic_save(output / "last.pt", checkpoint)


def run(output, *, resume=False, extend_map_to=None, extend_render_to=None, fork_map_from=None, fork_render_from=None):
    if fork_map_from and fork_render_from:
        raise ValueError("select exactly one fork stage")
    output, execution = prepare(output)
    if (output / "smoke.json").exists():
        raise ValueError("real training requires a separate output from smoke")
    fork = fork_map_from or fork_render_from
    effective_resume = resume or bool(fork)
    with process_lock(output, resume=effective_resume) as process:
        environment = runtime_environment()
        # No historical pickle is loaded; fork and resume accept only this package's safe schema.
        if fork:
            if resume:
                raise ValueError("new fork cannot also request --resume")
            validate_fork_environment(fork, environment)
            fork_checkpoint(output, execution, fork, "map" if fork_map_from else "render")
        if effective_resume:
            path = output / "last.pt"
            checkpoint = load_checkpoint(path if path.exists() else output / "previous.pt")
            state = checkpoint["state"]
            if extend_map_to is not None and state["stage"] != "map":
                raise ValueError("map extension after transition requires a new --fork-map-from lineage")
            if state["status"] == "complete" and (extend_map_to or extend_render_to):
                raise ValueError("completed-stage extension requires a new fork lineage")
            if checkpoint["identity"] != execution or checkpoint["policy"] != POLICY:
                raise ValueError("checkpoint scientific/source/input identity mismatch")
        runtime_path = output / "runtime.json"
        if runtime_path.exists() and read_json(runtime_path)["environment"] != environment:
            raise ValueError("strict resume runtime environment changed")
        caps = budget(output, resume=effective_resume, extend_map_to=extend_map_to, extend_render_to=extend_render_to)
        record_runtime(output, environment, process)
        pipeline = Pipeline(execution)
        engine = Engine(output, pipeline.model.net_g, POLICY, execution, pipeline.objective, pipeline.evaluate,
                        adam_betas=tuple(pipeline.settings["training"]["adam_betas"]), clip_norm=pipeline.settings["training"]["gradient_clip_norm"])
        engine.memory_guard = lambda: memory_is_low(memory_observation())
        original_write = engine._write_status
        def write_status(status=None, error=None):
            original_write(status, error)
            document = read_json(output / "status.json")
            document.update(process, memory=memory_observation(), output_directory=str(output))
            atomic_write_json(output / "status.json", document)
        engine._write_status = write_status
        try:
            return engine.run(caps, resume=effective_resume)
        except (torch.OutOfMemoryError, MemoryError) as error:
            traceback.clear_frames(error.__traceback__)
            engine.optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            path = output / "last.pt"
            engine.restore(load_checkpoint(path if path.exists() else output / "previous.pt"))
            engine.committed_step = engine.state["total_step"]
            _truncate = __import__("experiments.regularization.run", fromlist=["_truncate_log"])._truncate_log
            _truncate(output / "train.jsonl", engine.committed_step)
            write_status("memory_paused", error)
            return engine.state


def smoke(output):
    output, execution = prepare(output)
    with process_lock(output, resume=False) as process:
        environment = runtime_environment()
        record_runtime(output, environment, process)
        pipeline = Pipeline(execution)
        network = pipeline.model.net_g
        engine = Engine(output, network, POLICY, execution, pipeline.objective, pipeline.evaluate)
        stages = {}
        for stage in ("map", "render"):
            network.train()
            engine._new_optimizer(stage)
            engine.optimizer.zero_grad(set_to_none=True)
            loss, metrics = pipeline.objective(network, [0, 2, 4, 6], [0, 1, 4, 7], stage)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite smoke loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0, error_if_nonfinite=True)
            if norm <= 0:
                raise FloatingPointError("zero smoke gradient")
            engine.optimizer.step()
            stages[stage] = {"loss": float(loss.detach()), "gradient_norm": float(norm), **metrics}
        with neutral_evaluation(network):
            populations = pipeline.evaluate(network, "render", 1)
        from experiments.plateau_followup.engine import metric_values
        metric_values(populations)
        result = {"status": "smoke_complete", "implementation_only": True, "parameters": parameter_count(network),
                  "policy": POLICY, "stages": stages, "populations": populations, "memory": memory_observation()}
        atomic_write_json(output / "smoke.json", result)
        atomic_write_json(output / "status.json", {"status": "smoke_complete", **process})
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status", "smoke", "run"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-map-to", type=int)
    parser.add_argument("--extend-render-to", type=int)
    parser.add_argument("--fork-map-from", type=Path)
    parser.add_argument("--fork-render-from", type=Path)
    args = parser.parse_args(argv)
    controls = (args.resume, args.extend_map_to, args.extend_render_to, args.fork_map_from, args.fork_render_from)
    if args.command != "run" and any(controls):
        parser.error("resume/extension/fork controls are only accepted by run")
    if args.command == "prepare":
        output, _ = prepare(args.out)
        print(output)
    elif args.command == "status":
        print(json.dumps(read_json(args.out / "status.json"), indent=2))
    elif args.command == "smoke":
        print(json.dumps(smoke(args.out), indent=2))
    else:
        state = run(args.out, resume=args.resume, extend_map_to=args.extend_map_to,
                    extend_render_to=args.extend_render_to, fork_map_from=args.fork_map_from, fork_render_from=args.fork_render_from)
        print(json.dumps({key: state[key] for key in ("status", "stage", "stage_step", "total_step")}))


if __name__ == "__main__":
    main()
