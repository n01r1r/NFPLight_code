"""Authorized 15M/24crop extension preserving the original suite sources."""
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path
from experiments.regularization.config import atomic_write_json, file_identity
from experiments.robustness_suite import config, engine
from experiments.robustness_suite import run as runner
from experiments.robustness_suite.controller import cuda_busy, gpu_ready, utc
ARM = "capacity15m_extra24_p01"
ROOT = config.ROOT / "artifacts/capacity15m_extra24_seed19_20261010"
GIB = 2**30
PARAMETERS = 15085034
FULL_NAMES = {"last.pt", "previous.pt", "map_stage_end.pt", "render_stage_end.pt"}
_original_identity = config.execution_identity
_original_save = engine.atomic_save
class DiskReserveError(OSError):
    """Retain committed checkpoints and require an explicit reviewed resume."""
def storage_limits(directory):
    directory = Path(directory)
    weight = max(PARAMETERS * 4, max((p.stat().st_size for p in directory.glob("*.pt") if p.name not in FULL_NAMES), default=0))
    full = max(10 * weight, max((p.stat().st_size for p in directory.glob("*.pt") if p.name in FULL_NAMES), default=0))
    envelope = 4 * full + 4 * weight + 64 * 2**20
    critical = max(5 * GIB, 2 * full)
    return max(15 * GIB, envelope + 5 * GIB, critical), critical

def require_space(directory, *, launch=False, free_bytes=None):
    directory = Path(directory)
    warning, critical = storage_limits(directory)
    free = shutil.disk_usage(directory).free if free_bytes is None else free_bytes
    threshold = warning if launch else critical
    observation = {"at_utc": utc(), "free_bytes": free, "warning_bytes": warning, "critical_bytes": critical, "required_bytes": threshold, "phase": "launch" if launch else "checkpoint_save"}
    if free < threshold:
        atomic_write_json(directory / "storage_incident.json", observation)
        raise DiskReserveError("disk reserve gate blocked; explicit reviewed resume required")
    return observation

def guarded_save(path, value, *, rotate=False):
    require_space(Path(path).parent)
    return _original_save(path, value, rotate=rotate)

def execution_identity(arm):
    if arm != ARM:
        raise ValueError("this extension authorizes only the 15M/24crop arm")
    execution = _original_identity(arm)
    execution["file_identities"].append(file_identity(Path(__file__)))
    execution["file_identities"].sort(key=lambda row: row["path"])
    execution["control_roles"] = {"historical_capacity_control": str(config.DEFAULT_OUT / "arms_final/capacity15m_p01"), "historical_crop_control": str(config.DEFAULT_OUT / "arms_final/extra_crop7m_p01"), "historical_baseline": str(config.DEFAULT_OUT / "arms_final/baseline7m_p01"), "scope": "single-seed exploratory missing combination; no interaction significance claim"}
    execution["storage_guard"] = {"full_multiplier": 10, "warning_floor_bytes": 15 * GIB, "critical_floor_bytes": 5 * GIB, "max_normal_files": 8, "on_block": "stop before save; retain committed checkpoint; no automatic retry"}
    return execution

def install():
    config.ARMS[ARM] = {"parameters": PARAMETERS, "middle_blocks": 12, "extra_training_crops": True}
    config.execution_identity = execution_identity
    engine.atomic_save = guarded_save

def is_followup_trainer(info):
    args = info.get("cmdline") or []
    return "python" in (info.get("name") or "").lower() and any(args[i:i+3] in (["-m", "experiments.capacity_crop_followup", "run"], ["-m", "experiments.capacity_crop_followup", "smoke"]) for i in range(len(args)-2))

def launch_gate(output):
    import psutil
    busy = cuda_busy((os.getpid(),))
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if process.pid != os.getpid() and is_followup_trainer(process.info):
                busy.append({"pid": process.pid, "command": command})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    ready, gpu = gpu_ready()
    storage = require_space(output, launch=True)
    ram = psutil.virtual_memory().available
    report = {"at_utc": utc(), "gpu": gpu, "other_trainers": busy, "storage": storage, "available_ram_bytes": ram, "ready": ready and not busy and ram >= 2 * GIB}
    atomic_write_json(Path(output) / "launch_gate.json", report)
    if not report["ready"]:
        raise RuntimeError("GPU/RAM gate blocked; no training started")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "run", "status"))
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.resume and args.command != "run":
        parser.error("--resume only applies to explicitly resumed training")
    install()
    if args.command == "status":
        print(config.read_json(args.out / "status.json"))
        return
    output, _ = config.prepare(args.out, arm=ARM)
    if args.command == "prepare":
        print(output)
        return
    launch_gate(output)
    try:
        result = runner.smoke(output, arm=ARM) if args.command == "smoke" else runner.run(output, arm=ARM, resume=args.resume)
        print(result.get("status"))
    except DiskReserveError as error:
        status = config.read_json(output / "status.json")
        status.update(status="storage_paused", error=str(error), recovery="last.pt/previous.pt committed state only; explicit resume required")
        atomic_write_json(output / "status.json", status)
        raise
if __name__ == "__main__":
    main()
