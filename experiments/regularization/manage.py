"""Prepare, inspect, run/resume, and report the local regularization probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.regularization.config import (
    load_config, load_execution_config, prepare_run, resolve_repo_path,
)


def _output_path(config_path=None, out=None):
    if out is not None:
        return resolve_repo_path(out)
    config, _ = load_config(config_path)
    return resolve_repo_path(config["output"]["default_directory"])


def _status(output_dir):
    path = output_dir / "status.json"
    if not path.is_file():
        return {"status": "unprepared", "output_directory": str(output_dir)}
    result = json.loads(path.read_text(encoding="utf-8"))
    recorded = result.get("status", "unknown")
    result["recorded_status"] = recorded
    if recorded == "running":
        from experiments.bottleneck_fc.config import process_identity, process_identity_matches
        pid, identity = result.get("pid"), result.get("process_identity")
        if identity is not None and process_identity_matches(pid, identity):
            result["status"] = "running"
        elif identity is None and process_identity(pid) is not None:
            result["status"] = "unverified"
            result["status_detail"] = "process exists but no creation identity was recorded"
        else:
            result["status"] = "interrupted"
            result["status_detail"] = "recorded process is absent or has a different identity"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="validate and save immutable config/data identities")
    prepare_parser.add_argument("--config", type=Path)
    prepare_parser.add_argument("--out", type=Path)

    status_parser = commands.add_parser("status", help="read status and verify live process identity")
    status_parser.add_argument("--config", type=Path)
    status_parser.add_argument("--out", type=Path)

    run_parser = commands.add_parser("run", help="start or resume the three new conditions")
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--out", type=Path)
    run_parser.add_argument("--resume", action="store_true")

    report_parser = commands.add_parser("report", help="read saved results and build a partial/complete JSON report")
    report_parser.add_argument("--run-dir", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        output_dir = prepare_run(args.config, args.out)
        result = {
            "status": _status(output_dir)["status"],
            "output_directory": str(output_dir),
            "execution_config": str(output_dir / "execution_config.json"),
            "training_started": False,
            "device_initialized": False,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return output_dir
    if args.command == "status":
        output_dir = _output_path(args.config, args.out)
        if (output_dir / "execution_config.json").is_file():
            load_execution_config(output_dir)
        result = _status(output_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return result
    if args.command == "run":
        from experiments.regularization.run import run_experiment
        return run_experiment(args.config, args.out, resume=args.resume)
    if args.command == "report":
        from experiments.regularization.report import build_report
        result = build_report(resolve_repo_path(args.run_dir))
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return result
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
