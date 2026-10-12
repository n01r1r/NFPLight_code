"""Prepare, inspect, run, and report the bottleneck FC probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bottleneck_fc.config import (
    load_config,
    load_execution_config,
    prepare_run,
    process_identity,
    process_identity_matches,
    resolve_repo_path,
)


def _status(output_dir):
    status_path = output_dir / "status.json"
    if not status_path.is_file():
        return {"status": "unprepared", "output_directory": str(output_dir)}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    effective = status.get("status", "unknown")
    result = dict(status)
    result["recorded_status"] = effective
    if effective == "running":
        matched = process_identity_matches(
            status.get("pid"), status.get("process_identity")
        )
        if matched:
            result["status"] = "running"
        elif status.get("process_identity") is None:
            current = process_identity(status.get("pid"))
            result["status"] = "unverified" if current is not None else "interrupted"
            result["status_detail"] = (
                "process exists but no creation identity was recorded"
                if current is not None else "recorded PID is absent"
            )
        else:
            result["status"] = "interrupted"
            result["status_detail"] = "recorded PID is absent or belongs to a different process"
    return result


def _default_output(config_path):
    config, _ = load_config(config_path)
    return resolve_repo_path(config["output"]["default_directory"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare_parser = commands.add_parser("prepare", help="validate paths and save immutable run config")
    prepare_parser.add_argument("--config", type=Path, help="experiment JSON config")
    prepare_parser.add_argument("--out", type=Path, help="run directory; relative paths use repository root")

    status_parser = commands.add_parser("status", help="read run status and verify process identity")
    status_parser.add_argument("--config", type=Path, help="select the config's default output directory")
    status_parser.add_argument("--out", type=Path, help="run directory")

    run_parser = commands.add_parser("run", help="start or resume configured experiment runs")
    run_parser.add_argument("--config", type=Path, help="experiment JSON config")
    run_parser.add_argument("--out", type=Path, help="run directory; relative paths use repository root")
    run_parser.add_argument("--resume", action="store_true", help="resume only an identity-matched run")
    run_parser.add_argument("--arm", help="run one configured arm; omit to run all three")

    report_parser = commands.add_parser("report", help="build read-only summaries from saved run files")
    report_parser.add_argument("--run-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        output_dir = prepare_run(args.config, args.out)
        result = {
            "status": _status(output_dir)["status"],
            "output_directory": str(output_dir),
            "execution_config": str(output_dir / "config.json"),
            "started_training": False,
            "device_initialized": False,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return output_dir
    if args.command == "status":
        output_dir = resolve_repo_path(args.out) if args.out else _default_output(args.config)
        if (output_dir / "config.json").is_file():
            load_execution_config(output_dir)
        print(json.dumps(_status(output_dir), indent=2, ensure_ascii=False, allow_nan=False))
        return output_dir
    if args.command == "run":
        from experiments.bottleneck_fc.run import run_experiment
        return run_experiment(
            args.config, args.out, resume=args.resume, arm_id=args.arm
        )
    if args.command == "report":
        from experiments.bottleneck_fc.report import build_report
        return build_report(resolve_repo_path(args.run_dir))
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
