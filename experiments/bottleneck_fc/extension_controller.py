"""Wait for the approved baseline run to finish, then conditionally extend it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.bottleneck_fc.config import (
    atomic_write_json, process_identity, process_identity_matches, resolve_repo_path,
)
from experiments.bottleneck_fc.extend import POLICY_PATH, load_policy, run_extension


def _state(path, status, parent_state, *, error=None):
    result = {
        "controller_version": 1, "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pid": os.getpid(),
        "process_identity": process_identity(os.getpid()),
        "parent_status": parent_state,
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    atomic_write_json(path, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _parent_ready(parent_dir):
    status_path = parent_dir / "status.json"
    if not status_path.is_file():
        return False, "waiting_for_status"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    current = status.get("status")
    if current in ("failed", "interrupted"):
        raise RuntimeError(f"parent baseline run ended as {current}")
    if current != "complete":
        if current in ("prepared", "waiting", "running"):
            if current == "running":
                if process_identity_matches(status.get("pid"), status.get("process_identity")):
                    return False, "parent_training_active"
                raise RuntimeError("parent status says running but its process identity is no longer live")
            return False, "waiting_for_parent_training"
        raise ValueError(f"unknown parent run status: {current!r}")
    if status.get("process_identity") is None:
        raise RuntimeError("completed parent status lacks process identity")
    if process_identity_matches(status.get("pid"), status.get("process_identity")):
        return False, "parent_finalizing"
    archive_path = parent_dir / "stage_archive_status.json"
    if not archive_path.is_file():
        return False, "waiting_for_map_checkpoint_archive"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    archive_state = archive.get("status")
    if archive_state == "complete":
        return True, "ready"
    if archive_state in ("watching", "running", "pending"):
        return False, "waiting_for_map_checkpoint_archive"
    raise RuntimeError(f"map-stage archive ended as {archive_state!r}")


def run_controller(parent_dir=None, output_dir=None, policy_path=POLICY_PATH, poll_seconds=30):
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)) or poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    policy = load_policy(policy_path)
    parent_dir = resolve_repo_path(parent_dir or policy["parent_run_directory"])
    output_dir = resolve_repo_path(output_dir or policy["output_directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    controller_path = output_dir / "controller_status.json"
    if controller_path.is_file():
        previous = json.loads(controller_path.read_text(encoding="utf-8"))
        if (previous.get("status") in ("waiting_parent", "ready")
                and process_identity_matches(previous.get("pid"), previous.get("process_identity"))):
            raise RuntimeError(f"extension controller is already active as PID {previous.get('pid')}")
    parent_state = "waiting_for_status"
    _state(controller_path, "waiting_parent", parent_state)
    try:
        while True:
            ready, parent_state = _parent_ready(parent_dir)
            _state(controller_path, "ready" if ready else "waiting_parent", parent_state)
            if ready:
                break
            time.sleep(poll_seconds)
        result = run_extension(parent_dir, output_dir, policy_path)
        _state(controller_path, "complete", "baseline_complete_and_archived")
        return result
    except KeyboardInterrupt as exc:
        _state(controller_path, "interrupted", parent_state, error=exc)
        raise
    except BaseException as exc:
        _state(controller_path, "failed", parent_state, error=exc)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run-dir", type=Path, help="completed baseline run; relative paths use repository root")
    parser.add_argument("--out", type=Path, help="extension output; relative paths use repository root")
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    return run_controller(args.parent_run_dir, args.out, args.policy, args.poll_seconds)


if __name__ == "__main__":
    main()