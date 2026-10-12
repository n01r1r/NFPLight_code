"""Read saved regularization metrics; effects are withheld until the full 2x2 x 3-seed set exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.regularization.config import (
    atomic_write_json, load_execution_config, resolve_repo_path, verify_baseline_extension,
)


NEW_CONDITIONS = ("d4_only", "dropout_only", "d4_dropout")
METRICS = ("ndr_mean", "render_l1")


def _completed_rows(run_dir, execution):
    seeds = execution["experiment_settings"]["training"]["seeds"]
    rows = {}
    baseline_snapshot_path = run_dir / "baseline_reference.json"
    if baseline_snapshot_path.is_file():
        baseline_snapshot = json.loads(baseline_snapshot_path.read_text(encoding="utf-8"))
        current = verify_baseline_extension(execution, run_dir, record=False)
        if current != baseline_snapshot:
            raise ValueError("baseline extension output identity changed since its snapshot")
    else:
        baseline_snapshot = None
        try:
            baseline_snapshot = verify_baseline_extension(execution, run_dir, record=False)
        except (FileNotFoundError, RuntimeError, ValueError):
            return rows, False

    for seed in seeds:
        baseline_seed = baseline_snapshot["seeds"][str(seed)]
        record = baseline_seed["complete"]
        baseline_status = baseline_seed["status"]
        extension = record.get("extension") or {}
        map_updates = extension.get("actual_map_updates", record.get("actual_map_updates", record.get("map_updates")))
        if map_updates is None:
            map_updates = baseline_status.get("actual_map_updates", baseline_status.get("map_updates"))
        render_updates = extension.get("render_updates", record.get("render_updates"))
        if render_updates is None:
            render_updates = baseline_status.get("render_updates")
        total_updates = extension.get("total_updates", record.get("total_updates", record.get("updates")))
        if total_updates is None and map_updates is not None and render_updates is not None:
            total_updates = map_updates + render_updates
        rows[seed, "baseline"] = {
            "final": record["final"],
            "map_updates": map_updates,
            "render_updates": render_updates,
            "total_updates": total_updates,
            "source": "existing_baseline_extension",
        }
        for condition in NEW_CONDITIONS:
            path = run_dir / f"seed{seed}_{condition}" / "complete.json"
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                if record.get("seed") != seed or record.get("condition") != condition:
                    raise ValueError(f"condition completion identity mismatch: {path}")
                rows[seed, condition] = record
    expected = len(seeds) * 4
    return rows, len(rows) == expected


def _effects(rows, seeds):
    per_seed = {}
    summaries = {metric: {"d4_main_effect": [], "dropout_main_effect": [], "interaction": []}
                 for metric in METRICS}
    for seed in seeds:
        per_seed[str(seed)] = {}
        for metric in METRICS:
            values = {}
            for condition in ("baseline", "d4_only", "dropout_only", "d4_dropout"):
                values[condition] = rows[seed, condition]["final"]["heldout"]["mean"][metric]
            e00, e10, e01, e11 = (
                values["baseline"], values["d4_only"], values["dropout_only"], values["d4_dropout"]
            )
            effects = {
                "baseline": e00, "d4_only": e10, "dropout_only": e01, "d4_dropout": e11,
                "d4_main_effect": (e10 + e11 - e00 - e01) / 2,
                "dropout_main_effect": (e01 + e11 - e00 - e10) / 2,
                "interaction": e11 - e10 - e01 + e00,
            }
            per_seed[str(seed)][metric] = effects
            for key in summaries[metric]:
                summaries[metric][key].append(effects[key])
    return {
        "formula": {
            "d4_main_effect": "(E10+E11-E00-E01)/2",
            "dropout_main_effect": "(E01+E11-E00-E10)/2",
            "interaction": "E11-E10-E01+E00",
            "cells": {"E00": "baseline", "E10": "D4 only", "E01": "dropout only", "E11": "D4 + dropout"},
        },
        "direction": "These are error metrics; negative effects indicate lower error under the factor contrast.",
        "per_seed": per_seed,
        "three_seed_mean": {
            metric: {key: sum(values) / len(values) for key, values in effects.items()}
            for metric, effects in summaries.items()
        },
        "replication_note": "Paired seeds repeat initialization on the same fixed dataset; seeds are not material samples.",
    }


def build_report(run_dir):
    run_dir = Path(run_dir).resolve()
    execution = load_execution_config(run_dir)
    settings = execution["experiment_settings"]
    seeds = settings["training"]["seeds"]
    rows, complete = _completed_rows(run_dir, execution)
    output_rows = []
    for (seed, condition), record in sorted(rows.items()):
        heldout = record["final"]["heldout"]["mean"]
        output_rows.append({
            "seed": seed, "condition": condition, "source": record.get("source", "new_training"),
            "map_updates": record.get("map_updates"),
            "render_updates": record.get("render_updates"),
            "total_updates": record.get("total_updates"),
            "heldout_ndr_mean": heldout.get("ndr_mean"),
            "heldout_render_l1": heldout.get("render_l1"),
        })
    result = {
        "report_version": 1,
        "experiment": "regularization_probe",
        "run_directory": str(run_dir),
        "identity_method": settings["provenance"]["identity_method"],
        "condition_contract": settings["design"]["conditions"],
        "seeds": seeds,
        "expected_condition_seed_pairs": len(seeds) * 4,
        "completed_condition_seed_pairs": len(rows),
        "comparison_complete": complete,
        "status": "complete" if complete else "partial",
        "rows": output_rows,
        "effects": _effects(rows, seeds) if complete else None,
        "interpretation_limit": "The fixed material/crop probe does not establish generalization or causal capacity claims.",
    }
    atomic_write_json(run_dir / "report.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_report(resolve_repo_path(args.run_dir))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
