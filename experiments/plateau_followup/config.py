"""Frozen scientific contract and path/size/mtime identities (no hashes)."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments.regularization.config import atomic_write_json, file_identity

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "artifacts/regularization_probe_20261009_checked"
DEFAULT_OUT = ROOT / "artifacts/plateau_followup_7m_seed19_20261009"
POLICY = {
    "seed": 19, "parameters": 7609834, "d4": True, "dropout_probability": 0.1,
    "initialization": "scratch", "precision": "float32", "amp": False, "tf32": False,
    "evaluation_interval": 100, "checkpoint_interval": 100,
    "map_cap": 50000, "render_cap": 15000, "floor_observations": 20,
    "map_lr": 5e-4, "map_min_lr": 1e-5, "render_lr": 1e-4, "render_min_lr": 1e-6,
    "factor": 0.5, "patience": 5, "threshold": 0.005, "threshold_mode": "rel",
    "cooldown": 2, "eps": 1e-8, "mode": "min", "render_ndr_guard": 1.02,
    "map_monitor": "heldout.ndr_mean", "render_monitor": "heldout.render_l1",
    "map_transition": "floor20_no_meaningful_improvement_restore_raw_best_map",
    "cap_action": "pause_requires_explicit_monotone_extension",
    "render_selection": "eligible_raw_lowest_render_l1_else_best_map",
    "sample_stream": "continuous Random(seed+10000); render starts at actual map stop; not paired with old render prefix",
    "floor_rule": "first_floor_arrival_is_zero; count only subsequent floor-trained evaluations against meaningful best",
}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def execution_identity():
    reference_path = CONTROL / "execution_config.json"
    reference = read_json(reference_path)
    complete = read_json(CONTROL / "seed19_d4_dropout/complete.json")
    if (complete.get("condition"), complete.get("seed"), complete.get("parameters")) != ("d4_dropout", 19, 7609834):
        raise ValueError("control condition/seed/parameters changed")
    scientific = copy.deepcopy(reference["experiment_settings"]["scientific_settings"])
    inputs = reference["resolved_inputs"]
    if scientific["baseline_parameters"] != 7609834 or scientific["sampling"]["batch_size"] != 4:
        raise ValueError("control architecture or batch contract changed")
    expected_render = {"feature_version": "raw_calibrated_gain9_v3", "precision": "float32",
                       "amp": False, "tf32": False, "near_distance": 4.0,
                       "far_distance": 12.0, "gain": 9.0, "f0": 0.04}
    if any(scientific["rendering"].get(k) != v for k, v in expected_render.items()):
        raise ValueError("control rendering contract changed")
    if reference["experiment_settings"]["training"]["d4_seed_offset"] != 20000:
        raise ValueError("control D4 RNG contract changed")
    if [len(inputs[k]) for k in ("train", "unseen_crop", "heldout")] != [16, 8, 8]:
        raise ValueError("control populations changed")
    input_paths = {inputs["manifest"]}
    input_paths.update(row["shard_path"] for name in ("train", "unseen_crop", "heldout") for row in inputs[name])
    historical = {str(Path(row["path"]).resolve()): row for row in reference["file_identities"]}
    for path in input_paths:
        current = file_identity(path)
        if historical.get(current["path"]) != current:
            raise ValueError(f"control input identity changed: {path}")
    sources = [ROOT / path for path in reference["experiment_settings"]["provenance"]["source_files"]]
    sources += sorted((ROOT / "experiments/plateau_followup").glob("*.py"))
    paths = set(sources) | {Path(path) for path in input_paths}
    paths |= {reference_path, CONTROL / "experiment_config.json", CONTROL / "seed19_d4_dropout/complete.json",
              CONTROL / "seed19_d4_dropout/evaluation.json"}
    return {"version": 1, "policy": POLICY, "scientific_settings": scientific,
            "resolved_inputs": inputs, "control": str(CONTROL / "seed19_d4_dropout"),
            "file_identities": sorted((file_identity(p) for p in paths), key=lambda r: r["path"]),
            "identity_method": "canonical resolved path, size, mtime_ns; no content hashes"}


def prepare(output_dir=DEFAULT_OUT):
    output_dir = Path(output_dir).resolve()
    if CONTROL == output_dir or CONTROL in output_dir.parents:
        raise ValueError("output must be a new lineage outside the control")
    candidate = execution_identity()
    path = output_dir / "execution_config.json"
    if path.exists():
        if read_json(path) != candidate:
            raise ValueError("immutable scientific/source/input identity mismatch")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("new output directory must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, candidate)
        atomic_write_json(output_dir / "budget_ledger.json", [{
            "at_utc": datetime.now(timezone.utc).isoformat(), "event": "initial",
            "map_cap": POLICY["map_cap"], "render_cap": POLICY["render_cap"]}])
        atomic_write_json(output_dir / "status.json", {"status": "prepared", "stage": "map", "step": 0, "total_step": 0})
    return output_dir, candidate


def budget(output_dir, *, resume=False, extend_map_to=None, extend_render_to=None):
    path = Path(output_dir) / "budget_ledger.json"
    ledger = read_json(path)
    current = {key: ledger[-1][key] for key in ("map_cap", "render_cap")}
    requested = {"map_cap": extend_map_to, "render_cap": extend_render_to}
    if any(value is not None for value in requested.values()) and not resume:
        raise ValueError("budget extension requires --resume")
    for key, value in requested.items():
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int) or value <= current[key] or value % POLICY["evaluation_interval"]:
                raise ValueError("extension must strictly increase the cap in evaluation-interval multiples")
            current[key] = value
    if any(value is not None for value in requested.values()):
        ledger.append({"at_utc": datetime.now(timezone.utc).isoformat(), "event": "explicit_extension", **current})
        atomic_write_json(path, ledger)
    return current
