"""Regularization experiment config and immutable identities; no model or GPU imports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
DEFAULT_CONFIG = PACKAGE_DIR / "config.json"
IDENTITY_METHOD = "resolved path, file size, and mtime_ns; no content hashes"
CONDITION_CONTRACT = (
    ("baseline", False, False, "existing_baseline_extension"),
    ("d4_only", True, False, "train"),
    ("dropout_only", False, True, "train"),
    ("d4_dropout", True, True, "train"),
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _keys(value, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(set(required) - value.keys())
    unknown = sorted(value.keys() - set(required))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"{name}: " + "; ".join(details))


def _int(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _safe_repo_relative(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must remain repository-relative")


def _baseline_scientific_settings(config):
    training = config["training"]
    return {
        "dataset": config["dataset"],
        "backbone": config["model"]["backbone"],
        "baseline_parameters": config["model"]["baseline_parameters"],
        "sampling": config["sampling"],
        "optimization": {
            key: training[key] for key in (
                "map_learning_rate", "render_learning_rate", "learning_rate_schedule",
                "adam_betas", "fresh_adam_at_render_stage", "gradient_clip_norm",
                "render_loss_weight",
            )
        },
        "evaluation": {
            "checkpoint_interval": training["checkpoint_interval"],
            "evaluation_interval": training["evaluation_interval"],
        },
        "loss": config["loss"],
        "rendering": config["rendering"],
        "runtime": config["runtime"],
    }


def validate_config(config):
    _keys(config, (
        "schema_version", "experiment_id", "output", "baseline", "design",
        "training", "provenance", "scientific_settings",
    ), "config")
    if config["schema_version"] != 1 or config["experiment_id"] != "regularization_probe":
        raise ValueError("unsupported regularization experiment config")
    _keys(config["output"], ("default_directory",), "output")
    _safe_repo_relative(config["output"]["default_directory"], "output.default_directory")
    _keys(config["baseline"], ("config", "historical_run", "extension_run"), "baseline")
    for key in ("config", "historical_run", "extension_run"):
        _safe_repo_relative(config["baseline"][key], f"baseline.{key}")

    design = config["design"]
    _keys(design, ("dropout_probability", "conditions"), "design")
    if design["dropout_probability"] != 0.1:
        raise ValueError("dropout_probability is frozen at the approved 0.1")
    rows = design["conditions"]
    if not isinstance(rows, list) or len(rows) != len(CONDITION_CONTRACT):
        raise ValueError("design.conditions must contain the frozen 2x2 contract")
    for row, expected in zip(rows, CONDITION_CONTRACT):
        _keys(row, ("id", "d4", "dropout", "source"), "design condition")
        if (row["id"], row["d4"], row["dropout"], row["source"]) != expected:
            raise ValueError(f"invalid condition contract for {expected[0]}")

    training = config["training"]
    _keys(training, (
        "seeds", "map_cap", "render_cap", "consecutive_intervals",
        "relative_improvement_threshold", "d4_seed_offset",
    ), "training")
    if training["seeds"] != [7, 19, 31]:
        raise ValueError("training.seeds are frozen at [7, 19, 31]")
    for key, expected in (
        ("map_cap", 10000), ("render_cap", 3000),
        ("consecutive_intervals", 10), ("d4_seed_offset", 20000),
    ):
        _int(training[key], f"training.{key}", 1)
        if training[key] != expected:
            raise ValueError(f"training.{key} is frozen at {expected}")
    threshold = training["relative_improvement_threshold"]
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold != 0.005):
        raise ValueError("training.relative_improvement_threshold is frozen at 0.005")

    scientific = config["scientific_settings"]
    _keys(scientific, (
        "dataset", "backbone", "baseline_parameters", "sampling", "optimization",
        "evaluation", "loss", "rendering", "runtime",
    ), "scientific_settings")
    from experiments.bottleneck_fc.config import load_config as load_baseline_config
    baseline, _ = load_baseline_config(resolve_repo_path(config["baseline"]["config"]))
    if _baseline_scientific_settings(baseline) != scientific:
        raise ValueError("local scientific settings differ from the frozen baseline reference contract")

    sampling = scientific["sampling"]
    if sampling["batch_size"] != 4 or sampling["sample_order_seed_offset"] != 10000:
        raise ValueError("paired sample order and batch size must remain baseline-identical")
    if sampling["d4_augmentation"] is not False:
        raise ValueError("baseline sampling contract must retain D4 disabled; per-arm D4 is separate")
    if scientific["baseline_parameters"] != 7609834:
        raise ValueError("baseline parameter count must be 7,609,834")
    if scientific["backbone"] != {
        "name": "TwoBranchNet", "width": 32, "middle_blocks": 4,
        "encoder_blocks": [2, 2, 4], "decoder_blocks": [2, 2, 2],
        "input_channels": 21, "output_channels": 10, "input_resolution": 256,
    }:
        raise ValueError("the TwoBranchNet baseline shape is frozen")
    if scientific["evaluation"] != {"checkpoint_interval": 100, "evaluation_interval": 100}:
        raise ValueError("checkpoint and evaluation intervals are frozen at 100")
    if scientific["rendering"]["precision"] != "float32" or scientific["rendering"]["amp"] or scientific["rendering"]["tf32"]:
        raise ValueError("regularization runs require FP32 with AMP and TF32 disabled")
    if scientific["runtime"]["device"] != "cuda":
        raise ValueError("training device must remain CUDA")
    provenance = config["provenance"]
    _keys(provenance, ("identity_method", "source_files"), "provenance")
    if provenance["identity_method"] != IDENTITY_METHOD:
        raise ValueError("unsupported provenance identity method")
    files = provenance["source_files"]
    if not isinstance(files, list) or not files or len(files) != len(set(files)):
        raise ValueError("provenance.source_files must be a nonempty unique list")
    for path in files:
        _safe_repo_relative(path, "provenance.source_files entry")
    return config


def resolve_repo_path(value, *, repo_root=REPO_ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def load_config(path=None):
    config_path = resolve_repo_path(path or DEFAULT_CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    return validate_config(config), config_path


def file_identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _local_resolved_inputs(scientific, repo_root):
    from experiments.bottleneck_fc.config import (
        _manifest_rows,
        resolve_repo_path as baseline_resolve,
        select_materials,
    )

    dataset = scientific["dataset"]
    root = Path(repo_root).resolve()
    manifest = baseline_resolve(dataset["manifest"], repo_root=root)
    records = _manifest_rows(manifest)
    train_materials = select_materials(
        records, dataset["train_split"], dataset["categories"], dataset["materials_per_category"]
    )
    heldout_materials = select_materials(
        records, dataset["heldout_split"], dataset["categories"], dataset["materials_per_category"]
    )

    def slots(materials, crops):
        result = []
        for row in materials:
            shard = Path(row["shard_path"])
            shard = shard.resolve() if shard.is_absolute() else (root / shard).resolve()
            for crop in crops:
                result.append({
                    "material_id": row["material_id"],
                    "category": row["category"],
                    "shard_path": str(shard),
                    "crop": crop,
                })
        return result

    return {
        "manifest": str(manifest.resolve()),
        "train": slots(train_materials, dataset["train_crops"]),
        "unseen_crop": slots(train_materials, [dataset["unseen_crop"]]),
        "heldout": slots(heldout_materials, [dataset["heldout_crop"]]),
    }


def _historical_identity(config, repo_root):
    root = resolve_repo_path(config["baseline"]["historical_run"], repo_root=repo_root)
    manifest = root / "initial_result_identities.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"invalid historical baseline identity manifest: {manifest}")
    verified = []
    for row in entries:
        if not isinstance(row, dict) or not {"path", "bytes", "mtime_ns"}.issubset(row):
            raise ValueError(f"invalid historical baseline identity entry: {manifest}")
        current = file_identity(row["path"])
        if current["size"] != row["bytes"] or current["mtime_ns"] != row["mtime_ns"]:
            raise ValueError(f"historical baseline artifact changed: {current['path']}")
        verified.append(current)
    return {
        "directory": str(root),
        "identity_manifest": file_identity(manifest),
        "files": sorted(verified, key=lambda row: row["path"]),
    }


def build_execution_config(config, config_path, output_dir, *, repo_root=REPO_ROOT):
    validate_config(config)
    root = Path(repo_root).resolve()
    output_dir = resolve_repo_path(output_dir, repo_root=root)

    from experiments.bottleneck_fc.config import (
        build_execution_config as build_baseline_execution,
        load_config as load_baseline_config,
        resolve_repo_path as baseline_resolve,
    )

    baseline_config_path = baseline_resolve(config["baseline"]["config"], repo_root=root)
    baseline_config, baseline_config_path = load_baseline_config(baseline_config_path)
    baseline_execution = build_baseline_execution(
        baseline_config, baseline_config_path, output_dir, repo_root=root
    )
    baseline_execution.pop("created_at_utc", None)
    if _baseline_scientific_settings(baseline_config) != config["scientific_settings"]:
        raise ValueError("regularization config does not match the frozen baseline control contract")

    resolved_inputs = _local_resolved_inputs(config["scientific_settings"], root)
    if resolved_inputs != baseline_execution["resolved_inputs"]:
        raise ValueError("local resolved data slots do not match the frozen baseline reference")
    identities = {
        file_identity(config_path)["path"]: file_identity(config_path),
        file_identity(baseline_config_path)["path"]: file_identity(baseline_config_path),
        resolved_inputs["manifest"]: file_identity(resolved_inputs["manifest"]),
    }
    for rows in (resolved_inputs["train"], resolved_inputs["heldout"]):
        for row in rows:
            identities[row["shard_path"]] = file_identity(row["shard_path"])
    for relative in config["provenance"]["source_files"]:
        source = resolve_repo_path(relative, repo_root=root)
        if not source.is_file():
            raise FileNotFoundError(f"provenance source is missing: {source}")
        row = file_identity(source)
        identities[row["path"]] = row
    historical = _historical_identity(config, root)
    identities[historical["identity_manifest"]["path"]] = historical["identity_manifest"]
    for row in historical["files"]:
        identities[row["path"]] = row

    return {
        "execution_config_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository_root": str(root),
        "output_directory": str(output_dir),
        "experiment_config_source": file_identity(config_path),
        "experiment_settings": config,
        "baseline_execution_identity": {
            "execution_config_version": baseline_execution["execution_config_version"],
            "config_source": baseline_execution["config_source"],
            "file_identities": baseline_execution["file_identities"],
            "settings": baseline_execution["settings"],
        },
        "resolved_inputs": resolved_inputs,
        "historical_baseline_identity": historical,
        "baseline_extension_reference": str(
            resolve_repo_path(config["baseline"]["extension_run"], repo_root=root)
        ),
        "file_identities": [identities[key] for key in sorted(identities)],
    }


def same_execution_config(saved, candidate):
    ignored = {"created_at_utc"}
    return (
        isinstance(saved, dict) and isinstance(candidate, dict)
        and {key: value for key, value in saved.items() if key not in ignored}
        == {key: value for key, value in candidate.items() if key not in ignored}
    )


def replace_with_retry(temporary, target, attempts=10):
    for index in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if index + 1 == attempts:
                raise
            time.sleep(0.05 * (index + 1))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    replace_with_retry(temporary, path)


def prepare_run(config_path=None, output_dir=None):
    config, config_path = load_config(config_path)
    output_dir = resolve_repo_path(output_dir or config["output"]["default_directory"])
    candidate = build_execution_config(config, config_path, output_dir)
    execution_path = output_dir / "execution_config.json"
    if output_dir.exists() and not execution_path.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory contains files but is not a prepared run: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if execution_path.is_file():
        saved = json.loads(execution_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not same_execution_config(saved, candidate):
            raise ValueError(f"immutable execution identity mismatch: {execution_path}")
    else:
        atomic_write_json(execution_path, candidate)
        atomic_write_json(output_dir / "experiment_config.json", config)
    status_path = output_dir / "status.json"
    if not status_path.exists():
        atomic_write_json(status_path, {
            "status": "prepared",
            "run_scope": "three_new_conditions; existing_baseline_extension_is_reference_only",
            "selected_condition_ids": ["d4_only", "dropout_only", "d4_dropout"],
            "comparison_expected_runs": 12,
            "comparison_complete": False,
            "scheduled_training_runs": 9,
            "stage": None,
            "step": 0,
            "total_step": 0,
            "last_committed_step": 0,
            "output_directory": str(output_dir),
            "training_started": False,
        })
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        atomic_write_json(metadata_path, {
            "metadata_version": 1,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity_method": IDENTITY_METHOD,
            "training_started": False,
            "baseline_reference": candidate["baseline_extension_reference"],
            "baseline_reference_required_before_run": True,
        })
    return output_dir


def load_execution_config(output_dir):
    path = Path(output_dir) / "execution_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared execution config not found: {path}")
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if result.get("execution_config_version") != 1:
        raise ValueError(f"unsupported execution config: {path}")
    return result


def _validate_baseline_extension_policy(policy, seeds):
    if policy.get("arm_id") != "baseline" or policy.get("seeds") != seeds:
        raise ValueError("baseline extension does not cover the frozen baseline seeds")
    expected_early_stop = {
        "consecutive_intervals": 10,
        "relative_improvement_threshold": 0.005,
        "relative_improvement_formula": "(previous-current)/max(abs(previous),1e-12)",
        "map_metric": "heldout.ndr_mean",
        "render_metric": "heldout.render_l1",
    }
    if (policy.get("map_total_limit") != 10000 or policy.get("render_total_limit") != 3000
            or policy.get("evaluation_interval") != 100
            or policy.get("checkpoint_interval") != 100
            or policy.get("early_stop") != expected_early_stop):
        raise ValueError("baseline extension evaluation and plateau policy differ from the regularization contract")


def verify_baseline_extension(execution, output_dir, *, record=True):
    """Require and freeze a completed, identity-matched baseline extension."""
    reference = Path(execution["baseline_extension_reference"])
    manifest_path = reference / "extension_manifest.json"
    status_path = reference / "status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"completed baseline extension not found: {reference}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = execution["baseline_execution_identity"]
    if manifest.get("parent_execution_identity") != expected:
        raise ValueError("baseline extension source/data/config identity does not match the frozen baseline")
    if status.get("status") != "complete":
        raise RuntimeError(f"baseline extension is not complete: status={status.get('status')!r}")
    policy_path = reference / "extension_policy.json"
    if not policy_path.is_file():
        raise FileNotFoundError(f"baseline extension policy is missing: {policy_path}")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    seeds = execution["experiment_settings"]["training"]["seeds"]
    _validate_baseline_extension_policy(policy, seeds)

    # Extension runs retain their execution config in the immutable parent run.
    parent_config_path = Path(manifest["parent_directory"]) / "config.json"
    recorded_config = manifest.get("parent_files", {}).get("root", {}).get("config.json")
    if file_identity(parent_config_path) != recorded_config:
        raise ValueError("baseline parent config identity changed after extension")
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    if {key: parent_config.get(key) for key in expected} != expected:
        raise ValueError("baseline parent config does not match the frozen execution identity")
    files = [manifest_path, status_path, policy_path, parent_config_path]
    seed_records = {}
    for seed in seeds:
        run_dir = reference / f"seed{seed}_baseline"
        complete_path = run_dir / "complete.json"
        seed_status_path = run_dir / "status.json"
        required = (
            complete_path, seed_status_path, run_dir / "last.pt",
            run_dir / "train.jsonl", run_dir / "evaluation.json", run_dir / "validation.json",
        )
        if any(not path.is_file() for path in required):
            raise FileNotFoundError(f"baseline extension seed{seed} lacks final output files")
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        seed_status = json.loads(seed_status_path.read_text(encoding="utf-8"))
        if complete.get("arm_id") != "baseline" or complete.get("parameters") != 7609834:
            raise ValueError(f"baseline extension seed{seed} has unexpected model identity")
        if seed_status.get("status") != "complete" or seed_status.get("seed") != seed:
            raise RuntimeError(f"baseline extension seed{seed} is not complete")
        files.extend(required)
        seed_records[str(seed)] = {"complete": complete, "status": seed_status}
    snapshot = {
        "reference_version": 1,
        "directory": str(reference.resolve()),
        "parent_execution_identity": expected,
        "seed_ids": seeds,
        "seeds": seed_records,
        "file_identities": sorted((file_identity(path) for path in files), key=lambda row: row["path"]),
    }
    snapshot_path = Path(output_dir) / "baseline_reference.json"
    if snapshot_path.is_file():
        saved = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if saved != snapshot:
            raise ValueError("baseline extension identity changed after training began")
    elif record:
        atomic_write_json(snapshot_path, snapshot)
    return snapshot
