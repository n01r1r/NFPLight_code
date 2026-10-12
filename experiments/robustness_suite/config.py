"""Immutable seed19 arm definitions and checked source/input identities; no hashes."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from experiments.plateau_followup.config import ROOT, CONTROL, POLICY as BASE_POLICY, execution_identity as base_identity, read_json, budget
from experiments.regularization.config import atomic_write_json, file_identity

DEFAULT_OUT = ROOT / "artifacts/robustness_suite_seed19_20261009"
ARMS = {
    "baseline7m_p01": {},
    "capacity15m_p01": {"parameters": 15085034, "middle_blocks": 12},
    "dropout7m_p05": {"dropout_probability": 0.5},
    "gt_aug7m_p01": {"gt_augmentation": True},
    "extra_crop7m_p01": {"extra_training_crops": True},
    "ema7m_p01": {"ema_decay": 0.999},
    "adamw7m_p01": {"optimizer": "AdamW", "weight_decay": 1e-4},
    "noise7m_p01": {"input_noise": True},
}


def policy_for_arm(arm):
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    return {**copy.deepcopy(BASE_POLICY), "arm": arm, "middle_blocks": 4,
            "optimizer": "Adam", "weight_decay": 0.0, "ema_decay": None,
            "gt_augmentation": False, "extra_training_crops": False, "input_noise": False,
            "augmentation_seed_offset": 30000, "noise_seed_offset": 40000, "noise_sigma": .005,
            "ema_selection": "EMA monitor/selection follows same heldout rules; online metrics reported separately",
            "operational_settings": "approved operating choices; not optimized or scientific convergence evidence",
            **ARMS[arm]}


def validate_policy(policy, scientific):
    expected = policy_for_arm(policy["arm"])
    if policy != expected:
        raise ValueError("scientific policy differs from the approved arm")
    rendering = scientific["rendering"]
    contract = {"feature_version": "raw_calibrated_gain9_v3", "precision": "float32", "amp": False,
                "tf32": False, "near_distance": 4.0, "far_distance": 12.0, "gain": 9.0, "f0": .04}
    if any(rendering.get(k) != v for k, v in contract.items()):
        raise ValueError("FP32 fixed-view rendering contract changed")
    backbone = scientific["backbone"]
    expected_backbone = {"width": 32, "encoder_blocks": [2, 2, 4], "decoder_blocks": [2, 2, 2], "middle_blocks": policy["middle_blocks"]}
    if any(backbone.get(k) != v for k, v in expected_backbone.items()):
        raise ValueError("approved architecture changed")
    optimization = scientific["optimization"]
    if optimization.get("adam_betas") != [.9, .999] or optimization.get("gradient_clip_norm") != 1. or optimization.get("render_loss_weight") != 1.:
        raise ValueError("approved loss/optimizer contract changed")
    if scientific["sampling"]["sample_order_seed_offset"] != 10000 or scientific["sampling"]["batch_size"] != 4:
        raise ValueError("sampling contract changed")


def execution_identity(arm):
    # This validates the frozen control input identities before any CUDA/artifact mutation.
    execution = base_identity()
    policy = policy_for_arm(arm)
    scientific = copy.deepcopy(execution["scientific_settings"])
    scientific["backbone"]["middle_blocks"] = policy["middle_blocks"]
    scientific["baseline_parameters"] = policy["parameters"]
    scientific["sampling"]["d4_augmentation"] = True
    scientific["sampling"]["sample_order"] = policy["sample_stream"]
    scientific["optimization"].update(learning_rate_schedule="ReduceLROnPlateau", optimizer=policy["optimizer"], weight_decay=policy["weight_decay"])
    scientific["dropout_probability"] = policy["dropout_probability"]
    scientific["ema_decay_per_optimizer_update"] = policy["ema_decay"]
    validate_policy(policy, scientific)
    execution.update(version=1, package="robustness_suite", arm=arm, policy=policy, scientific_settings=scientific,
                     control_roles={"fresh_matched_control": "baseline7m_p01", "historical_reference": execution["control"]})
    from experiments.robustness_suite.data import audit_inputs
    execution["data_audit"] = audit_inputs(execution)
    if policy["extra_training_crops"]:
        slots = copy.deepcopy(execution["data_audit"]["safe_train_slots"])
        if not execution["data_audit"]["eligible_extra_crop"] or len(slots) != 24:
            raise ValueError("extra_crop is blocked: no provenance-proven safe additional training crop")
        execution["training_slots"] = slots
    else:
        execution["training_slots"] = copy.deepcopy(execution["resolved_inputs"]["train"])
    sources = set(Path(row["path"]) for row in execution["file_identities"])
    sources.update((ROOT / "experiments/robustness_suite").glob("*.py"))
    sources.update(Path(row["path"]) for row in execution["data_audit"]["evidence_files"])
    execution["file_identities"] = sorted((file_identity(p) for p in sources), key=lambda row: row["path"])
    return execution


def prepare(output_dir=None, *, arm="baseline7m_p01"):
    output_dir = Path(output_dir or DEFAULT_OUT / arm).resolve()
    if CONTROL == output_dir or CONTROL in output_dir.parents:
        raise ValueError("output must be a new lineage outside the historical control")
    candidate = execution_identity(arm)
    path = output_dir / "execution_config.json"
    if path.exists():
        if read_json(path) != candidate:
            raise ValueError("immutable scientific/source/input identity mismatch")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError("new output directory must be empty")
        output_dir.mkdir(parents=True, exist_ok=True)
        for row in candidate["file_identities"]:
            source = Path(row["path"])
            if source.suffix == ".py" and source.is_relative_to(ROOT):
                target = output_dir / "source_snapshot" / source.relative_to(ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        atomic_write_json(path, candidate)
        atomic_write_json(output_dir / "budget_ledger.json", [{"at_utc": datetime.now(timezone.utc).isoformat(),
                          "event": "initial", "map_cap": candidate["policy"]["map_cap"], "render_cap": candidate["policy"]["render_cap"]}])
        atomic_write_json(output_dir / "status.json", {"status": "prepared", "stage": "map", "step": 0, "total_step": 0, "arm": arm})
    return output_dir, candidate
