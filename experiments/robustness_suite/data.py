"""Train-only crop provenance audit and physical material augmentation.

No hashes, source modifications, official-test shards, or GPU work occur here.
Crop rectangles use native pixel y/x origins and half-open [y,y+edge) ranges.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch


CANDIDATE_CROPS = (1, 2, 3, 5, 6, 7)
AUGMENTATION_SETTINGS = {
    "diffuse_scale_uniform": [0.8, 1.2],
    "diffuse_scale_scope": "one scalar per item shared by RGB and all pixels",
    "roughness_offset_uniform": [-0.05, 0.05],
    "roughness_offset_scope": "one physical offset per item shared by all pixels",
    "physical_clamp": [0.0, 1.0],
    "normal": "unchanged; D4 vector-aware transform is owned by the pipeline",
    "specular": "unchanged here; pipeline enforces fixed F0=0.04",
    "rng": "dedicated Python random.Random(seed+30000); two draws per item",
    "feature_contract": "rerender augmented GT at fixed geometry before features",
    "selection": "fixed operational limits; no optimization of augmentation strength",
}


def _validate_target(target):
    if not isinstance(target, torch.Tensor) or target.ndim != 4 or target.shape[1] != 10 or min(target.shape) < 1:
        raise ValueError("target must be nonempty [B,10,H,W]")
    if target.dtype != torch.float32:
        raise ValueError("target must use FP32")
    if not torch.isfinite(target).all():
        raise ValueError("target must contain finite values")
    if (target < -1).any() or (target > 1).any():
        raise ValueError("target must lie in [-1,1]")
    lengths = target[:, :3].norm(dim=1)
    if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4, rtol=0):
        raise ValueError("target normals must be unit vectors")


def augment_material(target, rng):
    """Packed FP32 maps -> augmented packed maps and the actual per-item draws.

    Decode linear diffuse/physical roughness as (packed+1)/2; normals and
    specular remain byte-for-byte equal. Does not mutate target or global RNG.
    """
    _validate_target(target)
    if not isinstance(rng, random.Random):
        raise TypeError("augmentation requires a dedicated Python random.Random")
    draws = [{"diffuse_scale": rng.uniform(0.8, 1.2),
              "roughness_offset": rng.uniform(-0.05, 0.05)} for _ in range(target.shape[0])]
    scales = target.new_tensor([d["diffuse_scale"] for d in draws])[:, None, None, None]
    offsets = target.new_tensor([d["roughness_offset"] for d in draws])[:, None, None, None]
    result = target.clone()
    result[:, 3:6] = (((target[:, 3:6] + 1) / 2 * scales).clamp(0, 1) * 2 - 1)
    result[:, 6:7] = (((target[:, 6:7] + 1) / 2 + offsets).clamp(0, 1) * 2 - 1)
    _validate_target(result)
    return result, {"items": draws, "settings": copy.deepcopy(AUGMENTATION_SETTINGS)}


def overlap_pixels(first_yx, second_yx, edge):
    """Count shared native pixels; touching edges have no overlap."""
    if edge < 1 or len(first_yx) != 2 or len(second_yx) != 2:
        raise ValueError("rectangles require two coordinates and positive edge")
    return max(0, min(first_yx[0] + edge, second_yx[0] + edge) - max(first_yx[0], second_yx[0])) * max(
        0, min(first_yx[1] + edge, second_yx[1] + edge) - max(first_yx[1], second_yx[1]))


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _forbid_test(path):
    if any(part.lower() == "test" for part in Path(path).parts):
        raise ValueError(f"official test path is forbidden: {path}")


def _native_map(path, key):
    with Image.open(path) as image:
        result = np.asarray(image).copy()
    if key == "roughness" and result.dtype == np.uint16:
        result = np.rint(result.astype(np.float32) / 257).astype(np.uint8)
    if key != "roughness" and result.ndim == 3 and result.shape[2] == 4:
        result = result[:, :, :3].copy()
    if result.dtype != np.uint8:
        raise ValueError(f"native source must use uint8 (uint16 roughness allowed): {path}")
    return result


def _prove_coordinates(slot, record, crop_count, edge, identities):
    path = Path(slot["shard_path"]).resolve()
    _forbid_test(path)
    if record is None or record.get("official_split") != "train" or record.get("split") == "test":
        raise ValueError("selected shard needs an official-train manifest record")
    if Path(record["shard_path"]).resolve() != path:
        raise ValueError("selected slot and manifest shard identities differ")
    source = Path(record["source_dir"]).resolve()
    _forbid_test(source)
    identities.append(_identity(path))
    source_metadata = source / "metadata.json"
    if source_metadata.exists():
        identities.append(_identity(source_metadata))
    with np.load(path, allow_pickle=False) as archive:
        if "origins" not in archive.files or "resolution" not in archive.files:
            raise ValueError("missing crop origins/resolution provenance")
        origins = archive["origins"]
        if origins.shape != (crop_count, 2) or not np.issubdtype(origins.dtype, np.integer) or (origins < 0).any():
            raise ValueError("origins must be nonnegative integer [crop_count,2]")
        if archive["resolution"].shape != () or int(archive["resolution"]) != edge:
            raise ValueError("stored crop resolution differs from fixed input resolution")
        source_size = None
        verified = list(range(9))
        orientations = {"yx": origins, "xy": origins[:, ::-1]}
        possible_orders = set(orientations)
        # One original map and one shard map at a time bounds RAM at ~66 MiB.
        for key in ("normal", "diffuse", "roughness", "specular"):
            source_path = source / f"{key}.png"
            identities.append(_identity(source_path))
            native = _native_map(source_path, key)
            maps = archive[key]
            if maps.dtype != np.uint8 or maps.shape[0] != crop_count or maps.shape[1:3] != (edge, edge):
                raise ValueError(f"invalid shard map shape/dtype: {key}")
            if source_size is None:
                source_size = list(native.shape[:2])
            elif source_size != list(native.shape[:2]):
                raise ValueError("source map dimensions differ")
            for order in tuple(possible_orders):
                for crop in verified:
                    y, x = (int(v) for v in orientations[order][crop])
                    if (y + edge > native.shape[0] or x + edge > native.shape[1]
                            or not np.array_equal(maps[crop], native[y:y + edge, x:x + edge])):
                        possible_orders.remove(order)
                        break
            if not possible_orders:
                raise ValueError(f"crop {key} differs from stated source rectangles in both axis orders")
            del native, maps
        if len(possible_orders) > 1 and not np.array_equal(origins, origins[:, ::-1]):
            raise ValueError("crop coordinate axis order ambiguous after exact four-map matching")
        order = "yx" if "yx" in possible_orders else "xy"
        normalized = orientations[order]
        sidecar = path.with_suffix(".json")
        sidecar_info = None
        if sidecar.exists():
            identities.append(_identity(sidecar))
            details = json.loads(sidecar.read_text(encoding="utf-8"))
            coords = details.get("origins_yx", details.get("coordinates_yx"))
            if coords is not None and coords != normalized.tolist():
                raise ValueError("sidecar coordinates differ from NPZ")
            if details.get("source_size_hw") is not None and details["source_size_hw"] != source_size:
                raise ValueError("sidecar source dimensions differ from PNGs")
            sidecar_info = {"path": str(sidecar), "format": details.get("format"), "coordinates_match": coords is not None}
        return {"origins_yx": normalized.tolist(), "stored_origins": origins.tolist(),
                "stored_axis_order_verified": order, "source_size_hw": source_size,
                "crop_size_hw": [edge, edge], "verified_crops": verified,
                "verification": "exact four-map equality to original train PNG y/x slices; no hashes",
                "sidecar": sidecar_info, "source_dir": str(source),
                "group_id": record.get("group_id"), "source": record.get("source")}


def audit_inputs(execution):
    """Audit original16 and crop1/2/3/5/6/7; return balanced safe slots.

    Safety means no shared native pixels with original train0/8 or unseen4.
    This is spatial independence only; repeated material/textures remain shared.
    Unknown provenance blocks the complete extra-crop arm, never just a material.
    """
    inputs = execution["resolved_inputs"]
    original = copy.deepcopy(inputs["train"])
    unseen = inputs["unseen_crop"]
    heldout = inputs["heldout"]
    dataset = execution["scientific_settings"]["dataset"]
    crop_count = dataset["crops_per_shard"]
    edge = execution["scientific_settings"].get("backbone", {}).get("input_resolution", 256)
    if crop_count < 9:
        raise ValueError("crop audit requires at least nine stored crops")
    groups = {}
    for slot in original:
        _forbid_test(slot["shard_path"])
        groups.setdefault(slot["material_id"], []).append(slot)
    for material, slots in groups.items():
        if sorted(s["crop"] for s in slots) != [0, 8]:
            raise ValueError(f"fixed training slots must be crop0/8: {material}")
    if {s["material_id"] for s in heldout} & groups.keys():
        raise ValueError("heldout materials must differ from train materials")
    if {s["material_id"] for s in unseen} != groups.keys() or any(s["crop"] != 4 for s in unseen):
        raise ValueError("unseen population must be train materials crop4")
    manifest = Path(inputs["manifest"])
    identities = [_identity(manifest)]
    # Selected records only are retained; no official-test shards/maps are opened.
    needed = set(groups) | {s["material_id"] for s in heldout}
    records = {}
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("material_id") in needed and record.get("official_split") != "test" and record.get("split") != "test":
                records[record["material_id"]] = record
    train_groups = {records[m].get("group_id") for m in groups if m in records}
    heldout_groups = {records[s["material_id"]].get("group_id") for s in heldout if s["material_id"] in records}
    group_overlap = sorted((train_groups & heldout_groups) - {None})
    materials, extras, blockers, baseline_overlap = [], [], [], []
    if any(s["material_id"] not in records or not records[s["material_id"]].get("group_id") for s in heldout):
        blockers.append("heldout source-group manifest provenance missing")
    for material, slots in groups.items():
        row = {"material_id": material, "category": slots[0]["category"], "shard_path": slots[0]["shard_path"]}
        try:
            provenance = _prove_coordinates(slots[0], records.get(material), crop_count, edge, identities)
            row.update(provenance)
            origins = provenance["origins_yx"]
            baseline = [{"train_crop": crop, "unseen_crop": 4, "overlap_pixels": overlap_pixels(origins[crop], origins[4], edge)} for crop in (0, 8)]
            row["baseline_train_unseen_overlap"] = baseline
            baseline_overlap.extend(dict(material_id=material, **item) for item in baseline if item["overlap_pixels"] > 0)
            candidates = []
            for crop in CANDIDATE_CROPS:
                overlaps = {str(other): overlap_pixels(origins[crop], origins[other], edge) for other in (0, 8, 4)}
                candidates.append({"crop": crop, "origin_yx": origins[crop], "overlap_pixels_by_crop": overlaps,
                                   "verified_safe": not any(overlaps.values())})
            row["candidates"] = candidates
            safe = [c["crop"] for c in candidates if c["verified_safe"]]
            row["selected_extra_crop"] = safe[0] if safe else None
            if safe:
                extras.append(dict(slots[0], crop=safe[0]))
            else:
                blockers.append(f"{material}: no candidate independent of train0/8 and unseen4")
        except (OSError, ValueError, KeyError) as error:
            row.update(provenance_status="unknown", selected_extra_crop=None, cause=str(error))
            blockers.append(f"{material}: {error}")
        materials.append(row)
    if group_overlap:
        blockers.append(f"train/heldout manifest source groups overlap: {group_overlap}")
    eligible = len(extras) == len(groups) and not blockers
    evidence = {item["path"]: item for item in identities}
    generator = Path(__file__).resolve().parents[2] / "prepare_fabric_leather_subset.py"
    if generator.exists():
        evidence[str(generator)] = _identity(generator)
    return {"version": 1, "identity_method": "resolved path, size, mtime_ns; no content hashes",
            "candidate_crops": list(CANDIDATE_CROPS), "selection": "first verified safe candidate per material in numeric order; exactly one extra per material",
            "materials": materials, "eligible_extra_crop": eligible, "blocking_reasons": blockers,
            "safe_train_slots": original + (extras if eligible else []), "selected_balanced_extras": extras if eligible else [],
            "original_train_slot_count": len(original), "extra_train_slot_count": len(extras) if eligible else 0,
            "baseline_train_unseen_overlap": baseline_overlap, "heldout_source_group_overlap": group_overlap,
            "heldout_scope": "different materials checked by manifest identity only; heldout maps not opened by audit",
            "official_test_access": "no official-test shards, images, or maps opened; manifest selected non-test records only",
            "claim_limit": "spatial disjointness is not independent material or texture evidence; baseline overlapping crops are retained and disclosed",
            "supported_alternative": "keep original16 slots and use the separately authorized physical augmentation arm if extra-crop provenance fails",
            "augmentation_settings": copy.deepcopy(AUGMENTATION_SETTINGS), "evidence_files": list(evidence.values())}


def additional_training_slots(execution, auditpath=None):
    """Return original slots plus balanced verified extras; blocked means original only."""
    if auditpath is None:
        audit = audit_inputs(execution)
    else:
        audit = json.loads(Path(auditpath).read_text(encoding="utf-8"))
        if audit.get("version") != 1 or audit.get("safe_train_slots", [])[:len(execution["resolved_inputs"]["train"])] != execution["resolved_inputs"]["train"]:
            raise ValueError("audit does not match execution original train slots")
        extras = audit["safe_train_slots"][len(execution["resolved_inputs"]["train"]):]
        materials = {slot["material_id"] for slot in execution["resolved_inputs"]["train"]}
        if audit.get("eligible_extra_crop"):
            if len(extras) != len(materials) or {slot["material_id"] for slot in extras} != materials:
                raise ValueError("eligible crop audit must include exactly one extra per material")
            verified = {row["material_id"]: row["selected_extra_crop"] for row in audit["materials"]}
            if any(slot["crop"] != verified.get(slot["material_id"]) for slot in extras):
                raise ValueError("saved crop slots differ from verified balanced selection")
        elif extras:
            raise ValueError("blocked crop audit cannot expose extra training slots")
        for identity in audit["evidence_files"]:
            if _identity(identity["path"]) != identity:
                raise ValueError(f"crop audit evidence changed: {identity['path']}")
    return copy.deepcopy(audit["safe_train_slots"])
