"""User-authorized full-data arms; no dependence on discarded checkpoints."""
from pathlib import Path
import json
import zipfile
from collections import Counter
import numpy as np
from model.relation_geometry import RELATION_VERSION
from experiments.regularization.config import atomic_write_json

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path("C:/MatSynth_PRIMA_Fabric/manifests/fabric_leather_training_manifest.jsonl")
ARMS = ("baseline7m_p01", "capacity15m_p01", "dropout7m_p05")
TRAIN_LIGHTS = {"radius":4., "theta_degrees":[5,10,20,30,45],
                "phi_degrees":list(range(0,360,45)), "view":[0.,0.,2.75]}
NOVEL_LIGHTS = {"radius":4., "theta_degrees":[12,27,40],
                "phi_degrees":[22.5,112.5,202.5,292.5], "view":[0.,0.,2.75]}

def policy_for_arm(arm):
    if arm not in ARMS:
        raise ValueError("only supported full-dataset arms are permitted")
    return {"arm":arm, "seed":19, "parameters":15085034 if arm==ARMS[1] else 7609834,
            "middle_blocks":12 if arm==ARMS[1] else 4,
            "dropout_probability":.5 if arm==ARMS[2] else .1,
            "optimizer":"Adam", "weight_decay":0., "ema_decay":None,
            "map_cap":400000, "render_cap":100000,
            "evaluation_interval":1000, "checkpoint_interval":1000,
            "floor_observations":20, "map_lr":5e-4, "map_min_lr":1e-5,
            "render_lr":1e-4, "render_min_lr":1e-6, "factor":.5,
            "patience":5, "threshold":.005, "threshold_mode":"rel",
            "cooldown":2, "eps":1e-8, "mode":"min", "render_ndr_guard":1.02,
            "map_monitor":"heldout.ndrs_sum", "render_monitor":"heldout.render_l1",
            "cap_action":"pause phase; never force transition or claim convergence"}

def dataset_rows(manifest=MANIFEST):
    rows=[json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
    train=[r for r in rows if r["split"]=="train"]
    val=[r for r in rows if r["split"]=="val"]
    if len(train)!=566 or len(val)!=164:
        raise ValueError("full train566/val164 required")
    if len({r["material_id"] for r in train+val})!=730:
        raise ValueError("duplicate sample folder identity")
    if {r["group_id"] for r in train} & {r["group_id"] for r in val}:
        raise ValueError("source groups cross train/validation")
    if any("test" in Path(r["shard_path"]).parts for r in train+val):
        raise ValueError("official test payload forbidden")
    expected = {("train", "Fabric"):358, ("train", "Leather"):208,
                ("val", "Fabric"):98, ("val", "Leather"):66}
    if Counter((r["split"], r["category"]) for r in train+val) != expected:
        raise ValueError("full Fabric/Leather category populations differ")
    paths = [str(Path(r["shard_path"]).resolve()).casefold() for r in train+val]
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate shard payload across sample folders")
    return train,val

def file_identity(path):
    path=Path(path).resolve();stat=path.stat()
    return {"path":str(path),"size":stat.st_size,"mtime_ns":stat.st_mtime_ns}

def execution_identity(arm, *, purpose="training"):
    if purpose not in {"training", "smoke"}:
        raise ValueError("purpose must be training or smoke")
    train,val=dataset_rows()
    policy=policy_for_arm(arm)
    sources=set((ROOT/"experiments/full_dataset_suite").glob("*.py"))
    for folder in ("model","network","utils","data","experiments/bottleneck_fc",
                   "experiments/regularization","experiments/plateau_followup","experiments/robustness_suite"):
        sources.update((ROOT/folder).glob("*.py"))
    sources.update(ROOT/name for name in ("train_fabric.py","benchmark_fabric_leather.py",
        "experiments/regularization/run.py","experiments/regularization/config.py",
        "experiments/robustness_suite/run.py","experiments/bottleneck_fc/pipeline.py",
        "experiments/bottleneck_fc/config.py"))
    inputs={MANIFEST,*(Path(r["shard_path"]) for r in train+val)}
    return {"version":1,"package":"full_dataset_suite", "purpose":purpose, "authority":"2026-10-16 spatial relation preparation; inherited arm settings; no automatic launch authorization",
            "policy":policy,"manifest":str(MANIFEST),"train":train,"validation":val,
            "scientific_settings":{"dataset":{"categories":["Fabric","Leather"],"train_sample_folders":566,
                "validation_sample_folders":164,"train_crop_indices":list(range(16)),"train_slots":9056,
                "validation_crop_indices":[0,4,8,12],"sampling":"uniform full slots; batch4 without replacement per update, replacement across updates",
                "family_grouping":"all folder variants retained; groups used only for split separation"},
                "backbone":{"width":32,"encoder_blocks":[2,2,4],"decoder_blocks":[2,2,2],"middle_blocks":policy["middle_blocks"]},
                "rendering":{"relation_version":RELATION_VERSION,"precision":"float32","amp":False,"tf32":False,"feature_version":"raw_calibrated_gain9_v3",
                    "near":4.,"far":12.,"gain":9.,"f0":.04,"training_lights":TRAIN_LIGHTS,"novel_lights":NOVEL_LIGHTS},
                "loss":{"normal":"unit XYZ L1","map_terms":["N","D","R","S"],"map_aggregation":"sum",
                    "specular_target":.04,"render_weight":1.,"render_guard":"NDRS sum <= best-map sum x1.02"}},
            "file_identities":sorted([file_identity(p) for p in sources|inputs],key=lambda x:x["path"]),
            "official_test_payloads_read":0,"sha256_computed":False}

def prepare(output,arm, *, purpose="training"):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"artifacts"):
        raise ValueError("new run must remain in workspace artifacts")
    candidate=execution_identity(arm, purpose=purpose)
    path=output/"execution_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))!=candidate:
        raise ValueError("immutable source/input contract changed")
    output.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        data_check = inspect_shard_headers(candidate["train"] + candidate["validation"])
        atomic_write_json(output / "data_preflight.json", data_check)
        atomic_write_json(path,candidate)
    return candidate


def inspect_shard_headers(rows):
    """Check every training/validation map header without loading test payloads."""
    shapes = {"normal": {(16,256,256,3)}, "diffuse": {(16,256,256,3)},
              "roughness": {(16,256,256), (16,256,256,1)}, "specular": {(16,256,256,3)}}
    total_bytes = 0
    for row in rows:
        if row["split"] not in {"train", "val"}:
            raise ValueError("preflight forbids official test payloads")
        path = Path(row["shard_path"])
        total_bytes += path.stat().st_size
        with zipfile.ZipFile(path) as archive:
            for key, allowed in shapes.items():
                with archive.open(key + ".npy") as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1,0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2,0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(f"unsupported NPY header: {path}/{key}: {version}")
                if shape not in allowed or dtype != np.dtype("uint8"):
                    raise ValueError(f"invalid bank map: {path}/{key}: {shape}, {dtype}")
    return {"passed": True, "sample_folders":len(rows), "map_headers":len(rows)*len(shapes),
            "shard_bytes":total_bytes, "official_test_payloads_read":0, "sha256_computed":False,
            "scope":"map names, uint8 dtype and 16 native 256 crops; pixel contents checked by bounded GPU smoke"}
