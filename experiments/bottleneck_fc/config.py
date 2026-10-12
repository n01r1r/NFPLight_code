"""Strict configuration and immutable run identity helpers."""
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
ARM_CONTRACT = (
    ("baseline", "baseline", "identity"),
    ("bottleneck_fc", "baseline+Bottleneck_FC", "per_channel_shared_spatial_fc"),
    ("pixel_mlp", "baseline+Pixel_MLP", "per_pixel_channel_mlp"),
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


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")


def _string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _nonempty_number_list(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list")
    for value in values:
        _number(value, name, minimum=0)


def validate_config(config):
    _keys(config, (
        "schema_version", "experiment_id", "output", "dataset", "model",
        "sampling", "training", "loss", "rendering", "runtime", "provenance",
    ), "config")
    if config["schema_version"] != 1:
        raise ValueError(f"unsupported config schema_version: {config['schema_version']!r}")
    if config["experiment_id"] != "bottleneck_fc_probe":
        raise ValueError("experiment_id must be 'bottleneck_fc_probe'")

    output = config["output"]
    _keys(output, ("default_directory",), "output")
    _string(output["default_directory"], "output.default_directory")
    if Path(output["default_directory"]).is_absolute() or ".." in Path(output["default_directory"]).parts:
        raise ValueError("output.default_directory must stay under the repository root")

    dataset = config["dataset"]
    _keys(dataset, (
        "manifest", "shard_path_base", "train_split", "heldout_split", "categories",
        "materials_per_category", "train_crops", "unseen_crop", "heldout_crop",
        "crops_per_shard", "cache_shards", "map_keys", "diffuse_decode_gamma",
        "normal_decode", "roughness_decode", "specular_target",
    ), "dataset")
    _string(dataset["manifest"], "dataset.manifest")
    if dataset["shard_path_base"] != "repo_root":
        raise ValueError("dataset.shard_path_base must be 'repo_root'")
    _string(dataset["train_split"], "dataset.train_split")
    _string(dataset["heldout_split"], "dataset.heldout_split")
    if dataset["train_split"] == dataset["heldout_split"]:
        raise ValueError("training and heldout splits must differ")
    if dataset["categories"] != ["Fabric", "Leather"]:
        raise ValueError("dataset.categories must be ['Fabric', 'Leather']")
    if (dataset["train_split"], dataset["heldout_split"]) != ("train", "val"):
        raise ValueError("dataset splits must remain train and val")
    if dataset["materials_per_category"] != 4 or dataset["train_crops"] != [0, 8]:
        raise ValueError("the probe requires four materials/category and training crops [0,8]")
    if (dataset["unseen_crop"], dataset["heldout_crop"], dataset["crops_per_shard"]) != (4, 4, 16):
        raise ValueError("the probe requires unseen/heldout crop 4 of 16")
    if dataset["diffuse_decode_gamma"] != 2.2 or dataset["specular_target"] != 0.04:
        raise ValueError("the probe requires gamma 2.2 and F0 0.04")
    _integer(dataset["materials_per_category"], "dataset.materials_per_category", minimum=1)
    _integer(dataset["crops_per_shard"], "dataset.crops_per_shard", minimum=1)
    _integer(dataset["cache_shards"], "dataset.cache_shards", minimum=1)
    if dataset["map_keys"] != ["normal", "diffuse", "roughness", "specular"]:
        raise ValueError("dataset.map_keys must be normal, diffuse, roughness, specular")
    if not isinstance(dataset["train_crops"], list) or not dataset["train_crops"]:
        raise ValueError("dataset.train_crops must be a nonempty list")
    for crop in [*dataset["train_crops"], dataset["unseen_crop"], dataset["heldout_crop"]]:
        _integer(crop, "dataset crop", minimum=0)
        if crop >= dataset["crops_per_shard"]:
            raise ValueError(f"dataset crop index {crop} is outside the shard")
    if len(set(dataset["train_crops"])) != len(dataset["train_crops"]):
        raise ValueError("dataset.train_crops must not contain duplicates")
    _number(dataset["diffuse_decode_gamma"], "dataset.diffuse_decode_gamma", minimum=0.1)
    if dataset["normal_decode"] != "uint8/127.5-1 then unit normalize":
        raise ValueError("unsupported normal decode contract")
    if dataset["roughness_decode"] != "uint8/255":
        raise ValueError("unsupported roughness decode contract")
    _number(dataset["specular_target"], "dataset.specular_target", minimum=0, maximum=1)

    model = config["model"]
    _keys(model, ("backbone", "bottleneck", "baseline_parameters", "expected_parameters", "arms"), "model")
    backbone = model["backbone"]
    _keys(backbone, (
        "name", "width", "middle_blocks", "encoder_blocks", "decoder_blocks",
        "input_channels", "output_channels", "input_resolution",
    ), "model.backbone")
    if backbone["name"] != "TwoBranchNet":
        raise ValueError("model.backbone.name must be 'TwoBranchNet'")
    if (backbone["width"], backbone["middle_blocks"], backbone["encoder_blocks"], backbone["decoder_blocks"]) != (32, 4, [2, 2, 4], [2, 2, 2]):
        raise ValueError("the baseline TwoBranchNet shape is frozen")
    if (backbone["input_channels"], backbone["output_channels"], backbone["input_resolution"]) != (21, 10, 256):
        raise ValueError("the probe requires 21->10 maps at 256x256")
    for key in ("width", "middle_blocks", "input_channels", "output_channels", "input_resolution"):
        _integer(backbone[key], f"model.backbone.{key}", minimum=1)
    for key in ("encoder_blocks", "decoder_blocks"):
        values = backbone[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"model.backbone.{key} must be a nonempty integer list")
        for value in values:
            _integer(value, f"model.backbone.{key} entry", minimum=1)
    if len(backbone["encoder_blocks"]) != len(backbone["decoder_blocks"]):
        raise ValueError("encoder and decoder stage counts must match")
    if backbone["input_channels"] != 21 or backbone["output_channels"] != 10:
        raise ValueError("the frozen backbone requires 21 input and 10 output channels")
    if backbone["input_resolution"] % (2 ** len(backbone["encoder_blocks"])):
        raise ValueError("input_resolution must divide evenly through all encoder stages")
    side = backbone["input_resolution"] // (2 ** len(backbone["encoder_blocks"]))
    bottleneck = model["bottleneck"]
    _keys(bottleneck, (
        "channels", "side", "normalization", "residual_gate_initial",
        "spatial_fc_hidden", "pixel_mlp_hidden",
    ), "model.bottleneck")
    for key in ("channels", "side", "spatial_fc_hidden", "pixel_mlp_hidden"):
        _integer(bottleneck[key], f"model.bottleneck.{key}", minimum=1)
    if bottleneck["side"] != side:
        raise ValueError(
            f"model.bottleneck.side must equal input_resolution / 2^encoder_stages ({side})"
        )
    if bottleneck["channels"] != backbone["width"] * (2 ** len(backbone["encoder_blocks"])):
        raise ValueError("bottleneck channel count does not match the backbone")
    if bottleneck["normalization"] != "LayerNorm2d":
        raise ValueError("bottleneck normalization must be LayerNorm2d")
    if (bottleneck["channels"], bottleneck["side"], bottleneck["spatial_fc_hidden"], bottleneck["pixel_mlp_hidden"]) != (256, 32, 32, 128):
        raise ValueError("the bottleneck operators have frozen dimensions 256x32x32 / 32 / 128")
    _number(bottleneck["residual_gate_initial"], "model.bottleneck.residual_gate_initial")
    if bottleneck["residual_gate_initial"] != 0:
        raise ValueError("residual gates must start at zero for identical initial functions")
    _integer(model["baseline_parameters"], "model.baseline_parameters", minimum=1)
    expected = model["expected_parameters"]
    _keys(expected, tuple(row[0] for row in ARM_CONTRACT), "model.expected_parameters")
    for key, value in expected.items():
        _integer(value, f"model.expected_parameters.{key}", minimum=1)
    if expected != {"baseline": 7609834, "bottleneck_fc": 7741932, "pixel_mlp": 7741932}:
        raise ValueError("expected parameter counts are frozen at 7,609,834 / 7,741,932")
    arms = model["arms"]
    if not isinstance(arms, list) or len(arms) != len(ARM_CONTRACT):
        raise ValueError("model.arms must contain the three frozen arms in canonical order")
    for row, contract in zip(arms, ARM_CONTRACT):
        _keys(row, ("id", "label", "operator", "color"), "model arm")
        if (row["id"], row["label"], row["operator"]) != contract:
            raise ValueError(
                f"invalid arm contract for {contract[0]}: id, label, and operator are frozen"
            )
        if not isinstance(row["color"], str) or len(row["color"]) != 7 or not row["color"].startswith("#"):
            raise ValueError(f"model arm {row['id']} color must be #RRGGBB")
    if model["baseline_parameters"] != expected["baseline"]:
        raise ValueError("baseline parameter total disagrees with expected_parameters.baseline")

    sampling = config["sampling"]
    _keys(sampling, (
        "batch_size", "uniform_selection", "sample_order", "sample_order_seed_offset",
        "d4_augmentation",
    ), "sampling")
    _integer(sampling["batch_size"], "sampling.batch_size", minimum=1)
    if sampling["batch_size"] != 4:
        raise ValueError("sampling.batch_size must remain 4")
    _integer(sampling["sample_order_seed_offset"], "sampling.sample_order_seed_offset", minimum=0)
    if sampling["uniform_selection"] != "manifest-order evenly spaced indices; not loss-guided":
        raise ValueError("unsupported selection rule")
    if sampling["sample_order"] != "Python random.Random(seed+sample_order_seed_offset), paired across arms":
        raise ValueError("unsupported sample-order rule")
    if sampling["d4_augmentation"] is not False:
        raise ValueError("D4 augmentation must remain disabled for this fixed-fit probe")

    training = config["training"]
    _keys(training, (
        "seeds", "map_updates", "render_updates", "map_learning_rate",
        "render_learning_rate", "learning_rate_schedule", "adam_betas",
        "fresh_adam_at_render_stage", "gradient_clip_norm", "render_loss_weight",
        "checkpoint_interval", "evaluation_interval",
    ), "training")
    seeds = training["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("training.seeds must be a nonempty list")
    for seed in seeds:
        _integer(seed, "training seed", minimum=0)
    if len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds must be unique")
    for key in ("map_updates", "render_updates", "checkpoint_interval", "evaluation_interval"):
        _integer(training[key], f"training.{key}", minimum=1)
    for key in ("map_learning_rate", "render_learning_rate", "gradient_clip_norm", "render_loss_weight"):
        _number(training[key], f"training.{key}", minimum=0)
        if training[key] == 0:
            raise ValueError(f"training.{key} must be > 0")
    if training["learning_rate_schedule"] != "constant within each stage":
        raise ValueError("unsupported learning-rate schedule")
    betas = training["adam_betas"]
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("training.adam_betas must contain two values")
    for beta in betas:
        _number(beta, "training.adam_betas entry", minimum=0, maximum=1)
    if training["fresh_adam_at_render_stage"] is not True:
        raise ValueError("render stage must start with fresh Adam state")

    loss = config["loss"]
    _keys(loss, ("map_terms", "map_aggregation", "normal_degrees", "invalid_or_nonfinite"), "loss")
    if loss["map_terms"] != ["unit_xyz_normal_l1", "diffuse_rgb_l1", "roughness_l1"]:
        raise ValueError("unsupported map loss terms")
    if loss["map_aggregation"] != "equal_mean":
        raise ValueError("map terms must use equal mean aggregation")
    if loss["normal_degrees"] != "evaluation_only":
        raise ValueError("normal degrees are evaluation-only")
    if loss["invalid_or_nonfinite"] != "fail":
        raise ValueError("invalid or non-finite values must fail the run")

    rendering = config["rendering"]
    _keys(rendering, (
        "feature_version", "precision", "amp", "tf32", "near_distance", "far_distance",
        "gain", "f0", "training_lights", "novel_lights",
    ), "rendering")
    if rendering["feature_version"] != "raw_calibrated_gain9_v3":
        raise ValueError("feature_version must remain raw_calibrated_gain9_v3")
    if rendering["precision"] != "float32" or rendering["amp"] is not False or rendering["tf32"] is not False:
        raise ValueError("the probe requires FP32 with AMP and TF32 disabled")
    for key in ("near_distance", "far_distance", "gain"):
        _number(rendering[key], f"rendering.{key}", minimum=0)
        if rendering[key] == 0:
            raise ValueError(f"rendering.{key} must be > 0")
    _number(rendering["f0"], "rendering.f0", minimum=0, maximum=1)
    if (rendering["near_distance"], rendering["far_distance"], rendering["gain"], rendering["f0"]) != (4.0, 12.0, 9.0, 0.04):
        raise ValueError("the probe requires near/far/gain/F0 = 4/12/9/0.04")
    expected_lights = {
        "training_lights": {"radius": 4.0, "theta_degrees": [5, 10, 20, 30, 45], "phi_degrees": [0, 45, 90, 135, 180, 225, 270, 315], "view": [0.0, 0.0, 2.75]},
        "novel_lights": {"radius": 4.0, "theta_degrees": [12, 27, 40], "phi_degrees": [22.5, 112.5, 202.5, 292.5], "view": [0.0, 0.0, 2.75]},
    }
    for key, expected_values in expected_lights.items():
        if any(rendering[key][field] != value for field, value in expected_values.items()):
            raise ValueError(f"rendering.{key} lighting grid is frozen")
    for key in ("training_lights", "novel_lights"):
        lights = rendering[key]
        _keys(lights, ("radius", "theta_degrees", "phi_degrees", "view"), f"rendering.{key}")
        _number(lights["radius"], f"rendering.{key}.radius", minimum=0.001)
        _nonempty_number_list(lights["theta_degrees"], f"rendering.{key}.theta_degrees")
        if any(value > 90 for value in lights["theta_degrees"]):
            raise ValueError(f"rendering.{key}.theta_degrees must be in [0,90]")
        _nonempty_number_list(lights["phi_degrees"], f"rendering.{key}.phi_degrees")
        if any(value >= 360 for value in lights["phi_degrees"]):
            raise ValueError(f"rendering.{key}.phi_degrees must be in [0,360)")
        if not isinstance(lights["view"], list) or len(lights["view"]) != 3:
            raise ValueError(f"rendering.{key}.view must be a 3-vector")
        for value in lights["view"]:
            _number(value, f"rendering.{key}.view")
        if lights["view"][2] <= 0:
            raise ValueError(f"rendering.{key}.view must be above the surface")

    runtime = config["runtime"]
    _keys(runtime, ("device", "torch_num_threads", "cudnn_deterministic", "cudnn_benchmark"), "runtime")
    if runtime["device"] != "cuda":
        raise ValueError("runtime.device must remain cuda")
    _integer(runtime["torch_num_threads"], "runtime.torch_num_threads", minimum=1)
    if runtime["cudnn_deterministic"] is not True or runtime["cudnn_benchmark"] is not False:
        raise ValueError("runtime requires deterministic cuDNN with benchmarking disabled")

    provenance = config["provenance"]
    _keys(provenance, ("identity_method", "source_files"), "provenance")
    if provenance["identity_method"] != "resolved path, file size, and mtime_ns; no content hashes":
        raise ValueError("unsupported provenance identity method")
    sources = provenance["source_files"]
    if not isinstance(sources, list) or not sources or len(sources) != len(set(sources)):
        raise ValueError("provenance.source_files must be a nonempty unique list")
    for source in sources:
        _string(source, "provenance.source_files entry")
        if Path(source).is_absolute() or ".." in Path(source).parts:
            raise ValueError(f"source path must stay under the repository root: {source}")
    return config


def resolve_repo_path(value, *, repo_root=REPO_ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(repo_root) / path).resolve()


def load_config(path=None):
    config_path = resolve_repo_path(path or DEFAULT_CONFIG)
    config = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    return validate_config(config), config_path


def file_identity(path):
    resolved = Path(path).resolve(strict=True)
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def select_materials(records, split, categories, count):
    selected = []
    for category in categories:
        indices = [i for i, row in enumerate(records)
                   if row.get("split") == split and row.get("category") == category]
        if len(indices) < count:
            raise ValueError(
                f"manifest split {split!r} has {len(indices)} {category} materials; {count} required"
            )
        chosen = [indices[(i * (len(indices) - 1)) // (count - 1)] if count > 1 else indices[0]
                  for i in range(count)]
        selected.extend(records[i] for i in chosen)
    return selected


def _manifest_rows(path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: manifest rows must be objects")
        for key in ("split", "category", "material_id", "shard_path"):
            if key not in row:
                raise ValueError(f"{path}:{line_number}: missing {key}")
        rows.append(row)
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def build_execution_config(config, config_path, output_dir, *, repo_root=REPO_ROOT):
    validate_config(config)
    root = Path(repo_root).resolve()
    output_dir = resolve_repo_path(output_dir, repo_root=root)
    manifest = resolve_repo_path(config["dataset"]["manifest"], repo_root=root)
    if not manifest.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest}")
    records = _manifest_rows(manifest)
    data_cfg = config["dataset"]
    train = select_materials(
        records, data_cfg["train_split"], data_cfg["categories"], data_cfg["materials_per_category"]
    )
    heldout = select_materials(
        records, data_cfg["heldout_split"], data_cfg["categories"], data_cfg["materials_per_category"]
    )

    def slot_rows(materials, crops):
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

    train_slots = slot_rows(train, data_cfg["train_crops"])
    unseen_slots = slot_rows(train, [data_cfg["unseen_crop"]])
    heldout_slots = slot_rows(heldout, [data_cfg["heldout_crop"]])
    identity_paths = [config_path, manifest]
    identity_paths.extend(resolve_repo_path(path, repo_root=root) for path in config["provenance"]["source_files"])
    identity_paths.extend(Path(row["shard_path"]) for row in train_slots + heldout_slots)
    identities = {str(Path(path).resolve()): file_identity(path) for path in identity_paths}
    return {
        "execution_config_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository_root": str(root),
        "output_directory": str(output_dir),
        "config_source": file_identity(config_path),
        "settings": config,
        "resolved_inputs": {
            "manifest": str(manifest.resolve()),
            "train": train_slots,
            "unseen_crop": unseen_slots,
            "heldout": heldout_slots,
        },
        "file_identities": [identities[key] for key in sorted(identities)],
    }


def same_execution_config(saved, candidate):
    ignored = {"created_at_utc"}
    return (
        isinstance(saved, dict)
        and isinstance(candidate, dict)
        and {key: value for key, value in saved.items() if key not in ignored}
        == {key: value for key, value in candidate.items() if key not in ignored}
    )


def replace_with_retry(temporary, target, *, attempts=10):
    for attempt in range(attempts):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    replace_with_retry(temporary, path)


def prepare_run(config_path=None, output_dir=None):
    config, resolved_config_path = load_config(config_path)
    output_dir = resolve_repo_path(
        output_dir if output_dir is not None else config["output"]["default_directory"]
    )
    candidate = build_execution_config(config, resolved_config_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_path = output_dir / "config.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
        if not same_execution_config(saved, candidate):
            raise ValueError(f"immutable execution config mismatch: {saved_path}")
    else:
        atomic_write_json(saved_path, candidate)
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        atomic_write_json(metadata_path, {
            "metadata_version": 1,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "identity_method": config["provenance"]["identity_method"],
        })
    status_path = output_dir / "status.json"
    if not status_path.exists():
        atomic_write_json(status_path, {
            "status": "prepared",
            "pid": None,
            "stage": None,
            "total_step": 0,
            "last_committed_step": 0,
            "output_directory": str(output_dir),
        })
    return output_dir


def load_execution_config(output_dir):
    path = Path(output_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"prepared execution config not found: {path}")
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(config, dict) or config.get("execution_config_version") != 1:
        raise ValueError(f"unsupported execution config: {path}")
    validate_config(config.get("settings"))
    return config


def process_identity(pid):
    """Return a live process creation identity using query-only access."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        ]
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                return None
            creation, exit_time, kernel_time, user_time = FileTime(), FileTime(), FileTime(), FileTime()
            ok = kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            )
            if not ok:
                return None
            return (creation.high << 32) | creation.low
        finally:
            kernel32.CloseHandle(handle)
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="utf-8").rsplit(")", 1)[1].split()
            return fields[19]
        except (OSError, IndexError):
            return None
    return None


def process_identity_matches(pid, expected_identity):
    if expected_identity is None:
        return False
    current = process_identity(pid)
    return current is not None and str(current) == str(expected_identity)
