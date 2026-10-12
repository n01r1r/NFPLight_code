"""Read-only FP32 simulation stress and pre-backward loss-gradient diagnostics.

Perturbations affect linear near/far observations before the existing gain9 feature
builder. They do not change GT or support and are not measured camera calibration.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.bottleneck_fc.pipeline import (
    Bank, fixed_f0, make_lights, masked_l1, ndr_loss, prepare_features,
    render_batch, render_error, render_input_images_batched, top_view_positions,
    unit_prediction,
)
from experiments.regularization.config import atomic_write_json, file_identity

ROOT = Path(__file__).resolve().parents[2]
FOLLOWUP = ROOT / "artifacts/plateau_followup_7m_seed19_20261009"
OLD_FINAL = ROOT / "artifacts/regularization_probe_20261009_checked/seed19_d4_dropout"
NOISE_CONTRACT = {
    "sigma": 0.005, "units": "linear RGB observation intensity in [0,1]",
    "representation": "raw_calibrated_gain9_v3", "near_distance": 4.0,
    "far_distance": 12.0, "gain": 9.0, "distribution": "zero-mean iid Gaussian before clipping",
    "clamp": [0.0, 1.0], "valid_mask": "unchanged; invalid observations remain zero",
    "rng": "dedicated CPU torch.Generator(seed+40000); checkpoint get_state/set_state",
    "claim_scope": "simulation stress; not real calibrated camera noise",
}
CONDITIONS = (
    {"id": "clean", "axis": "clean"},
    *({"id": f"gaussian_{sigma:g}", "axis": "gaussian", "sigma": sigma} for sigma in (0.001, 0.005, 0.01)),
    {"id": "pixel_dropout_0.01", "axis": "pixel_dropout", "fraction": 0.01},
    {"id": "saturation_1.2", "axis": "saturation", "multiplier": 1.2},
)
METRICS = ("ndr_mean", "map_normal", "map_diffuse", "map_roughness", "normal_degrees", "render_l1", "near_l1")


def make_noise_generator(seed):
    """Return an independent CPU stream; callers persist get_state()/set_state()."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("noise seed must be a nonnegative integer")
    return torch.Generator(device="cpu").manual_seed(seed + 40000)


def _support(values, valid, channels):
    if values.ndim != 4 or values.shape[0] < 1 or values.shape[1] != channels or values.dtype != torch.float32:
        raise ValueError(f"expected nonempty FP32 [B,{channels},H,W]")
    if valid.dtype != torch.bool or valid.shape != (values.shape[0], 1, *values.shape[-2:]) or valid.device != values.device:
        raise ValueError("valid must be bool [B,1,H,W] on the input device")
    if not valid.flatten(1).any(1).all():
        raise ValueError("every image needs valid pixels")
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite diagnostic input")


def perturb_observations(observations, valid, condition, generator):
    """FP32 [B,6,H,W] linear RGB -> same shape/range; one stress axis per call.

    Pixel dropout sets exactly floor(H*W*fraction) spatial pixels per image to
    zero in all six channels. The GT support is retained, including dropped pixels.
    """
    _support(observations, valid, 6)
    if observations.min() < 0 or observations.max() > 1:
        raise ValueError("observations must be linear RGB in [0,1]")
    if not isinstance(generator, torch.Generator) or str(generator.device) != "cpu":
        raise ValueError("stress requires a dedicated CPU torch.Generator")
    axis = condition.get("axis")
    expected = {"clean": {"id", "axis"}, "gaussian": {"id", "axis", "sigma"},
                "pixel_dropout": {"id", "axis", "fraction"}, "saturation": {"id", "axis", "multiplier"}}
    if axis not in expected or set(condition) - expected[axis]:
        raise ValueError("unknown or mixed stress axes")
    result = observations.clone()
    if axis == "gaussian":
        sigma = condition["sigma"]
        if isinstance(sigma, bool) or not math.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if sigma:
            noise = torch.randn(result.shape, generator=generator, dtype=torch.float32, device="cpu")
            result = (result + noise.to(result.device) * sigma).clamp(0, 1)
    elif axis == "pixel_dropout":
        fraction = condition["fraction"]
        if isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("pixel dropout fraction must be in [0,1]")
        pixels = result.shape[-2] * result.shape[-1]
        count = int(pixels * fraction)
        for image in result:
            indices = torch.randperm(pixels, generator=generator, device="cpu")[:count].to(result.device)
            image.reshape(6, pixels)[:, indices] = 0
    elif axis == "saturation":
        multiplier = condition["multiplier"]
        if isinstance(multiplier, bool) or not math.isfinite(multiplier) or multiplier < 1:
            raise ValueError("saturation multiplier must be finite and >=1")
        result = (result * multiplier).clamp(0, 1)
    return result * valid


def noise_features(model, target, valid, generator, params):
    """Render canonical packed GT [B,10,H,W], add Gaussian noise, build features.

    target is prepare_features' safe fixed-F0 output. params['sigma'] defaults to
    the approved .005; optional params['rendering'] must preserve near4/far12/gain9.
    No network forward, optimizer step, or global RNG is used.
    """
    _support(target, valid, 10)
    rendering = params.get("rendering")
    if rendering is not None:
        contract = {"near_distance": 4.0, "far_distance": 12.0, "gain": 9.0,
                    "feature_version": "raw_calibrated_gain9_v3", "precision": "float32"}
        if any(rendering.get(key) != value for key, value in contract.items()):
            raise ValueError("noise requires the fixed FP32 near4/far12/gain9 contract")
    if hasattr(model, "feature_version") and model.feature_version != NOISE_CONTRACT["representation"]:
        raise ValueError("noise features require raw_calibrated_gain9_v3")
    with torch.no_grad():
        observations = render_input_images_batched(model, target) * valid
        changed = perturb_observations(observations, valid,
            {"axis": "gaussian", "sigma": params.get("sigma", 0.005)}, generator)
        features = model.build_features(changed, valid_mask=valid)
    features = features[0] if isinstance(features, tuple) else features
    if not torch.isfinite(features).all():
        raise FloatingPointError("non-finite noise features")
    return features


def gradient_metrics(loss_components, named_parameters):
    """Measure pre-clip parameter gradients without writing .grad or stepping.

    Norms are global parameter-space L2; cosine is dot(map,render)/(norms).
    Components are passed with their objective weights already applied. A constant
    stage-map render loss has norm/cosine None. Zero norms make cosine undefined.
    GPU gradients are transferred parameter by parameter, never concatenated.
    """
    names = ("normal", "diffuse", "roughness", "map", "render")
    if set(loss_components) != set(names):
        raise ValueError(f"expected loss components {names}")
    parameters = [value for _, value in named_parameters if value.requires_grad]
    if not parameters:
        raise ValueError("gradient diagnostics need trainable parameters")
    norms, undefined, map_gradients = {}, {}, None
    cosine = None
    for name in names:
        loss = loss_components[name]
        if not torch.is_tensor(loss) or loss.numel() != 1:
            raise ValueError("each loss component must be a scalar tensor")
        if not torch.isfinite(loss.detach()).all():
            raise FloatingPointError(f"non-finite {name} loss")
        if not loss.requires_grad:
            norms[name] = None
            undefined[name] = "loss has no gradient graph (inactive stage component)"
            continue
        grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        squared, dot = 0.0, 0.0
        saved = [] if name == "map" else None
        any_used = False
        for index, grad in enumerate(grads):
            if grad is None:
                if saved is not None:
                    saved.append(None)
                continue
            any_used = True
            values = grad.detach().to(device="cpu", dtype=torch.float64)
            if not torch.isfinite(values).all():
                raise FloatingPointError(f"non-finite {name} gradient")
            squared += float(values.square().sum())
            if saved is not None:
                saved.append(values)
            if name == "render" and map_gradients is not None and map_gradients[index] is not None:
                dot += float((values * map_gradients[index]).sum())
        del grads
        norms[name] = math.sqrt(squared) if any_used else None
        if not any_used:
            undefined[name] = "no trainable parameter is connected"
        if name == "map":
            map_gradients = saved
        if name == "render" and norms[name] is not None and norms.get("map") is not None:
            denominator = norms[name] * norms["map"]
            if denominator > 0:
                cosine = max(-1.0, min(1.0, dot / denominator))
    if cosine is None:
        undefined["map_render_cosine"] = "inactive/disconnected component or zero gradient norm"
    return {"finite": True, "norms": norms, "map_render_cosine": cosine, "undefined": undefined,
            "trainable_parameters": sum(value.numel() for value in parameters),
            "definition": "pre-clip global parameter L2; weighted components; loss values alone do not establish gradient dominance"}


def prediction_disagreement(left, right, valid):
    """N/D/R parameter L1 proxies between models; no calibrated uncertainty claim."""
    return {key: float(masked_l1(left[:, section], right[:, section], valid))
            for key, section in (("normal", slice(0, 3)), ("diffuse", slice(3, 6)), ("roughness", slice(6, 7)))}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _settings(execution):
    scientific = execution.get("scientific_settings") or execution["experiment_settings"]["scientific_settings"]
    return {"dataset": scientific["dataset"], "model": {"backbone": scientific["backbone"]},
            "rendering": scientific["rendering"], "runtime": scientific["runtime"]}


def _memory_guard():
    from experiments.plateau_followup.run import memory_is_low, memory_observation
    observation = memory_observation()
    if memory_is_low(observation):
        raise MemoryError(f"diagnostics batch-one memory guard: {observation}")


def _metrics(model, pred, target, valid, lights, rendering):
    map_loss, parts = ndr_loss(pred, target, valid)
    angle = torch.rad2deg(torch.acos((pred[:, :3] * target[:, :3]).sum(1, keepdim=True).clamp(-1, 1))).masked_select(valid).mean()
    aggregate = render_error(model, pred, target, valid, lights, rendering)
    views = top_view_positions(len(lights), rendering["training_lights"]["view"], device=model.device)
    difference = (render_batch(model, fixed_f0(pred, rendering["f0"]), lights, views)
                  - render_batch(model, target, lights, views)).abs()
    light_values = [float(image.masked_select(valid.expand_as(image)).mean()) for image in difference.unbind(1)]
    near_pred = render_input_images_batched(model, fixed_f0(pred, rendering["f0"]))[:, :3]
    near_gt = render_input_images_batched(model, target)[:, :3]
    near = (near_pred - near_gt).abs().masked_select(valid.expand_as(near_pred)).mean()
    values = [float(map_loss), *(float(part) for part in parts), float(angle), float(aggregate), float(near)]
    if not all(math.isfinite(value) for value in values + light_values):
        raise FloatingPointError("non-finite diagnostic evaluation")
    return {**dict(zip(METRICS, values)), "novel_light_l1": light_values}


def assert_clean_reproduction(actual, expected, *, atol=2e-6, rtol=2e-5):
    """Compare every original metric and material identity, never just population mean."""
    for population, rows in actual.items():
        reference = expected[population]["materials"]
        if len(rows) != len(reference):
            raise AssertionError("clean material population differs")
        for row, old in zip(rows, reference):
            if any(row[key] != old[key] for key in ("material_id", "category", "crop")):
                raise AssertionError("clean material identities differ")
            for key in METRICS:
                if not math.isclose(row[key], old[key], abs_tol=atol, rel_tol=rtol):
                    raise AssertionError(f"clean reproduction failed: {population}/{row['material_id']}/{key}: {row[key]} vs {old[key]}")
    return {"passed": True, "atol": atol, "rtol": rtol, "metrics": list(METRICS), "scope": "all 16 materials; all original seven metrics"}


def _default_specs():
    return [
        {"label": "selected_map500", "checkpoint": FOLLOWUP / "selected.pt",
         "execution": FOLLOWUP / "execution_config.json", "reference": FOLLOWUP / "evaluation.json"},
        {"label": "old_final", "checkpoint": OLD_FINAL / "last.pt",
         "execution": OLD_FINAL.parent / "execution_config.json", "reference": OLD_FINAL / "evaluation.json", "reference_index": -1,
         "identity_reference": OLD_FINAL.parent / "baseline_reference.json"},
    ]


def _correlation(pairs):
    if len(pairs) < 2:
        return None
    x = [pair[0] for pair in pairs]
    y = [pair[1] for pair in pairs]
    dx = [value - sum(x) / len(x) for value in x]
    dy = [value - sum(y) / len(y) for value in y]
    denominator = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denominator if denominator else None


def evaluate(output, *, checkpoint_specs=None, seed=19, runroot=None):
    """Evaluate fixed validation crops batch-one, write primitive JSON to a new directory.

    checkpoint_specs is a list of {label,checkpoint,execution,reference,
    optional reference_index} mappings. Default specs are the two frozen existing weights.
    Custom checkpoints require a matching clean evaluation reference.
    """
    from experiments.regularization.run import RegularizationModel, _set_deterministic_settings
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("diagnostic output directory must be new or empty")
    specs = copy.deepcopy(checkpoint_specs or _default_specs())
    if runroot is not None:
        specs.extend(_suite_specs(runroot))
    if len({spec["label"] for spec in specs}) != len(specs):
        raise ValueError("checkpoint labels must be unique")
    files = {Path(__file__).resolve()}
    for spec in specs:
        files.update(Path(spec[key]).resolve() for key in ("checkpoint", "execution", "reference"))
        if "identity_reference" in spec:
            files.add(Path(spec["identity_reference"]).resolve())
    executions = [_read(spec["execution"]) for spec in specs]
    base_inputs = executions[0]["resolved_inputs"]
    for execution in executions:
        if any(execution["resolved_inputs"][key] != base_inputs[key] for key in ("heldout", "unseen_crop", "manifest")):
            raise ValueError("checkpoint populations must match exactly")
    data_paths = {str(Path(base_inputs["manifest"]).resolve())}
    data_paths.update(str(Path(row["shard_path"]).resolve()) for name in ("heldout", "unseen_crop") for row in base_inputs[name])
    for execution in executions:
        historical = {row["path"]: row for row in execution["file_identities"]}
        for path in data_paths:
            if file_identity(path) != historical.get(path):
                raise ValueError(f"frozen input identity changed: {path}")
    for execution in executions:
        files.update(Path(row["path"]) for row in execution["file_identities"] if Path(row["path"]).suffix == ".py")
    files.add(Path(base_inputs["manifest"]))
    files.update(Path(row["shard_path"]) for name in ("heldout", "unseen_crop") for row in base_inputs[name])
    identities = sorted((file_identity(path) for path in files), key=lambda row: row["path"])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"schema_version": 1, "seed": seed, "precision": "float32", "amp": False, "tf32": False,
        "dropout": "off", "batch_size": 1, "cpu_prediction_cache_max_images": 96, "conditions": list(CONDITIONS), "observation_contract": NOISE_CONTRACT,
        "scientific_settings_per_model": {spec["label"]: _settings(execution) for spec, execution in zip(specs, executions)},
        "stress_rng": "CPU Generator(seed+slot ordinal+40000), reset per axis/severity/model; paired standardized Gaussian draws",
        "pixel_dropout": "floor(H*W*fraction) pixels per image; all 6 raw RGB channels; GT support unchanged",
        "checkpoint_specs": [{key: str(value) if isinstance(value, Path) else value for key, value in spec.items()} for spec in specs],
        "inputs": {key: base_inputs[key] for key in ("manifest", "heldout", "unseen_crop")}, "identities": identities,
        "identity_method": "resolved path/size/mtime_ns; no hashes", "python": platform.python_version(), "torch": str(torch.__version__),
        "claim_scope": "fixed validation reuse; synthetic stress one axis at a time; disagreement is descriptive, not calibrated uncertainty or labeled OOD detection"}
    atomic_write_json(output / "config.json", metadata)
    results, reference_predictions, disagreement = {}, {}, []
    try:
        for model_index, (spec, execution) in enumerate(zip(specs, executions)):
            _memory_guard()
            settings = _settings(execution)
            rendering = settings["rendering"]
            expected = {"near_distance": 4.0, "far_distance": 12.0, "gain": 9.0, "feature_version": "raw_calibrated_gain9_v3", "precision": "float32"}
            if any(rendering.get(key) != value for key, value in expected.items()):
                raise ValueError("checkpoint observation representation differs")
            _set_deterministic_settings({"runtime": settings["runtime"]})
            device = settings["runtime"]["device"]
            model = RegularizationModel(settings, 0.0, device)
            # Explicit local historical artifacts contain numpy RNG state; never load remote pickles.
            checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False)
            expected_identity = execution
            if "identity_reference" in spec:
                from experiments.regularization.run import _execution_identity
                expected_identity = _execution_identity(execution, _read(spec["identity_reference"]))
            if checkpoint.get("identity") != expected_identity:
                raise ValueError("checkpoint frozen execution identity differs")
            model.net_g.load_state_dict(checkpoint["net_g"])
            reference_history = _read(spec["reference"])
            if "reference_index" in spec:
                reference = reference_history[spec["reference_index"]]["populations"]
            else:
                matches = [row for row in reference_history if all(row.get(key) == checkpoint.get(key)
                           for key in ("stage", "step", "total_step"))]
                if len(matches) != 1:
                    raise ValueError("selected checkpoint has no unique matching clean history row")
                reference = matches[0]["populations"]
            del checkpoint
            model.net_g.eval()
            lights = make_lights(rendering["novel_lights"], model.device)
            if len(lights) != 12:
                raise ValueError("diagnostics require all 12 frozen novel lights")
            bank = Bank(base_inputs["manifest"], settings["dataset"])
            per_condition = {condition["id"]: {name: [] for name in ("heldout", "unseen_crop")} for condition in CONDITIONS}
            with torch.no_grad():
                slot_index = 0
                for population in ("heldout", "unseen_crop"):
                    if len(base_inputs[population]) != 8:
                        raise ValueError("diagnostics require eight materials per population")
                    for slot in base_inputs[population]:
                        _memory_guard()
                        safe, valid, clean_features = prepare_features(model, bank.crop(slot)[None], rendering["f0"])
                        clean_observations = render_input_images_batched(model, safe) * valid
                        for condition in CONDITIONS:
                            if condition["axis"] == "clean":
                                features = clean_features
                            else:
                                generator = make_noise_generator(seed + slot_index)
                                observations = perturb_observations(clean_observations, valid, condition, generator)
                                features = model.build_features(observations, valid_mask=valid)
                                features = features[0] if isinstance(features, tuple) else features
                            pred = unit_prediction(model.net_g(features))
                            row = {key: slot[key] for key in ("material_id", "category", "crop")}
                            row.update(_metrics(model, pred, safe, valid, lights, rendering))
                            per_condition[condition["id"]][population].append(row)
                            key = (condition["id"], population, slot_index)
                            if model_index == 0 and len(specs) > 1:
                                reference_predictions[key] = pred.cpu()
                            elif model_index == 1:
                                proxies = prediction_disagreement(reference_predictions.pop(key), pred.cpu(), valid.cpu())
                                disagreement.append({"condition": condition["id"], "population": population,
                                    **{name: row[name] for name in ("material_id", "category", "crop")}, "proxies": proxies,
                                    "actual_errors": {specs[0]["label"]: {name: results[specs[0]["label"]]["conditions"][condition["id"]][population][slot_index % 8][f"map_{name}"] for name in proxies},
                                                      spec["label"]: {name: row[f"map_{name}"] for name in proxies}}})
                            del pred
                        slot_index += 1
                        del safe, valid, clean_features, clean_observations, features
                        bank.cache.clear()
            reproduction = assert_clean_reproduction(per_condition["clean"], reference)
            means = {condition: {name: {metric: sum(row[metric] for row in rows) / len(rows) for metric in METRICS}
                                 for name, rows in populations.items()} for condition, populations in per_condition.items()}
            results[spec["label"]] = {"conditions": per_condition, "means": means, "clean_reproduction": reproduction}
            atomic_write_json(output / "metrics.json", results)
            del model, bank
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        relationship = {}
        for condition in CONDITIONS:
            selected = [row for row in disagreement if row["condition"] == condition["id"]]
            relationship[condition["id"]] = {label: {component: _correlation([(row["proxies"][component], row["actual_errors"][label][component]) for row in selected])
                                                    for component in ("normal", "diffuse", "roughness")}
                                            for label in (specs[0]["label"], specs[1]["label"])} if len(specs) >= 2 else {}
        atomic_write_json(output / "disagreement.json", {"rows": disagreement, "pearson_with_actual_component_error": relationship,
            "definition": "same-image model N/D/R L1 difference and actual GT error; Pearson descriptive over 16 fixed crops per condition; constant vectors -> null",
            "limitation": "No labeled OOD validation set, calibrated uncertainty, false-positive guarantee, or independent test claim."})
        if any(file_identity(row["path"]) != row for row in identities):
            raise RuntimeError("input/source/checkpoint identity changed during read-only evaluation")
        atomic_write_json(output / "complete.json", {"status": "complete", "models": list(results), "conditions": len(CONDITIONS),
            "materials_per_model_condition": 16, "novel_lights": lights.cpu().tolist(), "source_identities_unchanged": True})
    except Exception as error:
        atomic_write_json(output / "failure.json", {"status": "failed", "error": repr(error), "partial_models": list(results)})
        raise
    return results


def _suite_specs(runroot):
    """Completed fresh-arm selections only; unfinished arms are not silently included."""
    runroot = Path(runroot).resolve()
    specs = []
    for complete in sorted(runroot.glob("*/complete.json")):
        directory = complete.parent
        specs.append({"label": directory.name, "checkpoint": directory / "selected.pt",
            "execution": directory / "execution_config.json", "reference": directory / "evaluation.json"})
    if not specs:
        raise ValueError("runroot has no completed fresh arms")
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("evaluate", help="read-only fixed validation simulation stress")
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seed", type=int, default=19)
    command.add_argument("--runroot", type=Path)
    args = parser.parse_args()
    evaluate(args.out, seed=args.seed, runroot=args.runroot)


if __name__ == "__main__":
    main()
