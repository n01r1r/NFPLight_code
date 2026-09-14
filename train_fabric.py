"""Train the 21-channel Fabric estimator from scratch.

The RAW Fabric protocol is deliberately small and reproducible: native cached
tiles are sampled without D4/photometric augmentation, observations remain
float32 linear RGB, and all target/render losses use the native-normal mask.
"""
import argparse
import copy
import json
import math
import platform
import random
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.fabric import (DEFAULT_MANIFEST, TARGET_POLICY, FabricDataset,
                         canonical_target, file_hash, write_json)
from model.normal_head import (NORMAL_HEADS, POLE_EPS, apply_normal_head)
from train_scratch import ScratchModel

MAP_NAMES = ("normal", "diffuse", "roughness", "specular")
MAP_SLICES = (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10))
FEATURE_VERSION = "raw_calibrated_v2"
CHECKPOINT_FORMAT = "nfplight.fabric.training.v2"
LOSS_PROBE_SEED = 10000
TEST_PROBE_SEED = 20000
RENDER_NEAR = 2.414
RENDER_FAR = 10.0
F0_BASELINE = 0.04
NORMAL_EPS = 1e-8
POLE_EPS = 1e-6
NORMAL_LOSSES = ("l1", "cosine", "phi_theta", "geodesic")
# The one pre-normal-objective Fabric baseline that may be resumed after the
# checkpoint-cadence migration. It is a single known digest, not a bypass for
# ordinary source-contract checks.
LEGACY_L1_TRAIN_SOURCE_SHA256 = "44dc5017fa8ee914a341284c5feed564a7d01c049d137234f31c715c26a2d6e6"


def light_positions(count, *, generator):
    """Generate the fixed historical point-light distribution."""
    if count < 1:
        raise ValueError("light count must be positive")
    xy = torch.rand(count, 2, generator=generator) * 2.4 - 1.2
    z = torch.rand(count, 1, generator=generator) * 6 + 2
    return torch.cat((xy, z), dim=-1)


def fixed_light_positions(count=4, *, seed=LOSS_PROBE_SEED, device=None):
    """Return deterministic rendering probes without consuming global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return light_positions(count, generator=generator).to(device or "cpu")


def cosine_lr(step, steps, start, end=1e-5):
    if steps < 1 or not 0 <= step < steps:
        raise ValueError("cosine step must lie in [0, steps)")
    progress = step / max(1, steps - 1)
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def _valid_mask(mask, shape):
    if mask is None:
        return torch.ones((shape[0], 1, shape[2], shape[3]), dtype=torch.bool, device="cpu")
    if mask.ndim != 4 or tuple(mask.shape) != (shape[0], 1, shape[2], shape[3]) or mask.dtype != torch.bool:
        raise ValueError(f"valid_mask must be bool [{shape[0]},1,{shape[2]},{shape[3]}]")
    if not mask.flatten(1).any(1).all():
        raise ValueError("valid_mask has a sample with zero valid support")
    return mask


def masked_l1(pred, target, valid_mask=None):
    """Mean absolute error over valid pixels only, retaining gradient."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("masked L1 expects equal [B,C,H,W] tensors")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError("masked L1 received non-finite tensors")
    mask = _valid_mask(valid_mask, pred.shape).to(pred.device)
    expanded = mask.expand_as(pred)
    support = expanded.sum()
    if support.item() == 0:
        raise ValueError("masked L1 has zero valid support")
    return (pred - target).abs().masked_select(expanded).sum() / support


def _unit_normal_vectors(pred, target, valid_mask=None):
    """Select valid signed XYZ [B,3,H,W] vectors and normalize to [N,3]."""
    if pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3:
        raise ValueError("normal loss expects equal [B,3,H,W] tensors")
    if pred.dtype not in (torch.float32, torch.float64) or target.dtype != pred.dtype or target.device != pred.device:
        raise ValueError("normal loss expects matching float32/float64 tensors on one device")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError("normal loss received non-finite tensors")
    mask = _valid_mask(valid_mask, pred.shape).to(pred.device)[:, 0]
    p, t = (value.movedim(1, -1)[mask] for value in (pred, target))
    if (p.norm(dim=-1) < NORMAL_EPS).any() or (t.norm(dim=-1) < NORMAL_EPS).any():
        raise FloatingPointError("normal loss received a zero or near-zero valid vector")
    return F.normalize(p, dim=-1, eps=NORMAL_EPS), F.normalize(t, dim=-1, eps=NORMAL_EPS)


def _spherical_angles(value):
    """Convert unit [N,3] normals to colatitude theta and azimuth phi safely."""
    xy = torch.linalg.vector_norm(value[:, :2], dim=-1)
    theta = torch.atan2(xy, value[:, 2])
    defined = xy >= POLE_EPS
    # atan2(0,0) has an undefined backward.  A pole has no meaningful azimuth,
    # so use a fixed safe phi there while retaining theta supervision.
    phi = torch.atan2(torch.where(defined, value[:, 1], 0.0),
                      torch.where(defined, value[:, 0], 1.0))
    return theta, phi, defined


def _phi_theta_error(p, t):
    """Direct coordinate L1 / pi on unit [N,3] normals; phi is omitted at poles."""
    ptheta, pphi, pdefined = _spherical_angles(p)
    ttheta, tphi, tdefined = _spherical_angles(t)
    theta_error = (ptheta - ttheta).abs()
    defined = pdefined & tdefined
    delta = pphi - tphi
    phi_error = torch.atan2(delta.sin(), delta.cos()).abs()
    phi_error = torch.where(defined, phi_error, torch.zeros_like(phi_error))
    return (theta_error + phi_error) / math.pi, defined


def _geodesic_error(p, t):
    """Normalized shortest arc after converting normals to spherical coordinates."""
    ptheta, pphi, _ = _spherical_angles(p)
    ttheta, tphi, _ = _spherical_angles(t)
    sp, cp = ptheta.sin(), ptheta.cos()
    st, ct = ttheta.sin(), ttheta.cos()
    dphi = pphi - tphi
    cosine = (cp * ct + sp * st * dphi.cos()).clamp(-1, 1)
    # Express the sine of the same spherical arc through its spherical
    # coordinate components.  vector_norm has a finite zero subgradient at
    # identical/antipodal endpoints, unlike direct acos backpropagation.
    cross_spherical = torch.stack((
        sp * pphi.sin() * ct - cp * st * tphi.sin(),
        cp * st * tphi.cos() - sp * pphi.cos() * ct,
        sp * st * (-dphi.sin()),
    ), dim=-1)
    sine = torch.linalg.vector_norm(cross_spherical, dim=-1)
    return torch.atan2(sine, cosine) / math.pi


def normal_loss(pred, target, valid_mask=None, *, kind="l1"):
    """Valid-pixel mean: signed XYZ L1, 1-cos(angle), geodesic angle/pi, or wrapped angle L1/pi.

    Direction losses ignore positive vector scale. Geodesic is the normalized
    shortest-arc distance in [0,1], evaluated from spherical theta/phi
    coordinates. Phi/theta L1 comparison is coordinate dependent and is not
    the spherical geodesic error.
    """
    if kind == "l1":
        return masked_l1(pred, target, valid_mask)
    if kind not in NORMAL_LOSSES:
        raise ValueError(f"unknown normal loss: {kind}")
    p, t = _unit_normal_vectors(pred, target, valid_mask)
    if kind == "cosine":
        return (1 - (p * t).sum(-1).clamp(-1, 1)).mean()
    if kind == "geodesic":
        return _geodesic_error(p, t).mean()
    return _phi_theta_error(p, t)[0].mean()


def map_loss(pred, target, valid_mask=None, w=(1.0, 1.0, 1.0, 1.0), *, normal_kind="l1"):
    """Sum four weighted map terms; default L1 also preserves evaluation metrics."""
    if len(w) != 4 or any(not math.isfinite(float(value)) or float(value) < 0 for value in w):
        raise ValueError("map weights must be finite and nonnegative")
    weight_sum = float(sum(w))
    if weight_sum <= 0:
        raise ValueError("map weights must have positive sum")
    parts = {"normal": normal_loss(pred[:, :3], target[:, :3], valid_mask, kind=normal_kind)}
    parts.update({name: masked_l1(pred[:, channels], target[:, channels], valid_mask)
                  for name, channels in zip(MAP_NAMES[1:], MAP_SLICES[1:])})
    # Keep the established summed four-component objective.  ``map_mean`` is
    # the equally weighted mean metric; dividing the optimized loss here would
    # silently make the frozen 0.5 render weight four times stronger.
    return sum(weight * parts[name] for weight, name in zip(w, MAP_NAMES)), parts


def render_pair(model, svbrdf, light, view):
    light = light.reshape(1, 3).to(model.device)
    view = view.reshape(1, 3).to(model.device)
    ld, vd, distance, _ = model.renderer.torch_generate(view, light, pos=model.surface)
    return model.renderer._render(svbrdf, ld[:, None], vd[:, None], distance[:, None]).squeeze(1)


def _prepare_target(model, target):
    """Canonicalize a dataset target and build a mask-aware RAW network input."""
    if target.ndim == 3:
        target = target.unsqueeze(0)
    target = target.to(model.device, non_blocking=True)
    safe_target, valid = canonical_target(target)
    rendered = model.render_input_images(safe_target, toLDR=False)
    rendered = rendered * valid.to(rendered.dtype)
    features = model.build_features(rendered, valid_mask=valid)
    return safe_target, valid, rendered, features[0] if isinstance(features, tuple) else features


def build_net_input(model, svbrdf, *, quantize=False, valid_mask=None):
    """Build RAW estimator input from normalized targets; quantization is forbidden."""
    if quantize:
        raise ValueError("RAW Fabric training forbids observation quantization")
    safe_target, valid, _, features = _prepare_target(model, svbrdf)
    if valid_mask is not None and not torch.equal(valid, valid_mask.to(valid.device)):
        raise ValueError("provided valid_mask differs from canonical target")
    return features


def render_loss(model, pred, target, n_lights, light_positions=None, valid_mask=None):
    """Equal valid-pixel L1 on unclipped linear-HDR renders."""
    if n_lights < 1:
        raise ValueError("n_lights must be positive")
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("render loss expects equal [B,10,H,W] maps")
    mask = _valid_mask(valid_mask, pred.shape).to(pred.device)
    if light_positions is None:
        lights = fixed_light_positions(n_lights, seed=LOSS_PROBE_SEED, device=pred.device)
    else:
        if tuple(light_positions.shape) != (n_lights, 3):
            raise ValueError(f"light_positions shape {tuple(light_positions.shape)} != ({n_lights}, 3)")
        lights = light_positions.to(pred.device)
    if not torch.isfinite(lights).all() or (lights[:, 2] <= 0).any():
        raise ValueError("render lights must be finite and above the surface")
    loss = pred.new_zeros(())
    expanded = mask.expand(-1, 3, -1, -1)
    support = expanded.sum()
    for light in lights:
        rendered_pred = render_pair(model, pred, light, light)
        rendered_target = render_pair(model, target, light, light)
        if not torch.isfinite(rendered_pred).all() or not torch.isfinite(rendered_target).all():
            raise FloatingPointError("renderer produced a non-finite HDR value")
        loss = loss + (rendered_pred - rendered_target).abs().masked_select(expanded).sum() / support
    return loss / n_lights


def _sample_batch(dataset, rng, batch_size, *, fixed=False):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if fixed:
        slots = [(0, 0, 0, False)] * batch_size
        return torch.stack([dataset.get_crop(index, crop) for index, crop, _, _ in slots]), slots
    if hasattr(dataset, "sample"):
        # FabricDataset's default sampler is the frozen no-augmentation path.
        return dataset.sample(rng, batch_size)
    else:
        crop_count = int(dataset.manifest["crop_count"])
        slots = [(rng.randrange(len(dataset)), rng.randrange(crop_count), 0, False)
                 for _ in range(batch_size)]
    return torch.stack([dataset.get_crop(index, crop) for index, crop, _, _ in slots]), slots


def optimizer_step(model, dataset, optimizer, rng, *, batch_size, accumulate, weight,
                   light_count=4, lights=None, fixed_crop=False, normal_kind="l1",
                   normal_head="xyz"):
    """Perform one effective batch and return loss/gradient diagnostics."""
    if accumulate < 1:
        raise ValueError("accumulate must be positive")
    model.net_g.train()
    optimizer.zero_grad(set_to_none=True)
    probes = fixed_light_positions(light_count, seed=LOSS_PROBE_SEED, device=model.device) if lights is None else lights
    total_map = total_render = total_normal = 0.0
    sampled = []
    for _ in range(accumulate):
        target, slots = _sample_batch(dataset, rng, batch_size, fixed=fixed_crop)
        sampled.extend(slots)
        safe, valid, _, features = _prepare_target(model, target)
        prediction = apply_normal_head(model.net_g(features), normal_head=normal_head)
        maps, parts = map_loss(prediction, safe, valid, normal_kind=normal_kind)
        render = render_loss(model, prediction, safe, light_count, probes, valid) if weight else maps.new_zeros(())
        loss = (maps + weight * render) / accumulate
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss; sampled slots={sampled}")
        loss.backward()
        total_map += float(maps.detach()) / accumulate
        total_render += float(render.detach()) / accumulate
        total_normal += float(parts["normal"].detach()) / accumulate
    preclip = nn.utils.clip_grad_norm_(model.net_g.parameters(), 1.0, error_if_nonfinite=True)
    postclip = torch.sqrt(sum((parameter.grad.detach().square().sum()
                               for parameter in model.net_g.parameters() if parameter.grad is not None),
                              torch.zeros((), device=model.device)))
    optimizer.step()
    return {"map_loss": total_map, "render_loss": total_render,
            "normal_loss_kind": normal_kind, "normal_head": normal_head,
            "normal_loss": total_normal,
            "loss": total_map + weight * total_render,
            "preclip_grad_norm": float(preclip), "postclip_grad_norm": float(postclip),
            "gradient_norm": float(postclip),
            "clip_max_norm": 1.0, "slots": sampled}


def _scalar_metrics(pred, target, valid):
    mask = valid.to(pred.device)
    pnormal = F.normalize(pred[:, :3], dim=1)
    cosine = (pnormal * target[:, :3]).sum(1, keepdim=True).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    values = {"normal_degrees": masked_l1(angle, torch.zeros_like(angle), mask)}
    p, t = _unit_normal_vectors(pred[:, :3], target[:, :3], mask)
    phi_theta, phi_defined = _phi_theta_error(p, t)
    geodesic = _geodesic_error(p, t)
    values.update(normal_cosine=(1 - (p * t).sum(-1).clamp(-1, 1)).mean(),
                  normal_phi_theta=phi_theta.mean(), normal_geodesic=geodesic.mean(),
                  normal_phi_defined_fraction=phi_defined.float().mean())
    pphys = (pred + 1) / 2
    tphys = (target + 1) / 2
    for name, channels in (("diffuse", slice(3, 6)), ("roughness", slice(6, 7)), ("specular", slice(7, 10))):
        error = pphys[:, channels] - tphys[:, channels]
        values[f"{name}_mae"] = masked_l1(error, torch.zeros_like(error), mask)
        values[f"{name}_rmse"] = torch.sqrt(masked_l1(error.square(), torch.zeros_like(error), mask))
    baseline_error = masked_l1(tphys[:, 7:10], torch.full_like(tphys[:, 7:10], F0_BASELINE), mask)
    values["f0_baseline_mae"] = baseline_error
    values["specular_f0_baseline_mae"] = baseline_error
    return values


@torch.no_grad()
def evaluate(model, dataset, *, crop_count=4, light_count=4, seed=LOSS_PROBE_SEED,
             independent_views=False, prediction_dir=None, prediction_metadata=None,
             normal_head="xyz"):
    """Evaluate fixed cached crops with valid-pixel and input diagnostics."""
    if not 1 <= crop_count <= dataset.manifest["crop_count"]:
        raise ValueError("evaluation crop_count exceeds the cache or is nonpositive")
    if independent_views:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lights = light_positions(light_count, generator=generator).to(model.device)
        views = light_positions(light_count, generator=generator).to(model.device)
    else:
        lights = fixed_light_positions(light_count, seed=seed, device=model.device)
        views = lights
    was_training = model.net_g.training
    model.net_g.eval()
    rows = []
    saved_predictions = []
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, record in enumerate(dataset.records):
            crop_rows = []
            for crop in range(crop_count):
                raw_target = dataset.get_crop(index, crop).unsqueeze(0)
                safe, valid, rendered, features = _prepare_target(model, raw_target)
                prediction = apply_normal_head(model.net_g(features), normal_head=normal_head)
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError(f"non-finite prediction: {record.get('id', index)}/{crop}")
                if prediction_dir is not None and hasattr(model, "save_svbrdf_maps"):
                    stem = str(record.get("name", record.get("id", index))).replace("\\", "_").replace("/", "_")
                    path = prediction_dir / f"{index:04d}_{stem}_crop{crop:02d}"
                    metadata = {"split": "evaluation", "crop": crop,
                                "valid_fraction": float(valid.float().mean())}
                    metadata.update(prediction_metadata or {})
                    model.save_svbrdf_maps(prediction, path, metadata=metadata)
                    saved_predictions.extend([str(path) + ".npz", str(path) + ".json"])
                normal_length = prediction[:, :3].norm(dim=1, keepdim=True)
                if (normal_length.masked_select(valid)).lt(1e-8).any():
                    raise FloatingPointError(f"zero predicted normal: {record.get('id', index)}/{crop}")
                maps, parts = map_loss(prediction, safe, valid)
                metrics = {f"map_{name}": float(value) for name, value in parts.items()}
                metrics["map_mean"] = float(sum(parts.values()) / 4)
                metrics.update({key: float(value) for key, value in _scalar_metrics(prediction, safe, valid).items()})
                predicted_normal = F.normalize(prediction[:, :3], dim=1)
                metrics["pred_negative_normal_z_fraction"] = float((predicted_normal[:, 2:3].lt(0) & valid).float().sum() / valid.sum())
                expanded = valid.to(rendered.device).expand(-1, 3, -1, -1)
                render_errors = []
                for light, view in zip(lights, views):
                    rp = render_pair(model, prediction, light, view)
                    rg = render_pair(model, safe, light, view)
                    error = rp - rg
                    render_errors.append((float(error.abs().masked_select(expanded).mean()),
                                         float(torch.sqrt(error.square().masked_select(expanded).mean()))))
                metrics["render_l1"], metrics["render_rmse"] = np.mean(render_errors, axis=0).tolist()
                valid_float = valid.float()
                metrics["valid_fraction"] = float(valid_float.mean())
                metrics["roughness_adjusted_fraction"] = float(((safe[:, 6:7] != raw_target.to(safe.device)[:, 6:7]) & valid).float().mean())
                input_valid = valid.expand(-1, 6, -1, -1).to(rendered.device)
                saturation = (rendered >= 1 - 1e-6) & input_valid
                dark = (rendered <= 1e-6) & input_valid
                metrics["input_saturation_fraction"] = float(saturation.float().sum() / input_valid.sum())
                metrics["input_dark_fraction"] = float(dark.float().sum() / input_valid.sum())
                near_valid, far_valid = input_valid[:, :3], input_valid[:, 3:]
                metrics["near_saturation_fraction"] = float(saturation[:, :3].float().sum() / near_valid.sum())
                metrics["far_saturation_fraction"] = float(saturation[:, 3:].float().sum() / far_valid.sum())
                metrics["near_dark_fraction"] = float(dark[:, :3].float().sum() / near_valid.sum())
                metrics["far_dark_fraction"] = float(dark[:, 3:].float().sum() / far_valid.sum())
                cy, cx = rendered.shape[-2] // 2, rendered.shape[-1] // 2
                metrics["near_center_mean"] = float(rendered[:, :3, cy, cx].mean())
                metrics["far_center_mean"] = float(rendered[:, 3:, cy, cx].mean())
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError(f"non-finite evaluation: {record.get('id', index)}/{crop}")
                crop_rows.append(metrics)
            average = {key: sum(row[key] for row in crop_rows) / crop_count for key in crop_rows[0]}
            rows.append({"id": record.get("id", str(index)), "name": record.get("name", str(index)),
                         "crops": crop_count, **average})
    finally:
        model.net_g.train(was_training)
    if not rows:
        raise ValueError("evaluation has no materials")
    numeric_keys = [key for key in rows[0] if key not in ("id", "name", "crops")]
    mean = {key: sum(row[key] for row in rows) / len(rows) for key in numeric_keys}
    tails = {key: {"p50": float(np.percentile([row[key] for row in rows], 50)),
                   "p95": float(np.percentile([row[key] for row in rows], 95)),
                   "max": float(max(row[key] for row in rows))} for key in numeric_keys}
    return {"mean": mean, "material_macro": mean, "materials": rows, "tails": tails,
            "light_positions": lights.cpu().tolist(), "view_positions": views.cpu().tolist(),
            "seed": seed, "independent_views": independent_views,
            "prediction_files": saved_predictions}


def render_candidate(metrics, baseline, best_render, tolerance):
    return (metrics["render_l1"] < best_render
            and all(metrics[f"map_{name}"] <= baseline[f"map_{name}"] + tolerance for name in MAP_NAMES))


def _package_version(module, fallback="unavailable"):
    return str(getattr(module, "__version__", fallback))


def _environment():
    try:
        import cv2
        cv_version = _package_version(cv2)
    except ImportError:
        cv_version = "unavailable"
    try:
        import PIL
        pil_version = _package_version(PIL)
    except ImportError:
        pil_version = "unavailable"
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "numpy": _package_version(np), "PIL": pil_version, "cv2": cv_version,
                   "cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available()),
                   "cudnn": torch.backends.cudnn.version(),
                   "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                   "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32)}
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
    return environment


def build_contract(args, manifest_path):
    source_root = Path(__file__).resolve().parent
    source_names = ("train_fabric.py", "train_scratch.py", "finetune_fabric.py", "data/fabric.py", "data/matsynth_crops.py",
                    "model/nfplight_model.py", "model/normal_head.py",
                    "network/nfplight_net.py", "utils/render_util.py")
    sources = {name: file_hash(source_root / name) for name in source_names if (source_root / name).is_file()}
    normal_head = getattr(args, "normal_head", "xyz")
    if normal_head not in NORMAL_HEADS:
        raise ValueError(f"unknown normal head: {normal_head}")
    settings = {key: value for key, value in vars(args).items()
                if key not in ("manifest", "out", "resume", "stop_after", "allow_legacy_l1_resume_migration")}
    return {"format": CHECKPOINT_FORMAT, "settings": settings,
            "manifest_sha256": file_hash(manifest_path), "source_sha256": sources,
            "normal_objective": {"kind": args.normal_loss, "weight": 1.0,
                                 "head": normal_head,
                                 "formula": {"l1": "mean(abs(pred_xyz-target_xyz))",
                                             "cosine": "mean(1-dot(unit_pred,unit_target))",
                                             "geodesic": "mean(spherical_shortest_arc(theta_phi(unit_pred),theta_phi(unit_target))/pi)",
                                             "phi_theta": "mean((abs(delta_theta)+abs(wrap_pi(delta_phi)))/pi)"}[args.normal_loss],
                                 "head_formula": {"xyz": "identity(raw_xyz)",
                                                   "phi_theta": "raw_xyz -> theta=pi*(raw_z+1)/4, phi=atan2(raw_y,raw_x) -> unit_normal_on_z_nonnegative_hemisphere"}[normal_head],
                                 "reduction": "mean_over_valid_pixels; l1_also_means_xyz",
                                 "normal_norm_epsilon": NORMAL_EPS, "pole_xy_epsilon": POLE_EPS,
                                 "phi_pole_policy": "omit_phi_if_either_unit_xy_norm_below_epsilon",
                                 "selection_metrics": "unchanged_normalized_map_l1"},
            "data_policy": copy.deepcopy(TARGET_POLICY),
            "input": {"feature_version": FEATURE_VERSION, "quantize": False, "gamma": 0,
                      "near": RENDER_NEAR, "far": RENDER_FAR, "lamp_intensity": 16,
                      "denoiser": False, "augmentation": False, "loss_probe_seed": LOSS_PROBE_SEED},
            "environment": _environment()}


def _is_legacy_l1_resume_migration(checkpoint_contract, current_contract):
    """Accept exactly the historical L1 run when only runtime policy evolved."""
    stored = copy.deepcopy(checkpoint_contract)
    current = copy.deepcopy(current_contract)
    stored_sources = stored.pop("source_sha256", None)
    current_sources = current.pop("source_sha256", None)
    if not isinstance(stored_sources, dict) or not isinstance(current_sources, dict):
        return False
    if stored_sources.pop("train_fabric.py", None) != LEGACY_L1_TRAIN_SOURCE_SHA256:
        return False
    current_sources.pop("train_fabric.py", None)
    if stored_sources != current_sources:
        return False
    stored_settings = stored.get("settings")
    current_settings = current.get("settings")
    if not isinstance(stored_settings, dict) or not isinstance(current_settings, dict):
        return False
    if "checkpoint_every" in stored_settings or "normal_loss" in stored_settings:
        return False
    if stored_settings.pop("normal_head", "xyz") != "xyz":
        return False
    if current_settings.pop("checkpoint_every", None) != 1000:
        return False
    if current_settings.pop("normal_head", "xyz") != "xyz":
        return False
    if current_settings.pop("normal_loss", None) != "l1":
        return False
    objective = current.pop("normal_objective", None)
    if (not isinstance(objective, dict) or objective.get("kind") != "l1"
            or objective.get("weight") != 1.0 or objective.get("head", "xyz") != "xyz"):
        return False
    return stored == current


def _atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, contract, state, *, optimizer=None, rng=None, metrics=None,
                    refs=None, generation=None):
    """Atomically save a v2 candidate or full resumable state."""
    if (optimizer is None) != (rng is None):
        raise ValueError("optimizer and rng must be supplied together")
    feature_version = getattr(model, "feature_version", FEATURE_VERSION)
    if contract.get("input", {}).get("feature_version") == FEATURE_VERSION and feature_version != FEATURE_VERSION:
        raise ValueError("RAW Fabric checkpoints must use raw_calibrated_v2")
    payload = {"format": CHECKPOINT_FORMAT, "feature_version": getattr(model, "feature_version", FEATURE_VERSION),
               "params": copy.deepcopy(model.net_g.state_dict()), "contract": copy.deepcopy(contract),
               "state": copy.deepcopy(state), "metrics": copy.deepcopy(metrics),
               "refs": copy.deepcopy(refs if refs is not None else state.get("refs", {})),
               "generation": generation}
    if optimizer is not None:
        payload.update(optimizer=optimizer.state_dict(), sample_rng=rng.getstate(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                       torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    _atomic_torch_save(path, payload)
    return Path(path)


def load_checkpoint(path, model, contract, *, optimizer=None, rng=None,
                    allow_legacy_l1_resume_migration=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a Fabric training v2 checkpoint")
    if contract.get("input", {}).get("feature_version") == FEATURE_VERSION and payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("RAW Fabric checkpoints must use raw_calibrated_v2")
    if payload.get("contract") != contract and not (
            allow_legacy_l1_resume_migration
            and _is_legacy_l1_resume_migration(payload.get("contract"), contract)):
        raise ValueError("checkpoint experiment/dataset/code contract differs from this run")
    model.net_g.load_state_dict(payload["params"], strict=True)
    model.feature_version = payload.get("feature_version", FEATURE_VERSION)
    if (optimizer is None) != (rng is None):
        raise ValueError("optimizer and rng must be supplied together")
    if optimizer is not None:
        if "optimizer" not in payload or "sample_rng" not in payload:
            raise ValueError("checkpoint has no resumable optimizer/RNG state")
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["sample_rng"])
        if "python_rng" in payload:
            random.setstate(payload["python_rng"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def _candidate_path(output, kind, total_step):
    return output / f"{kind}.step-{int(total_step):08d}.pth"


def _write_candidate(output, kind, model, contract, state, metrics):
    path = _candidate_path(output, kind, state["total_step"])
    if path.exists():
        raise FileExistsError(f"immutable selection candidate already exists: {path.name}")
    save_checkpoint(path, model, contract, state, metrics=metrics, refs=state.get("refs", {}))
    return {"path": path.name, "sha256": file_hash(path), "step": state["total_step"]}


def _verify_refs(output, state):
    for kind in ("best_map", "best_render", "stage_a"):
        ref = state.get("refs", {}).get(kind)
        if ref is None:
            continue
        path = (output / ref["path"]).resolve()
        if path.parent != output.resolve() or path.name != ref["path"]:
            raise ValueError(f"checkpoint reference escapes output directory: {ref['path']}")
        if not path.is_file() or file_hash(path) != ref["sha256"]:
            raise ValueError(f"checkpoint reference hash mismatch: {path.name}")


def _cleanup_checkpoint_candidates(output):
    keep = set()
    for name in ("last.pt", "last.prev.pt"):
        path = output / name
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise ValueError(f"cannot safely clean checkpoint generations; unreadable {name}: {error}") from error
        keep.add(payload.get("generation"))
        for kind, ref in payload.get("refs", {}).items():
            if not isinstance(ref, dict) or "path" not in ref or "sha256" not in ref:
                raise ValueError(f"cannot safely clean checkpoint generations; invalid {kind} reference in {name}")
            path = (output / ref["path"]).resolve()
            if path.parent != output.resolve() or path.name != ref["path"]:
                raise ValueError(f"checkpoint reference escapes output directory: {ref['path']}")
            if not path.is_file() or file_hash(path) != ref["sha256"]:
                raise ValueError(f"checkpoint reference hash mismatch: {path.name}")
            keep.add(ref["path"])
    patterns = ("best_map.step-*.pth", "best_render.step-*.pth", "stage_a.step-*.pth", "last.step-*.pt")
    for pattern in patterns:
        for path in output.glob(pattern):
            if path.name not in keep:
                path.unlink()


def _commit_last(output, model, contract, state, optimizer, rng):
    generation = f"last.step-{int(state['total_step']):08d}.{state['stage']}.pt"
    generation_path = output / generation
    if generation_path.exists():
        previous = output / "last.pt"
        if previous.is_file():
            payload = torch.load(previous, map_location="cpu", weights_only=False)
            if payload.get("generation") == generation and payload.get("state") == state:
                return previous
        raise FileExistsError(f"immutable last generation already exists: {generation}")
    save_checkpoint(generation_path, model, contract, state, optimizer=optimizer, rng=rng,
                    refs=state.get("refs", {}), generation=generation)
    previous = output / "last.pt"
    if previous.exists():
        _atomic_copy(previous, output / "last.prev.pt")
    _atomic_copy(generation_path, previous)
    _cleanup_checkpoint_candidates(output)
    return previous


def _materialize_aliases(output, refs):
    for kind, ref in refs.items():
        if kind not in ("best_map", "best_render", "stage_a"):
            continue
        source = output / ref["path"]
        if not source.is_file() or file_hash(source) != ref["sha256"]:
            raise ValueError(f"cannot materialize uncommitted selection: {kind}")
        _atomic_copy(source, output / f"{kind}.pth")


def _checkpoint_due(step, stage_steps, checkpoint_every):
    """Return whether an intermediate step may commit a resumable checkpoint."""
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    return step >= stage_steps or step % checkpoint_every == 0


def _rewrite_train_log(path, committed_step):
    if not path.exists():
        return
    valid = []
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines(keepends=True):
        if not line.endswith(("\n", "\r")):
            break
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            break
        if int(entry.get("total_step", -1)) > committed_step:
            break
        valid.append(line if line.endswith("\n") else line + "\n")
    path.write_text("".join(valid), encoding="utf-8")


def _write_validation(output, state, latest=None):
    value = {"history": state.get("history", [])}
    if latest is not None:
        value["latest"] = latest
    write_json(output / "validation.json", value)


def run(args):
    positive = ("map_steps", "render_steps", "batch", "accumulate", "lights", "val_every", "log_every",
                "eval_crops", "test_crops", "test_lights", "render_ramp", "checkpoint_every")
    if any(getattr(args, key) < 1 for key in positive) or args.stop_after < 0:
        raise ValueError("step counts, batch, accumulation, crop/light counts, and intervals must be positive")
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in (args.render_weight, args.map_tolerance)):
        raise ValueError("render weight and map tolerance must be finite and nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("the NFPLight renderer requires CUDA for a training run")
    output = Path(args.out).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output.iterdir()):
        raise FileExistsError("new runs require an empty output directory; use --resume for an existing run")
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError("resume must use a checkpoint in the original output directory")
    limit = 1 if args.overfit else (2 if args.smoke else None)
    train_data = FabricDataset(args.manifest, split="train", limit=limit)
    eval_data = train_data if args.overfit else FabricDataset(args.manifest, split="val", limit=(2 if args.smoke else None))
    test_data = None if args.overfit else FabricDataset(args.manifest, split="test", limit=(2 if args.smoke else None))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract = build_contract(args, args.manifest)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = ScratchModel(SimpleNamespace(save_root=str(output), feature_version=FEATURE_VERSION))
    rng = random.Random(args.seed)
    optimizer = torch.optim.Adam(model.net_g.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0)
    state = {"stage": "map", "step": 0, "total_step": 0, "best_map": None, "best_render": None,
             "baseline": None, "history": [], "refs": {}, "overfit": bool(args.overfit)}
    resumed_with_legacy_policy_migration = False
    if args.resume:
        payload = load_checkpoint(args.resume, model, contract, optimizer=optimizer, rng=rng,
                                  allow_legacy_l1_resume_migration=args.allow_legacy_l1_resume_migration)
        resumed_with_legacy_policy_migration = payload["contract"] != contract
        state = payload["state"]
        _verify_refs(output, state)
        # Remove only orphan candidates/generations after both retained commits
        # have been read and their referenced selection hashes verified.
        _cleanup_checkpoint_candidates(output)
        _rewrite_train_log(output / "train.jsonl", state["total_step"])
        _write_validation(output, state)
        if resumed_with_legacy_policy_migration:
            legacy_config = output / "config.pre_migration.json"
            if not legacy_config.exists():
                write_json(legacy_config, payload["contract"])
            write_json(output / "resume_migration.json", {
                "kind": "legacy_l1_checkpoint_policy_only",
                "checkpoint": str(Path(args.resume).resolve()),
                "resumed_total_step": state["total_step"],
                "legacy_train_fabric_sha256": payload["contract"]["source_sha256"]["train_fabric.py"],
                "current_train_fabric_sha256": contract["source_sha256"]["train_fabric.py"],
                "checkpoint_every": args.checkpoint_every,
                "normal_loss": args.normal_loss,
            })
            failed = output / "failure.json"
            recovered = output / "failure.recovered.json"
            if failed.is_file() and not recovered.exists():
                failed.replace(recovered)
    write_json(output / "config.json", contract)
    initial_path = output / "initial.json"
    if args.resume:
        if not initial_path.is_file():
            raise ValueError("resume requires the committed initial.json evaluation artifact")
        initial = json.loads(initial_path.read_text(encoding="utf-8"))["evaluation"]
    else:
        initial = evaluate(model, eval_data, crop_count=1 if args.overfit else args.eval_crops,
                           light_count=args.lights, seed=LOSS_PROBE_SEED,
                           normal_head=args.normal_head)
        write_json(initial_path, {"status": "before_training", "held_out": not args.overfit,
                                  "scope": "train_crop" if args.overfit else "validation",
                                  "evaluation": initial})
        # A crash before the first validation still has an exact step-0 state.
        _commit_last(output, model, contract, state, optimizer, rng)
    started = time.perf_counter()
    invocation_steps = 0
    try:
        while state["stage"] != "complete":
            stage = state["stage"]
            steps = args.map_steps if stage == "map" else args.render_steps
            if state["step"] >= steps:
                if stage == "map":
                    ref = state["refs"].get("best_map")
                    if ref is None:
                        raise ValueError("map stage ended without a best_map selection")
                    selected = load_checkpoint(output / ref["path"], model, contract,
                                               allow_legacy_l1_resume_migration=resumed_with_legacy_policy_migration)
                    state.update(stage="render", step=0, baseline=selected["metrics"],
                                 best_render=selected["metrics"]["render_l1"])
                    optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)
                    stage_ref = _write_candidate(output, "stage_a", model, contract, state, state["baseline"])
                    render_ref = _write_candidate(output, "best_render", model, contract, state, state["baseline"])
                    state["refs"].update(stage_a=stage_ref, best_render=render_ref)
                    _commit_last(output, model, contract, state, optimizer, rng)
                    _materialize_aliases(output, state["refs"])
                    continue
                state["stage"] = "complete"
                _commit_last(output, model, contract, state, optimizer, rng)
                _materialize_aliases(output, state["refs"])
                break
            lr = cosine_lr(state["step"], steps, 5e-4 if stage == "map" else 1e-4)
            for group in optimizer.param_groups:
                group["lr"] = lr
            weight = 0.0 if stage == "map" else args.render_weight * min(1.0, (state["step"] + 1) / args.render_ramp)
            values = optimizer_step(model, train_data, optimizer, rng, batch_size=args.batch,
                                    accumulate=args.accumulate, weight=weight, light_count=args.lights,
                                    fixed_crop=args.overfit, normal_kind=args.normal_loss,
                                    normal_head=args.normal_head)
            state["step"] += 1
            state["total_step"] += 1
            invocation_steps += 1
            log = {"stage": stage, "step": state["step"], "total_step": state["total_step"],
                   "lr": lr, "render_weight": weight, **values}
            with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(log, allow_nan=False) + "\n")
            validation_due = state["step"] % args.val_every == 0 or state["step"] == steps
            committed = False
            if validation_due:
                latest = evaluate(model, eval_data, crop_count=1 if args.overfit else args.eval_crops,
                                  light_count=args.lights, seed=LOSS_PROBE_SEED,
                                  normal_head=args.normal_head)
                metrics = latest["mean"]
                state["history"].append({"stage": stage, "step": state["step"], **metrics})
                if state["best_map"] is None or metrics["map_mean"] < state["best_map"]:
                    state["best_map"] = metrics["map_mean"]
                    state["refs"]["best_map"] = _write_candidate(output, "best_map", model, contract, state, metrics)
                if stage == "render" and render_candidate(metrics, state["baseline"], state["best_render"], args.map_tolerance):
                    state["best_render"] = metrics["render_l1"]
                    state["refs"]["best_render"] = _write_candidate(output, "best_render", model, contract, state, metrics)
                _commit_last(output, model, contract, state, optimizer, rng)
                _materialize_aliases(output, state["refs"])
                _write_validation(output, state, latest)
                print(f"val: map={metrics['map_mean']:.6f} render={metrics['render_l1']:.6f}", flush=True)
                committed = True
            elif _checkpoint_due(state["step"], steps, args.checkpoint_every):
                _commit_last(output, model, contract, state, optimizer, rng)
                committed = True
            if args.stop_after and invocation_steps >= args.stop_after:
                if not committed:
                    _commit_last(output, model, contract, state, optimizer, rng)
                _materialize_aliases(output, state["refs"])
                write_json(output / "status.json", {"status": "paused", "stage": stage,
                                                     "step": state["step"], "total_step": state["total_step"]})
                return
            if args.log_every and state["total_step"] % args.log_every == 0:
                print(f"{stage} {state['step']}/{steps}: loss={values['loss']:.6f} lr={lr:.3g}", flush=True)
        chosen_ref = state["refs"].get("best_render")
        if chosen_ref is None:
            raise ValueError("complete run has no committed best_render")
        _verify_refs(output, state)
        _materialize_aliases(output, state["refs"])
        chosen = output / "best_render.pth"
        selected = load_checkpoint(chosen, model, contract,
                                   allow_legacy_l1_resume_migration=resumed_with_legacy_policy_migration)
        if args.overfit:
            final = evaluate(model, eval_data, crop_count=1, light_count=args.test_lights,
                             seed=TEST_PROBE_SEED, independent_views=True,
                             prediction_dir=output / "predictions" / "train_crop",
                             prediction_metadata={"checkpoint_ref": chosen_ref},
                             normal_head=args.normal_head)
            test = {"held_out_evidence": False, "reason": "--overfit evaluates the same fixed train crop",
                    "train_crop": final, "initial": initial, "checkpoint": str(chosen),
                    "checkpoint_sha256": file_hash(chosen)}
        else:
            final = evaluate(model, test_data, crop_count=args.test_crops, light_count=args.test_lights,
                             seed=TEST_PROBE_SEED, independent_views=True,
                             prediction_dir=output / "predictions" / "test",
                             prediction_metadata={"checkpoint_ref": chosen_ref},
                             normal_head=args.normal_head)
            test = {**final, "held_out_evidence": True, "initial": initial,
                    "checkpoint": str(chosen), "checkpoint_sha256": file_hash(chosen),
                    "selection_metrics": selected["metrics"], "smoke_only": args.smoke}
        write_json(output / "test.json", test)
        write_json(output / "status.json", {"status": "complete", "total_step": state["total_step"],
                                             "smoke_only": args.smoke, "overfit": args.overfit,
                                             "checkpoint": str(chosen)})
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"error": repr(error), "stage": state["stage"], "step": state["step"]})
        raise
    finally:
        runtime = {"optimizer_steps_this_invocation": invocation_steps,
                   "elapsed_seconds": time.perf_counter() - started}
        if torch.cuda.is_available():
            runtime.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output / "runtime.json", runtime)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", required=True)
    parser.add_argument("--map-steps", type=int, default=100000)
    parser.add_argument("--render-steps", type=int, default=25000)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lights", type=int, default=4)
    parser.add_argument("--render-weight", type=float, default=0.5)
    parser.add_argument("--render-ramp", type=int, default=1000)
    parser.add_argument("--map-tolerance", type=float, default=0.02)
    parser.add_argument("--eval-crops", type=int, default=4)
    parser.add_argument("--test-crops", type=int, default=16)
    parser.add_argument("--test-lights", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="commit resumable last checkpoint every N intermediate steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal-head", choices=NORMAL_HEADS, default="xyz",
                        help="normal output parameterization; phi_theta is a fixed +z-hemisphere decoder")
    parser.add_argument("--normal-loss", choices=NORMAL_LOSSES, default="l1",
                        help="normal objective only; validation/selection keep existing map L1")
    parser.add_argument("--resume", default="")
    parser.add_argument("--allow-legacy-l1-resume-migration", action="store_true",
                        help="allow only the recorded pre-policy L1 baseline to resume under the 1000-step policy")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--overfit", action="store_true", help="one material/crop; not held-out evidence")
    parser.add_argument("--smoke", action="store_true", help="four map + four render steps; implementation check only")
    args = parser.parse_args(argv)
    if args.smoke:
        args.map_steps = args.render_steps = 4
        args.eval_crops = args.test_crops = args.render_ramp = 1
        args.val_every = 2
        args.test_lights = 2
    if args.overfit:
        args.eval_crops = args.test_crops = 1
    return args


if __name__ == "__main__":
    run(parse_args())
