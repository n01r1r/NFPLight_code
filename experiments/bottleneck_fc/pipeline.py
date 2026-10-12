"""Local data, feature, loss, and renderer helpers for the probe."""
from collections import OrderedDict
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data.fabric import canonical_target, pack_svbrdf


def fixed_f0(maps, f0):
    return torch.cat((maps[:, :7], torch.full_like(maps[:, 7:10], 2 * f0 - 1)), dim=1)


def unit_prediction(raw):
    if not torch.isfinite(raw).all() or (raw[:, :3].norm(dim=1) < 1e-8).any():
        raise FloatingPointError("invalid predicted normal")
    return torch.cat((F.normalize(raw[:, :3], dim=1), raw[:, 3:]), dim=1)


class Bank:
    def __init__(self, manifest, dataset_config):
        import json

        rows = [line for line in Path(manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.records = [json.loads(line) for line in rows]
        self.map_keys = tuple(dataset_config["map_keys"])
        self.crop_count = dataset_config["crops_per_shard"]
        self.cache_limit = dataset_config["cache_shards"]
        self.diffuse_gamma = dataset_config["diffuse_decode_gamma"]
        self.specular_target = dataset_config["specular_target"]
        self.cache = OrderedDict()

    def crop(self, slot):
        path = str(Path(slot["shard_path"]).resolve())
        if path not in self.cache:
            with np.load(path, allow_pickle=False) as archive:
                maps = {key: archive[key] for key in self.map_keys}
            if any(value.dtype != np.uint8 for value in maps.values()):
                raise ValueError(f"bank must use uint8 maps: {path}")
            if any(value.shape[0] != self.crop_count for value in maps.values()):
                raise ValueError(f"bank must contain {self.crop_count} crops per map: {path}")
            self.cache[path] = maps
            if len(self.cache) > self.cache_limit:
                self.cache.popitem(last=False)
        maps = self.cache[path]
        self.cache.move_to_end(path)
        crop_index = slot["crop"]
        normal = maps["normal"][crop_index].astype(np.float32) / 127.5 - 1
        length = np.linalg.norm(normal, axis=-1, keepdims=True)
        if (length < 0.01).any():
            raise ValueError("invalid source normal")
        diffuse = (maps["diffuse"][crop_index].astype(np.float32) / 255) ** self.diffuse_gamma
        rough = maps["roughness"][crop_index].astype(np.float32) / 255
        if rough.ndim == 2:
            rough = rough[..., None]
        specular = np.full_like(diffuse, self.specular_target)
        return pack_svbrdf(normal / length, diffuse, rough, specular)


def prepare_features(model, target, f0):
    safe, valid = canonical_target(target.to(model.device))
    safe = fixed_f0(safe, f0)
    with torch.no_grad():
        observations = render_input_images_batched(model, safe) * valid
        features = model.build_features(observations, valid_mask=valid)
    return safe, valid, features[0] if isinstance(features, tuple) else features


def cache_slots(model, bank, slots, f0):
    rows = []
    for slot in slots:
        safe, valid, features = prepare_features(model, bank.crop(slot)[None], f0)
        rows.append({
            "safe": safe,
            "valid": valid,
            "features": features,
            "material_id": slot["material_id"],
            "category": slot["category"],
            "crop": slot["crop"],
        })
    return rows


def tensors(rows, indices):
    return [torch.cat([rows[index][key] for index in indices])
            for key in ("safe", "valid", "features")]


def masked_l1(pred, target, valid_mask):
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("masked L1 expects equal [B,C,H,W] tensors")
    if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
        raise FloatingPointError("masked L1 received non-finite tensors")
    if valid_mask.dtype != torch.bool or valid_mask.shape != (pred.shape[0], 1, pred.shape[2], pred.shape[3]):
        raise ValueError("valid_mask must be bool [B,1,H,W]")
    expanded = valid_mask.to(pred.device).expand_as(pred)
    support = expanded.sum()
    if support.item() == 0:
        raise ValueError("masked L1 has zero valid support")
    return (pred - target).abs().masked_select(expanded).sum() / support


def ndr_loss(pred, target, mask):
    parts = [
        masked_l1(pred[:, section], target[:, section], mask)
        for section in (slice(0, 3), slice(3, 6), slice(6, 7))
    ]
    return sum(parts) / 3, parts


def make_lights(lighting_config, device):
    rows = []
    for theta in lighting_config["theta_degrees"]:
        for phi in lighting_config["phi_degrees"]:
            t, p = math.radians(theta), math.radians(phi)
            radius = lighting_config["radius"]
            rows.append((
                radius * math.sin(t) * math.cos(p),
                radius * math.sin(t) * math.sin(p),
                radius * math.cos(t),
            ))
    return torch.tensor(rows, dtype=torch.float32, device=device)


def top_view_positions(count, view, *, device):
    return torch.tensor(view, dtype=torch.float32, device=device).repeat(count, 1)


def render_input_images_batched(model, svbrdf):
    """Render the fixed near/far observations in the original flattened batch order."""
    if svbrdf.ndim == 3:
        svbrdf = svbrdf.unsqueeze(0)
    if svbrdf.ndim != 4 or svbrdf.shape[1] != 10:
        raise ValueError("batched input rendering expects [B,10,H,W] maps")
    batch = svbrdf.shape[0]
    if batch < 1:
        raise ValueError("batched input rendering needs a nonempty batch")
    near_direction = model.near_light_dir.expand(batch, -1, -1, -1)
    far_direction = model.far_light_dir.expand(batch, -1, -1, -1)
    near_distance = model.near_light_dis.expand(batch, -1, -1, -1)
    far_distance = model.far_light_dis.expand(batch, -1, -1, -1)
    probes = torch.cat((near_direction, far_direction), dim=0).unsqueeze(1)
    distances = torch.cat((near_distance, far_distance), dim=0).unsqueeze(1)
    maps = torch.cat((svbrdf, svbrdf), dim=0)
    rendered = model.renderer._render(maps, probes, probes, distances).squeeze(1)
    rendered = rendered.reshape(2, batch, 3, rendered.shape[-2], rendered.shape[-1])
    rendered = rendered.permute(1, 0, 2, 3, 4).reshape(
        batch, 6, rendered.shape[-2], rendered.shape[-1]
    )
    if not torch.isfinite(rendered).all() or (rendered < 0).any():
        raise FloatingPointError("renderer observations must be finite nonnegative linear RGB")
    return rendered.clamp(0, 1)


def render_batch(model, svbrdf, lights, views):
    """Render B maps under L light/view pairs with one flattened renderer call."""
    if svbrdf.ndim != 4 or svbrdf.shape[1] != 10 or svbrdf.shape[0] < 1:
        raise ValueError("batched rendering expects nonempty [B,10,H,W] maps")
    if lights.ndim != 2 or views.ndim != 2 or lights.shape != views.shape or lights.shape[1] != 3:
        raise ValueError("batched rendering expects matching [L,3] lights and views")
    if lights.shape[0] < 1:
        raise ValueError("batched rendering needs at least one light")
    lights = lights.to(model.device)
    views = views.to(model.device)
    if (
        not torch.isfinite(lights).all()
        or not torch.isfinite(views).all()
        or (lights[:, 2] <= 0).any()
        or (views[:, 2] <= 0).any()
    ):
        raise ValueError("render lights and views must be finite and above the surface")
    light_dir, view_dir, distance, _ = model.renderer.torch_generate(
        views, lights, pos=model.surface
    )
    batch, count = svbrdf.shape[0], lights.shape[0]
    maps = svbrdf[:, None].expand(batch, count, -1, -1, -1).reshape(
        batch * count, *svbrdf.shape[1:]
    )
    flat_light = light_dir.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(
        batch * count, *light_dir.shape[1:]
    ).unsqueeze(1)
    flat_view = view_dir.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(
        batch * count, *view_dir.shape[1:]
    ).unsqueeze(1)
    flat_distance = distance.unsqueeze(0).expand(batch, -1, -1, -1, -1).reshape(
        batch * count, *distance.shape[1:]
    ).unsqueeze(1)
    rendered = model.renderer._render(maps, flat_light, flat_view, flat_distance).squeeze(1)
    return rendered.reshape(batch, count, 3, rendered.shape[-2], rendered.shape[-1])


def render_error(model, pred, target, valid, lights, rendering_config):
    views = top_view_positions(
        len(lights), rendering_config["training_lights"]["view"], device=model.device
    )
    with torch.no_grad():
        ground_truth = render_batch(model, target, lights, views)
    predicted = render_batch(
        model, fixed_f0(pred, rendering_config["f0"]), lights, views
    )
    mask = valid[:, None].expand_as(predicted)
    return (predicted - ground_truth).abs().masked_select(mask).mean()


def evaluation_views(model, lighting_config):
    return top_view_positions(
        len(lighting_config["theta_degrees"]) * len(lighting_config["phi_degrees"]),
        lighting_config["view"],
        device=model.device,
    )
