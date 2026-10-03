"""Train the 21-channel Fabric estimator from scratch.

The RAW Fabric protocol is deliberately small and reproducible: native cached
tiles default to no D4/photometric augmentation, observations remain
float32 linear RGB, and all target/render losses use the native-normal mask.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.fabric import (
    DEFAULT_MANIFEST,
    TARGET_POLICY,
    FabricDataset,
    canonical_target,
    file_hash,
    write_json,
)
from model.nfplight_model import NFPLightModel
from model.normal_head import NORMAL_HEADS, POLE_EPS, apply_normal_head
from network.nfplight_net import TwoBranchNet


class ScratchModel(NFPLightModel):
    """NFPLightModel but with a FRESH (random-init) net -- no checkpoint load.
    Reuses all the renderer / input-assembly machinery from the base."""
    def init_network(self):
        self.net_g = self.model_to_device(TwoBranchNet())


MAP_NAMES = ("normal", "diffuse", "roughness", "specular")
MAP_SLICES = (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10))
FEATURE_VERSION = "raw_calibrated_v2"
CHECKPOINT_FORMAT = "nfplight.fabric.training.v2"
LOSS_PROBE_SEED = 10000
TEST_PROBE_SEED = 20000
RENDER_LOSS_LIGHT_DISTANCE = 4.0
RENDER_LOSS_POLAR_DEGREES = 45.0
RENDER_LOSS_TOP_VIEW_DISTANCE = 2.75
RENDER_NEAR = 2.414
RENDER_FAR = 10.0
F0_BASELINE = 0.04
NORMAL_EPS = 1e-8
NORMAL_LOSSES = ("l1", "cosine", "phi_theta", "geodesic")
# The one pre-normal-objective Fabric baseline that may be resumed after the
# checkpoint-cadence migration. It is a single known digest, not a bypass for
# ordinary source-contract checks.
LEGACY_L1_TRAIN_SOURCE_SHA256 = "44dc5017fa8ee914a341284c5feed564a7d01c049d137234f31c715c26a2d6e6"
# Explicit transitions from the completed Fabric runs accept only these
# audited historical trainer snapshots.  Every other source hash remains part
# of the compatibility contract.  The archived baseline source is the first
# digest; the second is the trainer that produced the committed stage-A
# checkpoint after the checkpoint-policy migration; the third is the trainer
# used by the paused phi-theta run.
KNOWN_TRAINER_TRANSITION_SOURCE_SHA256 = frozenset({
    LEGACY_L1_TRAIN_SOURCE_SHA256,
    "2c6d466ce0c90a4dd9f366dc27bf1e2eda2907761a1354f235e9ad0766feabb0",
    "bf93f8a85cad73b86e7b30bdcb6f0c9f021287e3553c0c2cee8c0b9bd77797fc",
    "e150e3518c433e8d68e65b0670a3766c6b1b9fd12a11c9adea432dacaf508661",
})
# These exact pairs were audited as either import/dead-code cleanup or the
# no-augmentation-preserving D4 extension.  They are deliberately enumerated
# instead of treating arbitrary source-hash differences as safe.
KNOWN_NONTRAINER_SOURCE_TRANSITIONS = {
    "model/nfplight_model.py": {
        "d5ea7cafe6449bbd0ab10dc2bb8074f139afe9ef6147449eed308408b828d0c5":
            "7d48dc575f024744d3dd821a4e85c7f272b9614ce1f50b89be8b11d1979a8b20",
        "86ed09d2347138be239d87adb58a390314cbbac2e79616c6188b62e85b3cc5b4":
            "7d48dc575f024744d3dd821a4e85c7f272b9614ce1f50b89be8b11d1979a8b20",
    },
    "data/fabric.py": {
        "452ba5d2e9d46b968a0b34de1db7b61b67dda3cc8a5e76cbab25074598923964":
            "8b0fd88777fe977dc9a1fcd5d9f8c2be16e51f0616683028995b7923984184f0",
    },
    "model/normal_head.py": {
        "891e624c53f0c08f0d63114c41c4e9cbf8ab86a5b8a24e416310ccfabfa9ef1f":
            "287c54ab124c9db8fc743ae2edd3becd90eed42b3fa3980311b837f4857b0173",
    },
    "network/nfplight_net.py": {
        "ef2f008ce505c386638348461e780a4ee78893e6de35131a2172585e75c7d599":
            "3577db41302b749e9d08afec55e341ef358c617dbc563ec12df12a7ddfe77f6b",
    },
    "utils/render_util.py": {
        "a45dee81255a0a0f296ebc3cd25ee9290d908a2e51fe0feebecfd2283cb513fa":
            "fd0236320b0548976102492f476d4baa292f1e33c989716a09f7617bd765f44c",
        "c04bcaae0152fd642c678200a9ec4e0e4759dd333ea16853f40599cbefefbb41":
            "fd0236320b0548976102492f476d4baa292f1e33c989716a09f7617bd765f44c",
    },
}
RENDER_RESTART_KIND = "l1_render_restart"
RENDER_SCHEDULE_MIGRATION_KIND = "render_schedule_migration"
OBSERVATION_CACHE_FORMAT = "nfplight.fabric.observation.v1"
PERFORMANCE_MIGRATION_KIND = "performance_only_resume_migration"
PERFORMANCE_MIGRATION_VERSION = "vectorized-render-cache-v1"
AMP_DTYPE = torch.float16
AMP_DTYPE_NAME = "float16"
AMP_INIT_SCALE = 1024.0
AMP_GROWTH_INTERVAL = 2000
D4_AUGMENTATION = "d4"
NO_AUGMENTATION = "none"
# The paused 2026-09-15 run was created by this exact pre-optimization
# trainer.  Performance migration is explicit and only accepts that audited
# source snapshot; ordinary resume remains strict.
KNOWN_PERFORMANCE_TRAINER_SOURCE_SHA256 = frozenset({
    "55f77955f105f8435dfbed19790d58e862a5c8dd4afdb5f5ad5a49b6384b1867",
})


def _amp_context(model, enabled):
    """Use CUDA FP16 only for the network forward; losses/rendering stay FP32."""
    if not enabled:
        return nullcontext()
    if model.device.type != "cuda":
        raise RuntimeError("FP16 AMP requires a CUDA model device")
    return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)


def _new_grad_scaler(enabled):
    """Create the current AMP scaler API with a compatibility fallback."""
    if not enabled:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 AMP requires CUDA")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(
            "cuda", enabled=True, init_scale=AMP_INIT_SCALE,
            growth_interval=AMP_GROWTH_INTERVAL,
        )
    return torch.cuda.amp.GradScaler(
        enabled=True, init_scale=AMP_INIT_SCALE,
        growth_interval=AMP_GROWTH_INTERVAL,
    )


def _predict_network(model, features, normal_head, scaler):
    """Run only the convolutional estimator under AMP, then restore FP32 maps."""
    amp_enabled = scaler is not None and scaler.is_enabled()
    with _amp_context(model, amp_enabled):
        raw_prediction = model.net_g(features)
    # Normal decoding and all scientific losses/rendering intentionally remain
    # float32.  This keeps renderer HDR arithmetic and angular diagnostics
    # stable while still accelerating the convolution-heavy network forward.
    return apply_normal_head(raw_prediction.float(), normal_head=normal_head)


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


def augmentation_rng(seed, augmentation):
    """Create a repeatable transform RNG without consuming training RNG state."""
    if augmentation == NO_AUGMENTATION:
        return None
    if augmentation != D4_AUGMENTATION:
        raise ValueError(f"unknown augmentation: {augmentation}")
    material = f"nfplight.fabric.augmentation.{augmentation}:{seed}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived_seed)


def sample_render_loss_lights(count, *, rng, device):
    """Sample point lights at fixed polar angle over random azimuth."""
    if count < 1:
        raise ValueError("light count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(rng.randrange(2**63))
    polar = math.radians(45.0)
    azimuth = torch.rand(count, generator=generator) * (2.0 * math.pi)
    directions = torch.stack((math.sin(polar) * torch.cos(azimuth),
                              math.sin(polar) * torch.sin(azimuth),
                              torch.full_like(azimuth, math.cos(polar))), dim=1)
    return (directions * 4.0).to(device)


def top_view_positions(count, *, device):
    return torch.tensor((0.0, 0.0, 2.75), dtype=torch.float32, device=device).repeat(count, 1)


def cosine_lr(step, steps, start, end=1e-5):
    if steps < 1 or not 0 <= step < steps:
        raise ValueError("cosine step must lie in [0, steps)")
    progress = step / max(1, steps - 1)
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def _valid_mask(mask, shape, *, validate=True):
    if mask is None:
        return torch.ones((shape[0], 1, shape[2], shape[3]), dtype=torch.bool, device="cpu")
    if validate and (mask.ndim != 4 or tuple(mask.shape) != (shape[0], 1, shape[2], shape[3]) or mask.dtype != torch.bool):
        raise ValueError(f"valid_mask must be bool [{shape[0]},1,{shape[2]},{shape[3]}]")
    if validate and not mask.flatten(1).any(1).all():
        raise ValueError("valid_mask has a sample with zero valid support")
    return mask


def masked_l1(pred, target, valid_mask=None, *, validate=True):
    """Mean absolute error over valid pixels only, retaining gradient."""
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("masked L1 expects equal [B,C,H,W] tensors")
    if validate and (not torch.isfinite(pred).all() or not torch.isfinite(target).all()):
        raise FloatingPointError("masked L1 received non-finite tensors")
    mask = _valid_mask(valid_mask, pred.shape, validate=validate).to(pred.device)
    expanded = mask.expand_as(pred)
    support = expanded.sum()
    if validate and support.item() == 0:
        raise ValueError("masked L1 has zero valid support")
    return (pred - target).abs().masked_select(expanded).sum() / support


def _unit_normal_vectors(pred, target, valid_mask=None, *, validate=True):
    """Select valid signed XYZ [B,3,H,W] vectors and normalize to [N,3]."""
    if validate and (pred.shape != target.shape or pred.ndim != 4 or pred.shape[1] != 3):
        raise ValueError("normal loss expects equal [B,3,H,W] tensors")
    if validate and (pred.dtype not in (torch.float32, torch.float64) or target.dtype != pred.dtype or target.device != pred.device):
        raise ValueError("normal loss expects matching float32/float64 tensors on one device")
    if validate and (not torch.isfinite(pred).all() or not torch.isfinite(target).all()):
        raise FloatingPointError("normal loss received non-finite tensors")
    mask = _valid_mask(valid_mask, pred.shape, validate=validate).to(pred.device)[:, 0]
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


def normal_loss(pred, target, valid_mask=None, *, kind="l1", validate=True):
    """Valid-pixel mean: signed XYZ L1, 1-cos(angle), geodesic angle/pi, or wrapped angle L1/pi.

    Direction losses ignore positive vector scale. Geodesic is the normalized
    shortest-arc distance in [0,1], evaluated from spherical theta/phi
    coordinates. Phi/theta L1 comparison is coordinate dependent and is not
    the spherical geodesic error.
    """
    if kind == "l1":
        return masked_l1(pred, target, valid_mask, validate=validate)
    if kind not in NORMAL_LOSSES:
        raise ValueError(f"unknown normal loss: {kind}")
    p, t = _unit_normal_vectors(pred, target, valid_mask, validate=validate)
    if kind == "cosine":
        return (1 - (p * t).sum(-1).clamp(-1, 1)).mean()
    if kind == "geodesic":
        return _geodesic_error(p, t).mean()
    return _phi_theta_error(p, t)[0].mean()


def map_loss(pred, target, valid_mask=None, w=(1.0, 1.0, 1.0, 1.0), *, normal_kind="l1", validate=True):
    """Sum four weighted map terms; default L1 also preserves evaluation metrics."""
    if len(w) != 4 or any(not math.isfinite(float(value)) or float(value) < 0 for value in w):
        raise ValueError("map weights must be finite and nonnegative")
    weight_sum = float(sum(w))
    if weight_sum <= 0:
        raise ValueError("map weights must have positive sum")
    parts = {"normal": normal_loss(pred[:, :3], target[:, :3], valid_mask,
                                     kind=normal_kind, validate=validate)}
    parts.update({name: masked_l1(pred[:, channels], target[:, channels], valid_mask,
                                  validate=validate)
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


def render_input_images_batched(model, svbrdf):
    """Render the fixed near/far RAW observations in one renderer launch.

    The network still receives exactly six float32 linear channels.  The
    original implementation launched the same renderer twice (near, then
    far); flattening the two fixed probes into a batch preserves the formula
    while removing one Python/launch boundary.
    """
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
    rendered = rendered.permute(1, 0, 2, 3, 4).reshape(batch, 6, rendered.shape[-2], rendered.shape[-1])
    if not torch.isfinite(rendered).all() or (rendered < 0).any():
        raise FloatingPointError("renderer observations must be finite nonnegative linear RGB")
    return rendered.clamp(0, 1)


def render_batch(model, svbrdf, lights, views):
    """Render B maps under L light/view pairs with one flattened call."""
    if svbrdf.ndim != 4 or svbrdf.shape[1] != 10 or svbrdf.shape[0] < 1:
        raise ValueError("batched rendering expects nonempty [B,10,H,W] maps")
    if lights.ndim != 2 or views.ndim != 2 or lights.shape != views.shape or lights.shape[1] != 3:
        raise ValueError("batched rendering expects matching [L,3] lights and views")
    if lights.shape[0] < 1:
        raise ValueError("batched rendering needs at least one light")
    lights = lights.to(model.device)
    views = views.to(model.device)
    if not torch.isfinite(lights).all() or not torch.isfinite(views).all() or (lights[:, 2] <= 0).any() or (views[:, 2] <= 0).any():
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


class FabricObservationCache:
    """Lossless on-disk cache for deterministic six-channel RAW observations.

    Fabric manifests contain SVBRDF maps, not paired near/far captures.  This
    cache therefore stores the exact float32 result of the fixed forward
    renderer, never a quantized image or a replacement target.  One file is
    kept per material so lazy population can amortize the renderer while
    retaining a bounded working set.
    """

    def __init__(self, root, dataset, split):
        self.root = Path(root).resolve() / split
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.split = split
        self.metadata_path = self.root / "metadata.json"
        self._keys = [
            f"{index:04d}-{hashlib.sha256(str(record.get('id', index)).encode('utf-8')).hexdigest()[:16]}"
            for index, record in enumerate(dataset.records)
        ]
        metadata = {
            "format": OBSERVATION_CACHE_FORMAT,
            "manifest_sha256": file_hash(dataset.path),
            "split": split,
            "crop_count": int(dataset.manifest["crop_count"]),
            "shape": [int(dataset.manifest["crop_count"]), 6, 256, 256],
            "dtype": "float32",
            "feature_version": FEATURE_VERSION,
            "renderer_version": PERFORMANCE_MIGRATION_VERSION,
            "near": RENDER_NEAR,
            "far": RENDER_FAR,
            "lamp_intensity": 16,
            "record_keys": self._keys,
        }
        if self.metadata_path.is_file():
            try:
                existing = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"observation cache metadata is unreadable: {self.metadata_path}") from error
            if existing != metadata:
                raise ValueError(f"observation cache contract mismatch: {self.metadata_path}")
        else:
            write_json(self.metadata_path, metadata)

    def _path(self, index):
        if not 0 <= index < len(self.dataset.records):
            raise IndexError(f"observation cache material index out of range: {index}")
        return self.root / f"{self._keys[index]}.npy"

    def _read_material(self, index):
        path = self._path(index)
        if not path.is_file():
            return None
        try:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"observation cache file is unreadable: {path}") from error
        expected = (self.dataset.manifest["crop_count"], 6, 256, 256)
        if values.shape != expected or values.dtype != np.float32:
            self._close_memmap(values)
            raise ValueError(f"observation cache shape/dtype mismatch: {path}")
        return values

    @staticmethod
    def _close_memmap(values):
        mmap = getattr(values, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def load_slots(self, slots):
        """Return CPU float32 observations for slots, or None when uncached."""
        materials = {}
        try:
            for index, _, rotation, hflip in slots:
                if rotation not in range(4) or not isinstance(hflip, bool):
                    raise ValueError("observation cache slots require rotation 0..3 and a boolean hflip")
                if index not in materials:
                    materials[index] = self._read_material(index)
                if materials[index] is None:
                    return None
            values = []
            for index, crop, rotation, hflip in slots:
                value = torch.from_numpy(np.array(materials[index][crop], dtype=np.float32, copy=True))
                if rotation:
                    value = torch.rot90(value, rotation, dims=(-2, -1))
                if hflip:
                    value = torch.flip(value, dims=(-1,))
                values.append(value)
            return torch.stack(values)
        finally:
            for material in materials.values():
                if material is not None:
                    self._close_memmap(material)

    def _write_material(self, index, values):
        values = np.asarray(values, dtype=np.float32)
        expected = (self.dataset.manifest["crop_count"], 6, 256, 256)
        if values.shape != expected or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
            raise ValueError(f"invalid observation cache payload for material {index}: {values.shape}")
        path = self._path(index)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    @torch.inference_mode()
    def ensure_material(self, model, index):
        """Render and atomically persist all fixed crops of one material."""
        if self._path(index).is_file():
            material = self._read_material(index)
            self._close_memmap(material)
            return
        targets = torch.stack([
            self.dataset.get_crop(index, crop)
            for crop in range(self.dataset.manifest["crop_count"])
        ]).to(model.device, non_blocking=True)
        safe, _ = canonical_target(targets)
        rendered = render_input_images_batched(model, safe).float().cpu().numpy()
        self._write_material(index, rendered)

    def ensure_materials(self, model, indices):
        for index in sorted(set(int(value) for value in indices)):
            self.ensure_material(model, index)

    def build(self, model):
        for index in range(len(self.dataset.records)):
            self.ensure_material(model, index)


def _prepare_target(model, target, *, rendered_input=None):
    """Canonicalize GT and build the RAW input synthesized from that GT.

    The cache stores only this deterministic forward-render result.  GT maps
    remain supervision; they are not concatenated into the estimator input.
    If real paired near/far captures are introduced later, this is the single
    boundary where those captures can replace the synthesized observations.
    """
    if target.ndim == 3:
        target = target.unsqueeze(0)
    target = target.to(model.device, non_blocking=True)
    safe_target, valid = canonical_target(target)
    rendered = (render_input_images_batched(model, safe_target)
                if rendered_input is None else rendered_input.to(model.device, non_blocking=True))
    if rendered.shape != (len(safe_target), 6, 256, 256):
        raise ValueError(f"rendered input must be [B,6,256,256], got {tuple(rendered.shape)}")
    if not rendered.is_floating_point() or not torch.isfinite(rendered).all() or rendered.amin() < 0 or rendered.amax() > 1:
        raise ValueError("rendered input must be finite linear RGB in [0,1]")
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


def _render_loss_groups(model, pred, target, n_lights, lights, valid_mask, group_count):
    """Return one valid-pixel render loss per accumulated microbatch group."""
    if n_lights < 1:
        raise ValueError("n_lights must be positive")
    if pred.shape != target.shape or pred.ndim != 4:
        raise ValueError("render loss expects equal [B,10,H,W] maps")
    mask = _valid_mask(valid_mask, pred.shape).to(pred.device)
    if tuple(lights.shape) != (n_lights, 3):
        raise ValueError(f"light_positions shape {tuple(lights.shape)} != ({n_lights}, 3)")
    lights = lights.to(pred.device)
    if not torch.isfinite(lights).all() or (lights[:, 2] <= 0).any():
        raise ValueError("render lights must be finite and above the surface")
    views = top_view_positions(n_lights, device=pred.device)

    if pred.shape[0] % group_count:
        raise ValueError("render loss batch must divide evenly into groups")
    rendered = render_batch(model, torch.cat((pred, target), dim=0), lights, views)
    rendered_pred, rendered_target = rendered[:len(pred)], rendered[len(pred):]
    if not torch.isfinite(rendered_pred).all() or not torch.isfinite(rendered_target).all():
        raise FloatingPointError("renderer produced a non-finite HDR value")
    expanded = mask[:, None].expand(-1, n_lights, 3, -1, -1)
    error = (rendered_pred - rendered_target).abs().masked_fill(~expanded, 0)
    numerator = error.sum(dim=(1, 2, 3, 4))
    support = mask.expand(-1, 3, -1, -1).sum(dim=(1, 2, 3))
    group_size = pred.shape[0] // group_count
    numerator = numerator.reshape(group_count, group_size).sum(dim=1)
    support = support.reshape(group_count, group_size).sum(dim=1)
    if (support <= 0).any():
        raise ValueError("render loss has zero valid support")
    return numerator / support / n_lights


def render_loss(model, pred, target, n_lights, light_positions=None, valid_mask=None):
    """Equal valid-pixel L1 on fixed-top-view unclipped linear-HDR renders."""
    if light_positions is None:
        lights = fixed_light_positions(n_lights, seed=LOSS_PROBE_SEED, device=pred.device)
    else:
        lights = light_positions
    return _render_loss_groups(model, pred, target, n_lights, lights, valid_mask, 1)[0]


def _d4_slots(slots, augmentation_rng):
    if augmentation_rng is None:
        raise ValueError("D4 augmentation requires a dedicated augmentation_rng")
    transformed = []
    for index, crop, _, _ in slots:
        transform = augmentation_rng.randrange(8)
        transformed.append((index, crop, transform % 4, transform >= 4))
    return transformed


def _sample_batch(dataset, rng, batch_size, *, fixed=False,
                  augmentation=NO_AUGMENTATION, augmentation_rng=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if augmentation not in (NO_AUGMENTATION, D4_AUGMENTATION):
        raise ValueError(f"unknown augmentation: {augmentation}")
    if fixed:
        slots = [(0, 0, 0, False)] * batch_size
        if augmentation == D4_AUGMENTATION:
            slots = _d4_slots(slots, augmentation_rng)
        return torch.stack([dataset.get_crop(index, crop, rotation=rotation, hflip=hflip)
                            for index, crop, rotation, hflip in slots]), slots
    if hasattr(dataset, "sample"):
        # Keep the historical call path (and RNG consumption) exactly for none.
        if augmentation == NO_AUGMENTATION:
            return dataset.sample(rng, batch_size)
        return dataset.sample(rng, batch_size, augmentation=augmentation,
                              augmentation_rng=augmentation_rng)
    else:
        crop_count = int(dataset.manifest["crop_count"])
        slots = [(rng.randrange(len(dataset)), rng.randrange(crop_count), 0, False)
                 for _ in range(batch_size)]
    if augmentation == D4_AUGMENTATION:
        slots = _d4_slots(slots, augmentation_rng)
    return torch.stack([dataset.get_crop(index, crop, rotation=rotation, hflip=hflip)
                        for index, crop, rotation, hflip in slots]), slots


def optimizer_step(model, dataset, optimizer, rng, *, batch_size, accumulate, weight,
                   light_count=4, lights=None, fixed_crop=False, normal_kind="l1",
                   normal_head="xyz", observation_cache=None, fuse_accumulation=False,
                   scaler=None, augmentation=NO_AUGMENTATION, augmentation_rng=None):
    """Perform one effective batch and return loss/gradient diagnostics."""
    if accumulate < 1:
        raise ValueError("accumulate must be positive")
    model.net_g.train()
    optimizer.zero_grad(set_to_none=True)
    probes = sample_render_loss_lights(light_count, rng=rng, device=model.device) if lights is None else lights
    total_map = total_render = total_normal = None
    sampled = []
    if fuse_accumulation:
        targets = []
        batches = []
        for _ in range(accumulate):
            target, slots = _sample_batch(dataset, rng, batch_size, fixed=fixed_crop,
                                          augmentation=augmentation,
                                          augmentation_rng=augmentation_rng)
            targets.append(target)
            batches.append(slots)
            sampled.extend(slots)
        target = torch.cat(targets, dim=0)
        slots = [slot for batch in batches for slot in batch]
        rendered_input = None
        if observation_cache is not None:
            observation_cache.ensure_materials(model, [slot[0] for slot in slots])
            rendered_input = observation_cache.load_slots(slots)
        safe, valid, _, features = _prepare_target(model, target, rendered_input=rendered_input)
        prediction = _predict_network(model, features, normal_head, scaler)
        map_values = []
        normal_values = []
        for group in range(accumulate):
            begin, end = group * batch_size, (group + 1) * batch_size
            maps, parts = map_loss(prediction[begin:end], safe[begin:end], valid[begin:end],
                                   normal_kind=normal_kind, validate=False)
            map_values.append(maps)
            normal_values.append(parts["normal"])
        total_map = torch.stack(map_values).mean()
        total_normal = torch.stack(normal_values).mean()
        if weight:
            render_values = _render_loss_groups(
                model, prediction, safe, light_count, probes, valid, accumulate
            )
            total_render = render_values.mean()
        else:
            total_render = total_map.new_zeros(())
        loss = total_map + weight * total_render
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss; sampled slots={sampled}")
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
    else:
        for _ in range(accumulate):
            target, slots = _sample_batch(dataset, rng, batch_size, fixed=fixed_crop,
                                          augmentation=augmentation,
                                          augmentation_rng=augmentation_rng)
            sampled.extend(slots)
            rendered_input = None
            if observation_cache is not None:
                observation_cache.ensure_materials(model, [slot[0] for slot in slots])
                rendered_input = observation_cache.load_slots(slots)
            safe, valid, _, features = _prepare_target(model, target, rendered_input=rendered_input)
            prediction = _predict_network(model, features, normal_head, scaler)
            maps, parts = map_loss(prediction, safe, valid, normal_kind=normal_kind, validate=False)
            render = render_loss(model, prediction, safe, light_count, probes, valid) if weight else maps.new_zeros(())
            loss = (maps + weight * render) / accumulate
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss; sampled slots={sampled}")
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            total_map = maps.detach() / accumulate if total_map is None else total_map + maps.detach() / accumulate
            total_render = render.detach() / accumulate if total_render is None else total_render + render.detach() / accumulate
            normal = parts["normal"].detach()
            total_normal = normal / accumulate if total_normal is None else total_normal + normal / accumulate
    if scaler is not None:
        scaler.unscale_(optimizer)
    preclip = nn.utils.clip_grad_norm_(
        model.net_g.parameters(), 1.0, error_if_nonfinite=scaler is None
    )
    if scaler is not None and not bool(torch.isfinite(preclip).item()):
        # GradScaler intentionally skips an overflowing update and backs off
        # its scale.  Account for the skipped optimizer step explicitly in the
        # JSONL record instead of hiding it or aborting the whole run.
        scaler.step(optimizer)
        scaler.update()
        amp_scale = float(scaler.get_scale())
        diagnostics = torch.stack((total_map.detach(), total_render.detach(), total_normal.detach())).cpu().tolist()
        total_map_value, total_render_value, total_normal_value = diagnostics
        return {"map_loss": total_map_value, "render_loss": total_render_value,
                "normal_loss_kind": normal_kind, "normal_head": normal_head,
                "normal_loss": total_normal_value,
                "loss": total_map_value + weight * total_render_value,
                "preclip_grad_norm": None, "postclip_grad_norm": None,
                "gradient_norm": None, "clip_max_norm": 1.0,
                "amp_scale": amp_scale, "amp_overflow": True,
                "optimizer_step_applied": False, "slots": sampled}
    postclip = torch.sqrt(sum((parameter.grad.detach().square().sum()
                               for parameter in model.net_g.parameters() if parameter.grad is not None),
                              torch.zeros((), device=model.device)))
    if scaler is None:
        optimizer.step()
        amp_scale = 1.0
    else:
        scaler.step(optimizer)
        scaler.update()
        amp_scale = float(scaler.get_scale())
    diagnostics = torch.stack((total_map, total_render, total_normal, preclip.detach(), postclip.detach())).cpu().tolist()
    total_map, total_render, total_normal, preclip_value, postclip_value = diagnostics
    return {"map_loss": total_map, "render_loss": total_render,
            "normal_loss_kind": normal_kind, "normal_head": normal_head,
            "normal_loss": total_normal,
            "loss": total_map + weight * total_render,
            "preclip_grad_norm": preclip_value, "postclip_grad_norm": postclip_value,
            "gradient_norm": postclip_value,
            "clip_max_norm": 1.0, "amp_scale": amp_scale,
            "amp_overflow": False, "optimizer_step_applied": True,
            "slots": sampled}


def _scalar_metrics(pred, target, valid, *, validate=True):
    mask = valid.to(pred.device)
    pnormal = F.normalize(pred[:, :3], dim=1)
    cosine = (pnormal * target[:, :3]).sum(1, keepdim=True).clamp(-1, 1)
    angle = torch.rad2deg(torch.acos(cosine))
    values = {"normal_degrees": masked_l1(angle, torch.zeros_like(angle), mask, validate=validate)}
    p, t = _unit_normal_vectors(pred[:, :3], target[:, :3], mask, validate=validate)
    phi_theta, phi_defined = _phi_theta_error(p, t)
    geodesic = _geodesic_error(p, t)
    values.update(normal_cosine=(1 - (p * t).sum(-1).clamp(-1, 1)).mean(),
                  normal_phi_theta=phi_theta.mean(), normal_geodesic=geodesic.mean(),
                  normal_phi_defined_fraction=phi_defined.float().mean())
    pphys = (pred + 1) / 2
    tphys = (target + 1) / 2
    for name, channels in (("diffuse", slice(3, 6)), ("roughness", slice(6, 7)), ("specular", slice(7, 10))):
        error = pphys[:, channels] - tphys[:, channels]
        values[f"{name}_mae"] = masked_l1(error, torch.zeros_like(error), mask, validate=validate)
        values[f"{name}_rmse"] = torch.sqrt(masked_l1(error.square(), torch.zeros_like(error), mask, validate=validate))
    baseline_error = masked_l1(tphys[:, 7:10], torch.full_like(tphys[:, 7:10], F0_BASELINE), mask, validate=validate)
    values["f0_baseline_mae"] = baseline_error
    values["specular_f0_baseline_mae"] = baseline_error
    return values


def _evaluation_crop_metrics(prediction, safe, valid, rendered, raw_target,
                             rendered_pred, rendered_target, light_count):
    """Compute the established metrics for one crop from batched results."""
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("non-finite prediction")
    normal_length = prediction[:, :3].norm(dim=1, keepdim=True)
    if (normal_length.masked_select(valid)).lt(1e-8).any():
        raise FloatingPointError("zero predicted normal")
    maps, parts = map_loss(prediction, safe, valid, validate=False)
    metrics = {f"map_{name}": float(value) for name, value in parts.items()}
    metrics["map_mean"] = float(sum(parts.values()) / 4)
    metrics.update({key: float(value) for key, value in _scalar_metrics(
        prediction, safe, valid, validate=False
    ).items()})
    predicted_normal = F.normalize(prediction[:, :3], dim=1)
    metrics["pred_negative_normal_z_fraction"] = float(
        (predicted_normal[:, 2:3].lt(0) & valid).float().sum() / valid.sum()
    )
    expanded = valid.to(rendered.device).expand(-1, 3, -1, -1)
    error = rendered_pred - rendered_target
    expanded_render = expanded[:, None].expand(-1, light_count, -1, -1, -1)
    absolute = error.abs().masked_select(expanded_render)
    squared = error.square().masked_select(expanded_render)
    metrics["render_l1"] = float(absolute.mean())
    metrics["render_rmse"] = float(torch.sqrt(squared.mean()))
    valid_float = valid.float()
    metrics["valid_fraction"] = float(valid_float.mean())
    metrics["roughness_adjusted_fraction"] = float(
        ((safe[:, 6:7] != raw_target.to(safe.device)[:, 6:7]) & valid).float().mean()
    )
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
        raise FloatingPointError("non-finite evaluation")
    return metrics


@torch.inference_mode()
def evaluate(model, dataset, *, crop_count=4, light_count=4, seed=LOSS_PROBE_SEED,
             independent_views=False, prediction_dir=None, prediction_metadata=None,
             normal_head="xyz", observation_cache=None, eval_batch_size=8):
    """Evaluate fixed cached crops with valid-pixel and input diagnostics.

    Network inference and novel-light rendering are batched, while metrics
    are reduced crop-by-crop exactly as in the established evaluator.
    """
    if not 1 <= crop_count <= dataset.manifest["crop_count"]:
        raise ValueError("evaluation crop_count exceeds the cache or is nonpositive")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")
    if getattr(model, "feature_version", FEATURE_VERSION) == "legacy_batch_v0":
        # That legacy feature path intentionally normalizes over the whole
        # tensor; batching it would change its historical semantics.
        eval_batch_size = 1
    if independent_views:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        lights = light_positions(light_count, generator=generator).to(model.device)
        views = light_positions(light_count, generator=generator).to(model.device)
    else:
        lights = fixed_light_positions(light_count, seed=seed, device=model.device)
        views = top_view_positions(light_count, device=model.device)
    was_training = model.net_g.training
    model.net_g.eval()
    rows = []
    items = [(index, crop, record)
             for index, record in enumerate(dataset.records)
             for crop in range(crop_count)]
    crop_metrics = {index: [] for index, _, _ in items}
    saved_predictions = []
    if prediction_dir is not None:
        prediction_dir = Path(prediction_dir)
        prediction_dir.mkdir(parents=True, exist_ok=True)
    try:
        for start in range(0, len(items), eval_batch_size):
            chunk = items[start:start + eval_batch_size]
            slots = [(index, crop, 0, False) for index, crop, _ in chunk]
            raw_target = torch.stack([dataset.get_crop(index, crop) for index, crop, _ in chunk])
            rendered_input = None
            if observation_cache is not None:
                observation_cache.ensure_materials(model, [index for index, _, _ in chunk])
                rendered_input = observation_cache.load_slots(slots)
            safe, valid, rendered, features = _prepare_target(
                model, raw_target, rendered_input=rendered_input
            )
            prediction = apply_normal_head(model.net_g(features), normal_head=normal_head)
            render_pair_values = render_batch(
                model, torch.cat((prediction, safe), dim=0), lights, views
            )
            rendered_pred = render_pair_values[:len(chunk)]
            rendered_target = render_pair_values[len(chunk):]
            for local, (index, crop, record) in enumerate(chunk):
                metrics = _evaluation_crop_metrics(
                    prediction[local:local + 1], safe[local:local + 1],
                    valid[local:local + 1], rendered[local:local + 1],
                    raw_target[local:local + 1], rendered_pred[local:local + 1],
                    rendered_target[local:local + 1], light_count
                )
                if prediction_dir is not None and hasattr(model, "save_svbrdf_maps"):
                    stem = str(record.get("name", record.get("id", index))).replace("\\", "_").replace("/", "_")
                    path = prediction_dir / f"{index:04d}_{stem}_crop{crop:02d}"
                    metadata = {"split": "evaluation", "crop": crop,
                                "valid_fraction": metrics["valid_fraction"]}
                    metadata.update(prediction_metadata or {})
                    model.save_svbrdf_maps(prediction[local:local + 1], path, metadata=metadata)
                    saved_predictions.extend([str(path) + ".npz", str(path) + ".json"])
                crop_metrics[index].append(metrics)
        for index, record in enumerate(dataset.records):
            crop_rows = crop_metrics[index]
            if len(crop_rows) != crop_count:
                raise ValueError(f"evaluation missing crops for {record.get('id', index)}")
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
    if hasattr(torch, "get_float32_matmul_precision"):
        environment["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(0)
    return environment


def _configure_performance_profile(profile):
    """Configure measured CUDA throughput knobs without hiding them in code."""
    if profile == "strict":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        return
    if profile == "throughput":
        # This changes floating-point kernel selection and is therefore an
        # explicit, contract-recorded performance choice rather than the
        # default research baseline.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return
    raise ValueError(f"unknown performance profile: {profile}")


def build_contract(args, manifest_path):
    source_root = Path(__file__).resolve().parent
    source_names = ("train_fabric.py", "data/fabric.py", "model/nfplight_model.py",
                    "model/normal_head.py", "network/nfplight_net.py", "utils/render_util.py")
    sources = {name: file_hash(source_root / name) for name in source_names if (source_root / name).is_file()}
    normal_head = getattr(args, "normal_head", "xyz")
    if normal_head not in NORMAL_HEADS:
        raise ValueError(f"unknown normal head: {normal_head}")
    augmentation = getattr(args, "augmentation", NO_AUGMENTATION)
    if augmentation not in (NO_AUGMENTATION, D4_AUGMENTATION):
        raise ValueError(f"unknown augmentation: {augmentation}")
    settings = {key: value for key, value in vars(args).items()
                if key not in ("manifest", "out", "resume", "stop_after",
                               "render_restart_source", "allow_legacy_l1_resume_migration",
                               "allow_render_schedule_migration", "allow_performance_migration")}
    settings["augmentation"] = augmentation
    data_policy = copy.deepcopy(TARGET_POLICY)
    data_policy["augmentation"] = augmentation
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
             "data_policy": data_policy,
             "input": {"feature_version": FEATURE_VERSION, "quantize": False, "gamma": 0,
                       "near": RENDER_NEAR, "far": RENDER_FAR, "lamp_intensity": 16,
                       "denoiser": False, "augmentation": augmentation,
                       "loss_probe_seed": LOSS_PROBE_SEED},
            "environment": _environment()}


def _json_sha256(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_transition_contract(contract, *, allow_render_steps, require_known_trainer=True):
    """Normalize only documented historical omissions for an explicit transition."""
    if not isinstance(contract, dict):
        raise ValueError("transition source has no training contract")
    value = copy.deepcopy(contract)
    _normalize_contract_augmentation(value)
    settings = value.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("transition source contract has no settings")
    settings.setdefault("normal_head", "xyz")
    if allow_render_steps:
        settings.pop("render_steps", None)
        settings.pop("lights", None)
    objective = value.get("normal_objective")
    if not isinstance(objective, dict):
        raise ValueError("transition source contract has no normal objective")
    objective.setdefault("head", "xyz")
    if objective.get("head") == "xyz":
        objective.setdefault("head_formula", "identity(raw_xyz)")
    sources = value.get("source_sha256")
    if not isinstance(sources, dict):
        raise ValueError("transition source contract has no source hashes")
    trainer_sha = sources.get("train_fabric.py")
    if require_known_trainer and trainer_sha not in KNOWN_TRAINER_TRANSITION_SOURCE_SHA256:
        raise ValueError(f"transition source trainer hash is not an approved historical snapshot: {trainer_sha}")
    return value


def _normalize_contract_augmentation(contract):
    """Treat omitted/boolean legacy augmentation fields as the historical none mode."""
    settings = contract.get("settings")
    if isinstance(settings, dict):
        settings.setdefault("augmentation", NO_AUGMENTATION)
    for section in ("data_policy", "input"):
        value = contract.get(section)
        if isinstance(value, dict) and value.get("augmentation") is False:
            value["augmentation"] = NO_AUGMENTATION


def _assert_transition_contract(source_contract, current_contract, *, allow_render_steps):
    source = _canonical_transition_contract(source_contract, allow_render_steps=allow_render_steps)
    current = _canonical_transition_contract(current_contract, allow_render_steps=allow_render_steps,
                                             require_known_trainer=False)
    source_sources = source["source_sha256"]
    current_sources = current["source_sha256"]
    # The explicit transition is the only path that permits the trainer code
    # itself to evolve; its historical source digest was checked above.
    source_sources.pop("train_fabric.py", None)
    current_sources.pop("train_fabric.py", None)
    # The historical XYZ trainer predates the separate normal-head module.  A
    # missing module hash is accepted only for that documented XYZ case.
    if "model/normal_head.py" not in source_sources and current_sources.get("model/normal_head.py"):
        if source["settings"].get("normal_head") != "xyz":
            raise ValueError("historical source omits normal_head only for XYZ")
        source_sources["model/normal_head.py"] = current_sources["model/normal_head.py"]
    if set(source_sources) != set(current_sources):
        raise ValueError("transition source model/data source hashes differ")
    for name, source_sha in source_sources.items():
        current_sha = current_sources.get(name)
        if source_sha == current_sha:
            continue
        if KNOWN_NONTRAINER_SOURCE_TRANSITIONS.get(name, {}).get(source_sha) != current_sha:
            raise ValueError(f"transition source hash differs for {name}")
    source.pop("source_sha256", None)
    current.pop("source_sha256", None)
    if source != current:
        raise ValueError("transition source contract differs in a non-render setting")


def _validate_render_restart_source(source_path, current_contract):
    source_path = Path(source_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"render restart source does not exist: {source_path}")
    if not (source_path.name == "stage_a.pth"
            or (source_path.name.startswith("stage_a.step-") and source_path.name.endswith(".pth"))):
        raise ValueError("render restart source must be a stage_a checkpoint")
    payload = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("render restart source is not a Fabric training v2 checkpoint")
    if payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("render restart source does not use raw_calibrated_v2")
    state = payload.get("state")
    if not isinstance(state, dict) or (state.get("stage"), state.get("step")) != ("render", 0):
        raise ValueError("render restart source must be a step-0 render boundary")
    baseline = state.get("baseline")
    metrics = payload.get("metrics")
    if not isinstance(baseline, dict) or not isinstance(metrics, dict) or metrics != baseline:
        raise ValueError("render restart source must carry the selected stage_a metrics")
    for key in ("map_mean", "render_l1"):
        if key not in baseline or not math.isfinite(float(baseline[key])) or float(baseline[key]) < 0:
            raise ValueError("render restart source has invalid stage_a metrics")
    settings = payload["contract"].get("settings", {})
    if settings.get("normal_loss") != "l1" or settings.get("normal_head", "xyz") != "xyz":
        raise ValueError("render restart source must be the L1 XYZ baseline")
    if not isinstance(settings.get("map_steps"), int) or state.get("total_step") != settings["map_steps"]:
        raise ValueError("render restart source must follow a completed map stage")
    _assert_transition_contract(payload.get("contract"), current_contract, allow_render_steps=True)
    initial_path = source_path.parent / "initial.json"
    if not initial_path.is_file():
        raise ValueError("render restart source parent has no initial.json artifact")
    try:
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("render restart source initial.json is unreadable") from error
    if not isinstance(initial, dict) or not isinstance(initial.get("evaluation"), dict):
        raise ValueError("render restart source initial.json has no evaluation artifact")
    if "params" not in payload or not isinstance(payload["params"], dict):
        raise ValueError("render restart source has no model parameters")
    return payload, initial_path


def _validate_render_schedule_migration(payload, current_contract):
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("render schedule migration source is not a Fabric training v2 checkpoint")
    state = payload.get("state")
    if not isinstance(state, dict) or state.get("stage") != "map":
        raise ValueError("render schedule migration requires a map-stage checkpoint")
    settings = payload.get("contract", {}).get("settings", {})
    if settings.get("map_steps") != 100000 or settings.get("render_steps") != 25000:
        raise ValueError("render schedule migration requires the historical 100k + 25k schedule")
    current_settings = current_contract.get("settings", {})
    if current_settings.get("map_steps") != 100000 or current_settings.get("render_steps") != 100000:
        raise ValueError("render schedule migration requires the new 100k render schedule")
    if state.get("step", -1) < 0 or state.get("step", 100001) >= settings["map_steps"]:
        raise ValueError("render schedule migration requires a checkpoint before the map boundary")
    if state.get("total_step") != state.get("step"):
        raise ValueError("render schedule migration requires a committed map checkpoint")
    if "optimizer" not in payload or "sample_rng" not in payload:
        raise ValueError("render schedule migration requires optimizer and RNG state")
    if (settings.get("augmentation", NO_AUGMENTATION) == D4_AUGMENTATION
            and "augmentation_rng" not in payload):
        raise ValueError("D4 render schedule migration requires augmentation RNG state")
    _assert_transition_contract(payload.get("contract"), current_contract, allow_render_steps=True)


def _render_restart_state(payload, *, overfit=False):
    """Create the fresh render-stage state without inheriting optimizer/RNG."""
    baseline = copy.deepcopy(payload["state"]["baseline"])
    return {"stage": "render", "step": 0, "total_step": 100000,
            "best_map": baseline["map_mean"], "best_render": baseline["render_l1"],
            "baseline": baseline, "history": [], "refs": {}, "overfit": bool(overfit)}


def _is_legacy_l1_resume_migration(checkpoint_contract, current_contract):
    """Accept exactly the historical L1 run when only runtime policy evolved."""
    stored = copy.deepcopy(checkpoint_contract)
    current = copy.deepcopy(current_contract)
    _normalize_contract_augmentation(stored)
    _normalize_contract_augmentation(current)
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


def _validate_performance_migration(source_contract, current_contract):
    """Allow only the audited renderer/cache implementation transition."""
    source = copy.deepcopy(source_contract)
    current = copy.deepcopy(current_contract)
    _normalize_contract_augmentation(source)
    _normalize_contract_augmentation(current)
    source_sources = source.pop("source_sha256", None)
    current_sources = current.pop("source_sha256", None)
    if not isinstance(source_sources, dict) or not isinstance(current_sources, dict):
        raise ValueError("performance migration source has no source hashes")
    if source_sources.get("train_fabric.py") not in KNOWN_PERFORMANCE_TRAINER_SOURCE_SHA256:
        raise ValueError("performance migration source trainer hash is not an approved paused run")
    changed = {name for name in set(source_sources) | set(current_sources)
               if source_sources.get(name) != current_sources.get(name)}
    if changed != {"train_fabric.py"}:
        raise ValueError(f"performance migration changed unexpected sources: {sorted(changed)}")
    for settings in (source.get("settings", {}), current.get("settings", {})):
        if not isinstance(settings, dict):
            raise ValueError("performance migration contract has invalid settings")
        # ``map_only`` was added after the paused baseline was written.  Its
        # historical omission means the default (full map+render run), not an
        # implicit request to change the resumed scientific schedule.
        settings.setdefault("map_only", False)
        settings.setdefault("amp", False)
        settings.pop("observation_cache", None)
        settings.pop("performance_profile", None)
        settings.pop("fuse_accumulation", None)
        settings.pop("log_flush_every", None)
        settings.pop("eval_batch_size", None)
        settings.pop("amp", None)
    for environment in (source.get("environment", {}), current.get("environment", {})):
        if not isinstance(environment, dict):
            raise ValueError("performance migration contract has invalid environment")
        for key in ("tf32_matmul", "tf32_cudnn", "float32_matmul_precision"):
            environment.pop(key, None)
    source.pop("performance_migration", None)
    current.pop("performance_migration", None)
    if source != current:
        raise ValueError("performance migration differs in a scientific or dataset setting")


def _atomic_copy(source, destination):
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_link_or_copy(source, destination):
    """Atomically materialize an immutable checkpoint alias.

    Checkpoint generations and selection candidates are never mutated after
    publication.  A same-volume hard link therefore has identical bytes and
    crash behavior while avoiding repeated 257--771MB copies.  Filesystems
    without hard-link support fall back to the old atomic copy path.
    """
    source = Path(source)
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.link(source, temporary)
    except (OSError, NotImplementedError):
        shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_checkpoint(path, model, contract, state, *, optimizer=None, rng=None, metrics=None,
                    refs=None, generation=None, scaler=None, augmentation_rng=None):
    """Atomically save a v2 candidate or full resumable state."""
    if (optimizer is None) != (rng is None):
        raise ValueError("optimizer and rng must be supplied together")
    if augmentation_rng is not None and optimizer is None:
        raise ValueError("augmentation_rng is only stored with resumable optimizer state")
    if (optimizer is not None
            and contract.get("settings", {}).get("augmentation", NO_AUGMENTATION) == D4_AUGMENTATION
            and augmentation_rng is None):
        raise ValueError("D4 resumable checkpoints require augmentation_rng state")
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
        if augmentation_rng is not None:
            payload["augmentation_rng"] = augmentation_rng.getstate()
        if scaler is not None:
            payload["amp_scaler"] = scaler.state_dict()
    _atomic_torch_save(path, payload)
    return Path(path)


def load_checkpoint(path, model, contract, *, optimizer=None, rng=None, augmentation_rng=None,
                    allow_legacy_l1_resume_migration=False,
                    allow_render_schedule_migration=False,
                    allow_performance_migration=False, payload=None, scaler=None):
    if payload is None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a Fabric training v2 checkpoint")
    if contract.get("input", {}).get("feature_version") == FEATURE_VERSION and payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("RAW Fabric checkpoints must use raw_calibrated_v2")
    schedule_migration = False
    performance_migration = False
    if payload.get("contract") != contract and allow_render_schedule_migration:
        _validate_render_schedule_migration(payload, contract)
        schedule_migration = True
    if payload.get("contract") != contract and not schedule_migration and allow_performance_migration:
        _validate_performance_migration(payload.get("contract"), contract)
        performance_migration = True
    if payload.get("contract") != contract and not schedule_migration and not performance_migration and not (
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
        is_d4 = contract.get("settings", {}).get("augmentation", NO_AUGMENTATION) == D4_AUGMENTATION
        if is_d4 and augmentation_rng is None:
            raise ValueError("D4 resume requires a dedicated augmentation_rng")
        if is_d4 and "augmentation_rng" not in payload:
            raise ValueError("D4 checkpoint has no augmentation RNG state")
        if not is_d4 and augmentation_rng is not None:
            raise ValueError("augmentation_rng is only valid for D4 resume")
        optimizer.load_state_dict(payload["optimizer"])
        rng.setstate(payload["sample_rng"])
        if is_d4:
            augmentation_rng.setstate(payload["augmentation_rng"])
        if "python_rng" in payload:
            random.setstate(payload["python_rng"])
        if "numpy_rng" in payload:
            np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        if scaler is not None:
            if "amp_scaler" not in payload:
                if scaler.is_enabled():
                    raise ValueError("AMP resume checkpoint has no grad-scaler state")
            else:
                scaler.load_state_dict(payload["amp_scaler"])
    return payload


def _candidate_path(output, kind, total_step):
    return output / f"{kind}.step-{int(total_step):08d}.pth"


def _write_candidate(output, kind, model, contract, state, metrics):
    path = _candidate_path(output, kind, state["total_step"])
    if path.exists():
        raise FileExistsError(f"immutable selection candidate already exists: {path.name}")
    save_checkpoint(path, model, contract, state, metrics=metrics, refs=state.get("refs", {}))
    return {"path": path.name, "sha256": file_hash(path), "step": state["total_step"]}


def _rewrite_selection_candidate(output, ref, contract, *, migration_tag):
    """Copy an old immutable selection under the migrated strict contract."""
    if not isinstance(ref, dict) or not isinstance(ref.get("path"), str):
        raise ValueError("invalid selection reference during schedule migration")
    source = Path(output) / ref["path"]
    if not source.is_file() or file_hash(source) != ref.get("sha256"):
        raise ValueError("selection reference is missing or has a hash mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("params"), dict):
        raise ValueError("selection candidate is not a valid checkpoint")
    comparison_contract = copy.deepcopy(contract)
    comparison_contract.pop("render_schedule_migration", None)
    comparison_contract.pop("render_restart", None)
    _assert_transition_contract(payload.get("contract"), comparison_contract, allow_render_steps=True)
    # Keep migrated selections outside the generation glob so cleanup cannot
    # mistake them for an unreferenced historical candidate.
    destination = source.with_name(f"{migration_tag}.{source.name}")
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if existing.get("contract") != contract:
            raise FileExistsError(f"migrated selection already exists: {destination.name}")
    else:
        migrated = copy.deepcopy(payload)
        migrated["contract"] = copy.deepcopy(contract)
        _atomic_torch_save(destination, migrated)
    return {"path": destination.name, "sha256": file_hash(destination), "step": ref.get("step")}


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


def _commit_last(output, model, contract, state, optimizer, rng, *, scaler=None,
                 augmentation_rng=None):
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
                    refs=state.get("refs", {}), generation=generation, scaler=scaler,
                    augmentation_rng=augmentation_rng)
    previous = output / "last.pt"
    if previous.exists():
        _atomic_link_or_copy(previous, output / "last.prev.pt")
    _atomic_link_or_copy(generation_path, previous)
    _cleanup_checkpoint_candidates(output)
    return previous


def _rewrite_last_contract(output, contract, state, *, migration_tag):
    """Commit a transition contract before training can run another step."""
    previous = Path(output) / "last.pt"
    if not previous.is_file():
        raise ValueError("schedule migration requires a committed last.pt")
    payload = torch.load(previous, map_location="cpu", weights_only=False)
    generation = payload.get("generation")
    if not isinstance(generation, str):
        raise ValueError("schedule migration source last.pt has no generation")
    migrated_generation = f"{Path(generation).stem}.{migration_tag}.pt"
    migrated_path = Path(output) / migrated_generation
    migrated = copy.deepcopy(payload)
    migrated["contract"] = copy.deepcopy(contract)
    migrated["state"] = copy.deepcopy(state)
    migrated["refs"] = copy.deepcopy(state.get("refs", {}))
    migrated["generation"] = migrated_generation
    if migrated_path.exists():
        existing = torch.load(migrated_path, map_location="cpu", weights_only=False)
        if (existing.get("generation") != migrated_generation
                or existing.get("contract") != migrated["contract"]
                or existing.get("state") != migrated["state"]):
            raise FileExistsError(f"migrated last generation already exists: {migrated_path.name}")
    else:
        _atomic_torch_save(migrated_path, migrated)
    _atomic_link_or_copy(previous, Path(output) / "last.prev.pt")
    _atomic_link_or_copy(migrated_path, previous)
    _cleanup_checkpoint_candidates(output)
    return previous


def _materialize_aliases(output, refs):
    for kind, ref in refs.items():
        if kind not in ("best_map", "best_render", "stage_a"):
            continue
        source = output / ref["path"]
        if not source.is_file() or file_hash(source) != ref["sha256"]:
            raise ValueError(f"cannot materialize uncommitted selection: {kind}")
        _atomic_link_or_copy(source, output / f"{kind}.pth")


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
                "eval_crops", "test_crops", "test_lights", "render_ramp", "checkpoint_every",
                "log_flush_every", "eval_batch_size")
    if args.map_steps < 1 or args.render_steps < 0 or any(getattr(args, key) < 1 for key in positive if key != "render_steps") or args.stop_after < 0:
        raise ValueError("step counts, batch, accumulation, crop/light counts, and intervals must be positive")
    if args.render_steps == 0 and not args.map_only:
        raise ValueError("render_steps=0 requires --map-only")
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in (args.render_weight, args.map_tolerance)):
        raise ValueError("render weight and map tolerance must be finite and nonnegative")
    render_restart_source = getattr(args, "render_restart_source", "")
    allow_render_schedule_migration = bool(getattr(args, "allow_render_schedule_migration", False))
    if render_restart_source and args.resume:
        raise ValueError("--render-restart-source is incompatible with --resume")
    if render_restart_source and allow_render_schedule_migration:
        raise ValueError("render restart and render schedule migration are mutually exclusive")
    if allow_render_schedule_migration and not args.resume:
        raise ValueError("--allow-render-schedule-migration requires --resume")
    allow_performance_migration = bool(getattr(args, "allow_performance_migration", False))
    if allow_performance_migration and not args.resume:
        raise ValueError("--allow-performance-migration requires --resume")
    if allow_performance_migration and allow_render_schedule_migration:
        raise ValueError("performance migration and render schedule migration are mutually exclusive")
    if render_restart_source and args.render_steps != 100000:
        raise ValueError("render restart requires render_steps=100000")
    if render_restart_source and args.map_only:
        raise ValueError("render restart is incompatible with --map-only")
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
    _configure_performance_profile(args.performance_profile)
    contract = build_contract(args, args.manifest)
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if allow_performance_migration:
            _validate_performance_migration(resume_payload.get("contract"), contract)
    # Transition provenance is persisted in the output contract.  On ordinary
    # resume it is restored from the committed checkpoint so config.json is
    # only a human-readable mirror and cannot mask a partial commit.
    if args.resume:
        committed_contract = resume_payload.get("contract", {}) if isinstance(resume_payload, dict) else {}
        for key in ("render_restart", "render_schedule_migration"):
            if key == "render_schedule_migration" and allow_render_schedule_migration:
                # A prior process may have recorded the transition before
                # rewriting last.pt. Revalidate the old source contract on an
                # explicit retry instead of making that partial record part of
                # the source comparison.
                continue
            if key in committed_contract:
                contract[key] = copy.deepcopy(committed_contract[key])
    render_restart_payload = None
    render_restart_initial_path = None
    render_restart_record = None
    render_schedule_migration_payload = None
    render_schedule_migration_record = None
    if render_restart_source:
        render_restart_payload, render_restart_initial_path = _validate_render_restart_source(
            render_restart_source, contract)
        render_restart_record = {
            "kind": RENDER_RESTART_KIND,
            "source_checkpoint": str(Path(render_restart_source).resolve()),
            "source_checkpoint_sha256": file_hash(render_restart_source),
            "source_contract_sha256": _json_sha256(render_restart_payload["contract"]),
            "parent_initial_sha256": file_hash(render_restart_initial_path),
            "parent_total_step": render_restart_payload["state"]["total_step"],
            "source_settings": copy.deepcopy(render_restart_payload["contract"]["settings"]),
            "new_settings": {"map_steps": args.map_steps, "render_steps": args.render_steps,
                             "render_weight": args.render_weight, "render_ramp": args.render_ramp,
                             "normal_head": args.normal_head, "normal_loss": args.normal_loss},
            "reset_semantics": {
                "rng": "fresh_seeded_rng",
                "optimizer": "fresh_adam_lr_1e-4",
                "stage": "render",
                "step": 0,
                "total_step": 100000,
                "baseline": "recomputed_under_new_render_loss",
                "best_render": "reset_to_stage_a_render_l1",
                "initial": "recomputed_under_new_render_loss",
            },
        }
        contract["render_restart"] = copy.deepcopy(render_restart_record)
    if allow_render_schedule_migration:
        render_schedule_migration_payload = resume_payload
        _validate_render_schedule_migration(render_schedule_migration_payload, contract)
        source_settings = render_schedule_migration_payload["contract"]["settings"]
        render_schedule_migration_record = {
            "kind": RENDER_SCHEDULE_MIGRATION_KIND,
            "source_checkpoint": str(Path(args.resume).resolve()),
            "source_checkpoint_sha256": file_hash(args.resume),
            "source_contract_sha256": _json_sha256(render_schedule_migration_payload["contract"]),
            "source_render_steps": source_settings["render_steps"],
            "new_render_steps": args.render_steps,
            "preserved": ["map_optimizer", "optimizer_state", "sample_rng", "python_rng",
                           "numpy_rng", "torch_rng", "map_step", "map_schedule"],
            "reset": [],
            "semantics": "explicit_25k_to_100k_render_schedule_extension_before_render",
        }
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = ScratchModel(SimpleNamespace(save_root=str(output), feature_version=FEATURE_VERSION))
    observation_caches = {}
    if args.observation_cache != "off":
        datasets = [train_data, eval_data] + ([] if test_data is None else [test_data])
        for dataset in datasets:
            key = id(dataset)
            if key not in observation_caches:
                observation_caches[key] = FabricObservationCache(
                    output / "observations", dataset, dataset.records[0]["split"]
                )
        if args.observation_cache == "eager":
            for dataset in datasets:
                observation_caches[id(dataset)].build(model)
    train_observation_cache = observation_caches.get(id(train_data))
    eval_observation_cache = observation_caches.get(id(eval_data))
    test_observation_cache = (None if test_data is None
                              else observation_caches.get(id(test_data)))
    rng = random.Random(args.seed)
    aug_rng = augmentation_rng(args.seed, args.augmentation)
    optimizer = torch.optim.Adam(model.net_g.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=0)
    scaler = _new_grad_scaler(args.amp)
    state = {"stage": "map", "step": 0, "total_step": 0, "best_map": None, "best_render": None,
             "baseline": None, "history": [], "refs": {}, "overfit": bool(args.overfit)}
    resumed_with_legacy_policy_migration = False
    resumed_with_render_schedule_migration = False
    resumed_with_performance_migration = False
    if render_restart_payload is not None:
        model.net_g.load_state_dict(render_restart_payload["params"], strict=True)
        model.feature_version = render_restart_payload.get("feature_version", FEATURE_VERSION)
        state = _render_restart_state(render_restart_payload, overfit=args.overfit)
        restarted_evaluation = evaluate(
            model, eval_data, crop_count=1 if args.overfit else args.eval_crops,
            light_count=args.lights, seed=LOSS_PROBE_SEED, normal_head=args.normal_head,
            observation_cache=eval_observation_cache, eval_batch_size=args.eval_batch_size,
        )
        baseline = restarted_evaluation["mean"]
        state.update(baseline=baseline, best_map=baseline["map_mean"],
                     best_render=baseline["render_l1"], history=[])
        optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)
        write_json(output / "config.json", contract)
        write_json(output / "initial.json", {"status": "before_render_restart", "held_out": not args.overfit, "scope": "validation", "evaluation": restarted_evaluation})
        write_json(output / "render_restart.json", render_restart_record)
        stage_ref = _write_candidate(output, "stage_a", model, contract, state, baseline)
        render_ref = _write_candidate(output, "best_render", model, contract, state, baseline)
        state["refs"].update(stage_a=stage_ref, best_render=render_ref)
        _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                     augmentation_rng=aug_rng)
        _materialize_aliases(output, state["refs"])
    elif args.resume:
        payload = load_checkpoint(args.resume, model, contract, optimizer=optimizer, rng=rng,
                                  augmentation_rng=aug_rng,
                                  allow_legacy_l1_resume_migration=args.allow_legacy_l1_resume_migration,
                                  allow_render_schedule_migration=allow_render_schedule_migration,
                                  allow_performance_migration=allow_performance_migration,
                                  payload=resume_payload, scaler=scaler)
        resumed_with_render_schedule_migration = allow_render_schedule_migration and payload["contract"] != contract
        resumed_with_legacy_policy_migration = (not allow_render_schedule_migration
                                                and not allow_performance_migration
                                                and payload["contract"] != contract)
        resumed_with_performance_migration = allow_performance_migration and payload["contract"] != contract
        state = payload["state"]
        _verify_refs(output, state)
        if resumed_with_render_schedule_migration:
            contract["render_schedule_migration"] = copy.deepcopy(render_schedule_migration_record)
            # The map best candidate was written under the historical
            # 25k-render contract and may be needed at the map boundary even
            # when no later validation improves it.  Retain its parameters
            # and metrics under a new strict current-contract candidate.
            if state.get("refs", {}).get("best_map") is not None:
                state["refs"]["best_map"] = _rewrite_selection_candidate(
                    output, state["refs"]["best_map"], contract,
                    migration_tag="render100k")
            _verify_refs(output, state)
            write_json(output / "config.json", contract)
            _rewrite_last_contract(output, contract, state, migration_tag="render100k")
        if resumed_with_performance_migration:
            performance_record = {
                "kind": PERFORMANCE_MIGRATION_KIND,
                "version": PERFORMANCE_MIGRATION_VERSION,
                "source_checkpoint": str(Path(args.resume).resolve()),
                "source_checkpoint_sha256": file_hash(args.resume),
                "source_contract_sha256": _json_sha256(payload["contract"]),
                "accepted_changes": [
                    "batched_fixed_near_far_forward_render",
                    "batched_novel_light_render_loss",
                    "loss_diagnostic_host_sync_coalescing",
                    "immutable_checkpoint_hardlink_aliases",
                    "buffered_train_log",
                    "fp16_autocast_grad_scaling",
                ],
                "scientific_contract": "unchanged_raw_calibrated_v2_float32_linear_no_augmentation",
            }
            contract["performance_migration"] = performance_record
            write_json(output / "config.json", contract)
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
        if resumed_with_render_schedule_migration:
            write_json(output / "render_schedule_migration.json", render_schedule_migration_record)
    write_json(output / "config.json", contract)
    initial_path = output / "initial.json"
    if render_restart_payload is not None:
        initial = json.loads(initial_path.read_text(encoding="utf-8"))["evaluation"]
    elif args.resume:
        if not initial_path.is_file():
            raise ValueError("resume requires the committed initial.json evaluation artifact")
        initial = json.loads(initial_path.read_text(encoding="utf-8"))["evaluation"]
    else:
        initial = evaluate(model, eval_data, crop_count=1 if args.overfit else args.eval_crops,
                           light_count=args.lights, seed=LOSS_PROBE_SEED,
                           normal_head=args.normal_head,
                           observation_cache=eval_observation_cache,
                           eval_batch_size=args.eval_batch_size)
        write_json(initial_path, {"status": "before_training", "held_out": not args.overfit,
                                  "scope": "train_crop" if args.overfit else "validation",
                                  "evaluation": initial})
        # A crash before the first validation still has an exact step-0 state.
        _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                     augmentation_rng=aug_rng)
    started = time.perf_counter()
    invocation_steps = 0
    train_stream = (output / "train.jsonl").open("a", encoding="utf-8", buffering=1024 * 1024)
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
                                               allow_legacy_l1_resume_migration=resumed_with_legacy_policy_migration,
                                               allow_render_schedule_migration=resumed_with_render_schedule_migration,
                                               allow_performance_migration=resumed_with_performance_migration)
                    state.update(stage="render", step=0, baseline=selected["metrics"],
                                 best_render=selected["metrics"]["render_l1"])
                    if args.map_only:
                        state.update(stage="complete", step=args.map_steps,
                                     baseline=selected["metrics"], best_render=None)
                        train_stream.flush()
                        _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                                     augmentation_rng=aug_rng)
                        _materialize_aliases(output, state["refs"])
                        break
                    optimizer = torch.optim.Adam(model.net_g.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=0)
                    stage_ref = _write_candidate(output, "stage_a", model, contract, state, state["baseline"])
                    render_ref = _write_candidate(output, "best_render", model, contract, state, state["baseline"])
                    state["refs"].update(stage_a=stage_ref, best_render=render_ref)
                    _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                                 augmentation_rng=aug_rng)
                    _materialize_aliases(output, state["refs"])
                    continue
                state["stage"] = "complete"
                _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                             augmentation_rng=aug_rng)
                _materialize_aliases(output, state["refs"])
                break
            lr = cosine_lr(state["step"], steps, 5e-4 if stage == "map" else 1e-4)
            for group in optimizer.param_groups:
                group["lr"] = lr
            weight = 0.0 if stage == "map" else args.render_weight * min(1.0, (state["step"] + 1) / args.render_ramp)
            values = optimizer_step(model, train_data, optimizer, rng, batch_size=args.batch,
                                    accumulate=args.accumulate, weight=weight, light_count=args.lights,
                                    fixed_crop=args.overfit, normal_kind=args.normal_loss,
                                    normal_head=args.normal_head,
                                    observation_cache=train_observation_cache,
                                    fuse_accumulation=args.fuse_accumulation,
                                    scaler=scaler, augmentation=args.augmentation,
                                    augmentation_rng=aug_rng)
            state["step"] += 1
            state["total_step"] += 1
            invocation_steps += 1
            log = {"stage": stage, "step": state["step"], "total_step": state["total_step"],
                   "lr": lr, "render_weight": weight, **values}
            train_stream.write(json.dumps(log, allow_nan=False) + "\n")
            if state["total_step"] % args.log_flush_every == 0:
                train_stream.flush()
            validation_due = state["step"] % args.val_every == 0 or state["step"] == steps
            committed = False
            if validation_due:
                train_stream.flush()
                latest = evaluate(model, eval_data, crop_count=1 if args.overfit else args.eval_crops,
                                  light_count=args.lights, seed=LOSS_PROBE_SEED,
                                  normal_head=args.normal_head,
                                  observation_cache=eval_observation_cache,
                                  eval_batch_size=args.eval_batch_size)
                metrics = latest["mean"]
                state["history"].append({"stage": stage, "step": state["step"], **metrics})
                if state["best_map"] is None or metrics["map_mean"] < state["best_map"]:
                    state["best_map"] = metrics["map_mean"]
                    state["refs"]["best_map"] = _write_candidate(output, "best_map", model, contract, state, metrics)
                if stage == "render" and render_candidate(metrics, state["baseline"], state["best_render"], args.map_tolerance):
                    state["best_render"] = metrics["render_l1"]
                    state["refs"]["best_render"] = _write_candidate(output, "best_render", model, contract, state, metrics)
                _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                             augmentation_rng=aug_rng)
                _materialize_aliases(output, state["refs"])
                _write_validation(output, state, latest)
                print(f"val: map={metrics['map_mean']:.6f} render={metrics['render_l1']:.6f}", flush=True)
                committed = True
            elif _checkpoint_due(state["step"], steps, args.checkpoint_every):
                train_stream.flush()
                _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                             augmentation_rng=aug_rng)
                committed = True
            if args.stop_after and invocation_steps >= args.stop_after:
                train_stream.flush()
                if not committed:
                    _commit_last(output, model, contract, state, optimizer, rng, scaler=scaler,
                                 augmentation_rng=aug_rng)
                _materialize_aliases(output, state["refs"])
                write_json(output / "status.json", {"status": "paused", "stage": stage,
                                                     "step": state["step"], "total_step": state["total_step"]})
                return
            if args.log_every and state["total_step"] % args.log_every == 0:
                print(f"{stage} {state['step']}/{steps}: loss={values['loss']:.6f} lr={lr:.3g}", flush=True)
        chosen_kind = "best_map" if args.map_only else "best_render"
        chosen_ref = state["refs"].get(chosen_kind)
        if chosen_ref is None:
            raise ValueError(f"complete run has no committed {chosen_kind}")
        _verify_refs(output, state)
        _materialize_aliases(output, state["refs"])
        chosen = output / f"{chosen_kind}.pth"
        selected = load_checkpoint(chosen, model, contract,
                                   allow_legacy_l1_resume_migration=resumed_with_legacy_policy_migration,
                                   allow_performance_migration=resumed_with_performance_migration)
        if args.overfit:
            final = evaluate(model, eval_data, crop_count=1, light_count=args.test_lights,
                             seed=TEST_PROBE_SEED, independent_views=True,
                             prediction_dir=output / "predictions" / "train_crop",
                             prediction_metadata={"checkpoint_ref": chosen_ref},
                             normal_head=args.normal_head,
                             observation_cache=eval_observation_cache,
                             eval_batch_size=args.eval_batch_size)
            test = {"held_out_evidence": False, "reason": "--overfit evaluates the same fixed train crop",
                    "train_crop": final, "initial": initial, "checkpoint": str(chosen),
                    "checkpoint_sha256": file_hash(chosen)}
        else:
            final = evaluate(model, test_data, crop_count=args.test_crops, light_count=args.test_lights,
                             seed=TEST_PROBE_SEED, independent_views=True,
                             prediction_dir=output / "predictions" / "test",
                             prediction_metadata={"checkpoint_ref": chosen_ref},
                             normal_head=args.normal_head,
                             observation_cache=test_observation_cache,
                             eval_batch_size=args.eval_batch_size)
            test = {**final, "held_out_evidence": True, "initial": initial,
                    "checkpoint": str(chosen), "checkpoint_sha256": file_hash(chosen),
                    "selection_metrics": selected["metrics"], "smoke_only": args.smoke}
        write_json(output / "test.json", test)
        write_json(output / "status.json", {"status": "complete", "total_step": state["total_step"],
                                             "smoke_only": args.smoke, "overfit": args.overfit,
                                             "map_only": args.map_only,
                                             "checkpoint": str(chosen)})
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"error": repr(error), "stage": state["stage"], "step": state["step"]})
        raise
    finally:
        train_stream.flush()
        train_stream.close()
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
    parser.add_argument("--render-steps", type=int, default=100000)
    parser.add_argument("--map-only", action="store_true",
                        help="stop after the map stage; render_steps is set to zero")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--lights", type=int, default=9,
                        help="random-azimuth lights per render-loss step; view is fixed top-down")
    parser.add_argument("--render-weight", type=float, default=0.5)
    parser.add_argument("--render-ramp", type=int, default=1000)
    parser.add_argument("--map-tolerance", type=float, default=0.02)
    parser.add_argument("--eval-crops", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8,
                        help="validation/test inference batch size; lower it if GPU memory is tight")
    parser.add_argument("--test-crops", type=int, default=16)
    parser.add_argument("--test-lights", type=int, default=30)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--log-flush-every", type=int, default=100,
                        help="flush buffered training logs every N steps")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="commit resumable last checkpoint every N intermediate steps")
    parser.add_argument("--performance-profile", choices=("strict", "throughput"), default="strict",
                        help="strict is deterministic FP32; throughput enables measured TF32/cuDNN autotuning")
    parser.add_argument("--amp", action="store_true",
                        help="enable CUDA FP16 autocast for the estimator with gradient scaling; losses/rendering stay FP32")
    parser.add_argument("--observation-cache", choices=("off", "lazy", "eager"), default="off",
                        help="lossless float32 cache for deterministic GT->near/far observations")
    parser.add_argument("--augmentation", choices=(NO_AUGMENTATION, D4_AUGMENTATION),
                        default=NO_AUGMENTATION,
                        help="training-only spatial augmentation; none preserves the native-tile baseline")
    parser.add_argument("--fuse-accumulation", action="store_true",
                        help="forward/backward the effective batch once; preserves per-microbatch loss reductions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normal-head", choices=NORMAL_HEADS, default="xyz",
                        help="normal output parameterization; phi_theta is a fixed +z-hemisphere decoder")
    parser.add_argument("--normal-loss", choices=NORMAL_LOSSES, default="l1",
                        help="normal objective only; validation/selection keep existing map L1")
    parser.add_argument("--resume", default="")
    parser.add_argument("--render-restart-source", default="",
                        help="start a fresh render-only run from a validated map100k L1 XYZ stage_a checkpoint")
    parser.add_argument("--allow-legacy-l1-resume-migration", action="store_true",
                        help="allow only the recorded pre-policy L1 baseline to resume under the 1000-step policy")
    parser.add_argument("--allow-render-schedule-migration", action="store_true",
                        help="explicitly migrate a committed map-stage 25k-render checkpoint to the 100k render schedule")
    parser.add_argument("--allow-performance-migration", action="store_true",
                        help="explicitly resume the audited paused run after performance-only code changes")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--overfit", action="store_true", help="one material/crop; not held-out evidence")
    parser.add_argument("--smoke", action="store_true", help="four map + four render steps; implementation check only")
    args = parser.parse_args(argv)
    if args.smoke:
        args.map_steps = args.render_steps = 4
        args.eval_crops = args.test_crops = args.render_ramp = 1
        args.val_every = 2
        args.test_lights = 2
    if args.map_only:
        args.render_steps = 0
    if args.overfit:
        args.eval_crops = args.test_crops = 1
    return args


if __name__ == "__main__":
    run(parse_args())
