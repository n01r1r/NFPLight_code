from collections import OrderedDict
import json
from pathlib import Path

import torch

from einops import rearrange
import numpy as np


class MaterialBank:
    def __init__(self, manifest, split, *, cache_size, f0, crop_count):
        if split not in ('train', 'val') or cache_size < 1:
            raise ValueError('training bank permits train/val only and a positive cache size')
        rows = [json.loads(line) for line in Path(manifest).read_text(encoding = 'utf-8').splitlines() if line.strip()]
        self.records = [row for row in rows if row['split'] == split]
        if not self.records:
            raise ValueError('empty material population')
        self.cache = OrderedDict()
        self.cache_size, self.f0, self.crop_count = cache_size, f0, crop_count

    def crop(self, index, crop, rotation = 0, flip = False):
        if not 0 <= index < len(self.records) or not 0 <= crop < self.crop_count or rotation not in range(4):
            raise ValueError('invalid material/crop/rotation index')
        path = self.records[index]['shard_path']
        if path not in self.cache:
            with np.load(path, allow_pickle = False) as archive:
                maps = {key: archive[key] for key in ('normal', 'diffuse', 'roughness', 'specular')}
            for key, value in maps.items():
                channels = 1 if key == 'roughness' else 3
                expected = (self.crop_count, 256, 256, channels)
                if value.dtype != np.uint8 or (value.shape != expected and not (channels == 1 and value.shape == expected[:-1])):
                    raise ValueError(f'invalid bank map: {key}')
            self.cache[path] = maps
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last = False)
        maps = self.cache[path]
        self.cache.move_to_end(path)
        normal = maps['normal'][crop].astype(np.float32) / 127.5 - 1
        length = np.linalg.norm(normal, axis = -1, keepdims = True)
        if (length < .01).any():
            raise ValueError('invalid source normal')
        normal = normal / length
        diffuse = (maps['diffuse'][crop].astype(np.float32) / 255) ** 2.2
        roughness = maps['roughness'][crop].astype(np.float32) / 255
        if roughness.ndim == 2:
            roughness = roughness[..., None]
        specular = np.full_like(diffuse, self.f0)
        values = [normal, diffuse, roughness, specular]
        if rotation:
            values = [np.rot90(value, rotation, axes = (0, 1)).copy() for value in values]
            normal = values[0]
            x, y = normal[..., 0].copy(), normal[..., 1].copy()
            normal[..., 0], normal[..., 1] = ((-y, x), (-x, -y), (y, -x))[rotation - 1]
        if flip:
            values = [value[:, ::-1].copy() for value in values]
            values[0][..., 0] = -values[0][..., 0]
        normal, diffuse, roughness, specular = values
        values = [normal] + [np.clip(value, 0., 1.) * 2. - 1. for value in (diffuse, roughness, specular)]
        maps = np.ascontiguousarray(np.concatenate(values, axis = -1), dtype = np.float32)
        return rearrange(torch.from_numpy(maps), 'h w c -> c h w')


def canonical_target(target, *, roughness_floor):
    if (target.ndim != 4 or target.shape[1] != 10 or min(target.shape) < 1
            or target.dtype != torch.float32 or not torch.isfinite(target).all()
            or target.amin() < -1.00001 or target.amax() > 1.00001):
        raise ValueError('target must be finite FP32 B10HW in [-1,1]')
    lengths = target[:, :3].norm(dim = 1, keepdim = True)
    if not torch.allclose(lengths, torch.ones_like(lengths), atol = 1e-4, rtol = 0):
        raise ValueError('target normals must already have unit length')
    valid = target[:, 2:3] > 0
    if not valid.flatten(1).any(1).all():
        raise ValueError('crop has no front-facing target support')
    safe = target.clamp(-1, 1).clone()
    safe[:, 6:7].clamp_(min = 2 * roughness_floor - 1)
    fill = safe.new_tensor([0, 0, 1, -1, -1, -1, 1, -1, -1, -1])[None, :, None, None]
    return torch.where(valid, safe, fill), valid


def fixed_f0(maps, f0):
    return torch.cat((maps[:, :7], torch.full_like(maps[:, 7:10], 2 * f0 - 1)), dim = 1)


def map_loss(prediction, target, mask):
    if (prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 10
            or mask.dtype != torch.bool or mask.shape != (len(prediction), 1, *prediction.shape[-2:])
            or not mask.flatten(1).any(1).all()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()):
        raise ValueError('invalid NDRS loss inputs/support')
    parts = []
    for section in (slice(0, 3), slice(3, 6), slice(6, 7), slice(7, 10)):
        delta = (prediction[:, section] - target[:, section]).abs()
        support = mask.expand_as(delta)
        parts.append(delta.masked_select(support).sum() / support.sum())
    return sum(parts), parts
