"""Fabric-only native PNG cache and manifest. Run: python -m data.fabric --help.

Cache: uint8 [crop,256,256,10], encoded n/d/r/s; d/s are gamma22 RGB.
Dataset: float32 [10,256,256], unit OpenGL n and normalized linear d/r/s.
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError



def _rot90_normal_xy(n, k):
    """In-place remap of a normal map's in-plane (x,y) AFTER a spatial np.rot90(n,k)
    (k CCW quarter-turns), so the vectors stay attached to the rotated surface.
    k=2 negates both (convention-free 180deg); k=1/3 swap-with-sign, sign verified
    against the heightfield in test_aug_rot.py. Orthogonal -> unit norm preserved."""
    k %= 4
    if k == 0:
        return n
    nx = n[..., 0].copy()
    ny = n[..., 1].copy()
    if k == 1:                             # y-up (OpenGL) map: verified vs heightfield
        n[..., 0], n[..., 1] = -ny, nx
    elif k == 2:
        n[..., 0], n[..., 1] = -nx, -ny
    else:                                  # k == 3
        n[..., 0], n[..., 1] = ny, -nx
    return n


def pack_svbrdf(n, d, r, s, *, hflip=False, vflip=False, rot=0):
    """Unit normals and linear d/r/s -> augmented normalized CHW ndrs tensor."""

    if rot:
        d = np.rot90(d, rot, axes=(0, 1))
        s = np.rot90(s, rot, axes=(0, 1))
        r = np.rot90(r, rot, axes=(0, 1))
        n = np.rot90(n, rot, axes=(0, 1)).copy()               # writable for xy remap
        _rot90_normal_xy(n, rot)

    if hflip:
        d, s, r, n = d[:, ::-1], s[:, ::-1], r[:, ::-1], n[:, ::-1].copy()
        n[..., 0] = -n[..., 0]
    if vflip:
        d, s, r, n = d[::-1], s[::-1], r[::-1], n[::-1].copy()
        n[..., 1] = -n[..., 1]

    d = np.clip(d, 0.0, 1.0) * 2.0 - 1.0
    s = np.clip(s, 0.0, 1.0) * 2.0 - 1.0
    r = np.clip(r, 0.0, 1.0) * 2.0 - 1.0
    svbrdf = np.ascontiguousarray(
        np.concatenate([n, d, r, s], axis=-1), dtype=np.float32)   # HxWx10
    return torch.from_numpy(svbrdf).permute(2, 0, 1)


FORMAT = 'nfplight.fabric.native.v1'
MAP_CHANNELS = {'normal': 3, 'diffuse': 3, 'roughness': 1, 'specular': 3}
DEFAULT_MANIFEST = 'data/fabric_native256_v1/manifest.json'
TARGET_POLICY = {
    'version': 'nfplight.fabric.target.v2',
    'valid_pixels': 'native_normal_z_strictly_positive',
    'invalid_render_fill': 'flat_normal_zero_diffuse_zero_F0_roughness_one',
    'roughness_floor': 0.05,
    'gt_color_encoding': 'gamma22_compat',
    'augmentation': 'none',
}


def canonical_target(target):
    """Native normalized [B,10,H,W] -> renderer target and bool [B,1,H,W].

    Preserve supplied front-facing unit normals and decoded linear d/F0.
    Mask back-facing source pixels; never flip them into fabricated labels.
    Roughness below .05 has the same GGX response, so supervise its effective
    floor. The safe fill is only a numerical rendering placeholder, not GT.
    """
    if target.ndim != 4 or target.shape[1] != 10 or min(target.shape) < 1:
        raise ValueError('target must be nonempty [B,10,H,W]')
    if not target.is_floating_point() or not torch.isfinite(target).all():
        raise ValueError('target must contain finite floating-point values')
    if target.amin() < -1.00001 or target.amax() > 1.00001:
        raise ValueError('target must be normalized to [-1,1]')
    lengths = target[:, :3].norm(dim=1, keepdim=True)
    if not torch.allclose(lengths, torch.ones_like(lengths), atol=1e-4, rtol=0):
        raise ValueError('target normals must already be unit vectors')
    valid = target[:, 2:3] > 0
    if not valid.flatten(1).any(1).all():
        raise ValueError('target contains a crop with no valid front-facing pixels')
    safe = target.clamp(-1, 1).clone()
    safe[:, 6:7].clamp_(min=2 * TARGET_POLICY['roughness_floor'] - 1)
    fill = safe.new_tensor([0, 0, 1, -1, -1, -1, 1, -1, -1, -1])[None, :, None, None]
    return torch.where(valid, safe, fill), valid


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_map(path, channels):
    with Image.open(path) as image:
        pixels = np.asarray(image)
    if pixels.dtype != np.uint8:
        raise ValueError(f'{path}: expected uint8, got {pixels.dtype}')
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.ndim != 3 or pixels.shape[2] != channels:
        raise ValueError(f'{path}: expected {channels} channels, got {pixels.shape}')
    return pixels


def material_groups(records):
    """Join shared asset URLs and normal/roughness hashes, including transitive matches."""
    parents = {row['id']: row['id'] for row in records}

    def find(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen = {}
    for row in records:
        keys = [('geometry', row['hashes']['normal'], row['hashes']['roughness'])]
        link = row['link'].rstrip('/')
        # ponytail: exact asset links/hashes only; add curated groups for unmatched variants.
        if urlsplit(link).path.strip('/'):
            keys.append(('asset', row['source'], link))
        for key in keys:
            if key in seen:
                a, b = find(row['id']), find(seen[key])
                parents[max(a, b)] = min(a, b)
            seen[key] = row['id']
    return {row['id']: find(row['id']) for row in records}


def split_records(records, *, val_ratio, seed):
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be strictly between zero and one')
    groups = material_groups(records)
    test_groups = {groups[row['id']] for row in records if row['original_split'] == 'test'}
    kept, rejected = [], []
    for row in records:
        row = {**row, 'group': groups[row['id']]}
        if row['original_split'] == 'train' and row['group'] in test_groups:
            rejected.append({'id': row['id'], 'reason': 'shares_group_with_official_test'})
        else:
            kept.append(row)
    train_groups = sorted({row['group'] for row in kept if row['original_split'] == 'train'})
    if len(train_groups) < 2:
        raise ValueError('at least two independent train groups are required')
    random.Random(seed).shuffle(train_groups)
    sizes = Counter(row['group'] for row in kept if row['original_split'] == 'train')
    target = max(1, round(sum(sizes.values()) * val_ratio))
    val_groups, count = set(), 0
    for group in train_groups[:-1]:
        if count >= target:
            break
        val_groups.add(group)
        count += sizes[group]
    for row in kept:
        row['split'] = 'test' if row['original_split'] == 'test' else ('val' if row['group'] in val_groups else 'train')
    return kept, rejected


def cache_material(directory, destination, *, crop_count, seed, identity):
    metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))['metadata']
    opacity_declared = 'opacity' in metadata['maps']
    hashes = {key: file_hash(directory / f'{key}.png') for key in (*MAP_CHANNELS, 'opacity')}
    hashes['metadata'] = file_hash(directory / 'metadata.json')
    settings = {'hashes': hashes, 'crop_count': crop_count, 'seed': seed, 'format': FORMAT,
                'opacity_declared': opacity_declared}
    index_path = destination.with_suffix('.json')
    if destination.exists() and index_path.exists():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('settings') != settings or index['cache_sha256'] != file_hash(destination):
            raise RuntimeError(f'cache identity changed: {destination}; use a new output directory')
        return index

    opacity = read_map(directory / 'opacity.png', 1)
    nonopaque_fraction = float((opacity < 254).mean())
    if opacity_declared and nonopaque_fraction > 0:
        raise ValueError(f'nonopaque: opacity min={int(opacity.min())}/255')
    if not opacity_declared and nonopaque_fraction > 1e-6:
        raise ValueError(f'undeclared opacity is not an opaque placeholder: fraction={nonopaque_fraction}')
    opacity_audit = {'declared': opacity_declared, 'minimum': int(opacity.min()),
                     'fraction_below_254': nonopaque_fraction,
                     'policy': 'strict_declared_map' if opacity_declared else 'opaque_missing_map_placeholder'}
    del opacity
    maps = {key: read_map(directory / f'{key}.png', channels) for key, channels in MAP_CHANNELS.items()}
    shapes = {pixels.shape[:2] for pixels in maps.values()}
    if len(shapes) != 1:
        raise ValueError(f'map resolution mismatch: {sorted(shapes)}')
    height, width = shapes.pop()
    if min(height, width) < 256:
        raise ValueError(f'native map too small: {height}x{width}')
    # Validate all source normals row-wise, without allocating a full 4K float map.
    for strip in maps['normal']:
        normal = strip.astype(np.float32) / 127.5 - 1
        if (np.linalg.norm(normal, axis=-1) < 0.01).any():
            raise ValueError('invalid near-zero source normal')
    derived_seed = int(hashlib.sha256(f'{seed}:{identity}'.encode()).hexdigest()[:16], 16)
    rng = random.Random(derived_seed)
    coordinates = set()
    if crop_count > (height - 255) * (width - 255):
        raise ValueError('requested more unique crops than available windows')
    while len(coordinates) < crop_count:
        coordinates.add((rng.randrange(height - 255), rng.randrange(width - 255)))
    coordinates = sorted(coordinates)
    rng.shuffle(coordinates)
    tiles = np.stack([np.concatenate([pixels[y:y+256, x:x+256] for pixels in maps.values()], axis=-1)
                      for y, x in coordinates])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.npy.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, tiles, allow_pickle=False)
    temporary.replace(destination)
    result = {'settings': settings, 'coordinates_yx': coordinates, 'source_size_hw': [height, width],
              'opacity_audit': opacity_audit,
              'cache_sha256': file_hash(destination),
              'roughness_below_renderer_floor_fraction': float((maps['roughness'] < 0.05 * 255).mean())}
    write_json(index_path, result)
    return result


def prepare(root, output, *, crop_count=16, seed=0, val_ratio=0.1):
    if crop_count < 1:
        raise ValueError('crop_count must be positive')
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be strictly between zero and one')
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:
        raise ValueError('cache output must be outside the source dataset')
    records, rejected, test_errors = [], [], []
    for split in ('train', 'test'):
        folder = root / split / 'Fabric'
        if not folder.is_dir():
            raise FileNotFoundError(f'expected original Fabric directory: {folder}')
        for index, directory in enumerate(sorted(folder.iterdir())):
            if not directory.is_dir():
                continue
            identity = f'{split}/Fabric/{directory.name}'
            metadata = json.loads((directory / 'metadata.json').read_text(encoding='utf-8'))['metadata']
            if metadata['category'] != 'Fabric':
                raise ValueError(f'category metadata mismatch: {directory}')
            if metadata['source'] == 'deschaintre_2020':
                rejected.append({'id': identity, 'reason': 'excluded_source_deschaintre_2020'})
                continue
            cache = Path('crops') / split / f'{directory.name}.npy'
            try:
                details = cache_material(directory, output / cache, crop_count=crop_count, seed=seed, identity=identity)
            except (FileNotFoundError, UnidentifiedImageError, ValueError) as error:
                # Rejections remain visible; an invalid official test aborts publication.
                rejected.append({'id': identity, 'reason': str(error)})
                if split == 'test':
                    test_errors.append(identity)
                continue
            records.append({'id': identity, 'name': directory.name, 'category': 'Fabric',
                            'original_split': split, 'source': metadata['source'],
                            'link': metadata.get('link') or '', 'cache': cache.as_posix(),
                            'hashes': details['settings']['hashes'],
                            'cache_sha256': details['cache_sha256'],
                            'coordinates_yx': details['coordinates_yx'],
                            'opacity_audit': details['opacity_audit'],
                            'source_size_hw': details['source_size_hw'],
                            'roughness_below_renderer_floor_fraction': details['roughness_below_renderer_floor_fraction']})
            if (index + 1) % 20 == 0:
                print(f'{split}: inspected {index + 1}, accepted {len(records)}', flush=True)
    write_json(output / 'rejections.json', rejected)
    if test_errors:
        raise ValueError(f'invalid official test materials: {test_errors}; inspect rejections.json')
    records, group_rejections = split_records(records, val_ratio=val_ratio, seed=seed)
    rejected += group_rejections
    manifest = {'format': FORMAT, 'root': str(root), 'seed': seed, 'val_ratio': val_ratio,
                'crop_count': crop_count, 'crop_size': 256, 'gt_workflow': 'direct_diffuse_specular',
                'encoding': 'gamma22_compat', 'normal_convention': 'OpenGL_y_up',
                'excluded_sources': ['deschaintre_2020'], 'opacity_min': 254 / 255,
                'undeclared_opacity_max_defect_fraction': 1e-6,
                'counts': dict(Counter(row['split'] for row in records)),
                'records': records, 'rejected': rejected}
    if set(manifest['counts']) != {'train', 'val', 'test'}:
        raise ValueError(f'empty split: {manifest["counts"]}')
    path = output / 'manifest.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('existing manifest differs; use a new output directory')
    write_json(output / 'rejections.json', rejected)
    write_json(path, manifest)
    print(f'{path}: {manifest["counts"]}, rejected={len(rejected)}', flush=True)
    return manifest


def decode_tile(tile, *, rotation=0, hflip=False):
    if tile.dtype != np.uint8 or tile.shape != (256, 256, 10):
        raise ValueError(f'expected uint8 [256,256,10], got {tile.dtype} {tile.shape}')
    if rotation not in range(4):
        raise ValueError('rotation must be 0, 1, 2, or 3')
    values = tile.astype(np.float32) / 255
    normal = values[..., :3] * 2 - 1
    lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
    if (lengths < 0.01).any():
        raise ValueError('invalid near-zero cached normal')
    return pack_svbrdf(normal / lengths, values[..., 3:6] ** 2.2, values[..., 6:7],
                       values[..., 7:10] ** 2.2, rot=rotation, hflip=hflip)


class FabricDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, *, split, verify=True, limit=None):
        self.path = Path(manifest).resolve()
        self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
        document = self.manifest
        if (document['format'] != FORMAT or document['crop_size'] != 256
                or document['encoding'] != 'gamma22_compat' or document['gt_workflow'] != 'direct_diffuse_specular'):
            raise ValueError('unsupported Fabric manifest contract')
        seen, group_splits = set(), {}
        for row in document['records']:
            if row['category'] != 'Fabric' or row['source'] == 'deschaintre_2020':
                raise ValueError('manifest contains an excluded material')
            if row['id'] in seen:
                raise ValueError('duplicate material ID')
            seen.add(row['id'])
            previous = group_splits.setdefault(row['group'], row['split'])
            if previous != row['split']:
                raise ValueError('material group leaks across splits')
            if row['original_split'] == 'test' and row['split'] != 'test':
                raise ValueError('official test material was moved into training')
        if limit is not None and limit < 1:
            raise ValueError('limit must be positive')
        self.records = [row for row in document['records'] if row['split'] == split][:limit]
        if not self.records:
            raise ValueError(f'empty Fabric split: {split}')
        self.crops = []
        for row in self.records:
            path = (self.path.parent / row['cache']).resolve()
            if self.path.parent not in path.parents:
                raise ValueError('cache path escapes manifest directory')
            if verify and file_hash(path) != row['cache_sha256']:
                raise ValueError(f'cache hash mismatch: {path}')
            crops = np.load(path, mmap_mode='r', allow_pickle=False)
            if crops.shape != (document['crop_count'], 256, 256, 10) or crops.dtype != np.uint8:
                raise ValueError(f'cache shape/dtype mismatch: {path}')
            self.crops.append(crops)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.get_crop(index, 0)

    def get_crop(self, index, crop, *, rotation=0, hflip=False):
        return decode_tile(self.crops[index][crop], rotation=rotation, hflip=hflip)

    def sample(self, rng, batch_size, *, augmentation='none', augmentation_rng=None):
        """Sample cached tiles; optional D4 transforms use an independent RNG."""
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if augmentation not in ('none', 'd4'):
            raise ValueError(f'unknown augmentation: {augmentation}')
        if augmentation == 'd4' and augmentation_rng is None:
            raise ValueError('D4 augmentation requires a dedicated augmentation_rng')
        # ponytail: synchronous mmap reads; add workers only if measured I/O stalls the GPU.
        slots = [(rng.randrange(len(self)), rng.randrange(self.manifest['crop_count']),
                  0, False) for _ in range(batch_size)]
        if augmentation == 'd4':
            # 4 rotations x 2 horizontal-flip states enumerate D4 uniformly.
            transformed = []
            for index, crop, _, _ in slots:
                transform = augmentation_rng.randrange(8)
                transformed.append((index, crop, transform % 4, transform >= 4))
            slots = transformed
        batch = torch.stack([self.get_crop(i, c, rotation=r, hflip=f) for i, c, r, f in slots])
        return batch, slots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='D:/MatSynth_preprocessed')
    parser.add_argument('--out', default=str(Path(DEFAULT_MANIFEST).parent))
    parser.add_argument('--crops', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()
    prepare(args.root, args.out, crop_count=args.crops, seed=args.seed, val_ratio=args.val_ratio)
