"""MatSynth_crops256 loader for the from-scratch latter-stage estimator.

Source layout: root/<split>/<Category>/<name>.npz, each npz holding N (=4) crops
with keys basecolor(N,256,256,3 uint8, sRGB), metallic(N,256,256 uint8, linear),
roughness(N,256,256 uint8, linear), normal(N,256,256,3 uint8, map). No specular /
diffuse are stored -> both are DERIVED from the metallic workflow here, decoding
sRGB first (the trap the finetune_fabric loader falls into by reading sRGB maps
raw). Output is a 10-ch ndrs svbrdf in [-1,1] with LINEAR d/s, matching what the
renderer expects (utils/render_util._render: d,s used as d*0.5+0.5 linear).

See NFPLight_matsynth_scratch_plan.md sec 2 for the derivation and gotchas.
Drops straight into finetune_fabric.py's scan/split/balanced_weights/train loop
because items are (category, path) just like MatSynthMapDataset.
"""
import os
import random

import numpy as np
import torch

from utils import numpy_norm

DIELECTRIC_F0 = 0.04


def scan_crops(root, split):
    """root/<split>/<Category>/<name>.npz -> list of (category, npz_path)."""
    base = os.path.join(root, split)
    items = []
    if not os.path.isdir(base):
        return items
    for cat in sorted(os.listdir(base)):
        cdir = os.path.join(base, cat)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            if f.endswith(".npz"):
                items.append((cat, os.path.join(cdir, f)))
    return items


def metallic_to_ds(basecolor_srgb, metallic):
    """Metallic workflow -> (diffuse, F0), both linear. basecolor_srgb,metallic in
    [0,1]. F0 = lerp(0.04, basecolor_lin, m); diffuse = basecolor_lin*(1-m).
    Verified on disk to reproduce MatSynth's own specular-workflow specular.png."""
    bl = np.power(basecolor_srgb, 2.2)                 # sRGB -> linear
    m = metallic[..., None] if metallic.ndim == 2 else metallic
    F0 = DIELECTRIC_F0 * (1.0 - m) + bl * m
    diffuse = bl * (1.0 - m)
    return diffuse, F0


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


def build_svbrdf_from_crop(z, idx, hflip=False, vflip=False, rot=0):
    """One crop -> 10-ch ndrs svbrdf tensor [10,256,256] in [-1,1].
    rot in {0,1,2,3} = CCW 90deg turns (pure pixel permutation: no interpolation,
    no empty corners -- the frame stays fully valid, matching inference)."""
    bc = z["basecolor"][idx].astype(np.float32) / 255.0        # sRGB HxWx3
    m = z["metallic"][idx].astype(np.float32) / 255.0          # linear HxW
    rg = z["roughness"][idx].astype(np.float32) / 255.0        # linear HxW(x?)
    nm = z["normal"][idx].astype(np.float32) / 255.0           # map HxWx3

    d, s = metallic_to_ds(bc, m)                               # linear 3ch each
    r = (rg.mean(-1) if rg.ndim == 3 else rg)[..., None]       # 1ch
    n = numpy_norm(nm * 2.0 - 1.0, dim=-1)                     # unit, [-1,1]
    return pack_svbrdf(n, d, r, s, hflip=hflip, vflip=vflip, rot=rot)


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


class MatSynthCropsDataset(torch.utils.data.Dataset):
    """train: random crop-in-npz + flips. eval: crop 0, no aug. Returns
    {'svbrdf':[10,256,256], 'cat':str, 'name':str} to match MatSynthMapDataset."""

    def __init__(self, items, train=True):
        self.items = items                 # list of (category, npz_path)
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        cat, path = self.items[k]
        z = np.load(path)
        ncrop = z["normal"].shape[0]
        idx = random.randrange(ncrop) if self.train else 0
        hflip = self.train and random.random() < 0.5
        rot = random.randrange(4) if self.train else 0
        # vflip dropped: rot{0,90,180,270} x hflip already spans the full D4 dihedral
        # group (8 distinct transforms). Adding vflip only double-labels each element
        # (hflip o vflip == rot180), giving 16 params for the same 8 images -- redundant,
        # no extra coverage. build_svbrdf_from_crop keeps its vflip arg for test_aug_rot.
        sv = build_svbrdf_from_crop(z, idx, hflip, False, rot)
        return {"svbrdf": sv, "cat": cat, "name": os.path.basename(path)}
