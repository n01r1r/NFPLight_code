"""
v0 exploratory finetune of the final SVBRDF enc-dec (TwoBranchNet) on a
fabric-heavy MatSynth mixture.

Scope (deliberately minimal — this is an exploratory probe, not the paper's
main result):
  - Supervision: map-space L1 only. NO render loss (excluded by design for v0).
  - Adaptation: LoRA on the 1x1 convs of every NAFBlock + unfreeze the two
    output heads. Everything else frozen -> small-data safe, forgetting-light.
  - Data: D:/MatSynth, each material folder holds gt/{normal,diffuse,roughness,
    specular}.png plus pre-rendered near/far.png. Category = folder-name prefix
    (Fabric_, Metal_, ...), used for category-balanced sampling and per-category
    eval. Inputs are re-rendered from gt at 256 via the frozen NFPLight
    renderer, so train/inference input assembly is identical.

The pretrained inference pipeline (renderer, 21-ch input assembly, checkpoint
loading) is reused wholesale from NFPLightModel; we only inject LoRA into its
net_g and add a training loop.

Run a pipeline sanity check first (validates the gt->render->input path on
whatever materials exist):
    python finetune_fabric.py --sanity --data_root D:/MatSynth \
        --loadpath checkpoints/net_g_syn.pth

Then finetune once populated:
    python finetune_fabric.py --data_root D:/MatSynth \
        --loadpath checkpoints/net_g_syn.pth --out checkpoints/fabric_lora.pth
"""
import os
import argparse
import hashlib
import random
from types import SimpleNamespace
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nfplight_model import NFPLightModel
from network.nfplight_net import NAFBlock
from utils import numpy_norm


# ----------------------------------------------------------------------------
# LoRA on a conv layer. For the 1x1 convs that dominate a NAFBlock this is an
# exact low-rank factorization of the channel-mixing weight: A: in->r (1x1),
# B: r->out (1x1). B is zero-init so the adapter starts as a no-op (output ==
# pretrained). The base conv is frozen inside the wrapper.
# ----------------------------------------------------------------------------
class LoRAConv2d(nn.Module):
    def __init__(self, base: nn.Conv2d, r=16, alpha=16):
        super().__init__()
        assert base.groups == 1, "LoRA wrapper assumes non-grouped conv"
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride,
                           base.padding, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def inject_lora(net, r=16, alpha=16):
    """Wrap the 1x1 convs of every NAFBlock (conv1/3/4/5 + sca projection).
    The 3x3 depthwise conv2 is left frozen (grouped -> not a channel mixer)."""
    n = 0
    for m in net.modules():
        if isinstance(m, NAFBlock):
            for name in ("conv1", "conv3", "conv4", "conv5"):
                setattr(m, name, LoRAConv2d(getattr(m, name), r, alpha))
                n += 1
            # m.sca is Sequential(AdaptiveAvgPool2d, Conv2d 1x1)
            m.sca[1] = LoRAConv2d(m.sca[1], r, alpha)
            n += 1
    return n


def set_trainable(net):
    """Freeze base, keep LoRA (A/B) + the two output heads trainable."""
    for p in net.parameters():
        p.requires_grad_(False)
    for m in net.modules():
        if isinstance(m, LoRAConv2d):
            m.A.weight.requires_grad_(True)
            m.B.weight.requires_grad_(True)
    for head in (net.albedoMap, net.specularMap):
        for p in head.parameters():
            p.requires_grad_(True)
    return [p for p in net.parameters() if p.requires_grad]


# ----------------------------------------------------------------------------
# Dataset: read the four gt maps, apply scale-jitter + random crop + flips,
# assemble a [-1,1] 10-ch svbrdf in ndrs order following svBRDF._norm.
# ----------------------------------------------------------------------------
MAP_KEYS = ("normal", "diffuse", "roughness", "specular")


def _align_maps(raw, size):
    """The four maps of a material are usually the same resolution, but some
    MatSynth materials ship a map at a different res. Bring every map to a
    common target = max dims across maps (>= crop size). Only ever UPSAMPLES a
    smaller map (INTER_LINEAR) -- never downsamples, so native detail of the
    largest map is preserved. Returns (raw, h, w)."""
    hs = [v.shape[0] for v in raw.values()] + [size]
    ws = [v.shape[1] for v in raw.values()] + [size]
    th, tw = max(hs), max(ws)
    out = {}
    for k, v in raw.items():
        if v.shape[0] != th or v.shape[1] != tw:
            v = cv2.resize(v, (tw, th), interpolation=cv2.INTER_LINEAR)
        out[k] = v
    return out, th, tw


def _read01(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    scale = 65535.0 if img.dtype == np.uint16 else 255.0
    if img.ndim == 2:
        img = img[:, :, None].repeat(3, 2)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / scale


def _load_crop(mat_dir, size=256, train=True, rng=random):
    """Random (train) / center (eval) crop of the four maps at NATIVE resolution.
    NEVER downsamples — fabric thread detail must be preserved; scale-jitter by
    resize is forbidden. Reads uint8, slices the crop window, THEN casts to
    float32 so a full-res float array is never allocated (which also avoids the
    cv2 OOM a 4K resize caused). The four maps share size within a material."""
    raw = {k: cv2.imread(os.path.join(mat_dir, k + ".png"), cv2.IMREAD_UNCHANGED)
           for k in MAP_KEYS}
    for k, v in raw.items():
        if v is None:
            raise FileNotFoundError(os.path.join(mat_dir, k + ".png"))
    raw, h, w = _align_maps(raw, size)       # common res, upsample-only
    if train:
        top, left = rng.randint(0, h - size), rng.randint(0, w - size)
    else:
        top, left = (h - size) // 2, (w - size) // 2
    maps = {}
    for k in MAP_KEYS:
        c = raw[k][top:top + size, left:left + size]
        div = 65535.0 if c.dtype == np.uint16 else 255.0
        if c.ndim == 2:
            c = c[:, :, None].repeat(3, 2)
        else:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        maps[k] = c.astype(np.float32) / div
    return maps


# ----------------------------------------------------------------------------
# Tile cache: decoding a native 2K-4K PNG four times per __getitem__ just to
# take one 256 crop dominates wall-clock on a shared box. Pre-extract K native
# 256 tiles per material ONCE (random offsets, frozen), pack the four maps into
# a 10-ch uint8 array (normal3,diffuse3,rough1,spec3 -- lossless, maps are
# uint8), store one .npz per material with one array per tile so a read
# decompresses only the tile it needs. This is a decode-once cache, NOT a
# downsample: every cached pixel is a native-resolution crop.
# ----------------------------------------------------------------------------
def _cache_key(path):
    return hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:16]


def _pack_tiles(mat_dir, k=16, size=256, rng=random):
    raw = {key: cv2.imread(os.path.join(mat_dir, key + ".png"), cv2.IMREAD_UNCHANGED)
           for key in MAP_KEYS}
    for key, v in raw.items():
        if v is None:
            raise FileNotFoundError(os.path.join(mat_dir, key + ".png"))
    raw, h, w = _align_maps(raw, size)       # common res, upsample-only
    tiles = []
    for _ in range(k):
        top, left = rng.randint(0, h - size), rng.randint(0, w - size)
        chans = []
        for key in MAP_KEYS:
            c = raw[key][top:top + size, left:left + size]
            if c.ndim == 2:                              # roughness -> keep 1ch
                c = c[:, :, None]
            else:
                c = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)   # BGR->RGB (matches _load_crop)
                if key == "roughness":
                    c = c[:, :, :1]
            chans.append(c)
        tiles.append(np.concatenate(chans, axis=2))       # H,W,10 uint8
    return tiles


def _tile_to_maps(tile):
    f = tile.astype(np.float32) / 255.0
    return {"normal": f[..., 0:3], "diffuse": f[..., 3:6],
            "roughness": f[..., 6:7], "specular": f[..., 7:10]}


def build_cache(items, cache_dir, k=16, size=256, seed=0):
    """Pre-tile every material into `k` native 256 crops. Resumable (skips
    materials already cached). One .npz per material, one array per tile."""
    os.makedirs(cache_dir, exist_ok=True)
    done = skip = 0
    for i, (_, path) in enumerate(items):
        out = os.path.join(cache_dir, _cache_key(path) + ".npz")
        if os.path.isfile(out):
            skip += 1
            continue
        rng = random.Random(seed * 100003 + i)
        tiles = _pack_tiles(path, k, size, rng)
        np.savez_compressed(out, **{f"t{t}": tiles[t] for t in range(len(tiles))})
        done += 1
        if (done + skip) % 100 == 0:
            print(f"  cache {done + skip}/{len(items)} (new {done}, skip {skip})", flush=True)
    print(f"cache done: {done} built, {skip} existing -> {cache_dir}", flush=True)


class MatSynthMapDataset(torch.utils.data.Dataset):
    def __init__(self, items, size=256, train=True, cache_dir=None):
        self.items = items            # list of (category, folder_path)
        self.size = size
        self.train = train
        self.cache_dir = cache_dir    # if set, read pre-tiled crops instead of PNGs

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _flip(maps):
        hflip, vflip = random.random() < 0.5, random.random() < 0.5
        if hflip:
            maps = {k: v[:, ::-1].copy() for k, v in maps.items()}
        if vflip:
            maps = {k: v[::-1, :].copy() for k, v in maps.items()}
        return maps, hflip, vflip

    def _build_svbrdf(self, maps, hflip, vflip):
        # normal: [0,1] rgb -> unit vector, renormalized (matches svBRDF._norm)
        n = numpy_norm(maps["normal"] * 2 - 1, dim=-1)          # [-1,1], unit
        if hflip:
            n[..., 0] = -n[..., 0]                              # flip nx
        if vflip:
            n[..., 1] = -n[..., 1]                              # flip ny
        d = maps["diffuse"] ** 2.2 * 2 - 1                     # gamma22 -> linear
        r = maps["roughness"].mean(-1, keepdims=True) * 2 - 1   # 1ch [-1,1]
        s = maps["specular"] ** 2.2 * 2 - 1                    # gamma22 -> linear F0
        svbrdf = np.concatenate([n, d, r, s], axis=-1)          # H,W,10
        return torch.from_numpy(svbrdf).permute(2, 0, 1).float()

    def __getitem__(self, i):
        cat, path = self.items[i]
        if self.cache_dir:
            with np.load(os.path.join(self.cache_dir, _cache_key(path) + ".npz")) as z:
                idx = random.randrange(len(z.files)) if self.train else 0
                maps = _tile_to_maps(z[f"t{idx}"])
        else:
            maps = _load_crop(path, self.size, self.train)
        if self.train:
            maps, hflip, vflip = self._flip(maps)
        else:
            hflip = vflip = False
        return {"svbrdf": self._build_svbrdf(maps, hflip, vflip), "cat": cat}


def scan_materials(root, split="train"):
    """Layout: root/<split>/<Category>/<Material>/{normal,diffuse,...}.png.
    Falls back to the flat root/<Material>/gt/ sample layout if no split dir."""
    base = os.path.join(root, split)
    items = []
    if os.path.isdir(base):
        for cat in sorted(os.listdir(base)):
            cdir = os.path.join(base, cat)
            if not os.path.isdir(cdir):
                continue
            for name in sorted(os.listdir(cdir)):
                mdir = os.path.join(cdir, name)
                if os.path.isfile(os.path.join(mdir, "normal.png")):
                    items.append((cat, mdir))
    else:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if os.path.isfile(os.path.join(p, "gt", "normal.png")):
                items.append((name.split("_")[0], os.path.join(p, "gt")))
    return items


def split_by_material(items, val_ratio=0.15, seed=0):
    """Per-category material-level split (no crop leakage across splits)."""
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it[0]].append(it)
    rng = random.Random(seed)
    train, val = [], []
    for cat, lst in by_cat.items():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, int(round(len(lst) * val_ratio))) if len(lst) > 1 else 0
        val += lst[:k]
        train += lst[k:]
    return train, val


def balanced_weights(items, fabric_tilt=2.0):
    """Category-balanced sampling weights (each category equal expected mass),
    with an extra multiplicative tilt toward Fabric. The real deployment mix is
    unobservable, so we do NOT match a target ratio; the tilt is a knob chosen
    later by a no-regression gate on non-fabric val error."""
    counts = defaultdict(int)
    for c, _ in items:
        counts[c] += 1
    w = []
    for c, _ in items:
        base = 1.0 / counts[c]
        w.append(base * (fabric_tilt if c == "Fabric" else 1.0))
    return torch.tensor(w, dtype=torch.double)


# ----------------------------------------------------------------------------
# Input assembly: mirror NFPLightModel.test() (lines 116-132) but keep grad and
# take svbrdf as an argument. Returns the 21-ch net input.
# ----------------------------------------------------------------------------
def build_net_input(model, svbrdf, *, quantize=False):
    """Reuse inference features; historical callers keep unquantized rendering."""
    inputs = model.render_input_images(svbrdf, toLDR=quantize)
    return model.build_features(inputs)[0]


def map_loss(pred, gt, w=(1.0, 1.0, 1.0, 1.0)):
    pn, pd, pr, ps = torch.split(pred, [3, 3, 1, 3], 1)
    gn, gd, gr, gs = torch.split(gt, [3, 3, 1, 3], 1)
    parts = {
        "normal": F.l1_loss(pn, gn),
        "diffuse": F.l1_loss(pd, gd),
        "roughness": F.l1_loss(pr, gr),
        "specular": F.l1_loss(ps, gs),
    }
    total = (w[0] * parts["normal"] + w[1] * parts["diffuse"]
             + w[2] * parts["roughness"] + w[3] * parts["specular"])
    return total, parts


# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, items, size=256):
    """Per-category mean map-L1 on val materials (center crop, no aug)."""
    model.net_g.eval()
    ds = MatSynthMapDataset(items, size=size, train=False)
    per_cat = defaultdict(lambda: defaultdict(float))
    per_cat_n = defaultdict(int)
    for i in range(len(ds)):
        b = ds[i]
        svbrdf = b["svbrdf"].unsqueeze(0).cuda()
        x = build_net_input(model, svbrdf)
        pred = model.net_g(x)
        _, parts = map_loss(pred, svbrdf)
        for k, v in parts.items():
            per_cat[b["cat"]][k] += v.item()
        per_cat_n[b["cat"]] += 1
    out = {}
    for c, d in per_cat.items():
        out[c] = {k: v / per_cat_n[c] for k, v in d.items()}
        out[c]["mean"] = sum(out[c][k] for k in
                             ("normal", "diffuse", "roughness", "specular")) / 4
    return out


def print_eval(tag, ev):
    print(f"\n[{tag}] per-category map-L1")
    hdr = f"{'category':<10} {'normal':>8} {'diffuse':>8} {'rough':>8} {'spec':>8} {'mean':>8}"
    print(hdr)
    for c in sorted(ev):
        e = ev[c]
        print(f"{c:<10} {e['normal']:>8.4f} {e['diffuse']:>8.4f} "
              f"{e['roughness']:>8.4f} {e['specular']:>8.4f} {e['mean']:>8.4f}")


# ----------------------------------------------------------------------------
def sanity(model, items):
    """Validate the gt->render->input path: render near from gt and compare to
    the pre-rendered near.png (gamma-matched). Large error => a normalization or
    color-space mismatch in the loader."""
    ds = MatSynthMapDataset(items[:8], train=False)
    print("\n[sanity] svbrdf range + (if present) gt->render vs near.png")
    for i in range(len(ds)):
        b = ds[i]
        svbrdf = b["svbrdf"].unsqueeze(0).cuda()
        _, path = items[i]
        line = (f"  {b['cat']:<10} {os.path.basename(path):<28} "
                f"svbrdf[min/max]={svbrdf.min().item():.2f}/{svbrdf.max().item():.2f}")
        near_path = os.path.join(path, "near.png")
        if not os.path.isfile(near_path):
            near_path = os.path.join(os.path.dirname(path), "near.png")
        if os.path.isfile(near_path):
            rendered = model.render_input_images(svbrdf, toLDR=False)[:, :3]
            rendered = torch.clip(rendered, 0, 1) ** (1 / 2.2)
            s = model.renderer.size
            near = cv2.resize(_read01(near_path), (s, s), interpolation=cv2.INTER_AREA)
            near = torch.from_numpy(near).permute(2, 0, 1)[None].cuda()
            line += f"  dnear={F.l1_loss(rendered, near).item():.4f}"
        print(line)


def train(model, train_items, val_items, args):
    model.net_g.cuda()                 # move freshly-injected LoRA convs to GPU
    params = set_trainable(model.net_g)
    n_train = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.net_g.parameters())
    print(f"trainable params: {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.2f}%)")

    ds = MatSynthMapDataset(train_items, size=args.size, train=True)
    weights = balanced_weights(train_items, args.fabric_tilt)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=args.steps * args.batch, replacement=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.9),
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    if val_items:
        print_eval("baseline (val)", evaluate(model, val_items, args.size))

    model.net_g.train()
    best = float("inf")
    step = 0
    for batch in loader:
        svbrdf = batch["svbrdf"].cuda(non_blocking=True)
        x = build_net_input(model, svbrdf)
        pred = model.net_g(x)
        loss, parts = map_loss(pred, svbrdf, args.loss_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % args.log_every == 0:
            print(f"step {step:>5}/{args.steps}  loss {loss.item():.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  "
                  + " ".join(f"{k[:1]}={v.item():.3f}" for k, v in parts.items()))
        if val_items and step % args.val_every == 0:
            ev = evaluate(model, val_items, args.size)
            print_eval(f"step {step} (val)", ev)
            fab = ev.get("Fabric", {}).get("mean", float("inf"))
            if fab < best:
                best = fab
                save_adapter(model.net_g, args.out)
                print(f"  saved (Fabric val mean {fab:.4f}) -> {args.out}")
            model.net_g.train()
    if not val_items:
        save_adapter(model.net_g, args.out)
        print(f"saved -> {args.out}")


def save_adapter(net, path):
    state = {k: v for k, v in net.state_dict().items()
             if (".A." in k or ".B." in k or "albedoMap" in k or "specularMap" in k)}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"adapter": state}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="D:/MatSynth_preprocessed")
    ap.add_argument("--loadpath", default="checkpoints/net_g_syn.pth")
    ap.add_argument("--out", default="checkpoints/fabric_lora.pth")
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--scale_min", type=float, default=0.5)
    ap.add_argument("--scale_max", type=float, default=2.0)
    ap.add_argument("--fabric_tilt", type=float, default=2.0)
    ap.add_argument("--max_per_cat", type=int, default=0,
                    help="cap materials per category (0=all) for a faster probe")
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--val_every", type=int, default=300)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.loss_w = (1.0, 1.0, 1.0, 1.0)   # normal, diffuse, roughness, specular

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # NFPLightModel expects an args-like object with these fields.
    model = NFPLightModel(SimpleNamespace(
        loadpath_network_g=args.loadpath, save_root="./_ft_tmp"))

    all_train = scan_materials(args.data_root, "train")
    all_test = scan_materials(args.data_root, "test")
    if args.max_per_cat:
        all_train = cap_per_cat(all_train, args.max_per_cat, args.seed)
    cats = defaultdict(int)
    for c, _ in all_train:
        cats[c] += 1
    print(f"found {len(all_train)} train / {len(all_test)} test materials")
    print("  train per category:", dict(sorted(cats.items(), key=lambda x: -x[1])))

    if args.sanity:
        sanity(model, all_train)
        return

    n_lora = inject_lora(model.net_g, args.rank, args.alpha)
    print(f"injected LoRA into {n_lora} convs")
    train_items, val_items = split_by_material(all_train, args.val_ratio, args.seed)
    print(f"split: {len(train_items)} train / {len(val_items)} val materials")

    train(model, train_items, val_items, args)

    print_eval("FINAL test (official split)", evaluate(model, all_test, args.size))


def cap_per_cat(items, cap, seed=0):
    by = defaultdict(list)
    for it in items:
        by[it[0]].append(it)
    rng = random.Random(seed)
    out = []
    for lst in by.values():
        rng.shuffle(lst)
        out += lst[:cap]
    return out


if __name__ == "__main__":
    main()
