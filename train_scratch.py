"""From-scratch training of the latter-stage estimator (TwoBranchNet, 21-ch) on
MatSynth_crops256. Historical all-category experiments use clean clipped inputs
without LDR quantization. Use train_fabric.py for the quantized Fabric protocol.
Denoise stage is out of scope.

Loss = map-L1  +  render_w * novel-light render loss.
The novel-light render loss is the TRAINING embodiment of the repeat-capture
consistency intent: it forces the predicted svbrdf to match GT not only under the
near/far colocated cue the net sees as input, but under randomly-placed lights it
does NOT see -- constraining specular/roughness that a single colocated config
under-observes (redteam critical-4). Same intent as a real repeat-capture
consistency check, but with synthetic GT so it is fully supervised.

    # smoke (few steps, tiny subset):
    python train_scratch.py --steps 3 --max_per_cat 2 --val_every 0 --workers 0
    # real run:
    python train_scratch.py --crops D:/MatSynth_crops256 \
        --out checkpoints/net_g_matsynth.pth --steps 60000 --batch 4
"""
import os
import math
import random
import argparse
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model.nfplight_model import NFPLightModel
from network.nfplight_net import TwoBranchNet
from data.matsynth_crops import (
    build_svbrdf_from_crop, scan_crops, MatSynthCropsDataset,
)
from finetune_fabric import (
    build_net_input, map_loss, print_eval, split_by_material,
    balanced_weights, cap_per_cat,
)


class ScratchModel(NFPLightModel):
    """NFPLightModel but with a FRESH (random-init) net -- no checkpoint load.
    Reuses all the renderer / input-assembly machinery from the base."""
    def init_network(self):
        self.net_g = self.model_to_device(TwoBranchNet())


class FixedIndexSampler(torch.utils.data.Sampler):
    """Replay a precomputed post-resume training index stream exactly."""

    def __init__(self, indices, population_size):
        self.indices = torch.as_tensor(indices, dtype=torch.int64).flatten()
        if self.indices.numel() == 0:
            raise ValueError("sampler index stream must not be empty")
        if torch.any(self.indices < 0) or torch.any(self.indices >= population_size):
            raise ValueError(
                f"sampler index stream contains values outside [0, {population_size})"
            )

    def __iter__(self):
        return iter(self.indices.tolist())

    def __len__(self):
        return self.indices.numel()


class PlannedMatSynthCropsDataset(torch.utils.data.Dataset):
    """Training dataset driven by an immutable per-slot replay plan."""

    def __init__(self, items, plan):
        self.items = items
        self.material_indices = plan["material_indices"]
        self.crop_indices = plan["crop_indices"]
        self.hflips = plan["hflips"]
        self.rotations = plan["rotations"]
        if not all(len(values) == len(self.material_indices) for values in (
            self.crop_indices, self.hflips, self.rotations,
        )):
            raise ValueError("replay plan per-slot arrays must have equal lengths")

    def __len__(self):
        return len(self.material_indices)

    def __getitem__(self, slot):
        material_index = int(self.material_indices[slot])
        category, path = self.items[material_index]
        with np.load(path) as archive:
            svbrdf = build_svbrdf_from_crop(
                archive,
                int(self.crop_indices[slot]),
                bool(self.hflips[slot]),
                False,
                int(self.rotations[slot]),
            )
        return {
            "svbrdf": svbrdf,
            "cat": category,
            "name": os.path.basename(path),
        }


def load_sampler_indices(path):
    """Load a trusted ``torch.save`` replay payload or raw tensor/list."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    indices = payload.get("indices") if isinstance(payload, dict) else payload
    if indices is None:
        raise ValueError(f"sampler index payload has no 'indices': {path}")
    return torch.as_tensor(indices, dtype=torch.int64).flatten()


def load_train_plan(path, args, start_step, train_items):
    """Load and validate an immutable 30k->60k replay plan."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"replay plan must contain metadata: {path}")
    metadata = payload["metadata"]
    expected_steps = args.steps - start_step
    if metadata.get("parent_step") != start_step:
        raise ValueError(
            f"replay plan parent step {metadata.get('parent_step')} != resumed step {start_step}"
        )
    if metadata.get("target_steps") != args.steps:
        raise ValueError(
            f"replay plan target steps {metadata.get('target_steps')} != args.steps {args.steps}"
        )
    if metadata.get("batch") != args.batch:
        raise ValueError(
            f"replay plan batch {metadata.get('batch')} != args.batch {args.batch}"
        )
    if metadata.get("n_lights") != args.n_lights:
        raise ValueError(
            f"replay plan n_lights {metadata.get('n_lights')} != args.n_lights {args.n_lights}"
        )
    arrays = {
        key: torch.as_tensor(payload[key], dtype=dtype).flatten()
        for key, dtype in (
            ("material_indices", torch.int64),
            ("crop_indices", torch.int64),
            ("hflips", torch.bool),
            ("rotations", torch.int64),
        )
    }
    if any(len(values) != expected_steps * args.batch for values in arrays.values()):
        raise ValueError("replay plan slot arrays do not match remaining step count")
    if "render_lights" not in payload:
        raise ValueError(f"replay plan has no render_lights: {path}")
    render_lights = torch.as_tensor(payload["render_lights"], dtype=torch.float32)
    expected_shape = (expected_steps, args.n_lights, 3)
    if tuple(render_lights.shape) != expected_shape:
        raise ValueError(
            f"replay plan render_lights shape {tuple(render_lights.shape)} != {expected_shape}"
        )
    if torch.any(arrays["material_indices"] < 0) or torch.any(
        arrays["material_indices"] >= len(train_items)
    ):
        raise ValueError("replay plan material index is outside the train split")
    if torch.any(arrays["rotations"] < 0) or torch.any(arrays["rotations"] > 3):
        raise ValueError("replay plan rotation must be in [0, 3]")
    arrays["render_lights"] = render_lights
    print(
        f"train plan: {path} ({expected_steps} steps, {expected_steps * args.batch} slots)"
    )
    return arrays


# ----------------------------------------------------------------------------
# Novel-light render loss
# ----------------------------------------------------------------------------
def sample_light_positions(k, device, xy=1.2, zmin=2.0, zmax=8.0):
    """k random point-light positions above the surface (surface spans [-1,1])."""
    pxy = (torch.rand(k, 2, device=device) * 2 - 1) * xy
    pz = torch.rand(k, 1, device=device) * (zmax - zmin) + zmin
    return torch.cat([pxy, pz], dim=1)                       # [k,3]


def render_colocated(model, svbrdf, light_pos):
    """Render svbrdf under one colocated light (view==light) at light_pos [3].
    Mirrors NFPLightModel.render_input_images. Returns [B,3,256,256] (HDR)."""
    pos = light_pos.view(1, 3)
    ld, vd, ldis, _ = model.renderer.torch_generate(pos, pos, pos=model.surface)
    img = model.renderer._render(
        svbrdf, ld.unsqueeze(1), vd.unsqueeze(1), ldis.unsqueeze(1)).squeeze(1)
    return img


def render_weight(step, steps, render_w, warmup_frac):
    """Render-loss weight schedule: 0 for the first `warmup_frac` of training, then
    LINEAR ramp 0 -> render_w over the remainder. Baseline choice -- the net first
    fits the maps under direct supervision, then the novel-light consistency term is
    introduced gradually (common in render-loss papers); the exact schedule is an
    experiment knob, this is just the starting point."""
    if warmup_frac >= 1.0:
        return 0.0
    prog = step / max(1, steps)
    ramp = (prog - warmup_frac) / (1.0 - warmup_frac)
    return render_w * float(min(1.0, max(0.0, ramp)))


def render_loss(model, pred, gt, n_lights, light_positions=None):
    """L1 on raw (linear HDR) renders -- no tonemap. Highlights are NOT compressed,
    so the specular lobe dominates the term; that is the intended emphasis here."""
    if n_lights < 1:
        raise ValueError('n_lights must be positive')
    if light_positions is None:
        lights = sample_light_positions(n_lights, pred.device)
    else:
        if tuple(light_positions.shape) != (n_lights, 3):
            raise ValueError(
                f"light_positions shape {tuple(light_positions.shape)} != ({n_lights}, 3)"
            )
        lights = light_positions.to(pred.device, non_blocking=True)
    if not torch.isfinite(lights).all() or (lights[:, 2] <= 0).any():
        raise ValueError('render lights must be finite and above the surface')
    loss = 0.0
    for i in range(n_lights):
        rp = render_colocated(model, pred, lights[i])
        rg = render_colocated(model, gt, lights[i])
        loss = loss + F.l1_loss(rp, rg)
    return loss / n_lights


# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, items, n_lights=2):
    """Per-category map-L1 (stratified; Metal reported separately by print_eval)
    plus mean render loss on val."""
    model.net_g.eval()
    if not items or n_lights < 1:
        raise ValueError('evaluation needs nonempty materials and positive light count')
    generator = torch.Generator().manual_seed(0)
    lights = torch.cat((torch.rand(n_lights, 2, generator=generator) * 2.4 - 1.2,
                        torch.rand(n_lights, 1, generator=generator) * 6 + 2), dim=1)
    ds = MatSynthCropsDataset(items, train=False)
    per_cat = defaultdict(lambda: defaultdict(float))
    per_cat_n = defaultdict(int)
    rloss = 0.0
    for i in range(len(ds)):
        b = ds[i]
        sv = b["svbrdf"].unsqueeze(0).cuda()
        x = build_net_input(model, sv)
        pred = model.net_g(x)
        _, parts = map_loss(pred, sv)
        for k, v in parts.items():
            per_cat[b["cat"]][k] += v.item()
        per_cat_n[b["cat"]] += 1
        rloss += render_loss(model, pred, sv, n_lights, lights).item()
    out = {}
    for c, d in per_cat.items():
        out[c] = {k: v / per_cat_n[c] for k, v in d.items()}
        out[c]["mean"] = sum(out[c][k] for k in
                             ("normal", "diffuse", "roughness", "specular")) / 4
    return out, rloss / max(1, len(ds))


def train(model, train_items, val_items, args):
    params = list(model.net_g.parameters())
    n = sum(p.numel() for p in params)
    print(f"from-scratch trainable params: {n:,}")

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    # Historical experiment default, not the paper schedule. NFPLight section 4.1
    # uses Adam and cosine decay in both stages; train_fabric.py implements it.
    # --lr_decay cosine opts in to a DELAYED cosine squeeze: lr is held
    # constant through the render-warmup phase (map-only, decay_start = render_warmup
    # * steps) so the map converges at full lr exactly as the constant-lr r30 run did,
    # then CosineAnnealingLR anneals lr -> lr*0.01 over the render fine-tuning phase.
    # This isolates "does decay help the render squeeze" without starving the map
    # phase. NOT paper-faithful, so it stays behind the flag.
    decay_start = int(args.render_warmup * args.steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.steps - decay_start), eta_min=args.lr * 0.01) \
        if args.lr_decay == "cosine" else None

    # Resume from a full-state checkpoint (weights+opt+sched+step+early-stop+RNG) so a
    # post-prefix sweep can skip the shared 0..decay_start map phase instead of retraining
    # it. Restore BEFORE building the sampler so `remaining` sizes the loader correctly.
    start_step, best, bad = 0, float("inf"), 0
    if args.resume:
        start_step, best, bad = load_full(args.resume, model.net_g, opt, sched)
        print(f"resumed from {args.resume}: step {start_step}, best {best:.4f}, bad {bad}")
    if args.disable_early_stop:
        print("early stopping: disabled (strict replay contract)")

    if start_step >= args.steps:
        raise ValueError('resume step must be less than target steps')
    remaining = args.steps - start_step
    train_plan = None
    if args.train_plan:
        train_plan = load_train_plan(args.train_plan, args, start_step, train_items)
        ds = PlannedMatSynthCropsDataset(train_items, train_plan)
        sampler = FixedIndexSampler(torch.arange(len(ds)), len(ds))
        print("sampler: fixed sequential slots from immutable train plan")
    else:
        ds = MatSynthCropsDataset(train_items, train=True)
        weights = balanced_weights(train_items, fabric_tilt=1.0)      # no tilt: broad coverage
        # A replay branch can use a dedicated CPU generator initialized from the
        # restored torch RNG state.  With workers=0 this produces the same sampler
        # stream as the historical global-generator path, while making the stream
        # independent of unrelated future torch.random calls in the process.
        sampler_generator = None
        sampler_indices = getattr(args, "sampler_indices", "")
        if sampler_indices:
            replay_indices = load_sampler_indices(sampler_indices)
            expected_indices = remaining * args.batch
            if len(replay_indices) != expected_indices:
                raise ValueError(
                    f"sampler index stream length {len(replay_indices)} does not match "
                    f"remaining*batch={expected_indices}"
                )
            sampler = FixedIndexSampler(replay_indices, len(train_items))
            print(f"sampler: fixed replay indices from {sampler_indices}")
        elif getattr(args, "controlled_sampler", False):
            sampler_generator = torch.Generator(device="cpu")
            sampler_generator.set_state(torch.get_rng_state())
            print("sampler: controlled CPU generator from restored torch RNG")
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=remaining * args.batch, replacement=True,
                generator=sampler_generator)
        else:
            print("sampler: legacy global CPU generator")
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=remaining * args.batch, replacement=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if val_items and start_step == 0:
        ev, rl = evaluate(model, val_items)
        print_eval("baseline (random init, val)", ev)
        print(f"  val render-loss {rl:.4f}")

    model.net_g.train()
    step = start_step
    stopped = False
    for batch in loader:
        sv = batch["svbrdf"].cuda(non_blocking=True)
        x = build_net_input(model, sv)
        pred = model.net_g(x)
        ml, parts = map_loss(pred, sv, args.loss_w)
        rw = render_weight(step, args.steps, args.render_w, args.render_warmup)
        plan_lights = None
        if train_plan:
            plan_step = step - start_step
            if plan_step < 0 or plan_step >= len(train_plan["render_lights"]):
                raise RuntimeError(f"train plan has no render lights for step {step}")
            plan_lights = train_plan["render_lights"][plan_step]
        rl = render_loss(model, pred, sv, args.n_lights, plan_lights) if rw > 0 else \
            torch.zeros((), device=sv.device)
        loss = ml + rw * rl
        if not torch.isfinite(loss):
            raise FloatingPointError(f'non-finite loss at step {step}')
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float('inf'), error_if_nonfinite=True)
        opt.step()
        step += 1
        if sched is not None and step >= decay_start:
            sched.step()                       # cosine only in the render fine-tune phase
        if step % args.log_every == 0:
            cur_lr = opt.param_groups[0]["lr"]
            print(f"step {step:>6}/{args.steps}  loss {loss.item():.4f} "
                  f"(map {ml.item():.4f} rend {float(rl):.4f} rw {rw:.3f})  "
                  f"lr {cur_lr:.2e}  "
                  + " ".join(f"{k[:1]}={v.item():.3f}" for k, v in parts.items()))
        if val_items and args.val_every and step % args.val_every == 0:
            ev, vrl = evaluate(model, val_items)
            print_eval(f"step {step} (val)", ev)
            print(f"  val render-loss {vrl:.4f}   "
                  f"Metal specular {ev.get('Metal', {}).get('specular', float('nan')):.4f}")
            score = np.mean([ev[c]["mean"] for c in ev])          # macro over categories
            if score < best - args.min_delta:
                best = score
                bad = 0
                save(model.net_g, args.out)
                print(f"  saved (macro map-L1 {score:.4f}) -> {args.out}")
            else:
                bad += 1
                print(f"  no improve {bad}/{args.patience} (best {best:.4f})")
                if bad >= args.patience and not args.disable_early_stop:
                    print(f"  EARLY STOP at step {step} (no improvement in "
                          f"{args.patience} vals)")
                    stopped = True
            model.net_g.train()
        if args.full_every and step % args.full_every == 0:
            rp = resume_path(args, step)
            save_full(rp, model.net_g, opt, sched, step, best, bad)
            print(f"  [resumable] full state (opt+sched+rng) -> {rp}")
        if stopped:
            break
    if not (val_items and args.val_every):
        save(model.net_g, args.out)
        print(f"saved -> {args.out}")
    if stopped:
        print(f"stopped early; best macro map-L1 {best:.4f} at {args.out}")


def save(net, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"params": net.state_dict(), "feature_version": "sample_v1"}, path)


def resume_path(args, step):
    """checkpoints/resume/<out-stem>_step<N>.pth for the full-state snapshots."""
    stem = os.path.splitext(os.path.basename(args.out))[0]
    return os.path.join(args.resume_dir, f"{stem}_step{step}.pth")


def save_full(path, net, opt, sched, step, best, bad):
    """Resumable snapshot: weights + optimizer(AdamW moments) + scheduler + step +
    early-stop counters + all RNG states. The "params" key is the same one save()
    writes, so a full snapshot is ALSO directly loadable by NFPLightModel as a plain
    model checkpoint. ponytail: the loader's per-batch position is NOT restored (fresh
    data stream on resume) -- fine for branching post-prefix sweeps, which want new
    post-decay_start trajectories anyway; the learned state (weights+opt) is what must
    carry over. Upgrade to a resumable sampler only if bit-identical continuation ever
    matters."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "params": net.state_dict(),
        "feature_version": "sample_v1",
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "step": step, "best": best, "bad": bad,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
    }, path)


def load_full(path, net, opt, sched):
    """Restore a save_full() snapshot in place; returns (step, best, bad). Load to CPU
    (RNG ByteTensors must stay on CPU for set_rng_state), then move the optimizer moment
    tensors onto the param device -- Optimizer.load_state_dict does NOT do that itself."""
    ck = torch.load(path, map_location="cpu", weights_only=False)   # trusted self-produced
    net.load_state_dict(ck["params"])
    opt.load_state_dict(ck["opt"])
    dev = next(net.parameters()).device
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    if sched is not None and ck.get("sched") is not None:
        sched.load_state_dict(ck["sched"])
    torch.set_rng_state(ck["torch_rng"])
    if ck.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ck["cuda_rng"])
    np.random.set_state(ck["np_rng"])
    random.setstate(ck["py_rng"])
    return ck["step"], ck["best"], ck["bad"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="D:/MatSynth_crops256")
    ap.add_argument("--out", default="checkpoints/net_g_matsynth.pth")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=4,
                    help="training batch size; 4 is the RTX/OOM-safe default")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_decay", choices=["none", "cosine"], default="none",
                    help="historical constant/delayed-cosine experiment; see train_fabric.py for stage-wise training")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--render_w", type=float, default=0.5)
    ap.add_argument("--render_warmup", type=float, default=0.5,
                    help="fraction of steps with render_w=0 before linear ramp to render_w")
    ap.add_argument("--n_lights", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=15,
                    help="early stop after this many vals with no macro map-L1 improvement")
    ap.add_argument("--min_delta", type=float, default=1e-4,
                    help="min macro map-L1 drop to count as improvement")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--full_every", type=int, default=10000,
                    help="save a resumable full-state snapshot every N steps (0=off)")
    ap.add_argument("--resume_dir", default="checkpoints/resume",
                    help="where full-state snapshots go")
    ap.add_argument("--resume", default="",
                    help="path to a save_full() snapshot to resume from (skips its prefix)")
    ap.add_argument("--controlled_sampler", action="store_true",
                    help="use a dedicated sampler generator initialized from the restored torch RNG")
    ap.add_argument("--sampler_indices", default="",
                    help="path to a torch.save() explicit post-resume sampler index stream")
    ap.add_argument("--train_plan", default="",
                    help="path to an immutable material/augmentation/render-light replay plan")
    ap.add_argument("--disable_early_stop", action="store_true",
                    help="run through --steps even when validation patience is exceeded")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_per_cat", type=int, default=0, help="cap materials/cat (0=all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.loss_w = (1.0, 1.0, 1.0, 1.0)
    # ponytail: input-degradation augmentation (colored flash/vignette/LDR) is the
    # syn-real gap lever discussed in the plan but NOT wired here yet -- add as an
    # aug hook on build_net_input inputs when repeat-capture consistency needs it.

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    model = ScratchModel(SimpleNamespace(save_root="./_scratch_tmp"))

    all_train = scan_crops(args.crops, "train")
    all_test = scan_crops(args.crops, "test")
    if args.max_per_cat:
        all_train = cap_per_cat(all_train, args.max_per_cat, args.seed)
    cats = defaultdict(int)
    for c, _ in all_train:
        cats[c] += 1
    print(f"crops256: {len(all_train)} train / {len(all_test)} test materials")
    print("  per category:", dict(sorted(cats.items(), key=lambda x: -x[1])))
    assert all_train, f"no crops under {args.crops}/train"

    tr, va = split_by_material(all_train, args.val_ratio, args.seed)
    print(f"split: {len(tr)} train / {len(va)} val materials")

    train(model, tr, va, args)

    if all_test:
        checkpoint = torch.load(args.out, map_location='cpu', weights_only=False)
        model.net_g.load_state_dict(checkpoint['params'], strict=True)
        ev, rl = evaluate(model, all_test)
        print_eval("FINAL test (official split)", ev)
        print(f"  test render-loss {rl:.4f}")


if __name__ == "__main__":
    main()
