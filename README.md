# NFPLight
This is the source code of research paper "NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point lighting" (SIGGRAPH Asia 2024).
**More information (include our paper, supplementary, video) can be found at** [My Personal Page](https://cgliwang.github.io/) 
![Alt](Teaser.jpg)

# Pretrained models
Our pretrained models can be downloaded from [here](https://drive.google.com/drive/folders/171Krs3DUGqI-IkejbtOsbPtrKpOlds2p?usp=sharing). Unzip these files to 'checkpoints' folder.

# Dependencies
The maintained portability path is Python 3.10 or 3.11. Install a PyTorch
wheel that matches the machine first, then install the small runtime dependency
set in `requirements.txt`:

```powershell
git clone https://github.com/n01r1r/NFPLight_code.git
cd NFPLight_code
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CPU-only example:
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
# CUDA example (choose the supported pair for your driver instead):
# python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The current maintainer smoke environment is Python 3.11.0rc2 with
`torch==2.5.1+cu121` and CUDA 12.1. The repository requirements intentionally
do not pin NVIDIA runtime packages or Triton, so the PyTorch CPU/CUDA choice is
made by the wheel command above. Fabric/MatSynth21 inference is supported with
`--device cpu`; `--device auto` (the default) uses CUDA when available. The
historical legacy33 denoiser path retains its CUDA-only behavior.

On Linux or macOS, activate the environment with `source .venv/bin/activate`.
Fabric training requires CUDA; CPU support here is for 21-channel inference.
Optional utility functions `utils.img_util.color_map` and `utils.options.parse`
require `matplotlib` and `PyYAML`, respectively; the documented training and
inference commands do not use those utilities.

The release was checked in a clean checkout and a new Windows CPU environment
with PyTorch 2.5.1, torchvision 0.20.1, NumPy 2.4.6, OpenCV 4.14.0.94, and
Pillow 12.3.0. The 43 contract tests and estimator smoke passed. Existing XYZ
and `phi_theta` checkpoint files both produced finite float32 maps from a
synthetic input fixture, without training caches or denoiser weights. This
checks execution and output contracts on one host, not reconstruction quality
or execution on another physical machine. Run the self-contained checks with:

```powershell
python -m unittest test_fabric_training test_fabric_raw_core test_raw_capture test_normal_losses test_normal_heads -v
python test_real_estimator.py
```

This fork adds the Fabric-only training and portable 21-channel inference
path. It does not include checkpoints or capture data. A checkpoint and its
input captures remain separate inputs to every inference run.

# Usage
- Test on synthetic data

This script tests the method on synthetic data under ideal capture settings, 
aiming to evaluate its theoretical upper-bound performance.
```Python
python test.py --save_root ./results/syn --test_data_root ./input_data/syn_data --loadpath_network_g ./checkpoints/net_g_syn.pth
```

`test.py` keeps the original synthetic XYZ path and accepts legacy or
`sample_v1` checkpoints. It rejects Fabric `raw_calibrated_v2` or fixed
`phi_theta` checkpoints before loading the model; use the `real.py`
MatSynth21 command below for those checkpoints and their linear near/far
capture contract.

- Test on real data

This script incorporates a denoising network for real material capture.
The model is trained at a resolution of 256, but the proposed method supports inference at 1024
to produce higher-resolution results.
```Python
python real.py --save_root ./results/real --test_data_root ./input_data/real_data --loadpath_network_g ./checkpoints/net_g_real.pth --loadpath_network_denoise ./checkpoints/net_denoise_real.pth --image_size 1024
```

For an older MatSynth-trained 21-channel estimator, use pre-aligned real-capture
near/far folders at 256px. Encoded PNG pairs use the explicit gamma-2.2 approximation;
registration and exposure alignment remain caller responsibilities. This route
intentionally skips DenoiseNet and keeps the synthetic feature geometry
(`near=2.414`, `far=10`) for input compatibility:
```powershell
python real.py --save_root ./results/real_matsynth21 --test_data_root ./input_data/real_data --loadpath_network_g ./checkpoints/net_g_matsynth_r30d.pth --estimator-family matsynth21 --image_size 256
```
With `--estimator-family auto` (the default), `real.py` selects 21-channel
versus legacy 33-channel inference from `intro_albedo.weight` in the estimator
checkpoint.

For a new RAW Fabric checkpoint, each capture directory contains `near.npy`,
`far.npy` (float32 HWC RGB in [0,1]) and `metadata.json`. Alternatively use linear
uint16 `near.png`/`far.png`, already mapped to the full [0,65535] reference range.
Do not provide both file formats for the same shot. Arrays must already be
black/white corrected, demosaiced, converted with fixed WB and a camera matrix
to linear sRGB primaries, and spatially aligned. The loader does not decode Bayer
RAW files or infer camera calibration. Minimal metadata:

```json
{
  "working_space": "linear_srgb",
  "near_distance": 2.414,
  "far_distance": 10.0,
  "reference_exposure": "equal",
  "reference_flux": "equal",
  "reference_scale": 1.0,
  "preprocessing_done_externally": true
}
```

Distances are in patch-half-width units, not millimeters. Metadata declares
conditions that must actually be calibrated; it does not prove that calibration.
No extra gamma or per-image brightness normalization is applied. Any resize occurs
in linear space. Float32 NPZ/JSON maps are authoritative; the PNG is a preview.

```powershell
python real.py --device auto --save_root results/raw_fabric --test_data_root input_data/raw_pairs --loadpath_network_g checkpoints/fabric_raw_full_20260912/best_render.pth --estimator-family matsynth21 --input-encoding linear_rgb --image_size 256
```

The default `--input-encoding auto` selects linear RGB for `raw_calibrated_v2`
checkpoints and the legacy gamma22 PNG route for older checkpoints. RAW checkpoints
reject `--input-encoding gamma22_png`.

The checkpoint payload is the inference contract. New Fabric checkpoints carry
both `feature_version` and `contract.normal_objective.head` (also mirrored in
`contract.settings.normal_head`); the two values must agree. The fixed
`phi_theta` decoder is applied during inference exactly as during training and
produces the same `[B,10,256,256]` output shape. A legacy checkpoint without
these metadata fields uses the established `xyz` meaning. Passing only a bare
`params` mapping discards this metadata and therefore selects legacy `xyz`; use
the original checkpoint file when the selected head or RAW feature version
matters. Conflicting or malformed metadata is rejected.

For a portable CPU smoke run with an existing checkpoint and separately
prepared captures:

```powershell
python real.py --device cpu --save_root results/raw_fabric_cpu --test_data_root C:/data/raw_pairs --loadpath_network_g C:/weights/fabric_checkpoint.pth --estimator-family matsynth21 --input-encoding auto --image_size 256
```

The run writes authoritative float32 `*.npz` maps and a matching `*.json`
provenance document. `metadata.json`, linear `[B,6,256,256]` near/far inputs,
the calibrated distances (2.414 and 10), equal exposure/flux, and external
linear-sRGB preprocessing remain part of the capture contract.

## Fabric-only from-scratch training (without DenoiseNet)

The current Fabric pipeline accepts a user-provided MatSynth preprocessed root, excludes the
`deschaintre_2020` source, preserves the official Fabric test split, and creates
native 256px crops of the original normal/diffuse/roughness/specular maps.
The immutable manifest records source and cache hashes, grouping, crop coordinates,
opacity checks, and every rejected material. Large mmap crop files stay local.
The prepared v1 split has **338 train / 38 validation / 5 official test** materials,
with 16 crops each. It excludes 73 materials from the source above and 9 transparent
materials. Declared opacity maps require every pixel to be at least 254/255;
undeclared exported placeholders allow a fraction of at most 1e-6 below that value.

```powershell
python -m data.fabric --root C:/data/MatSynth_preprocessed --out C:/data/fabric_native256_v1  # one-time preparation; an identical existing cache is verified
python -m unittest test_fabric_training -v
python train_fabric.py --manifest C:/data/fabric_native256_v1/manifest.json --out checkpoints/fabric_smoke --smoke
python train_fabric.py --manifest C:/data/fabric_native256_v1/manifest.json --out checkpoints/fabric_scratch
```

The full-run default is 100K map steps followed by 25K map/render steps, Adam with
cosine decay per stage, microbatch 2 with accumulation 2, FP32, and clean linear
near/far inputs clipped to [0,1]. Observation gamma and quantization are disabled.
All augmentation is disabled: the sampler selects only the fixed cached tiles,
without rotations, flips, new crops, resize jitter, color, exposure or noise changes.
Rendering supervision uses fixed light probes. This is an estimator-only adaptation of the paper, whose full
pipeline used 400K/100K steps. DenoiseNet and pretrained estimator weights are
never loaded. `train_scratch.py` remains the older all-category experiment runner;
use `train_fabric.py` for the fixed Fabric protocol.

Resumable `last.step-*.pt` checkpoints are committed every 1000 optimizer steps by
default. Validation points, stage transitions, normal completion, and `--stop-after`
always commit. Use `--checkpoint-every 1` only when reproducing the older per-step
checkpoint behavior. Selected `best_*.pth` files omit optimizer state but retain
the parameter mapping, feature version, and normal-head contract; training
outputs are local generated artifacts and are excluded from the published source tree.

`--normal-loss cosine` replaces only the normal training term with
`mean(1 - dot(unit_pred, unit_target))`. `--normal-loss phi_theta` instead compares
theta from +z and the wrapped phi=atan2(y,x), using `(abs(delta_theta) +
abs(wrapped_delta_phi))/pi`. At either pole (unit xy norm below 1e-6), only phi
is omitted. `--normal-loss geodesic` first converts each unit XYZ normal to
the spherical coordinates `(theta, phi)` on the +z hemisphere convention and
then evaluates the normalized shortest-arc spherical distance, using the
spherical law-of-cosines equivalent with safe endpoint gradients. It is not
the coordinate-wise phi/theta L1 objective and does not change the XYZ head:
the conversion is loss-side only. The default remains `l1` pending comparison.
The selected follow-up candidate keeps the existing XYZ head and uses the
loss-side spherical coordinate objective explicitly as
`--normal-head xyz --normal-loss phi_theta`; this is not the fixed output-head
decoder described below. The default remains `l1` for baseline reproducibility.
`--normal-head phi_theta` is a separate output-head variant: it deterministically
maps the raw normal channels to `theta=pi*(raw_z+1)/4` and
`phi=atan2(raw_y,raw_x)`, then reconstructs a unit normal on the +z hemisphere.
It adds no learnable parameters and preserves the `[B,10,H,W]` output contract;
the default head remains `xyz`. Head and loss choices are recorded in the
checkpoint contract and must be matched for an exact resume.
The shared decoder and trainer source hashes are part of that resume contract;
the portability changes therefore make an older training run fail its exact
resume check until it is reproduced with the matching source archive. This
does not alter inference of an existing checkpoint whose metadata is intact.
The other maps, RAW input, no-augmentation policy and L1 selection metrics are
unchanged; degrees, cosine, phi-theta and geodesic diagnostics are recorded
separately. Objective/source changes are incompatible with exact optimizer resume.
Resume with the same manifest and experiment flags:

```powershell
python train_fabric.py --manifest C:/data/fabric_native256_v1/manifest.json --out checkpoints/fabric_scratch --resume checkpoints/fabric_scratch/last.pt
```

`--stop-after N` pauses after N optimizer steps and saves the complete state.
`--smoke` uses two materials per split and four steps per stage; its metrics are
implementation checks, not benchmark results. Validation uses fixed multiple crops
and a separate light RNG. Test reloads the selected `best_render.pth`, which retains
the stage-A model if no stage-B candidate meets the map-regression guardrail.
`train.jsonl`, `validation.json`, `test.json`, and `runtime.json` record actual work.

New checkpoints declare `raw_calibrated_v2`: per-image relation normalization and
the calibrated equal-flux/exposure scale `(10/2.414)^2`, independent of the center
pixel's brightness. The official coefficient approximation is retained explicitly.
Older `sample_v1` and unversioned `legacy_batch_v0` checkpoints retain their feature
semantics. MatSynth diffuse/specular GT still use `gamma22_compat` decoding; normal
and roughness have no gamma conversion. This GT transfer is independent of RAW
observation encoding.

The source cache and material split remain immutable. Target policy v2 excludes
native nonpositive-normal-z pixels from all losses and metrics and marks them in
the existing feature mask. Effective roughness is floored at the renderer's .05.
Training records preclip gradient norms and clips the actual norm to 1.0.
The manifest and run logs are the source of truth for a particular local
training run. This repository does not publish checkpoints, caches, private
captures, result reports, or benchmark claims.

# Citation
If you use our code or pretrained models, please cite as following:
```
@article{10.1145/3687978,
author = {Wang, Li and Zhang, Lianghao and Gao, Fangzhou and Kang, Yuzhen and Zhang, Jiawan},
title = {NFPLight:  Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting},
year = {2024},
issue_date = {December 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3687978},
doi = {10.1145/3687978},
abstract = {Recovering spatial-varying bi-directional reflectance distribution function (SVBRDF) from a few hand-held captured images has been a challenging task in computer graphics. Benefiting from the learned priors from data, single-image methods can obtain plausible SVBRDF estimation results. However, the extremely limited appearance information in a single image does not suffice for high-quality SVBRDF reconstruction. Although increasing the number of inputs can improve the reconstruction quality, it also affects the efficiency of real data capture and adds significant computational burdens. Therefore, the key challenge is to minimize the required number of inputs, while keeping high-quality results. To address this, we propose maximizing the effective information in each input through a novel co-located capture strategy that combines near-field and far-field point lighting. To further enhance effectiveness, we theoretically investigate the inherent relation between two images. The extracted relation is strongly correlated with the slope of specular reflectance, substantially enhancing the precision of roughness map estimation. Additionally, we designed the registration and denoising modules to meet the practical requirements of hand-held capture. Quantitative assessments and qualitative analysis have demonstrated that our method achieves superior SVBRDF estimations compared to previous approaches. All source codes will be publicly released.},
journal = {ACM Trans. Graph.},
month = nov,
articleno = {274},
numpages = {11},
keywords = {material reflectance modeling, SVBRDF, deep learning, rendering}
}
```
