# NFPLight

Source for *NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting*, SIGGRAPH Asia 2024. This fork retains the checkpoint-compatible estimator and Fabric training core.

## Fabric DNG comparison

The current capture contract is [docs/CAPTURE_REQUIREMENTS.md](docs/CAPTURE_REQUIREMENTS.md), with the active author-real pipeline detailed in [docs/DNG_AUTHOR_REAL_PIPELINE_20261001.md](docs/DNG_AUTHOR_REAL_PIPELINE_20261001.md). The maintained entry point processes five supplied groups independently, in FP32, using `checkpoints/net_g_real.pth`:

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepare-only --out artifacts/fabric_capture_20261001_author_original
```

Preparation writes per-frame numeric archives, marker/crop overlays, aligned crop sheets, marker residuals and display-only frame00-versus-FP32-mean candidates. Inspect each `prepare_review.html` and its images. Averaging is eligible only when each side's five frames meet all marker RMS ≤1 px in final 256 coordinates, full valid crop support and compatible raw/photometric metadata. Texture registration still requires visual acceptance. Before inference, record the review in `root_review.json` for every group (`status: reviewed`, `texture_registration_passed`, and a non-empty note). A failed numeric or visual gate selects that group's `near_00.dng` / `far_00.dng`; passing groups average aligned RGB counts in FP32.

Run inference only after all five review records are present:

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepared artifacts/fabric_capture_20261001_author_original
```

Each group report includes only the `both=(D-B)/(W-B)` compensation condition, the fixed 512 → crop46 → 420 → 256 geometry, the author's 33 input and 10 raw prediction channels, clipping, source/code hashes, environment and verification. The estimator uses the author's real weights without instantiating, loading, or running a denoiser; its historical copy feature slots duplicate the original RGB inputs. The root artifact directory has a combined `index.html`; each group has its own report and provenance manifest. Camera-RGB display copies apply the side's normalized WB and camera-to-linear-sRGB matrix once; already-whitebalanced RGB receives only the matrix, and linear-sRGB arrays are displayed directly. The report and preparation-review HTML have a checkbox that optionally selects separate `x^(1/2.2)` RGB preview PNGs; it defaults off and does not alter scalar/signed maps, numerical arrays or statistics. Bayer remains in NPZ/stats; its photographic selector uses the same-frame WB-demosaic RGB fallback and labels that distinction. Raw prediction arrays remain unchanged by display decoding. The preserved `best_render` outputs in `artifacts/fabric_capture_20261001_clean` remain archived; current reports exclude MatSynth/custom-weight comparison rows at the user's request.

To refresh semantic feature PNGs from existing arrays, then refresh only HTML without rerunning model inference, call the display and HTML-only helpers for each group directory:

```powershell
python -c "from pathlib import Path; from capture_processing.report import regenerate_feature_gallery_only; print(regenerate_feature_gallery_only(Path('artifacts/fabric_capture_20261001_author_original/260930_152732_008')))"
python -c "from pathlib import Path; from capture_processing.report import regenerate_capture_html_only; print(regenerate_capture_html_only(Path('artifacts/fabric_capture_20261001_author_original/260930_152732_008'), include_preserved_checkpoint=False))"
```

This reads existing NPZ arrays only for display previews, preserves numeric archives and execution manifest, and writes separate generator/asset SHA-256 records. Historical `checkpoint_comparison_display.json` files remain archived; current author reports omit the custom-weight comparison rows.

When only report HTML needs a text/table refresh, use the HTML-only helper. It reads the saved manifest and array statistics, does not open NPZ files, leaves PNGs and the execution manifest unchanged, and writes `report_html_generation.json` with the HTML and generator SHA-256 values:

```powershell
python -c "from pathlib import Path; from capture_processing.report import regenerate_capture_html_only; print(regenerate_capture_html_only(Path('artifacts/fabric_capture_20261001_author_original/260930_152732_008'), include_preserved_checkpoint=False))"
```

For a preparation review page, `regenerate_prepare_html_only(group_dir)` updates the WB-diagonal HTML table and only the `prepare_review.html` entry in `preparation.json.review_assets.sha256`; the existing PNGs and numeric archives remain unchanged.

Every input frame is processed afresh from Bayer uint16 through FP32 AHD, marker geometry, spatial interpolation, black/white comparison, camera WB, the embedded linear-sRGB matrix, feature construction and checkpoint inference. The original DNGs and checkpoint remain unchanged. The older `best_render` report is preserved for comparison and is not rerun.

The old PNG, FP32 sensor ablation, compensation and one-off report scripts have been removed. See [cleanup record](docs/cleanup_removed.json). Do not regenerate input PNGs in the original DNG folder.

## Environment

Python 3.10/3.11 and a matching PyTorch wheel are required. Install PyTorch for the desired CPU/CUDA environment, then:

```powershell
python -m pip install -r requirements.txt
```

Actual DNG unpacking requires rawpy. FP32 inference disables AMP, autocast and TF32. CUDA is optional for the 33-channel author-real estimator; runtime and memory requirements depend on the selected device.

## Fabric training

`train_fabric.py` is the maintained Fabric training entry point. Dataset preparation is available through `python -m data.fabric --help`. The estimator, feature construction, renderer, D4 augmentation and linear prediction export remain checkpoint-compatible. Old LoRA, all-category training and PNG inference entry points have been removed.

```powershell
python train_fabric.py --manifest data/fabric_native256_v1/manifest.json --out checkpoints/fabric_scratch
```

The source contract changed during cleanup. Exact optimizer resume from checkpoints recording the previous sources is rejected; no migration whitelist was expanded. Inference using the selected unchanged checkpoint remains supported.

Tests are kept in `tests/`: capture processing, AHD, capture checkpoint loading, Fabric training/resume, linear map export, and normal heads/losses. Run them together from the repository root:

```powershell
python -m pip install pytest
python -m pytest tests -q -p no:cacheprovider
```

The three earlier training-design and red-team documents are historical records in [docs/archive/](docs/archive/); the active contract remains [docs/CAPTURE_REQUIREMENTS.md](docs/CAPTURE_REQUIREMENTS.md). Root documentation is limited to this README and AGENTS.md.

Original capture/training data was removed by the user and will be supplied separately. The preserved report documents its recorded run; executing new capture or training runs requires the corresponding data.

Capture data, checkpoints, training caches and experiment outputs are local inputs/artifacts. `.git`, `checkpoints`, `results`, and `data/fabric_native256_v1` may be junctions to an external disk: never recursively clean their targets.

## Citation

```bibtex
@article{10.1145/3687978,
  author = {Wang, Li and Zhang, Lianghao and Gao, Fangzhou and Kang, Yuzhen and Zhang, Jiawan},
  title = {NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting},
  journal = {ACM Transactions on Graphics},
  volume = {43}, number = {6}, articleno = {274}, year = {2024},
  doi = {10.1145/3687978}
}
```

## Capture reports

Use each capture's `report.html` and the combined `index.html`. The separate
`paper_comparison` report and `--paper-equations` runner option were retired at
the user's request on 2026-10-02. Original author inference arrays and historical
best_render results remain preserved. Numerical equation helpers remain only for
existing diagnostics; they do not generate a separate comparison report.
