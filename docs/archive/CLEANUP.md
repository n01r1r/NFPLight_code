# Cleanup and rebuild

The user confirmed retaining the checkpoint-compatible model, Fabric training core and meaningful tests. The old DNG/PNG conversion, FP32 ablation, compensation, denoiser experiments, one-off reports and launch/watch scripts were removed from the active source tree. The 44 root files are listed with recovery paths and SHA256 in `cleanup_removed.json`; nine denoiser source files were verified in `denoise_ext.zip` before their directory was removed.

Recovery directory:
`C:/Users/hdy/.codex/visualizations/2026/09/30/01a0f214-4a44-7b61-aa0c-2a78870df452/cleanup_recovery`

`pre_cleanup_sources.zip` contains 95 verified original source/data files, including the two DNGs and immutable capture metadata. `best_render.pth` is a verified copy of the selected checkpoint. The pre-cleanup diff and Git status are saved separately. `.work`, `.fork-work` and `plots` were moved outside the active tree, preserving roughly 6.8 GB of historical artifacts. The derived DNG outputs were subsequently deleted; recovery retains original inputs and source copies, never reusable experiment evidence.

The external junction targets `.git`, `checkpoints`, `results`, and `data/fabric_native256_v1` were not cleaned. Existing core modifications were preserved; no hard reset, commit or checkpoint rewrite was performed. Registered worktree/agent state was not removed.

Current FP32 implementation verification: 98 capture and retained-core tests passed. The 15/45cm numerical experiment and interrupted 10/30cm FP64 outputs were deleted following the user precision change; nine older DNG experiment output directories plus capture-processed/replay and fixture-inference results were also removed. Only the new 10/30cm captures may appear in the current report; the current run uses only the latest near_00/far_00 pair without averaging, as requested after post-marker residuals persisted. Previous averaged FP32 outputs were removed.

Current decisions: explicit FP32 only selection, default/current FP32; fresh marker detection from a display-only copy of the new AHD result; shared per-side geometry with the single active `both=(D-B)/(W-B)` compensation condition. See `CAPTURE_REQUIREMENTS.md`. Unresolved choices are handled through `grill-me`, one question at a time.

## Minimal credible reset — 2026-10-01

Target: `C:/workspace/NFPLight_code`, resolved from both referenced capture chats. The paper-template repository is a separate project and was not reset.

Remote comparison (live `git ls-remote`, 2026-10-01): origin/main = `cc79c9b98f06dccc0a78974e4b625c2f78ac5936`, identical to local HEAD; upstream/main = `cd16babaa79e79d8c02feaee03a7dd0f85118ae4`. No reset, commit, pull, or push. All existing local training/model changes remain present. The origin difference is the uncommitted training/resume/augmentation changes, simplification of utility I/O, FP32 capture additions, and removal of obsolete utilities. `rawpy` and `scipy` are required by capture unpacking and FP32 LAPACK respectively.

Latest human decisions override the initial FP64 specification: FP32 only, latest 10/30cm near_00/far_00 pair, no averaging, fresh markers and inner crop, camera WB plus linear-sRGB matrix, gamma off, only `both` black/white compensation, checkpoint relation/features unchanged.

Removed local leftovers: `input_data/capture_260813` (old generated near/far PNGs), `data/Our-Capture - 바로 가기.lnk`, `harness/contracts/example.contract.yaml` (unused scaffold referencing absent eval.py), and root/package Python and pytest caches. Registered `.claude/worktrees`, harness state, dataset/checkpoint/results/.git junctions were preserved.

Audit retention decisions: active FP32 report/arrays/previews implement the explicit intermediate/channel visualization request. Original DNGs/metadata, upstream sample data, source/hash recovery records, research notes and meaningful numerical/training tests remain useful. `finetune_fabric.py` supplies six helpers consumed by retained training and participates in checkpoint source hashes; its old standalone LoRA path was not removed because strict resume compatibility must not be silently relaxed. No source-hash migration whitelist was extended.

The old report remains evidence for its recorded source hashes only; this reset does not claim a newly executed DNG experiment or new model accuracy evidence.

Applied capture cleanup: removed the burst-average selector and reduction helpers, cross-frame-only compatibility helpers, legacy near.dng/far.dng fallback, and externally supplied geometry override. Each side now processes exactly one selected metadata-declared DNG and returns its arrays directly. FP32 AHD/photometry/feature tracing, LibRaw comparison, source/NPZ integrity checks and checkpoint semantics remain. `select_source_frames` defaults to 0 and rejects None; `run`/CLI no longer accept geometry overrides. Manifest v3 retains list-shaped source/hash/metadata records and `frame_average.enabled=false`; `frame_average.stage` is now `none`. Source archive names/indices reflect the selected frame index.

Ponytail review/audit findings applied:
- delete: retired average/legacy/geometry branches and their tests. One selected pair with fresh markers. [run_fabric_capture.py, test_capture_processing.py]
- delete: old capture PNGs, unused shortcut, non-runnable harness example, disposable caches. Nothing replaces them.
- shrink: per-side frame loop, sum/division and intersection boilerplate. Direct single-frame arrays.
- net: -185 Python lines, -0 dependencies (runner 798 ->635; capture tests 264 ->242).

Verification on 2026-10-01: 55 retained-core unittest cases, 23 AHD pytest cases, 18 capture unittest cases, REAL_ESTIMATOR_SMOKE_OK, CLI help and first-pair selector/signature smoke passed. 40 DNG files, four metadata files and the selected checkpoint retain their exact pre-reset SHA256 hashes. Eight actual saved spatial arrays are byte-for-byte equal between prior sum/divide-by-one and direct selection; an additional deterministic FP32 check covered 1,512,240 values with zero differing bits. Modified README/.gitignore pass git diff --check; unrelated pre-existing whitespace warnings remain in train_fabric.py/utils/render_util.py.

Full runner integration replay also passed using recorded unpack/spatial fixtures and actual unchanged checkpoint inference on CUDA: 12 shared source arrays matched exactly, 138 condition arrays matched the previous report within rtol=1e-5/atol=1e-6. The replay exercised all three conditions, feature/model consistency, NPZ round trips, source/checkpoint hashes and report generation. Temporary outputs were removed on exit. This is implementation regression evidence; DNG unpack/AHD/marker detection were fixture inputs, not a fresh physical-data experiment.

## Deeper source cleanup — 2026-10-01

This pass supersedes the conservative legacy-retention decision above, following the user's explicit "Remove all unnecessary codes, scripts." request. Luna max delegation reviewed active callers and staged essential helper extraction.

Deleted 11 files: finetune_fabric.py, train_scratch.py, real.py, test.py, test_aug_rot.py, test_crops_loader.py, test_raw_capture.py, data/dataset.py, data/matsynth_crops.py, model/nfplight_real_model.py, utils/img_util.py. The small ScratchModel subclass now lives in train_fabric.py; pack_svbrdf and its D4 normal rotation helper live in data/fabric.py.

Removed unused base-model inference/export wrappers, old capture dataset validation and metadata bookkeeping, TwoBranchRealNet/DenoiseNet, renderer samplers/PNG conversion/unused physical modes and LDR quantization. Retained save_svbrdf_maps because Fabric evaluation exports linear predictions through it. Retained TwoBranchNet, LayerNorm2d, 21-channel features, checkpoint heads, fixed renderer mathematics and D4 training checks. Current runtime entry points are train_fabric.py, run_fabric_capture.py and dataset preparation through python -m data.fabric. Removed torchvision and tqdm from requirements. Net Python reduction this pass: 3620 lines.

All changed/deleted sources were backed up with SHA256 guards and rollback before applying. Recovery stages: C:/Users/hdy/AppData/Local/Temp/nfplight_minimum_core_92ce8amw and C:/Users/hdy/AppData/Local/Temp/fabric_training_cleanup_9c7b6588ea63477da39c504aa4afc528. No Git reset, commit or push. Source contracts now include only current training files. Old exact optimizer resumes with different recorded source hashes remain rejected; migration allowlists were not expanded.

Validation: 66 unittest cases, 23 AHD pytest cases, CAPTURE_MODEL_SMOKE_OK and all three CLI help checks passed. Active imports resolve without removed modules. Fixed renderer regression matched the original default path bitwise on 6,144 values. Actual selected checkpoint inference on CUDA matched all three saved condition predictions exactly (maximum absolute difference 0), using recorded input arrays. This is inference regression evidence, not a fresh DNG experiment.

Data availability: the user confirmed deleting the original DNG directories and training-data junction, and will supply data separately. These absent inputs are not restored. No fresh DNG experiment or real-dataset training was run during this source cleanup. The unchanged selected checkpoint SHA256 remains 6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd; original-data hashes cannot be checked until data is supplied.

## Root documentation and test layout — 2026-10-01

Root Markdown is now limited to README.md (current usage) and AGENTS.md (agent instructions). Three historical research/design documents were moved to docs/archive/, retaining their recorded content, with a historical-status note and relocated links. Active capture and cleanup contracts remain under docs/.

All seven root test scripts were removed from the root and organized into six modules under tests/. Normal head and normal loss tests were combined into test_normals.py with deduplicated imports and unchanged test bodies. The old independently invoked estimator smoke is now three pytest-discovered tests in test_capture_model.py; no custom runner or test path bootstrap was added. Numerical, D4, training/resume, prediction export and input/checkpoint validation checks remain. Standard command: python -m pytest tests -q -p no:cacheprovider. pytest is a test-only install, not an added inference/training dependency.

Layout validation: 92 tests passed with the single pytest command; active README/AGENTS/capture-contract Markdown targets exist; git diff --check passed. Recovery originals and SHA256 move ledger are in C:/Users/hdy/AppData/Local/Temp/nfplight_test_layout_4cxgskah. Runtime sources and training source hashes were not modified in this layout pass.
