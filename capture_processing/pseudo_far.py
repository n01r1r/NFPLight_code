"""Copied-near inverse-square control of fresh, aligned FP32 capture inputs."""

from __future__ import annotations

import argparse
import html
import json
import platform
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .paper_comparison import _difference, verify_paper_equations
from .raw import sha256_file
from model.author_real_capture_adapter import AuthorRealCaptureAdapter
from run_fabric_capture import CAPTURE_GROUPS, _hwc, _verified_author_checkpoint


ATTENUATION = np.float32((4 / 12) ** 2)


def _paper_formulas() -> list:
    from .report import _AUTHOR_REAL_FORMULAS
    stages = [
        ("Near intensity", "I_N", "mean_RGB(near)", "선형 RGB 평균; 값 제한된 입력에서 계산", "paper_near_intensity"),
        ("Far intensity", "I_F", "mean_RGB(far_pseudo)", "near/9의 선형 RGB 평균", "paper_far_intensity"),
        ("Gain", "K_L", "I_N[128,128] / I_F[128,128]", "중앙 한 픽셀 강도 비율", "paper_gain"),
        ("Angle", "K_θ", "cos θ_N / cos θ_F", "거리 4/12의 각도 보정", "paper_time_map"),
        ("Scaled far", "K I_F", "K_L K_θ I_F", "스칼라 강도; 곱한 뒤 clipping하지 않음", "paper_far_scaled_intensity"),
        ("Signed difference", "I_N − K I_F", "I_N − K_L K_θ I_F", "절댓값 적용 전 부호 있는 스칼라 차이", "paper_signed_difference"),
        ("Numerator", "|I_N − K I_F|", "abs(I_N − K_L K_θ I_F)", "RGB 평균 후 차이를 계산하고 절댓값 적용", "paper_numerator"),
        ("Denominator", "C_M", "cos θ_N |cos θ_N − cos θ_F|", "논문 분모; 저자 코드 분모 정규화·floor 제외", "paper_denominator"),
        ("Raw RM", "R_M", "|I_N − K I_F| / C_M", "논문 식 (4)–(5)의 원래 관계값", "paper_relation_raw"),
    ]
    return list(_AUTHOR_REAL_FORMULAS[:2]) + stages + [
        row for row in _AUTHOR_REAL_FORMULAS
        if row[-1] in {"relation_log_raw", "relation", "relation_log", "non_saturated_mask",
                       "log_original_6_near_rgb", "log_original_6_far_rgb", "features_legacy33_rgb_preview"}]


def _interactive_reports(output: Path, pseudo: dict, record: dict, model_contract: dict) -> None:
    """Reuse the capture stage UI; saved experiment arrays remain authoritative."""
    from .report import (_AUTHOR_REAL_FORMULAS, _PAPER_NOTATION_NOTE, _burst_summary_html, _display_configs,
                         _figure_gallery, _make_html, _render_array, _save_display_previews,
                         _save_prediction_component_previews, _stats)

    for mode, source in pseudo.items():
        folder = output if mode == "author_code" else output / mode
        folder.mkdir(exist_ok=True)
        formulas = _AUTHOR_REAL_FORMULAS if mode == "author_code" else _paper_formulas()
        arrays = dict(source)
        arrays.update(near_input_clipped=source["original_rgb_6"][..., :3],
                      far_input_clipped=source["original_rgb_6"][..., 3:],
                      features_legacy33_rgb_preview=(source["features_legacy33"][..., :3] + np.float32(1)) / np.float32(2),
                      log_original_6_near_rgb=source["log_original_6"][..., :3],
                      log_original_6_far_rgb=source["log_original_6"][..., 3:])
        # Author archives retain upstream's reverse-signed trace. Only its display is reversed.
        keys = [row[-1] for row in formulas]
        if mode == "author_code":
            keys[keys.index("near_minus_scaled_far")] = "signed_diff"
        for channel in range(33):
            key = f"features_legacy33_c{channel}"
            arrays[key] = source["features_legacy33"][..., channel:channel + 1]
            keys.append(key)
        conditions = {"both": {"formula": "near–pseudo far · both · " + mode,
                               "arrays": {key: _stats(arrays[key]) for key in keys}}}
        conditions["both"]["arrays"].update({f"prediction_c{channel}": _stats(source["prediction_raw_10"][..., channel])
                                             for channel in range(10)})
        manifest = {"group": f"{output.name} · near–pseudo far · {mode}",
                    "model_contract": model_contract, "checkpoint_sha256": record["checkpoint_sha256"],
                    "conditions": conditions, "report": {"display_only": True,
                    "manifest": "display_manifest.json", "verification": "display_verification.json"}}
        ranges = {key: [0.0, 1.0] for key in keys}
        _save_display_previews(folder, ranges, conditions, keys, {"both": arrays}, manifest)
        for key in ("features_legacy33_rgb_preview", "log_original_6_near_rgb", "log_original_6_far_rgb"):
            ranges[key] = [0.0, 1.0]
            _render_array(arrays[key], key, folder / "both/display" / f"{key}.png", (0.0, 1.0), "gray")
            if key == "features_legacy33_rgb_preview":
                _render_array(arrays[key], key, folder / "both/display_gamma22" / f"{key}.png",
                              (0.0, 1.0), "gray", gamma22=True)
        _save_prediction_component_previews(folder, {"both": source})
        configs = _display_configs(keys, ranges, {"both": arrays}, {})
        galleries = _figure_gallery(folder, {"both": source}, model_contract, filename_suffix="_both_only")
        note = _PAPER_NOTATION_NOTE if mode == "author_code" else (
            "논문 Fig. 3·식 (4)–(5): RGB 평균 강도 I_N, I_F를 먼저 계산하고 "
            "K_L은 중앙 한 픽셀의 비율로 계산합니다. R_M=|I_N−K_L K_θ I_F|/C_M. "
            "분자는 곱한 뒤 clipping하지 않습니다. 원래 R_M과 log(R_M)은 고정된 저자 모델에 "
            "입력하기 위해 각각 min/max 정규화 후 1에서 뺍니다. 입력·33채널 feature·출력은 저장된 FP32 값입니다.")
        page = _make_html(manifest, keys, ranges, configs, {}, galleries,
                          formula_records=formulas, notation_note=note)
        page = page.replace(_burst_summary_html(manifest),
                            '<p class="run-strip">near는 기존 선택 프레임의 입력을 유지하고, '
                            'far_pseudo는 그 near를 1/9로 감쇠한 영상입니다. 프레임 선택·정렬·photometry 규정은 동일합니다.</p>')
        prefix = "" if mode == "author_code" else "../"
        author_link = "report.html" if mode == "author_code" else "../report.html"
        paper_link = "paper_equations/report.html" if mode == "author_code" else "report.html"
        nav = (f'<p class="run-strip">far_pseudo=near/9 · FP32 · both · 표시만 재구성 · '
               f'<a href="{author_link}">저자 코드 단계</a> · <a href="{paper_link}">논문 수식 단계</a> · '
               f'<a href="{prefix}comparison.html">actual far 대조 · 컴포넌트 비교</a> · '
               f'<a href="{prefix}provenance.json">추론 출처·변화 지표</a> · '
               f'<a href="{prefix}../index.html">5개 묶음</a></p>')
        page = page.replace('<div class="workspace">', nav + '<div class="workspace">', 1)
        (folder / "report.html").write_text(page, encoding="utf-8")
        manifest["display_ranges"] = ranges
        (folder / "display_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_reports(capture_root: Path, groups: list[str]) -> None:
    """Display-only refresh: never construct a model or rewrite inference provenance."""
    capture_root = capture_root.resolve()
    root = capture_root / "pseudo_far"
    protected = [root / "index.json"]
    for group in groups:
        protected.extend([root / group / "provenance.json", root / group / "author_code_arrays.npz",
                          root / group / "paper_equations_arrays.npz", capture_root / group / "manifest.json",
                          capture_root / group / "both/arrays.npz", capture_root / group / "both/paper_arrays.npz"])
    before = {str(path.relative_to(capture_root)): sha256_file(path) for path in protected}
    for group in groups:
        output = root / group
        record = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        manifest = json.loads((capture_root / group / "manifest.json").read_text(encoding="utf-8"))
        pseudo = {mode: _archive(output / f"{mode}_arrays.npz") for mode in ("author_code", "paper_equations")}
        if not (output / "comparison.html").exists():
            (output / "comparison.html").write_bytes((output / "report.html").read_bytes())
        _interactive_reports(output, pseudo, record, manifest["model_contract"])
        print(f"{group}: saved FP32 components rendered; inference unchanged", flush=True)
    after = {str(path.relative_to(capture_root)): sha256_file(path) for path in protected}
    if before != after:
        raise AssertionError("display refresh changed protected inference results")
    verification = {"display_only": True, "inference_run": False, "protected_sha256_before": before,
                    "protected_sha256_after": after, "numerical_results_unchanged": True,
                    "generator_sha256": sha256_file(Path(__file__)),
                    "renderer_sha256": sha256_file(Path(__file__).with_name("report.py")),
                    "groups": groups, "historical_report_hash_superseded": "report.html only; provenance preserved"}
    for group in groups:
        for folder in (root / group, root / group / "paper_equations"):
            (folder / "display_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    (root / "display_verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    cards = "".join(f'<article><h2>{group}</h2><p>near–pseudo far · both · FP32</p>'
                    f'<a href="{group}/report.html">저자 코드 단계</a> · '
                    f'<a href="{group}/paper_equations/report.html">논문 수식 단계</a><p>'
                    f'<a href="{group}/comparison.html">actual far 대조</a></p></article>' for group in groups)
    (root / "index.html").write_text('<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Near–pseudo far components</title>'
        '<style>body{font:15px Arial;background:#111;color:#ddd;margin:20px}main{max-width:1800px;margin:auto}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}'
        'article{background:#181818;border:1px solid #555;padding:12px}h2{font-size:17px}a{color:#8ccaff}</style>'
        '<main><h1>Near–pseudo far · 5개 묶음</h1><p>far_pseudo=near×(4/12)²=near/9 · 공간 좌표 유지 · 원본 net_g_real.pth · denoiser 제외</p>'
        '<p>입력 → gain·각도 → 차이·절댓값·분모 → RM → 33개 feature → normal / diffuse / roughness / specular</p>'
        '<p><a href="../index.html">실제 near–far report</a> · <a href="display_verification.json">수치 보존 검사</a></p>'
        f'<div class="grid">{cards}</div></main></html>', encoding="utf-8")


def pseudo_far_input(inputs: np.ndarray) -> np.ndarray:
    """Preserve clipped linear near; set far=near/9 without spatial resampling."""
    if inputs.shape != (256, 256, 6) or inputs.dtype != np.float32:
        raise ValueError("pseudo far requires FP32 HWC [256,256,6]")
    if not np.isfinite(inputs).all() or inputs.min() < 0 or inputs.max() > 1:
        raise ValueError("pseudo far requires finite clipped linear RGB in [0,1]")
    if inputs[128, 128, :3].mean(dtype=np.float32) <= 0:
        raise ValueError("pseudo far requires positive near center intensity")
    return np.concatenate((inputs[..., :3], inputs[..., :3] * ATTENUATION), axis=-1)


def _archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"invalid FP32 baseline archive: {path}")
    return arrays


def _save_rgb(folder: Path, name: str, value: np.ndarray, *, gamma_allowed: bool = True) -> str:
    for gamma, suffix in ((False, ""), (True, "_gamma22")):
        shown = np.clip(value, 0, 1)
        if gamma and gamma_allowed:
            shown = np.power(shown, np.float32(1 / 2.2))
        Image.fromarray(np.rint(shown * 255).astype(np.uint8)).save(folder / f"{name}{suffix}.png")
    return (f'<figure><figcaption>{html.escape(name)}</figcaption><img class="rgb" '
            f'src="{name}.png" data-linear="{name}.png" data-gamma="{name}_gamma22.png"></figure>')


def _report(output: Path, baseline: dict, pseudo: dict, record: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = baseline["author_code"]["original_rgb_6"]
    rgb = [_save_rgb(output, name, value) for name, value in (
        ("near", inputs[..., :3]), ("actual_far", inputs[..., 3:]),
        ("pseudo_far", pseudo["author_code"]["original_rgb_6"][..., 3:]))]
    components = (("normal", 0, 3), ("diffuse", 3, 6), ("roughness", 6, 7), ("specular", 7, 10))
    ranges, figures, predictions = {}, [], []
    for mode in baseline:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), layout="constrained")
        for row, key in enumerate(("relation_raw", "relation", "relation_log")):
            actual, control = baseline[mode][key].squeeze(), pseudo[mode][key].squeeze()
            low = min(float(actual.min()), float(control.min()))
            high = max(float(actual.max()), float(control.max()), low + 1e-7)
            original_range = [low, high]
            delta = control - actual
            extent = max(float(np.abs(delta).max()), 1e-7)
            if key == "relation_raw":
                low, high = (float(v) for v in np.percentile(np.concatenate((actual.ravel(), control.ravel())), [1, 99]))
                high = max(high, low + 1e-7)
                extent = max(float(np.percentile(np.abs(delta), 99)), 1e-7)
            ranges[f"{mode}/{key}"] = {"actual_pseudo_common": [low, high], "delta": [-extent, extent],
                                       "original_common_min_max": original_range,
                                       "display": "common 1–99 percentile; delta abs p99" if key == "relation_raw" else "common min-max; delta symmetric"}
            for col, (values, title, limits) in enumerate(((actual, "actual far", (low, high)),
                                                          (control, "pseudo far", (low, high)),
                                                          (delta, "pseudo - actual", (-extent, extent)))):
                ax = axes[row, col]
                im = ax.imshow(values, cmap="gray", vmin=limits[0], vmax=limits[1], interpolation="nearest")
                ax.set_title(f"{key}: {title}", fontsize=9)
                ax.axis("off")
                fig.colorbar(im, ax=ax, shrink=0.8)
        filename = f"{mode}_relations.png"
        fig.savefig(output / filename, dpi=130)
        plt.close(fig)
        figures.append(f'<h2>{mode}: RM</h2><img class="maps" src="{filename}">')
        for label, arrays in (("actual", baseline[mode]), ("pseudo", pseudo[mode])):
            tiles = []
            for name, start, end in components:
                values = (arrays["prediction_raw_10"][..., start:end] + np.float32(1)) / np.float32(2)
                if end - start == 1:
                    values = np.repeat(values, 3, axis=-1)
                tiles.append(_save_rgb(output, f"{mode}_{label}_{name}", values, gamma_allowed=name in {"diffuse", "specular"}))
            predictions.append(f'<h3>{mode}: {label}</h3><div class="tiles">{"".join(tiles)}</div>')
    record["display_ranges"] = ranges
    details = html.escape(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
    page = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>far_pseudo · {output.name}</title>
<style>body{{font:15px Arial;background:#111;color:#ddd;margin:20px}}main{{max-width:1300px;margin:auto}}.tiles{{display:flex;flex-wrap:wrap;gap:12px}}figure{{margin:0}}img.rgb{{width:256px}}img.maps{{width:100%}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#8cf}}</style>
<main><h1>far_pseudo · {output.name}</h1><p>near 그대로 · far_pseudo=near×(4/12)²=near/9 · FP32 · 원본 net_g_real.pth · both · denoiser 제외</p>
<p>near와 far_pseudo는 같은 공간 좌표입니다. 현재 real 모델의 정규화 거리 4/12와 실제 촬영 10/30cm는 거리비 3을 공유합니다.</p>
<p>gain은 매 입력에서 다시 계산합니다. pure global attenuation은 K_L≈9로 상쇄되므로, 이 실험은 실제 far의 공간·색·반사 차이를 제거한 copied-near 대조입니다. 논문 분자는 ≈|I_N(1−cosθ_N/cosθ_F)|이며 각도 역보정은 적용하지 않습니다.</p>
<p>이를 분모로 나누면 raw RM≈I_N/(cosθ_N·cosθ_F)입니다. 따라서 pseudo raw RM은 near 밝기 무늬를 주로 반영하며 실제 거리 변화의 specular 각도 효과를 재현하지 않습니다. 중앙 작은 분모에서는 FP32 상쇄 오차가 증폭될 수 있습니다.</p>
<p>단독으로 optics 원인 또는 물리 정확도를 입증하지 않습니다. near/far BRDF 차이, 센서·조명·photometry 영향도 함께 제거됩니다. clipping된 near를 그대로 복사하므로 clipping 이전 radiance 복원 실험은 아닙니다.</p>
<p><a href="../../{output.name}/report.html">실제 far 보고서</a> · <a href="provenance.json">수치·출처</a></p>
<label><input type="checkbox" onchange="document.querySelectorAll('img.rgb').forEach(i=>i.src=this.checked?i.dataset.gamma:i.dataset.linear)">RGB 표시 gamma x^(1/2.2)</label>
<p>입력 RGB 공통 [0,1]. 출력 normal XYZ, diffuse/specular RGB, roughness 회색 [0,1]. 감마는 표시만 변경합니다.</p>
<div class="tiles">{"".join(rgb)}</div>{"".join(predictions)}<p>raw RM은 중앙 spike를 포함한 원래 수치를 저장하고 두 입력 공통 1–99 percentile로 표시합니다. raw 차이는 절댓값 p99 대칭 범위이며 범위 밖 값은 표시에서만 포화됩니다. adapted/log는 actual/pseudo 공통 min/max 회색, 차이는 대칭 범위입니다. 각 RM 단위를 분리하며 출력 변화 지표는 raw tanh [−1,1] 단위입니다.</p>{"".join(figures)}
<details><summary>측정·출처</summary><pre>{details}</pre></details></main></html>'''
    (output / "report.html").write_text(page, encoding="utf-8")


def run(capture_root: Path, groups: list[str], device: str = "cuda") -> list[dict]:
    capture_root = capture_root.resolve()
    checkpoint = Path("checkpoints/net_g_real.pth").resolve()
    digest = _verified_author_checkpoint(checkpoint, "estimator")
    code_paths = [Path(__file__), Path("model/author_real_capture_adapter.py"), Path("model/paper_equations.py"),
                  Path("network/nfplight_net.py"), Path("network/base_net.py"), Path("run_fabric_capture.py")]
    code_hashes = {str(path.resolve()): sha256_file(path) for path in code_paths}
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AuthorRealCaptureAdapter(checkpoint, device=device)
    if any(p.dtype != torch.float32 for p in model.net_g.parameters()):
        raise AssertionError("estimator parameters must be FP32")
    root = capture_root / "pseudo_far"
    root.mkdir(exist_ok=True)
    records = []
    for group in groups:
        if group not in CAPTURE_GROUPS:
            raise ValueError(f"unknown active capture group: {group}")
        source = capture_root / group
        paths = [source / "both/arrays.npz", source / "both/paper_arrays.npz", source / "manifest.json"]
        hashes = {str(path.relative_to(capture_root)): sha256_file(path) for path in paths}
        manifest = json.loads(paths[2].read_text(encoding="utf-8"))
        if manifest["checkpoint_sha256"] != digest:
            raise ValueError("baseline checkpoint differs")
        baseline = {"author_code": _archive(paths[0]), "paper_equations": _archive(paths[1])}
        inputs = baseline["author_code"]["original_rgb_6"]
        if not np.array_equal(inputs, baseline["paper_equations"]["original_rgb_6"]):
            raise AssertionError("baseline modes differ in common input")
        control_input = pseudo_far_input(inputs)
        output = root / group
        output.mkdir(exist_ok=False)
        tensor = torch.from_numpy(control_input.transpose(2, 0, 1)[None].copy()).to(device)
        pseudo, metrics = {}, {}
        for mode in baseline:
            with torch.autocast(device_type=torch.device(device).type, enabled=False):
                prediction, trace = model.infer(tensor, equation_mode=mode)
            if not torch.equal(trace["features_legacy33"], model.last_estimator_input):
                raise AssertionError("saved features differ from actual estimator input")
            arrays = {key: _hwc(value) for key, value in trace.items()}
            arrays["prediction_nchw"] = prediction.cpu().numpy()
            if not np.array_equal(inputs[..., :3], arrays["original_rgb_6"][..., :3]):
                raise AssertionError("near changed")
            np.savez_compressed(output / f"{mode}_arrays.npz", **arrays)
            restored = _archive(output / f"{mode}_arrays.npz")
            if any(not np.array_equal(value, restored[key]) for key, value in arrays.items()):
                raise AssertionError("pseudo archive roundtrip changed array values")
            pseudo[mode] = arrays
            metrics[mode] = {key: _difference(baseline[mode][key], arrays[key])
                             for key in ("relation_raw", "relation", "relation_log")}
            metrics[mode]["prediction_raw_tanh_units"] = {
                name: _difference(baseline[mode]["prediction_raw_10"][..., start:end], arrays["prediction_raw_10"][..., start:end])
                for name, start, end in (("normal", 0, 3), ("diffuse", 3, 6), ("roughness", 6, 7), ("specular", 7, 10))}
            metrics[mode]["gain_actual_pseudo"] = [float(baseline[mode]["far_gain_scalar"].ravel()[0]), float(arrays["far_gain_scalar"].ravel()[0])]
        verified = verify_paper_equations(control_input, pseudo["paper_equations"])
        paper = pseudo["paper_equations"]
        expected_raw = paper["paper_near_intensity"] / (paper["paper_cos_near"] * paper["paper_cos_far"])
        raw_delta = np.abs(paper["paper_relation_raw"] - expected_raw)
        bound = np.float32(4e-7) / paper["paper_denominator"] + np.float32(5e-5) * np.abs(expected_raw)
        if not np.all(raw_delta <= bound):
            raise AssertionError("pseudo raw RM does not match copied-near analytic identity within FP32 cancellation bound")
        record = {"format": "nfplight.pseudo-far-inverse-square.v1", "group": group,
                  "distance_ratio": 3, "normalized_real_geometry": [4, 12], "attenuation_fp32": float(ATTENUATION),
                  "construction": "clipped near retained; far replaced by near/9; no spatial resampling or angular inverse correction",
                  "input_source_sha256": hashes, "selected_frames": manifest["selected_frames"],
                  "source_dng_sha256": manifest["source_dng_sha256_before"],
                  "preparation_source_hashes": manifest["preparation_source_hashes"],
                  "checkpoint_sha256": digest, "code_sha256": code_hashes,
                  "dtype": "float32", "denoiser_used": False, "seed": 0,
                  "precision_flags": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "autocast_enabled": False},
                  "archive_roundtrip_equal": True,
                  "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "device": device},
                  "metrics_pseudo_minus_actual": metrics, "independent_paper_formula_verification": verified}
        record["copied_near_analytic_identity"] = {"expected": "I_N/(cos_near*cos_far)",
            "passed": True, "max_abs_error": float(raw_delta.max()), "mean_abs_error": float(raw_delta.mean()),
            "fp32_bound": "4e-7/C_M + 5e-5*abs(expected); near-axis cancellation amplification"}
        _report(output, baseline, pseudo, record)
        if any(sha256_file(path) != hashes[str(path.relative_to(capture_root))] for path in paths):
            raise AssertionError("baseline source mutated during experiment")
        if any(sha256_file(path) != code_hashes[str(path.resolve())] for path in code_paths):
            raise AssertionError("experiment implementation changed during the run")
        if sha256_file(checkpoint) != digest:
            raise AssertionError("checkpoint changed during the run")
        record["baseline_unchanged"] = True
        record["code_unchanged"] = True
        record["checkpoint_unchanged"] = True
        record["output_sha256"] = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
        (output / "provenance.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        records.append(record)
        print(f"{group}: pseudo far author+paper completed, gain {metrics['paper_equations']['gain_actual_pseudo']}", flush=True)
    if sha256_file(checkpoint) != digest:
        raise AssertionError("checkpoint changed")
    all_records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*/provenance.json"))]
    (root / "index.json").write_text(json.dumps(all_records, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    links = "".join(f'<li><a href="{record["group"]}/report.html">{record["group"]}</a></li>' for record in all_records)
    (root / "index.html").write_text(f'<!doctype html><meta charset="utf-8"><h1>far_pseudo = near / 9</h1><p>FP32 · both · original net_g_real · gain recalculated · no resizing</p><ul>{links}</ul>', encoding="utf-8")
    refresh_reports(capture_root, [record["group"] for record in all_records])
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=CAPTURE_GROUPS, default=list(CAPTURE_GROUPS))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--report-only", action="store_true", help="Render saved components without inference")
    arguments = parser.parse_args()
    if arguments.report_only:
        refresh_reports(arguments.capture_root, arguments.groups)
    else:
        run(arguments.capture_root, arguments.groups, arguments.device)
