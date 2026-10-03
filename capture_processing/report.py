"""Display-only HTML and figure generation for the FP32 capture artifacts."""

from __future__ import annotations

import ast
import html
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .raw import sha256_file


_CONDITIONS = ("both",)
_BOTH_ONLY_GALLERIES = (
    "features_gallery_both_only.png",
    "prediction_semantic_gallery_both_only.png",
    "prediction_semantic_gallery_both_only_gamma22.png",
    "relation_gallery_both_only.png",
)
_FEATURE_CHANNELS = (
    "near.R", "near.G", "near.B", "far.R", "far.G", "far.B",
    "far_gained.R", "far_gained.G", "far_gained.B",
    "log_near.R", "log_near.G", "log_near.B",
    "log_far.R", "log_far.G", "log_far.B",
    "log_far_gained.R", "log_far_gained.G", "log_far_gained.B",
    "relation", "log_relation", "valid_mask",
)
_PREDICTION_CHANNELS = (
    "normal_x", "normal_y", "normal_z", "diffuse_r", "diffuse_g",
    "diffuse_b", "roughness", "specular_r", "specular_g", "specular_b",
)
_PREDICTION_COMPONENTS = (
    ("prediction_normal_rgb", "Normal RGB", 0, 3, True, False),
    ("prediction_diffuse_rgb", "Diffuse RGB", 3, 6, True, True),
    ("prediction_roughness", "Roughness", 6, 7, False, False),
    ("prediction_specular_rgb", "Specular RGB", 7, 10, True, True),
)
_BEST_RENDER_SHA256 = "6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd"
_FORMULAS = (
    ("N · F", "N = near input", "F = far input", "각 조건에서 photometry 및 linear-sRGB를 거쳐 [0,1]로 clip한 256×256 선형 RGB 입력입니다.", "near_input_clipped"),
    ("gain g", "raw_calibrated_v2: g=(d_f/d_n)^2", "sample_v1: g=mean(N[:,127,127]) / max(mean(F[:,127,127]), 1e-5)", "checkpoint에 기록된 feature version의 gain을 그대로 사용합니다. 거리 상수는 모델 기하와 실제 촬영 거리를 별도로 보고합니다.", "far_gain_map"),
    ("F·g", "F_g = F × g", "F_g,clip = clip(F_g, 0, 1)", "F_g,clip은 appearance와 saturation mask에 씁니다. relation에서는 clip 전 F_g에 t를 곱한 뒤 별도로 clip합니다.", "far_gained_clipped"),
    ("t · F_t", "t(x,y)=cos(atan(r/d_n))/cos(atan(r/d_f))", "F_t=clip(t × F_g, 0, 1)", "r=sqrt((x−127.5)^2+(y−127.5)^2)/128. F_t는 차이 계산에 쓰는 time-scaled far RGB입니다.", "far_time_clipped"),
    ("signed / absolute RGB difference", "Δ = F_t − N", "|Δ| = abs(Δ)", "R/G/B를 보존합니다. signed 값은 대칭 공통 범위 gray, absolute 값은 공통 gray 범위로 채널별 표시합니다.", "signed_diff"),
    ("mean absolute difference", "M = mean_RGB(|Δ|)", "shape [256,256,1], nonnegative", "세 RGB 절대 차이의 산술 평균입니다. 보고서의 relation numerator입니다.", "mean_abs_diff"),
    ("raw optical coefficient", "c_raw=(cos θ_f − cos θ_n)/cos θ_n", "c_norm=c_raw/max(c_raw)", "θ_n=atan(r/d_n), θ_f=atan(r/d_f). c_raw는 model geometry에서 계산되며 입력과 독립입니다.", "coefficient_raw"),
    ("denominator", "c_den=max(c_norm, 1e-5)", "relation_raw=M/c_den", "분모 floor는 coefficient가 0에 가까운 위치에서 나눗셈을 안정화합니다. raw ratio에는 고정 upper clip이 없습니다.", "denominator"),
    ("log ratio", "relation_floor=max(relation_raw, 1e-5)", "relation_log_raw=ln(relation_floor)", "floor는 log가 유한하도록 적용합니다. raw 로그는 음수가 될 수 있고 보고서는 저장된 display range를 사용합니다.", "relation_log_raw"),
    ("normalize and invert", "ClipToOne(x)=(x−min(x))/max(max(x)−min(x), 1e-5)", "relation=1−ClipToOne(relation_raw)", "log_relation=1−ClipToOne(relation_log_raw). 이 min–max 결과는 최종 relation/log relation 양쪽 모두 [0,1] 고정 표시 범위입니다.", "relation"),
    ("log appearance", "L(x)=(ln(x+0.01)−ln(0.01))/(ln(1.01)−ln(0.01))", "x∈[0,1] → L(x)∈[0,1]", "near, far, far_gained RGB에 채널별 적용됩니다. 이 연산은 camera RGB 표시 gamma나 sRGB OETF와 다릅니다.", "log_near"),
    ("21-channel model input", "[N, F, F_g,clip, L(N), L(F), L(F_g,clip), relation, log_relation, valid_mask] × 2 − 1", "HWC channels → network input [1,21,256,256], range [−1,1]", "채널 순서는 아래 표에 나열했습니다. 모델 입력 값은 FP32 NPZ에 있고 PNG는 표시 전용입니다.", "features_c0"),
    ("saturation mask", "sat = 1[max_RGB(N)>0.95 OR max_RGB(F_g,clip)>0.95]", "valid_mask = 1 − sat", "이 0/1 feature는 기하학적 warp support와 다른 포화 제외 mask입니다. 값 1은 채널 최대값 기준으로 포화되지 않은 픽셀입니다.", "valid_mask"),
)
_AUTHOR_REAL_FORMULAS = (
    ("linear input", "X = [I_N, I_F]", "선형 색상값、0–1", "가까운 조명과 먼 조명에서 촬영한 영상을 사용합니다. 현재 실험은 denoiser를 제외하므로 복사 입력도 같은 영상입니다.", "near_input_clipped"),
    ("far input", "I_F", "선형 색상값、0–1", "먼 조명에서 촬영한 입력 영상입니다. 밝기 배율을 곱하기 전 값입니다.", "far_input_clipped"),
    ("center-patch gain", "K_L = mean(I_N[118:138,118:138]) / max(mean(I_F[118:138,118:138]),10⁻⁵)", "논문 식 (4)의 중앙 밝기 비율에 대응", "저자 코드는 중앙 20×20 영역의 빨강·초록·파랑을 평균하여 먼 영상의 밝기 배율을 구합니다.", "far_gain_scalar"),
    ("angle ratio", "K_θ = cos θ_N / cos θ_F", "θ_N = atan(ρ/4), θ_F = atan(ρ/12)", "입사각에 따른 밝기 차이를 보정합니다. ρ = √((x−127.5)²+(y−127.5)²)/128이며 4·12는 저자 모델의 기하 상수입니다.", "time_co_map"),
    ("gained far", "K_L I_F", "색상 입력에 쓰는 복사본은 0–1로 제한", "먼 영상의 밝기를 가까운 영상의 밝기에 맞춘 결과입니다.", "original_copy_far_gained_clipped"),
    ("time-scaled far", "K = K_L K_θ; clip(K I_F,0,1)", "논문 Fig. 3의 K I_F에 대응", "밝기와 입사각 보정을 모두 적용한 먼 영상입니다. time은 시간 보정이 아니라 코드 내부 이름입니다.", "far_time_clipped"),
    ("difference", "Δ = I_N − clip(K I_F,0,1)", "near − scaled far、절댓값 적용 전", "논문과 같은 뺄셈 방향으로 표시합니다. 저장된 저자 trace의 반대 부호는 표시용으로만 반전합니다. R/G/B 순서입니다.", "near_minus_scaled_far"),
    ("absolute difference", "|I_N − clip(K I_F,0,1)|", "빨강·초록·파랑별 절대 차이", "관계 맵 분자의 색상별 값입니다.", "abs_diff"),
    ("mean difference", "mean_RGB(|I_N − clip(K I_F,0,1)|)", "색상별 절대 차이의 산술 평균", "저자 코드가 관계 맵 분자로 사용하는 값입니다.", "mean_abs_diff_rgb"),
    ("relation denominator", "논문: C_M = cos θ_N |cos θ_N − cos θ_F|", "코드: c_raw=(cos θ_F−cos θ_N)/cos θ_N; c=c_raw/max(c_raw); 분모=max(c,10⁻⁵)", "C_M에 대응하는 계산 단계입니다. 저장된 코드 분모는 논문 식과 같지 않으며 원래 계산을 그대로 표시합니다.", "denominator"),
    ("raw relation", "q = mean_RGB(|Δ|) / max(c,10⁻⁵)", "R_M 대응 단계、정규화 전", "분자를 코드 분모로 나눈 관계 맵입니다. 원래 값에는 상한 제한이 없습니다.", "relation_raw"),
    ("raw log relation", "ln(max(q,10⁻⁵))", "정규화 전 관계 맵의 로그", "0의 로그를 피하도록 최솟값을 제한합니다. 원래 값은 음수일 수 있습니다.", "relation_log_raw"),
    ("normalized relation", "R_M 입력 = 1 − Normalize(q)", "Normalize(z)=(z−min(z))/max(max(z)−min(z),10⁻⁵)", "저자 코드는 관계 맵을 0–1로 정규화한 뒤 밝고 어두운 값을 반전하여 모델에 넣습니다.", "relation"),
    ("normalized log relation", "log(R_M) 입력 = 1 − Normalize(ln(max(q,10⁻⁵)))", "0–1、로그 적용 후 별도로 정규화·반전", "이미 정규화된 R_M의 로그가 아니라 원래 q에 로그를 적용한 결과입니다.", "relation_log"),
    ("saturation mask", "M_R = 1[max_RGB(I_N,I_F,clip(K_L I_F)) ≤ 0.95]", "흰색 1: 유효、검정 0: 과다 노출 제외", "관계 맵의 신뢰 영역입니다. 원본 영상이 존재하는지를 표시하는 기하 정렬 마스크와 구분합니다.", "non_saturated_mask"),
    ("log operator · near RGB", "L(I_N); L(x)=(ln(x+0.01)−ln(0.01))/(ln(1.01)−ln(0.01))", "0–1、어두운 부분을 펼치는 로그 변환", "저장된 로그 변환 영상입니다. 추가 화면 감마를 적용하지 않습니다.", "log_original_6_near_rgb"),
    ("log operator · far RGB", "L(I_F)", "가까운 영상과 같은 로그 변환", "먼 입력 영상의 어두운 부분을 펼칩니다. 추가 화면 감마를 적용하지 않습니다.", "log_original_6_far_rgb"),
    ("33-channel input", "[I_N,I_F,copy(I_N),copy(I_F),clip(K_L I_F),각 영상의 L(x),R_M,log(R_M),M_R] × 2 − 1", "15 + 15 + 3 = 33개 채널、저장값 −1–1", "여기서는 첫 가까운 영상만 미리 봅니다. 전체 채널은 아래 갤러리에서 확인할 수 있습니다.", "features_legacy33_rgb_preview"),
)


# Paper Fig. 3 / Eqs. 4–5 notation. These labels never change stored array keys.
_STAGE_LABELS = {
    "paper_near_intensity": "I_N · 가까운 입력의 RGB 평균 강도",
    "paper_far_intensity": "I_F · pseudo far 입력의 RGB 평균 강도",
    "paper_gain": "K_L · 중앙 픽셀의 강도 비율",
    "paper_time_map": "K_θ · 조명 입사각의 코사인 비율",
    "paper_far_scaled_intensity": "K I_F · 값 제한 없는 보정된 강도",
    "paper_signed_difference": "I_N − K I_F · 절댓값 적용 전 스칼라 차이",
    "paper_numerator": "|I_N − K I_F| · 스칼라 차이의 절댓값",
    "paper_denominator": "C_M · 논문 식의 기하 분모",
    "paper_relation_raw": "R_M · 논문 식의 정규화 전 관계 맵",
    "near_input_clipped": "I_N · 가까운 조명에서 촬영한 모델 입력 영상",
    "far_input_clipped": "I_F · 먼 조명에서 촬영한 모델 입력 영상",
    "far_gain_scalar": "K_L · 가까운 영상에 맞추는 먼 영상의 밝기 배율",
    "far_gain_map": "K_L · 먼 영상의 밝기 배율",
    "time_co_map": "K_θ · 조명 입사각의 코사인 비율",
    "original_copy_far_gained_unclipped": "K_L I_F · 밝기를 맞춘 먼 영상 — 값 제한 전",
    "original_copy_far_gained_clipped": "K_L I_F · 밝기를 맞춘 먼 영상 — 0–1 값 제한 후",
    "far_gained_clipped": "K_L I_F · 밝기를 맞춘 먼 영상 — 0–1 값 제한 후",
    "far_time_unclipped": "K I_F · 밝기와 입사각을 맞춘 먼 영상 — 값 제한 전",
    "far_time_clipped": "K I_F · 밝기와 입사각을 맞춘 먼 영상 — 0–1 값 제한 후",
    "signed_diff": "저장된 저자 trace · scaled far − near",
    "near_minus_scaled_far": "I_N − clip(K I_F,0,1) · 절댓값 적용 전 색상별 차이",
    "abs_diff": "|I_N − clip(K I_F,0,1)| · 색상별 차이의 절댓값",
    "mean_abs_diff_rgb": "mean_RGB(|I_N − clip(K I_F,0,1)|) · 저자 코드 분자",
    "mean_abs_diff": "|I_N − K I_F| · 색상별 절대 차이의 평균",
    "coefficient": "C_M 대응 단계 · 저자 코드의 정규화된 기하 계수",
    "coefficient_raw": "C_M 대응 단계 · 저자 코드의 정규화 전 기하 계수",
    "coefficient_normalization_max": "기하 계수를 정규화할 때 나누는 최댓값",
    "denominator": "C_M 대응 단계 · 0으로 나누지 않도록 제한한 분모",
    "relation_raw": "R_M 대응 단계 · 정규화 전 관계 맵",
    "relation_floored": "관계 맵에 로그를 적용하기 전 최솟값 제한",
    "relation_log_raw": "관계 맵의 로그 — 정규화 전",
    "relation_normalized": "관계 맵 — 0–1 정규화 후, 반전 전",
    "relation_log_normalized": "관계 맵의 로그 — 0–1 정규화 후, 반전 전",
    "relation": "R_M · 두 조명 영상의 관계 맵 — 모델 입력",
    "relation_log": "log(R_M) · 관계 맵의 로그 — 모델 입력",
    "log_relation": "log(R_M) · 관계 맵의 로그 — 모델 입력",
    "non_saturated_mask": "M_R · 과다 노출을 제외한 유효 영역",
    "valid_mask": "M_R · 과다 노출을 제외한 유효 영역",
    "log_original_6_near_rgb": "log(I_N) · 가까운 입력 영상의 로그 변환",
    "log_original_6_far_rgb": "log(I_F) · 먼 입력 영상의 로그 변환",
    "features_legacy33_rgb_preview": "신경망 입력 33개 채널 · 가까운 영상 미리보기",
    "prediction_normal_rgb": "n · 표면이 향하는 방향",
    "prediction_diffuse_rgb": "d · 확산 반사 색상",
    "prediction_roughness": "r · 표면 거칠기",
    "prediction_specular_rgb": "s · 정반사 색상",
}


def _stage_label(key: str) -> str:
    if key in _STAGE_LABELS:
        return _STAGE_LABELS[key]
    base, channel = _split_channel_key(key)
    if channel is not None:
        if base in {"features_legacy33", "features_unscaled_33"}:
            groups = ("I_N · 가까운 영상", "I_F · 먼 영상", "I_N · 가까운 영상의 동일 복사본",
                      "I_F · 먼 영상의 동일 복사본", "K_L I_F · 밝기를 맞춘 먼 영상")
            if channel < 30:
                label = groups[(channel % 15) // 3]
                if channel >= 15:
                    label += "의 로그 변환"
                label += " · " + ("빨강", "초록", "파랑")[channel % 3]
            else:
                label = ("R_M · 관계 맵", "log(R_M) · 관계 맵의 로그", "M_R · 유효 영역")[channel - 30]
            return f"신경망 입력 채널 {channel + 1} · {label}" + (
                " — 저장값 −1–1" if base == "features_legacy33" else " — 변환 전 0–1")
        return _stage_label(base) + f" · 채널 {channel + 1}"
    if key.startswith("frame_"):
        _, side, index, name = key.split("_", 3)
        return f"{'가까운' if side == 'near' else '먼'} 조명 · 촬영 {int(index) + 1} · " + _stage_label(name)
    if key.startswith(("near_", "far_")):
        side, name = key.split("_", 1)
        return ("가까운 조명 · " if side == "near" else "먼 조명 · ") + _stage_label(name)
    names = {
        "bayer_u16": "카메라 원시 센서값 — 정수 저장; 화면은 색상 복원 영상",
        "bayer_selected_dtype": "카메라 원시 센서값 — FP32 저장; 화면은 색상 복원 영상",
        "demosaic_rgb_selected_dtype": "원시 센서의 색상을 복원한 영상",
        "libraw_demosaic_rgb_selected_dtype": "LibRaw 색상 복원 비교 영상",
        "warp512": "같은 재료 영역으로 정렬한 영상 — 512×512",
        "crop420": "가장자리 제거 후 영상 — 420×420",
        "output256": "신경망 해상도로 축소한 영상 — 256×256",
        "valid512": "정렬 후 원본 영상이 존재하는 영역 — 512×512",
        "valid420": "가장자리 제거 후 원본 영상이 존재하는 영역 — 420×420",
        "valid": "원본 영상이 존재하는 영역 — 기하 정렬 기준",
        "counts": "선택한 촬영의 카메라 색상값",
        "compensated": "검정·흰색 기준으로 밝기를 보정한 영상",
        "whitebalanced": "카메라 화이트밸런스를 적용한 영상",
        "linear_srgb": "선형 sRGB 색 공간으로 변환한 영상",
        "input_unclipped": "모델 입력 영상 — 0–1 값 제한 전",
        "input_clip_delta": "모델 입력의 값 제한으로 제거된 차이",
        "libraw_counts": "LibRaw로 복원한 카메라 색상값",
        "libraw_output256": "LibRaw 색상 복원 비교 영상 — 256×256",
        "libraw_difference256": "색상 복원 방법 사이의 차이 — 256×256",
        "ahd_rectified_difference": "색상 복원 방법 사이의 정렬 후 차이",
        "original_rgb_6": "I_N / I_F · 두 입력 영상",
        "original_copy_pair_6": "I_N / I_F · 두 입력 영상의 동일 복사본",
        "log_original_6": "log(I_N) / log(I_F) · 두 입력 영상의 로그 변환",
        "log_original_copy_6": "두 입력 영상 복사본의 로그 변환",
        "log_original_copy_far_gained_3": "log(K_L I_F) · 밝기를 맞춘 먼 영상의 로그 변환",
        "libraw_relation_difference": "색상 복원 방법 사이의 관계 맵 차이",
        "libraw_log_relation_difference": "색상 복원 방법 사이의 로그 관계 맵 차이",
        "libraw_prediction_difference": "색상 복원 방법 사이의 재료 추정값 차이",
        "features": "신경망 입력",
        "prediction": "재료 추정값 — 저장된 원래 값",
        "prediction_raw_10": "재료 추정값 — 저장된 원래 값",
        "prediction_display": "재료 추정값 — 0–1 표시값",
        "prediction_display_10": "재료 추정값 — 0–1 표시값",
    }
    return names.get(key, key.replace("_", " "))


_PAPER_NOTATION_NOTE = (
    "논문 Fig. 3, 식 (4)–(5): K = K_L K_θ, C_M = cos θ_N |cos θ_N − cos θ_F|, "
    "R_M = |I_N − K I_F| / C_M. "
    "현재 결과는 denoiser를 제외한 저자 코드의 저장값입니다. K_L은 중앙 20×20 영역의 색상 평균 비율, "
    "K_θ = cos θ_N / cos θ_F입니다. 코드의 분모는 "
    "c_raw = (cos θ_F − cos θ_N) / cos θ_N, c = c_raw / max(c_raw), max(c, 10⁻⁵)이며 "
    "논문 C_M 식과 같지 않습니다. 모델 입력 R_M과 log(R_M)은 각각 원래 관계값과 그 로그를 "
    "0–1 정규화한 뒤 1에서 뺀 값입니다. M_R의 흰색(1)은 과다 노출되지 않은 영역입니다. "
    "센서 복원·정렬·크기 변경은 논문 기호가 없는 촬영 전처리 단계로 표시합니다."
)

def _stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError("report input must be non-empty and finite")
    work = array.astype(np.float32, copy=False)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(work.mean(dtype=np.float32)),
        "std": float(work.std(dtype=np.float32)),
        "p01": float(np.percentile(work, np.float32(1))),
        "p99": float(np.percentile(work, np.float32(99))),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _only_both(records: Any) -> dict[str, Any]:
    """Return a render-only copy of condition-keyed data containing only both."""
    if not isinstance(records, dict) or "both" not in records:
        return {}
    return {"both": records["both"]}


def _both_only_comparison(comparison: Any) -> dict[str, Any] | None:
    """Filter condition-indexed comparison metadata without changing its source JSON."""
    if not isinstance(comparison, dict):
        return None
    filtered = dict(comparison)
    for key in ("assets", "conditions", "reference_npz_sha256_before", "reference_npz_sha256_after"):
        if isinstance(filtered.get(key), dict):
            filtered[key] = _only_both(filtered[key])
    asset_hashes = filtered.get("asset_sha256")
    if isinstance(asset_hashes, dict):
        filtered["asset_sha256"] = {
            name: digest for name, digest in asset_hashes.items()
            if "/both/" in str(name).replace("\\", "/")
        }
    return filtered


def _split_channel_key(key: str) -> tuple[str, int | None]:
    base, separator, channel = key.rpartition("_c")
    if separator and channel.isdigit():
        return base, int(channel)
    return key, None


def _signed_key(key: str) -> bool:
    base, _ = _split_channel_key(key)
    return (base in {"signed_diff", "near_minus_scaled_far", "far_gain_clip_delta", "time_clip_delta"}
            or "difference" in base or base.endswith("_delta"))


def _side_for_key(key: str) -> str | None:
    base, _ = _split_channel_key(key)
    if base.startswith("frame_near_") or base.startswith("near_") or base == "near":
        return "near"
    if base.startswith("frame_far_") or base.startswith("far_") or base == "far":
        return "far"
    return None


def _rgb_display_action(key: str) -> str:
    """Select only the display transform required by an RGB stage."""
    base, _ = _split_channel_key(key)
    if _signed_key(key) or base == "abs_diff" or "difference" in base:
        return "scalar"
    if "whitebalanced" in base:
        return "matrix"
    if (base in {"near", "far"} or "linear_srgb" in base or "input_" in base
            or base in {"far_gained_clipped", "far_time_clipped"}):
        return "linear_rgb"
    camera_tokens = ("compensated", "demosaic_rgb", "libraw_", "counts", "warp512", "crop420", "output256")
    if any(token in base for token in camera_tokens):
        return "wb_matrix"
    return "linear_rgb"


def _is_rgb_visual(key: str, value: np.ndarray) -> bool:
    array = np.asarray(value)
    base, _ = _split_channel_key(key)
    return (array.ndim == 3 and array.shape[-1] == 3
            and not (_signed_key(key) or base == "abs_diff" or "absolute_difference" in base))


def _camera_transform(manifest: dict[str, Any], side: str,
                      frame_metadata: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized camera WB gains and the camera-to-linear-sRGB matrix."""
    transforms = manifest.get("color_transforms", {}) or {}
    side_transform = transforms.get(side, {}) if isinstance(transforms, dict) else {}
    metadata = frame_metadata or {}
    gains_value = side_transform.get("normalized_rgb_wb") or side_transform.get("camera_wb")
    if gains_value is None:
        gains_value = metadata.get("camera_whitebalance")
    matrix_value = side_transform.get("camera_to_linear_srgb")
    if matrix_value is None:
        matrix_value = metadata.get("color_matrix")
    if gains_value is None or matrix_value is None:
        raise ValueError(f"WB and camera color matrix are required for {side} camera-RGB previews")
    gains = np.asarray(gains_value, dtype=np.float32).reshape(-1)[:3]
    positive = gains[np.isfinite(gains) & (gains > 0)]
    if gains.shape != (3,) or positive.size != 3:
        raise ValueError(f"invalid camera WB gains for {side}: {gains_value}")
    gains = gains / np.float32(positive.min())
    matrix = np.asarray(matrix_value, dtype=np.float32)
    if matrix.shape == (3, 4):
        matrix = matrix[:, :3]
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"invalid camera-to-linear-sRGB matrix for {side}: {matrix_value}")
    return gains.astype(np.float32, copy=False), matrix.astype(np.float32, copy=False)


def _display_rgb(value: np.ndarray, key: str, manifest: dict[str, Any],
                 *, frame_metadata: dict[str, Any] | None = None) -> np.ndarray:
    """Apply WB/matrix to a display copy only; preserve input arrays and statistics."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[-1] != 3:
        return array
    action = _rgb_display_action(key)
    side = _side_for_key(key)
    if action == "scalar" or action == "linear_rgb":
        return array
    if side is None:
        raise ValueError(f"cannot determine camera side for RGB preview {key}")
    gains, matrix = _camera_transform(manifest, side, frame_metadata)
    if action == "wb_matrix":
        array = array * gains.reshape(1, 1, 3)
    # whitebalanced input receives only the matrix; camera RGB receives WB first.
    return (array @ matrix.T).astype(np.float32, copy=False)


def _fixed_range(key: str) -> tuple[float, float] | None:
    base, _ = _split_channel_key(key)
    if base in {"relation", "log_relation", "relation_log", "relation_normalized", "relation_log_normalized",
                "near_input_clipped", "far_input_clipped", "far_gained_clipped",
                "far_time_clipped"}:
        return 0.0, 1.0
    if base in {"features", "features_legacy33", "prediction", "prediction_raw_10"}:
        return -1.0, 1.0
    if base in {"features_unscaled_33", "original_rgb_6", "original_copy_pair_6",
                "log_original_6", "log_original_copy_6", "non_saturated_mask"}:
        return 0.0, 1.0
    if (base in {"valid_mask", "near_valid", "far_valid"}
            or base.endswith(("_valid", "valid420", "valid512"))):
        return 0.0, 1.0
    return None


def _display_contract(key: str, suggested: list[float] | tuple[float, float]) -> tuple[float, float, str]:
    fixed = _fixed_range(key)
    if fixed is not None:
        return fixed[0], fixed[1], "gray" if _split_channel_key(key)[0] in {"features", "prediction"} else "gray"
    low, high = float(suggested[0]), float(suggested[1])
    if not np.isfinite((low, high)).all():
        raise FloatingPointError(f"non-finite display range for {key}")
    if high < low:
        raise ValueError(f"display range is reversed for {key}: [{low}, {high}]")
    if _signed_key(key):
        extent = max(abs(low), abs(high), 1e-12)
        return -extent, extent, "gray"
    if _split_channel_key(key)[0] in {"features", "prediction"}:
        return -1.0, 1.0, "gray"
    return low, max(high, low + 1e-12), "gray"


def _gamma22(linear: np.ndarray) -> np.ndarray:
    """Optional display-only gamma x^(1/2.2) on normalized RGB values."""
    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.power(values, np.float32(1.0 / 2.2)).astype(np.float32, copy=False)


def _prediction_hwc(value: np.ndarray) -> np.ndarray:
    """Read a 10-channel prediction layout without changing its stored values."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 10:
        return array
    if array.ndim == 3 and array.shape[0] == 10:
        return array.transpose(1, 2, 0)
    raise ValueError(f"expected one 10-channel prediction map, got {array.shape}")


def _prediction_array(arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    """Resolve a stored raw prediction, preferring the un-decoded array."""
    for key in ("prediction", "prediction_nchw", "prediction_raw_10"):
        if key in arrays:
            return np.asarray(arrays[key])
    return None


def _contract_channel_count(contract: dict[str, Any] | None) -> int | None:
    contract = contract or {}
    for key in ("input_channel_count", "feature_channels", "input_channels"):
        value = contract.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_author_real_contract(contract: dict[str, Any] | None) -> bool:
    contract = contract or {}
    family = str(contract.get("estimator_family", "")).lower()
    architecture = str(contract.get("estimator_architecture", contract.get("architecture", ""))).lower()
    return (_contract_channel_count(contract) == 33
            or "author_legacy33" in family or "legacy33" in family
            or "twobranchrealnet" in architecture
            or contract.get("feature_version") == "legacy_batch_v0")


def _feature_key(arrays: dict[str, np.ndarray], contract: dict[str, Any] | None = None) -> str | None:
    """Find the contract-specific feature tensor without guessing from old metadata."""
    if "features_legacy33" in arrays:
        return "features_legacy33"
    if "features" in arrays:
        return "features"
    return None


def _identity_copy_features_match(feature_maps: dict[str, np.ndarray]) -> bool:
    """Verify the author 33-channel copy slots before omitting duplicate tiles."""
    if not feature_maps:
        return False
    for feature in feature_maps.values():
        if feature.shape[-1] != 33:
            return False
        if not (np.array_equal(feature[..., 0:3], feature[..., 6:9])
                and np.array_equal(feature[..., 3:6], feature[..., 9:12])
                and np.array_equal(feature[..., 15:18], feature[..., 21:24])
                and np.array_equal(feature[..., 18:21], feature[..., 24:27])):
            return False
    return True


def _prediction_decoded(value: np.ndarray) -> np.ndarray:
    """Display-only channel mapping; no normal-vector renormalization."""
    return np.clip((_prediction_hwc(value) + np.float32(1.0)) * np.float32(0.5),
                   np.float32(0.0), np.float32(1.0))


def _save_prediction_component_previews(
        output: Path, condition_arrays: dict[str, dict[str, np.ndarray]]) -> list[str]:
    """Write semantic prediction previews while leaving raw10 arrays untouched."""
    from matplotlib import colormaps

    written: list[str] = []
    for mode in _CONDITIONS:
        if mode not in condition_arrays:
            continue
        prediction = _prediction_array(condition_arrays[mode])
        if prediction is None:
            continue
        decoded = _prediction_decoded(prediction)
        for key, _, start, end, is_rgb, gammaable in _PREDICTION_COMPONENTS:
            values = decoded[..., start:end]
            preview = values if is_rgb else colormaps["gray"](values[..., 0])[..., :3]
            variants = ((False, "display"), (True, "display_gamma22")) if gammaable else ((False, "display"),)
            for gamma22, folder in variants:
                shown = _gamma22(preview) if gamma22 else preview
                relative = Path(mode) / folder / f"{key}.png"
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                pixels = np.rint(np.clip(shown, 0.0, 1.0) * 255).astype(np.uint8)
                Image.fromarray(pixels).save(target)
                written.append(relative.as_posix())
    return written


def _prediction_component_preview(value: np.ndarray, start: int, end: int,
                                  is_rgb: bool, gamma22: bool) -> np.ndarray:
    from matplotlib import colormaps

    decoded = _prediction_decoded(value)[..., start:end]
    if is_rgb:
        shown = _gamma22(decoded) if gamma22 else decoded
    else:
        shown = colormaps["gray"](decoded[..., 0])[..., :3]
    return np.rint(np.clip(shown, 0.0, 1.0) * 255).astype(np.uint8)


def _comparison_reference_dir(output: Path, manifest: dict[str, Any]) -> Path | None:
    record = manifest.get("comparison_reference")
    if isinstance(record, dict) and record.get("path"):
        return (output / str(record["path"])).resolve()
    if output.parent.name != "fabric_capture_20261001_author_original":
        return None
    group = str(manifest.get("group") or output.name)
    return output.parent.parent / "fabric_capture_20261001_clean" / group


def _write_checkpoint_comparison(output: Path, manifest: dict[str, Any],
                                 condition_arrays: dict[str, dict[str, np.ndarray]]) -> tuple[dict[str, Any] | None, list[str]]:
    """Render author-vs-preserved-checkpoint predictions from immutable raw NPZs."""
    if not _is_author_real_contract(manifest.get("model_contract", {})):
        return None, []
    reference = _comparison_reference_dir(output, manifest)
    if reference is None or not reference.is_dir():
        return {"status": "unavailable", "reason": "preserved best_render group directory not found"}, []
    reference_manifest_path = reference / "manifest.json"
    if not reference_manifest_path.is_file():
        return {"status": "unavailable", "reason": "preserved group manifest missing"}, []
    reference_manifest_sha = sha256_file(reference_manifest_path)
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    if reference_manifest.get("checkpoint_sha256") != _BEST_RENDER_SHA256:
        return {"status": "unavailable", "reason": "reference checkpoint SHA does not match preserved best_render"}, []

    modes = [mode for mode in _CONDITIONS if mode in condition_arrays]
    if modes != list(_CONDITIONS):
        return {"status": "unavailable", "reason": "author report is missing one or more compensation conditions"}, []
    old_arrays: dict[str, np.ndarray] = {}
    reference_npz_paths: dict[str, Path] = {}
    for mode in modes:
        path = reference / mode / "arrays.npz"
        if not path.is_file():
            return {"status": "unavailable", "reason": f"preserved condition archive missing: {mode}"}, []
        with np.load(path, allow_pickle=False) as archive:
            prediction = _prediction_array({key: np.asarray(archive[key]) for key in archive.files})
            if prediction is None:
                return {"status": "unavailable", "reason": f"preserved raw prediction missing: {mode}"}, []
            old_arrays[mode] = prediction.copy()
        reference_npz_paths[mode] = path

    old_hashes_before = {mode: sha256_file(path) for mode, path in reference_npz_paths.items()}
    assets: dict[str, dict[str, dict[str, str]]] = {}
    written: list[str] = []
    for mode in modes:
        author_prediction = _prediction_array(condition_arrays[mode])
        if author_prediction is None:
            return {"status": "unavailable", "reason": f"author raw prediction missing: {mode}"}, []
        author_hwc = _prediction_hwc(author_prediction)
        old_hwc = _prediction_hwc(old_arrays[mode])
        if author_hwc.shape != old_hwc.shape:
            return {"status": "unavailable", "reason": f"prediction shapes differ for {mode}"}, []
        assets[mode] = {}
        for key, _, start, end, is_rgb, gammaable in _PREDICTION_COMPONENTS:
            assets[mode][key] = {}
            for branch, prediction in (("author_real_no_denoiser", author_hwc),
                                       ("best_render_preserved", old_hwc)):
                for gamma22 in ((False, True) if gammaable else (False,)):
                    suffix = "_gamma22" if gamma22 else ""
                    relative = Path("comparison") / mode / f"{branch}_{key}{suffix}.png"
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    pixels = _prediction_component_preview(prediction, start, end, is_rgb, gamma22)
                    Image.fromarray(pixels).save(target)
                    written.append(relative.as_posix())
                    assets[mode][key][f"{branch}{suffix}"] = relative.as_posix()
    old_hashes_after = {mode: sha256_file(path) for mode, path in reference_npz_paths.items()}
    if old_hashes_before != old_hashes_after or sha256_file(reference_manifest_path) != reference_manifest_sha:
        raise AssertionError("preserved best_render manifest or numeric archives changed during read-only comparison")
    report_path = os.path.relpath(reference / "report.html", output).replace("\\", "/")
    comparison = {
        "status": "ready",
        "author_label": "저자 real · denoiser 제외",
        "reference_label": "best_render · 보존 결과",
        "reference_group_relative": os.path.relpath(reference, output).replace("\\", "/"),
        "reference_report_relative": report_path,
        "reference_checkpoint": reference_manifest.get("checkpoint"),
        "reference_checkpoint_sha256": reference_manifest.get("checkpoint_sha256"),
        "reference_manifest_sha256": reference_manifest_sha,
        "reference_npz_sha256_before": old_hashes_before,
        "reference_npz_sha256_after": old_hashes_after,
        "reference_numeric_archives_unchanged": old_hashes_before == old_hashes_after,
        "same_group_and_conditions": True,
        "weight_only_comparison": False,
        "interpretation": "The feature and gain pipelines differ; this is a same-input sensitivity comparison, not a weight-only experiment.",
        "assets": assets,
        "asset_sha256": {name: sha256_file(output / name) for name in written},
    }
    return comparison, written


def _prediction_component_configs() -> dict[str, dict[str, Any]]:
    from matplotlib import colormaps

    scalar_colors = [colormaps["gray"](index / 31) for index in range(32)]
    scalar_stops = [f"rgb({round(c[0]*255)},{round(c[1]*255)},{round(c[2]*255)}) {i*100/31:.2f}%"
                    for i, c in enumerate(scalar_colors)]
    return {
        key: {"range": [0.0, 1.0], "cmap": "gray" if not is_rgb else "rgb",
              "rgb": is_rgb, "gammaable": gammaable,
              "display_fallback": None, "exposure": [0.0, 1.0] if is_rgb else None,
              "stops": [] if is_rgb else scalar_stops}
        for key, _, _, _, is_rgb, gammaable in _PREDICTION_COMPONENTS
    }


def _feature_groups(channel_count: int, model_contract: dict[str, Any] | None = None
                    ) -> tuple[list[tuple[str, tuple[int, int, int]]], list[tuple[str, int]], tuple[float, float] | None]:
    """Return only channel groupings and ranges supported by the recorded model contract."""
    contract = model_contract or {}
    version = contract.get("feature_version")
    if channel_count == 21 and version == "raw_calibrated_v2":
        rgb_titles = ("N", "F", "F × g", "log N", "log F", "log(F × g)")
        rgb_groups = [(title, (index * 3, index * 3 + 1, index * 3 + 2))
                      for index, title in enumerate(rgb_titles)]
        return rgb_groups, [("relation", 18), ("log relation", 19), ("valid mask", 20)], (-1.0, 1.0)
    if channel_count == 33 and _is_author_real_contract(contract):
        rgb_groups = [
            ("original near", (0, 1, 2)),
            ("original far", (3, 4, 5)),
            ("copy of near", (6, 7, 8)),
            ("copy of far", (9, 10, 11)),
            ("gained far", (12, 13, 14)),
            ("log original near", (15, 16, 17)),
            ("log original far", (18, 19, 20)),
            ("log copy of near", (21, 22, 23)),
            ("log copy of far", (24, 25, 26)),
            ("log gained far", (27, 28, 29)),
        ]
        return rgb_groups, [("relation", 30), ("log relation", 31), ("valid mask", 32)], (-1.0, 1.0)
    raw_range = contract.get("feature_value_range", contract.get("input_range"))
    fixed_range = tuple(map(float, raw_range[:2])) if isinstance(raw_range, (list, tuple)) and len(raw_range) >= 2 else None
    return [], [(f"channel {index}", index) for index in range(channel_count)], fixed_range


def _render_array(value: np.ndarray, key: str, target: Path,
                  value_range: tuple[float, float], cmap_name: str, *,
                  manifest: dict[str, Any] | None = None,
                  frame_metadata: dict[str, Any] | None = None,
                  gamma22: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colormaps

    array = np.asarray(value, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FloatingPointError(f"non-finite display array: {key}")
    low, high = value_range
    base, _ = _split_channel_key(key)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    channel_triptych = (_signed_key(key) or base == "abs_diff" or "absolute_difference" in base)
    if array.ndim == 3 and array.shape[-1] == 3 and channel_triptych:
        cmap = colormaps[cmap_name]
        components = [cmap(np.clip((array[..., channel] - low) / (high - low), 0, 1))[..., :3]
                      for channel in range(3)]
        shown = np.concatenate(components, axis=1)
    elif array.ndim == 2 or (array.ndim == 3 and array.shape[-1] == 1):
        if array.ndim == 3:
            array = array[..., 0]
        normalized = np.clip((array - low) / (high - low), 0, 1)
        shown = colormaps[cmap_name](normalized)[..., :3]
    elif array.ndim == 3 and array.shape[-1] == 3:
        if manifest is not None:
            array = _display_rgb(array, key, manifest, frame_metadata=frame_metadata)
        normalized = np.clip((array - low) / (high - low), 0, 1)
        shown = _gamma22(normalized) if gamma22 else normalized
    else:
        raise ValueError(f"unsupported display array shape for {key}: {array.shape}")
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(shown, 0, 1) * 255).astype(np.uint8)).save(target)


def _array_for_key(archive: Any, key: str) -> np.ndarray | None:
    base, channel = _split_channel_key(key)
    if base == "near_minus_scaled_far" and "signed_diff" in archive:
        value = -archive["signed_diff"]
        return value[..., channel] if channel is not None else value
    if base not in archive:
        return None
    value = archive[base]
    if channel is not None:
        if value.ndim != 3 or channel >= value.shape[-1]:
            return None
        return value[..., channel]
    return value


def _collect_condition_arrays(output: Path, conditions: dict[str, Any], display_keys: list[str]) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for mode in conditions:
        path = output / mode / "arrays.npz"
        with np.load(path, allow_pickle=False) as archive:
            arrays[mode] = {key: np.asarray(value).copy() for key, value in archive.items()}
            for key in display_keys:
                record = conditions[mode].setdefault("arrays", {})
                value = _array_for_key(archive, key)
                if value is not None:
                    record[key] = _stats(value)
    return arrays


def _make_range(key: str, arrays: dict[str, dict[str, np.ndarray]],
                current: dict[str, list[float]],
                manifest: dict[str, Any] | None = None) -> tuple[float, float]:
    fixed = _fixed_range(key)
    if fixed is not None:
        return fixed
    if key in {"near_linear_srgb", "far_linear_srgb"}:
        # Fixed exposure is shared by near/far images and every condition.
        return 0.0, 1.0
    base, _ = _split_channel_key(key)
    paired_rgb = {
        "near_compensated": "far_compensated", "far_compensated": "near_compensated",
        "near_whitebalanced": "far_whitebalanced", "far_whitebalanced": "near_whitebalanced",
        "near_input_unclipped": "far_input_unclipped", "far_input_unclipped": "near_input_unclipped",
    }.get(base)
    values = []
    for data in arrays.values():
        _, channel = _split_channel_key(key)
        paired_key = f"{paired_rgb}_c{channel}" if paired_rgb and channel is not None else paired_rgb
        keys = [key, paired_key] if paired_rgb else [key]
        for candidate in keys:
            if candidate is None:
                continue
            resolved = _array_for_key(data, candidate)
            if resolved is not None:
                value = np.asarray(resolved, dtype=np.float32)
                if manifest is not None and value.ndim == 3 and value.shape[-1] == 3:
                    value = _display_rgb(value, candidate, manifest)
                values.append(value.reshape(-1))
    if values:
        pooled = np.concatenate(values)
        if _signed_key(key):
            extent = float(np.percentile(np.abs(pooled), np.float32(99)))
            extent = max(extent, 1e-12)
            return -extent, extent
        return (float(np.percentile(pooled, np.float32(1))),
                float(np.percentile(pooled, np.float32(99))))
    return float(current.get(key, [0.0, 1.0])[0]), float(current.get(key, [0.0, 1.0])[1])


def _save_display_previews(output: Path, display_ranges: dict[str, list[float]],
                           conditions: dict[str, Any], display_keys: list[str],
                           condition_arrays: dict[str, dict[str, np.ndarray]],
                           manifest: dict[str, Any]) -> list[str]:
    gamma_files: list[str] = []
    for key in display_keys:
        base, channel = _split_channel_key(key)
        if base.startswith("prediction") and channel is not None:
            continue
        values = [value for data in condition_arrays.values()
                  if (value := _array_for_key(data, key)) is not None]
        if not values:
            continue
        current = display_ranges.get(key, [0.0, 1.0])
        low, high = _make_range(key, condition_arrays, display_ranges, manifest)
        low, high, cmap = _display_contract(key, (low, high))
        display_ranges[key] = [low, high]
        for mode, data in condition_arrays.items():
            value = _array_for_key(data, key)
            if value is not None:
                is_rgb = _is_rgb_visual(key, value)
                _render_array(value, key, output / mode / "display" / f"{key}.png", (low, high), cmap,
                              manifest=manifest)
                if base == "signed_diff" and _is_author_real_contract(manifest.get("model_contract", {})):
                    alias = key.replace("signed_diff", "near_minus_scaled_far", 1)
                    _render_array(-value, alias, output / mode / "display" / f"{alias}.png",
                                  (-high, -low), cmap)
                if is_rgb:
                    relative = Path(mode) / "display_gamma22" / f"{key}.png"
                    _render_array(value, key, output / relative, (low, high), cmap,
                                  manifest=manifest, gamma22=True)
                    gamma_files.append(relative.as_posix())
    return gamma_files


def _common_preview_group(key: str) -> str:
    base, _ = _split_channel_key(key)
    if base.startswith("frame_"):
        pieces = base.split("_", 3)
        if len(pieces) == 4 and pieces[1] in {"near", "far"}:
            base = pieces[3]
    elif base.startswith("near_") or base.startswith("far_"):
        base = base.split("_", 1)[1]
    return base


def _common_rgb_ranges(preview_values: list[tuple[str, np.ndarray, np.ndarray, str]],
                       display_ranges: dict[str, list[float]]) -> dict[str, tuple[float, float]]:
    pools: dict[str, list[np.ndarray]] = {}
    for _, _, shown, display_key in preview_values:
        if _is_rgb_visual(display_key, shown):
            sample = shown
            if sample.shape[0] * sample.shape[1] > 1_000_000:
                step = int(np.ceil(np.sqrt(sample.shape[0] * sample.shape[1] / 1_000_000)))
                sample = sample[::step, ::step]
            pools.setdefault(_common_preview_group(display_key), []).append(sample.reshape(-1))
    ranges = {}
    for stage, samples in pools.items():
        values = np.concatenate(samples)
        if "linear_srgb" in stage or "input_clipped" in stage:
            ranges[stage] = (0.0, 1.0)
        else:
            low = float(np.percentile(values, np.float32(1)))
            high = float(np.percentile(values, np.float32(99)))
            ranges[stage] = (low, max(high, low + 1e-12))
    return ranges


def _save_common_previews(output: Path, manifest: dict[str, Any],
                          display_ranges: dict[str, list[float]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    common_stats: dict[str, dict[str, Any]] = {}
    common_dir = output / "common_display"
    common_dir.mkdir(exist_ok=True)
    sources = [output / "source_arrays.npz", *sorted((output / "source_frames").glob("*.npz"))]
    preview_values: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    for path in sources:
        if not path.is_file():
            continue
        prefix = "" if path.name == "source_arrays.npz" else f"frame_{path.stem}_"
        with np.load(path, allow_pickle=False) as archive:
            values = {name: np.asarray(archive[name]) for name in archive.files}
        for name, value in values.items():
            key = f"{prefix}{name}"
            stats = _stats(value)
            common_stats[key] = stats
            is_bayer = "bayer" in name.lower()
            display_value = values.get("demosaic_rgb_selected_dtype") if is_bayer else value
            display_key = f"{prefix}demosaic_rgb_selected_dtype" if is_bayer else key
            if display_value is None:
                continue
            shown_value = np.asarray(display_value, dtype=np.float32)
            if _is_rgb_visual(display_key, shown_value):
                shown_value = _display_rgb(shown_value, display_key, manifest)
            preview_values.append((key, value, shown_value, display_key))
    pooled_rgb_ranges = _common_rgb_ranges(preview_values, display_ranges)
    gamma_files = []
    for key, value, shown_value, display_key in preview_values:
        stats = common_stats[key]
        if _is_rgb_visual(key, shown_value) or "bayer" in key.lower():
            group = _common_preview_group(display_key)
            low, high = pooled_rgb_ranges[group]
            # _render_array transforms camera-space pixels. Here values are already
            # transformed for pooled exposure, so pass a direct linear-RGB key.
            render_key = "linear_srgb"
            low, high, cmap = low, high, "gray"
            step = max(1, int(np.ceil(max(shown_value.shape[:2]) / 1200)))
            sample = shown_value[::step, ::step]
            _render_array(sample, render_key, common_dir / f"{key}.png", (low, high), cmap)
            alt = output / "common_display_gamma22" / f"{key}.png"
            _render_array(sample, render_key, alt, (low, high), cmap, gamma22=True)
            gamma_files.append(alt.relative_to(output).as_posix())
            display_ranges[key] = [low, high]
        elif _is_rgb_visual(key, value):
            proposed = display_ranges.get(key) or [stats["p01"], stats["p99"]]
            low, high, cmap = _display_contract(key, proposed)
            display_ranges[key] = [low, high]
            step = max(1, int(np.ceil(max(value.shape[:2]) / 1200))) if value.ndim >= 2 else 1
            _render_array(value[::step, ::step] if value.ndim >= 2 else value,
                          key, common_dir / f"{key}.png", (low, high), cmap)
        else:
            proposed = display_ranges.get(key) or [stats["p01"], stats["p99"]]
            low, high, cmap = _display_contract(key, proposed)
            display_ranges[key] = [low, high]
            step = max(1, int(np.ceil(max(value.shape[:2]) / 1200))) if value.ndim >= 2 else 1
            _render_array(value[::step, ::step] if value.ndim >= 2 else value,
                          key, common_dir / f"{key}.png", (low, high), cmap)
    return common_stats, gamma_files


def _display_configs(display_keys: list[str], display_ranges: dict[str, list[float]],
                     condition_arrays: dict[str, dict[str, np.ndarray]],
                     common_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from matplotlib import colormaps

    configs: dict[str, dict[str, Any]] = {}
    for key in sorted(set(display_keys) | set(common_stats)):
        low, high, cmap = _range_and_cmap(key, display_ranges)
        sample = next((_array_for_key(data, key) for data in condition_arrays.values()
                       if _array_for_key(data, key) is not None), None)
        if sample is None and key in common_stats:
            shape = common_stats[key]["shape"]
        else:
            shape = list(np.asarray(sample).shape) if sample is not None else []
        bayer_fallback = "bayer" in key.lower() and key in common_stats
        rgb = (((len(shape) == 3 and shape[-1] == 3)
                and not (_signed_key(key) or key == "abs_diff" or "absolute_difference" in key))
               or bayer_fallback)
        colors = [colormaps[cmap](index / 31) for index in range(32)]
        configs[key] = {
            "range": [low, high], "cmap": cmap, "rgb": rgb,
            "display_fallback": "same-frame WB-demosaic RGB; raw Bayer remains scalar NPZ/stat only" if bayer_fallback else None,
            "exposure": [low, high] if rgb else None,
            "stops": [f"rgb({round(c[0]*255)},{round(c[1]*255)},{round(c[2]*255)}) {i*100/31:.2f}%"
                      for i, c in enumerate(colors)],
        }
    configs.update(_prediction_component_configs())
    return configs


def _figure_gallery(output: Path, condition_arrays: dict[str, dict[str, np.ndarray]],
                    model_contract: dict[str, Any] | None = None,
                    *, filename_suffix: str = "") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    produced = []
    modes = [mode for mode in _CONDITIONS if mode in condition_arrays]
    if not modes:
        return produced
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 8, "figure.facecolor": "white", "axes.facecolor": "white"})
    feature_keys = {mode: _feature_key(condition_arrays[mode], model_contract) for mode in modes}
    if all(feature_keys[mode] is not None for mode in modes):
        feature_maps = {}
        for row, mode in enumerate(modes):
            features = np.asarray(condition_arrays[mode][feature_keys[mode]], dtype=np.float32)
            if features.ndim == 4 and features.shape[0] == 1:
                features = features[0]
            if features.ndim != 3:
                raise ValueError(f"expected 3D feature map for {mode}, got {features.shape}")
            expected_channels = _contract_channel_count(model_contract)
            if expected_channels is None:
                version = (model_contract or {}).get("feature_version")
                expected_channels = 21 if version == "raw_calibrated_v2" else (
                    33 if _is_author_real_contract(model_contract) else None)
            if expected_channels is not None and features.shape[0] == expected_channels:
                features = features.transpose(1, 2, 0)
            elif expected_channels is not None and features.shape[-1] != expected_channels:
                raise ValueError(f"recorded model expects {expected_channels} feature channels, got {features.shape}")
            elif expected_channels is None and features.shape[0] < features.shape[-1] and features.shape[0] in {21, 33}:
                features = features.transpose(1, 2, 0)
            feature_maps[mode] = features
        channel_count = next(iter(feature_maps.values())).shape[-1]
        if any(value.shape[-1] != channel_count for value in feature_maps.values()):
            raise ValueError("condition feature maps have different channel counts")
        rgb_groups, scalar_groups, feature_range = _feature_groups(channel_count, model_contract)
        if (channel_count == 33 and _is_author_real_contract(model_contract)
                and _identity_copy_features_match(feature_maps)):
            duplicate_indices = {2, 3, 7, 8}
            rgb_groups = [group for index, group in enumerate(rgb_groups) if index not in duplicate_indices]
        groups: list[tuple[str, tuple[int, ...], bool]] = [
            (title, channels, True) for title, channels in rgb_groups
        ] + [(title, (channel,), False) for title, channel in scalar_groups]
        bands = int(np.ceil(len(groups) / 5))
        if feature_range is None:
            pooled = np.concatenate([value.reshape(-1) for value in feature_maps.values()])
            feature_range = (float(np.percentile(pooled, np.float32(1))),
                             float(np.percentile(pooled, np.float32(99))))
        low, high = feature_range
        high = max(high, low + 1e-12)
        rows = bands * len(modes)
        fig = plt.figure(figsize=(14.5, max(5.5, bands * len(modes) * 2.05)))
        grid = fig.add_gridspec(rows, 6, width_ratios=[1] * 5 + [0.055],
                                left=0.075, right=0.975, bottom=0.04, top=0.96,
                                wspace=0.045, hspace=0.28)
        cmap = plt.get_cmap("gray")
        scalar_axes = []
        for band in range(bands):
            for condition_row, mode in enumerate(modes):
                grid_row = band * len(modes) + condition_row
                features = feature_maps[mode]
                start = band * 5
                for column, (title, channels, is_rgb) in enumerate(groups[start:start + 5]):
                    axis = fig.add_subplot(grid[grid_row, column])
                    if is_rgb:
                        rgb = np.clip((features[..., list(channels)] + 1.0) * 0.5, 0.0, 1.0)
                        axis.imshow(rgb, interpolation="nearest")
                    else:
                        scalar_axes.append(axis)
                        axis.imshow(np.clip((features[..., channels[0]] - low) / (high - low), 0, 1), cmap=cmap, vmin=0, vmax=1,
                                    interpolation="nearest")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if condition_row == 0:
                        axis.set_title(title)
                    if column == 0:
                        axis.set_ylabel(mode, rotation=0, ha="right", va="center")
        if scalar_axes:
            color_axis = fig.add_subplot(grid[:, 5])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=color_axis,
                         label=f"black 0 / white 1; stored [{low:.3g}, {high:.3g}]")
        path = output / f"features_gallery{filename_suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        produced.append(path.name)
    if all(_prediction_array(condition_arrays[mode]) is not None for mode in modes):
        titles = [record[1] for record in _PREDICTION_COMPONENTS]
        for gamma22 in (False, True):
            fig = plt.figure(figsize=(11.6, max(7.2, len(modes) * 2.35)))
            grid = fig.add_gridspec(len(modes), 5, width_ratios=[1, 1, 1, 1, 0.055],
                                    left=0.085, right=0.975, bottom=0.04, top=0.95,
                                    wspace=0.08, hspace=0.045)
            axes = np.empty((len(modes), 4), dtype=object)
            roughness_plot = None
            for row, mode in enumerate(modes):
                decoded = _prediction_decoded(_prediction_array(condition_arrays[mode]))
                for col, (key, _, start, end, is_rgb, gammaable) in enumerate(_PREDICTION_COMPONENTS):
                    axis = fig.add_subplot(grid[row, col])
                    axes[row, col] = axis
                    values = decoded[..., start:end]
                    if is_rgb:
                        shown = _gamma22(values) if gamma22 and gammaable else values
                        axis.imshow(shown, interpolation="nearest")
                    else:
                        roughness_plot = axis.imshow(values[..., 0], cmap="gray", vmin=0, vmax=1,
                                                     interpolation="nearest")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row == 0:
                        axis.set_title(titles[col])
                axes[row, 0].set_ylabel(mode, rotation=0, ha="right", va="center")
            if roughness_plot is not None:
                color_axis = fig.add_subplot(grid[:, 4])
                fig.colorbar(roughness_plot, cax=color_axis, label="roughness [0,1]")
            suffix = "_gamma22" if gamma22 else ""
            path = output / f"prediction_semantic_gallery{filename_suffix}{suffix}.png"
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            produced.append(path.name)
    # Shared row-wise color limits keep condition comparisons directly comparable.
    relation_keys = ("relation_raw", "relation_log_raw", "relation", "relation_log")
    if all(all(key in condition_arrays[mode] for key in relation_keys) for mode in modes):
        fig = plt.figure(figsize=(11.6, 9.0))
        grid = fig.add_gridspec(len(relation_keys), len(modes) + 1,
                                width_ratios=[1] * len(modes) + [0.05],
                                left=0.09, right=0.975, bottom=0.045, top=0.96,
                                wspace=0.08, hspace=0.09)
        row_labels = {"relation_raw": "raw relation", "relation_log_raw": "raw log ratio",
                      "relation": "relation [0,1]", "relation_log": "log relation [0,1]"}
        for row, key in enumerate(relation_keys):
            stacked = [np.asarray(condition_arrays[mode][key], dtype=np.float32).squeeze() for mode in modes]
            if key in {"relation", "relation_log"}:
                low, high = 0.0, 1.0
            else:
                low = min(float(np.percentile(value, np.float32(1))) for value in stacked)
                high = max(float(np.percentile(value, np.float32(99))) for value in stacked)
            high = max(high, low + 1e-12)
            plot = None
            for col, (mode, image) in enumerate(zip(modes, stacked)):
                axis = fig.add_subplot(grid[row, col])
                plot = axis.imshow(np.clip((image - low) / (high - low), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                axis.set_xticks([])
                axis.set_yticks([])
                if row == 0:
                    axis.set_title(mode)
                if col == 0:
                    axis.set_ylabel(row_labels[key], rotation=0, ha="right", va="center")
            if plot is not None:
                color_axis = fig.add_subplot(grid[row, len(modes)])
                fig.colorbar(plot, cax=color_axis, label=f"black 0 / white 1; stored [{low:.3g}, {high:.3g}]")
        path = output / f"relation_gallery{filename_suffix}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        produced.append(path.name)
    return produced


def _feature_channel_names(model_contract: dict[str, Any] | None) -> list[str]:
    contract = model_contract or {}
    if _is_author_real_contract(contract):
        return [
            "original near RGB · R", "original near RGB · G", "original near RGB · B",
            "original far RGB · R", "original far RGB · G", "original far RGB · B",
            "identity copy of original near · R", "identity copy of original near · G", "identity copy of original near · B",
            "identity copy of original far · R", "identity copy of original far · G", "identity copy of original far · B",
            "clipped gain-scaled original far · R", "clipped gain-scaled original far · G", "clipped gain-scaled original far · B",
            "log original near · R", "log original near · G", "log original near · B",
            "log original far · R", "log original far · G", "log original far · B",
            "log identity copy of original near · R", "log identity copy of original near · G", "log identity copy of original near · B",
            "log identity copy of original far · R", "log identity copy of original far · G", "log identity copy of original far · B",
            "log clipped gain-scaled original far · R", "log clipped gain-scaled original far · G", "log clipped gain-scaled original far · B",
            "relation", "log relation", "non-saturated mask",
        ]
    if contract.get("feature_version") == "raw_calibrated_v2":
        return list(_FEATURE_CHANNELS)
    channel_count = _contract_channel_count(contract)
    if channel_count:
        return [f"input channel {index}" for index in range(channel_count)]
    return list(_FEATURE_CHANNELS)


def _channel_table(model_contract: dict[str, Any] | None = None) -> str:
    names = _feature_channel_names(model_contract)
    rows = "".join(f"<tr><th>{index}</th><td>{html.escape(name)}</td><td>[-1, 1]</td></tr>"
                   for index, name in enumerate(names))
    if _is_author_real_contract(model_contract):
        rows += "<tr><td colspan=3>Feature gallery omits only the copied RGB/log-RGB tiles when their NPZ values exactly match the original slots; all 33 stored channels remain unchanged.</td></tr>"
    return f"<table><thead><tr><th>Channel</th><th>Meaning</th><th>Stored feature range</th></tr></thead><tbody>{rows}</tbody></table>"


def _prediction_table(model_contract: dict[str, Any]) -> str:
    order = model_contract.get("output_channel_order", model_contract.get("prediction_channel_order", list(_PREDICTION_CHANNELS)))
    rows = "".join(f"<tr><th>{index}</th><td>{html.escape(str(name))}</td><td>raw [-1,1]</td></tr>"
                   for index, name in enumerate(order))
    head = html.escape(str(model_contract.get("normal_head", "unspecified")))
    return (f"<p>checkpoint normal head: {head}. `xyz`는 XYZ 단위 길이를 보장하지 않습니다. "
            "Normal RGB preview는 `(n+1)/2` component encoding이며 normal vector를 정규화하지 않습니다.</p>"
            "<p>Diffuse/specular physical values는 display-only `(raw+1)/2` linear [0,1] 변환 후 gamma off 기본 RGB 또는 선택형 `x^(1/2.2)`로 표시합니다. "
            "Roughness는 `(raw+1)/2` scalar [0,1]입니다. raw 10 channels는 NPZ에 그대로 남습니다.</p>"
            f"<table><thead><tr><th>Channel</th><th>Meaning</th><th>Stored range</th></tr></thead><tbody>{rows}</tbody></table>")


def _range_and_cmap(key: str, ranges: dict[str, list[float]]) -> tuple[float, float, str]:
    low, high = ranges.get(key, [0.0, 1.0])
    return _display_contract(key, (low, high))


def _burst_summary_html(manifest: dict[str, Any]) -> str:
    source = str(manifest.get("source", ""))
    group = str(manifest.get("group") or Path(source).name or "unknown")
    frame_average = manifest.get("frame_average", {}) or {}
    selected = manifest.get("selected_frames", {}) or {}
    selected_gate = selected.get("averaging_gate", {}) if isinstance(selected, dict) else {}
    enabled = frame_average.get("enabled", selected.get("averaged") if isinstance(selected, dict) else None)
    near_count = frame_average.get("near_count")
    far_count = frame_average.get("far_count")
    if enabled is True:
        decision = f"정렬 프레임 FP32 평균 사용 (near {near_count or 5}, far {far_count or 5}장)"
    elif enabled is False or (isinstance(selected, dict) and selected.get("averaged") is False):
        decision = "원본 near_00/far_00 단일 쌍 fallback"
    else:
        decision = "프레임 선택 정보 없음"
    quantitative = frame_average.get("quantitative_gate_passed", selected_gate.get("quantitative_passed"))
    texture_status = frame_average.get("texture_review_status", selected_gate.get("texture_review_status"))
    root_review = manifest.get("root_visual_review", {}) or {}
    display_review = manifest.get("display_visual_review", {}) or {}
    display_review_record = display_review.get("root_review", {}) if isinstance(display_review, dict) else {}
    if quantitative is False:
        review_text = "수치 gate 실패: 평균 texture review는 해당 없음"
    elif texture_status in {"passed", "accepted", "reviewed_pass"}:
        review_text = "텍스처 정합 시각 검토 통과"
    elif texture_status in {"failed", "rejected", "reviewed_fail"}:
        review_text = "텍스처 정합 시각 검토 실패"
    elif texture_status == "not_applicable":
        review_text = "수치 gate 실패: 평균 texture review는 해당 없음"
    else:
        review_text = "텍스처 정합 시각 검토 대기"
    review_note = root_review.get("note") if isinstance(root_review, dict) else None
    display_review_status = display_review.get("status") if isinstance(display_review, dict) else None
    display_review_note = display_review_record.get("note") if isinstance(display_review_record, dict) else None
    display_preparation_sha = (display_review.get("display_preparation_manifest_sha256")
                               if isinstance(display_review, dict) else None)
    fallback = frame_average.get("fallback_reason")
    if not fallback and isinstance(selected, dict):
        fallback = selected_gate.get("reasons")
    raw_reasons = [fallback] if isinstance(fallback, str) else list(fallback) if isinstance(fallback, (list, tuple)) else [fallback] if fallback else []
    fallback_items = []
    for item in raw_reasons:
        if not item:
            continue
        if isinstance(item, str):
            try:
                parsed = ast.literal_eval(item)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                for side, reasons in parsed.items():
                    if isinstance(reasons, (list, tuple)):
                        fallback_items.extend(f"{side}: {reason}" for reason in reasons)
                    else:
                        fallback_items.append(f"{side}: {reasons}")
                continue
            fallback_items.append(item)
        else:
            fallback_items.append(json.dumps(item, ensure_ascii=False, allow_nan=False))
    fallback_items = list(dict.fromkeys(fallback_items))
    shown_reasons = fallback_items[:5]
    selected_names = selected.get("selected_filenames", {}) if isinstance(selected, dict) else {}
    if isinstance(selected_names, dict) and (selected_names.get("near") or selected_names.get("far")):
        names_text = "선택 파일: near=" + ", ".join(map(str, selected_names.get("near", [])))
        names_text += "; far=" + ", ".join(map(str, selected_names.get("far", [])))
    else:
        names_text = ""
    reason_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in shown_reasons)
    if len(fallback_items) > len(shown_reasons):
        reason_html += f"<li>{len(fallback_items) - len(shown_reasons)} additional reasons are in the manifest.</li>"
    review_note_html = f"<p>Root review: {html.escape(str(review_note))}</p>" if review_note else ""
    display_status_text = ("WB/matrix 표시 자료를 다시 확인했습니다" if display_review_status == "reviewed"
                           else "WB/matrix 표시 자료 검토 대기")
    display_review_html = (f'<p><b>Current display review:</b> {display_status_text}'
                           f'{f" · {html.escape(str(display_review_note))}" if display_review_note else ""}</p>')
    if display_preparation_sha:
        display_review_html += f'<p>Current display preparation manifest SHA-256: <code>{html.escape(str(display_preparation_sha))}</code></p>'
    links = []
    report_assets = manifest.get("report", {}) or {}
    if report_assets.get("preparation_review"):
        path = html.escape(str(report_assets["preparation_review"]), quote=True)
        links.append(f'<a href="{path}">프레임 정렬 및 crop 검토</a>')
    if report_assets.get("independent_formula_verification"):
        path = html.escape(str(report_assets["independent_formula_verification"]), quote=True)
        links.append(f'<a href="{path}">독립 수식 검증</a>')
    link_html = " · ".join(links)
    return (f'<p class="run-strip">{html.escape(decision)} · {html.escape(review_text)}</p>'
            f'<details class="run-details"><summary>Frame review and selection details</summary>'
            f'<p>{html.escape(review_text)}{f" · {html.escape(names_text)}" if names_text else ""}</p>'
            f'{review_note_html}{display_review_html}{f"<ul>{reason_html}</ul>" if reason_html else ""}'
            f'{f"<p>{link_html}</p>" if link_html else ""}</details>')


def _case_image(source: str | None, caption: str, alt: str, *, gamma_source: str | None = None) -> str:
    caption_html = html.escape(caption)
    if not source:
        return (f'<figure class="case-tile"><div class="case-image missing" aria-hidden="true"></div>'
                f'<figcaption>{caption_html}</figcaption></figure>')
    escaped_source = html.escape(str(source), quote=True)
    gamma_attrs = (f' data-linear-src="{escaped_source}"'
                   f' data-gamma-src="{html.escape(str(gamma_source), quote=True)}"') if gamma_source else ""
    return (f'<figure class="case-tile"><img class="case-image" src="{escaped_source}"{gamma_attrs} '
            f'alt="{html.escape(alt, quote=True)}"><figcaption>{caption_html}</figcaption></figure>')


def _case_row(label: str, tiles: list[str]) -> str:
    return (f'<div class="case-row"><div class="case-label">{html.escape(label)}</div>'
            f'<div class="case-tiles">{"".join(tiles)}</div></div>')


def _case_results_html(modes: list[str], conditions: dict[str, Any], display_configs: dict[str, dict[str, Any]],
                       comparison: dict[str, Any] | None, *, author_real: bool,
                       model_label: str) -> str:
    comparison_ready = isinstance(comparison, dict) and comparison.get("status") == "ready"
    comparison_assets = comparison.get("assets", {}) if comparison_ready else {}
    cases = []
    input_stages = (
        ("near_input_clipped", ("near_input_clipped", "near"), "near"),
        ("far_input_clipped", ("far_input_clipped", "far"), "far"),
        ("relation", ("relation", "relation_normalized"), "relation"),
        ("relation_log", ("relation_log", "log_relation", "relation_log_normalized"), "log relation"),
    )
    for mode in modes:
        record = conditions.get(mode, {}) or {}
        array_keys = record.get("arrays", {}) if isinstance(record, dict) else {}
        mode_tiles = []
        for fallback, candidates, caption in input_stages:
            key = next((candidate for candidate in candidates if candidate in array_keys), None)
            if key is None and fallback in display_configs:
                key = fallback
            source = display_configs.get(key, {}).get("scalar_file", f"{mode}/display/{key}.png") if key else None
            gamma_source = (f"{mode}/display_gamma22/{key}.png"
                            if key in {"near_input_clipped", "far_input_clipped"} else None)
            mode_tiles.append(_case_image(source, _stage_label(key) if key else caption, f"{mode} · {caption}", gamma_source=gamma_source))
        rows = [_case_row("Inputs", mode_tiles)]

        mode_assets = comparison_assets.get(mode, {}) if isinstance(comparison_assets, dict) else {}
        for branch, label in (("author_real_no_denoiser", "저자 real"),
                              ("best_render_preserved", "best_render")):
            if branch == "author_real_no_denoiser" and not author_real:
                continue
            if branch == "best_render_preserved" and not comparison_ready:
                continue
            row_tiles = []
            for key, title, _, _, _, gammaable in _PREDICTION_COMPONENTS:
                component = mode_assets.get(key, {}) if isinstance(mode_assets, dict) else {}
                if comparison_ready:
                    source = component.get(branch)
                    gamma_source = component.get(branch + "_gamma22") if gammaable else None
                elif branch == "author_real_no_denoiser":
                    source = f"{mode}/display/{key}.png"
                    gamma_source = f"{mode}/display_gamma22/{key}.png" if gammaable else None
                else:
                    source = None
                    gamma_source = None
                row_tiles.append(_case_image(source, _stage_label(key), f"{mode} · {label} · {title}",
                                             gamma_source=gamma_source))
            rows.append(_case_row(label, row_tiles))

        if not author_real and not comparison_ready:
            best_tiles = []
            for key, title, _, _, _, gammaable in _PREDICTION_COMPONENTS:
                source = f"{mode}/display/{key}.png"
                gamma_source = f"{mode}/display_gamma22/{key}.png" if gammaable else None
                best_tiles.append(_case_image(source, title.replace(" RGB", ""), f"{mode} · {model_label} · {title}",
                                              gamma_source=gamma_source))
            rows.append(_case_row(model_label or "Prediction", best_tiles))

        cases.append(f'<section class="case"><h2>{html.escape(mode)}</h2>{"".join(rows)}</section>')

    context = ""
    if comparison_ready:
        report_link = comparison.get("reference_report_relative")
        link = (f' · <a href="{html.escape(str(report_link), quote=True)}">best_render report</a>'
                if report_link else "")
        interpretation = html.escape(str(comparison.get("interpretation", "")))
        context = (f'<details class="comparison-context"><summary>저자 real · denoiser 제외 / best_render · 보존 결과 · Comparison notes{link}</summary>'
                   f'<p>{interpretation}</p></details>')
    return '<section class="results"><h2>Results</h2>' + "".join(cases) + context + '</section>'


def _make_html(manifest: dict[str, Any], display_keys: list[str], display_ranges: dict[str, list[float]],
               display_configs: dict[str, dict[str, Any]], common_stats: dict[str, dict[str, Any]], gallery_files: list[str],
               *, prepare_files: list[str] | None = None,
               formula_records: list | None = None, notation_note: str | None = None) -> str:
    source_conditions = manifest.get("conditions", {})
    active_conditions = _only_both(source_conditions)
    if not active_conditions:
        raise ValueError("active capture reports require the 'both' condition")
    manifest = dict(manifest)
    manifest["conditions"] = {mode: dict(record, arrays=dict(record.get("arrays", {})))
                              for mode, record in active_conditions.items()}
    source_report = manifest.get("report", {})
    report_assets = dict(source_report) if isinstance(source_report, dict) else {}
    report_assets["checkpoint_comparison"] = _both_only_comparison(report_assets.get("checkpoint_comparison"))
    manifest["report"] = report_assets
    gallery_files = [name for name in gallery_files if name in _BOTH_ONLY_GALLERIES]
    model_contract = manifest.get("model_contract", {}) or {}
    author_real = _is_author_real_contract(model_contract)
    if author_real:
        display_keys = list(display_keys)
        display_ranges = dict(display_ranges)
        display_configs = dict(display_configs)
        display_aliases = (("features_legacy33_rgb_preview", True),
                           ("log_original_6_near_rgb", False),
                           ("log_original_6_far_rgb", False))
        for key, gammaable in display_aliases:
            display_ranges.setdefault(key, [0.0, 1.0])
            if key not in display_keys:
                display_keys.append(key)
            display_configs[key] = {
                "range": [0.0, 1.0], "cmap": "gray", "rgb": True,
                "gammaable": gammaable, "stops": [],
            }
    model_label = "저자 real · denoiser 제외" if author_real else (
        "best_render · 보존 결과" if manifest.get("checkpoint_sha256") == _BEST_RENDER_SHA256 else "")
    all_keys = sorted(set(display_keys) | set(common_stats))
    if author_real:
        # Retain historical NPZ keys/statistics; expose only near-minus-far.
        for key in list(all_keys):
            if _split_channel_key(key)[0] != "signed_diff":
                continue
            alias = key.replace("signed_diff", "near_minus_scaled_far", 1)
            config = dict(display_configs.get(key, {"range": display_ranges[key], "rgb": False,
                                                    "cmap": "gray", "stops": []}))
            low, high = config["range"]
            config.update(range=[-high, -low], scalar_file=f"both/display/{alias}.png")
            display_configs[alias] = config
            all_keys[all_keys.index(key)] = alias
            for condition in manifest["conditions"].values():
                stats = condition.get("arrays", {}).get(key)
                if stats:
                    condition["arrays"][alias] = dict(stats, min=-stats["max"], max=-stats["min"],
                                                     mean=-stats["mean"], p01=-stats["p99"], p99=-stats["p01"])
    all_keys = [key for key in all_keys
                if not (_split_channel_key(key)[0].startswith("prediction")
                        and _split_channel_key(key)[1] is not None)]
    semantic_keys = [item[0] for item in _PREDICTION_COMPONENTS]
    selectable_keys = all_keys + [key for key in semantic_keys if key not in all_keys]
    if formula_records is None:
        formula_records = _AUTHOR_REAL_FORMULAS if author_real else _FORMULAS
    priority = {key: index for index, key in enumerate([record[-1] for record in formula_records] + semantic_keys)}
    selectable_keys.sort(key=lambda key: (priority.get(key, len(priority)),
                                         _split_channel_key(key)[0], _split_channel_key(key)[1] or 0))
    display_configs = {key: dict(value) for key, value in display_configs.items()}
    for config in display_configs.values():
        if not config.get("rgb"):
            config["source_range"] = list(config["range"])
            config["range"] = [0.0, 1.0]
            config["cmap"] = "gray"
            config["stops"] = ["rgb(0,0,0) 0%", "rgb(255,255,255) 100%"]
    semantic_titles = {record[0]: record[1] for record in _PREDICTION_COMPONENTS}
    options = "".join(
        f'<option value="{html.escape(key)}">{html.escape(_stage_label(key))}</option>'
        for key in selectable_keys)
    panels = []
    modes = [mode for mode in _CONDITIONS if mode in manifest.get("conditions", {})]
    for mode in _CONDITIONS:
        if mode not in manifest.get("conditions", {}):
            continue
        record = manifest["conditions"][mode]
        panels.append(
            f'<section class="condition"><h2>{html.escape(mode)}</h2>'
            f'<img class="stage" id="image_{mode}" alt="{html.escape(mode)} selected array">'
            f'<div id="bar_{mode}" class="colorbar"></div>'
            f'<h3 id="label_{mode}"></h3><p class="imgnote" id="meaning_{mode}"></p>'
            f'<details class="stage-info"><summary>저장된 원래 수치와 계산 정보</summary>'
            f'<p>{html.escape(str(record.get("formula", "")))}</p>'
            f'<p class="imgnote" id="note_{mode}"></p><pre id="stats_{mode}"></pre></details></section>')
    formula_rows = []
    for title, lhs, rhs, meaning, key in formula_records:
        gallery_link = (' <a href="#feature-gallery-details" onclick="document.getElementById(\'feature-gallery-details\').open=true">전체 feature gallery</a>'
                        if author_real and title == "33-channel input" else "")
        formula_rows.append(
            f'<tr><th>{html.escape(_stage_label(key))}</th><td><code>{html.escape(lhs)}</code><br><code>{html.escape(rhs)}</code></td>'
            f'<td>{html.escape(meaning)}</td><td><button onclick="selectStage(\'{html.escape(key, quote=True)}\')">보기</button>{gallery_link}</td></tr>')
    formulas = "".join(formula_rows)
    friendly_titles = {
        "features_gallery_both_only.png": "Feature maps · both",
        "prediction_semantic_gallery_both_only.png": "Prediction maps · both · gamma off",
        "prediction_semantic_gallery_both_only_gamma22.png": "Prediction maps · both · gamma 2.2",
        "relation_gallery_both_only.png": "Relation maps · both",
        "features_gallery.png": "Feature maps · 300 dpi",
        "prediction_semantic_gallery.png": "Prediction maps · gamma off",
        "prediction_semantic_gallery_gamma22.png": "Prediction maps · gamma 2.2",
        "relation_gallery.png": "Relation maps · common scales",
    }
    galleries = "".join(
        f'<figure class="gallery {"prediction-static-gallery gamma22-gallery" if path.endswith("_gamma22.png") else "prediction-static-gallery" if path.startswith("prediction_semantic_gallery") else ""}">'
        f'<a href="{html.escape(path, quote=True)}" target="_blank"><img src="{html.escape(path, quote=True)}" alt="{html.escape(friendly_titles.get(path, path))}"></a>'
        f'<figcaption>{html.escape(friendly_titles.get(path, path))}</figcaption></figure>'
        for path in gallery_files)
    run_conditions = {
        mode: {key: manifest["conditions"][mode].get(key) for key in
               ("formula", "arrays", "input_lower_clip_fraction", "input_upper_clip_fraction", "clipping")}
        for mode in modes
    }
    ui_manifest = {
        "conditions": run_conditions,
        "common_array_stats": common_stats,
        "display_ranges": display_ranges,
    }
    data = json.dumps(ui_manifest, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    stage_notes = {
        "linear_srgb": "저장 NPZ 값은 gamma 없는 linear sRGB입니다. 기본 표시에서는 값을 정규화한 뒤 그대로 RGB 바이트로 냅니다. 선택형 gamma는 display-only x^(1/2.2)입니다.",
        "input_clipped": "모델 입력 linear sRGB [0,1]입니다. 기본 RGB 표시 gamma는 꺼져 있고, 선택형 gamma만 x^(1/2.2)를 적용합니다.",
        "counts": "AHD/camera RGB count 표시 copy에 WB를 적용하고 camera→linear-sRGB matrix를 한 번 적용했습니다. raw NPZ/stats는 변경하지 않았고 shared RGB exposure를 씁니다.",
        "bayer": "원시 Bayer는 NPZ/stats에만 있습니다. 선택 시 같은 frame의 WB 적용 demosaic RGB와 camera→linear-sRGB matrix 결과를 표시하며 scalar Bayer raster로 오인하지 않도록 대체 표시입니다.",
        "signed": "부호 있는 스칼라 차이는 한 장으로, RGB 차이는 빨강·초록·파랑 순서로 표시합니다. 대칭 범위의 흑백 표시값 0.5가 차이 0입니다.",
    }
    note_json = json.dumps(stage_notes, ensure_ascii=False).replace("<", "\\u003c")
    scales_json = json.dumps(display_configs, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    report_assets = manifest.get("report", {}) or {}
    artifact_links = [("manifest", report_assets.get("manifest", "manifest.json")),
                      ("run verification", report_assets.get("verification", "verification.json"))]
    for field, label in (("preparation_review", "registration review"),
                         ("independent_formula_verification", "formula verification"),
                         ("both_only_gallery_generation", "both-only gallery record"),
                         ("active_condition_contract_snapshot", "active both-only contract"),
                         ("author_real_pipeline_snapshot", "author pipeline audit"),
                         ("html_generation_record", "HTML refresh record")):
        if report_assets.get(field):
            artifact_links.append((label, str(report_assets[field])))
    artifact_links_html = " · ".join(
        f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>' for label, path in artifact_links)
    group_label = str(manifest.get("group") or Path(str(manifest.get("source", ""))).name or "comparison")
    burst_summary = _burst_summary_html(manifest)
    color_transforms = manifest.get("color_transforms", {}) or {}
    transform_rows = []
    for side in ("near", "far"):
        transform = color_transforms.get(side, {}) if isinstance(color_transforms, dict) else {}
        if not isinstance(transform, dict):
            continue
        wb = transform.get("normalized_rgb_wb", transform.get("camera_wb"))
        matrix = transform.get("camera_to_linear_srgb")
        if wb is None and matrix is None:
            continue
        normalized_wb = np.asarray(wb, dtype=np.float32).reshape(-1)[:3] if wb is not None else None
        wb_diagonal = np.diag(normalized_wb).astype(np.float32).tolist() if normalized_wb is not None else None
        transform_rows.append(
            f'<tr><th>{side}</th><td>WB normalized RGB gains: <code>{html.escape(json.dumps(wb, ensure_ascii=False))}</code><br>'
            f'WB diagonal matrix (3×3): <code>{html.escape(json.dumps(wb_diagonal, ensure_ascii=False))}</code></td>'
            f'<td>camera→linear RGB matrix: <code>{html.escape(json.dumps(matrix, ensure_ascii=False))}</code></td></tr>')
    transform_table = ("<details class=technical><summary>Camera RGB display transform · WB diagonal and matrix</summary>"
                       "<p>Display copy only: camera RGB applies WB then the camera-to-linear-sRGB matrix; already-WB RGB applies the matrix once.</p>"
                       f"<table><thead><tr><th>Side</th><th>WB gains / diagonal</th><th>Camera→linear RGB matrix</th></tr></thead><tbody>{''.join(transform_rows)}</tbody></table></details>" if transform_rows else "")
    feature_count = _contract_channel_count(model_contract) or (33 if author_real else 21)
    channel_details = ("<details class=technical><summary>Feature and prediction channel order</summary>"
                       f"<h3>{feature_count} estimator input channels</h3>" + _channel_table(model_contract) +
                       "<h3>10 estimator output channels</h3>" + _prediction_table(model_contract) + "</details>")
    formula_details = ("<details class=technical><summary>Estimator formula, meaning, display range, and stage controls</summary>"
                       "<div class=formula-wrap><table><thead><tr><th>Stage</th><th>Formula</th><th>Meaning and display range</th><th></th></tr></thead>"
                       f"<tbody>{formulas}</tbody></table></div></details>")
    stage_buttons = []
    for title, _, _, _, key in formula_records:
        if key in selectable_keys:
            stage_buttons.append((_stage_label(key), key))
    stage_buttons.extend((_stage_label(key), key) for key in semantic_keys)
    stage_nav = "".join(
        f'<button type="button" class="stage-button" data-stage="{html.escape(key, quote=True)}" '
        f'aria-pressed="false" onclick="selectStage(\'{html.escape(key, quote=True)}\')">{html.escape(label)}</button>'
        for label, key in stage_buttons)
    initial_stage = "near_input_clipped" if "near_input_clipped" in selectable_keys else selectable_keys[0]
    case_results = _case_results_html(modes, manifest.get("conditions", {}), display_configs,
                                      report_assets.get("checkpoint_comparison"),
                                      author_real=author_real, model_label=model_label)
    figure_details = ("<details class=technical id=feature-gallery-details><summary>Semantic feature and prediction galleries</summary>" + galleries + "</details>")
    formula_links = artifact_links_html
    return f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fabric capture {html.escape(group_label)}</title>
<style>
*{{box-sizing:border-box}}
body{{font:15px Arial,Helvetica,sans-serif;margin:16px;background:#111;color:#ddd}}
main{{max-width:1800px;margin:auto;min-width:0}}
h1,h2,h3{{font-family:Arial,Helvetica,sans-serif}}
h1{{font-size:22px;margin:0 0 6px}}
h2{{font-size:17px;margin:6px 0}}
h3{{font-size:15px}}
.run-strip{{margin:4px 0 10px;color:#bbb}}
.workspace{{display:grid;grid-template-columns:240px minmax(0,1fr);gap:12px;align-items:start}}
.stage-nav{{position:sticky;top:12px;align-self:start;max-height:calc(100vh - 24px);overflow:auto;background:#181818;border:1px solid #555;padding:10px}}
.stage-nav label{{display:block;margin-bottom:8px}}
.stage-nav select{{display:block;width:100%;padding:7px;background:#222;color:#ddd;border:1px solid #555}}
.stage-buttons{{display:grid;gap:4px;margin-top:8px}}
button{{padding:6px 7px;text-align:left;border:1px solid #555;background:#222;color:#ddd;border-radius:3px;cursor:pointer}}
.stage-button[aria-pressed=true]{{background:#394b50;border-color:#aaa;font-weight:bold}}
.stage-results{{min-width:0}}
.conditions{{display:grid;grid-template-columns:minmax(0,512px);justify-content:start;gap:10px}}
section.condition{{min-width:0;max-width:512px;background:#181818;padding:8px;border:1px solid #555}}
.condition h2{{font-size:14px;margin:0 0 6px}}
.stage{{display:block;width:100%;height:auto;image-rendering:pixelated;background:#111}}
.colorbar{{margin-top:4px}}
.colorbar .bar{{height:12px;border:1px solid #aaa}}
.ticks{{display:flex;justify-content:space-between;font:11px Arial,sans-serif}}
.stage-info{{margin-top:6px}}
.stage-info>summary,.technical>summary{{cursor:pointer;color:#bbb;font-size:12px}}
.stage-info p,.stage-info pre{{font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:220px;overflow:auto}}
.results{{margin-top:16px}}
.case{{border-top:1px solid #555;padding:7px 0 12px}}
.case h2{{font-size:17px;margin:5px 0}}
.case-row{{display:flex;flex-wrap:nowrap;gap:12px;overflow-x:auto;align-items:flex-start;padding:5px 0 8px}}
.case-label{{flex:0 0 125px;position:sticky;left:0;z-index:1;background:#111;padding:5px 4px 0 0;font-size:12px}}
.case-tiles{{display:flex;flex:0 0 auto;gap:12px}}
.case-tile{{flex:0 0 145px;width:145px;margin:0}}
.case-image{{display:block;width:100%;height:auto;aspect-ratio:1;object-fit:contain;background:#181818;image-rendering:pixelated}}
.case-image.missing{{border:1px solid #555}}
.case-tile figcaption{{font-size:10px;margin-top:3px;color:#bbb;line-height:1.2}}
.comparison-context{{margin:4px 0 12px;font-size:12px}}
.comparison-context>summary{{cursor:pointer}}
.gamma22-gallery{{display:none}}
.gallery{{margin:12px 0}}
.gallery img{{display:block;width:100%;height:auto}}
.gallery figcaption{{font-size:11px;margin-top:3px;color:#bbb}}
.technical{{margin:12px 0;background:#181818;padding:8px;border:1px solid #555}}
.technical h3{{font-size:13px}}
.formula-wrap,.table-wrap{{overflow-x:auto;max-width:100%}}
table{{border-collapse:collapse;width:100%;background:#181818;margin:8px 0;font-size:12px}}
th,td{{border:1px solid #555;padding:5px;vertical-align:top;text-align:left}}
td code{{white-space:normal;overflow-wrap:anywhere}}
.technical pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.artifact-links{{margin:10px 0;font-size:12px;line-height:1.7}}
a{{color:#8ccaff}}
@media(min-width:900px){{.case-tile{{flex-basis:clamp(192px,16vw,256px);width:clamp(192px,16vw,256px)}}}}
@media(max-width:1050px){{
 body{{margin:8px}}
 .workspace{{grid-template-columns:1fr;gap:6px}}
 .stage-nav{{position:sticky;top:0;z-index:10;max-height:112px;overflow:hidden;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:3px;padding:5px}}
 .stage-nav label{{display:flex;align-items:center;gap:4px;margin:0;font-size:11px}}
 .stage-nav select{{width:100%;min-width:0;padding:4px;font-size:12px}}
 .stage-nav label:nth-of-type(2){{white-space:nowrap}}
 #range{{display:none;margin:0}}
 .stage-buttons{{grid-column:1/-1;display:flex;flex-wrap:nowrap;gap:4px;margin:0;overflow-x:auto;overflow-y:hidden;white-space:nowrap;padding-bottom:2px}}
 .stage-button{{flex:0 0 auto;padding:4px 7px;font-size:11px}}
 .conditions{{grid-template-columns:minmax(0,1fr);gap:4px}}
 section.condition{{max-width:100%;padding:4px}}
 .condition h2{{font-size:12px}}
}}
@media(max-width:520px){{.stage-buttons{{gap:3px}}.stage-button{{font-size:10px;padding:4px 6px}}.case-row{{gap:8px}}.case-tiles{{gap:8px}}.case-tile{{flex-basis:145px}}}}
@media(max-width:420px){{.conditions{{grid-template-columns:minmax(0,1fr)}}section.condition{{max-width:100%}}}}
</style><main>
<h1>Fabric capture · {html.escape(group_label)}{f' · {html.escape(model_label)}' if model_label else ''}</h1>
{burst_summary}
<details class="technical"><summary>논문 기호와 현재 계산의 대응</summary><p>{html.escape(notation_note if notation_note is not None else _PAPER_NOTATION_NOTE)}</p><a href="https://cgliwang.github.io/NFPLight/pdf/NFPLight.pdf">논문 Fig. 3 및 식 (4)–(5)</a></details>
<div class="workspace"><aside class="stage-nav"><label>보고 있는 계산 단계<select id="stage" onchange="showStage()">{options}</select></label><label><input id="gamma22" type="checkbox" onchange="toggleGammaDisplay()"> 화면 밝기 감마 2.2 (기본 꺼짐)</label><p id="range" class="small"></p><div class="stage-buttons" aria-label="Estimator stages">{stage_nav}</div></aside>
<div class="stage-results"><div class="conditions">{''.join(panels)}</div></div></div>
{case_results}
{formula_details}
{channel_details}
{transform_table}
{figure_details}
<details class="technical"><summary>Artifact links</summary><p class="artifact-links">{formula_links}</p></details>
<script>const manifest={data};const stageNotes={note_json};const scales={scales_json};
const stageLabels={json.dumps({key: _stage_label(key) for key in selectable_keys}, ensure_ascii=False)};
const stageDescriptions={json.dumps({key: lhs+' · '+rhs+' · '+meaning for _,lhs,rhs,meaning,key in formula_records}, ensure_ascii=False)};
const predictionStages={json.dumps({key:{'channels':list(range(start,end)),'label':title} for key,title,start,end,_,_ in _PREDICTION_COMPONENTS}, ensure_ascii=False)};
function selectedNote(key){{if(key==='features_legacy33_rgb_preview')return 'Display-only alias of the near-input RGB preview for feature channels 0:3 decoded to [0,1]; full 33-channel semantics are in the linked feature gallery.';if(key.startsWith('log_original_6_'))return 'Recorded log-transformed RGB triplet; display only, with no additional WB or gamma.';if(predictionStages[key])return 'Display decode: raw10 channels '+predictionStages[key].channels.join(',')+' → '+predictionStages[key].label+'; raw prediction and channel statistics unchanged.';if(key.includes('signed_diff')||key.includes('near_minus_scaled_far')||key.includes('_difference')||key.endsWith('_delta'))return stageNotes.signed;if(key.includes('bayer'))return stageNotes.bayer;if(key.includes('linear_srgb')||key.includes('input_clipped')||key.includes('input_unclipped')||key.includes('far_gained')||key.includes('far_time'))return stageNotes.linear_srgb;if(key.includes('counts')||key.includes('demosaic_rgb')||key.includes('compensated')||key.includes('whitebalanced')||key.includes('warp512')||key.includes('crop420')||key.includes('output256')||key.includes('libraw'))return stageNotes.counts;return 'FP32 numeric array; PNG is display-only.'}}
function selectStage(key){{document.getElementById('stage').value=key;document.querySelectorAll('.stage-button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.stage===key)));showStage();document.querySelector('.workspace').scrollIntoView({{block:'start'}})}}
function showStage(){{const key=document.getElementById('stage').value;document.querySelectorAll('.stage-button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.stage===key)));const common=Object.hasOwn(manifest.common_array_stats||{{}},key);const sc=scales[key]||{{range:(manifest.display_ranges||{{}})[key]||[0,1],rgb:false,cmap:'gray',stops:[]}};const gamma=Boolean(document.getElementById('gamma22').checked&&sc.rgb&&sc.gammaable!==false);const r=sc.range;document.getElementById('range').textContent=sc.rgb?'shared RGB exposure ['+r[0]+', '+r[1]+']; '+(gamma?'display-only x^(1/2.2)':'gamma off')+(sc.display_fallback?'; '+sc.display_fallback:''):'common scalar range ['+r[0]+', '+r[1]+']; '+sc.cmap+'; gamma/WB not applied.';for(const mode of {json.dumps(modes)}){{const image=document.getElementById('image_'+mode);if(!image)continue;const folder=common?(gamma?'common_display_gamma22':'common_display'):mode+(gamma?'/display_gamma22':'/display');image.src=(sc.scalar_file||folder+'/'+key+'.png');document.getElementById('label_'+mode).textContent=stageLabels[key];const source=sc.source_range||r;document.getElementById('meaning_'+mode).textContent=(stageDescriptions[key]||stageLabels[key])+(sc.rgb?' · 색상 영상입니다.':' · 검정 0 → 흰색 1; 표시값 = clip((원래 값 − '+source[0]+') / ('+source[1]+' − '+source[0]+'), 0, 1). 원래 수치는 아래에 보존됩니다.');const bar=document.getElementById('bar_'+mode);const note=document.getElementById('note_'+mode);if(sc.rgb){{bar.style.display='none'}}else{{bar.style.display='block';const middle=sc.cmap==='gray'&&r[0]<0&&r[1]>0?0:(r[0]+r[1])/2;bar.innerHTML='<div class="bar" style="background:linear-gradient(90deg,'+sc.stops.join(',')+')"></div><div class="ticks"><span>'+r[0]+'</span><span>'+middle+'</span><span>'+r[1]+'</span></div>'}}note.textContent=selectedNote(key);const rec=(manifest.conditions||{{}})[mode]||{{}};let shownStats;if(predictionStages[key]){{const a=rec.arrays||{{}};shownStats={{display_only:true,raw_channels:Object.fromEntries(predictionStages[key].channels.map(index=>['prediction_c'+index,a['prediction_c'+index]]))}}}}else{{const base=key.replace(/_c[0-9]+$/,'');const stats=common?(manifest.common_array_stats||{{}})[key]:((rec.arrays||{{}})[key]||(rec.arrays||{{}})[base]);shownStats={{array:stats,input_lower_clip_fraction:rec.input_lower_clip_fraction,input_upper_clip_fraction:rec.input_upper_clip_fraction,clipping:rec.clipping}}}}document.getElementById('stats_'+mode).textContent=JSON.stringify(shownStats,null,2)}}}}
function toggleGammaDisplay(){{const on=document.getElementById('gamma22').checked;document.querySelectorAll('.case-image[data-gamma-src]').forEach(image=>image.src=on?image.dataset.gammaSrc:image.dataset.linearSrc);document.querySelectorAll('.prediction-static-gallery.gamma22-gallery').forEach(x=>x.style.display=on?'block':'none');document.querySelectorAll('.prediction-static-gallery:not(.gamma22-gallery)').forEach(x=>x.style.display=on?'none':'block');showStage()}}
document.querySelectorAll('.stage-button').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.stage==={json.dumps(initial_stage)})));document.getElementById('stage').value={json.dumps(initial_stage)};toggleGammaDisplay();</script></main></html>"""


def write_prepare_review(output: Path, frame_records_by_side: dict[str, Any] | None = None,
                         *, group_label: str | None = None) -> dict[str, Any]:
    """Write marker/crop, alignment-difference, and display-only mean review assets.

    Each record contains ``index``, ``filename``, ``archive``, ``geometry`` and
    optional ``alignment``. Archives retain the source camera-RGB frame and a
    final aligned ``output256`` HWC array. No preview is an inference input.
    """
    output = Path(output)
    if frame_records_by_side is None:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        frame_records_by_side = manifest.get("geometry", {}).get("content", {}).get("per_frame", {})
    elif isinstance(frame_records_by_side.get("sides"), dict):
        frame_records_by_side = frame_records_by_side["sides"]
    preparation_manifest = frame_records_by_side if isinstance(frame_records_by_side, dict) else {}
    side_gates: dict[str, dict[str, Any]] = {}
    records = {}
    for side in ("near", "far"):
        side_value = frame_records_by_side.get(side, [])
        if isinstance(side_value, dict):
            side_gates[side] = side_value
            rows = side_value.get("frames", [])
        else:
            side_gates[side] = {}
            rows = side_value
        records[side] = [dict(row) for row in rows]
        for row in records[side]:
            row.setdefault("metadata_compatible", side_gates[side].get("metadata_compatible", True))
            metadata_list = side_gates[side].get("metadata", [])
            frame_index = row.get("index")
            if isinstance(metadata_list, list) and isinstance(frame_index, int) and 0 <= frame_index < len(metadata_list):
                row["_frame_metadata"] = metadata_list[frame_index]
    for side, rows in records.items():
        if not rows:
            records[side] = [{"side": side, "index": None, "filename": None,
                              "archive": None, "error": "no frame records were produced"}]
    frames: dict[str, list[tuple[dict[str, Any], np.ndarray | None, np.ndarray | None,
                                 np.ndarray | None, np.ndarray | None]]] = {}
    failed_records: list[dict[str, Any]] = []
    for side, rows in records.items():
        frames[side] = []
        for row in rows:
            rgb: np.ndarray | None = None
            crop: np.ndarray | None = None
            preview_rgb: np.ndarray | None = None
            preview_crop: np.ndarray | None = None
            archive_name = row.get("archive")
            path = output / str(archive_name) if archive_name else None
            try:
                if path is None or not path.is_file():
                    raise FileNotFoundError(f"frame archive missing: {archive_name or '(not recorded)'}")
                with np.load(path, allow_pickle=False) as archive:
                    if "output256" not in archive or "demosaic_rgb_selected_dtype" not in archive:
                        raise ValueError("archive lacks output256 or demosaic_rgb_selected_dtype")
                    rgb = np.asarray(archive["demosaic_rgb_selected_dtype"], dtype=np.float32).copy()
                    crop = np.asarray(archive["output256"], dtype=np.float32).copy()
                if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
                    raise ValueError("invalid camera-RGB sensor preview array")
                if crop.shape != (256, 256, 3) or not np.isfinite(crop).all():
                    raise ValueError("invalid aligned output256 crop")
                frame_metadata = row.get("_frame_metadata")
                frame_prefix = f"frame_{side}_{int(row.get('index') or 0):02d}_"
                preview_rgb = _display_rgb(rgb, frame_prefix + "demosaic_rgb_selected_dtype",
                                           preparation_manifest, frame_metadata=frame_metadata)
                preview_crop = _display_rgb(crop, frame_prefix + "output256",
                                            preparation_manifest, frame_metadata=frame_metadata)
            except (OSError, ValueError, KeyError, EOFError) as error:
                row["review_error"] = str(error)
                failed_records.append({"side": side, "index": row.get("index"), "error": str(error)})
                rgb, crop, preview_rgb, preview_crop = None, None, None, None
            frames[side].append((row, rgb, crop, preview_rgb, preview_crop))
    exposure_samples = [shown.reshape(-1, 3) for rows in frames.values()
                        for _, _, _, rgb, crop in rows
                        for shown in (rgb, crop) if shown is not None]
    if exposure_samples:
        pool = np.concatenate(exposure_samples, axis=0)
        common_rgb_exposure = (float(np.percentile(pool, np.float32(1))),
                               float(np.percentile(pool, np.float32(99))))
    else:
        common_rgb_exposure = (0.0, 1.0)
    exposure_span = max(common_rgb_exposure[1] - common_rgb_exposure[0], 1e-12)

    contact_files = []
    for side in ("near", "far"):
        rows = frames[side]
        width, tile, label_h = 256, 256, 28
        for gamma22 in (False, True):
            sheet = Image.new("RGB", (width * len(rows), (tile + label_h) * 2), "white")
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default()
            for column, (record, _, _, rgb, crop) in enumerate(rows):
                x = column * width
                if rgb is not None:
                    frame = np.clip((rgb - common_rgb_exposure[0]) / exposure_span, 0, 1)
                    if gamma22:
                        frame = _gamma22(frame)
                    overlay = Image.fromarray(np.rint(frame * 255).astype(np.uint8))
                    _draw_geometry_overlay(overlay, record)
                    overlay.thumbnail((width, tile), Image.Resampling.LANCZOS)
                    sheet.paste(overlay, (x, label_h + (tile - overlay.height) // 2))
                else:
                    draw.text((x + 4, label_h + 12), "WB-demosaic preview unavailable", fill="black", font=font)
                if crop is not None:
                    crop_preview = np.clip((crop - common_rgb_exposure[0]) / exposure_span, 0, 1)
                    if gamma22:
                        crop_preview = _gamma22(crop_preview)
                    sheet.paste(Image.fromarray(np.rint(crop_preview * 255).astype(np.uint8)), (x, tile + label_h * 2))
                else:
                    draw.rectangle((x, tile + label_h * 2, x + width - 1, tile + label_h * 2 + tile - 1), fill=(225, 228, 230))
                    draw.text((x + 4, tile + label_h * 2 + 12), "256 crop unavailable", fill="black", font=font)
                alignment = record.get("alignment", {})
                rms = alignment.get("residual_rms_px")
                rms_label = f"RMS={float(rms):.3f}px" if rms is not None else "RMS unavailable"
                frame_label = f"{side} {record.get('index', column):02d} {rms_label}"
                draw.text((x + 3, 3), frame_label, fill="black", font=font)
                draw.text((x + 3, tile + label_h + 3), "WB + camera→linear RGB · 256×256 · display only", fill="black", font=font)
                if record.get("review_error"):
                    draw.text((x + 3, tile + label_h + 17), str(record["review_error"])[:42], fill="red", font=font)
            suffix = "_gamma22" if gamma22 else ""
            path = output / f"prepare_{side}_contact_sheet{suffix}.png"
            sheet.save(path)
            if not gamma22:
                contact_files.append(path.name)

    difference_files = []
    for side in ("near", "far"):
        rows = frames[side]
        reference = rows[0][2] if rows else None
        diffs = [np.abs(crop - reference).mean(axis=-1, dtype=np.float32)
                 if crop is not None and reference is not None else None
                 for _, _, crop, _, _ in rows]
        valid_diffs = [diff for diff in diffs if diff is not None]
        extent = (max(float(np.percentile(np.concatenate([d.reshape(-1) for d in valid_diffs]), np.float32(99))), 1e-12)
                  if valid_diffs else 1.0)
        from matplotlib import colormaps
        for gamma22 in (False, True):
            sheet = Image.new("RGB", (256 * len(rows), 2 * 256 + 46), "white")
            draw = ImageDraw.Draw(sheet)
            for column, ((record, _, _, _, crop_preview), diff) in enumerate(zip(rows, diffs)):
                x = column * 256
                if crop_preview is not None:
                    preview = np.clip((crop_preview - common_rgb_exposure[0]) / exposure_span, 0, 1)
                    if gamma22:
                        preview = _gamma22(preview)
                    sheet.paste(Image.fromarray(np.rint(preview * 255).astype(np.uint8)), (x, 0))
                else:
                    draw.text((x + 4, 12), "WB-RGB crop unavailable", fill="black")
                if diff is not None:
                    colors = colormaps["gray"](np.clip(diff / extent, 0, 1))[..., :3]
                    sheet.paste(Image.fromarray(np.rint(colors * 255).astype(np.uint8)), (x, 256))
                else:
                    draw.rectangle((x, 256, x + 255, 511), fill=(225, 228, 230))
                    draw.text((x + 4, 268), "difference unavailable", fill="black")
                draw.text((x + 3, 3), f"frame {record.get('index', column):02d}", fill="white")
                draw.text((x + 3, 259), "|Δ camera-RGB counts| mean · no WB/gamma", fill="white")
            gradient = colormaps["gray"](np.linspace(0, 1, sheet.width, dtype=np.float32))[None, :, :3]
            bar = np.repeat(np.rint(gradient * 255).astype(np.uint8), 14, axis=0)
            sheet.paste(Image.fromarray(bar), (0, 512))
            draw = ImageDraw.Draw(sheet)
            draw.text((0, 528), "mean per-pixel abs camera-RGB count difference", fill="black")
            draw.text((sheet.width - 62, 528), f"0 .. {extent:.4g}", fill="black")
            suffix = "_gamma22" if gamma22 else ""
            path = output / f"prepare_{side}_alignment_difference{suffix}.png"
            sheet.save(path)
            if not gamma22:
                difference_files.append(path.name)

    gate_failures: list[str] = []
    if failed_records:
        gate_failures.append("one or more review arrays are missing or invalid")
    for side, rows in records.items():
        gate = side_gates[side]
        if len(rows) != 5:
            gate_failures.append(f"{side}: expected five frame records, got {len(rows)}")
        if gate.get("averaging_eligible") is not True:
            gate_failures.append(f"{side}: preprocessing averaging eligibility is false or unrecorded")
        if gate.get("metadata_compatible") is not True:
            gate_failures.append(f"{side}: burst metadata compatibility is false or unrecorded")
        for row in rows:
            index = row.get("index", "?")
            if row.get("review_error"):
                continue
            if row.get("error"):
                gate_failures.append(f"{side} frame {index}: preprocessing error")
            if row.get("alignment", {}).get("within_rms_limit") is not True:
                gate_failures.append(f"{side} frame {index}: alignment RMS gate failed or unrecorded")
            if row.get("metadata_compatible", gate.get("metadata_compatible")) is not True:
                gate_failures.append(f"{side} frame {index}: metadata compatibility failed or unrecorded")
            processing = row.get("processing", {})
            valid_fraction = processing.get("valid_fraction", processing.get("valid_intersection_fraction"))
            if processing.get("valid420_all") is not True or valid_fraction is None or float(valid_fraction) != 1.0:
                gate_failures.append(f"{side} frame {index}: full valid-crop support failed or unrecorded")
    gate_ok = not gate_failures
    mean_files = []
    if gate_ok and all(len(rows) == 5 for rows in records.values()):
        for side in ("near", "far"):
            rows = frames[side]
            stack = np.stack([crop for _, _, crop, _, _ in rows], axis=0).astype(np.float32, copy=False)
            average = stack.mean(axis=0, dtype=np.float32)
            record, _, first, _, _ = rows[0]
            metadata = record.get("_frame_metadata")
            preview_first = _display_rgb(first, f"frame_{side}_00_output256", preparation_manifest,
                                         frame_metadata=metadata)
            preview_average = _display_rgb(average, f"frame_{side}_00_output256", preparation_manifest,
                                           frame_metadata=metadata)
            low, high = common_rgb_exposure
            scale = max(high - low, 1e-12)
            for gamma22 in (False, True):
                first_display = np.clip((preview_first - low) / scale, 0, 1)
                mean_display = np.clip((preview_average - low) / scale, 0, 1)
                if gamma22:
                    first_display, mean_display = _gamma22(first_display), _gamma22(mean_display)
                comparison = Image.new("RGB", (512, 284), "white")
                comparison.paste(Image.fromarray(np.rint(first_display * 255).astype(np.uint8)), (0, 28))
                comparison.paste(Image.fromarray(np.rint(mean_display * 255).astype(np.uint8)), (256, 28))
                ImageDraw.Draw(comparison).text((4, 4), "frame 00 · WB + camera→linear RGB", fill="black", font=ImageFont.load_default())
                ImageDraw.Draw(comparison).text((260, 4), "FP32 mean candidate · display only", fill="black", font=ImageFont.load_default())
                suffix = "_gamma22" if gamma22 else ""
                path = output / f"prepare_{side}_frame00_vs_mean_display_only{suffix}.png"
                comparison.save(path)
                if not gamma22:
                    mean_files.append(path.name)

    names = contact_files + difference_files + mean_files
    escaped_label = html.escape(group_label or output.name)
    figures = []
    for name in names:
        stem = Path(name).stem
        gamma_name = f"{stem}_gamma22.png"
        gamma_path = output / gamma_name
        alt = gamma_name if gamma_path.is_file() else name
        side = "near" if name.startswith("prepare_near_") else "far"
        if "contact_sheet" in name:
            caption = f"{side} · five frames + marker footprint"
        elif "alignment_difference" in name:
            caption = f"{side} · aligned crops + scalar Δ"
        else:
            caption = f"{side} · frame 00 vs mean candidate · display only"
        figures.append(
            f'<figure class="native-review"><figcaption>{html.escape(caption)}</figcaption>'
            f'<div class="review-image-scroll"><a href="{html.escape(name, quote=True)}" target="_blank">'
            f'<img class="prepare-rgb-image" data-linear="{html.escape(name, quote=True)}" '
            f'data-gamma22="{html.escape(alt, quote=True)}" src="{html.escape(name, quote=True)}" '
            f'alt="{html.escape(caption, quote=True)}"></a></div></figure>')
    images = "".join(figures)
    records_json = json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False)
    transform_rows = []
    transform_manifest = {}
    for side in ("near", "far"):
        metadata = side_gates[side].get("metadata", [])
        metadata = metadata[0] if isinstance(metadata, list) and metadata else {}
        try:
            gains, matrix = _camera_transform(preparation_manifest, side, metadata)
            wb_diagonal = np.diag(gains).astype(np.float32).tolist()
            transform_manifest[side] = {"normalized_rgb_wb": gains.tolist(),
                                        "normalized_rgb_wb_diagonal": wb_diagonal,
                                        "camera_to_linear_srgb": matrix.tolist()}
            transform_rows.append(
                f'<tr><th>{side}</th><td>gains <code>{html.escape(json.dumps(gains.tolist()))}</code><br>'
                f'diagonal (3×3) <code>{html.escape(json.dumps(wb_diagonal))}</code></td>'
                f'<td><code>{html.escape(json.dumps(matrix.tolist()))}</code></td></tr>')
        except ValueError:
            continue
    transform_table = ("<details class=technical><summary>WB diagonal and camera matrix</summary>"
                       f"<table><thead><tr><th>Side</th><th>Normalized WB</th><th>Camera→linear-sRGB matrix</th></tr></thead><tbody>{''.join(transform_rows)}</tbody></table></details>" if transform_rows else "")
    review_path = output / "root_review.json"
    try:
        root_review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        root_review = {}
    if not gate_ok:
        gate_summary = "near_00 / far_00 fallback · averaging review not applicable"
    elif root_review.get("status") == "reviewed" and root_review.get("texture_registration_passed") is True:
        gate_summary = "numeric gate passed · texture review passed"
    elif root_review.get("status") == "reviewed" and root_review.get("texture_registration_passed") is False:
        gate_summary = "numeric gate passed · texture review failed"
    else:
        gate_summary = "numeric gate passed · texture review pending"
    gate_detail = "; ".join(gate_failures) or "numeric marker, metadata, and crop checks passed"
    display_contract = (
        "RGB photo previews use the side WB diagonal, then the camera→linear-sRGB matrix and shared exposure. "
        "The gamma toggle changes only RGB display copies (off by default; optional x^(1/2.2)). "
        "Each Δ panel shows WB/matrix RGB crops above a scalar mean absolute raw camera-RGB count difference below; "
        "the scalar uses gray with no WB or gamma. Bayer remains numeric-only; its photo view uses WB-demosaic RGB. "
        "All previews are display-only; NPZ arrays and inference inputs are unchanged.")
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><title>Frame registration review · {escaped_label}</title>
<style>*{{box-sizing:border-box}}body{{font:14px Arial,Helvetica,sans-serif;margin:10px;background:#111;color:#ddd}}main{{max-width:1600px;margin:auto;min-width:0}}h1{{font-size:20px;margin:0 0 8px}}.review-summary{{background:#181818;padding:8px;margin:0 0 8px}}.review-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}figure.native-review{{background:#181818;margin:0;padding:6px;min-width:0}}figure.native-review figcaption{{font-size:11px;margin-bottom:4px}}.review-image-scroll{{max-width:100%;overflow:auto}}.review-image-scroll img{{display:block;width:auto;max-width:none;height:auto;image-rendering:pixelated}}.technical{{background:#181818;margin:8px 0;padding:8px}}.technical>summary{{cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #555;padding:5px;text-align:left;vertical-align:top}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}.controls{{position:sticky;top:0;z-index:5;background:#111;padding:4px 0}}@media(max-width:650px){{.review-grid{{grid-template-columns:1fr}}}}</style><main><h1>Frame registration · {escaped_label}</h1>
<p class="review-summary"><b>{html.escape(gate_summary)}</b> · <span>5 near + 5 far frames</span></p>
<div class="controls"><label><input id="gamma22" type="checkbox" onchange="toggleGamma22()"> RGB display gamma x^(1/2.2) (off by default)</label></div>
<div class="review-grid">{images}</div>
<details class="technical"><summary>Display contract, gate detail, and WB transforms</summary><p>{html.escape(display_contract)}</p>
<p>Numeric gate: <code>{str(gate_ok).lower()}</code> · {html.escape(gate_detail)}</p>{transform_table}</details>
<details class="technical"><summary>Marker IDs and residuals</summary><pre>{html.escape(records_json)}</pre></details>
<script>function toggleGamma22(){{const on=document.getElementById('gamma22').checked;document.querySelectorAll('.prepare-rgb-image').forEach(image=>image.src=on?image.dataset.gamma22:image.dataset.linear)}}</script></main></html>"""
    review_path = output / "prepare_review.html"
    review_path.write_text(page, encoding="utf-8")
    all_display_files = names + [f"{Path(name).stem}_gamma22.png" for name in names
                                 if (output / f"{Path(name).stem}_gamma22.png").is_file()]
    return {"html": review_path.name, "contact_sheets": contact_files[0:2],
            "alignment_differences": difference_files, "mean_candidates": mean_files,
            "marker_metadata_crop_gate_passed": gate_ok, "gate_failures": gate_failures,
            "failed_frames": failed_records, "display_exposure": list(common_rgb_exposure),
            "display_gamma": {"default": "off", "optional": "x^(1/2.2)", "transform_order": transform_manifest,
                              "raw_arrays_changed": False,
                              "asset_sha256": {name: sha256_file(output / name) for name in all_display_files}},
            "visual_review_status": "awaiting_recorded_review" if gate_ok else "not_applicable_numeric_gate_failed",
            "group": group_label or output.name}


def _draw_geometry_overlay(image: Image.Image, record: dict[str, Any]) -> None:
    """Add marker IDs and the inverse homography crop footprint to a display copy."""
    geometry = record.get("geometry", {}) or {}
    alignment = record.get("alignment", {}) or {}
    H = alignment.get("homography_final", geometry.get("homography"))
    if H is not None:
        matrix = np.asarray(H, dtype=np.float64)
        if matrix.shape == (3, 3) and np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-12:
            corners = np.asarray([[46, 46, 1], [466, 46, 1], [466, 466, 1], [46, 466, 1]], dtype=np.float64).T
            source = np.linalg.inv(matrix) @ corners
            source = (source[:2] / source[2:3]).T
            raw_shape = record.get("raw_shape", geometry.get("raw_shape", [image.height, image.width]))
            sx, sy = image.width / max(1, int(raw_shape[-1])), image.height / max(1, int(raw_shape[0]))
            polygon = [(float(x * sx), float(y * sy)) for x, y in source]
            ImageDraw.Draw(image).polygon(polygon, outline=(255, 230, 0), width=2)
    marker_corners = geometry.get("marker_corners_raw", {})
    if isinstance(marker_corners, dict):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        raw_shape = record.get("raw_shape", geometry.get("raw_shape", [image.height, image.width]))
        sx = image.width / max(1, int(raw_shape[-1]))
        sy = image.height / max(1, int(raw_shape[0]))
        for marker_id, points in marker_corners.items():
            points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
            xy = [(float(x * sx), float(y * sy)) for x, y in points]
            if xy:
                draw.polygon(xy, outline=(0, 255, 255), width=2)
                draw.text((xy[0][0], xy[0][1]), str(marker_id), fill=(255, 255, 0), font=font)


def write_capture_report(output: Path, display_ranges: dict[str, list[float]],
                         conditions: dict[str, Any], display_keys: list[str]) -> None:
    """Build shared-scale previews, static galleries, manifest fields, and HTML."""
    if set(conditions) != {"both"}:
        raise ValueError("new capture reports accept only the active 'both' condition")
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition_arrays = _collect_condition_arrays(output, conditions, display_keys)
    manifest["conditions"] = conditions
    manifest["report_generation_sha256"] = sha256_file(Path(__file__))
    common_stats, common_gamma_files = _save_common_previews(output, manifest, display_ranges)
    condition_gamma_files = _save_display_previews(
        output, display_ranges, conditions, display_keys, condition_arrays, manifest)
    prediction_preview_files = _save_prediction_component_previews(output, condition_arrays)
    display_configs = _display_configs(display_keys, display_ranges, condition_arrays, common_stats)
    gallery_files = _figure_gallery(output, condition_arrays, manifest.get("model_contract", {}))
    # Current reports compare paper equations with author code, not MatSynth weights.
    comparison, comparison_files = None, []
    manifest["common_array_stats"] = common_stats
    manifest["display_ranges"] = display_ranges
    manifest["report"] = {
        "html": "report.html",
        "active_condition": "both",
        "static_figures": gallery_files,
        "display_only": True,
        "raw_prediction_arrays_unchanged": True,
        "color_scales": {key: {"range": value, "cmap": _display_contract(key, value)[2]}
                         for key, value in display_ranges.items()},
        "report_only_refresh": "regenerate_capture_report(output) reads existing manifest/NPZ files and does not run inference",
        "rgb_display": {"default_gamma": "off", "optional_gamma": "x^(1/2.2)",
                        "camera_rgb_order": "camera RGB × normalized WB, then camera-to-linear-sRGB matrix",
                        "already_whitebalanced": "camera-to-linear-sRGB matrix only",
                        "linear_srgb": "direct normalized RGB values",
                        "raw_bayer": "stats/NPZ only; photo view falls back to same-frame WB-demosaic RGB"},
    }
    if comparison is not None:
        manifest["report"]["checkpoint_comparison"] = comparison
    if (output / "prepare_review.html").is_file():
        manifest["report"]["preparation_review"] = "prepare_review.html"
    if (output / "independent_formula_verification.json").is_file():
        manifest["report"]["independent_formula_verification"] = "independent_formula_verification.json"
    if (output / "review_at_inference").is_dir() and not manifest.get("inference_review_snapshot"):
        manifest["inference_review_snapshot"] = "review_at_inference/"
    contract_path = Path("docs/CAPTURE_REQUIREMENTS.md")
    if contract_path.is_file():
        snapshot = output / "capture_requirements_snapshot.md"
        snapshot.write_text(contract_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest["report"]["requirements_snapshot"] = snapshot.name
        manifest["report"]["requirements_sha256"] = sha256_file(snapshot)
    execution_audit_path = Path("docs/DNG_PIPELINE_AUDIT_20261001.md")
    if execution_audit_path.is_file():
        snapshot = output / "dng_pipeline_audit_snapshot.md"
        snapshot.write_text(execution_audit_path.read_text(encoding="utf-8"), encoding="utf-8")
        manifest["report"]["execution_audit_snapshot"] = snapshot.name
        manifest["report"]["execution_audit_sha256"] = sha256_file(snapshot)
    active_contract_path = Path("docs/BOTH_ONLY_CAPTURE_CONTRACT_20261001.md")
    if active_contract_path.is_file():
        snapshot = output / "both_only_capture_contract_snapshot.md"
        snapshot.write_bytes(active_contract_path.read_bytes())
        manifest["report"]["active_condition_contract_snapshot"] = snapshot.name
        manifest["report"]["active_condition_contract_sha256"] = sha256_file(snapshot)
    display_asset_paths = sorted(set(
        common_gamma_files + condition_gamma_files + prediction_preview_files + gallery_files
        + comparison_files
        + ["common_display/" + key + ".png" for key in common_stats]
        + [f"{mode}/display/{key}.png" for mode in conditions for key in display_keys
           if (output / mode / "display" / f"{key}.png").is_file()]
    ))
    display_asset_hashes = {name: sha256_file(output / name) for name in display_asset_paths
                            if (output / name).is_file()}
    root_review_path = output / "root_review.json"
    root_review_record = None
    root_review_hash = None
    root_display_status = "awaiting_root_review"
    if root_review_path.is_file():
        root_review_hash = sha256_file(root_review_path)
        try:
            root_review_record = json.loads(root_review_path.read_text(encoding="utf-8"))
            inspected = root_review_record.get("inspected_assets", []) if isinstance(root_review_record, dict) else []
            evidence_valid = bool(inspected) and all(
                isinstance(item, dict) and item.get("path") and item.get("sha256")
                and (output / str(item["path"])).is_file()
                and sha256_file(output / str(item["path"])) == item["sha256"]
                for item in inspected)
            if root_review_record.get("status") == "reviewed" and evidence_valid:
                root_display_status = "reviewed"
        except (OSError, ValueError, TypeError):
            root_review_record = None
    previous_display_review = manifest.get("display_visual_review", {})
    if not isinstance(previous_display_review, dict):
        previous_display_review = {}
    current_preparation_assets = {}
    current_preparation_hash = None
    preparation_path = output / "preparation.json"
    if preparation_path.is_file():
        current_preparation_hash = sha256_file(preparation_path)
        try:
            current_preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            current_preparation_assets = (current_preparation.get("review_assets", {}) or {}).get("sha256", {})
        except (OSError, ValueError, TypeError):
            current_preparation_assets = {}
    manifest["display_visual_review"] = {
        "revision": "wb-matrix-linear-rgb-gamma22-v1",
        "status": root_display_status,
        "root_review_file": "root_review.json" if root_review_record is not None else None,
        "root_review_sha256": root_review_hash,
        "root_review": root_review_record,
        "asset_sha256": display_asset_hashes,
        "preparation_assets_sha256": current_preparation_assets,
        "display_preparation_manifest_sha256": current_preparation_hash,
        "default_gamma": "off",
        "optional_gamma": "x^(1/2.2)",
        "raw_numeric_arrays_changed": False,
        "prior_display_review": previous_display_review if previous_display_review.get("revision") != "wb-matrix-linear-rgb-gamma22-v1" else None,
    }
    _write_json(manifest_path, manifest)
    if (output / "verification.json").exists():
        verification = json.loads((output / "verification.json").read_text(encoding="utf-8"))
        verification["report_assets_generated"] = True
        verification["raw_prediction_arrays_unchanged"] = True
        _write_json(output / "verification.json", verification)
        manifest["verification"] = verification
        _write_json(manifest_path, manifest)
    (output / "report.html").write_text(
        _make_html(manifest, display_keys, display_ranges, display_configs, common_stats, gallery_files), encoding="utf-8")


def regenerate_capture_report(output: Path) -> Path:
    """Refresh the HTML view only; preserve historical multi-condition files."""
    return regenerate_capture_html_only(Path(output))


def _saved_common_shapes(output: Path, recorded: dict[str, Any]) -> dict[str, Any]:
    """Read NPZ headers only when author manifests omit common array metadata."""
    stats = dict(recorded)
    for path in [output / "source_arrays.npz", *sorted((output / "source_frames").glob("*.npz"))]:
        if not path.is_file():
            continue
        prefix = "" if path.name == "source_arrays.npz" else f"frame_{path.stem}_"
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".npy") or prefix + name[:-4] in stats:
                    continue
                with archive.open(name) as stream:
                    version = np.lib.format.read_magic(stream)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(stream)
                stats[prefix + name[:-4]] = {"shape": list(shape), "dtype": str(dtype),
                                           "metadata_only": True}
    return stats


def regenerate_scalar_display(output: Path) -> Path:
    """Render grayscale display copies from saved arrays; never execute inference.

    Source ranges are preserved. Black/white represent display coordinates 0/1,
    with the source-to-display transform printed beside every selected map.
    Historical stage PNGs, execution manifest, and numeric archives stay intact.
    """
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protected = [manifest_path, *sorted(output.rglob("*.npz"))]
    before = {str(path): sha256_file(path) for path in protected}
    conditions = _only_both(manifest.get("conditions"))
    if not conditions:
        raise ValueError("scalar display refresh requires saved both results")
    ranges = manifest["display_ranges"]
    common = _saved_common_shapes(output, manifest.get("common_array_stats", {}))
    configs = _display_configs_from_manifest(ranges, conditions, common)
    stage_files = {}
    sources = [("", output / "both" / "arrays.npz"),
               ("", output / "source_arrays.npz")]
    sources.extend((f"frame_{path.stem}_", path)
                   for path in sorted((output / "source_frames").glob("*.npz")))
    for prefix, path in sources:
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as archive:
            for key, config in configs.items():
                if config.get("rgb") or key in stage_files:
                    continue
                if prefix and not key.startswith(prefix):
                    continue
                name = key[len(prefix):] if prefix else key
                value = _array_for_key(archive, name)
                if value is None:
                    continue
                target = output / "scalar_display" / f"{key}.png"
                step = max(1, int(np.ceil(max(value.shape[:2]) / 1200))) if value.ndim >= 2 else 1
                _render_array(value[::step, ::step] if value.ndim >= 2 else value,
                              key, target, tuple(config["range"]), "gray")
                stage_files[key] = target.relative_to(output).as_posix()
    # Semantic roughness is decoded from the original prediction, just as RGB outputs are.
    with np.load(output / "both" / "arrays.npz", allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    prediction_files = _save_prediction_component_previews(output, {"both": arrays})
    stage_files["prediction_roughness"] = "both/display/prediction_roughness.png"
    comparison, comparison_files = _write_checkpoint_comparison(output, manifest, {"both": arrays})
    if comparison is not None:
        _write_json(output / "checkpoint_comparison_display.json", comparison)
    after = {str(path): sha256_file(path) for path in protected}
    if before != after:
        raise AssertionError("scalar display refresh changed numeric archives or execution manifest")
    asset_names = list(stage_files.values()) + prediction_files + comparison_files
    record = {
        "format": "nfplight.scalar-display.v1", "cmap": "gray", "display_range": [0, 1],
        "stage_files": stage_files,
        "source_ranges": {key: configs[key]["range"] for key in stage_files},
        "asset_sha256": {name: sha256_file(output / name) for name in asset_names},
        "protected_sha256_before": before, "protected_sha256_after": after,
        "numeric_arrays_changed": False, "inference_executed": False,
    }
    _write_json(output / "scalar_display_generation.json", record)
    regenerate_both_only_gallery(output)
    return regenerate_capture_html_only(output)


def _display_configs_from_manifest(display_ranges: dict[str, list[float]],
                                   conditions: dict[str, Any],
                                   common_stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Rebuild HTML selector metadata from saved statistics without opening NPZs."""
    from matplotlib import colormaps

    configs: dict[str, dict[str, Any]] = {}
    keys = sorted(set(display_ranges) | set(common_stats))
    for key in keys:
        low, high, cmap = _range_and_cmap(key, display_ranges)
        stats = common_stats.get(key)
        if not isinstance(stats, dict):
            base, _ = _split_channel_key(key)
            for condition in conditions.values():
                record = condition.get("arrays", {}) if isinstance(condition, dict) else {}
                stats = record.get(key, record.get(base)) if isinstance(record, dict) else None
                if isinstance(stats, dict):
                    break
        shape = stats.get("shape", []) if isinstance(stats, dict) else []
        bayer_fallback = False
        if "bayer" in key.lower():
            candidate = key.replace("bayer_u16", "demosaic_rgb_selected_dtype")
            candidate = candidate.replace("bayer_selected_dtype", "demosaic_rgb_selected_dtype")
            fallback_stats = common_stats.get(candidate, {})
            fallback_shape = fallback_stats.get("shape", []) if isinstance(fallback_stats, dict) else []
            bayer_fallback = len(fallback_shape) == 3 and fallback_shape[-1] == 3
        rgb = (((len(shape) == 3 and shape[-1] == 3)
                and not (_signed_key(key) or key == "abs_diff" or "absolute_difference" in key))
               or bayer_fallback)
        colors = [colormaps[cmap](index / 31) for index in range(32)]
        configs[key] = {
            "range": [low, high], "cmap": cmap, "rgb": rgb,
            "display_fallback": "same-frame WB-demosaic RGB; raw Bayer remains scalar NPZ/stat only" if bayer_fallback else None,
            "exposure": [low, high] if rgb else None,
            "stops": [f"rgb({round(c[0]*255)},{round(c[1]*255)},{round(c[2]*255)}) {i*100/31:.2f}%"
                      for i, c in enumerate(colors)],
        }
    configs.update(_prediction_component_configs())
    return configs


def regenerate_both_only_gallery(output: Path) -> Path:
    """Create new both-only display galleries from the saved both NPZ, preserving old PNGs."""
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = _only_both(manifest.get("conditions"))
    if not conditions:
        raise ValueError("gallery refresh requires a saved 'both' condition")
    npz_path = output / "both" / "arrays.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    numeric_sha_before = sha256_file(npz_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        feature_key = _feature_key({key: None for key in archive.files}, manifest.get("model_contract", {}))
        if feature_key is None:
            raise ValueError("saved both condition has no recognized feature tensor")
        condition_arrays = {"both": {key: np.asarray(value).copy() for key, value in archive.items()}}
    if feature_key not in condition_arrays["both"]:
        raise KeyError(f"{feature_key} is missing from {npz_path}")

    target_names = [
        "features_gallery_both_only.png",
        "prediction_semantic_gallery_both_only.png",
        "prediction_semantic_gallery_both_only_gamma22.png",
        "relation_gallery_both_only.png",
    ]
    prior_record_path = output / "both_only_gallery_generation.json"
    prior_assets: dict[str, str] = {}
    if prior_record_path.is_file():
        try:
            prior = json.loads(prior_record_path.read_text(encoding="utf-8"))
            prior_assets = prior.get("asset_sha256", {}) if isinstance(prior, dict) else {}
        except (OSError, ValueError, TypeError):
            prior_assets = {}
    original_png_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.rglob("*.png") if path.is_file()
    }



    for name in target_names:
        target = output / name
        if target.exists() and prior_assets.get(name) != sha256_file(target):
            raise FileExistsError(f"refusing to overwrite an untracked display image: {target}")

    produced = _figure_gallery(output, condition_arrays, manifest.get("model_contract", {}),
                               filename_suffix="_both_only")
    if "features_gallery_both_only.png" not in produced:
        raise ValueError("both-only feature gallery was not generated")
    numeric_sha_after = sha256_file(npz_path)
    if numeric_sha_before != numeric_sha_after or manifest_sha != sha256_file(manifest_path):
        raise AssertionError("both-only gallery refresh changed numeric data or execution manifest")
    display_hashes_after = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in output.rglob("*.png") if path.is_file()
    }
    old_assets_unchanged = all(display_hashes_after.get(name) == digest
                               for name, digest in original_png_hashes.items()
                               if name not in target_names)
    if not old_assets_unchanged:
        raise AssertionError("both-only gallery refresh changed a pre-existing PNG")
    asset_hashes = {name: sha256_file(output / name) for name in produced}
    gallery_record = {
        "format": "nfplight.fabric-capture-both-only-gallery.v1",
        "active_condition": "both",
        "gallery_files": produced,
        "asset_sha256": asset_hashes,
        "html_generator_sha256": sha256_file(Path(__file__)),
        "manifest_sha256_before": manifest_sha,
        "manifest_sha256_after": sha256_file(manifest_path),
        "numeric_npz_sha256_before": numeric_sha_before,
        "numeric_npz_sha256_after": numeric_sha_after,
        "numeric_archives_unchanged": numeric_sha_before == numeric_sha_after,
        "preexisting_pngs_unchanged": old_assets_unchanged,
        "numeric_arrays_changed": False,
        "inference_executed": False,
    }
    _write_json(prior_record_path, gallery_record)
    return output / "features_gallery_both_only.png"


def regenerate_relation_difference_display(output: Path) -> Path:
    """Refresh near-minus-far display aliases without changing archived computation."""
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not _is_author_real_contract(manifest.get("model_contract", {})):
        raise ValueError("difference display refresh requires an author-real report")
    archive_path = output / "both" / "arrays.npz"
    protected = {str(path): sha256_file(path) for path in (manifest_path, archive_path)}
    assets = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for key, value_range in manifest["display_ranges"].items():
            if _split_channel_key(key)[0] != "signed_diff":
                continue
            alias = key.replace("signed_diff", "near_minus_scaled_far", 1)
            value = _array_for_key(archive, alias)
            low, high = value_range
            target = output / "both" / "display" / f"{alias}.png"
            _render_array(value, alias, target, (-high, -low), "gray")
            assets[target.relative_to(output).as_posix()] = sha256_file(target)
    if protected != {path: sha256_file(Path(path)) for path in protected}:
        raise AssertionError("difference display refresh changed archived computation")
    _write_json(output / "relation_difference_display.json", {
        "formula": "I_N - clip(K I_F,0,1)", "source": "-signed_diff (author RGB trace)",
        "asset_sha256": assets, "protected_sha256": protected,
        "numeric_arrays_changed": False, "inference_executed": False,
    })
    return regenerate_capture_html_only(output, include_preserved_checkpoint=False)


def regenerate_capture_html_only(output: Path, *, include_preserved_checkpoint: bool = True) -> Path:
    """Regenerate only report.html from saved manifest/statistics; never read arrays or render PNGs.

    A sidecar records the HTML/generator hashes and the unchanged execution manifest.
    Existing PNGs and NPZ archives are untouched; NPZ checksums are compared as bytes.
    """
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest_sha_before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_conditions = manifest.get("conditions")
    display_ranges = manifest.get("display_ranges")
    common_stats = _saved_common_shapes(output, manifest.get("common_array_stats", {}))
    report_assets = manifest.get("report", {})
    conditions = _only_both(source_conditions)
    if not conditions or not isinstance(display_ranges, dict):
        raise ValueError("manifest lacks saved conditions or display_ranges")
    # New author-real manifests retain per-condition array metadata and display
    # ranges, but do not duplicate the older common_array_stats summary block.
    if not isinstance(common_stats, dict):
        common_stats = {}
    if not isinstance(report_assets, dict):
        raise ValueError("manifest report record is unavailable")
    report_assets = dict(report_assets)
    manifest = dict(manifest)
    manifest["conditions"] = conditions
    # Some runner-authored manifests omit a report block; keep links added by
    # this HTML-only renderer available to the template's artifact-link list.
    if (output / "prepare_review.html").is_file():
        report_assets.setdefault("preparation_review", "prepare_review.html")
    if (output / "independent_formula_verification.json").is_file():
        report_assets.setdefault("independent_formula_verification", "independent_formula_verification.json")
    manifest["report"] = report_assets
    author_real = _is_author_real_contract(manifest.get("model_contract", {}))
    report_assets["static_figures"] = [
        name for name in _BOTH_ONLY_GALLERIES if (output / name).is_file()
    ]
    contract_source = Path("docs/BOTH_ONLY_CAPTURE_CONTRACT_20261001.md")
    contract_snapshot_sha = None
    if contract_source.is_file():
        contract_name = "both_only_capture_contract_snapshot.md"
        contract_snapshot_path = output / contract_name
        contract_snapshot_path.write_bytes(contract_source.read_bytes())
        report_assets["active_condition_contract_snapshot"] = contract_name
        contract_snapshot_sha = sha256_file(contract_snapshot_path)
    # Keep historical snapshots byte-for-byte, but do not link obsolete run
    # contracts as the current condition policy.
    report_assets.pop("requirements_snapshot", None)
    if not author_real:
        report_assets.pop("execution_audit_snapshot", None)
    comparison_path = output / "checkpoint_comparison_display.json"
    comparison_sha = None
    if not include_preserved_checkpoint:
        report_assets["checkpoint_comparison"] = None
    elif author_real and comparison_path.is_file():
        comparison_sha = sha256_file(comparison_path)
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if isinstance(comparison, dict) and comparison.get("status") == "ready":
            # Display-only comparison provenance is separate from the immutable
            # inference manifest and is consumed only by this HTML copy.
            report_assets["checkpoint_comparison"] = _both_only_comparison(comparison)
    else:
        report_assets["checkpoint_comparison"] = _both_only_comparison(
            report_assets.get("checkpoint_comparison"))
    display_keys = sorted(display_ranges)
    display_configs = _display_configs_from_manifest(display_ranges, conditions, common_stats)
    gallery_files = [name for name in report_assets["static_figures"] if (output / name).is_file()]
    preparation_sha = None
    preparation_assets_sha = {}
    preparation_path = output / "preparation.json"
    if preparation_path.is_file():
        preparation_sha = sha256_file(preparation_path)
        try:
            preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
            preparation_assets_sha = (preparation.get("review_assets", {}) or {}).get("sha256", {})
        except (OSError, ValueError, TypeError):
            preparation_assets_sha = {}
        display_review = manifest.get("display_visual_review", {})
        if isinstance(display_review, dict):
            # Current display provenance is refreshed in the HTML copy only; the
            # on-disk inference manifest and its historical fields remain untouched.
            display_review["display_preparation_manifest_sha256"] = preparation_sha
            display_review["preparation_assets_sha256"] = preparation_assets_sha
    feature_record_path = output / "both_only_gallery_generation.json"
    feature_record = {}
    if feature_record_path.is_file():
        try:
            feature_record = json.loads(feature_record_path.read_text(encoding="utf-8"))
            report_assets["both_only_gallery_generation"] = feature_record_path.name
        except (OSError, ValueError):
            feature_record = {}
    audit_snapshot = None
    audit_snapshot_sha = None
    if author_real:
        audit_source = Path("docs/DNG_AUTHOR_REAL_PIPELINE_20261001.md")
        if not audit_source.is_file():
            raise FileNotFoundError(f"author real pipeline audit is unavailable: {audit_source}")
        audit_snapshot = "author_real_pipeline_snapshot.md"
        snapshot_path = output / audit_snapshot
        snapshot_path.write_bytes(audit_source.read_bytes())
        audit_snapshot_sha = sha256_file(snapshot_path)
        # The historical execution manifest remains untouched; this in-memory
        # report copy links to the active author-real contract snapshot.
        report_assets["execution_audit_snapshot"] = audit_snapshot
        report_assets["author_real_pipeline_snapshot"] = audit_snapshot
        report_assets["execution_audit_sha256"] = audit_snapshot_sha
    scalar_record_path = output / "scalar_display_generation.json"
    if scalar_record_path.is_file():
        scalar_record = json.loads(scalar_record_path.read_text(encoding="utf-8"))
        for key, path in scalar_record["stage_files"].items():
            if key in display_configs:
                display_configs[key]["scalar_file"] = path
        report_assets["scalar_display_generation"] = scalar_record_path.name
    # This path is added to the in-memory copy only; the execution manifest stays byte-identical.
    report_assets["html_generation_record"] = "report_html_generation.json"
    report_html = output / "report.html"
    report_html.write_text(_make_html(manifest, display_keys, display_ranges, display_configs,
                                      common_stats, gallery_files), encoding="utf-8")
    checkpoint_record = manifest.get("checkpoint")
    checkpoint_sha256 = manifest.get("checkpoint_sha256")
    if checkpoint_sha256 is None and isinstance(checkpoint_record, dict):
        checkpoint_sha256 = checkpoint_record.get("sha256")
    numeric_hashes = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*.npz")) if path.is_file()
    }
    record = {
        "format": "nfplight.fabric-capture-html-refresh.v2",
        "html_file": report_html.name,
        "html_sha256": sha256_file(report_html),
        "html_generator_sha256": sha256_file(Path(__file__)),
        "manifest_sha256_before": manifest_sha_before,
        "manifest_sha256_after": sha256_file(manifest_path),
        "preparation_manifest_sha256_current": preparation_sha,
        "preparation_review_assets_sha256_current": preparation_assets_sha,
        "checkpoint_comparison_display_sha256_current": comparison_sha,
        "active_condition": "both",
        "active_condition_contract_snapshot": "both_only_capture_contract_snapshot.md" if contract_snapshot_sha else None,
        "active_condition_contract_snapshot_sha256": contract_snapshot_sha,
        "author_real_pipeline_snapshot": audit_snapshot,
        "author_real_pipeline_snapshot_sha256": audit_snapshot_sha,
        "feature_gallery_file": "features_gallery_both_only.png" if (output / "features_gallery_both_only.png").is_file() else None,
        "feature_gallery_sha256": sha256_file(output / "features_gallery_both_only.png") if (output / "features_gallery_both_only.png").is_file() else None,
        "both_only_gallery_generation_sha256": sha256_file(feature_record_path) if feature_record else None,
        "numeric_npz_sha256": numeric_hashes,
        "numeric_archives_unchanged": bool(feature_record.get("numeric_archives_unchanged", True)),
        "inference_executed": False,
        "png_assets_regenerated": False,
        "numeric_arrays_read": False,
        "numeric_arrays_changed": False,
        "execution_manifest_changed": False,
    }
    _write_json(output / "report_html_generation.json", record)
    return report_html


def regenerate_feature_gallery_only(output: Path) -> Path:
    """Compatibility alias for the saved both-only gallery refresh."""
    return regenerate_both_only_gallery(output)


def regenerate_capture_gallery_and_html(output: Path) -> Path:
    """Refresh only new both-only gallery files and HTML from existing arrays."""
    output = Path(output)
    regenerate_both_only_gallery(output)
    return regenerate_capture_html_only(output)


def regenerate_prepare_html_only(output: Path) -> Path:
    """Refresh compact preparation HTML using saved metadata and PNGs only."""
    output = Path(output)
    preparation_path = output / "preparation.json"
    html_path = output / "prepare_review.html"
    preparation_before = sha256_file(preparation_path)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    assets = preparation.setdefault("review_assets", {})
    old_asset_hashes = dict(assets.get("sha256", {})) if isinstance(assets, dict) else {}
    image_names = []
    for field in ("contact_sheets", "alignment_differences", "mean_candidates"):
        values = assets.get(field, []) if isinstance(assets, dict) else []
        if isinstance(values, list):
            for value in values:
                if isinstance(value, str) and Path(value).name == value and (output / value).is_file():
                    image_names.append(value)
    if not image_names:
        for pattern in ("prepare_*_contact_sheet.png", "prepare_*_alignment_difference.png",
                        "prepare_*_frame00_vs_mean_display_only.png"):
            image_names.extend(path.name for path in sorted(output.glob(pattern)))
    image_names = list(dict.fromkeys(image_names))
    display_hashes_before = {name: sha256_file(output / name) for name in image_names}

    figures = []
    for name in image_names:
        side = "near" if name.startswith("prepare_near_") else "far"
        if "contact_sheet" in name:
            caption = f"{side} · five frames + marker footprint"
        elif "alignment_difference" in name:
            caption = f"{side} · aligned crops + scalar Δ"
        else:
            caption = f"{side} · frame 00 vs mean candidate · display only"
        gamma_name = f"{Path(name).stem}_gamma22.png"
        if not (output / gamma_name).is_file():
            gamma_name = name
        figures.append(
            f'<figure class="native-review"><figcaption>{html.escape(caption)}</figcaption>'
            f'<div class="review-image-scroll"><a href="{html.escape(name, quote=True)}" target="_blank">'
            f'<img class="prepare-rgb-image" src="{html.escape(name, quote=True)}" '
            f'data-linear="{html.escape(name, quote=True)}" data-gamma22="{html.escape(gamma_name, quote=True)}" '
            f'alt="{html.escape(caption, quote=True)}"></a></div></figure>')

    sides = preparation.get("sides", {}) if isinstance(preparation.get("sides", {}), dict) else {}
    transform_rows = []
    for side in ("near", "far"):
        side_record = sides.get(side, {})
        metadata = side_record.get("metadata", []) if isinstance(side_record, dict) else []
        metadata = metadata[0] if isinstance(metadata, list) and metadata else {}
        try:
            gains, matrix = _camera_transform(preparation, side, metadata)
        except ValueError:
            continue
        diagonal = np.diag(gains).astype(np.float32).tolist()
        transform_rows.append(
            f'<tr><th>{side}</th><td><code>{html.escape(json.dumps(gains.tolist()))}</code><br>'
            f'WB diagonal matrix (3×3): <code>{html.escape(json.dumps(diagonal))}</code></td>'
            f'<td><code>{html.escape(json.dumps(matrix.tolist()))}</code></td></tr>')
    transform_table = (
        "<details><summary>WB diagonal and camera matrix</summary>"
        "<p>Display copy only: camera RGB × normalized WB diagonal, then camera→linear-sRGB matrix.</p>"
        f"<table><thead><tr><th>Side</th><th>WB gains / diagonal</th><th>Camera→linear-sRGB matrix</th></tr></thead><tbody>{''.join(transform_rows)}</tbody></table></details>"
        if transform_rows else "")

    gate = assets.get("marker_metadata_crop_gate_passed") if isinstance(assets, dict) else None
    averaging = preparation.get("averaging", {})
    if gate is None and isinstance(averaging, dict):
        gate = averaging.get("quantitative_eligible")
    try:
        root_review = json.loads((output / "root_review.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        root_review = {}
    if gate is not True:
        summary = "near_00 / far_00 fallback · averaging review not applicable"
    elif root_review.get("status") == "reviewed" and root_review.get("texture_registration_passed") is True:
        summary = "numeric gate passed · texture review passed"
    elif root_review.get("status") == "reviewed" and root_review.get("texture_registration_passed") is False:
        summary = "numeric gate passed · texture review failed"
    else:
        summary = "numeric gate passed · texture review pending"

    side_details = {}
    for side in ("near", "far"):
        record = sides.get(side, {})
        if isinstance(record, dict):
            side_details[side] = {key: record.get(key) for key in
                                  ("metadata_compatible", "metadata_mismatches", "averaging_eligible",
                                   "fallback_reason", "frames")}
    details_json = json.dumps({"sides": side_details, "root_review": root_review},
                              ensure_ascii=False, indent=2, allow_nan=False)
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Frame registration · {html.escape(str(preparation.get('source_name') or output.name))}</title>
<style>*{{box-sizing:border-box}}body{{font:14px Arial,Helvetica,sans-serif;margin:10px;background:#111;color:#ddd}}main{{max-width:1600px;margin:auto;min-width:0}}h1{{font-size:20px;margin:0 0 8px}}.review-summary{{background:#181818;padding:8px;margin:0 0 8px}}.review-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}figure.native-review{{background:#181818;margin:0;padding:6px;min-width:0}}figure.native-review figcaption{{font-size:11px;margin-bottom:4px}}.review-image-scroll{{max-width:100%;overflow:auto}}.review-image-scroll img{{display:block;width:auto;max-width:none;height:auto;image-rendering:pixelated}}details{{background:#181818;margin:8px 0;padding:8px}}details>summary{{cursor:pointer}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border:1px solid #555;padding:5px;text-align:left;vertical-align:top}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap}}.controls{{position:sticky;top:0;z-index:5;background:#111;padding:4px 0}}@media(max-width:650px){{.review-grid{{grid-template-columns:1fr}}}}</style><main>
<h1>Frame registration · {html.escape(str(preparation.get('source_name') or output.name))}</h1>
<p class="review-summary"><b>{html.escape(summary)}</b> · 5 near + 5 far frames</p>
<div class="controls"><label><input id="gamma22" type="checkbox" onchange="toggleGamma22()"> RGB display gamma x^(1/2.2)</label></div>
<div class="review-grid">{''.join(figures)}</div>
<details><summary>Display contract and gate details</summary><p>RGB previews use the side WB diagonal followed by the camera→linear-sRGB matrix and shared exposure. The gamma toggle affects RGB display copies only (off by default; optional x^(1/2.2)). Each Δ panel shows the WB/matrix RGB crop above and a scalar mean absolute raw camera-RGB count difference below; the scalar uses gray with no WB or gamma. Bayer is numeric-only and uses the same-frame WB-demosaic RGB photo preview. These figures are display-only; saved arrays are unchanged.</p>
<p>Numeric gate: <code>{str(gate is True).lower()}</code> · {html.escape('; '.join((assets.get('gate_failures') or []) if isinstance(assets, dict) else []) or 'no recorded gate failures')}</p>{transform_table}</details>
<details><summary>Marker IDs, residuals, and recorded review</summary><pre>{html.escape(details_json)}</pre></details>
<script>function toggleGamma22(){{const on=document.getElementById('gamma22').checked;document.querySelectorAll('.prepare-rgb-image').forEach(image=>image.src=on?image.dataset.gamma22:image.dataset.linear)}}</script></main></html>"""
    html_path.write_text(page, encoding="utf-8")
    assets.setdefault("sha256", {})[html_path.name] = sha256_file(html_path)
    preparation_path.write_text(json.dumps(preparation, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                encoding="utf-8")
    display_hashes_after = {name: sha256_file(output / name) for name in image_names}
    root_review_path = output / "root_review.json"
    root_review_sha = sha256_file(root_review_path) if root_review_path.is_file() else None
    record = {
        "format": "nfplight.fabric-capture-prepare-html-refresh.v2",
        "html_file": html_path.name,
        "html_sha256": sha256_file(html_path),
        "html_generator_sha256": sha256_file(Path(__file__)),
        "preparation_sha256_before": preparation_before,
        "preparation_sha256_after": sha256_file(preparation_path),
        "review_asset_sha256_before": old_asset_hashes,
        "review_asset_sha256_after": assets["sha256"],
        "display_png_sha256_before": display_hashes_before,
        "display_png_sha256_after": display_hashes_after,
        "display_pngs_unchanged": display_hashes_before == display_hashes_after,
        "root_review_sha256": root_review_sha,
        "numeric_arrays_read": False,
        "numeric_arrays_changed": False,
        "inference_executed": False,
    }
    _write_json(output / "prepare_html_generation.json", record)
    return html_path


def _render_capture_index_html(output: Path, index_records: list[dict[str, Any]]) -> Path:
    cards = []
    for item in index_records:
        group = str(item.get("group", "unknown"))
        status = str(item.get("status", "unknown"))
        frame_average = item.get("frame_average", {}) or {}
        average_used = item.get("average_used", frame_average.get("enabled"))
        selection = (f"FP32 mean · near {frame_average.get('near_count', 5)} / far {frame_average.get('far_count', 5)}"
                     if average_used is True else
                     "near_00/far_00 fallback" if average_used is False else "frame selection unavailable")
        report = str(item.get("report_html") or f"{group}/report.html").replace("\\", "/")
        report_link = f'<a href="{html.escape(report, quote=True)}">report</a>'
        details = {
            "fallback_reason": item.get("fallback_reason", frame_average.get("fallback_reason")),
            "selected_frames": item.get("selected_frames"),
            "clipping": _only_both(item.get("clipping")),
        }
        details_text = html.escape(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False))
        cards.append(
            f'<article class="index-card"><div class="index-heading"><h2>{html.escape(group)}</h2>'
            f'<span class="status">{html.escape(status)}</span></div>'
            f'<div class="index-selection">{html.escape(selection)}</div>{report_link}'
            f'<details><summary>RMS and clipping details</summary><pre>{details_text}</pre></details></article>')
    supplemental_links = " · ".join(
        f'<a href="{path}">{label}</a>' for path, label in (
            ("pseudo_far/index.html", "far_pseudo = near/9 대조 실험"),
            ("spatial_alignment_evidence.md", "배율 정렬 근거")) if (output / path).is_file())
    supplemental_links = f"<p>{supplemental_links}</p>" if supplemental_links else ""
    document = f"""<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fabric capture · groups</title><style>*{{box-sizing:border-box}}body{{font:14px Arial,Helvetica,sans-serif;margin:12px;background:#111;color:#ddd}}main{{max-width:1100px;margin:auto;min-width:0}}h1{{font-size:20px;margin:0 0 10px}}.group-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}.index-card{{min-width:0;background:#181818;border:1px solid #555;padding:9px}}.index-heading{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}.index-heading h2{{font-size:15px;margin:0}}.status{{font-size:12px;color:#bbb}}.index-selection{{margin:7px 0}}details{{margin-top:7px}}details>summary{{cursor:pointer;font-size:12px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}}a{{color:#8ccaff}}@media(max-width:620px){{.group-grid{{grid-template-columns:1fr}}}}</style>
<main><h1>Fabric capture · 5 groups</h1>{supplemental_links}<div class="group-grid">{''.join(cards)}</div><p><a href="index.json">index data</a></p></main></html>"""
    path = output / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def write_capture_index(output: Path, group_records: list[dict[str, Any]]) -> Path:
    """Write the compact index and its machine-readable group records."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for item in group_records:
        source = str(item.get("source", ""))
        group = str(item.get("group", item.get("label", Path(source).name or "unknown")))
        run_dir = Path(item.get("output", item.get("output_dir", group)))
        report = str(item.get("report_html", item.get("report", "")) or
                     item.get("prepare_review", (run_dir / "report.html").as_posix())).replace("\\", "/")
        frame_average = item.get("frame_average", {}) or {}
        prepared = item.get("prepared", {}) or {}
        records.append({
            "group": group,
            "source": source,
            "status": str(item.get("status", "unknown")),
            "frame_average": frame_average,
            "average_used": item.get("average_used", frame_average.get("enabled")),
            "averaging_eligible": prepared.get("averaging_eligible", item.get("averaging_eligible")),
            "fallback_reason": item.get("fallback_reason") or frame_average.get("fallback_reason") or prepared.get("fallback_reason"),
            "selected_frames": item.get("selected_frames", prepared.get("selected_frames")),
            "clipping": _only_both(item.get("clipping") or {
                name: record.get("clipping", {}) for name, record in (item.get("conditions", {}) or {}).items()
                if isinstance(record, dict)}),
            "report_html": report,
        })
    _write_json(output / "index.json", {"format": "nfplight.fabric-capture-index.v1", "groups": records})
    return _render_capture_index_html(output, records)


def regenerate_capture_index_html_only(output: Path) -> Path:
    """Render index.html from saved index.json without rewriting machine records."""
    output = Path(output)
    manifest_path = output / "index.json"
    before = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("groups")
    if not isinstance(records, list):
        raise ValueError("index data has no groups list")
    page = _render_capture_index_html(output, records)
    record = {
        "format": "nfplight.fabric-capture-index-html-refresh.v1",
        "active_condition": "both",
        "html_sha256": sha256_file(page),
        "html_generator_sha256": sha256_file(Path(__file__)),
        "index_json_sha256_before": before,
        "index_json_sha256_after": sha256_file(manifest_path),
        "index_json_unchanged": before == sha256_file(manifest_path),
        "inference_executed": False,
    }
    _write_json(output / "index_html_generation.json", record)
    return page


__all__ = ["write_capture_report", "regenerate_capture_report", "regenerate_capture_gallery_and_html",
           "regenerate_capture_html_only", "regenerate_feature_gallery_only", "regenerate_both_only_gallery",
           "regenerate_prepare_html_only", "regenerate_capture_index_html_only",
           "write_prepare_review", "write_capture_index"]
