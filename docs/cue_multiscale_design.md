# 색 단서와 다중 스케일 refinement: 탐색 실험 계약

완료된 refinement runner·날짜별 report/PDF 생성기와 전용 테스트는 [복구용 원본 ZIP](archive/completed_refinement_sources.zip)으로 정리했다. 아래 관련 Python 명령은 과거 기록이며 재현할 때만 ZIP의 원래 상대 경로로 복구한다. 현재 capture·학습 진입점은 유지한다.


2026-10-07. 목표는 색·무늬와 normal/roughness의 동반 변화를 줄이면서 원본의 세부 normal을 유지하는 것이다. 실제 fabric의 정답 맵은 없다. 이 문서는 실험 전에 정한 가설·설정이며 성공 결과를 뜻하지 않는다.

## 변경 범위와 가설

기존 `analyze_condition`의 `I_N` 선형 RGB MSE + 0.1 × 40방향 원본 prediction 렌더링 MSE를 유지한다. 원본 저자 pretrained prediction에서 동일하게 초기화하고 n/d/r을 모두 갱신한다. F0=0.04와 q는 고정하며 FP32, fresh Adam lr=0.01, seed=0을 사용한다. 공유 spatial constant를 적용하지 않는다.

가설: 색 단서와 **저주파 맵의 국소 공분산**을 벌점으로 주면 전역 normal 상수화보다 일부 세부 normal을 덜 파괴할 수 있다. 국소 공분산은 동일 위치에서 색에 따라 선형으로 바뀌는 맵 성분을 벌점으로 준다. 색 경계 전체를 평탄화하는 edge loss가 아니다. 제곱하기 때문에 양·음 상관이 상쇄되지 않는다. 색과 함께 변하는 실제 geometry/roughness에도 벌점이 걸릴 수 있다.

[Tappen 등, PAMI 2005](https://people.csail.mit.edu/billf/papers/pamiReflShade.pdf)은 diffuse와 같은 색 조명 아래의 색도 변화가 반사율 단서가 될 수 있지만, 회색 무늬와 specular가 이 가정을 위반하는 예를 보인다. 같은 논문은 shading과 reflectance 모두 여러 주파수에 존재하므로 필터만으로 분리할 수 없음을 보인다. 따라서 아래 lowpass는 **선택한 제약 대역**이며 재질/형상 분류기가 아니다. [Deschaintre 등의 공식 SVBRDF 자료](https://repo-sam.inria.fr/fungraph/deep-materials/)는 rendering-aware 학습의 원본 코드·데이터를 제공한다. 여기서 원본 렌더링 보존은 기준 응답의 유지이며 geometry 검증이 아니다.

## 최소 callable 계약

`analyze_condition(..., cue_config: dict | None = None)`만 추가한다. `None`은 기존 동작이다. 실험 dict의 항목은 `lowpass_kernel=9`, `local_kernel=31`, `cue_budget=0.1`, `detail_budget=1.0`이다. 첫 두 값은 픽셀 단위의 양의 홀수이고 마지막 두 값은 유한한 비음수다. 사용한 스케일은 출력 해상도의 픽셀 단위이며 실제 실의 물리 크기를 뜻하지 않는다.

활성 cue는 `mode='fixed'`, measured target, fresh optimizer를 요구한다. spatial constant/resume와 함께 사용하지 않는다. 입력 material은 FP32 `[B,10,H,W]`, target은 같은 장치의 비음수·유한 FP32 `[B,3,H,W]`다. normal은 `physical_maps`의 단위 normal 3채널, roughness는 같은 decoder의 물리 roughness 1채널이다. normal 좌표계는 renderer와 기존 prediction 계약을 따른다. 각 reflect padding 크기가 H와 W보다 작아야 한다. dtype/shape/kernel/weight/nonfinite 오류는 실행 전에 거부한다. Pilot에서는 sumRGB<1e-4인 dark pixel이 일부만 존재하는 입력을 `partially dark chromaticity unsupported` 오류로 거부한다. 모든 pixel이 dark이면 cue는 비활성화할 수 있다. 선택된 078·478 target의 dark fraction은 root 사전 검사에서 모두 0이었다.

## 고정된 손실

`P_k(x)`는 reflect padding 후 stride=1 box average pool이고 `Q=P_31`이다. `c = I_N / clamp(sum_RGB(I_N), min=1e-4)`는 고정·detach한 3채널 색도다. 합이 1e-4 이상이면 공통 scalar shading은 색도에서 상쇄된다. dark pixel 비율과 clipped RGB 픽셀 비율을 기록한다. 전부 dark인 입력은 c=0이므로 cue가 비활성화되고 detail 항목은 여전히 활성일 수 있다. 부분 dark 입력을 지원하려면 단순 clamp가 아니라 고정 validity mask로 lowpass 및 국소 moments를 가중하는 후속 변경이 필요하다. specular/색 조명에는 색도 해석의 한계가 있다. 최적화 diffuse에서 cue를 만들지 않는다.

각 `m in {N,r}`에 대해 `a=P_k(m)`, `b=P_k(c)`를 계산한다. cue 3채널과 map C채널의 모든 조합에 대해 다음을 계산한다.

```
cov = Q(b*a) - Q(b)*Q(a)
var_c = clamp(Q(b*b) - Q(b)*Q(b), min=0)
C_m = mean(cov*cov / (var_c + 1e-6))
```

평균은 batch·공간·cue channel·map channel 전체다. C는 색 cue가 국소적으로 설명하는 map 에너지의 대용량이며, 물리 truth 오차가 아니다. map 분산으로 나누지 않으므로 값을 평탄화해 에너지를 줄이는 방향도 허용한다. 이를 진단해야 한다.

FP32 moments를 계산하기 전에 a와 b에서 batch/channel별 전체 공간 평균을 빼는 수치 안정화를 적용한다. 공분산·분산은 상수 이동에 불변이므로 수학적 목적함수는 같다. 독립 CPU 검사에서 안정화 전 64² gray/brightness-only 입력은 roundoff로 C_N≈1.72e-7/7.99e-8을 만들어 잘못 활성화됐고, 색도 평균 중심화 후 두 예제의 N/r C가 모두 0이었다. 이 검사는 실제 fabric의 성공 증거가 아니다.

초기 고정-F0 상태의 `C_N0`, `C_r0`를 고정한다. 초기 값이 1e-8 이상인 항목만 `C_m/C_m0`로 정규화해 평균을 취한 것이 `L_cue`다. 모두 비활성이면 0이다. 활성 여부는 초기 상태에서만 정하고 이후 map 분산으로 비활성화하지 않는다.

```
H(N) = N - P_k(N)
E0 = mean(H(N_initial)^2)
D = mean((H(N)-H(N_initial))^2) / E0
```

E0<1e-8이면 D는 비활성이며 0이다. D는 에너지만 유지하는 제약이 아니라 잔차의 부호·위치·크기 변화에 반응한다. reference 세부의 correctness는 가정하지 않는다. 초기 reference에 고주파 색 누출이 있으면 같이 보존될 수 있다.

`R0`는 동일 초기 고정-F0 상태의 measured center MSE다. 고정·detach한 R0를 쓰며 0이면 cue/detail 가중값은 모두 0이다.

```
L = center_MSE + 0.1*mean40_preservation_MSE
    + cue_budget*R0*L_cue + detail_budget*R0*D
```

초기 활성 cue가 있으면 cue 추가 손실은 초기 measured 오차의 10%다. 원본 고주파 normal을 모두 없애면 detail 손실은 초기 measured 오차와 같다. 두 budget은 같은 손실 스케일에서의 개입 강도이며 통계적·물리적 최적값이 아니다. 초기 C·E·R, 활성 항목, epsilon, kernel, 각 비가중/가중 loss를 기록한다.

## 제한된 ablation

078을 주 대상, 478을 stress 대상으로 한다. 기존 결과를 수정하지 않고 새 artifact 디렉터리에 기록한다. 모든 조건을 동일 원본에서 1000 updates, 동일 Adam, 조명·q·입력·해상도·FP32로 계산한다. 안정화 조기 종료는 끈다. 초기·종료 배열과 수치 trajectory를 남긴다. 중간 맵 snapshot은 필수가 아니다.

|조건|k|cue budget|detail budget|
|---|---:|---:|---:|
|baseline|사용 안 함|0|0|
|fullres cue|1|0.1|0|
|lowpass cue|9|0.1|0|
|lowpass cue + detail|9|0.1|1|

이 8회가 초기 budget이다. 1000-step baseline과 동일하게 비교한다. 과거 10k 결과는 이 탐색 실험의 동등한 대조군이 아니다. 연장 실험은 탐색 결과를 본 뒤 별도 설정·경로로 기록하고 1k 결과를 보존한다.

## 평가와 해석

각 map의 full-resolution C(k1)와 coarse C(k9)를 **모든 조건에서 같은 evaluator**로 보고한다. 학습에 사용한 C 감소는 직접 최적화한 지표의 감소이므로 독립 증거가 아니다. local kernel을 53으로 바꾼 비학습 평가도 함께 보면 window 선택 민감도를 확인할 수 있다. C가 0에 가까운 초기 상태에서는 상대 감소를 `null`과 이유로 기록한다.

세부 normal 평가는 항상 같은 k9로 다음을 보고한다.

* HP energy ratio: `sqrt(E_final/E_original)`. 0은 잔차 소멸, 1은 같은 RMS이며 위상·정확성은 알 수 없다.
* HP normalized error: `sqrt(mean((H_final-H_original)^2)/E_original)`. 0이면 같은 잔차다.
* HP cosine: `mean(H_final*H_original)/sqrt(E_final*E_original)`. 부호·위치 일치를 보며, 에너지 부족 시 `null`과 이유를 기록한다.
* map 공간 분산과 normal 평균 각도 변화, roughness 범위·변화량을 함께 보고하여 평탄화로 C를 줄인 경우를 드러낸다.

center 측정 MSE, 40방향 원본 응답 보존 MSE, 미사용 40방향 원본 응답 MSE는 각각 분리한다. loss 합계는 조건별 regularizer가 다르므로 서로 다른 목적함수의 total만으로 조건 순위를 정하지 않는다. 고주파 에너지와 눈에 보이는 반복 직조 구조의 유지가 실제 normal correctness를 증명하지 않는다.

## CPU TDD와 반례

1. 공간적으로 일정한 RGB 색을 scalar brightness로만 바꾸면 색도가 일정하여 cue가 비활성/거의 0이어야 한다. 모두 0 입력·초기 C/E/R의 퇴화도 유한한 손실과 명시적 상태를 반환해야 한다.
2. 알려진 flat N와 constant r에 저주파 색과 연동한 오염을 주입한다. cue의 작은 gradient step이 C와 known-truth 오차를 줄여야 한다.
3. 알려진 독립 fine sinusoidal normal(예: 4px 주기)과 별도의 broad 색 cue(예: 32px 주기)를 합성한다. 저주파+detail이 fullres보다 reference fine 잔차 왜곡을 줄이는지 확인한다. reflect 경계 구간과 내부를 구분해 관찰한다. 정확한 GT normal 각도·roughness MAE도 보고한다.
4. 실제 truth N가 broad 색과 같은 위치에서 변하는 반례를 만든다. cue는 이를 벌점으로 주고 truth에서 멀어질 수 있음을 검사·보고한다. 이런 결과를 버리거나 generic 성공으로 포장하지 않는다.
5. 회색 broad 무늬에 N 누출을 주입한다. 색도가 일정해 cue가 감지하지 못하는 반례를 검사한다.
6. 고주파 normal의 부호를 뒤집거나 같은 에너지의 다른 주파수로 바꾸면 energy ratio는 1일 수 있으나 HP error는 양수, cosine은 낮아야 한다.
7. FP32·shape·kernel·유한성·nonnegative controls 오류와 fixed F0 유지, 추가 loss trajectory/terminal 재평가 일치를 검사한다.

TDD는 구현 계약 검증이고 fabric 실험은 조건부 관측이다. 정답 없는 실제 자료에서 leakage/geometry disentanglement가 입증됐다고 선언하지 않는다. 주장은 선택한 색도·스케일 아래의 국소 동반 변화 감소와 reference detail 유지까지로 제한한다.

## 독립 CPU 합성 관측

[실행 스크립트](../tmp/cue_tdd/sol_synthetic_experiment.py)와 [수치 증거](../tmp/cue_tdd/sol_synthetic_evidence.json)는 64², FP32 CPU 1thread, Adam lr=0.005, 100 updates의 별도 관측이다. 이 실험은 cue prior만 최적화했다. renderer·measured MSE·원본 응답 보존은 없으므로 fabric refinement의 성능을 예측하는 물리 검증이 아니다. 파일 size/mtime 기준으로 실행 중 cue module이 바뀌지 않았음을 기록했으며 SHA-256은 실행하지 않았다.

알려진 4px fine normal과 32px broad 색 연동 오염을 합성한 경우 초기 GT normal 평균 각도 오차는 5.418°였다.

|조건|최종 GT 각도 오차(°)|GT HP 상대 RMS 오차|GT HP cosine|GT HP RMS 비율|
|---|---:|---:|---:|---:|
|fullres|3.492|1.311|0.887|2.115|
|lowpass|3.370|1.190|0.922|2.048|
|lowpass + detail|2.757|0.304|0.956|1.040|

반대로 동일 broad 변화가 실제 GT normal·roughness의 일부인 반례는 초기 GT 오차가 거의 0이었다. 같은 prior를 최적화하면 fullres/lowpass/lowpass+detail의 GT normal 오차가 각각 7.044°/6.914°/3.845°가 됐고 roughness MAE는 0.0740/0.0719/0.0719가 됐다. lowpass+detail도 normalized cue를 1에서 0.0636으로 줄이면서 reference HP cosine 0.994를 유지했다. 따라서 작은 cue와 좋은 reference HP 일치만으로 coarse geometry 또는 roughness의 correctness를 확인할 수 없다.

## 승인된 mixed-scale 대조와 통합 탐색기

초기 pilot의478은 저주파 roughness cue가 감소하면서 전체 해상도 roughness cue가 증가했다. 다음 단일 대조를 사용자가 승인했다. `roughness_lowpass_kernel`을 선택 설정으로 추가하며 기본 `None`은 기존 `lowpass_kernel`을 따른다. mixed 조건은 N cue k9, roughness cue k1, normal detail k9다. 초기 normal/roughness 각 스케일의 고정 normalization을 그대로 사용한다.

```
L_cue_mixed = 0.5 * (C_N_k9/C_N_k9_initial + C_r_k1/C_r_k1_initial)
```

cue budget0.1, detail budget1, FP32·q·F0·fresh Adam·1000 updates를 유지한다. 기존 결과는 보존하고 별도 `artifacts/scale_cue_20261007/mixed`에 기록한다. mixed 결과의 숫자는 이 설계 문서에서 예측하지 않는다.

`build_cue_explorer.py`는 저장된 원본·공통 fixed-F0 초기·공간 상수 초기/종료·기존 fit·고정1k·현재 최대10k 조건을 패키징한다. 모든 상태에 동일한 공통 evaluator를 적용해 normal k1/5/9/17, roughness k1/5/9의 raw cue와 원본 대비 비율을 계산한다. 이 공통 평가와 조건별 활성 band·초기 분모를 사용하는 훈련 trajectory를 구분한다. 공통 원본·near/far·q와 모든 새 조건의 saved initial byte 동일성을 검사한다. `fixed_initial`은 저장된 공통 초기 맵에서 F0=.04인 상태를 별도 표시한다. 이 상태와 공간 상수 초기의 center render는 동일한 고정 camera FP32 CPU shader로 재평가한다. 원본 전체 spatial F0는 원본 teacher에 유지된다. 추론이나 최적화는 실행하지 않는다.

기록된 montage frame과 terminal 수치 맵을 내장하며 저장된 intermediate FP32 snapshot은 디스크 source/step manifest로 참조한다. 미저장 중간 상태를 생성하지 않는다. 텐서는 little-endian FP32 HWC bytes를 deflate 압축해 넣는다. `package_data()`의 trajectory는 검증용 plain JSON이며 HTML serialization 직전에만 UTF8 JSON을 lossless deflate/base64로 바꾼다. 브라우저의 native decompression은 null을 포함한 실제 기록을 그대로 복원한다. 세 TDD 디렉터리의 로그와 존재하는 최종 validation JSON도 내장한다. source stat과 byte equality로 비교하며 SHA-256은 실행하지 않는다.

## 승인된 최대10k 재실험과 다중 스케일 조건

사용자는 기존 baseline/fullres/lowpass/detail/mixed의 두 sample 모두를 최대10000 updates와 기존 안정화 종료 기준으로 다시 계산하도록 승인했다. 고정1k 결과는 보존한다. 새 경로는 `artifacts/scale_cue_10k_20261007/{condition}/{sample}`이다. 최소300 updates 이후100-update 창에서 loss span/mean≤0.001, normal 평균 각도 변화≤0.001°, diffuse/roughness MAE≤1e-5일 때 종료한다. cap 도달은 안정화 충족과 구분한다.

여섯 번째 `scalespace` 조건은 normal cue kernel5/9/17, roughness cue kernel1/5/9를 사용한다. normal HP anchor는 기존 k9이고 cue budget0.1/detail budget1을 유지한다. 각 band의 초기 cue 에너지와 활성 여부를 고정하며, component별 활성 band의 정규화 loss를 평균한 뒤 N/r component 평균을 취한다. 개별 band 값과 초기 normalization은 출력에 기록한다. 기본 단일-band 조건의 알고리즘을 바꾸지 않는다.

통합 탐색기는 새6조건을 `*_10k` ID와 `current` 그룹에 담고 기존1k를 `pilot` 그룹에 보존한다. `current`는 현재 비교의 데이터 그룹 식별자이며 종료 상태를 대신하지 않는다. 기본 final build는 새6조건×2sample의 실제 종료 결과가 모두 있어야 한다. 기록된 montage frame과 전체 scalar trajectory를 넣고 미저장 중간 상태를 생성하지 않는다.
