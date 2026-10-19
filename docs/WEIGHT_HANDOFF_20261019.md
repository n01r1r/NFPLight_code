# 7M weight 전달 안내

전달 순서는 **① relmap 수정 전 → ② relmap 수정 후 → ③ C 옵션**이다. 세 모델은 모두 7,609,834 parameters이며, C는 수정 후 모델의 best-map135k에서 render 학습을 새로 시작한 조건이다. A/B는 현재 후보에서 제외했고 과거 파일은 보존한다.

## 파일명과 전달 폴더

전달용 파일은 `artifacts/weight_handoff_20261019/weights/`에 모았다. 원본 checkpoint를 그대로 복사하고 이름만 바꿨다. 세 파일과 이 문서, `training_settings.json`을 함께 전달하면 된다. Weight는 기존 저장소 규약에 따라 Git에서 제외되며 코드·문서·설정 요약만 push한다.

|순서|모델|전달 파일명|
|---|---|---|
|01|relmap 수정 전 7M|`01_nfplight_7m_relmap-original_render-w1_initlr1e-4_step32k.pt`|
|02|relmap 수정 후 7M|`02_nfplight_7m_relmap-spatial_render-w1_initlr1e-4_step49k.pt`|
|03|C 옵션 7M|`03_nfplight_7m_relmap-spatial_C_render-w10_initlr1e-5_step10k.pt`|

`render-w`는 render L1의 가중치, `initlr`는 render 단계의 **초기** 학습률, `step`은 선택된 checkpoint의 render update 수다. 학습률은 ReduceLROnPlateau가 조절하므로 파일명의 값을 고정 학습률로 해석하면 안 된다.

전체 설정은 [training_settings_20261019.json](weights/training_settings_20261019.json)에 기록했다. 전달 폴더의 `training_settings.json`도 같은 내용이다.

## 세 조건의 차이

|항목|01 수정 전|02 수정 후|03 C 옵션|
|---|---|---|---|
|Relmap|`legacy_no_spatial`|`spatial_inverse_square_v1`|`spatial_inverse_square_v1`|
|선택한 map weight|map152k|map135k|02와 같은 map135k|
|Render 시작 전 실행한 map updates|153k|155k|새 map update 없음|
|Render 목적함수|L_map + 1 × L_render|L_map + 1 × L_render|L_map + 10 × L_render|
|Render 초기 LR|1e-4|1e-4|1e-5|
|Render optimizer|단계 전환 시 fresh Adam|단계 전환 시 fresh Adam|map135k에서 fresh Adam|
|선택한 render step|32k|49k|10k|
|Checkpoint의 total_step|185k = 153k + 32k|204k = 155k + 49k|145k = 상속 map135k + 새 render10k|
|실행 상태|메모리 일시정지 시 보존한 선택 weight|floor20 기준으로 운영 종료|10k 선별 예산 종료|

01과 02의 total_step은 선택한 best-map 시점보다 실제 실행한 map updates를 기준으로 기록한다. C는 과거 map155k 실행의 RNG를 이어받았다. 다만 시작 weight는 map135k이고 optimizer는 새로 만들었다. 따라서 C의 145k와 02의 204k를 같은 학습 경로의 누적량으로 비교하면 안 된다. C는 02의 render49k weight에서 이어 학습한 모델이 아니다.

C는 사용자가 유지하기로 선택한 후속 조건이다. 10k 종료를 수렴이나 통계적 유의성으로 해석하지 않는다. 01과 02도 학습 경로·선택 시점이 달라 둘의 차이를 relmap 하나의 효과로 단정할 수 없다.

|선택 checkpoint의 validation 지표|01 수정 전|02 수정 후|03 C 옵션|
|---|---:|---:|---:|
|NDRS L1 합|0.1247438307|0.1195049770|0.1207984001|
|Novel-light render L1|0.0035547363|0.0033044383|0.0032749340|

두 지표 모두 낮을수록 좋다. C의 render L1은 낮지만 NDRS 합은 02보다 높다. 실측 재질 정답은 없으므로 이 수치를 실측 물성 정확도로 해석하지 않는다.

## 공통 학습 설정

|항목|설정|
|---|---|
|데이터|Fabric + Leather 전체; train566(Fabric358/Leather208), validation164(Fabric98/Leather66)|
|Crop·batch|Train은 재질당16crop, 총9,056 slots; batch4; validation crop 0/4/8/12|
|Sampling·증강|전체 slots에서 batch 내 중복 없이 표집; update 간 재표집 허용; D4 회전·반사|
|구조|TwoBranchNet, width32, encoder [2,2,4], middle4, decoder [2,2,2], dropout0.1|
|Seed·정밀도|Seed19, FP32, AMP/TF32 off|
|입력 특징|21채널, `raw_calibrated_gain9_v3`, 선형 RGB|
|합성 입력 기하·gain|Near4 / far12, 고정 gain9, lamp intensity16|
|Material·손실|Unit XYZ normal; L_map = N/D/R/S 성분별 L1의 합; S 정답과 재렌더 F0=.04|
|Optimizer|Adam, betas=(0.9,0.999), weight_decay=0, gradient clip norm1; EMA 없음|
|Map LR|초기5e-4, 최저1e-5|
|Render LR|위 조건별 초기 LR, 최저1e-6|
|Scheduler|ReduceLROnPlateau; factor0.5, patience5, threshold0.005(relative), cooldown2, eps1e-8|
|학습 조명|반지름4; θ=5/10/20/30/45°, φ=0°부터45° 간격, 총40방향|
|Validation 조명|θ=12/27/40°, φ=22.5/112.5/202.5/292.5°, 총12방향|
|평가·선택|1,000 updates마다 평가; best-map은 validation NDRS 합, best-render는 validation render L1|
|선택 보호|Best-render는 NDRS 합 ≤ best-map NDRS 합 × 1.02를 충족해야 함|
|종료·예산|01/02 map cap400k·render cap100k, LR floor 이후20회 관측 기준; C는 render10k 선별 예산|

Test는 보고에만 사용하고 checkpoint 선택이나 scheduler에 사용하지 않는다. 이미 관측한 test이므로 독립 최종 검증으로 주장하지 않는다. 데이터 population과 train/validation 행은 세 checkpoint에서 정확히 같다.

## Inference에서 맞춰야 할 설정

`.pt`의 `net_g`를 읽고 `identity.policy`와 `identity.scientific_settings.backbone`으로 `experiments.robustness_suite.run.create_network`를 구성한다. 기본 `MatSynthCaptureModel`은 64M 구조를 만들기 때문에 이 7M weight를 그대로 로드할 수 없다. Eval 모드와 FP32를 사용하고 예측 normal XYZ를 단위 정규화한다. Specular head는 보존하되 재렌더링에서는 F0=.04를 쓴다.

학습·합성 validation은 `build_features`의 gain9를 사용한다. 실측은 `build_capture_features`의 중앙20×20 RGB 평균 near/far 비율(`author_patch20`)을 사용한다. 실측 gain에 학습 gain9를 곱하지 않는다. DNG 정합·crop·`both=(D-B)/(W-B)`는 [capture 요구사항](CAPTURE_REQUIREMENTS.md)을 따르며 denoiser를 사용하지 않는다.

**01의 과거 결과를 재현하려면 공간 거리 보정을 꺼야 한다.** 현재 코드는 옛 checkpoint의 relation metadata만으로 자동 전환하지 않는다. 21채널 모델의 `spatial_attenuation_factor`를 ones로 설정하고 `relation_version=legacy_no_spatial`을 명시한다. 02와 C는 현재 공간 보정 경로를 그대로 사용한다. 공통 보정의 수식·곱셈·clip 순서는 [relmap 규약](REAL_SPATIAL_ATTENUATION.md)에 있다.

기존 C의 `net_g_selected_fp32.pth`에는 feature 버전과 backbone 정보가 빠져 있으므로 이번 전달에서는 설정이 포함된 `.pt`를 사용한다. 향후 간이 export는 해당 정보를 보존하도록 수정했다. 보정 전 optimizer의 현행 학습 resume은 거부한다. Checkpoint 안의 로컬 경로·mtime는 학습 이력일 뿐, 수신 환경의 데이터 설치 경로나 학습 재개 권한을 나타내지 않는다.

## 원본과 검증 기록

|전달 파일|원본 checkpoint|
|---|---|
|01|`artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422/snapshot/best_render.pt`|
|02|`artifacts/spatial_7m_real_20261019/snapshot/best_render.pt`|
|03|`artifacts/render_weight_followup_20261019/runs/C_weight10_lr1e5/best_render.pt`|

세 전달 파일은 원본과 byte 단위로 일치한다. 모두 실제 7.61M 구조에 strict load했고 FP32·유한값을 확인했다. C의 기존 간이 export와 선택 `.pt`의 weight tensor도 정확히 같다. Weight와 기존 실험 산출물은 덮어쓰지 않았다.

코드 점검에서는 CPU 전체405개 통과·42개 건너뜀, export 수정 후 관련45개 통과, 보고서 JavaScript 검사 통과를 확인했다. SHA-256 호출은 정책에 따라 차단했고 GPU 검사는 제외했다. 새 학습·DNG 재추론·성능 검증은 실행하지 않았다.

2026-10-19 점검 시작 시 remote와 HEAD는 모두 `4b07b43`이었다. 로컬 relmap·gain 분리·runner·보고서 변경과 export metadata 보완을 `403d99a`에 기록했다. D worker의 Windows 경로 오류도 수정했으나 실패한 queue를 재실행하지 않았다. 현재 후보는 C만 유지한다. 로컬 진단·자원 설정 파일은 전달 commit에서 제외했다.
