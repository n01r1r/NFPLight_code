# 현재 Fabric DNG 리포트: 생성 규약과 데이터 흐름

2026-10-16: 거리비율 relation 보정을 train·eval·inference에 공통 적용한다. `relation_version=spatial_inverse_square_v1`; 학습 gain9·실측 patch20 및 과거 결과는 유지한다. [계산·호환성 규약](REAL_SPATIAL_ATTENUATION.md)을 따른다.

완료된 refinement runner·날짜별 report/PDF 생성기와 전용 테스트는 [복구용 원본 ZIP](archive/completed_refinement_sources.zip)으로 정리했다. 아래 관련 Python 명령은 과거 기록이며 재현할 때만 ZIP의 원래 상대 경로로 복구한다. 현재 capture·학습 진입점은 유지한다.


문서 기준일: 2026-10-03. 대상은 원본 저자 weight로 생성한 현재 Fabric 리포트다.
이 문서는 현재 규약의 단일 기준이다. 과거 결정과 실행 기록은 [문서 이력](archive/README.md)에서 확인한다.
새 지시가 과거 지시를 대체한 경우에는 현재 규약을 적용한다.

한국어 문장에는 ISO 24495-1:2023의 공개된 간결한 언어 원칙을 적용했다.
독자가 필요한 정보를 찾고, 이해하고, 사용할 수 있도록 목적·절차·조건·실패 처리를 구분했다.
이는 표준 인증이나 전문 전체에 대한 적합성 선언이 아니다.
근거: [ISO 표준 소개](https://www.iso.org/standard/78907.html), [국제 간결한 언어 연합의 원칙 설명](https://www.iplfederation.org/iso-standard/).
ASD-STE100은 통제된 영어를 위한 규칙이므로 한국어 문서의 작성 기준으로 선택하지 않았다.

## 1.4. 2026-10-13 사용자 확정: 학습과 실측 추론 분리

이 규약은 아래 §1.2·§1.3 및 과거 실측 보고의 고정 gain9 지시를 대체한다. 과거 결과는 실행 이력으로 보존한다.

- 학습 및 합성 validation/test: `raw_calibrated_gain9_v3`, near4/far12, 고정 `gain=(12/4)^2=9`. 기존 학습·가중치·공식 test 결과를 변경하지 않는다.
- 실측 inference: 교정·정합·프레임 선택 후 선형 FP32 near/far 입력의 중앙 `[118:138,118:138]` RGB 전체 평균으로 `K_L=mean(N)/max(mean(F),1e-5)`를 계산한다. 평균의 차이가 아닌 비율이다. 학습 gain9를 강제하거나 어두운 중앙 패치를 9로 대체하지 않는다. 분모 floor 활성 여부를 기록하고 비유한값은 거부한다.
- 구현: 합성 특징은 `NFPLightModel.build_features`; 실측 특징은 `build_capture_features`를 사용한다. checkpoint의 feature_version은 학습 provenance이며 실측 gain 모드를 결정하는 근거가 아니다. capture 전용 모델은 고정 gain 모드를 거부한다.
- 새 실측 보고는 `training_gain=9`, `capture_gain_mode=author_patch20`, 중앙 near/far 평균·실제 gain·분모 floor·평균 잔차를 기록한다. far gain, 각도 보정, relation/log-relation, mask, log appearance 및 예측·재렌더를 모두 새 계산한다.
- 현재 승인된 sample_crop_only_v1(inset0)로 정합·무늬를 새 검토한다. 지정 옛 snapshot의 inset0.035 입력과 달라 gain만의 효과를 분리한 비교는 아니다. 실측 기하 near4/far12·21채널·unit-normal decoder·F0=.04·lamp16·FP32·denoiser 금지·both·공통 정합·프레임 선택은 유지한다. gain 추정만 저자 실측 방식이며 denoiser를 포함한 원본 모델 전체 재현이라고 주장하지 않는다.
- 사용자 지정 snapshot_20261012_090422의 best-map/best-render 두 동결 checkpoint를 그대로 읽어 5묶음을 새 추론한다. 새 결과는 [중앙 평균 gain 보고](../artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422_patch20_20261013/index.html)에 저장한다. 과거 snapshot은 덮어쓰지 않는다. 이후 refresh와 완주 후 보고도 같은 실측 계약을 따른다. SHA-256은 실행하지 않는다.

## 1. 먼저 확인할 규약

2026-10-19 추가 사용자 승인: [기존 relmap3조건 보고](../artifacts/relmap_three_way_real_20261019/index.html)에 A–C의 validation 선정 checkpoint 실측 결과를 옆에 추가한다. 기존 완료된 조건별 CPU FP32 실측 보고를 읽고 공통 정합·프레임·중앙RGB평균·색변환을 검사한다. 같은 map135k에서 승인된 D render-only의 별도 실측도 기존 capture builder로 생성한다. 학습 gain9와 실측 author_patch20을 구분하고 both·FP32·5묶음·denoiser 금지·GT 부재·원본 보존을 유지한다. 실측 관측 잔차는 checkpoint나 주 비교 선정에 사용하지 않는다. [후속 계약](../artifacts/render_only_followup_20261019/EXPERIMENT_CONTRACT_KO.md)에 따른 별도 승인 경로이며 아래 저자 원본 기본 추론 경로를 바꾸지 않는다.

| 항목 | 현재 요구사항 |
|---|---|
| 자료 | 새로 제공한 5개 묶음의 10 cm/30 cm DNG. 묶음별 near 5장, far 5장 |
| 비교 단위 | 각 묶음을 독립 처리한다. 묶음 간 평균을 만들지 않는다 |
| 정밀도 | 입력 처리와 추론은 FP32. FP64 실험, AMP, autocast, TF32 사용 금지 |
| 광도 처리 | `both=(D-B)/(W-B)` 하나만 사용한다 |
| 추론 weight | 현재 runner는 원본 `net_g_real.pth`만 허용한다 |
| 노이즈 제거 | denoiser 생성·weight 로드·실행 및 denoising image 입력을 금지한다 |
| 공간 처리 | inset=0 · 공통 near_00 마커 평면 → bilinear 512 → 단일 crop46/420(샘플 5cm) → exact-area 256 |
| 평균 | 양쪽 5장 모두 수치·무늬 정합 조건을 통과할 때만 각각 동일 가중 평균 |
| 평균 실패 | 같은 묶음의 near_00/far_00만 사용한다. 실패 이유를 기록한다 |
| 출력 | 묶음별 `report.html`, 전체 `index.html`, 수치 배열 및 검증 기록 |
| 제외 | `paper_comparison`, best_render/MatSynth 비교 행, 새 custom-weight 추론 |
| 보존 | 원본 DNG·metadata·checkpoint와 기존 수치 결과를 보존한다 |

`white_only`와 `neither`는 입력 단계에서 거부한다. 15 cm/45 cm 자료는 사용하지 않는다.
기존 best_render 수치 자료를 현재 예측으로 재사용하거나 다시 추론하지 않는다.
과거 PNG/FP32 실험과 파생 리포트 스크립트를 활성 코드에 다시 넣지 않는다.
전역 허용 저자 estimator는 원본 real/syn weight이지만, 현재 real33 경로에는 synthetic21 추론이 없다.

## 1.1. 2026-10-08 사용자 승인 Fabric checkpoint 재비교

사용자가 400k 종료, 누적 500k 최종, 예전 best_render 세 weight의 새 추론을 명시적으로 승인했다.
이 별도 비교에는 위 custom-weight 제외 규칙의 예외를 적용한다. 저자 원본 리포트와 기존 best_render 수치 자료는 보존한다.
[재추론 리포트](../artifacts/fabric_checkpoint_comparison_20261008/index.html),
[실행기](../run_fabric_checkpoint_comparison.py), [실행 계약](../artifacts/fabric_checkpoint_comparison_20261008/experiment_plan.json)을 사용한다.
기존 continuous-AHD FP32 camera RGB를 읽기 전용으로 읽고 현재 공통 정합·crop·프레임 선택·both·WB/matrix를 새 계산한다.
입력 PNG 재읽기, 8-bit 양자화, 입력 gamma, denoiser, AMP, TF32, SHA-256 계산은 하지 않는다. 출력은 PNG·JSON·HTML이며 NPZ를 저장하지 않는다.
최초 비교는 Fabric 21채널 raw_calibrated_v2의 고정 gain=(10/2.414)²를 사용했다. 2026-10-08 추가 사용자 지시에 따라 새 비교는 저자 실측 코드의 중앙 20×20 `[118:138,118:138]` RGB 전체 평균 비율 K_L을 사용한다. 논문 식 (4)는 중앙 비율을 정의하며 영역 크기는 저자 실측 코드 L102에 따른다. 세 weight·21채널·기하 near=2.414/far=10·decoder·FP32·교정 입력·프레임 선택은 유지한다. 이는 학습 고정 gain 계약과 다른 승인 입력 조건이며 완전한 저자 원본 모델 재현을 뜻하지 않는다.
새 결과: [영역 평균 비교](../artifacts/fabric_checkpoint_comparison_patch20_20261008/index.html). 실행은 `python run_fabric_checkpoint_comparison.py --out artifacts/<새 경로> --gain-mode author_patch20 --review-from artifacts/fabric_checkpoint_comparison_20261008`이다. 기존 정합 기록이 새 계산과 완전히 같을 때만 이전 시각 검토를 재사용하고 출처를 기록한다. 기존 고정 gain report와 수치는 보존한다.
400k는 fabric_amp_l1map400k_render100k_20260915/stage_a.pth,
best_render는 400k+49k, 최종은 400k+100k다. 재시작 로그 total_step=149k/200k와 누적 학습량을 구분한다.
다섯 묶음 × 세 weight의 15개 예측을 새로 계산했다. 078·503은 5장 평균이며 나머지는 수치 게이트 실패로 near_00/far_00이다.
입력·특징 동일성, 반복 추론 일치, FP32·유한값, 원본 크기·mtime 보존 및 HTML 기능 검사는 artifact 검증 JSON에 기록한다.
기존 학습은 AMP였으며 새 추론은 FP32다. 실측 SVBRDF GT가 없으므로 맵 차이를 재질 정확도로 해석하지 않는다.

## 1.2. 2026-10-10 사용자 승인 robustness 후보 실측 비교

사용자가 완료된 7조건의 비교분석과 유망 후보 real data inference·기존 규약 리포트를 명시적으로 요청했다. 이번 범위는 seed19 extra_crop7m_p01 선택 가중치와 같은 모델 family의 baseline7m_p01 대조 가중치, 기존 5개 10/30cm 묶음의 새 FP32 추론이다. 기존 저자 원본 리포트와 §1.1의 세 checkpoint 비교는 보존한다.

새 출력은 `artifacts/robustness_real_extra_crop_20261010`이다. 두 모델은 원래 학습 계약인21채널 raw_calibrated_gain9_v3·near4/far12·고정 gain9·F0=.04와 해당 decoder를 그대로 사용한다. §1.1의 author_patch20 gain과near2.414/far10 기하를 이번 모델에 적용하지 않는다. 실측10/30cm와 모델좌표는 거리비만 대응하며 절대 crop-to-world 대응은 미확정이다.

기존 continuous-AHD FP32 camera RGB를 읽기 전용 입력으로 사용해 현재 공통 정합·512→420→256crop·프레임 선택·both·WB/matrix를 새 계산한다. 정합 재계산이 이전 검토와 일치할 때만 시각 검토를 출처와 함께 재사용한다. 수치 입력 PNG 재읽기·gamma·8bit 양자화·denoiser·AMP·TF32·SHA-256은 금지한다. 기존 capture 표시/차이/픽셀조회 규약과 템플릿을 재사용하고 묶음별 report.html·전체 index.html·PNG/JSON 검증을 저장한다. 기존 frozen 입력과 checkpoint를 변경하지 않는다.

실측 SVBRDF GT가 없어 모델간 map 차이나 관측 재렌더 오차를 재질 정확도 개선으로 해석하지 않는다. 학습·새 seed·외부자료·공식test·추가 모델 후보의 자동개시는 이번 승인에 포함하지 않는다.

### 2026-10-13 사용자 승인 relation 평균 방식 비교

기존 patch20 실측 report의 5묶음·best-map/best-render 동결 weight에서 RGB 평균과 가중 grayscale 추론을 비교한다. 기본 학습·실측 모델 계산을 변경하지 않고 별도 실험으로 추가한다. 실행기는 [run_relation_grayscale_comparison.py](../experiments/run_relation_grayscale_comparison.py), 계산은 [relation_grayscale_comparison.py](../experiments/relation_grayscale_comparison.py)다.

`Δ=clip(Kθ×gain×Far,0,1)−Near`에서 저자 방식은 `mean_RGB(abs(Δ))`, 비교 방식은 `abs(0.299ΔR+0.587ΔG+0.114ΔB)`다. 가중치는 선형 RGB에 gamma 없이 적용하는 명시적 grayscale 실험 조건이며 XYZ luminance 계수라고 해석하지 않는다. Grayscale 변환 후 절댓값이므로 채널 간 부호 상쇄가 포함되며 `sum(weights×abs(Δ))`와 다르다. 두 방식 모두 중앙20×20 **RGB 평균** gain을 사용하고 입력·프레임·checkpoint·기하·FP32·21채널 decoder를 공유한다. 달라지는 network 입력은 relation/log-relation인18/19번 채널뿐이다.

같은 sample의 두 방식·두 checkpoint에서 numerator/raw q/log q의 전체 extrema 표시 범위를 공유하고 relation/log relation 및 N/D/R/S는 [0,1]로 표시한다. 차이는 grayscale−RGB이며 N 각도(°), D/S 물리 RGB RMS, R 부호 차이를 기록한다. 예측 S head와 renderer F0=.04를 구분하고 실측 GT가 없는 민감도 비교를 정확도 개선으로 주장하지 않는다. 기존 수치·이미지를 보존하며 report.html/index.html에 비교 section을 추가한다. SHA 계산·새 학습·새 checkpoint 선택은 하지 않는다.

### 2026-10-13 실제 renderer의 diffuse-only 진단

사용자 요청으로 F0=.04, normal=(0,0,1), roughness=.5, near4/far12·lamp16·CPU FP32를 고정하고 diffuse 밝기 7수준과 좌우 .03/.6 패턴을 비교했다. [실행기](../experiments/verify_diffuse_gain_relation.py), [진단 보고서](../artifacts/diffuse_gain_relation_20261013_final/index.html), [수치](../artifacts/diffuse_gain_relation_20261013_final/results.json), [검증](../artifacts/diffuse_gain_relation_20261013_final/verification.json)을 따른다. 같은 기하의 대칭 pixel에서 중앙20×20 K는 albedo 잔차를 남기지만 renderer의 픽셀별 diffuse 전달계수 비율 K는 이를 FP32 오차 수준에서 소거한다. 보정 후 잔차의 albedo 항은 `(Ktotal×Afar−Anear)×diffuse`다. Near/Far 비율의 albedo 의존성과 이상적 보정 후 relation의 albedo 의존성을 혼동하지 않는다. 작은 중심 coefficient와 개별 minmax의 FP32 오차 증폭도 수치에 기록했다. 이상적 K는 합성 진단용 oracle이며 실측 gain 규약·network·학습을 변경하지 않는다. FP32 tensor를 `.pt`로 보존했고 SHA는 실행하지 않았다.

## 2. 대상 자료와 현재 리포트

대상 묶음은 다음 5개다. 각 묶음에는 `near_00`부터 `near_04`, `far_00`부터 `far_04`까지 있어야 한다.
파일 이름과 촬영 정보는 해당 묶음의 metadata에서 읽는다. 경로 밖 파일을 지정하는 이름은 거부한다.

- `260818_174845_478`
- `260818_175005_078`
- `260923_141241_844`
- `260923_161825_503`
- `260930_152732_008`

사용자가 지정한 현재 리포트는 다음 파일이다.

```text
artifacts/fabric_capture_20261002_aligned_paper_equations/260818_174845_478/report.html
```

디렉터리 이름의 `paper_equations`는 과거 실행 이름이다. 현재 납품 범위를 뜻하지 않는다.
현재 리포트는 실제 far DNG를 사용한 원본 저자 real33 결과를 보여준다.
기존 실행 manifest에 남은 paper 비교 기록은 당시 실행 이력이다. 삭제 후의 파일 목록은 별도 cleanup 증거로 확인한다.

## 3. 전체 처리 순서

```text
DNG + metadata + SHA256 확인
  → visible Bayer uint16 읽기
  → 값 손실 없이 float32 변환
  → continuous AHD로 camera RGB count 생성
  → 각 프레임 마커 검출 및 near_00 공통 평면 정합
  → 512 warp → [46:466] crop → 256 resize
  → 10장 수치 검사 + 무늬 정합 검토
  → 양쪽 각각 5장 평균 또는 양쪽 각각 frame00 선택
  → (D-B)/(W-B)
  → 정규화 camera WB → 내장 camera-to-linear-sRGB 행렬
  → unclipped RGB 보존 → [0,1] 입력 clipping
  → near RGB + far RGB: [1,6,256,256]
  → 원본 입력의 identity copy + 저자 real33 특징 구성
  → FP32 원본 estimator → 10채널 raw prediction
  → 배열·통계·검증 보존 → 표시용 이미지 → report.html
```

표시용 PNG, gamma preview, 마커 검출용 8-bit 영상은 위 수치 입력을 대체하지 않는다.
기본 배열은 HWC다. 모델 배열은 NCHW다. 두 레이아웃을 이름과 shape로 구분한다.

## 4. DNG 읽기와 continuous AHD

`rawpy.raw_image_visible`에서 2차원 Bayer `uint16`을 읽는다.
DNG 내부의 visible CFA, 색상 설명, 센서 원점과 margin을 확인한다.
전체 센서의 CFA 원점을 visible 배열의 원점으로 가정하지 않는다.
2×2 CFA의 반복성과 RGB 채널 대응을 검사한다.

각 파일에서 아래 정보를 기록한다.

| 종류 | 기록 항목 |
|---|---|
| 원본 | 파일 이름, SHA256, raw shape, 최소·최대 count |
| CFA | raw pattern, visible CFA index, RGB index, color description, visible origin |
| 광도 | 채널별 black level, white level, camera WB, 색 변환 행렬 |
| 촬영 | 방향 정보, 노출 설정과 확인 가능 여부 |
| 처리 | dtype, 각 단계 shape·통계, 유효 지지 영역, 코드 hash |

uint16의 모든 값은 float32에 정확히 표현할 수 있다. 변환 전후 값이 같은지 검사한다.
DNG에서 읽은 count는 디코딩된 Bayer 값이다. 원래 ADC 신호를 복원했다고 주장하지 않는다.
수치 처리에는 EXIF 방향 회전을 적용하지 않으며 `orientation_applied=False`로 기록한다.

continuous AHD는 LibRaw 0.21.4의 AHD를 바탕으로 만든 부동소수점 구현이다.
정수 저장과 반올림, Lab 계산의 `+0.5` 반올림 편향을 제거한다.
계산 결과는 유한한 FP32 camera RGB count여야 한다.

AHD의 제한 처리는 다음과 같다.

- 수평·수직 green 후보는 이웃 값의 최소·최대 범위로 제한한다.
- 빠진 red/blue 값을 보간하는 일부 단계에는 `[0,65535]` 제한이 있다.
- 방향 선택을 위한 Lab 평가에서 XYZ를 `[0,1]`로 제한한다.
- 경계 5픽셀은 유효한 3×3 이웃으로 빠진 색을 보완한다. 타일 halo는 8픽셀이며 경계 확장은 edge 방식이다.
- 마지막에 실제 측정한 CFA 채널 값을 복원한다. 전체 RGB에 일괄 `[0,65535]` clipping을 적용하지 않는다.

분기 선택은 정수 AHD와 달라질 수 있다. 차이를 단순한 저장 정밀도 차이로 해석하지 않는다.

## 5. 공간 정합과 crop

마커 검출은 각 프레임에서 새로 수행한다.
검출용 복사본만 8-bit로 만들고, 원본 해상도를 유지하며 `CORNER_REFINE_SUBPIX`로 모서리를 정밀화한다.
far의 작은 마커를 1600으로 축소해 생기는 좌표 오차를 줄이기 위한 규약이다.
검출용 1–99 percentile 대비 조정은 광도 정규화가 아니다.
AprilTag 36h11 마커를 최소 4개 검출해야 한다.

near_00의 보드 중심·대표 방향과 안쪽 모서리로 기준 사각형을 정한다.
사각형에 inset을 적용하지 않고(inset=0), 512 좌표의 네 꼭짓점에 대응시킨다. 종이 배제는 다음 단계의 고정 크롭 한 번으로만 수행한다.
나머지 9장은 near_00과 공유하는 최소 4개 tag의 모든 모서리를 사용한다.
공유 모서리 좌표를 Hartley 방식으로 정규화하고 FP32 최소제곱으로 homography를 구한다.
`h22=1`이며, outlier 제거와 임의 가중치 변경을 하지 않는다.
각 프레임의 상대 변환을 near_00의 기준 변환과 합성한다.

원본 demosaic에 합성 변환을 한 번만 적용한다.
역방향 bilinear warp로 512×512를 만들고 `[46:466,46:466]`을 잘라 420×420를 얻는다.
FP32 exact-area 가중치와 누적으로 256×256를 만든다.
반복 warp, 정수 resize, 중간 PNG 왕복을 사용하지 않는다.

마커 오차는 최종 256 좌표에서 측정한다.
512 좌표를 `(u-46)×256/420`으로 바꾸고, 대응점의 2차원 거리 제곱을 평균한 뒤 제곱근을 구한다.
모든 공유 모서리를 집계한다. 일부 점만 골라 RMS를 낮추지 않는다.
이 값은 마커 평면의 정합 오차다. 섬유 무늬의 정확한 정합을 보장하지 않는다.

## 6. 평균 조건과 실패 처리

각 side 내부에서 5장의 raw shape, CFA, visible origin, B/W, WB, 행렬, 방향과 노출 설정이 호환되어야 한다.
노출 정보가 없으면 평균 조건을 통과한 것으로 처리하지 않는다.
아래 조건은 near와 far 모두에 적용한다.

| 조건 | 통과 기준 |
|---|---|
| 자료 수 | side마다 정확히 5장 |
| 메타데이터 | side 내부의 검사 항목이 일치하고 필요한 노출 정보가 있음 |
| 마커 | 모든 프레임의 최종 256 좌표 RMS ≤ 1 px |
| crop | 420 crop 전체에 유효한 원본 지지 영역이 있음 |
| 처리 | 파일별 처리 오류와 metadata 오류가 없음 |
| 무늬 | 준비 리포트의 overlay·차이·contact sheet를 보고 정합을 승인함 |

수치 조건과 시각 검토가 모두 통과하면 정합된 256 camera RGB count를 side별로 FP32 합산하고 5로 나눈다.
광도 보정과 WB는 이 선택·평균 다음에 적용한다.
검토 화면의 평균 후보는 표시 자료다. 승인 전의 최종 수치 입력이 아니다.

하나라도 실패하면 양쪽 모두 frame00을 사용한다.
통과한 일부 프레임만 평균하거나 한쪽만 평균하지 않는다. RMS 한계도 임의로 높이지 않는다.
이때 frame00도 같은 near_00 평면으로 정합한다.
RMS가 1 px를 넘었다는 이유만으로 fallback 추론 자체를 금지하지는 않는다.
그러나 frame00에 처리 오류나 불완전한 crop이 있으면 해당 묶음을 중단한다.
다른 frame이나 다른 묶음으로 조용히 대체하지 않는다.

묶음별 `root_review.json`에는 다음 항목을 기록한다.

```json
{
  "status": "reviewed",
  "texture_registration_passed": false,
  "note": "실제로 본 영상과 무늬 정합 판단 이유를 적는다."
}
```

위 예시는 형식 설명이다. 실제 검토를 대신하는 승인 기록으로 복사하지 않는다.
`texture_registration_passed`는 JSON boolean이어야 하며 note는 비어 있으면 안 된다.
선택 항목 `inspected_assets`를 기록하면 각 자료의 상대 경로와 SHA256을 제공한다.
검사한 파일은 prepared 디렉터리 안에 있어야 하고, 기록 후 hash가 달라지면 추론을 거부한다.
시각 검토는 판단 기록이며 재료 GT나 수학적 정합 증명으로 표현하지 않는다.

## 7. 광도 보정과 모델 입력

선택한 camera RGB count를 `D`, DNG black level을 `B`, white level을 `W`로 쓴다.
각 파일에서 black level이 채널별로 같아야 하며 `0 ≤ B < W`이고 값이 유한해야 한다.
허용 식은 다음 하나다.

```text
S = (D-B)/(W-B)
w = camera_WB_RGB / min(camera_WB_RGB)
C = S × w                         # 채널별 곱
I_unclipped = C @ Mᵀ             # HWC, M은 camera-to-linear-sRGB
I = clip(I_unclipped, 0, 1)
```

RGB WB는 유한한 양수여야 한다. RGBG metadata의 네 번째 WB 값을 별도 RGB 채널로 쓰지 않는다.
행렬은 `rawpy.color_matrix[:, :3]`에서 읽으며 유한한 rank 3인지 FP32 SVD로 확인한다.
WB와 행렬을 각각 한 번만 적용한다. 음수와 1 초과 값을 가진 중간 배열도 clipping 전에 보존한다.
각 단계의 범위와 clipping 비율을 기록한다.

sRGB OETF, gamma, ICC 변환, tone curve, auto brightness는 수치 입력에 적용하지 않는다.
영상 최대값·percentile로 수치 밝기를 다시 정규화하지 않는다.
ISO·셔터·광원 flux·flat field에 대한 새 보정식을 추가하지 않는다.

near의 RGB 3채널 뒤에 far의 RGB 3채널을 붙인다.
모델 입력은 유한한 FP32 `[1,6,256,256]`이며 범위는 `[0,1]`이다.

## 8. 저자 real33 특징의 정확한 정의

`N`과 `F`는 위에서 clipping한 near/far 입력이다.
과거 denoised 슬롯에는 `N,F`의 identity copy를 넣는다. 실제 denoised 영상은 없다.

### 8.1 밝기와 각도 계수

```text
g = mean(N[:,:,118:138,118:138]) / max(mean(F[:,:,118:138,118:138]), 1e-5)
rho = sqrt((x-127.5)² + (y-127.5)²) / 128
θN = atan(rho/4)
θF = atan(rho/12)
t = cos(θN)/cos(θF)
c_raw = (cos(θF)-cos(θN))/cos(θN)
c = c_raw/max(c_raw)
Fg = g×F
Ft = clip(t×Fg, 0, 1)
```

중앙 밝기는 20×20 영역의 RGB 전체 평균이다. gain에 9 또는 이전 고정 값을 강제하지 않는다.
거리 상수 4와 12는 저자 모델의 정규화 좌표다. cm 또는 m로 해석하지 않는다.
실제 촬영 거리 10/30 cm와 광학 정보를 별도로 기록한다.
`time_co_map`은 코드상의 이름이며 여기서는 각도 계수 `t`다. 노출 시간 보정이 아니다.
`Ft`는 clipping 전 `Fg`에 `t`를 곱한 값이다. `t×clip(Fg,0,1)`로 바꾸면 안 된다.

기하 상수는 원본 NumPy 식과의 호환성을 위해 float64로 생성한 뒤 torch FP32 buffer로 저장한다.
이 초기 상수 생성은 FP64 영상 처리·추론 실험과 구분해 기록한다. 실제 특징과 네트워크 계산은 FP32다.

### 8.2 relation과 log-relation

```text
signed_diff = Ft-N
numerator = mean_RGB(abs(Ft-N))
q = numerator/max(c,1e-5)
log_q = ln(max(q,1e-5))
normalize(z) = (z-min(z))/max(max(z)-min(z),1e-5)
relation = 1-normalize(q)
relation_log = 1-normalize(log_q)
```

normalize의 최소·최대는 해당 tensor 전체에서 구한다. 현재 batch는 1이다.
`q`의 상한을 먼저 1로 clipping하지 않는다.
`mean(abs(RGB 차이))`를 `abs(mean(RGB 차이))`로 바꾸지 않는다.
`relation_log`는 `ln(relation)`이 아니다.
리포트에서 보이는 `N-Ft`는 저장된 `signed_diff`의 부호를 뒤집은 표시다. NPZ의 원래 부호를 바꾸지 않는다.
논문 식 표기를 사용해도 저자 코드 식과 인쇄된 식이 같다고 주장하지 않는다.

### 8.3 33채널 순서

appearance 로그 함수는 `L(x)=(ln(x+0.01)-ln(0.01))/(ln(1.01)-ln(0.01))`이다.
이는 display gamma가 아니다.

| 0부터 시작하는 채널 | 내용 | 채널 수 |
|---|---|---:|
| 0–2 | N RGB | 3 |
| 3–5 | F RGB | 3 |
| 6–8 | N identity copy | 3 |
| 9–11 | F identity copy | 3 |
| 12–14 | clip(Fg,0,1) | 3 |
| 15–29 | 앞의 15채널에 L을 적용한 값, 같은 순서 | 15 |
| 30 | relation | 1 |
| 31 | relation_log | 1 |
| 32 | non_saturated_mask | 1 |

mask는 N, F, N copy, clipping된 gain far 중 어느 RGB 최댓값이라도 `>0.95`이면 0, 아니면 1이다.
정확히 0.95인 값은 이 조건에서 포화로 처리하지 않는다. 공간 crop 유효 mask와 별개다.
마지막에 모든 채널에 `2×값-1`을 적용한다.
실제 estimator 입력은 FP32 `[1,33,256,256]`, 범위 `[-1,1]`이다.
첫 30채널은 albedo 분기로, 마지막 3채널은 다른 분기로 들어간다.

## 9. checkpoint와 추론 결과

현재 runner는 파일 이름과 다음 SHA256으로 원본 real weight를 확인한다.

```text
net_g_real.pth
75f23d38f1298e635d51b98cd82389117d124018cef1c2bae123689198f726b9
```

checkpoint의 `params` tensor는 모두 유한한 FP32여야 한다. state dict는 strict load한다.
기존 checkpoint 파일은 변경하지 않는다. denoiser checkpoint도 보존만 한다.

원본 real network는 width 32, encoder block 수 `[2,2,4,8]`, middle 12, decoder `[2,2,2,2]`를 사용한다.
공유 NAF block과 두 분기, downsample, PixelShuffle upsample 및 skip 결합을 유지한다.
`eval()`과 `no_grad()`로 추론한다. AMP/autocast와 CUDA matmul·cuDNN TF32를 사용하지 않는다.
cuDNN benchmark는 끄고 deterministic 설정을 기록한다.

| raw 출력 채널 | 의미 | 리포트 표시 |
|---|---|---|
| 0–2 | normal XYZ | `(raw+1)/2` RGB encoding |
| 3–5 | diffuse RGB | `(raw+1)/2` |
| 6 | roughness | `(raw+1)/2`, gray |
| 7–9 | specular RGB | `(raw+1)/2` |

raw prediction은 FP32 `[1,10,256,256]`, 저자 tanh 출력 범위 `[-1,1]`로 보존한다.
normal의 길이를 1로 다시 정규화하거나 각도로 바꾸지 않는다. roughness에 새 floor를 넣지 않는다.
출력에 camera WB나 camera matrix를 다시 적용하지 않는다.
GT 없이 재료 정확도 순위나 물리적 정답을 주장하지 않는다.

### 9.1. Prediction 이후 고정 F0 refinement

2026-10-06 사용자 지시에 따라 현재 5개 묶음에 별도 refinement 비교를 추가한다.
원본 estimator prediction은 그대로 보존하고, 원본의 가상 조명 렌더링을 고정 목표로 삼는다.
입력 DNG 영상은 refinement loss에 사용하지 않는다. 새 원본 추론과 refinement는 FP32이며
denoiser·AMP·TF32를 사용하지 않는다. 현재 검토의 선택 프레임과 정합을 그대로 따른다.

Specular RGB F0는 물리 값 `0.04`(raw `-0.92`)로 고정하고 optimizer에서 제외한다.
Normal·diffuse·roughness만 최적화한다. 이 별도 결과의 normal은 렌더링용 단위 벡터이며,
diffuse는 `[0,1]`, effective roughness는 기존 렌더러의 `[0.05,1]` 범위다.
원본 prediction의 저장·표시 규약을 바꾸는 후처리는 아니다.
Loss는 다중 가상 조명의 원본 렌더링 대비 MSE와 초기 고정-F0 맵 유지 항이다.
조명 좌표, 반복 수, 학습률, 유지 항 가중치, seed와 오차는 실행 JSON에 기록한다.

원본 near 렌더링의 RGB 평균과 실제 `I_N`의 RGB 평균 사이 제곱 오차를 최소화하는
양수 스칼라 광원 세기를 구하고 teacher/student에 동일하게 고정한다.
정제 후 맵에서도 같은 관측 영역으로 표시용 최적 세기를 따로 제안하며 loss에는 되돌려 넣지 않는다.
이 값은 모델 좌표와 현재 광도 처리에 맞춘 경험적 배율이며 측정 광량이 아니다.
원본 외관 재현 차이, 미사용 조명 렌더링 차이, 실제 `I_N` 일치도를 구분한다.
외관 재현 차이 감소를 실제 재질 정확도 향상으로 주장하지 않는다.

새 결과는 별도 artifact에 PNG·JSON·HTML만 저장한다. NPZ와 SHA-256 검증은 생성·실행하지 않는다.
기존 sample 리포트에는 같은 행·타일 형식으로 원본, 고정-F0 초기값, 정제 결과와 렌더링을 추가한다.
RGB 표시는 `[0,1]` 및 gamma off/2.2 선택을 유지하고 normal·roughness에는 gamma를 적용하지 않는다.

### 9.2. Rendering loss의 조명 변화·최적화 dynamics 분석

현재 5개 sample의 새 원본 pred에서 시작해 F0 자유와 `F0=0.04` 고정 조건을 비교한다.
고정 조건이 주 분석이며 자유 조건은 무변화 대조군이다. 원본 pred의 동일 조명 렌더링이
목표 영상이다. Perturbation, 실제 입력을 목표로 하는 loss, prior, 공간 규제는 추가하지 않는다.
순수 rendering MSE만 사용하므로 자유 조건은 초기 상태에서 loss·gradient가 0에 가까워야 한다.
수치 처리로 발생한 차이는 재질 개선이나 의도된 perturbation으로 해석하지 않는다.

카메라는 모델 좌표 `(0,0,4)`에 고정한다. 광원 z는 4로 고정하고 중심 하나,
중심과 xy 반경 0.25의 네 방향, 중심과 반경 1.5의 네 방향을 각각 비교한다.
조명·색 채널·픽셀에 대한 평균 MSE를 사용하며 조명 수와 좌표를 기록한다.
이전 §9.1의 카메라·광원 동시 이동 실험과 구분한다. 샘플별 원본 pred에서 맞춘
광원 세기는 F0 조건·조명 조건 전체에서 동일하게 고정한다.

Iteration별 loss, raw 맵 공간의 gradient RMS, 업데이트 크기, 물리 맵 변화량과
단위 normal의 각도 변화를 기록한다. 최종 iteration과 최저 loss iteration을 구분하고,
모든 조건을 동일한 미사용 조명에서 평가한다. 조명 조건에 대한 결과 민감도와
맵별 보상 양상을 분석하되 실제 재질 정확도 또는 전 조명 일반화를 주장하지 않는다.
실행 설정·trajectory·분석은 새 PNG·JSON·HTML artifact로 보존하며 NPZ·SHA-256은 사용하지 않는다.

## 10. LibRaw 비교의 범위

LibRaw AHD reference는 주 입력과 별도로 생성하는 비교 자료다.
`raw.postprocess`는 linear gamma `(1,1)`, 16-bit 출력, auto brightness/scale 및 WB 해제,
방향 회전 해제, black 보정 해제 등의 명시적 옵션으로 실행한다.
설치된 rawpy/LibRaw 버전은 실행 환경에 기록한다. 구현의 기준 버전 0.21.4와 혼동하지 않는다.

reference에는 같은 공간 변환과 최종 선택 프레임을 적용한다.
현재 runner는 이 reference 입력으로 보조 추론도 수행해 비교 기록을 남긴다.
reference 결과로 continuous AHD 주 입력이나 주 예측을 대체하지 않는다.
실측 DNG에 노이즈 제거를 적용한 것으로 설명하지 않는다.

## 11. 리포트 표시 규약

수치 배열과 화면 이미지는 분리한다. 표시 변환은 원본 NPZ나 통계를 바꾸지 않는다.
선형 RGB 입력·밝기 보정·입사각 보정 영상의 표시 범위는 항상 `[0,1]`로 고정한다.
단계별 percentile 자동 대비 조정은 금지한다. 범위 밖 값은 표시 복사본에서만 clip한다.
K_L처럼 공간적으로 일정한 계수는 실제 숫자로 표시하며 단색 영상으로 표현하지 않는다.
변하는 수치 계수(K_θ·coefficient·denominator)는 제한된 원값 범위로 표시하고 실제 값 눈금을 붙인다.
이는 선형 RGB `[0,1]` 고정 규칙과 구분한다. 계산 배열과 계수 값은 바꾸지 않는다.
2026-10-06 사용자가 지정한 `fabric_capture_20261002_aligned_paper_equations` 전면 갱신에는 NPZ 저장을 금지한다.
`run_fabric_capture.py --refresh-no-npz --all-groups --archive-root <보존 원본 FP32 경로> --out <지정 경로>`는
원본 camera RGB를 읽기 전용으로 읽고, 새 공통 좌표 정합·검토·FP32 추론을 메모리에서 수행한다.
새 artifact는 묶음별 `refresh_20261006/`이며 PNG·JSON·HTML만 저장한다. NPZ·SHA-256 검증은 실행하지 않는다.
이 모드에서는 수치 원본이 저장됐다고 주장하지 않는다. 과거 NPZ와 예측은 보존하며 현재 비교에서 제외한다.

| 자료 | 표시 규약 |
|---|---|
| camera RGB 사진 | 정규화 WB와 camera matrix를 한 번 적용한 복사본 |
| 이미 WB한 RGB | matrix만 적용 |
| 이미 linear RGB인 자료 | WB와 matrix를 다시 적용하지 않음 |
| Bayer | 수치는 원본 mosaic 보존. 사진 preview는 같은 프레임의 WB demosaic이며 대체 표시임을 명시 |
| relation, log-relation, roughness, mask | gray, black 0 → white 1 |
| estimator 특징 채널 | 실제 `[-1,1]` 범위를 표시 |
| log 원값과 중간 scalar | 저장된 표시 범위와 실제 최소·최대를 명시 |
| signed 차이 | 0을 포함하는 대칭 범위와 부호 의미를 표시 |
| normal | XYZ encoding RGB. 사진 gamma를 적용하지 않음 |

scalar 표시값은 명시된 범위 `[a,b]`를 이용해 `clip((x-a)/(b-a),0,1)`로 만든다.
2026-10-13 수정: 모든 scalar의 색상 범례는 원래 수치 범위 `[a,b]`를 표시하고 가운데 눈금은 `(a+b)/2`다. PNG 밝기0~1을 원래 값 범례로 대체하지 않는다. 범위 밖 값·색상 양자화의 검정/흰색 포화와 실제 값0을 구분한다. 기존 중간 scalar의 percentile 표시 범위는 유지하며 추론 배열·수식을 변경하지 않는다.
고정값 영상 등 퇴화 범위는 표시 helper의 별도 처리를 따른다. 수치 범위를 숨기지 않는다.
같은 의미의 영상을 비교할 때에는 공통 표시 범위를 사용한다.
채널별 signed 수치는 gray 영상과 눈금으로 보여주며 사진 RGB와 구분한다.
과거 cividis 표시는 현재 scalar 규약이 아니다.

사진 RGB에는 선택 가능한 display gamma 2.2를 제공한다.
체크박스 기본값은 off이며 on이면 표시 복사본에 지수 `1/2.2`를 적용한다.
로그 특징·scalar·signed 눈금의 수치 의미에는 적용하지 않는다.
256 crop은 확대 보간 없이 확인할 수 있어야 한다.
예측은 normal·diffuse·roughness·specular의 의미별 영상으로 묶는다.
채널마다 별도 예측 그림 페이지를 만들지 않는다.

단계 선택 UI는 해당 영상 옆이나 바로 위에 둔다.
식·입력 의미·표시 범위는 펼칠 수 있는 설명에 둔다. 영상 위 글자는 최소화하고 겹치지 않게 한다.
WB 대각행렬과 camera matrix를 명시한다.
리포트와 준비 정합 영상을 실제로 확인한다. 자동 수치 검사만으로 무늬 정합을 승인하지 않는다.

## 12. 실행 절차

entry point는 [run_fabric_capture.py](../run_fabric_capture.py)다.
새 추론에는 지정한 새 artifact 디렉터리를 사용한다. 기존 결과 디렉터리를 덮어쓰지 않는다.
아래 경로는 새 실행 이름의 예시다. 실행 전에 비어 있거나 존재하지 않는지 확인한다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepare-only --precision fp32 --out artifacts/fabric_capture_20261003_run01
```

1. 각 묶음의 `prepare_review.html`과 정합 자료를 확인한다.
2. 각 묶음에 실제 판단을 담은 `root_review.json`을 작성한다.
3. 동일 prepared 경로를 지정해 추론한다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepared artifacts/fabric_capture_20261003_run01 --precision fp32
```

`--all-groups`는 지정 source와 같은 parent의 5개 고정 묶음을 독립 처리한다.
`--device cuda`가 기본값이며 CPU 실행은 `--device cpu`로 지정한다.
`--prepare-only`와 `--prepared`는 동시에 사용할 수 없다. 두 단계 중 하나를 명시한다.
`--out`과 `--prepared`를 함께 쓰면 같은 resolved 경로여야 한다.
`--frame-index`는 0만 허용한다. `--paper-equations`는 폐지했다.
코드의 역사적 기본 output은 `fabric_capture_20261001_author_original`이다.
새 실행에서는 위와 같이 새 경로를 명시한다.

원본·metadata·준비 코드·검토 자료의 hash가 바뀌면 준비 결과를 재사용하지 않는다.
리뷰 누락, 잘못된 JSON, 다른 checkpoint, non-finite 배열 또는 호환되지 않는 dtype은 오류로 중단한다.
중단한 이유와 묶음을 기록한다. 이미 성공한 다른 묶음을 실패한 묶음의 결과로 재사용하지 않는다.

## 13. 파일과 재현 증거

| 파일/위치 | 용도 |
|---|---|
| `preparation.json` | 프레임별 준비 기록, 정합, 평균 수치 조건, 코드·원본 hash |
| 프레임별 NPZ | Bayer, cast, demosaic, warp512, crop420, output256, valid mask, reference |
| `prepare_review.html` 및 영상 | 무늬 정합 검토와 frame00/평균 후보 확인 |
| `root_review.json` | 실제 시각 검토 판단 |
| `source_arrays.npz` | 최종 선택/평균한 공통 입력 단계 |
| `both/`의 NPZ·이미지 | 광도·WB·linear RGB·33채널 특징·10채널 예측과 표시 자료 |
| 실행 manifest·verification | 선택 집단, 식·dtype·통계·clipping·검증·환경·checkpoint hash |
| `report.html` | 묶음별 현재 결과 |
| `index.html` | 5개 묶음의 결과 링크 |
| HTML/display generation JSON | 표시 갱신 코드·자료 hash와 수치 보존 확인 |

원본/공통 배열을 조건별로 중복 처리하지 않는다. NPZ 저장·재읽기 일치와 선택 프레임 재계산을 확인한다.
식 검증과 source 재검증은 추론 정확도에 대한 실험 증명과 구분한다.
원본 50개 파일, metadata, checkpoint와 관련 코드 hash를 실행 전후 확인한다.
Python·NumPy·PyTorch·rawpy·LibRaw·OpenCV, 장치·CUDA/cuDNN 정보와 dtype audit를 기록한다.
서로 다른 장치나 라이브러리 실행이 bit 단위로 동일하다고 가정하지 않는다.

HTML만 갱신할 때에는 `regenerate_capture_html_only(group_dir, include_preserved_checkpoint=False)`를 쓴다.
이 helper는 저장된 manifest·통계를 읽고 숫자 배열과 실행 manifest를 보존한다.
표시 PNG 갱신 helper는 기존 NPZ에서 표시 자료만 만든다. 새 모델 추론을 수행하지 않는다.
scalar와 차이 표시 갱신 기록에서는 NPZ hash 전후 일치를 확인한다.
HTML 전용 갱신과 표시 영상 갱신을 같은 작업으로 설명하지 않는다.

## 14. 폐기 항목과 보존 범위

`paper_comparison.html`, 해당 image 디렉터리와 branch 전용 `paper_arrays.npz`는 폐기했다.
새 runner는 해당 비교 분기를 생성하지 않는다. 실제 리포트와 준비 자료는 유지한다.
기존 numerical equation helper와 pseudo_far 진단은 별도 진단 코드이며 현재 리포트 요구사항이 아니다.
과거 best_render·custom-weight 결과는 이력으로 보존한다. 현재 비교 행에 넣지 않는다.

문서 정리는 과거 문서를 `docs/archive/`로 이동하는 작업이다.
원본 이미지·수치 NPZ·checkpoint의 삭제나 재추론을 뜻하지 않는다.
`.git`, `checkpoints`, `results`, dataset junction의 외부 대상을 재귀 삭제하거나 이동하지 않는다.

## 15. 과거·remote 결과와 비교할 때 확인할 항목

| 비교 지점 | 현재 경로 | remote 저자 경로 또는 확인 범위 |
|---|---|---|
| 입력 | DNG Bayer count → continuous FP32 AHD | 원본 PNG reader는 8-bit 읽기와 integer resize 후 `/255`, gamma `**2.2` |
| 공간 | near_00 공통 평면과 고정 crop | 같은 sample 이름만으로 crop·정합이 같다고 판단하지 않음 |
| 밝기 | DNG B/W, WB, 내장 행렬, clipping | 실제 이전 입력과 설정을 확인해야 비교 가능 |
| denoiser | 원본 입력 identity copy | upstream 기본 경로에는 denoiser. fork의 no_denoise 경로와 구분 |
| 특징·network | 저자 real33 순서·원본 state 구조 유지 | 동일 저장 입력에서 fork no_denoise 특징 구성은 CPU 기준 bit 일치 확인 |
| 정밀도 | 현 처리와 inference FP32 | upstream 모델도 float32. FP32라는 이름만으로 변화 원인을 설명하지 않음 |

이전 조사 기준 remote commit은 upstream `cd16babaa79e79d8c02feaee03a7dd0f85118ae4`,
origin/main `cc79c9b98f06dccc0a78974e4b625c2f78ac5936`이다. 현재 최신 remote라는 뜻은 아니다.
CPU 동일 입력 특징 검사는 estimator 예측 재실행 검사가 아니다.
저장 CUDA 특징과 CPU 기준의 최대 차이 약 `1.25e-5`는 장치별 비교 기록이다.

같은 원본 저자 weight의 9월 리포트는 이전 조사에서 특정하지 못했다.
9월 best_render 결과를 그 대용으로 쓰지 않는다.
확인된 10월 1일/2일 동일 저자 weight 비교에서는 near clipped 입력은 같고,
far 입력·중앙 gain·특징·예측이 달랐다. 공통 평면 정합 변경과 연결해 조사한 결과다.
현재 규약 문서가 확인되지 않은 9월 실행 옵션을 추정해 채우지는 않는다.
관련 로컬 증거는 `artifacts/author_report_input_comparison_20261002.json`과
`artifacts/remote_author_code_comparison_20261002.json`에 있다.

## 16. 구현 근거

| 처리 | 코드 |
|---|---|
| 단계 실행·입력/weight 검증 | [run_fabric_capture.py](../run_fabric_capture.py) |
| DNG 읽기 | [raw.py](../capture_processing/raw.py) |
| continuous AHD | [demosaic.py](../capture_processing/demosaic.py) |
| warp·crop·resize | [geometry.py](../capture_processing/geometry.py) |
| 공통 정합·평균 조건 | [burst.py](../capture_processing/burst.py) |
| 광도·WB·행렬 | [photometry.py](../capture_processing/photometry.py) |
| 저자 특징·추론 adapter | [author_real_capture_adapter.py](../model/author_real_capture_adapter.py) |
| 공유 network | [nfplight_net.py](../network/nfplight_net.py) |
| 리포트·표시 갱신 | [report.py](../capture_processing/report.py) |

실행별 실제 값은 해당 artifact의 manifest와 배열로 확인한다.
이 문서의 규약, 코드 구현, 실행 기록이 충돌하면 충돌을 먼저 보고하고 원인을 확인한다.

현재 보고서는 사용자 수정 지시에 따라 F0=0.04 고정 조건만 표시한다. 과거 자유 조건의 원시 실행 자료는 보존한다. 원본 prediction은 GT가 아니므로 원본 대비 렌더링·맵 지표는 error가 아닌 차이로 표기한다.

보고서 정성 비교는 본문에 열어 두고 원본 대비 관찰문을 함께 기록한다. n/d/renderer-effective r는 원본=고정 직후 baseline으로 표시하며 원본 F0 렌더와 고정 직후 렌더를 구분한다. 모든 상태를 동일 크기·동일 RGB 표시 변환으로 비교하고 실제 선택 iteration을 캡션에 쓴다. 저장된 렌더는 중앙 조명이므로 미사용 조명 수치의 공간적 원인을 그 영상으로 설명하지 않는다. 광원 세기와 실제 I_N의 사용 범위를 명시한다.

변화 분석의 본문은 픽셀별 normal angular change와 물리 diffuse/roughness/F0의 signed difference heatmap을 사용한다. RGB 차이는 채널별 부호를 보존한다. 공통 color scale·단위·원본 기준·선택 iteration을 표시하고, F0 고정 자체의 차이와 refinement 변화 0을 구분한다. 표시용 PNG 차감으로 수치 heatmap을 만들지 않는다. 광원 조건은 중앙 1광원 및 XY 반경 0.25/1.50의 각 5광원으로 표기한다.

보고서는 원래 pred의 n/d/r/s와 F0 고정 후 n/d/r를 열린 비교 표로 제시한다. 차이는 채널별 패널이 아니라 맵당 단일 영상이며 d/s는 signed RGB, r는 signed grayscale, n는 angular variation(도)로 표시한다. RGB의 중간 회색은 변화 0이며 배율과 비교 가능 범위를 캡션에 기록한다.

### 활성 Fabric 학습·추론 입력 계약 (2026-10-08)

새 raw_calibrated_gain9_v3 checkpoint는 near=4/far=12 모델 좌표와 고정 far gain=9, linear FP32를 공유한다. Lamp intensity16, surface[-1,1], colocated light/camera이며 입력 양자화·gamma·denoiser는 없다. 실측10/30cm와 동일 거리비이며 절대 crop-to-world 대응은 미확정이다. 저자 원본 및 §1.1 역사적 재비교는 기존 checkpoint별 계약을 유지한다.


## 2026-10-08 렌더링·재질 맵 차이 표시 갱신

사용자 지시에 따라 렌더링 comparison/residual의 `difference` 영상은 CIEDE2000 ΔE00으로 표시한다. 기존 최적화 loss와 MSE 수치는 유지한다. 모델 특징의 near−scaled-far, relation 및 촬영 정합 차이는 원래 정의를 유지한다.

색차 입력은 저장 FP32의 선형 sRGB다. clip·gamma·노출 없이 XYZ D65→Lab(2°)로 변환하고 kL=kC=kH=1의 ΔE00을 계산한다. 색 변환만 CPU FP64를 사용하며 추론·학습·저장 배열 정밀도는 바꾸지 않는다. 음수·비유한 RGB와 잘못된 shape는 오류로 처리한다. 1 초과 RGB의 색차는 white Y=1 기준 확장값이며 디스플레이 JND 판정을 뜻하지 않는다.

맵 difference는 normal 각도(°), roughness 부호 있는 물리값 차이, diffuse/F0의 선형 RGB 반사율 RMS 크기다. RGB RMS는 채널별 부호를 보존하지 않으므로 원래 채널은 대화형 픽셀 조회에서 확인한다. 각 difference PNG와 대화형 PNG export에는 colorbar, 범위, 단위, clipping 비율을 포함한다.

범위는 0부터 실제 최대 크기를 기본으로 사용한다. P98 절대값이 최대의 절반보다 작으면 고립된 꼬리를 표시에서 clipping하고 P98을 상한으로 사용한다. 지표와 원수치에는 모든 픽셀을 포함한다. 동일 sample의 original/updated 대 I_N은 같은 범위를 사용한다. A/B explorer는 전체 영상의 현재 pair 범위를 기준으로 하며 ROI가 범위를 바꾸지 않는다. 다른 pair의 색 강도는 범례를 함께 확인한다.

재생성: `python refresh_difference_reports.py`, `python build_cue_constraints_report.py`, `python build_cue_explorer.py`. 모두 새 최적화 없이 저장 결과를 표시한다. 검증·범위·제한 목록은 [difference 갱신 기록](../artifacts/difference_display_refresh_20261008/verification.json)에 있다. 저장 FP32 prediction/render가 없는 checkpoint·capture 결과는 PNG로 색차를 역산하지 않는다. 과거 PDF와 원수치가 없는 중간 montage는 기존 기록이며 이 갱신의 수치 색차 비교로 해석하지 않는다.


### 2026-10-10 PI 피드백: RAW 처리 감사

[LibRaw·실측 입력 감사](../artifacts/representation_inverse_council_20261010/LIBRAW_REAL_INPUT_AUDIT_KO.md)를 따른다. 별도 감사 작업이지만 처리 옵션·metadata·보정·표현·모델의 공동 영향으로 해석한다. 현행 수치 계약을 변경하지 않는다. 코드/옵션/저장 metadata 감사와 향후 픽셀·물리 검증을 구분한다.


## 1.3. 2026-10-12 사용자 승인 전체 데이터 7M 완주 후 실측 보고
사용자가 현재 전체 Fabric+Leather baseline7m_p01 완주 후 real-data inference report를 요청했다. 새 출력은 `artifacts/full_dataset_baseline7m_real_20261012`이다. 같은 실행의 best_map.pt와 최종 selected.pt를 비교하고 net_g_selected_fp32.pth의 tensor 동일성을 검사한다. 마지막 optimizer weight와 validation-selected weight를 혼동하지 않으며 실제 selection stage/step을 표기한다. 이전 §1.2 진단셋 가중치는 재사용하지 않는다.
§1.2의 공통 정합·512→420→256·5장 평균 게이트/실패 near00/far00·both·원본 FP32 camera RGB·gain9·near4/far12·unit-normal·F0=.04·lamp16 계약과 기존 HTML 표시·차이·픽셀조회 템플릿을 재사용한다. 정합 기록이 이전 승인 기록과 정확히 같을 때만 시각 검토를 재사용한다. 이전 예측/수치를 새 모델 결과로 재사용하지 않고 fresh FP32 추론 및 반복 일치를 검사한다. 묶음별 독립 비교, PNG/JSON/HTML만 저장하며 SHA-256·NPZ·denoiser·입력 gamma/8bit 재읽기·AMP/TF32는 실행하지 않는다. 실측 GT가 없으므로 관측 재렌더 차이를 재질 정확도로 주장하지 않는다. 원본 저자 real33은 다른 계약의 역사적 참조다. 브라우저 시각·조작 검사는 완료 근거가 없으면 미수행으로 표기한다.


### 2026-10-12 학습 중 best-map/best-render 갱신 보고 승인
사용자가 가용한 map-best/render-best로 현재 real-data HTML을 생성·갱신하도록 요청했다. `artifacts/full_dataset_baseline7m_best_real_20261012/refresh.py`가 일관된 snapshot을 별도 복사하고 CPU FP32로 새 추론한다. 완료 여부 gate 대신 두 best checkpoint의 현재 실행 identity·FP32 finite·strict load를 검사하며 실제 stage/step을 표기한다. 나머지 §1.3 입력·정합·프레임·표시·보존·claim 규약은 그대로 적용한다. 버전별 결과를 보존하고 성공한 최신 보고서만 index.html/latest.json으로 연결한다. 현재 학습과 완료 후 보고 대기열은 변경하지 않는다.


### 2026-10-12 사용자 승인 008·478 burst 평균 예외
사용자 지시로 008은 near/far 동일 인덱스0·1·3·4의4장 평균, 478은 near/far5장 평균을 진단한다. 원래 정합 RMS·무늬 검토 실패를 통과로 바꾸지 않고 user_average_override.json에 실패 사유와 권한을 함께 기록한다. 선택 프레임의 full crop 지원은 유지한다. 현재 best-map Map152k와 best-render Render32k snapshot을 그대로 사용해 frame selection 효과를 비교한다. 기존 단일 쌍 보고서는 보존하며 새 출력은 artifacts/full_dataset_478_008_burst_20261012다. FP32 CPU·both·gain9·near4/far12·F0=.04·unit-normal 및 나머지 표시 규약은 유지한다. 시간축 변동에는 noise·조명·잔여 움직임이 섞이므로 순수 sensor noise나 정확도 개선으로 주장하지 않는다.


### 2026-10-12 사용자 승인 far blur 민감도 진단
사용자가 이론적 긍정·반박 및 병렬 보고서 생성을 요청했다. 새 출력 artifacts/full_dataset_far_blur_20261012는 현재 승인 burst 조건의478/008에 far-only Gaussian σ0·0.5·1·2px(256grid) 민감도를 검사한다. 선형 calibrated unclipped FP32 RGB에 ±3σ kernel/reflect101 후 clip하며 near·gain9·geometry·F0·동일Map152k/Render32k snapshot은 고정한다. 모든 조건의 관측오차 target은 unblurred burst 평균으로 유지한다. 이는 명시 승인된 blur 진단으로 learned denoiser/denoising image를 사용하지 않으며 기존 정식 입력·학습을 변경하지 않는다. 478 정합 실패는 그대로 기록한다. 기존 평균보고서를 보존하며 객관적noise분리·세부보존·실측GT 정확도/최적σ를 검증했다고 주장하지 않는다.


## 2026-10-12 활성 입력: 크롭 한 번으로 5cm 샘플 ROI 확보

사용자가 확인한 5cm는 **종이를 제외한 샘플 영역**이며 마커 내부 전체의 폭이 아니다.
활성 저장형·메모리 갱신 경로는 `sample_crop_only_v1`로 통일한다.
마커 내부 quad는 inset=0으로 검출하고, 공통 near_00 기준 512×512로 정렬한다.
그 뒤 `[46:466,46:466]` 크롭 한 번으로 420×420 샘플 ROI를 선택하고,
전체 샘플 ROI를 FP32 exact-area 방식으로 256×256에 축소한다.
최종 샘플 ROI의 물리 폭은 사용자 확인값 5cm로 기록한다. 마커 내부 전체 폭을 5cm로
간주하거나 420/512를 곱해 최종 폭을 축소 계산하지 않는다. 추가 inset은 오류로 거부한다.

inset 제거로 homography와 최종 픽셀 기준 RMS가 바뀌므로 새 입력은 정합/RMS/support 및
실제 무늬 검토를 다시 수행한다. 과거 검토를 자동 재사용하지 않는다.
46px 고정 크롭이 실제 모든 촬영에서 종이를 배제하는지는 각 묶음의 새 미리보기로 확인하며,
이 코드 검증만으로 실측 종이 제거 성공이나 추론 개선을 주장하지 않는다.
gain9·near4/far12·FP32·both 및 프레임 선택 규약은 유지한다.
008·478의 승인된 평균 예외는 실제 실패 게이트를 보존한다.

이전 `marker_inner_full_v1`(crop=0) 시도는 사용자 ROI 정의와 달라 폐기된 진단이다.
`artifacts/full_dataset_inner_full_20261012/`의 부분 결과는 보존하지만 활성 리포트나
정식 5cm 입력 결과로 사용하지 않는다. 이전 inset=0.035 결과도 역사적 입력이다.

구현 검사: 크롭·inset 거부·FP32·메모리 준비 등 관련 테스트 34개 통과.
실측 첫 프레임 검토: [비교 이미지](../artifacts/crop_only_validation_20261012/crop_only_contact.png) · [검증 기록](../artifacts/crop_only_validation_20261012/verification.json). 5묶음 near_00/far_00의 crop support와 눈에 보이는 종이 테두리 제거를 확인했다. 전체 burst 정합 재검토나 새 모델 추론은 수행하지 않았다.

2026-10-14: [원문 Eq.2–5와 구현 항별 시각 검증](../artifacts/relation_equation_visual_20261014/index.html)은 중앙 픽셀 원문식·실측 patch20·합성 gain9·diffuse oracle을 구분한다. [실행기](../experiments/visualize_relation_equations.py)는 실제 renderer와 원문의 상수 조도 대조군을 새 FP32 계산하며, 구현 채널 일치·diffuse 분해·대칭 항등식을 검증한다. 기존 capture/학습 규약은 유지한다.

## 2026-10-19 사용자 승인: 재학습 relmap 7M 실측 정성 보고

사용자 “진행해줘”로 완료된 수정 relmap baseline7m_p01의 best-map135k / 최종 선택 best-render49k를 기존5개10/30cm 묶음에 새 추론했다. [결과](../artifacts/spatial_7m_real_20261019/index.html) · [검증](../artifacts/spatial_7m_real_20261019/verification.json). §1.4의 author_patch20·spatial_inverse_square_v1·FP32·both·현재 공통 정합/프레임 선택을 적용했다. 현재 render A/B/C 학습을 유지하며 CPU2스레드로 실행했다. 078·503은5장 평균, 나머지는near00/far00이며 실측GT/정확도 개선 주장은 없다. 동결checkpoint·기존 결과 보존, SHA-256·denoiser·새NPZ·입력PNG·AMP/TF32 없음.


## 2026-10-19 A/B/C 조건별 실측 보고 추가 승인
사용자가 각 조건 종료 후 동일 실측 5묶음 추론·별도 정성 보고서를 요청했다. artifacts/render_weight_followup_20261019/real_reports/contract.json에 따라 별도 CPU FP32 2thread watcher가 validation 선정 checkpoint와 공통 map135k를 비교한다. 조건별 runs/<arm>/real_report/index.html에 저장하며 기존 합성 report/index.html과 구분한다. 학습 runner/config/checkpoint는 변경하지 않는다. 실측 author_patch20·both·기존 정합/프레임 게이트를 유지하고 gain9를 실측에 적용하지 않는다. 학습 완료 및 메모리 여유를 기다리며 실패 시 자동 재시작하지 않는다. 실측 보고까지 모두 완료해야 heartbeat 완료 처리한다.

## 2026-10-19 사용자 승인: 기존64M·기존7M·변경7M relmap 3조건

사용자가 기존 relmap Fabric64M best-render449k / 기존 relmap 7M best-render32k(누적185k) / 변경 relmap 7M best-render49k(누적204k)의 실측5묶음 비교를 지정했다. [보고서](../artifacts/relmap_three_way_real_20261019/index.html) · [계약](../artifacts/relmap_three_way_real_20261019/experiment_plan.json) · [검증](../artifacts/relmap_three_way_real_20261019/verification.json). 이 artifact의 과거 두 조건에 한해 보존된 relation 함수를 읽기 전용으로 연결하고 공간 계수를 적용하지 않는다. 기존7M의 전체21채널 특징과 과거 함수의 exact equality를 검사했다. 활성 train/eval/inference의 spatial_inverse_square_v1 기본값은 유지한다.

같은 FP32 camera RGB에서 공통 정합·inset0 crop·both·WB/matrix·프레임 게이트와 author_patch20을 새 계산한다. 64M의 near2.414/far10·raw XYZ 및 두7M의 near4/far12·unit XYZ는 보존한다. 공통 재렌더는 near4/far12·lamp16·F0=.04이며 모델별 원래 기하 재렌더는 별도 표기한다. Relmap/log-relmap은 실제 network 채널에서 추출하며 NDRS·near/far 재렌더·ΔE00·L1·MSE를 함께 제공한다. ΔE00 heatmap은 sample/side별 세 조건 공통 범위·범례·clipping 비율을 쓴다. 원래 HTML 특징/픽셀 조회 템플릿도 재사용한다.

두7M의 train/val·seed·기본 설정은 relation 버전 외 일치하지만 학습 경로와 선정 시점은 다르며 기존7M은 memory_paused 실행이다. 크기만/relmap만의 인과 비교나 실측GT 정확도로 해석하지 않는다. 이전2조건(spatial relation을 두 모델 모두 적용) 보고서와 원본 checkpoint는 보존한다. GPU 학습·기본 코드 변경, SHA-256, 새NPZ, denoiser, 입력PNG, AMP/TF32 없이 CPU2스레드로 실행했다.
