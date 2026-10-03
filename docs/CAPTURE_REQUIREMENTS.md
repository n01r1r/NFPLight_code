# 현재 Fabric DNG 리포트: 생성 규약과 데이터 흐름

문서 기준일: 2026-10-03. 대상은 원본 저자 weight로 생성한 현재 Fabric 리포트다.
이 문서는 현재 규약의 단일 기준이다. 과거 결정과 실행 기록은 [문서 이력](archive/README.md)에서 확인한다.
새 지시가 과거 지시를 대체한 경우에는 현재 규약을 적용한다.

한국어 문장에는 ISO 24495-1:2023의 공개된 간결한 언어 원칙을 적용했다.
독자가 필요한 정보를 찾고, 이해하고, 사용할 수 있도록 목적·절차·조건·실패 처리를 구분했다.
이는 표준 인증이나 전문 전체에 대한 적합성 선언이 아니다.
근거: [ISO 표준 소개](https://www.iso.org/standard/78907.html), [국제 간결한 언어 연합의 원칙 설명](https://www.iplfederation.org/iso-standard/).
ASD-STE100은 통제된 영어를 위한 규칙이므로 한국어 문서의 작성 기준으로 선택하지 않았다.

## 1. 먼저 확인할 규약

| 항목 | 현재 요구사항 |
|---|---|
| 자료 | 새로 제공한 5개 묶음의 10 cm/30 cm DNG. 묶음별 near 5장, far 5장 |
| 비교 단위 | 각 묶음을 독립 처리한다. 묶음 간 평균을 만들지 않는다 |
| 정밀도 | 입력 처리와 추론은 FP32. FP64 실험, AMP, autocast, TF32 사용 금지 |
| 광도 처리 | `both=(D-B)/(W-B)` 하나만 사용한다 |
| 추론 weight | 현재 runner는 원본 `net_g_real.pth`만 허용한다 |
| 노이즈 제거 | denoiser 생성·weight 로드·실행 및 denoising image 입력을 금지한다 |
| 공간 처리 | 공통 near_00 마커 평면 → bilinear 512 → 중앙 crop 420 → exact-area 256 |
| 평균 | 양쪽 5장 모두 수치·무늬 정합 조건을 통과할 때만 각각 동일 가중 평균 |
| 평균 실패 | 같은 묶음의 near_00/far_00만 사용한다. 실패 이유를 기록한다 |
| 출력 | 묶음별 `report.html`, 전체 `index.html`, 수치 배열 및 검증 기록 |
| 제외 | `paper_comparison`, best_render/MatSynth 비교 행, 새 custom-weight 추론 |
| 보존 | 원본 DNG·metadata·checkpoint와 기존 수치 결과를 보존한다 |

`white_only`와 `neither`는 입력 단계에서 거부한다. 15 cm/45 cm 자료는 사용하지 않는다.
기존 best_render 수치 자료를 현재 예측으로 재사용하거나 다시 추론하지 않는다.
과거 PNG/FP32 실험과 파생 리포트 스크립트를 활성 코드에 다시 넣지 않는다.
전역 허용 저자 estimator는 원본 real/syn weight이지만, 현재 real33 경로에는 synthetic21 추론이 없다.

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
검출용 복사본만 8-bit로 만들고, 긴 변의 최대 크기를 1600으로 줄인다.
검출용 1–99 percentile 대비 조정은 광도 정규화가 아니다.
AprilTag 36h11 마커를 최소 4개 검출해야 한다.

near_00의 보드 중심·대표 방향과 안쪽 모서리로 기준 사각형을 정한다.
사각형을 중심 방향으로 0.035 inset하고, 512 좌표의 네 꼭짓점에 대응시킨다.
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
