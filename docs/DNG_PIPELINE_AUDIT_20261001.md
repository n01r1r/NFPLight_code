# DNG → NFPLight 읽기 전용 파이프라인 감사

작성일: 2026-10-01. 저장소 코드와 입력 메타데이터를 읽어 경로·수식·shape를 추적했다. 이 감사에서는 capture runner, 모델 inference, 새 배열 생성, 실험을 실행하지 않았다. 따라서 아래는 코드가 정의하는 처리와 확인된 입력 메타데이터이지, 출력 이미지·clipping 비율·예측값에 대한 실측 보고서가 아니다.

사용자가 새로 정한 범위는 **새 DNG 5묶음 각각의 보고서**다. 각 묶음에서 여러 프레임을 정렬해 평균할지는 별도 marker 정렬 acceptance rule로 결정한다. 이 결정은 현재 코드 상태와 구분한다. 현재 runner는 묶음당 near/far 각각 단일 `frame_index`만 고르며 burst 평균은 구현되어 있지 않다.

## 입력 경로와 현재 코드의 선택

유지 계약과 README, CLI 기본값은 `data/20260827_130311_10cm30cm_burst5`를 가리키지만, 이 작업공간에는 해당 경로가 없다. 반면 새 폴더 [260930_152732_008 metadata](../data/260930_152732_008/metadata.json)가 있으며 near/far 각 5개 DNG를 선언한다. `run_fabric_capture.py`의 `DEFAULT_SOURCE`는 아직 옛 경로다. CLI를 기본값 그대로 실행하면 `run()`의 source 검사에서 실패한다. 새 묶음을 처리하려면 `--source data/260930_152732_008`처럼 폴더를 지정해야 하지만, 실제 처리·source 선택은 사용자가 요청한 5묶음 범위와 marker acceptance 기준이 확정된 뒤다. 경로 선택 규칙은 [요구사항](CAPTURE_REQUIREMENTS.md#L5), [README](../README.md#L7), [runner 기본값](../run_fabric_capture.py#L31), [CLI 옵션](../run_fabric_capture.py#L619), [source 유효성 검사](../run_fabric_capture.py#L274)를 참조한다.

`input_data/real_data`에는 기존 PNG near/far 이미지가 있고 `input_data/syn_data`에는 PNG 학습자료가 남아 있지만, 활성 Python 코드에 `input_data`를 읽는 참조는 없다. README도 이전 PNG inference 경로를 제거했다고 기록한다. 현재 capture CLI의 `--source`는 개별 PNG나 `input_data`가 아니라 `metadata.json`과 그 파일명이 가리키는 DNG들이 있는 폴더를 요구한다. Fabric 학습 경로도 `data/fabric_native256_v1/manifest.json`을 사용한다. 따라서 새 입력이 PNG라면 이 CLI로 바로 전달되지 않는다. 근거: [CLI source resolver](../run_fabric_capture.py#L39), [README의 제거 기록](../README.md#L21), [유지되는 학습 entry point](../README.md#L33).

`select_source_frames()`는 metadata의 near/far `filenames` 목록을 검사하고, 목록 길이와 `frameCount`가 일치하는지 확인한 뒤 같은 index의 한 쌍을 고른다. 경로 구분자와 접두사도 검사한다. 기본 index는 0이므로 `near_00.dng`/`far_00.dng` 한 쌍이며 `averaged=False`로 기록된다. `--frame-index`로 다른 단일 index를 지정할 수 있지만, 다섯 프레임 반복이나 평균은 없다. [선택 로직](../run_fabric_capture.py#L55), [CLI default](../run_fabric_capture.py#L621), [한 프레임씩 처리](../run_fabric_capture.py#L311).

현재 코드 상태에서 다섯 묶음별 보고서를 만들려면 서로 다른 source/output으로 runner를 반복 호출해야 한다. 각 실행은 이미 내용이 있는 출력 폴더를 거부한다. 여러 프레임의 공통-marker 정렬과 acceptance rule이 통과했을 때의 평균은 별도 burst 처리 기능이 필요하다. 이 문서는 해당 후속 작업을 현재 동작으로 서술하지 않는다.

## DNG unpack과 raw 속성

주 경로는 `rawpy.imread()`에서 `raw.raw_image_visible`을 복사한다. 반드시 2D `uint16`이어야 하며, `raw.postprocess()`는 authoritative 입력 경로에서 호출하지 않는다. `uint16 → float32` cast의 모든 샘플이 동일한지 확인한 뒤 AHD가 H×W×3 RGB를 만든다. shape·CFA·비유한 값은 경계에서 검사한다. [RawCapture와 unpack](../capture_processing/raw.py#L24), [DNG 읽기·CFA 구성](../capture_processing/raw.py#L43), [정확 cast와 AHD 호출](../capture_processing/raw.py#L99).

unpack에서 보존하는 rawpy 값은 `raw_image_visible`, `raw_pattern`, `raw_colors_visible`, `color_desc`, `sizes.top_margin`, `sizes.left_margin`, `sizes.flip`, `black_level_per_channel`, `white_level`, `camera_whitebalance`, `color_matrix`다. manifest metadata에는 DNG 경로/hash, 보이는 raw shape/dtype/min/max, CFA와 visible CFA index, visible origin, color description, black/white, WB, color matrix, orientation 값 및 orientation 적용 여부가 들어간다. `raw_pattern`은 전체 raw 좌표의 CFA라 visible margin이 CFA 위상을 바꿀 수 있다. 코드가 `raw_colors_visible[:2,:2]`와 color description으로 visible Bayer 위상을 다시 만들고, 전체 visible mask가 주기적인 2×2인지 확인하는 이유다. [CFA phase와 저장 metadata](../capture_processing/raw.py#L60).

현재 공유된 read-only DNG 요약은 3024×4032 `uint16`; near raw 범위 528–2777, far 508–4095; 각 CFA black level 528, white level 4095다. `raw_pattern=[[0,1],[3,2]]`에서 RGBG description을 적용한 visible CFA는 `[[R,G],[G,B]]`다. visible origin은 (top,left)=(0,0), flip=6이지만 orientation은 적용되지 않는다. 두 파일의 camera WB는 `[2.142578125, 1, 1.8847655057907104, 0]`이고 코드가 쓰는 첫 세 RGB 값은 minimum=1로 이미 정규화되어 있다. 이 요약의 마지막 green 항목 0은 사용되지 않는다. 실제 capture distance metadata는 near optical-axis 0.10051784698351215 m, far 0.30422835898131617 m이며 plane distances와는 다른 값이다. capture metadata에는 near/far 각각 5개 filename이 있다. 이 문서 작성 중 DNG를 다시 열어 처리하지 않았고, color matrix 숫자·실제 marker 검출 결과·AHD 이후 범위는 산출하지 않았다.

### AHD의 코드 동작과 clipping

구현은 float32로 수평·수직 후보를 만들고 CIELab 차이의 동질성을 비교하는 continuous AHD다. 실제 처리 순서는 다음과 같다. [기본 배열과 CFA 검증](../capture_processing/demosaic.py#L65), [경계용 sparse RGB 구성](../capture_processing/demosaic.py#L163), [Lab 변환](../capture_processing/demosaic.py#L203), [후보 보간](../capture_processing/demosaic.py#L241), [동질성 선택](../capture_processing/demosaic.py#L353), [타일 orchestration과 sample 복원](../capture_processing/demosaic.py#L404).

1. CFA 위치마다 측정된 값만 해당 RGB 채널에 두고 나머지는 0인 sparse planes를 만든다. 원본 가장자리 5픽셀 폭에서 각 누락 채널은 유효한 이웃 범위 안의 3×3 sample 평균으로 채운다. 이 border helper의 `_shift()`는 배열 밖을 0으로 두고 count에는 실제 존재하는 이웃만 더한다.
2. 각 tile의 후보 두 개에서 green을 각각 horizontal/vertical로 계산한다. 각 식은 `2*(양쪽 인접 green + center의 측정 sample) - 같은 CFA 색의 양쪽 2칸 sample 합`을 4로 나눈 값이다. 결과는 그 방향의 두 인접 green 값 사이 `[min,max]`로 제한한다.
3. green 위치의 누락 red/blue는 `기준 green + 0.5 × (양옆 누락색 합 - 양옆 후보 green 합)`으로 채우며 각각 `[0,65535]`로 clip한다.
4. red/blue 위치의 반대색은 네 대각 sample의 차이를 사용한다. `후보 green + 0.25 × (누락색 대각합 - 후보 green 대각합)`이며 역시 `[0,65535]` clip이다. 측정 CFA sample은 각 후보에서 그대로 복사한다.
5. 동질성 평가용으로만 candidate RGB에 camera-to-linear-sRGB matrix, 고정 linear-sRGB→D65 XYZ matrix를 적용한다. XYZ를 `65535 × D65 white`로 나누고 [0,1] clip한 뒤, 0.008856 임계의 cube-root/선형 piecewise Lab 함수를 계산한다. L 차이는 절댓값, ab 차이는 제곱합이다. 수평 후보의 좌우 및 수직 후보의 상하 차이에서 공통 임계값을 만들고, 네 방향의 homogeneous 개수를 합산해 3×3 이웃 동질성을 비교한다. 큰 후보를 채택하고 동률이면 두 후보를 평균한다.
6. 8픽셀 halo tile 계산을 위해 Bayer 입력은 `np.pad(..., mode="edge")`한다. 출력 이미지 전체에 최종 global clip은 없다. 가장자리 5픽셀은 앞서 만든 border RGB로 덮고, 마지막에 모든 측정 CFA 위치의 원본 값을 다시 넣는다.

따라서 출력용 AHD RGB 전체가 단순히 `[0,65535]`로 제한되는 것은 아니다. 해당 범위 clip은 green site의 red/blue 후보와 red/blue site의 opposite-color 후보에 적용되고, Lab 내부의 XYZ clip은 분기 평가용뿐이다. green 후보는 인접 green 사이에 제한된다. 정수 저장·rounding은 없으며 측정 sample 보존이 마지막에 강제된다. `_source_rgb_with_border()` 주석은 `dcraw border_interpolate(5)`를 언급하지만 실제 코드는 위 3×3 valid-neighbor 평균이다. tile 내부 candidate 계산의 edge padding과 border RGB 채움은 서로 다른 경로다. 연속 float32 적응이므로 LibRaw와 branch 결정까지 bit-exact하지 않다.

LibRaw reference는 별도 비교 경로다. `raw.postprocess()`에 AHD, gamma=(1,1), auto bright/scale off, camera/auto WB off, raw color, 16-bit, flip=0, black=0을 지정한다. 이 uint16 RGB는 main input이 아니고 같은 custom-AHD에서 얻은 homography로 warp해 차이를 기록하고, 보조 prediction 비교에만 사용한다. [LibRaw comparison](../capture_processing/raw.py#L128), [custom AHD와 비교 image 처리](../run_fabric_capture.py#L311), [보조 replay](../run_fabric_capture.py#L441).

## Marker geometry와 resampling

near/far 각 단일 DNG의 custom-AHD RGB에서 독립적으로 marker를 다시 찾는다. 검출 전 detector-only copy를 긴 변 최대 1600으로 줄이고, 전체 RGB 값의 1–99 percentile을 8-bit 회색 영상으로 바꾸어 `DICT_APRILTAG_36h11` detector에 넣는다. 최소 4 tag가 필요하다. 이 uint8 copy는 homography 좌표 검출에만 쓰며 resampling·photometry의 수치 입력으로 쓰지 않는다. [탐지와 display copy](../capture_processing/geometry.py#L218).

검출된 모든 tag corner를 모아 중심을 구하고, tag 첫 edge 각도의 중앙값으로 board axis를 잡는다. 회전 좌표에서 각 사분면에 속하는 corner를 구해 tag ring의 바깥 경계가 아닌 안쪽 사각형을 만든다. quad를 중심 기준 3.5% inset하고 source 좌표로 되돌린다. 이 float32 quad를 `(0,0),(512,0),(512,512),(0,512)`에 매핑하는 8계수 homography를 single precision `sgesv`로 푼다. manifest용 기록에는 tag ID/corners, 각도, quad, homography와 detector display percentiles가 저장된다. DNG metadata의 사전 `tagIDs`, `boardCenter`, 또는 preview PNG를 homography 대신 사용하지 않는다.

공간 처리는 float32다. homography의 역행렬을 single precision LAPACK으로 구해 512×512 목적 pixel center마다 source 위치를 역매핑하고, 네 이웃 bilinear 가중합을 만든다. source 밖은 0이며 valid mask는 false다. 이어 `46:466`을 잘라 420×420으로 만들고 crop의 모든 pixel이 valid인지 요구한다. 마지막은 픽셀 영역 중첩 가중치를 separable accumulation해 정확 area 방식으로 256×256 downsample한다. [homography와 bilinear warp](../capture_processing/geometry.py#L13), [warp 좌표와 보간](../capture_processing/geometry.py#L61), [area resize](../capture_processing/geometry.py#L125), [512→420→256 고정 순서](../capture_processing/geometry.py#L182).

현재 runner는 marker ID/corners/homography를 실제 실행 때 fresh detect 결과로 archive/manifest에 남길 준비가 되어 있다. 이 읽기 전용 감사에서는 homography, 최종 256좌표 corner RMS, valid support, 프레임 간 texture overlap, residual을 계산하지 않았다. Metadata 안의 capture-side tag list는 실제 fresh detector 결과로 취급하지 않는다.

## Active photometry condition

현재 활성 실행은 각 side의 256×256 linear count `D`에 DNG별 black `B`와 white `W`를 적용하는 `both` 하나만 사용한다. CFA 4개 black 값이 모두 같고 `0≤B<W`인지 확인한다. `white_only`와 `neither`는 폐지되어 active API가 거부한다. [조건 정의](../capture_processing/photometry.py), [level 검사와 적용](../run_fabric_capture.py), [수식 구현](../capture_processing/photometry.py).

| 활성 조건 | 수식 | B 사용 | W 사용 |
|---|---|---:|---:|
| `both` | `(D - B) / (W - B)` | 예 | 예 |

아래 모델 세부 항목은 보존된 이전 best_render 실행의 구현 감사 기록이다. 현재 활성 체크포인트/feature contract는 [author real pipeline contract](DNG_AUTHOR_REAL_PIPELINE_20261001.md)에서 확인한다.

감산 후 signed 값은 WB 전까지 보존된다. rawpy camera WB 첫 3개 양수 RGB gain을 최소값으로 나누어 최소 gain=1로 만들고, camera matrix 앞 3열을 사용해 `balanced @ matrix.T`로 linear sRGB를 만든다. camera description은 RGBG여야 하고 matrix는 finite·rank 3이어야 한다. gamma, ICC 변환은 없다. custom AHD 내부에서도 matrix는 Lab 기준용으로 호출되지만, AHD가 반환하는 RGB는 camera-space counts다. [WB/matrix metadata 읽기](../capture_processing/photometry.py#L65), [선형 색 변환](../capture_processing/photometry.py#L92).

near/far linear sRGB 각각은 signed, unclipped 배열로 먼저 보존된다. 두 배열을 HWC 6채널 `[near RGB, far RGB]`로 붙이고 `[0,1]` clip 후 `[1,6,256,256]` FP32 tensor로 만들어 inference에 넘긴다. 실제 user-facing count/색/clipping 퍼센트는 run 결과가 있어야 알 수 있다. runner는 preclip, clipped, delta, lower/upper fractions를 저장하도록 되어 있다. [input clip과 tensor layout](../run_fabric_capture.py#L419).

## 거리 gain, relation/log relation, 21 features

체크포인트 feature version은 `raw_calibrated_v2`다. model 상수 거리는 near=2.414, far=10이다. 이는 촬영 메타데이터의 실제 optical-axis 거리 near=0.10051784698351215 m, far=0.30422835898131617 m와 별도다. 현재 feature mode는 실제 optical 거리로 자동 보정하지 않고 고정 `gain=(10/2.414)^2` (약 17.1603)을 사용한다. `coefficientGeneration()`은 256×256 중심(127.5,127.5) 기준 반경을 128로 나누고 각 거리에 대한 `atan(r/distance)`로 angle을 계산한다. `coefficient=(cos(angle_far)-cos(angle_near))/cos(angle_near)`를 최대값으로 나누어 spatial relation denominator를 만들고, `time_co_map=cos(angle_near)/cos(angle_far)`를 만든다. [거리와 map 생성](../model/nfplight_model.py#L52), [계수 공식](../model/nfplight_model.py#L72), [checkpoint mode의 고정 gain](../model/nfplight_model.py#L160).

실제 condition input을 `near`, `far` 3채널씩 나눈 뒤 `far_gained = far × gain`을 만든다. Relation 쪽은 `far_time = time_co_map × far_gained`를 `[0,1]`에 clip하고 `signed_diff=far_time_clipped-near`, `abs_diff=|signed_diff|`, `mean_abs_diff=mean_RGB(abs_diff)` 순으로 진행한다. `relation_raw=mean_abs_diff/max(coefficient,1e-5)`이며 log는 `log(max(relation_raw,1e-5))`다. `ClipToOne()`는 각 입력 이미지에서 H/W 전체 min/max를 구해 두 map을 각각 `[0,1]`로 min-max normalize한다. 반환 relation과 log relation은 각각 `1-normalized_map`이다. 구형 `sample_v1` 및 `legacy_batch_v0`에는 center-pixel-derived gain 또는 batch-global normalization branch가 있지만 이 checkpoint는 그 branch를 사용하지 않는다. [현재 구현](../model/nfplight_model.py#L125), [계산 trace](../capture_processing/trace.py#L26).

`far_gained`도 `[0,1]` clip해서 appearance의 세 번째 image로 쓴다. 원래의 `far`와 `near`는 그대로 appearance에 포함된다. Appearance 9채널 `[near(3), far(3), far_gained_clipped(3)]`에 각 채널 `log_normalization(x)=(log(x+0.01)-log(0.01))/(log(1.01)-log(0.01))` 9채널, relation 1채널, log relation 1채널, mask 1채널을 이어 총 **실제 21 feature**를 만든다. 마지막에 `2*x-1`을 적용한다. mask는 각 pixel에서 near RGB 또는 gained/clipped far RGB 중 어느 최대값이라도 `>0.95`이면 0, 아니면 1이다. [log map와 mask](../model/nfplight_model.py#L107), [21채널 조립](../model/nfplight_model.py#L125), [독립 trace와 equality 확인](../capture_processing/trace.py#L33).

`trace.py`는 `far_gain_map`, unclipped/clipped far gain, gain clip delta, time map 및 time clip, signed/absolute 차이, mean 차이, raw/normalized coefficient, denominator, raw/floored/log/normalized relation, valid mask, near/far/gained log appearance와 최종 21 feature를 따로 내보낸다. `build_features()` 결과와 feature tensor, log relation, aligned far가 bitwise `torch.equal`인지 점검한다. 입력 및 모든 중간 trace는 FP32, 유한값이어야 한다. 이 감사는 이 식들을 코드에서 추적했을 뿐 값이나 분포는 계산하지 않았다.

## 체크포인트, 네트워크 shape, prediction semantics

Runner는 체크포인트 SHA256을 `6b1afb28a14b39930736bd7d29da438b797fd770bf0a3042fb809c5919d8a3cd`와 비교한다. 전달된 read-only 체크포인트 점검에서 실제 해시가 일치했고 `feature_version=raw_calibrated_v2`, `normal_head=xyz`로 확인됐다. Adapter는 `TwoBranchNet`을 만들고 FP32 device로 옮긴 뒤 checkpoint state dict prefix `module`/`_orig_mod`를 정리해 `strict=True`로 로드한다. `torch.load(..., map_location="cpu", weights_only=False)`는 모델 초기화 중 호출된다. runner는 `image_size=256`, `input_mode="linear_rgb"`, `capture_precision=True`를 넘긴다. 해당 모델의 실행은 `eval()` + `torch.no_grad()`이며, capture run은 AMP/autocast/TF32를 사용하지 않도록 설계됐다. 이 감사에서 모델을 초기화하거나 forward하지 않았다. [checkpoint SHA 및 adapter args](../run_fabric_capture.py#L31), [strict load](../model/nfplight_matsynth_capture_model.py#L69), [state dict 정규화와 metadata](../model/nfplight_matsynth_capture_model.py#L25), [eval inference](../model/nfplight_matsynth_capture_model.py#L99), [FP32 model setup](../model/nfplight_model.py#L25).

고정 256 입력은 이미 16의 배수라 network padding으로 shape가 변하지 않는다. `TwoBranchNet`의 21채널은 18채널 albedo branch input과 3채널 specular branch input으로 분리된다. 두 branch는 각자 width=32로 시작하고 encoder block 수는 `[2,2,4,8]`, downsample 후 channels는 `[64,128,256,512]`다. 두 bottleneck은 512채널·16×16에서 12 NAFBlock씩 처리한다. NAFBlock 안에서는 LayerNorm, pointwise expand, depthwise convolution, SimpleGate, channel attention, residual beta와 gated FFN, residual gamma가 사용된다. [NAFBlock](../network/nfplight_net.py#L14), [branch/encoder 정의](../network/nfplight_net.py#L69).

| 단계 | Albedo branch | Specular branch |
|---|---|---|
| 입력/intro | 18 → 32 @ 256×256 | 3 → 32 @ 256×256 |
| encoder 0 | 2 blocks, 32 @ 256² → down 64 @ 128² | 동일 |
| encoder 1 | 2 blocks, 64 @ 128² → down 128 @ 64² | 동일 |
| encoder 2 | 4 blocks, 128 @ 64² → down 256 @ 32² | 동일 |
| encoder 3 | 8 blocks, 256 @ 32² → down 512 @ 16² | 동일 |
| middle | 12 blocks, 512 @ 16² | 12 blocks, 512 @ 16² |
| decoder | PixelShuffle up + matching skip; 256@32² → 128@64² → 64@128² → 32@256², 각 stage 2 blocks | 두 middle feature concat 1024@16²; matching 두 branch skip를 concat해 512@32² → 256@64² → 128@128² → 64@256², 각 stage 2 blocks |
| map head | 32 → 6, Tanh | 64 → 4, Tanh |

두 head 출력은 diffuse 3, specular 3, roughness 1, normal 3으로 재배열되어 `[B,10,256,256]`를 반환한다. Network 내부 출력의 저장 순서는 **normal(0:3), diffuse(3:6), roughness(6:7), specular(7:10)**다. [decoder와 10채널 순서](../network/nfplight_net.py#L136), [forward shape/concat](../network/nfplight_net.py#L170).

현재 checkpoint의 `xyz` normal head는 첫 3채널을 identity로 둔다. 따라서 runner의 `prediction`은 Tanh를 거친 network raw-10 tensor 그대로이며, 여기서 normal을 단위화하거나 diffuse/specular을 `[0,1]` 물리 map으로 역변환하거나 roughness floor를 적용하지 않는다. `save_svbrdf_maps()`는 별도 물리 map export 경로지만 runner는 호출하지 않는다. 저장하는 것은 `prediction_nchw`와 HWC로 바꾼 `prediction`이다. 즉 이 보고서에서 “10 prediction channels”는 물리 parameter map으로 후처리되기 전의 normalized network output을 뜻한다. [normal-head identity](../model/normal_head.py#L17), [prediction assignment](../model/nfplight_matsynth_capture_model.py#L99), [별도 map export](../model/nfplight_model.py#L204), [runner save keys](../run_fabric_capture.py#L456).

## 초기화 과정의 추가 상태와 runner 검증 경계

`NFPLightModel.init_rendering()`은 coefficient 이외에도 renderer 상태를 만든다. `utils/render_util.py::svBRDF`의 `lampIntensity`는 16이다. `surface(256,1)`은 x/y `linspace(-1,1,256)`와 meshgrid로 `[1,3,256,256]` FP32 위치 tensor를 구성하며 x는 오른쪽으로 증가, y는 아래로 감소, z는 0이다. near/far 위치는 각각 `[0,0,2.414]`, `[0,0,10]`이다. `torch_generate()`는 위치 차이의 제곱합으로 squared light distance `[1,1,256,256]`, `v/(sqrt(sum(v²))+1e-12)`로 light/view direction `[1,3,256,256]`을 만든다. 이 buffers와 identity map은 모델 초기화 때 생성되고 dtype 검사를 받는다. 현재 capture의 `test()`는 renderer `_render()`를 호출하지 않는다. 따라서 lamp intensity, GGX/Smith/Fresnel rendering은 이번 DNG 입력→추론 연산에 포함되지 않는다.

현재 runner는 `model.feed_data()`를 호출하지 않고 이미 FP32로 변환·clipping한 tensor를 `model.inputs`에 대입한다. 따라서 adapter의 tolerance `1e-5` 검사와 추가 `.clamp()`는 이번 경로에서 실행되지 않는다. 대신 feature trace와 `build_features()`가 shape `[B,6,256,256]`, 유한값, `[0,1]`을 검사한다. `net_g.eval()`과 `torch.no_grad()`는 `test()`에서 적용한다. 별도 trace는 `no_grad()` 바깥에서 호출되지만 입력과 기하 상수가 gradient를 요구하지 않으므로 최종 inference graph를 학습하지 않는다. NAFBlock dropout 기본값은 0이므로 Identity다. LayerNorm은 각 픽셀의 채널 평균과 분산을 사용하고 epsilon은 `1e-6`이다. 네트워크 intermediate activations에는 전역 범위 제약이 없으며 output Tanh와 input feature rescaling만 규정된 범위를 갖는다.

정밀도 표기의 경계도 구분한다. 영상 배열·보간 가중치·모델 parameter/buffer·feature tensor의 계산 dtype은 FP32다. 파일명/hash, CFA index, mask·homogeneity count는 문자열/정수/bool이다. `near_distance=2.414`, gain 식 `(10/2.414)**2`, `math.log(0.01)` 같은 Python scalar 상수 및 JSON 통계는 Python float를 사용하고, 영상 tensor를 구성하거나 연산에 적용할 때 FP32로 변환된다. 따라서 “모든 scalar 계산까지 binary32”라고 주장하지 않는다. 마커 검출에는 uint8 복사본과 OpenCV 내부 구현이 있고, 행렬 해석/warp의 지정된 경로는 single LAPACK와 FP32 arrays다. NumPy 및 OpenCV의 빌드 내부 구현 전체를 이 감사로 검증한 것도 아니다.

환경 확인값은 NumPy 2.3.5, SciPy 1.16.2, rawpy 0.26.0, 실제 LibRaw runtime 0.22.0, OpenCV 4.11.0, matplotlib 3.10.7, PyTorch 2.5.1+cu121, RTX 3090이다. custom AHD의 기록된 source reference는 LibRaw 0.21.4이며 비교 경로는 설치된 0.22.0 runtime을 사용한다. 둘의 차이를 정수 rounding만으로 설명할 수 없다.

## 2026-10-01 후속 실행 규약

사용자는 새 `data`의 다섯 묶음 각각을 선택했고, 공통 마커로 각 side의 frame 00에 정렬한 모든 프레임의 **최종 256 좌표 corner RMS <=1px**, crop 전체 유효, 메타데이터 호환, 무늬 중첩 시각검사를 통과할 때만 near/far 각각 5장 평균하도록 확정했다. 하나라도 실패하면 해당 묶음은 near_00/far_00 한 쌍으로 처리하고 실패 원인을 기록한다. 평균은 정렬된 camera RGB count에서 FP32로 계산하고, 그 뒤 기존 세 photometry 조건을 적용한다. 묶음 간 평균은 없다. 이 규약은 위 감사 시점의 단일 프레임 코드에서 후속 구현으로 변경할 부분이며, 최신 계약은 `CAPTURE_REQUIREMENTS.md`를 따른다.

원본 header 50개를 추가로 읽어 각 side의 5장 사이 shape/CFA/origin/black/white/WB/matrix/orientation 동일성을 확인했다. 각 묶음의 기록된 ISO=100.15123, shutter=0.049862s, torchIntensity=0.05도 동일하다. 이는 기록값의 호환성 검사이며 실제 광량이 완전히 동일함을 증명하지 않는다. 이 점검에서도 AHD나 inference는 실행하지 않았다. 새 산출물은 `artifacts/fabric_capture_20261001_clean/` 아래 묶음별로 작성하며, 이전 산출물은 수치 입력으로 사용하지 않는다.

원본 DNG EXIF도 Pillow의 `getexif().get_ifd(34665)`로 50개 모두 header만 읽었다. 각 burst 내부 및 near/far의 ExposureTime=0.05s, FNumber=1.6, ISOSpeedRatings=100, FocalLength=4.2mm가 동일했다. 촬영 JSON의 applied shutter=0.049862s/ISO=100.15123와 EXIF 기록값은 정밀도·기록 방식이 다르므로 두 값을 구분한다. 어느 쪽도 실제 photon flux 동일성의 측정은 아니다. 이 값을 이용해 기존 모델 거리 gain이나 입력 밝기를 추가 보정하지 않는다. orientation/EXIF/DNG의 다른 tag는 raw unpack에서 읽거나 명시적으로 사용한 항목 외에는 후처리 transform으로 자동 적용하지 않는다.

## 기록 예정 산출물과 미측정 항목

Runner가 실행되면 frame별 raw/cast/AHD/spatial 및 LibRaw comparison 배열을 `source_frames/*.npz`, 공유 selected-frame 수치배열을 `source_arrays.npz`, 각 photometry 조건별 중간값·feature trace·prediction을 `<condition>/arrays.npz`에 보존한다. 각 NPZ를 다시 열어 exact array equality를 확인하고, manifest에 source/checkpoint hashes·metadata·dtype·clipping 통계·거리·조건 formula 등을 쓴다. 표시용 PNG는 별도 폴더에 percentile range로 정규화된 preview이고 모델 입력으로 되돌려 쓰지 않는다. [frame archive와 roundtrip](../run_fabric_capture.py#L109), [공통 배열](../run_fabric_capture.py#L400), [condition archive](../run_fabric_capture.py#L456), [manifest](../run_fabric_capture.py#L538), [display/report](../run_fabric_capture.py#L178).

이 문서 작성 과정에서 새 capture 또는 inference를 실행하지 않았으므로, 아래 값은 아직 보고되지 않는다: per-condition linear-sRGB·far-gain·far-time·relation·feature·prediction 값과 clipping fraction, 새 detector corner/homography, frame 간 RMS residual과 valid support, texture overlap, 평균 전후 marker 품질, 10개 prediction 채널의 actual min/max. 보고서의 code path와 source metadata만으로 재료 정확도/registration 성공을 주장할 수 없다. runner 자체도 no-GT limitation을 manifest에 남긴다. [runner limitation](../run_fabric_capture.py#L588).


## 후속 구현의 burst 정렬·선택 경로

`capture_processing/burst.py::prepare_burst()`는 각 side의 00–04 원본을 새로 unpack/AHD/detect하고 frame 00을 기준으로 삼는다. `register_marker_frames_to_first()`는 공통 tag ID가 최소 4개인 경우 모든 tag의 네 corner를 사용한다. 이상점을 제거하거나 일부 프레임만 골라 평균하지 않는다. `geometry.py::fit_homography_lstsq()`는 Hartley 좌표 정규화 후 FP32 LAPACK `sgels` least-squares로 current raw 좌표→frame00 raw 좌표 homography를 계산한다. 최종 transform은 `H00 @ Hrelative`이며 원본 AHD 영상에 한 번만 warp한다. frame00은 identity relative transform이다. 마커 residual은 두 좌표를 각각 warp512 평면에 투영하고 `(coordinate−46)*256/420`으로 변환한 뒤, 모든 공통 corner의 Euclidean squared distance 평균에 제곱근을 취한 RMS다. 최대 corner residual도 별도 기록한다.

Burst metadata signature는 raw shape, CFA pattern/visible phase/color index, color description, visible origin, black/white, WB, camera matrix, orientation, capture settings와 DNG EXIF exposure를 정확히 비교한다. 모르는 EXIF는 평균 gate 실패다. 매 프레임의 crop420 전체 bilinear support 유효성과 RMS <=1px를 요구한다. 어느 한 side라도 실패하면 near/far 모두 frame00을 선택한다. 수치 gate가 통과하면 준비 단계는 평균 후보를 표시 자료로만 만들고 실제 numerical 선택은 `root_review.json`에 기록된 시각검사 후 진행한다. 이 기록은 에이전트의 실제 이미지 검사이며 사람의 검증이나 재질 정합의 수학적 증명을 뜻하지 않는다.

`run_fabric_capture.py`의 `--prepare-only`와 `--prepared`는 단계를 분리한다. 준비 manifest, 모든 DNG/metadata/code/NPZ SHA, root inspection 이미지 SHA가 유효해야 추론한다. 평균 승인 시 near와 far를 각각 정렬된 `output256` camera RGB count에서 동일 가중치 FP32로 평균한 다음 세 photometry 조건을 적용한다. fallback 시 frame00 배열 그대로 사용한다. 준비된 이번 실행의 배열을 이어 사용하며 과거 결과를 수치 입력으로 사용하지 않는다. 실패 이유와 선택 frame 목록은 그룹 manifest에 남긴다. LibRaw 비교 배열도 같은 선택·정렬·평균을 적용한다. 네트워크, checkpoint, 거리 gain과 relation 수식은 변경하지 않았다.

정렬 overlay·색상표·HTML/PNG/SVG 생성의 표시 전용 계산에는 NumPy float64 또는 Matplotlib 내부 dtype이 포함될 수 있다. 이 값은 모델 입력으로 돌아가지 않는다. 원본 Bayer uint16, detector uint8, mask/index 정수, 영상 처리/선택 평균/photometry/features/model FP32의 구분을 유지한다. 실행 결과는 이 문서의 사전 감사 시점 설명과 별도로 아래 후속 결과 기록 및 각 그룹 manifest에서 확인한다.


## 추가 표시 규약: WB 이후 RGB와 gamma 토글

사용자의 후속 요청에 따라 사진형 RGB 미리보기는 모두 WB 이후의 linear RGB로 표시한다. raw camera RGB count·compensated 단계는 표시용 복사본에 정규화 WB 대각행렬과 camera→linear RGB matrix를 적용하고, 이미 whitebalanced 단계는 matrix만, 이미 linear RGB 단계는 둘 다 추가 적용하지 않는다. 원본 Bayer mosaic는 WB 이전 수치 배열로 보존하며, 사진형 preview는 대응하는 WB 처리 demosaic RGB임을 표시한다. 각 단계의 수치 통계는 원래 저장 배열 기준이며 preview 값과 혼동하지 않는다. 보고서는 WB 대각행렬과 camera matrix를 명시한다.

`gamma=2.2` 토글은 표시 RGB의 `clip(normalized_linear_rgb,0,1)^(1/2.2)` 변환을 선택한다. 기본값은 gamma off이다. 단순 CSS 근사나 모델 입력 gamma 변경을 뜻하지 않는다. relation·log relation·signed difference colorbar·feature/prediction scalar 채널은 수치 map으로 유지하며 gamma를 적용하지 않는다. raw NPZ, 모델 입력, checkpoint 및 inference 수식은 변경하지 않는다. 앞서 사용하던 표시용 sRGB OETF 계약은 이 후속 사용자 요청에 맞춰 사진형 preview의 gamma off/2.2 선택 계약으로 교체한다.


## 실제 새 데이터 준비 결과

50개 DNG를 새로 처리했다. 다섯 묶음 모두 near burst는 corner RMS 기준을 통과했고, far burst에 1px 초과 프레임이 있어 묶음 전체가 첫 한 쌍 선택이다. 부분 프레임 평균이나 임계값 완화는 하지 않았다. 원본·메타데이터·코어 구현의 전후 해시와 NPZ 재읽기 검증은 각 preparation.json에 기록됐다. 아래 수치는 각 side의 5프레임 중 최대 RMS이며 frame00 RMS=0을 포함한다.

| 묶음 | near 최대 RMS (256px) | far 최대 RMS (256px) | 선택 |
|---|---:|---:|---|
| 260818_174845_478 | 0.473579 | 1.289965 | near_00 / far_00 |
| 260818_175005_078 | 0.506768 | 1.159251 | near_00 / far_00 |
| 260923_141241_844 | 0.575712 | 1.397419 | near_00 / far_00 |
| 260923_161825_503 | 0.521310 | 1.378969 | near_00 / far_00 |
| 260930_152732_008 | 0.574958 | 1.083380 | near_00 / far_00 |


## 최종 추론·독립 검증 결과

다섯 묶음 모두 새로 준비한 FP32 source 배열에서 기존 checkpoint로 세 조건(`white_only`, `neither`, `both`)을 추론했고 그룹 manifest의 `status=ok`를 확인했다. 모두 near_00/far_00 한 쌍이며 numerical burst mean은 생성되지 않았다. 원본 DNG·metadata·checkpoint는 변경되지 않았다. 각 실행 구간의 inference code 전후 hash 검사도 통과했다. 표시 변경에 앞서 끝난 세 그룹의 수치 NPZ 12개는 WB/gamma 표시 갱신 후에도 같은 SHA를 유지했다. 그 세 그룹의 원래 준비/검사 증거는 `review_at_inference/`에 보존되며 최신 표시 검사는 별도 `display_visual_review`로 구분한다.

`verify_prepared_arrays.py`의 독립 검증은 50개 프레임의 원본/NPZ hash, dtype·shape·유한성, uint16→FP32 정확 캐스트, measured CFA sample 보존을 확인했다. 최종 다섯 그룹의 공유 선택 배열도 각각 12개 항목이 frame00 배열과 정확히 일치했다. `verify_formulas.py`는 생산 helper나 새 model forward를 사용하지 않고 NumPy FP32 식으로 photometry/WB/matrix, geometry coefficient, gain·time·RGB difference·relation/log relation, mask·21 feature concat 및 prediction layout을 재확인했다. 그룹별 87개, 총 435개 비교가 통과했으며 CPU/GPU log 및 reduction의 작은 차이는 검사별 absolute tolerance와 실제 최대 오차로 기록했다. 이는 재료 추정 정확도 검증이나 GT 비교를 뜻하지 않는다.

통합 보고서는 `artifacts/fabric_capture_20261001_clean/index.html`이다. 각 그룹의 `report.html`은 WB 이후 RGB, gamma off/2.2 선택, relation 계산 중간값, 21 feature 채널과 raw10 prediction을 제공한다. `prepare_review.html`은 원래 크기 256 crop 5장을 한 줄(1280px)로 보존하며 WB/gamma 선택과 scalar 정렬 차이를 구분한다. 브라우저에서 5개 결과 페이지와 5개 준비 페이지의 gamma 이미지 로딩, relation의 고정 scalar 범위, Bayer numeric stats와 WB-demosaic 대체 표시를 확인했다. 실행/독립 검증의 machine-readable 증거는 그룹별 manifest/verification/independent JSON과 통합 검증 index에 있다.
