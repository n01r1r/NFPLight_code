# 실행과 결과 확인

학습·Drive 입력은 [Colab 안내](COLAB.md)를 따른다. 아래는 DNG 전처리와 명시적 author profile 추론이다. 저자 weight는 별도로 확보해야 한다.

## 1. Weight와 profile 고르기

[README의 weight 표](../README.md#weight와-profile-선택)에서 저자 원본 weight와 profile을 함께 지정한다. `net_g_real.pth`는 `author_real.json`, `net_g_syn.pth`는 `author_syn.json`을 사용한다. CLI 기본 모델은 없다.

구조가 같아서 weight를 읽을 수 있다는 것만으로 계산이 같은 것은 아니다. Profile은 입력 채널, relmap, 조명 거리와 normal decode도 결정한다. Metadata와 profile이 충돌하면 코드가 거부하며, metadata가 없는 weight는 사용자가 맞는 `--config`를 지정해야 한다.

## 2. DNG를 정합하고 검토하기

촬영 폴더의 `metadata.json`에 near와 far 각각 5개 DNG 파일명을 `filenames`로 기록한다. 촬영 폴더 밖의 새 준비 폴더를 지정한다.

```powershell
python preprocess_dng.py prepare --source <촬영묶음> --out <새준비폴더>
```

기본 `configs/capture.json`은 `near_00`의 target 사각형 하나를 기준으로 전체 near/far burst를 정합한다. 원본 FP32 AHD RGB에서 warp512 → area256을 수행하고, 각 프레임의 센서 count와 검토 자료를 저장한다. 태그 조건과 과거 crop46/420 경로의 차이는 [DNG 처리 설명](../README.md#dng에서-입력-만들기)을 따른다.

Samsung Camera2 DNG의 exposure EXIF는 Pillow로 먼저 읽고, Pillow가 식별하지 못한 TIFF/RAW 형식은 `tifffile`의 main IFD에서 확인한다. 노출시간·조리개·초점거리 rational 값과 ISO를 검증하며, 누락되거나 잘못된 태그를 기본값으로 채우지 않는다.

준비 폴더에서 다음을 확인한다.

- `target_boundary.png`: 사각형이 재질 영역을 올바르게 잡았는지 확인한다.
- near/far PNG 10개: 프레임 사이에 무늬가 같은 위치에 놓이는지 확인한다. PNG는 검토용이며 추론 입력이 아니다.
- `preparation.json`: 프레임별 처리 오류, 정합 판정, 전체 crop 유효성과 `eligible`을 확인한다. RMS만 보고 평균 가능 여부를 판단하지 않는다.

`review.example.json`을 `review.json`으로 복사하고 실제 판단을 기록한다.

|필드|기록할 값|
|---|---|
|`status`|`reviewed`|
|`target_boundary_passed`|기본 경로의 target 경계 검토 결과; 통과했을 때만 `true`|
|`texture_registration_passed`|무늬 정합 검토 결과 `true` 또는 `false`|
|`note`|판단 이유를 적은 비어 있지 않은 문자열|
|`preparation`|검토한 파일의 identity를 그대로 유지|

자동 검출 성공을 사람의 검토 통과로 기록하지 않는다.

## 3. 검토한 입력 저장하기

```powershell
python preprocess_dng.py finalize --prepared <준비폴더> --review <review.json> --out <새pair.pt>
```

기본 경로는 아래 순서로 판정한다.

|조건|처리|
|---|---|
|Target 경계 검토 미통과, 검토 형식 오류 또는 파일 identity 불일치|입력 생성 거부|
|`near_00` 또는 `far_00`에 처리 오류가 있거나 전체 crop이 유효하지 않음|입력 생성 거부|
|양쪽 모든 프레임의 `eligible=true`이고 무늬 검토 통과|near 5장과 far 5장을 각각 평균|
|첫 프레임은 유효하나 평균 조건 미충족|양쪽 모두 `near_00`/`far_00` 사용|

선택한 센서 count를 먼저 평균하고, 각 쪽의 첫 프레임 보정값으로 black/white 보정(`both`) → WB → camera-to-linear-sRGB → [0,1] clip을 수행한다. [세부 판정과 보정 수식](../README.md#3-프레임-선택과-밝기색-보정)은 README에 있다.

생성된 `pair.pt`에는 near 다음 far 순서의 `[1,6,256,256]` FP32 선형 RGB, 전체가 true인 bool `valid_mask`, clip 전 `unclipped_pair`, 프레임 선택·검토·보정 이력이 들어간다. 준비 단계의 count 배열이나 검토용 PNG를 대신 넣지 않는다.

## 4. FP32 추론하기

```powershell
python -m nfplight infer --config configs/author_real.json --checkpoint pretrained/net_g_real.pth --input <pair.pt> --out <새prediction.pt> --device cpu
```

`<...>`를 실제 경로로 바꾸고 아직 없는 출력 파일을 지정한다. GPU에서는 `--device cuda`를 사용한다. 다른 저자 weight의 실행 명령은 [README의 추론 예시](../README.md#추론-실행)를 따른다.

CLI는 입력을 검사하고 weight를 strict load한 뒤 다음 순서로 처리한다.

1. 저장된 선형 RGB의 중앙 20×20 패치에서 RGB 전체 평균을 구해 [밝기 보정 K](../README.md#밝기-보정-k는-어떻게-계산하는가)를 계산한다. 실측에 학습 gain9를 적용하지 않는다.
2. K와 profile의 각도·거리 항으로 [relmap과 입력 특징](../README.md#relmap은-어떻게-계산하는가)을 만든다. Appearance용 far와 relmap용 far의 보정 항은 서로 다르다.
3. FP32·eval 모드로 추론하고 profile에 맞게 normal을 decode한다. AMP/TF32와 denoiser는 사용하지 않는다.

직접 Python을 호출할 때의 경로도 `load_model` → `Features(..., purpose='capture')` → model → `decode`다. [추론 CLI](../nfplight/cli.py)와 [특징 코드](../nfplight/features.py)를 기준으로 한다.

## 5. 출력 확인하기

출력 `.pt`는 N3/D3/R1/S3 순서의 tensor와 trace를 담은 dictionary다. [전체 key 설명](../README.md#저장-결과-읽기)을 참고해 다음을 확인한다.

- `config`, `checkpoint_identity`, `input_identity`, `input_spatial_policy`: 의도한 profile·weight·입력·영역 정책인지 확인한다. Identity는 경로·크기·mtime 정보다.
- `trace['gain']`, `near_patch_mean`, `far_patch_mean`, `gain_floor_active`: K와 중앙 패치 평균, 분모 하한 적용 여부를 확인한다.
- `features`, `raw_prediction`, `prediction`: 모델 입력과 decode 전후 출력을 구분한다. 저자 profile은 raw tanh 출력을 유지한다. 저장값을 곧바로 [0,1] 이미지로 해석하지 않는다.

`trace['aligned_far']`는 K만 곱하고 clip한 far다. `relation`·`log_relation`은 최종 특징의 `2*v-1` 변환 전 map이다.

