# NFPLight clean core

기존 weight와 FP32 입력·계산 규약을 유지하는 독립 모델·추론·학습 코어다.
기본은 **7.61M / width32 / encoder [2,2,4] / middle4 / decoder [2,2,2] / dropout .1**이다.
모든 모델 구조는 config에서 읽는다. 기존 연구 실행이나 원본 파일을 자동으로 수정하지 않는다.

## 구조

```text
nfplight/          모델, 특징, 렌더러, 데이터, checkpoint, 학습, 센서 수치 처리
configs/           모델 profile, 학습·실행·DNG 설정
tests/             결정적 계약 검사와 기존 코드 비교
docs/              계약, 감사, 실제 검증 결과
preprocess_dng.py   DNG 준비 → 검토 → FP32 입력 저장의 단일 진입점
verify_compatibility.py   기존 저장소를 읽기 전용 기준으로 사용하는 호환성 검사
```

## 설치와 검사

PyTorch는 환경에 맞는 CPU/CUDA 배포판을 먼저 설치한다.

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

Formatter, 자동 인덱싱, SHA-256 검증은 실행하지 않는다.

## 모델 config

|Config|입력 특징|Relmap|Normal 출력|
|---|---|---|---|
|`configs/7m_spatial.json` (기본)|21ch, near4/far12|공간 거리 보정|unit XYZ|
|`configs/7m_original.json`|21ch, near4/far12|수정 전|unit XYZ|
|`configs/7m_c.json`|21ch, near4/far12|공간 거리 보정|unit XYZ|
|`configs/author_real.json`|33ch, near4/far12|수정 전|원래 tanh 출력|
|`configs/author_syn.json`|21ch, near2.414/far10|수정 전|원래 tanh 출력|

`network.dim`, `encoder_depths`, `middle_depth`, `decoder_depths`, `appearance_channels`, `dropout`로 구조를 구성한다.
21/33 채널 순서는 고정 입력 형식이다. 구조를 바꾸면 기존 weight strict load가 실패하는 것이 정상이다.
가중치 이름과 parameter 등록 순서는 기존 optimizer 호환성 때문에 유지한다.
Metadata가 없으면 명시된 profile을 사용하고, metadata가 profile과 충돌하면 거부한다.
저자 profile은 denoiser를 사용하지 않는 현재 승인 경로다. 저자 원본 전체 파이프라인 재현이라고 해석하지 않는다.

## DNG 전처리

```powershell
python preprocess_dng.py prepare --source <촬영묶음> --out <새준비폴더>
```

촬영 폴더의 `metadata.json`에 near/far 각각 5개 DNG filename이 있어야 한다.
기본은 FP32 continuous-AHD → near_00 target 사각형·공통 태그 정합 → warp512 → exact-area256이다. 기존 crop46/420은 capture_legacy.json에서만 사용한다.
센서 연산은 검증된 기존 수식을 보존했다. 표시용 PNG는 검토에만 사용한다.

생성된 10개 PNG와 `preparation.json`의 RMS·crop 결과를 검토한다. `review.example.json`을 새 `review.json`으로 복사하고
`status`를 `reviewed`, `texture_registration_passed`를 true/false, `note`를 실제 판단으로 채운다. 새 기본 경로는 `target_boundary.png`도 확인하고 `target_boundary_passed`를 true로 설정해야 한다.
`preparation` identity는 검토한 파일의 값을 유지한다. 검토를 자동 통과시키지 않는다.

```powershell
python preprocess_dng.py finalize --prepared <준비폴더> --review <review.json> --out <새pair.pt>
```

양쪽의 모든 수치 조건과 무늬 검토가 통과하면 5장 평균, 실패하면 양쪽 frame00을 사용한다.
Frame00 자체가 유효한 전체 crop이 아니면 거부한다. 평균 뒤 `both=(D-B)/(W-B)`·WB·camera-to-linear-sRGB를 적용한다.
`pair.pt`에는 `[1,6,256,256]` FP32 선형 RGB, bool mask, unclipped RGB, 프레임 선택·보정·검토 이력을 담는다.
경로·크기·mtime로 변경을 감지하며 암호학적 내용 검증을 주장하지 않는다.

## 추론

```powershell
python -m nfplight infer --input <pair.pt> --checkpoint <weight.pt> --out <새prediction.pt> --device cpu
python -m nfplight infer --config configs/author_real.json --input <pair.pt> --checkpoint <net_g_real.pth> --out <새prediction.pt>
```

실측 gain은 중앙 `[118:138,118:138]` RGB 평균 near/far 비율이다. 학습 gain9를 실측에 적용하지 않는다.
출력은 N(3)/D(3)/R(1)/S(3) 순서의 raw/decoded tensor와 특징·gain trace를 포함한다.
직접 Python을 사용하는 경우 `Features(config['features'], device = ...)`에 FP32 쌍과 mask를 전달한다.
`load_model` → `Features(..., purpose = 'capture')` → network → `decode`가 공개 추론 경로다.

## 일부 태그 누락과 target 영역

`configs/capture.json`은 새 기본값 `robust_target`이다. 기준 near_00에서 edge와 직선 경계로 target 사각형을 찾고, 검출된 공통 태그로 모든 프레임을 그 영역에 맞춘다. 종이를 제거하기 위한 고정 46px crop을 적용하지 않는다. 검출·검토용 영상만 uint8이며 입력 측정은 원본 FP32 AHD에서 수행한다.

- 최소 3개 공통 태그와 양 축의 충분한 분포가 필요하다. 전체 태그를 먼저 사용하고, 오검출이 있으면 일관된 최대 집합을 찾아 사용·제외 ID와 잔차를 기록한다.
- 1px 기준을 넘으면 평균을 금지하고, 기하가 유효할 때 near_00/far_00으로 fallback한다. 기하가 불충분하거나 target 경계가 검출되지 않으면 거부한다.
- `target_boundary.png`의 빨간 사각형을 확인하고 review JSON에 `target_boundary_passed: true`를 기록한다. 무늬 정합 검토도 필요하다. 자동 검출을 사람의 검토로 표시하지 않는다.
- 과거 입력을 정확히 재현할 때는 `prepare --config configs/capture_legacy.json`을 사용한다. 새 기본 영역은 시야와 예측값이 달라질 수 있으며, 과거 결과와 동일하다고 주장하지 않는다.

## 학습과 이어 학습

```powershell
python -m nfplight train --manifest <manifest.jsonl> --out <새학습폴더> --device cuda
python -m nfplight train --resume <기존last.pt> --out <새이어학습폴더> --device cuda
```

새 학습은 `configs/training.json`의 NDRS 합·ReduceLROnPlateau·map/render 정책과 `configs/runtime.json`을 사용한다.
C 이어 학습은 checkpoint의 loss weight10·Adam·scheduler·RNG·평가·종료 상태와 승인된 continuation cap을 그대로 읽는다.
`best_render.pt` 같은 model-only checkpoint는 resume가 아니다. Resume에는 전체 상태가 있는 `last.pt`가 필요하다. 과거 checkpoint는 같은 폴더의 `runtime.json`도 필요하며 Python·PyTorch·CUDA·장치 환경이 다르면 정확한 이어 학습을 거부한다.
과거 relation optimizer, 입력 identity 불일치, runtime 과학 설정 불일치, smoke→연구 학습 전환을 거부한다.
`configs/legacy_resume.json`은 과거 구현의 복원 규약이며 새 실험 설정으로 수정하지 않는다.

`--smoke-updates N`은 별도 출력에서 최대 N update를 수행하는 구현 검사다. 실제 데이터·연산을 사용하므로 가벼운 CPU unit test와 다르다.
평가 population을 임의 축소하지 않는다. 이 작업에서 실제 연구 학습이나 GPU production continuation은 실행하지 않았다.
기존 source identity는 checkpoint의 역사적 identity로 보존하며 새 코드 identity를 `implementation.json`에 별도 기록한다.
원래 실행 폴더를 덮어쓰지 않는다. 기존 상태를 재개하더라도 항상 새 출력 폴더를 사용한다.
메모리 부족은 마지막 checkpoint로 되돌아가 중단하고, 디스크 부족·오류는 기존 저장 checkpoint를 보존한다.
상한 도달은 pause이며 수렴이나 강제 단계 전환이 아니다.

## 기존 명령 전환과 검증

|기존|새 진입점|
|---|---|
|run_fabric_capture.py의 준비·입력 생성|preprocess_dng.py prepare / finalize|
|날짜별 실측 모델 로더|python -m nfplight infer --config ...|
|full_dataset_suite·C 전용 runner|python -m nfplight train --resume ...|
|과거 HTML/PDF 보고서 생성기|활성 코드에서 제외; 기존 checkout/Git 이력에 보존|

```powershell
python verify_compatibility.py --legacy-root <기존저장소> --out <새검증.json>
python tests/verify_training_compatibility.py --legacy-root <기존저장소> --out <새학습검증.json> --device cuda --checkpoint <현재last.pt>
python tests/verify_contract_compatibility.py --legacy-root <기존저장소> --out <새계약검증폴더>
python tests/verify_capture_compatibility.py --source-root <기존data> --prepared-root <새전처리폴더> --out <새DNG검증.json>
```

검증기는 과거 파일 경로를 비교 기준으로만 사용하며 코어의 실행 의존성이 아니다.
[확정 계약](docs/CLEAN_CORE_REWRITE.md) · [감사와 검증](docs/AUDIT.md) · [weight 비교](docs/compatibility.json) · [학습 수치 비교](docs/training_compatibility.json).
