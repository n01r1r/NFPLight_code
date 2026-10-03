> Historical research record. Current capture contract: [CAPTURE_REQUIREMENTS.md](../CAPTURE_REQUIREMENTS.md). Current commands: [README.md](../../README.md). Code citations and run status below describe their recorded version; removed code and old line numbers are not current implementation claims.

# MatSynth Fabric 전용 NFPLight from-scratch 학습 설계

## 2026-09-14 최신 실행 결정: 모든 full 실험 100k + 100k

사용자 승인: 현재 phi-theta 학습을 중단하고 L1 render 100k를 먼저 실행한 뒤
phi-theta를 재개한다. 앞으로 모든 full 실험은 map 100,000 + render 100,000
optimizer step으로 유지한다. 아래의 과거 100k + 25k 계획·결과는 당시 기록이며
새 실험의 기본값이 아니다. normal loss/head와 데이터·평가 목적은 변경하지 않는다.

- L1: 완료된 `fabric_raw_full_20260912/stage_a.pth`의 가중치 사용. 이 파일은
  map 100k 완료 시 선택한 map 87k 모델이다. 원본 run은 보존하고 새 출력
  `fabric_l1_render100k_20260914`에 render step 0 / total step 100k로 시작한다.
  Adam과 seed 0 RNG를 새로 초기화하며 render LR 1e-4 → 1e-5 cosine 100k,
  render weight 0.5 / ramp 1k를 사용한다. 기존 render 25k를 연장한 실험이 아니다.
- phi-theta: XYZ head / phi_theta loss 유지. PID 24712를 중단했고 저장된
  map 15,000 checkpoint에서 optimizer·RNG를 복원한다. 로그의 15,052까지 중
  미저장 52 step은 재계산한다. map 100k 스케줄은 그대로이고 아직 시작하지 않은
  render 기간만 25k → 100k로 바꾼다. 이후 총 200k까지 진행한다.
- 설정 이전은 알려진 원본 소스·동일 데이터·동일 objective 및 map 단계에 한정한다.
  일반 exact resume의 엄격한 검증은 유지하고 이전 출처와 설정을 기록한다.
- 실행 순서: L1 종료 코드 0, 완료 상태 및 200k/100k+100k 설정을 확인한 다음
  phi-theta를 자동 실행한다. 첫 run 실패 시 두 번째는 시작하지 않는다.
  두 run은 같은 검증된 소스 사본에서 실행한다.
- acceptance: 초기화·이전 허용/거부·resume CPU 검사, 순차 실행기 성공/실패 검사,
  실제 부모 checkpoint 검증과 L1 GPU 학습 로그 진행 확인. 실행 시작과
  100k 학습 완료·성능 검증은 구분해서 보고한다.

**실행 확인:** 2026-09-14 18:52:44 KST에 supervisor PID 9484 / L1 PID 44680으로
시작했다. `results/fabric_l1_render100k_20260914/launch.json`에 두 실행 명령과
소스 사본 해시를 저장했다. CPU 계약 40개, 순차 실행기 4개 검사 통과.
`startup_verification.json`에서 부모 가중치 bitwise 일치, 빈 Adam 상태,
첫 LR 1e-4 / render weight 0.0005, 초기 평가 파일 동일, render 89 step의
유한한 loss를 확인했다. L1 100k 완료와 후속 phi-theta 실행은 아직 진행/대기 상태다.

## 2026-09-14 Fork 배포 및 추론 호환성 계약

**완료:** Luna 구현 후 공개 대상 24개 파일을 commit
`cc79c9b98f06dccc0a78974e4b625c2f78ac5936`으로 기록하고
`https://github.com/n01r1r/NFPLight_code`의 `main`에 push했다.
로컬 브랜치는 `codex/fork-portability`, `origin`은 사용자 fork,
`upstream`은 원저자 저장소다. GitHub API에서 게시 commit과 파일 목록을 확인했다.
공개 파일만 복사한 clean checkout 및 새 CPU venv에서 43 contract tests,
독립 estimator smoke, pip check와 기존 XYZ/phi_theta checkpoint 실제 추론이
통과했다. NPZ shape/float32/finite/unit normal/roughness floor 및 head/feature
manifest를 확인했다. 다른 물리 머신·복원 품질·추가 학습은 검증하지 않았다.
선정 근거와 실행 증거는 로컬 `.fork-work/release-scope.json`,
`.fork-work/release-files.json`, `.fork-work/verification.json` 및 로그에 보존한다.
공개 제외 파일·가중치·데이터·기존 Git refs는 삭제하지 않았다.

현재 요청은 원저자 `CGLiWang/NFPLight_code`의 이력을 유지한 사용자 GitHub fork에
학습·추론에 필요한 수정만 공개하고, 새 clone에 별도로 전달한 checkpoint를 넣으면
동일한 입력 계약으로 추론할 수 있게 만드는 것이다. 상태 분석 후 Luna에 구현을
위임하며, 기존 Fabric 연구 목적과 완료된 실험의 의미는 변경하지 않는다.

- 공개 대상: 현재 Fabric trainer와 전이 import 의존 코드, 원본 호환 추론,
  관련 계약 검사, 설치·학습·추론 문서. 기존 원저자 코드·예제·인용은 보존한다.
- 제외 대상: checkpoint/optimizer, 데이터 cache, 개인 실측 촬영본, 결과·보고서·
  임시 파일, 로컬 agent/harness 상태 및 무관한 denoiser/실험 자동화.
  제외 파일은 로컬에 보존하며 Git history/ref 정리나 데이터 이동은 하지 않는다.
- checkpoint의 feature version 및 normal head 의미를 복원한다. metadata 없는
  원본 checkpoint의 legacy 동작과 strict state-dict 검증을 유지한다. 전달 단위는
  metadata를 포함한 기존 weight checkpoint 파일이며 bare parameter만으로 새
  head/RAW 의미를 추측하지 않는다. capture 입력·보정 정보는 입력 데이터 계약이다.
- RAW linear [B,6,256,256] -> SVBRDF [B,10,256,256], 기존 geometry·renderer,
  XYZ 기본 head/L1 기본 loss, 데이터 split·augmentation 금지 정책은 유지한다.
  학습 알고리즘 변경·재학습·기본 loss/head 변경은 이번 범위 밖이다.
- 설치 의존성과 사용자 지정 경로를 다른 머신에서도 사용할 수 있게 문서화한다.
  inference에는 학습 cache/source root/optimizer/외부 denoiser weight가 필요 없어야 한다.
- acceptance: 관련 CPU 계약 검사, 기존 checkpoint의 strict load와 actual inference,
  XYZ 및 phi_theta checkpoint의 학습/추론 decoder 일치, 공개 파일만으로 만든
  깨끗한 checkout의 clone-plus-weight 실행 및 유한한 출력 shape/manifest 확인.
  다른 물리 머신에서 실행하지 않은 경우 그 한계를 보고한다.
- 원격 목표는 인증 사용자 fork이며 원저자 저장소에는 push하지 않는다.
  publish는 선택된 코드만 검증한 후 수행하며 GitHub 접근 차단은 별도 보고한다.


작성·구현: 2026-09-12. 최신 요청: **RAW linear 기준 문제 수정, augmentation 없는 학습 실행·결과 보고, Luna delegation.**

**2026-09-14 Geodesic 후속 결정(현재 범위):** 현재 완료된 본 학습의 normal objective는
기존 XYZ head에 대한 signed XYZ L1이다. phi-theta head나 출력 채널 변경은 도입하지 않는다.
normal 개선 후보는 같은 XYZ head에서 정규화된 shortest-arc 구면거리
구좌표 변환 후 `cos(gamma) = cos(theta_p)cos(theta_t) +
sin(theta_p)sin(theta_t)cos(delta_phi)`로 계산하는 `geodesic` loss로 한정한다.
`gamma`는 안전한 `atan2(sin(gamma), cos(gamma))` shortest arc이고 loss는 `gamma/pi`다.
이는 [0,1] 무차원 loss이며 직접 `acos`를 역전파하지 않는다.
목적은 head 효과와 loss 효과를 분리하고 phi-theta의 좌표/pole 의존성을 피하는 것이다.
이번 변경의 acceptance는 known-angle 값, positive-scale 불변성, finite gradient/endpoint,
invalid mask·non-finite·zero-vector 거부, CLI/contract 식별, 4 map + 4 render CUDA smoke로 한다.
full training, matched pilot, 성능 비교 및 기본 loss 전환은 수행하지 않는다.

**2026-09-14 Head 후속 결정(현재 범위):** 사용자의 head 영향 확인 요청에 따라 기존
XYZ 출력과 고정 `phi_theta` head를 같은 `geodesic` objective·seed·smoke schedule로
나란히 비교한다. 후보 head는 추가 `nn.Parameter`나 별도 learnable layer를 만들지 않고,
네트워크의 normal 3채널을 `theta = pi*(raw_z+1)/4`, `phi = atan2(raw_y, raw_x)`로
읽은 뒤 unit normal `(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))`으로 재구성한다.
따라서 `z >= 0`, unit-length와 pole의 고정 azimuth 정책을 head 출력 계약으로 둔다.
diffuse/roughness/specular 채널과 기존 XYZ state-dict shape는 유지한다. acceptance는
head known-angle·unit/상반구·finite-gradient 검사, head/contract 식별, 동일 조건의
XYZ-vs-phi_theta 4 map + 4 render CUDA smoke로 한다. smoke는 초기 수치·gradient/trajectory
차이만 보여 주며 통계적 유의성이나 성능 우열을 주장하지 않는다. full training, pilot,
benchmark, 기본 head 전환은 수행하지 않는다.

**Head smoke evidence (2026-09-14):** 동일 seed와 `geodesic`, 각 4 map + 4 render
step의 CUDA smoke가 모두 `status=complete`, `total_step=8`로 종료했다. smoke의
2-material/2-light 최종 test artifact에서 XYZ head는 `normal_degrees=32.404268°`,
`normal_geodesic=0.180023707`, `map_mean=0.523075730`, `render_l1=0.047784900`;
고정 phi-theta head는 각각 `68.954762°`, `0.383082017`, `0.626572520`,
`0.028266595`였다. 즉 이 짧은 단일 seed smoke에서는 head에 따라 초기/단기
trajectory가 실제로 크게 달라졌지만, render와 map/normal 방향 지표가 엇갈리고
반복·held-out 규모가 없으므로 통계적 유의성 또는 우열의 근거로 해석하지 않는다.

**2026-09-14 XYZ head + phi-theta loss 결정(현재 범위):** 후속 normal 후보는 기존
XYZ output head를 유지하고, loss 단계에서만 unit normal을 `(theta, phi)`로 변환해
wrapped coordinate L1 `/ pi`를 계산하는 `--normal-head xyz --normal-loss phi_theta`로
고정한다. head decoder는 사용하지 않으며 diffuse/roughness/specular와 checkpoint
output shape는 기존과 같다. 사용자의 명시적 요청에 따라 이 조합의 full run을
`checkpoints/fabric_phi_theta_xyz_full_20260914`에서 시작한다. baseline은 완료된
`fabric_raw_full_20260912`이며, manifest/split/crop, RAW linear·무 augmentation,
seed=0, A100K+B25K, batch=2/accumulate=2, lights=4, render weight/ramp,
checkpoint-every=1000, validation과 L1 selection metric을 동일하게 유지한다.
CLI default와 완료된 baseline checkpoint는 변경하지 않는다. acceptance는
crash-consistent completion/status, 선택 checkpoint·test artifact·contract 및
baseline 대조이며, 추가 head/seed/pilot은 이 run이 끝날 때까지 시작하지 않는다.

**Full-run launch evidence (2026-09-14):** background PID 24712가
`checkpoints/fabric_phi_theta_xyz_full_20260914`에 기동되었다. 생성된 contract는
`normal_head=xyz`, `normal_loss=phi_theta`, `map_steps=100000`, `render_steps=25000`,
`seed=0`, `checkpoint_every=1000`을 확인했고, 첫 확인 시 map step 29까지
`map_loss=1.114407`, `normal_loss=0.571564`를 기록했다. 이는 실행 중간 상태이며
최종 성능 결과가 아니다.

**후속 normal-loss 비교 완료:** cosine과 wrapped phi–theta 재매개변수화 loss를 구현했다.
CPU31개 검사, 후보별 CUDA8-step smoke 및 300+100-step 비교를 완료했다.
두 후보의 초기 파라미터 SHA와 세 실행의 샘플/학습률/render-weight 순서가 일치했다.
선택 checkpoint test normal은 L1 13.6332°, cosine 13.6656°, phi–theta 16.2877°다.
짧은 단일 seed 결과에서 normal 개선은 없으므로 기본 L1은 유지하되, 사용자 요청에 따라 cosine·phi–theta도
동일한 본학습(A100K+B25K)으로 추가 비교한다. GPU 충돌을 피하기 위해 기존 L1 종료 후 두 후보를 순차 실행한다.
이 문장의 cosine·phi-theta queue 계획은 현재 명시된 XYZ head + phi-theta 단일 full run 결정으로 대체되었다.
[비교 보고서](../../results/fabric_normal_cosine_20260912/report.md), [비교 계약](../../results/fabric_normal_cosine_20260912/contract.md).
기존 L1 본학습은 19,000 step checkpoint 저장 중 디스크 부족으로 중단되었고, 정상 재개점은 18,000이다.
변경 전 20개 Python 소스는 `results/fabric_raw_training_20260912/baseline_source/`에 해시 검증해 보존했다.
checkpoint 정책은 기본 `--checkpoint-every 1000`으로 바꾸어 validation·stage 경계·정상 종료·stop-after만
항상 저장하며, 기존 동작은 `--checkpoint-every 1`로 재현한다. 후보 full run은 이 정책을 명시적으로 사용한다.

**실행 상태:** RAW 수정과 CPU31개 검사, GPU exact resume/adapter 일치 검증 완료. 고정 crop250-step overfit과 Fabric 전체 구성400-step pilot 완료. 2026-09-12 23:07 KST에 시작한 L1 본 학습은 2026-09-13 03:05 KST에 map 19,000 checkpoint 저장 중 C: 디스크 부족으로 실패했다. checkpoint·results·Fabric cache·`.git`은 `D:\NFPLight_runs\NFPLight_code`로 이동하고 C: 원래 경로에는 junction을 만들었다. 기존 generation 중 1000-step 배수가 아닌 41개(17.67GB)와 실패 임시 파일은 제거했다. `.git`의 41GB 중 대부분은 Codex turn-diff ref가 보유한 reachable checkpoint blob이므로 ref 삭제 전 사용자 확인이 필요하다. 새 full 후보는 checkpoint 정책 적용 및 충분한 D: 공간 확보 후 재개한다.

최신 실행 계약은 [RAW 학습 계약](../../results/fabric_raw_training_20260912/contract.md)을 따른다. `raw_calibrated_v2`, 관측 gamma/양자화 없음, 고정16 crop·D4 없음, invalid normal 픽셀 마스크, renderer roughness floor, 실제 gradient clipping, 1000-step resumable checkpoint cadence, crash-consistent checkpoint 및 float 맵 출력을 적용한다. 데이터 재질·분할·원본 cache는 유지하며 [target policy 전수 검사](../../results/fabric_raw_training_20260912/dataset_policy_v2.json)에 유효 픽셀과 변경 범위를 기록한다. 아래 LDR 구현 기록은 과거 증거다. 최신 실행 결과는 D: junction 뒤 `results/fabric_raw_training_20260912/`에 기록한다.

**과거 검토 기록:** [입출력·엔지니어링 검토](../../results/fabric_io_review_20260912/review.md)는 RAW 변경 전 상태와 central-scale invalid, normal z, HDR loss, checkpoint 비정상 종료 등의 근거를 보존한다. 최신 수정·학습 상태는 위 RAW 실행 계약과 실행 결과를 우선한다.

현재 실행 경로는 [train_fabric.py](../../train_fabric.py), 고정 데이터는 [manifest](../../data/fabric_native256_v1/manifest.json), 실제 검증 결과는 [구현 보고서](../../results/fabric_training_implementation/report.md)를 따른다. **train 338 / validation 38 / official test 5**, 재질별 native 256 crop 16개를 생성했다. 아래 §1–8은 구현 전 조사 기록으로 보존하며, 현재 결정과 완료 상태는 이 문서 상단 및 구현 보고서가 우선한다.

### 2026-09-12 구현 계약

사용자가 우려 사항 수정, Fabric 구성 확정, 코드 구현을 요청했다. 아래 본문은 최초 설계 기록이며 이 계약이 실행 결정을 구체화한다.

- root=`D:/MatSynth_preprocessed`, metadata.category=Fabric, source=deschaintre_2020 제외. 후보 385 train / 5 official test. 아래 opacity 보정 계약 및 map 유효성 검사를 적용하고 모든 제외 사유를 기록한다.
- 재질별 source link(홈페이지 제외) 또는 normal+roughness 원본 hash를 공유하는 재질은 한 group으로 묶는다. official test와 group이 겹치는 train 재질은 제외한다. 완전한 semantic variant 탐지를 보장하지 않는다.
- seed=0, group 기준 약 10% validation. diffuse/specular 원본을 보존한 uint8 native 256 crop 16개/재질을 workspace 안에 저장한다. 기존 D: 데이터는 수정하지 않는다.
- sample별 feature 정규화는 sample_v1, metadata 없는 기존 checkpoint는 legacy_batch_v0로 읽어 과거 추론을 보존한다. 새 trainer는 공통 입력 조립과 LDR 양자화를 사용한다.
- A=100K, B=25K; Adam+각 stage cosine, effective batch=4(기본 microbatch 2 × accumulation 2), FP32, K=4 colocated HDR-L1. B의 rendering 가중치는 1K step 동안 0→0.5.
- B 모델 선택은 validation rendering 개선 및 A 대비 각 normalized map-L1 증가 ≤0.02를 함께 요구한다. 이 허용치는 구현 초기값이며 정확성 보장이 아니다. 불충족 시 A model을 보존한다.
- mmap cache를 동기적으로 읽고 전용 Python RNG로 material/crop/D4를 선택한다. RNG와 optimizer/stage/step, dataset/config hash를 full checkpoint에 저장하여 worker=0 경로의 재개를 검증한다.
- 완료 기준: Fabric manifest/cache 생성, sample-batch invariant·PNG round trip·invalid 입력·고정 평가·checkpoint 재개 검사, 실제 GPU의 짧은 A/B 학습 smoke. 장시간 본 학습은 이번 구현 검증에 포함하지 않는다.

**전수 검사에 따른 opacity 보정:** 후보 중 380개는 metadata.maps에 opacity가 없지만 export된 대체 opacity PNG의 공통 4개 픽셀이 216/255다(4096² 중 약 2.38e-7). strict minimum만 적용하면 이들과 test 5개가 모두 탈락한다. 실제 opacity를 선언한 10개에는 strict minimum을 유지한다. 미선언 대체 맵은 254 미만 pixel 비율 ≤1e-6일 때만 opaque로 취급하며 min/비율/선언 여부를 manifest에 기록한다. 그 이상은 제외한다. GT 색·normal·roughness 픽셀은 변경하지 않는다.

**구현 확인:** CPU 계약 검사 8개 및 기존 실측 estimator smoke 통과. RTX 3090에서 A 4 + B 4 step 통과, 3 step 후 중단·재개한 실행과 parameter/optimizer/RNG/log/evaluation이 정확히 일치했다. 새 checkpoint의 실측 adapter 출력과 학습 경로 출력 차이는 0이었다. 본 학습 수렴·overfit 완료·일반화 성능을 검증한 결과는 아니다. 원 계획의 기존 runner 확장 대신 별도 `train_fabric.py`에 Fabric 실행을 모아 과거 전체 category 실험 CLI를 보존했다.

## 1. 목적과 결론

목표는 MatSynth의 **Fabric 재질만으로 near/far 두 이미지에서 SVBRDF를 추정하는 21채널 TwoBranchNet을 처음부터 학습**하는 것이다. DenoiseNet은 학습, checkpoint 로딩, 추론 모두에서 제외한다. 사전학습 weight와 LoRA를 쓰지 않는다. 두 번째 학습 단계는 첫 단계에서 직접 학습한 estimator를 이어서 최적화하므로 from-scratch 조건을 유지한다.

권장 방향은 **native 256 crop → GGX near/far 렌더 → 21채널 feature → estimator**, 그리고 **map-L1 선행 학습 → map-L1 + rendering loss 학습**이다. 기존 trainer는 출발점으로 쓸 수 있지만, 현 상태를 그대로 장시간 실행하는 것은 권하지 않는다. 특히 입력의 배치 의존성, 실제 데이터 경로, fabric 필터, checkpoint 선택을 먼저 수정해야 한다.

이 문서는 여러 카테고리와 metal/F0 범위 확장을 중심으로 작성된 [이전 계획](NFPLight_matsynth_scratch_plan.md)을 이번 작업 범위에서 대체한다. Fabric 밖의 성능 유지나 metal 개선은 이번 성공 기준에 넣지 않는다. 기존의 native crop 원칙, 10채널 출력, GGX renderer, capture-noise augmentation 제외는 유지한다.

## 2. 실제 데이터 확인

| 경로 | Fabric train / test | 확인 내용 | 용도 판단 |
|---|---:|---|---|
| `D:\MatSynth` | split 없음, fabric 1개 | 전체 5개 재질의 near/far/gt 예시. Fabric 맵과 입력은 1024² | 데이터셋 전체가 아니다. 시각 점검용 |
| `D:\MatSynth_preprocessed` | 458 / 5 | 원본 PNG와 metadata. train normal은 385개 4096², 73개 2048² | 원본·출처·opacity·색 공간 확인의 기준 |
| `D:\MatSynth_crops256` | 458 / 5 | 첫 NPZ는 재질당 256² crop 4개, basecolor/metallic/roughness/normal | 기존 코드 smoke에 바로 재사용 가능 |
| `D:\MatSynth_crops256_k16` | 458 / 5 | 첫 NPZ는 같은 4종 맵, crop 16개 | 후보 cache. 모든 파일의 좌표·생성 이력은 미검증 |
| `D:\MatSynth_crops256_fullstride` | 458 / 5 | 첫 NPZ는 crop 1개 | 이름만 보고 더 많은 crop이 있다고 가정하면 안 됨 |

NPZ의 crop 수·shape는 각 split의 첫 파일을 확인한 값이다. 전체 cache 내용과 원본 crop 대응을 전수 검증한 것은 아니다. 재현 가능한 목록과 확인값은 [evidence.json](C:/workspace/NFPLight_code/results/fabric_training_design_20260912/evidence.json)에 있다.

train 458개 중 **73개는 metadata.source=`deschaintre_2020`**, 나머지는 385개다. 따라서 로컬의 모든 Fabric을 쓴 실험과 원 MatSynth 출처만 선택한 실험은 데이터 모집단이 다르다. 공식 데이터 카드에도 Deschaintre 출처를 필터하는 사용 예가 있다. [MatSynth 공식 데이터 카드](https://huggingface.co/datasets/gvecchio/MatSynth)

**계획의 데이터 선택:** 사용자가 지목한 경로를 임의로 바꾸거나 73개를 자동 제외하지 않는다. 실행 전 manifest에서 실제 root와 출처 포함 여부를 확정한다. 원 MatSynth로 DeepMaterials를 대체한다는 이전 목적을 엄격히 유지하려면 385개 안을 권장한다. 단순히 로컬 MatSynth의 모든 Fabric으로 학습하려는 경우는 458개 안으로 명시하면 된다. 아래 학습 절차는 두 경우에 동일하게 적용한다.

`D:\MatSynth\Fabric_acg_fabric_009`는 official Fabric test ID와 같다. 이 파일을 학습에 사용한 후 해당 재질을 test 점수에 포함하면 안 된다. 이름이 다른 재질 사이의 중복·variant까지 검사한 것은 아니다.

## 3. 원 논문에서 확인한 설정

본문 §4.1, p.5를 텍스트와 렌더된 페이지로 확인했다. 보충자료는 등록 방법과 추가 비교 결과를 설명하며, 별도의 상세 training recipe는 찾지 못했다. [NFPLight 본문](https://cgliwang.github.io/NFPLight/pdf/NFPLight.pdf), [보충자료](https://cgliwang.github.io/NFPLight/pdf/SupplementalMaterial.pdf)

| 항목 | 논문의 명시 사항 |
|---|---|
| optimizer | Adam |
| 1단계 | 두 network 각각 400K iteration, GT에 대한 L1 |
| 1단계 LR | 5e-4 → 1e-5, cosine annealing |
| 2단계 | 두 network joint training 100K iteration, L1 + rendering loss |
| 2단계 LR | 1e-4 → 1e-5, cosine annealing |
| train 입력 해상도 | 256 × 256 |
| 합성 비교의 near / far | 2.414 / 10 |
| 실측 비교의 near / far | 4 / 12 |

거리 설정은 본문 §5.1의 합성/실측 실험 조건이다. 이번에는 기존 21채널 synthetic geometry인 2.414/10을 유지한다. 논문의 실측 설정 4/12를 그대로 가져오려면 renderer와 relation 계수, 실제 촬영 범위를 함께 맞춰야 하므로 이번 기본안에 섞지 않는다.

확인한 본문·보충자료에서는 batch size, map별 loss 가중치, training rendering loss의 정확한 광원 개수·분포·세부 수식을 확정할 수 없었다. 아래 값은 이번 구현을 위한 제안이다. 특히 기존 코드의 **“논문은 고정 LR” 주석은 틀리다.** DenoiseNet이 빠지고 dataset도 바뀌므로 이 실험은 논문의 전체 pipeline 재현이 아니라 논문을 참고한 estimator 전용 학습이다.

## 4. 기존 코드의 재사용 판단

| 부분 | 실제 상태와 판단 | 필요한 조치 |
|---|---|---|
| [ScratchModel](C:/workspace/NFPLight_code/train_scratch.py:41) | `init_network()`가 fresh `TwoBranchNet()` 생성. 64,102,378개 parameter, 21채널 | 유지. resume 없는 새 run에서 pretrained 로드가 없음을 검증 |
| [TwoBranchNet](C:/workspace/NFPLight_code/network/nfplight_net.py:69) | 18채널 appearance branch + 3채널 relation/mask branch, 10채널 출력 | 구조 유지. DenoiseNet과 legacy33 경로 제외 |
| [crop loader](C:/workspace/NFPLight_code/data/matsynth_crops.py:50) | gamma 2.2 decode 후 metallic workflow로 d/s 유도. D4 augmentation 구현 | 변환·회전 기능은 재사용. 원본 d/s를 쓰는 cache와 schema를 구분 |
| [기존 PNG loader](C:/workspace/NFPLight_code/finetune_fabric.py:245) | diffuse/specular를 decode 없이 정규화 | 원본 PNG용으로 그대로 쓰면 안 됨 |
| [21채널 입력 조립](C:/workspace/NFPLight_code/finetune_fabric.py:330) | 실제 호출은 `toLDR=False` | 문서의 “LDR 양자화”와 불일치. 공통 입력 함수와 명시적 설정 필요 |
| [relation 정규화](C:/workspace/NFPLight_code/model/nfplight_model.py:102) | `torch.min/max(feature)`가 배치 전체에 적용 | **우선 수정:** sample별 C/H/W reduction. train·infer 모두 같은 함수를 사용 |
| [현재 학습 일정](C:/workspace/NFPLight_code/train_scratch.py:246) | AdamW, 기본 constant 2e-4, 선택적 delayed cosine | 논문의 Adam + stage별 cosine으로 구성 가능하게 수정 |
| [render loss](C:/workspace/NFPLight_code/train_scratch.py:199) | raw linear-HDR L1. camera=view=light인 랜덤 위치 K개 | “log1p”, “off-colocated”라는 이전 설명은 실제 코드와 다름. 현재 방식은 baseline으로 재사용 가능 |
| [validation](C:/workspace/NFPLight_code/train_scratch.py:220) | material마다 crop 0만 사용, render light는 매번 새 random | 고정 crop 여러 개와 독립된 eval RNG 사용 |
| [best/test 연결](C:/workspace/NFPLight_code/train_scratch.py:500) | 마지막 메모리 weight로 최종 test. 저장한 best를 다시 읽지 않음 | test 직전에 선택한 checkpoint를 strict load하고 hash 기록 |
| [공식 paired eval](C:/workspace/NFPLight_code/eval_paired_official.py:108) | 전체 89개·전체 category를 고정 검증 | 검증 guardrail을 풀지 말고 Fabric manifest용 평가를 별도 구성 |
| [full resume](C:/workspace/NFPLight_code/train_scratch.py:393) | optimizer/scheduler/RNG 저장은 있음. 보통 경로의 sampler 위치는 복원 안 됨 | 실험 config·dataset hash·sampler/aug 상태·stage를 함께 저장 |
| `finetune_fabric*.py`, `exp_denoise.py`, autoresearch scripts | 이전 LoRA/실측/denoise/전체 category 실험용 | 새 from-scratch 실행 경로로 사용하지 않음 |

배치 의존성은 재현했다. 동일한 작은 feature를 단독 정규화했을 때와 다른 feature와 배치로 묶었을 때 최대 차이가 **0.9230769**였다. 실제 데이터 성능 저하량을 측정한 수치는 아니지만, batch 크기와 구성에 따라 입력 의미가 변하는 코드 결함을 직접 보여준다. 수정 후 acceptance는 **같은 이미지의 단독·배치 feature 일치**다.

추가로 확인할 사항은 `n_lights>0`, empty split, `resume_step>=target_step`, non-finite loss/gradient의 명시적 거부다. 현재 `render_weight`는 총 step 수에 종속되어 있어 재개 시 `--steps`를 바꾸면 일정의 의미도 달라진다. stage별 고정 기간으로 이를 해소한다.

## 5. 데이터와 수치 계약

### 데이터 선정과 split

1. 원본 metadata의 `category == Fabric`으로 선택하고 source, material ID, 원본 경로·hash, 허용/제외 사유를 manifest에 저장한다. category 가중치나 `fabric_tilt`는 제거하고 material 균등 sampling을 쓴다.
2. 공식 test 5개는 고정한다. train에서 약 10%를 validation으로 분리한다. 필터 전 단순 계산으로는 458 → 약 412/46, 385 → 약 346/39이며, 실제 개수는 source와 variant grouping 후 확정한다.
3. 같은 원본에서 나온 색 변형·blend·variant는 가능한 범위에서 같은 split에 묶는다. 이후에 crop을 생성한다. train/val 사이에 같은 재질의 서로 다른 crop을 나누지 않는다.
4. opaque planar GGX가 기본 계약이다. missing/corrupt map, NaN/Inf, invalid normal, alpha hole은 검사하고 명시적으로 제외한다. 기본 후보 기준은 opacity의 모든 pixel이 정규화값 `>=254/255`인 재질이며, 제거 수와 ID를 기록한다. 이는 본문 recipe가 아니라 renderer의 미지원 투명도를 피하기 위한 제안이다.
5. flat normal, 높은 roughness, 균일한 색, procedural 출처라는 이유만으로 제거하지 않는다. matte/plain fabric은 실제 목표 분포에 포함될 수 있다. 과거 문서의 “단서가 약하므로 제외” 기준은 사용하지 않는다.

### map과 좌표

| map | 학습/renderer 의미 | tensor 계약 |
|---|---|---|
| normal | tangent normal, x 오른쪽·y 위·z 표면 바깥 | unit GT, float32 `[B,3,H,W]`, [-1,1] |
| diffuse | linear RGB diffuse reflectance | 물리 공간 [0,1], 저장 tensor는 2d−1 |
| roughness | perceptual roughness. GGX 내부 alpha=r² | `[B,1,H,W]`, 물리 공간 [0,1], 저장 tensor는 2r−1 |
| specular | linear RGB Fresnel F0 | `[B,3,H,W]`, 물리 공간 [0,1], 저장 tensor는 2s−1 |

출력 순서는 **n3,d3,r1,s3**, 전체 `[B,10,256,256]`다. renderer는 예측 normal을 normalize하지만 network head 자체는 tanh이며 unit normal을 보장하지 않는다. map-L1에서는 현행 raw normal 채널을 사용하고, 각도 평가에서는 normalize 후 계산한다. zero-length 예측은 invalid 수로 보고한다.

MatSynth의 OpenGL normal 규약은 공식 자료에 명시되어 있으며, 로컬 `surface()`도 위쪽이 +y다. 기본 y-flip은 필요 없다. augmentation의 회전·반사에 따른 normal xy 변환은 별도로 적용한다. [MatSynth normal 규약](https://huggingface.co/datasets/gvecchio/MatSynth)

색 변환의 기본 제안은 기존 경로와 맞추는 **`gamma22_compat`**다. 색 PNG를 [0,1]로 읽고 `c**2.2`를 적용한다. 이는 정확한 piecewise sRGB EOTF가 아닌 근사임을 config에 기록한다. normal/roughness/metallic/opacity에는 gamma를 적용하지 않는다. 정확한 sRGB 변환으로 바꾸려면 GT·입력 인코딩·실측 로더를 함께 바꾸고 별도 버전으로 비교해야 한다.

**본 학습 GT는 원본 diffuse/specular 맵을 decode해서 사용하는 안을 권장한다.** 이미 있는 맵을 버리고 metallic=0인 fabric 전체의 F0를 상수로 만들 필요가 없다. 4-key crop cache는 빠른 smoke와 기존 결과 확인에는 유용하지만, d/s 원본을 보존한 cache와 동등한 GT라고 부르지 않는다. 기존 fallback은 `d=basecolor_lin*(1-m)`, `s=0.04*(1-m)+basecolor_lin*m`이며 이것을 쓸 때는 `gt_workflow=derived_metallic`으로 표시한다. 이 GT 경로 변경은 이번에 구현하지 않았다.

renderer에는 roughness 최솟값 0.05가 있다. GT r<0.05는 map supervision에서 보존하되 해당 pixel 비율을 보고하고, renderer가 그 이하를 구별한다고 해석하지 않는다.

### crop과 입력 생성

- native 256² crop을 재질마다 균등 추출한다. 첫 학습 후보 cache는 재질당 16개 이상이며, 좌표·seed·원본 해상도·변환을 기록한다. 고정 crop을 늘려도 독립 재질 수는 늘지 않는다.
- 원본 맵들의 UV/해상도가 같은지 먼저 확인한다. 해상도가 다른 nonconstant map을 동일 pixel 좌표로 잘라 합치거나, 가장 작은 map에 맞춰 일괄 축소하지 않는다. 상수 map은 broadcast 가능하며 그 외 mismatch는 별도 처리 대상으로 남긴다.
- 한꺼번에 4K float tensor를 만들지 않는다. uint 데이터에서 crop한 뒤 float32로 변환한다. PNG는 crop 전에 압축 전체를 decode할 수 있으므로 학습 loop 밖에서 cache를 만드는 것이 낫다.
- D4의 8개 변환(90° 배수 회전 × hflip)을 맵 전체에 함께 적용한다. arbitrary-angle 회전과 resize는 기본 OFF다. GT를 변환한 **뒤에** near/far를 재렌더한다.
- 표면은 `[-1,1]×[-1,1]`, z=0, near=(0,0,2.414), far=(0,0,10). 이는 물체 반폭을 1로 둔 상대 단위이며 cm가 아니다. 현재 `lampIntensity=16`, 거리 제곱 감쇠를 사용한다.
- 기본 입력은 현재 renderer의 clip → gamma22 encode → uint8 rounding → gamma22 decode를 거친 linear pair다. train에는 `toLDR=True`, infer에는 이미 decode한 실측 pair를 같은 feature 함수로 전달한다. 이는 기존 목표 문서와 맞추는 수정안이다.
- 21채널은 `[near3, far3, aligned_far3]`와 그 log 9채널, relation1, log_relation1, mask1이다. mask는 별도 feature이며 모든 map loss를 지우는 GT mask로 쓰지 않는다.
- sensor noise, 위치 jitter, colored flash, vignette, 새 denoiser는 기본안에서 제외한다. DenoiseNet을 빼면 실측 noise·registration 문제도 자동 해결된다는 뜻은 아니다. 실측 추론은 정렬·노출 조건을 맞춘 `matsynth21` 경로를 사용한다.

## 6. 권장 학습 일정

**첫 본 학습 후보는 A 100K + B 25K optimizer step**으로 잡는다. 논문의 400K+100K를 1/4로 줄인 초기 예산이며 충분한 수렴을 보장하는 수치가 아니다. 작은 Fabric 모집단에 원 논문의 iteration을 무조건 복사하기보다 validation과 계산 비용을 먼저 확인한다. 기간을 바꿀 때에는 test를 보지 않고 새 config로 고정한다.

| 항목 | A: map 학습 | B: rendering supervision 추가 |
|---|---|---|
| 시작 weight | fresh random initialization | A에서 선택한 estimator weight |
| 권장 시작 예산 | 100,000 step | 25,000 step |
| optimizer | Adam, betas=(0.9,0.999), weight_decay=0 | 새 Adam 상태로 시작 |
| LR | cosine 5e-4 → 1e-5 | cosine 1e-4 → 1e-5 |
| loss | Lmap | Lmap + λ Lrender |
| λ | 0 | 첫 1,000 step 동안 0→0.5, 이후 0.5 |
| batch | effective 4 | effective 4 |
| validation | 1,000 step마다 | 1,000 step마다 |

betas, weight decay, optimizer reset, 100K/25K, λ/ramp, batch는 이번 제안이며 논문에서 확인된 수치로 주장하지 않는다. stage B 전환 시 early-stop/best counter를 초기화한다. 전역 map score만으로 stage A에서 학습 전체를 끝내어 B가 생략되는 구조는 피한다.

`Lmap = mean_abs(n_pred−n_gt) + mean_abs(d_pred−d_gt) + mean_abs(r_pred−r_gt) + mean_abs(s_pred−s_gt)`로 시작한다. 각 항은 자기 채널과 pixel, batch에 대해 평균하므로 4개 map의 가중치가 같다. 전체 10채널 평균과는 다르다. 학습 loss는 정규화 tensor 공간의 값이다.

`Lrender = (1/K) Σ mean_abs(R(pred,l_k,v_k)−R(gt,l_k,v_k))`. 초기 baseline은 기존 코드의 **K=4 colocated** 위치 분포 x,y∈[-1.2,1.2], z∈[2,8]와 raw linear-HDR L1을 유지한다. 입력 near/far 외의 위치에서의 supervision이지만 light와 camera를 독립적으로 뽑는 것은 아니다. 논문이 이 정확한 loss sampling을 썼다고 주장하지 않는다. GT와 prediction은 반드시 같은 위치·강도로 렌더한다.

독립 light/view sampling은 novel-view 평가에는 포함하고, training loss로 바꾸는 실험은 baseline을 확인한 뒤 하나의 변경으로 분리한다. log1p loss와 raw HDR loss도 별도 설정이다. 단순히 이름만 같다고 과거 결과와 수치 비교하지 않는다.

현재 GPU는 RTX 3090 24GB다. 첫 구현 검증에서는 batch 2부터 memory peak를 측정하고 가능하면 4를 사용한다. 부족하면 2×accumulation 2로 effective batch 4를 유지한다. network mixed precision은 필요할 때 도입하되 relation/division/log와 GGX/loss는 FP32로 둔다. 이번에는 memory peak·step time·전체 학습 시간을 측정하지 않았으므로 소요 시간을 예측값으로 확정하지 않는다.

## 7. 검증, 모델 선택, 재현성

**장시간 학습 전 구현 acceptance**

- manifest에 Fabric만 있고 train/val/test의 material/group overlap이 없다. 거부된 재질 수·사유가 남는다.
- 두 Fabric 샘플의 D4 변환은 finite/range/unit-normal/pixel permutation 검사를 통과한다. normal 방향은 알려진 plane/heightfield로 별도 검사한다.
- 같은 입력의 batch=1과 batch=N feature가 허용 오차 내에서 같다. center가 어둡거나 saturated인 경우 scale, relation, mask가 finite다.
- synthetic encode→PNG→real decode 경로와 학습용 feature가 같은 계약으로 일치한다. 인코딩과 양자화 위치를 고정한다.
- 새 net에 DenoiseNet/adapter/pretrained state가 없다. forward/backward/optimizer 한 step에서 loss와 gradient가 finite이고, synthetic identical pred/GT의 rendering loss는 0에 가깝다.
- 작은 고정 train set 8–16 crop을 overfit할 수 있는지 확인한다. 이는 배선·최적화 점검이며 generalization 근거가 아니다.
- best save→strict load 후 고정 validation 결과가 재현된다. resume의 같은 다음 step 재현은 worker=0과 명시적 sampler/augmentation 상태로 우선 검증한다.

**모델 선택**

A의 best는 고정 validation의 material-macro map-L1로 고른다. B에서는 고정 validation render-L1을 주 지표로 하며 normal/diffuse/roughness/specular를 함께 보고한다. map regression 허용치는 A 결과를 확인한 후 B 실행 전에 config에 고정한다. `best_map`, `best_render`, `last`를 별도로 보존하고, 선택된 model 파일을 실제로 reload한 뒤 test를 한 번 수행한다.

validation crop은 재질당 고정 4개 이상, test는 가능하면 native 고정 16개로 한다. **crop/pixel → material 내 평균 → material 간 평균** 순서로 집계한다. test 5개는 개별 결과도 전부 표시한다. crop 수를 독립 재질 표본 수처럼 보고하지 않는다.

normal 각도 오차(degree, 단위화 후 arccos), d/r/s 물리 공간 [0,1]의 MAE/RMSE, linear-HDR rendering L1/RMSE를 구분한다. LPIPS를 추가한다면 encoder 버전과 tone mapping을 고정하며 “논문과 동일 구현”이라고 부르지 않는다. 논문은 합성 평가에서 30개의 random lighting/viewing direction과 RMSE/LPIPS를 보고한다. 이번 test에도 고정 seed의 30개 light/view pair를 사용할 수 있으나, 구체 sampling은 명시적으로 정의해야 한다. [NFPLight §5.1.1 및 Table 1](https://cgliwang.github.io/NFPLight/pdf/NFPLight.pdf)

eval RNG는 train RNG와 분리한다. run config에는 seed, dataset revision·source filter·manifest hash, crop coordinate/hash, GT workflow, gamma, geometry, light set, stage별 optimizer/scheduler, batch/accumulation, code hash, Python/PyTorch/CUDA/GPU를 기록한다. full checkpoint에는 현재 stage/step과 RNG, sampler/augmentation 위치를 추가한다. 기존 replay plan은 파일 목록 identity와 연결을 강화한 뒤 재사용한다.

실측 Fabric은 정렬된 near/far의 결과와 새로운 조명에서 촬영한 reference가 있으면 별도로 비교한다. 반복 촬영 출력이 일정하다는 사실만으로 정확성을 입증하지 않는다. 상수 출력을 내는 estimator도 일관성 점수는 좋을 수 있다. 검증 목표는 현행 isotropic GGX의 n/d/r/s 추정이며, 섬유의 anisotropy·sheen·transmission을 복원했다는 주장은 범위 밖이다.

## 8. 구현 순서와 이번 확인 범위

1. root/source 범위를 확정한 Fabric manifest, split, opacity/invalid 검사와 native crop cache를 만든다. cache가 기존 four-key와 다르면 version/key/encoding을 명시한다.
2. sample별 relation 정규화와 공통 21채널 입력 조립을 수정한다. 기존 checkpoint 비교용 legacy preprocessing은 명시적 version으로 보존한다.
3. `train_scratch.py`에 Fabric manifest, 원본 d/s GT 경로, Adam과 stage별 cosine, finite 검사, stage별 checkpoint 선택을 붙인다. 현재 CLI에 없는 옵션을 있는 것처럼 제시하지 않는다.
4. fixed multi-crop/light Fabric evaluator와 selected-checkpoint reload를 완성하고 acceptance를 확인한다.
5. 짧은 overfit/throughput 점검 후 A→B를 실행한다. 400K+100K 예산은 필요할 경우 별도 고정 config로 수행한다.

이번에는 본문·보충자료 확인, 데이터 경로/출처/normal 해상도 inventory, 두 Fabric crop의 8가지 변환 검사, batch 정규화 결함 재현, network parameter/입력 채널 확인을 수행했다. **학습 forward/backward, optimizer step, full-dataset opacity 검사, 전체 cache provenance 검증, 실제 성능 비교는 수행하지 않았다.** 과거 로그의 smoke나 평가 수치를 이번 결과로 재사용하지 않는다.

재실행 가능한 audit:

```powershell
python results/fabric_training_design_20260912/audit.py
```

현재 환경은 Python 3.11.0rc2, PyTorch 2.5.1+cu121, RTX 3090이다. README/requirements의 PyTorch 2.9.1 환경과 다르므로 requirements 파일만으로 이번 환경이 재현된다고 보면 안 된다.
