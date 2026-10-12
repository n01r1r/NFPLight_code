# NFPLight

완료된 refinement runner·날짜별 report/PDF 생성기와 전용 테스트는 [복구용 원본 ZIP](docs/archive/completed_refinement_sources.zip)으로 정리했다. 아래 관련 Python 명령은 과거 기록이며 재현할 때만 ZIP의 원래 상대 경로로 복구한다. 현재 capture·학습 진입점은 유지한다.


## 독립 세션의 현재 기준

[학습 scheduler 정책](docs/TRAINING_SCHEDULER_POLICY.md) → [현재 학습 상태](docs/CURRENT_TRAINING_STATE.md) → [artifact 인덱스](artifacts/training_artifact_index.json) 순으로 읽는다. 새 학습 기본은 **ReduceLROnPlateau**다. 2026-10-09 Go 후 [단일 seed19 후속 계약](docs/PLATEAU_FOLLOWUP.md)의 별도 runner 구현·CPU51검사·FP32 GPU smoke를 마치고 [7.61M D4+dropout·seed19 학습](artifacts/plateau_followup_7m_seed19_20261009/status.json)을 시작했다. 기존 전역 trainer는 cosine/고정 LR이며 새 scoped runner와 구분한다. 실제 실행 상태는 현재 학습 상태 문서를 따른다. 7.61M 선택 seed7/19 진단과15.09M D4-only seed19는 완료됐고, 아래 날짜별 고정 LR·준비·전환 설명은 이력이다. 새 후속은 단일 seed19이며 seed31 제외 incident를 현재 상태에서 확인한다. [Doctor 검사](artifacts/policy_doctor_20261009/report.json).

아래 과거 run 명령·기본 config를 새 정책 구현 또는 새 실행 승인으로 사용하지 않는다. Capture 추론의 현재 기준은 [CAPTURE_REQUIREMENTS](docs/CAPTURE_REQUIREMENTS.md)이며 학습 후속과 구분한다.

## Bottleneck FC / Pixel MLP 비교 실험

2026-10-08 사용자 확정: `baseline`, `baseline+Bottleneck_FC`, `baseline+Pixel_MLP` 세 조건으로 실험을 준비한다. 독립 실험 폴더의 [config·진입점·관리 명령](experiments/bottleneck_fc/README.md)을 기준으로 한다. 공간 FC의 구체 정의와 고정 비교 조건은 해당 폴더의 `config.json`에 기록한다. 2026-10-08 20:00 사용자 지시로 baseline만 실행했다. 출력은 `artifacts/bottleneck_fc_baseline_20261008`이며 seed 7/19/31마다 map 1,000 + render 300 update를 수행한다. 선택은 `run_selection.json`에 고정한다. Live viewer는 seed 7/19/31 순서로 포트 8767/8768/8769를 사용한다.
기존 exploratory run과 checkpoint는 `artifacts/bottleneck_fc_overfit_20261008`에 보존한다. 이전 runner의 `running` 상태는 실제 프로세스와 맞지 않으며 [실제 상태 확인](artifacts/bottleneck_fc_overfit_20261008/RUN_STATE_RECONCILIATION.json)에 기록했다. 이전 수치 결과를 새 규약으로 생성된 run처럼 취급하지 않는다.


2026-10-08 후속 승인: baseline 세 seed의 초기 1k map + 300 render 실행이 모두 완료된 후 단계별 수렴을 판정한다. 미수렴은 누적 map 10k / render 3k를 상한으로 연장한다. Early stop은 Validation 10회 연속 구간의 상대 개선 <0.5%이며 map=heldout NDR mean, render=heldout render L1이다. Map을 연장하면 map 종료 checkpoint·Adam에서 이어간 뒤 fresh Adam으로 render를 시작한다. 연장 결과는 `artifacts/bottleneck_fc_baseline_extended_20261008`에 분리하며 viewer 포트는 seed 7/19/31 순서 8777/8778/8779다. 기존 학습 프로세스와 과학 설정은 변경하지 않는다.

## D4 · Dropout 2×2 실험 준비

2026-10-08 사용자 승인: 기존 baseline을 대조군으로 재사용하고 D4 only, NAFBlock dropout p=0.1 only, D4+dropout을 seed 7/19/31로 비교한다. [독립 설정과 관리 명령](experiments/regularization/README.md) · [고정 과학 규약](docs/REGULARIZATION_PROBE.md). Map 10k / render 3k 상한과 기존 early stop을 유지한다. 당시 준비·CPU 검증 단계였으며 이후 선택 seed7/19 실행이 완료됐다. 최신 범위는 위 현재 학습 상태를 따른다.
2026-10-09 사용자 승인으로 D4/dropout 9개 run을 `artifacts/regularization_probe_20261009_checked`에서 시작했다. [진행 상태](artifacts/regularization_probe_20261009_checked/status.json)와 [실행·후속 조정 권한](artifacts/regularization_probe_20261009_checked/user_authorization_20261009.json)을 기준으로 한다. 초기 비교의 LR·상한은 유지하며, 허용된 LR schedule·상한 연장은 평가 근거를 확인한 별도 후속 lineage로 결정한다.
## Fabric 400k · best_render · 500k 교정 입력 재추론

2026-10-08 추가 지시: [중앙 20×20 평균 gain 재추론](artifacts/fabric_checkpoint_comparison_patch20_20261008/index.html). 저자 실측 코드의 `[118:138,118:138]` RGB 평균 비율로 K_L을 계산하고 세 weight의 21채널·기하·decoder를 유지한다. 아래 고정 gain 비교는 이전 기록이다.

[비교 리포트](artifacts/fabric_checkpoint_comparison_20261008/index.html) · [수치 검증](artifacts/fabric_checkpoint_comparison_20261008/verification.json) · [브라우저 기능 검사](artifacts/fabric_checkpoint_comparison_20261008/browser_checks.json).
2026-10-08 사용자 승인에 따라 다섯 묶음에서 400k 종료, 449k best_render, 누적 500k 최종 weight를 모두 새 FP32 추론했다.
현재 교정된 정합·photometry를 사용하고 동일 선형 RGB와 Fabric 21채널 특징을 세 weight에 전달한다. 입력 PNG·8-bit 양자화·gamma·denoiser는 사용하지 않는다.
078·503은 5장 평균이며 나머지는 near_00/far_00이다. 기존 리포트 형식과 감마 off 기본값을 유지하고 이전 결과는 보존했다.
세 weight 모두 raw_calibrated_v2 고정 gain=(10/2.414)²를 유지하며 저자 real33과 구분한다. best_render의 실제 누적량은 449k다.
재현은 `python run_fabric_checkpoint_comparison.py --out artifacts/<새 경로>`이며 새 정합 검토를 기록해야 한다. [승인 범위와 계약](docs/CAPTURE_REQUIREMENTS.md#11-2026-10-08-사용자-승인-fabric-checkpoint-재비교)을 따른다.


## 색 cue · Scale-space 대화형 통합 비교

[단일 대화형 HTML](artifacts/cue_explorer_20261007/report.html) · [설계와 반례](docs/cue_multiscale_design.md) · [UI 검사](artifacts/cue_explorer_20261007/UI_CHECKS.md).
078·478 각각 Baseline, Fullres, Lowpass, Detail, Mixed, Scale-space를 동일 원본 pretrained 초기값에서 새 FP32 계산했다.
모든 조건에 fresh Adam .01, F0=.04·q 고정, 최대10,000 step과 같은 안정화 종료 기준을 적용했다.
Baseline은078에서9900,478에서2100 step에 안정화 기준을 충족했다. 나머지10회는10,000 step 상한까지 실행했으며 안정화 기준을 충족하지 않았다.
Scale-space는 normal5·9·17px와 roughness1·5·9px의 초기 정규화 cue를 component별 평균한다. 총 cue 가중치는 Mixed와 같으며9px 미세 normal 보존항을 유지한다.
단일 파일에 현재12회, 보존된1k 탐색, 공간 상수 초기/종료, 기존 fit, 원본 및 공통 fixed-F0 초기 상태를 담았다.
방법 선택, 동기화 ROI, A/B·wipe·각도/부호 차이, 원시 FP32 값, 공통 scale별 수치, loss·종료 이력, 실제 저장 frame, CSV/JSON/PNG 및 현재 화면 인쇄를 제공한다.
색 cue는 색과 함께 변하는 성분에 대한 soft penalty다. 동일한 맵 값을 강제하지 않으며 원본 미세 normal 보존도 실밥의 물리적 정답을 보장하지 않는다.
Scale-space는 Mixed 대비 두 sample의5·9px roughness cue를 줄였지만 실측·보존 MSE와 원본 대비 normal 고주파 오차는 늘었다.
478의 roughness9px cue/원본 비율은7.13324→0.00208088, 실측 MSE는1.65695e-6→1.77731e-6, normal 고주파 상대 RMS 차이는0.0294022→0.0378413이다.
CPU 테스트30개와12회 저장 결과·설정 검사208개가 통과했다. 전체 수치는 [final_metrics.json](artifacts/cue_explorer_20261007/final_metrics.json), 검사 요약은 [data_checks.json](artifacts/cue_explorer_20261007/data_checks.json)에 있다.
HTML은 약239.6MiB이며32개 비교 상태와234개 실제 montage를 내장한다. 원본 파일의 크기·mtime 및 배열 byte 동일성을 검사하며 SHA-256은 실행하지 않았다.
재생성: `python build_cue_explorer.py`. 새 최적화 없이 저장 결과를 패키징하며 이전 artifact를 보존한다.

## Cue 제약·보존 가중치 통합 비교 · 478 / 078

[단일 HTML 리포트](artifacts/cue_constrained_refinement_20261007/summary.html) · [실험 계획](artifacts/cue_constraints_lambda_20261007/experiment_plan.json) · [검증 기록](artifacts/cue_constraints_lambda_20261007/verification.txt) · [Sol λ 검토](artifacts/cue_constraints_lambda_20261007/lambda_review.md).
478과 078을 별도 섹션으로 나누고 같은 열·크기·표시 변환을 사용한다. 각 조합에 render − I_N을 추가하고 샘플별 공통 범위로 표시한다. 판정은 `step 수 - 수렴/수렴X`다. 각 조건의 체크박스가 이미지와 수치 행을 함께 제어한다.
478은 diffuse 상수 / diffuse+roughness 상수, 078은 normal 상수 / normal+roughness 상수를 각각 λ=0.1/0.5로 비교한다.
기존 3개 결과를 재사용하고 5개 조건을 새 FP32 계산했다. 원본 pred, q, F0=.04, 40방향, fresh Adam .01, seed0, 최대10k와 종료 기준을 유지했다.
078 normal-only는 두 λ 모두10k 상한에서 안정화 미충족이며, 나머지 조건은 안정화 기준을 충족했다.
가중 전 측정·보존 MSE를 비교하며 λ가 다른 total loss를 직접 순위화하지 않는다. roughness 공간 변동은 정답 정확도의 근거가 아니다.
8개 조건의 입력 동일성·상수 제약·FP32·손실 및 기존 수치 파일 보존을 검사했다. 새 계산은 저장 맵 재렌더링 검사를 통과했다.
HTML에 모든 이미지와 스크립트를 포함한다. 재생성: `python build_cue_constraints_report.py`.
전체 검증: `python artifacts/cue_constraints_lambda_20261007/verify_cue_constraints_experiment.py --require-complete`.
새 실험 CLI: `python run_cue_constrained_refinement.py --sample 260818_175005_078 --constants n,r --lambda 0.5 --out <새 artifact 경로>`.

## 색 cue와 미세 normal: TDD 탐색 실험

[비교 HTML](artifacts/scale_cue_20261007/summary.html) · [PDF](output/pdf/scale_cue_20261007.pdf) · [설계·반례](docs/cue_multiscale_design.md) · [TDD 기록](artifacts/scale_cue_20261007/TDD.md).
078·478을 동일 원본 pred에서 no-cue/fullres/lowpass/lowpass+detail 네 조건으로 각1000 step 계산했다.
FP32·F0=.04·q 고정, 중심 실측 MSE+0.1×40방향 원본 응답 보존에 고정 색도 cue를 추가했다.
Lowpass+detail은 원본 미세 normal을 유지했으나478의 원래 해상도 roughness cue가 증가했다.
현재 설정은 기본 경로로 채택하지 않는다. 실제 정답은 없으며 cue 감소가 물리 정확도를 뜻하지 않는다.
관련 CPU 테스트19개와 합성 반례, 실제 renderer smoke 및 손실 재평가를 확인했다. 기존 수치는 보존한다.


## 이전 Cue 제약 refinement · diffuse / roughness 상수

[이전 비교 HTML](artifacts/cue_constrained_refinement_20261007/previous_summary.html) · [비교 이미지 PDF](output/pdf/cue_constrained_refinement_20261007.pdf) · [수치 검사](artifacts/cue_constrained_refinement_20261007/constraint_comparison.json).
저장된 저자 pretrained 원본 pred에서 시작하여 공간 평균으로 제약에 투영했다.
078은 normal 전체를 공유하고, 478은 diffuse RGB와 roughness를 공유하며 공유 값도 최적화했다.
FP32, F0=0.04·q 고정, I_N MSE + 0.1×40방향 원본 응답 보존, fresh Adam 0.01.
078은 10000 step 상한 종료(안정화 미충족), 478은 1400 step에 안정화 기준 충족.
원본 대비 측정 MSE는 각각 약99.2%·89.6% 감소했으나 기존 무제약 결과보다 측정·보존 오차가 높다.
공간 상수·원본 pred 동일성·최종 손실 재평가를 검사했다. 물리적 정답이나 geometry 보정 검증은 아니다.
기존 결과를 보존한다. 재실행은 `run_cue_constrained_refinement.py`의 OUTPUT을 새 경로로 지정한다.


## 현재 결과 · F0 고정 · 40방향 보존

[통합 HTML](artifacts/measured_refinement_20261007_unified/summary.html) · [PDF](output/pdf/measured_refinement_20261007_unified.pdf): 478·078·008의 원래 pred, 최종 pred, 차이와 re-rendering을 비교한다. 중앙 I_N MSE + 0.1 × 중심 제외 40방향 보존 MSE, F0=0.04 고정, FP32, 최대10k. 478은 원래 prediction에서 새로 계산해2100 step에 안정화 기준을 충족했다. 078·008의 기존40방향 결과(10000·7800 step)를 함께 사용한다. 세 샘플의 계산 이력은 통합 리포트의 source_manifest.json에 기록했다. 아래는 보존된 이전 기록이다.

## 이어 계산 · 40방향 보존 · 누적 10k 상한

[새 HTML](artifacts/measured_spherical_refinement_20261007_continue10k_40lights/summary.html)과 [PDF](output/pdf/measured_spherical_refinement_20261007_continue10k_40lights.pdf): center는 I_N에 맞추고 40개 off-center 방향은 원래 pred 응답을 보존한다(λ=0.1). 이전 3000-step 맵·loss를 정확히 재현하여 Adam 상태를 복원했다. 078은 누적10000, 008은7800 step에서 기존 안정화 기준을 모두 충족했다. 이전 리포트·수치는 보존했다.


## 측정 기반 refinement · λ=0.1

[새 HTML 리포트](artifacts/measured_spherical_refinement_20261007_lambda01/summary.html): I_N MSE + 0.1 × 41광원 응답 보존 MSE. 대상은 478·078·008이며 기존 q와 F0=0.04를 고정했다. 478은 2100 step에서 안정화 기준 충족, 078·008은 3000 step에서 기준 미충족으로 종료했다. [실험 해석](artifacts/measured_spherical_refinement_20261007_lambda01/ANALYSIS.md)과 [PDF](output/pdf/measured_spherical_refinement_20261007_lambda01.pdf)를 함께 제공하며 이전 리포트는 보존했다.


*NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting* (SIGGRAPH Asia 2024)의 fork다.
저자 checkpoint와 호환되는 estimator 및 Fabric 학습 코드를 유지한다.

## 현재 Fabric 리포트

2026-10-07 당시 refinement는 [구면 sampling runner](docs/archive/completed_refinement_sources.zip)다.
Refinement 대상은 `260818_174845_478`, `260818_175005_078`, `260930_152732_008`이다.
`260923_141241_844`와 `260923_161825_503`은 사용자 판정으로 성질이 달라 대상에서 제외한다.
제외 sample의 기존 수치 기록은 보존하며 새 실행·분석·리포트에는 포함하지 않는다.
이전 refinement·세 조명 배치 기록은 보존하고 새 실험 비교에서 보류한다.
R=4, θ=0/5/10/20/30/45°, θ>0에서 φ 45° 간격의 8방향(총41개)을
하나의 평균 rendering loss로 사용한다. F0=.04 고정, n/d/r만 갱신한다.
매 step 8개씩 gradient를 누적하되 전체41개 평균으로 정규화하고 Adam은 한 번 갱신한다.
평가 φ는22.5° offset이며 gradient·종료 판정에서 제외한다.
300–3000회와 100회 구간 안정화 기준을 사용하고 종료 맵이 주 결과다.
새 결과: `artifacts/spherical_refinement_20261007/summary.html`.
[전체 PDF 리포트](output/pdf/spherical_refinement_20261007.pdf)는 11페이지이며
대상3개 sample의 원래 맵·종료 맵·변화 영상과 dynamics·gradient를 포함한다.
초기 공동 Adam 평균 업데이트가 개별 θ ring 최댓값보다5%를 넘으면 ring별 구성으로 전환한다.
이는 보수적 업데이트 스케일 기준이며 발산·통계적 유의성 판정은 아니다.
현재 대상3개는 공동 구성에서 각각400/1300/1000회에 안정화 기준을 충족했다.
수치 trajectory는 각조건의 `condition.json`에만 저장하고 sample `metrics.json`은 metadata다.
위치는 균일 solid-angle 적분이나 GGX importance sampling이 아닌 선택 조명 평균이다.

[생성 규약과 상세 데이터 흐름](docs/CAPTURE_REQUIREMENTS.md)을 기준으로 실행한다.
활성 ROI는 inset=0 → 512 정렬 → 단일 46px 크롭(420×420, 종이 제외 샘플 5cm) → 256 축소다. 마커 내부 전체를 5cm로 간주하지 않는다.
문서는 한국어로 작성했으며 ISO 24495-1:2023의 공개된 간결한 언어 원칙을 적용했다.
입력 처리, 공통 정합, 평균 조건, 광도·색 변환, 33채널 특징, 추론과 표시를 단계별로 설명한다.

현재 경로는 5개 DNG 묶음을 독립 처리한다. 입력 처리와 추론은 FP32이며 원본 `net_g_real.pth`를 사용한다.
광도 조건은 `both=(D-B)/(W-B)` 하나다. denoiser는 사용하지 않는다.
양쪽 5장 모두 수치 검사와 무늬 정합 검토를 통과하면 side별로 평균한다.
실패하면 같은 묶음의 near_00/far_00을 사용하고 이유를 기록한다.

새 실행에는 새 artifact 디렉터리를 지정한다. 아래 경로는 예시다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepare-only --precision fp32 --out artifacts/fabric_capture_20261003_run01
```

각 묶음의 `prepare_review.html`을 확인하고 실제 판단을 `root_review.json`에 기록한다.
형식과 승인 조건은 [실행 절차](docs/CAPTURE_REQUIREMENTS.md#12-실행-절차)를 따른다.
그 다음 같은 prepared 경로에서 추론한다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepared artifacts/fabric_capture_20261003_run01 --precision fp32
```

묶음별 `report.html`과 전체 `index.html`을 확인한다.
별도 `paper_comparison`과 `--paper-equations` 옵션은 폐지했다.
기존 best_render 수치 결과는 보존하지만 현재 비교에서 제외한다.
표시 gamma는 기본 off이며 입력·예측 배열을 바꾸지 않는다.
2026-10-06 지정 리포트 전면 갱신은 NPZ 저장 없이 수행한다. 현재 결과는
`artifacts/fabric_capture_20261002_aligned_paper_equations/index.html`에서 확인한다.
각 묶음의 `refresh_20261006/`에 새 공통 좌표 정합·FP32 추론의 PNG·JSON·HTML만 저장했다.
기존 원본 해상도 FP32 배열을 읽어 갱신하는 `--refresh-no-npz` 절차는
[생성 규약](docs/CAPTURE_REQUIREMENTS.md#11-리포트-표시-규약)을 따른다.
원본 DNG 폴더에 입력 PNG를 다시 만들지 않는다.

2026-10-06 prediction-only refinement는 현재 5개 sample 리포트의
`Prediction refinement` 영역과 전체 index에 추가했다.
원본 가상 조명 렌더링을 목표로 `s=0.04`를 고정하고 n/d/r만 FP32로 최적화한다.
입력 영상은 광원 세기 산정에만 사용하며 refinement loss에는 넣지 않는다.
[전체 실행 설정·결과](artifacts/prediction_refinement_20261006_all_samples/summary.json)와
[비교 요약](artifacts/prediction_refinement_20261006_all_samples/summary.html)을 확인한다.
학습 조명 렌더링 차이는 5개 모두 줄었지만 미사용 조명에서는 2개 sample의 원본 대비 차이가 증가했다.
재실행은 새 출력 경로를 지정한다.

```powershell
python run_prediction_refinement.py --all-groups --device cuda --steps 300 --out-dir artifacts/prediction_refinement_new_run
```

카메라를 중앙에 고정한 후속 rendering-loss 분석은 F0=0.04 고정의 세 조명 조건을
center/small/wide 조명 조건에서 비교한다. Prior와 perturbation은 사용하지 않는다.
[결과·dynamics 분석](artifacts/rendering_loss_analysis_20261006/ANALYSIS.html)과
[전체 trajectory·설정](artifacts/rendering_loss_analysis_20261006/summary.json)을 확인한다.
현재 300-step 실행에서 small은 5개 중 4개 sample의 공통 미사용 조명 렌더링 차이가 가장 낮았지만,
`260923_161825_503`은 모든 조건에서 초기 고정-F0 상태보다 미사용 조명 렌더링 차이가 증가했다.

## 설치와 검증

[2026-10-12 push 전 ponytail 감사·수정·검증 범위](docs/PONYTAIL_AUDIT_20261012.md).

Python 3.10/3.11과 장치에 맞는 PyTorch를 설치한 뒤 의존성을 설치한다.
DNG 처리는 rawpy가 필요하다. 모델의 CPU 실행은 `--device cpu`로 지정한다.

```powershell
python -m pip install -r requirements.txt
python -m pip install "pytest>=7,<9"
python -m pytest tests -q -p no:cacheprovider
```

## Fabric 학습

MatSynth Fabric/Leather의 이전 run 폐기·재개 금지 원칙은 2026-10-08 사용자 지시로 철회했다. 이 재개 대기는 당시 기록이다. 현재 후속 실행 여부와 scheduler 구현 상태는 위 현재 학습 상태에서 확인한다. [현재 학습 계약](docs/FABRIC_LEATHER_TRAINING.md)을 따른다.

학습 entry point는 `train_fabric.py`다. 데이터 준비는 `python -m data.fabric --help`로 확인한다.
학습에서 만든 custom weight는 현재 DNG 리포트의 허용 추론 weight가 아니다.

```powershell
python train_fabric.py --manifest data/fabric_native256_v1/manifest.json --out checkpoints/fabric_scratch
```

이전 source 계약을 기록한 checkpoint의 exact optimizer resume는 거부한다.
정리 과정에서 migration whitelist를 늘리지 않았다.
학습과 재실행에는 해당 dataset과 로컬 checkpoint가 필요하다.

## 문서와 자료 보존

현재 규약은 [CAPTURE_REQUIREMENTS.md](docs/CAPTURE_REQUIREMENTS.md) 한 곳에서 관리한다.
[과거 문서와 정리 증거](docs/archive/README.md)는 당시 판단과 실행 이력을 보존한다.
과거 문서를 현재 실행 지시로 사용하지 않는다.

capture data, checkpoint, cache, artifact는 로컬 자료다.
`.git`, `checkpoints`, `results`, dataset junction의 외부 대상을 재귀 삭제하거나 이동하지 않는다.

## 인용

```bibtex
@article{10.1145/3687978,
  author = {Wang, Li and Zhang, Lianghao and Gao, Fangzhou and Kang, Yuzhen and Zhang, Jiawan},
  title = {NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting},
  journal = {ACM Transactions on Graphics},
  volume = {43}, number = {6}, articleno = {274}, year = {2024},
  doi = {10.1145/3687978}
}
```

F0 고정 분석의 샘플별 본문에는 정성 관찰과 같은 크기의 맵·중앙 렌더링 비교를 통합했다. [수정 보고서](artifacts/rendering_loss_analysis_20261006/summary.html), [Council 지적 반영 기록](artifacts/rendering_loss_analysis_20261006/council_review/integration.html)을 확인한다.

변화 분석은 픽셀별 normal angular change와 물리 diffuse/roughness/F0의 signed difference heatmap을 본문에 표시한다. 광원 이름은 중앙 1광원 및 XY 반경 0.25/1.50의 각 5광원이다. 기존 조건을 FP32로 재현해 수치 맵 차이를 계산했고 표시용 PNG를 차감하지 않았다. [재현 일치 검사](artifacts/rendering_loss_analysis_20261006/change_fields/checks.json)를 보존한다.

## Fabric + Leather 학습용 subset 준비

DeepMaterials는 학습에서 제외한다. `prepare_fabric_leather_subset.py`는 기존 C:/MatSynth_PRIMA_Fabric의 Fabric 자료를 보존하고 Leather의 diffuse/specular/roughness/normal 네 map만 추가한다. 통합 split은 `manifests/fabric_leather_split_manifest.jsonl`이며 기존 Fabric 전용 manifest와 구별한다. 생성·학습 계약은 [FABRIC_LEATHER_TRAINING.md](docs/FABRIC_LEATHER_TRAINING.md), 노출량·분포 점검은 `audit_fabric_leather_data.py`를 따른다. Dataset 준비는 학습 실행이나 trainer 연결 완료를 의미하지 않는다.

## 학습 live HTML / report 규약

Train/Val 통합 plot, 공통 범위·Y-only log 선택, 독립 raw/이동평균 축약, 정량 status/train/val 표와 Normal angle 집계는 [현재 HTML 규약](docs/TRAINING_HTML_REPORT_CONTRACT.md)을 따른다. 독립 세션은 해당 문서와 기존 live_training_plot.html 템플릿에서 시작한다. [학습 계약](docs/FABRIC_LEATHER_TRAINING.md)과 연결되어 있다.

## 2026-10-09 단일 seed 중간 모델

사용자 승인한15.09M(3단계·middle12)·D4only·단일seed19는 완료됐다. [설정과실행](experiments/intermediate_capacity/README.md), [전환상태](artifacts/intermediate_capacity_15m_seed19_20261009/transition_status.json), [학습상태](artifacts/intermediate_capacity_15m_seed19_20261009/status.json)를 기준으로 확인한다. 기존regularization은두seed결론을 저장했고, 가드 오류로 실행된seed31의224updates는 중단·제외했다. 새모델은같은seed19의7.61M D4only와 비교한다. 데이터·평가·초기LR·상한을 유지하고결과는별도보존한다.

2026-10-09 사용자 지시: 앞으로 새 학습의 기본 scheduler는 ReduceLROnPlateau다. [Scheduler 정책과 적용 상태](docs/TRAINING_SCHEDULER_POLICY.md), [두 용량 모델의 검토 근거](artifacts/scheduler_council_20261009/evidence.json)를 따른다. 기존 고정 LR 결과는 보존한다.
