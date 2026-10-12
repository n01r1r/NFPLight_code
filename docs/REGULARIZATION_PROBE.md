# D4·dropout 2×2 과거 고정 진단 규약

현재 범위는 [학습 상태](CURRENT_TRAINING_STATE.md), 후속 기본값은 [scheduler 정책](TRAINING_SCHEDULER_POLICY.md)을 먼저 확인한다. 아래2026-10-08 계약은 고정 config 보존용 이력이며 새 학습의 정책을 덮어쓰지 않는다.

사용자 승인: 2026-10-08. 기존 baseline을 대조군으로 재사용하고 D4만, dropout만, D4+dropout 세 조건을 새로 준비한다. Dropout p=0.1을 먼저 검사하며 더 높은 확률은 이번 규약에 포함하지 않는다. 현재 baseline extension과 기존 FC 세 조건의 소스·설정·결과는 보존한다.

| 조건 | D4 | NAFBlock residual dropout |
|---|---|---|
| 기존 baseline | off | 0 |
| D4 only | on | 0 |
| dropout only | off | 0.1 |
| D4 + dropout | on | 0.1 |

- 기존 7,609,834-parameter TwoBranchNet backbone과 초기 seed 7/19/31을 유지한다. Dropout은 모든 NAFBlock의 기존 dropout1/2 위치에 elementwise `nn.Dropout`으로 적용한다. 36개 NAFBlock의 dropout 위치 72곳이며 새로운 가중치는 없다. 평가에서는 dropout을 끈다.
- 같은 material/crop 분리, batch 4, Python sample RNG(seed+10000)를 사용한다. D4에는 별도 Python RNG(seed+20000)를 사용한다. Dropout과 model RNG 상태도 checkpoint에 보존한다.
- D4는 균등한 8개 직교 변환이다: 90° 배수 회전 후 horizontal flip 여부. normal XY 벡터와 모든 GT map의 공간 배치를 함께 변환한다. `data.fabric.pack_svbrdf`의 기존 방향 규약을 재사용한다. 증강은 map/render 학습에만 적용하고 평가 crop은 변환하지 않는다.
- 변환된 GT에서 valid mask·near/far 관측·21채널 features를 재생성한다. 이미 계산한 features 전체를 이미지처럼 회전하지 않는다. Canonical roughness floor 0.05, F0=0.04, unit normal과 source backface mask를 유지한다.
- FP32, near4/far12/gain9, AMP/TF32 off, map LR 5e-4, render LR 1e-4, Adam(.9,.999), clip norm1, 기존 loss와 조명을 유지한다. Scheduler와 best-weight 복원은 추가하지 않는다.
- 각 조건의 상한은 누적 map 10,000/render 3,000 updates다. map 종료 후 fresh Adam으로 render를 시작한다. 100회마다 평가/체크포인트를 기록하며 기존 기준인 Validation 10개 interval 연속 상대 개선<0.005로 조기 종료한다. map은 heldout NDR mean, render는 heldout novel-light render L1이다.
- 상한은 같아도 조기 종료 시 실제 updates는 다를 수 있으므로 map/render 실제 종료 횟수와 stop reason을 함께 보고한다. Render graph의 x축은 실제 map 종료 step부터 시작한다.
- 기존 pilot map1,000+render300은 historical diagnostic이며, 새 main lineage는 기존 baseline extension의 최종 map→render lineage와 비교한다. 기존 pilot render를 새 map 단계의 warm start로 사용하지 않는다. 새 세 조건은 동일 seed의 원래 초기 backbone에서 시작한다.
- 실행 진입점과 설정은 `experiments/regularization/`에 둔다. 새 결과는 `artifacts/regularization_probe_20261008/`에 분리한다. 이는 당시 prepare/CPU 검증 단계의 계약이었다. 이후실제선택실행은아래완료기록을따른다. 새 run은 기존 baseline extension의 세 seed가 모두 complete이고 참조 정체성이 맞는지 GPU 초기화 전에 검사한다.

## 판단과 제한

Train·같은 material의 미학습 crop·별도 material의 map 성분/normal 각도/novel-light render를 구분한다. 2×2 effect는 동일 seed의 네 조건이 모두 완료된 뒤만 보고한다. 부분 실행을 전체 비교 완료로 표시하지 않는다. 이 검사는 공간 방향 증강과 residual dropout의 효용을 평가하며 새 material·brightness의 추가나 관측 식별 한계의 해결을 가정하지 않는다. 3개 seed는 같은 데이터의 초기화 반복이다.

기존 baseline 대조군의 데이터·모델·precision·loss·예산·평가 정체성을 비교 metadata에 기록한다. 정체성이 맞지 않거나 결과가 아직 끝나지 않으면 effect 계산을 거부한다. Path/size/mtime_ns로 보존을 확인하며 SHA-256은 계산하지 않는다.

CPU 사전 검사: 기존 GT D4와 renderer의 near4/far12 공변성 16개 검사 및 normal 단위 길이 확인이 통과했다. [검사 기록](../artifacts/bottleneck_fc_analysis_20261008/d4_cpu_covariance_check.json). 실행 중 baseline의 과적합·입력 모호성 분석은 [독립 판정](../artifacts/bottleneck_fc_analysis_20261008/COUNCIL_VERDICT.md)을 참조한다.

## 현재 선택 완료와 제외 범위

[검증된 root](../artifacts/regularization_probe_20261009_checked/artifact_context.json)에서seed7/19의6개새run을완료했다. [결과](../artifacts/regularization_probe_20261009_checked/two_seed_conclusion.json)·[선택scope](../artifacts/regularization_probe_20261009_checked/user_scope_completion.json)·[제외incident](../artifacts/regularization_probe_20261009_checked/seed31_exclusion_incident.json)를확인한다. Seed31은가드오류로224updates실행후중단·제외했으며0updates가아니다. 원래3-seed12결과비교완료로표시하지않는다. 오래된20261008preparedsourceidentity는재사용하지않는다.

15.09M D4-only seed19도완료됐으며[별도screen](../experiments/intermediate_capacity/README.md)을따른다. ReduceLROnPlateau후속의세부조건·새학습은미확정/미구현이다. [Council](../artifacts/scheduler_council_20261009/COUNCIL_VERDICT.md)은권고이지사용자승인config가아니다.
