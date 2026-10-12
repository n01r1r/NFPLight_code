# 학습 scheduler 정책

## 독립 세션의 적용 순서

최신 사용자 지시와 [AGENTS.md](../AGENTS.md)를 확인한 뒤 이 문서 → [현재 학습 상태](CURRENT_TRAINING_STATE.md) → 해당 실행의 고정 config·identity·완료/복구 기록 순으로 읽는다. [기계 판독 요약](TRAINING_POLICY_STATE.json)은 이 문서의 확정·제안·미구현 구분을 그대로 기록한다. 문서 날짜가 최신이라는 이유만으로 제안을 승인 정책으로 승격하지 않는다.

## 확정된 사용자 지시

- 앞으로 새 네트워크 학습의 기본 scheduler는 **ReduceLROnPlateau**다. 다른 scheduler나 고정 LR은 별도 비교 조건·기존 실행 이력으로 명시한다. 기존 실측 map refinement의 승인된 optimizer 조건에 소급 적용하지 않는다.
- 현재 reduced-param 후속은 단일 seed19로 우선 진행한다. 이 seed 선택을 앞으로 모든 별도 연구의 영구 기본값으로 확대하지 않는다. 새 seed31 실행을 예약하지 않는다. 기존 baseline seed31 기록과 regularization의 제외된224 updates는 보존하되 후속 비교에 섞지 않는다.
- 7.61M D4+dropout이 미수렴이면 ReduceLROnPlateau를 도입하는 방향을 사용자가 요청했다. LR schedule·상한 연장은 허용돼 있다. 2026-10-09 Go 이후 현재 단일 seed19 후속은 [새 계약](PLATEAU_FOLLOWUP.md)의 validation best·단계 전환 규칙을 구현한다.
- FP32, AMP/TF32 off, near4/far12·gain9·F0=.04 등 해당 실험의 데이터·관측·평가 계약을 유지한다. 모델 용량과 dropout이 다른 결과를 capacity만의 효과로 해석하지 않는다.

## 제안이며 확정되지 않은 값

[Council 결론](../artifacts/scheduler_council_20261009/COUNCIL_VERDICT.md)은 validation NDR 기준 map 학습, best-map 복원 후 map+render, fresh rerun을 권장했다. Eval100, factor.5, patience5(6 bad evaluations 후 감소), threshold rel.005, cooldown2, map LR5e-4→1e-5·cap20k, render LR1e-4→1e-6·cap5k, 상대2% map 보호 기준은 **실험 제안**이다. Floor 이후10/20회 대기에는 합의가 없다. 이를 승인된 config·종료 기준·성능 개선 사실로 쓰지 않는다. 위 내용은 당시 Council 제안이다. 이후 사용자의 Go로 현재 후속 목표는 validation NDR best 확보로 정해졌으며 [새 계약](PLATEAU_FOLLOWUP.md)이 운영 예산과 전환·연장을 구체화한다. 과거 Council의 숫자와 합의 여부를 소급 변경하지 않는다.

Train loss 감소, validation 개선, budget 종료, low-LR 탐색은 다른 사건이다. Budget cap을 수렴 완료로 표기하지 않는다. 반복 사용한8재질은 validation이며 독립 test가 아니다. 초기 best의 NDR 평균이 낮아도 모든 map 성분이 동시에 최고였다는 뜻은 아니다.

## 구현·실행 상태와 적용 gate

**기존 전역 trainer의 기본값 변경은 미구현이다. 현재 승인 범위는 [별도 후속 runner](../experiments/plateau_followup/README.md)에서 구현·CPU51검사·FP32 GPU smoke를 통과했다. 실제 시작 여부는 현재 상태를 따른다.** `train_fabric.py`와 독립 `train_fabric_leather.py`는 기존 cosine 구현이고, reduced-param 두 패키지는 기존 고정 LR 구현이다. 이들의 옛 run 명령을 새 기본 정책 구현으로 간주하지 않는다.

새 기본 run은 계약을 구현하고 검사한 새 lineage에서 시작한다. 기존 실행의 immutable config·source/data/control identity·checkpoint·수치는 덮어쓰지 않는다. 이미 사용자에게 허용된 조정은 그 범위에서 진행하고, 미답인 과학적 종료·전환 의미를 승인됐다고 추정하지 않는다. 구현 확인에는 단계 내 Adam 보존, model/optimizer/scheduler/RNG/phase/eval/stop 상태 복원, eval dropout off 및 train RNG 영향 방지, raw best 저장과 scheduler threshold 분리, best-map에서의 단계 전환 검사가 필요하다.

진단은 `python doctor_training_policy.py`로 실행한다. [Doctor 결과](../artifacts/policy_doctor_20261009/report.json)는 문서·link·artifact scope 일관성과 보존 검사를 보고하며 scheduler의 성능이나 구현 완료를 인증하지 않는다.

Doctor는read-only 검사이며trainer 실행을기술적으로차단하지않는다. PASS는해당문서·기록의검사범위에서일관성을뜻하며독립세션의준수나미구현scheduler동작을보장하지않는다. `--out`은보고서위치만바꾸며고정된보존baseline을생략하지않는다.

2026-10-09 audit 수정: 새 기본 학습은 scheduler 미구현 상태에서 CUDA·학습 출력 생성 전에 차단한다. 명시적 기존 resume은 원래 contract·identity 검사를 계속 거친다. 기존 smoke 검사는 허용한다. 이는 scheduler 구현 완료를 뜻하지 않는다. 변경 전 source는 [ZIP](../artifacts/ponytail_repairs_20261009/before_sources.zip), 승인된 변경 전후 identity는 [수정 기록](../artifacts/ponytail_repairs_20261009/source_repairs.json)에 보존한다. 과거 lineage의 현재 source 불일치는 숨기지 않으며 재시작에는 보존 원본이 필요하다.

## 2026-10-09 Go 후속

사용자는 validation best 중심의 새 7.61M D4+dropout·seed19 실행을 승인했고 넉넉한 예산·메모리 제한·체크포인트 연장을 요청했다. [현재 후속 계약](PLATEAU_FOLLOWUP.md)을 적용한다. 최초 map50k/render15k, floor20회 평가 대기, budget cap에서 단계 유지·일시정지, 명시적 예산 증가 또는 stage-end에서 새 lineage로 연장한다. 기존 trainer 전역 기본 변경과 별도 scoped runner 구현은 구분한다. 구현·검사·실제 시작 여부는 기계 상태와 현재 상태 문서에 기록한다.

## 추가 suite 승인

2026-10-09 사용자가 추천한 별도조건 전체의 개시를 승인했다. [새 suite 계약](ROBUSTNESS_SUITE.md)에서 단일seed19의15.09M/.1,7.61M/.5,증강·EMA·AdamW와진단을 별도lineage로 구현·검증후실행한다. 이전Council·초안은이력이며현재승인은이번지시에근거한다. 기존전역trainer기본값미구현gate와원본후속보존규칙은유지한다.


## 2026-10-10 승인된 15M·24crop 단일 후속 실행

사용자 “진행하고 병렬로 15m에서 시도 가능한 다른 변주들을 비판적 사고로 논의” 승인으로 [새 계약](../artifacts/capacity15m_extra24_seed19_20261010/EXPERIMENT_CONTRACT_KO.md)의 seed19·FP32·15,085,034 parameters·24crop·dropout .1 fresh 실행을 개시했다. 실제 경로는 `artifacts/capacity15m_extra24_seed19_20261010/run`이다. 이전 suite7조건은 완료·보존이며 재개하지 않는다. smoke 및 *_prepared_gate_v1은 학습 개시 전 검증 이력이다. [실제 상태](../artifacts/capacity15m_extra24_seed19_20261010/run/status.json) · [검증](../artifacts/capacity15m_extra24_seed19_20261010/launch_verification.json) · [Live viewer](http://127.0.0.1:8783/) · [Sol/Astra 조건부 후속변주](../artifacts/capacity15m_extra24_seed19_20261010/analysis/VARIANTS_DECISION_KO.md). 다른 변주는 분석만 승인되었으며 자동 학습하지 않는다. 10분 heartbeat `15m-24crop-fp32`가 진행·PID identity·checkpoint·디스크를 매회 보고한다. 시작·저장 전 disk reserve gate를 새 wrapper에 적용했고 공간 부족 시 committed checkpoint를 보존하며 자동 재시작하지 않는다. 이전 heartbeat `7-61m-plateau`는 PAUSED 유지다.


## 2026-10-10 승인된 7M24 factor0.8 fresh 실행
사용자 추가 예산 및 7M·24crop·dropout.1 선택으로 map update0부터 새 seed19 FP32 학습을 개시했다. [계약](../artifacts/plateau08_7m24_seed19_20261010/EXPERIMENT_CONTRACT_KO.md) · [검증](../artifacts/plateau08_7m24_seed19_20261010/verification.json) · [상태](../artifacts/plateau08_7m24_seed19_20261010/run/status.json) · [Live viewer](http://127.0.0.1:8784/). 실제 run/만 활성이고 smoke/는 CPU25검사 후 두 단계 FP32 GPU 구현검증 이력이다. 원래 7M24와 초기 가중치·24crop·scientific settings 동일; map/render factor.5→.8 및 render cap15000→25000만 변경했다. map cap50000·floor20·초기 LR·손실·선택 규칙은 유지하며 cap은 일시정지이고 수렴이 아니다. render-only 15M24 factor.8 준비안은 실제 개시하지 않고 superseded.json으로 대체를 기록했다. 기존15M24 및7조건 완료 lineage와 heartbeat는 PAUSED 유지한다. 이 실행 외 후보·재시작·추가 연장은 자동 실행하지 않는다.


## 2026-10-11 새 손실 계약과 전체 데이터 범위

사용자는 normal 유지·NDRS 성분 평균 대신 합을 확정했다. [데이터·손실 정책](TRAINING_OBJECTIVE_POLICY.md)을 새 전체 학습에 적용한다. Unit-normal XYZ·S=.04 감독·NDRS 합, 기존 FP32/gain9/F0 계약을 유지한다. 기존8재질 scoped runner의 승인 당시 평균/no-S 계약은 역사적 lineage이며 소급 변경하지 않는다. 전체 trainer의 손실 v2 구현과 ReduceLROnPlateau 전역 미구현 gate는 별개다. 이번 지시는 GPU 학습 개시나 정지 run 자동 재개로 해석하지 않는다.


## 2026-10-11 전체 데이터3조건 자동진행

사용자는Extra-crop/EMA제외와1–3자동진행·적절한상한을요청했다. [새계약](../artifacts/full_dataset_three_20261011/EXPERIMENT_CONTRACT_KO.md)에따라새scoped전체runner에서ReduceLROnPlateau를구현·검증했다. Map/render monitor는NDRS합/novel12render,평가1000update·floor20·상한400k/100k를운영선택으로적용한다. 미구현전역기본값과이새scoped구현을구분한다. 승인1–3외자동학습은하지않는다. 상한도달은pause,실패·memory/diskpause는queue중단·검토이며자동재시작하지않는다.
