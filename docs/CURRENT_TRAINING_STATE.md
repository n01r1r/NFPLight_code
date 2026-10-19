# 현재 학습 상태와 결과 탐색

2026-10-19 전달 점검: A–C queue는 complete이며 C는 render10k best-render 선택이다. D queue는 Windows `psutil.disk_usage(Path)` TypeError로 failed/stopped다. 호출을 문자열 경로로 수정했으나 재실행하지 않았다. 기존 아래 D 대기 설명은 계획 이력이며 실제 상태는 queue를 따른다. [weight와 코드 점검](WEIGHT_HANDOFF_20261019.md).

2026-10-19 15:21 KST 확인: A–C10k 및 합성/실측 보고 완료, 통합 실측 HTML에 clean/A/B/C 반영. D watcher는15:12 자원 기록의 WindowsPath/psutil TypeError로 GPU smoke·학습 시작 전에 중단됐다. D 결과가 없어 주 비교 교체 판단은 유보한다. 실패 시 자동 재시작 금지에 따라 재실행하지 않고 heartbeat를 일시정지한다. [중단 원인·검증·미완료 범위](../artifacts/render_only_followup_20261019/FAILURE_KO.md) · [실패 검증](../artifacts/render_only_followup_20261019/failure_verification.json). 아래 D 대기/준비 문단은 개시 전 이력이며 실제 상태는 이 기록과 queue를 우선한다.

2026-10-19 최신 사용자 승인: A–C 학습을 유지하고 C의 학습·합성/실측 보고 정상 완료 뒤 같은 selected map135k에서 **D render-only, 초기 LR1e-5·10k**를 추가한다. B/C의 fresh Adam·초기 RNG·batch/D4·ReduceLROnPlateau·평가·NDRS합×1.02 선정 보호를 유지하고 map backward만 제거한다. 별도 worker가 [후속 queue](../artifacts/render_only_followup_20261019/queue.json)에서 대기하며 중복 실행하지 않는다. [계약](../artifacts/render_only_followup_20261019/EXPERIMENT_CONTRACT_KO.md) · [구현](../experiments/render_only_followup/README.md) · [준비 검증](../artifacts/render_only_followup_20261019/setup_verification.json). [기존3조건 실측 HTML](../artifacts/relmap_three_way_real_20261019/index.html)에 clean baseline/A/B를 옆 열로 추가했고 C/D는 검증된 실측 보고 완료 후 자동 반영한다. 원본 index 백업·기존 수치/이미지를 보존한다. D 실행·결과는 실제 queue/complete/COMPARISON_KO.md 생성 여부로 판단하며 아직 성능 우위를 주장하지 않는다. 별도 heartbeat `render-only`가 의미 있는 완료·실패·성분 tradeoff와 최종 주 비교 판정을 이 대화에 알린다.

2026-10-19 사용자 요청 map/render gradient 실측 완료: 현재 A–C 학습·설정을 유지하며 clean map135k/A render9k/기존 render49k의 동일 고정 batch에서 CPU FP32 12case를 계산했다. Eval render/map parameter norm은3.55–6.19%, cosine은9case 중2개 음수였다. 첫 batch의40조명 일치율.855–.900·가중 pair cosine+.724–+.804로 이 표본에서 심한 조명 상쇄설은 지지되지 않았다. 실제 Adam moment·일반화·render-only 우위는 미확정이다. A–C 폐기·새 학습은 실행하지 않았으며 clean baseline은 map135k다. [분석](../artifacts/render_gradient_diagnostic_20261019/ANALYSIS_KO.md) · [실측 전체](../artifacts/render_gradient_diagnostic_20261019/MEASUREMENTS_KO.md) · [검증](../artifacts/render_gradient_diagnostic_20261019/verification.json) · [Astra 반대 검토](../artifacts/render_gradient_diagnostic_20261019/COUNCIL_VERDICT_KO.md) · [미실행 후속 대조 초안](../artifacts/render_gradient_diagnostic_20261019/RENDER_ONLY_COMPARISON_DRAFT_KO.md).

2026-10-19 사용자 승인 실측 3조건 비교 완료: 기존 relmap Fabric64M(449k) / 기존 relmap 7M(누적185k, memory_paused 실행의 선택 weight) / 변경 relmap 7M(누적204k). [보고서](../artifacts/relmap_three_way_real_20261019/index.html) · [분석](../artifacts/relmap_three_way_real_20261019/ANALYSIS_KO.md) · [검증](../artifacts/relmap_three_way_real_20261019/verification.json). 5묶음×3모델을 새 CPU FP32 추론했고 relmap/log-relmap·NDRS·공통/원래 기하 재렌더·ΔE00을 분리했다. 변경7M의 공통 재렌더 ΔE00은 기존7M 대비10개 묶음·거리 조합 중7개에서 감소했으나 학습 경로·선택 시점 차이와 실측GT 부재로 relmap만의 효과나 정확도 우위는 주장하지 않는다. 학습·기본 추론 정책은 변경하지 않았다.

2026-10-19 최신 사용자 승인: map135k에서 A(weight1/LR1e-4) → B(weight1/LR1e-5) → C(weight10/LR1e-5)를 각10k update 순차 실행하고 조건별 정성 보고서를 제작한다. 현재 실행·단계는 [queue](../artifacts/render_weight_followup_20261019/queue.json)를 확인한다. 조건별 [라이브/오프라인 목록](../artifacts/render_weight_followup_20261019/REPORT_INDEX_KO.md): A8787·B8788·C8789, 별도 viewer와 snapshot을 사용한다. [계약](../artifacts/render_weight_followup_20261019/EXPERIMENT_CONTRACT_KO.md) · [구현/검증](../experiments/render_weight_followup/README.md). 10k는 선별 예산 종료이며 수렴이 아니다. 기존 relmap 학습은 render77k에서 floor20 정상 종료, best-render49k 선정·보존이며 기존 heartbeat는 완료 후 PAUSED였다. 새 queue는 학습→보고를 조건별로 순차 수행하고 실패·memory/disk pause 시 자동 재개하지 않는다.

2026-10-19 완료된 수정 relmap 7M의 실측5묶음×best-map135k/best-render49k 새 CPU FP32 추론·정성 보고 완료. [실측 보고서](../artifacts/spatial_7m_real_20261019/index.html) · [검증](../artifacts/spatial_7m_real_20261019/report_checks.json). A/B/C GPU 학습은 유지했으며 이 결과는 합성 test와 별도이고 실측GT 정확도 증거가 아니다.

2026-10-17 최신 사용자 지시: 15M 실행·viewer 중지 및 이번 15M 산출물 삭제 완료. 기존 전체 데이터 7M과 relmap만 비교하도록 baseline7m_p01 fresh 재학습 개시. 기존 policy·scientific settings(관계 버전 제외)·train/val 기록 정확히 일치, 7M smoke map2/render2 및 전체 복원 통과. [비교 계약](../artifacts/spatial_7m_retraining_20261017/comparison_contract.json) · [실행 기록](../artifacts/spatial_7m_retraining_20261017/launch.json) · [Live report](http://127.0.0.1:8786/). 종료 시 offline_report.html 자동 생성. 이전 7M 비교 기록은 보존한다.

2026-10-16 재학습 준비 완료: baseline7m_p01의 새 relmap 설정·730 shard 헤더 검사·CPU383 통과·GPU map2/render2 및 checkpoint 전체 복원 검증을 마쳤다. **본 학습은 미시작이며 15M 보류를 유지한다.** [준비 결과](../artifacts/spatial_retraining_20261016/README.md) · [실행 안내](../experiments/full_dataset_suite/README.md).

2026-10-16: 거리비율 relation 보정을 train·eval·inference에 공통 적용한다. `relation_version=spatial_inverse_square_v1`; 학습 gain9·실측 patch20 및 과거 결과는 유지한다. [계산·호환성 규약](REAL_SPATIAL_ATTENUATION.md)을 따른다.

2026-10-13 사용자 확정: 학습·합성 validation/test의 고정 gain9와 실측 중앙20×20 RGB 평균 비율 gain을 구분한다. 과거 실측 gain9 지시는 [CAPTURE_REQUIREMENTS §1.4](CAPTURE_REQUIREMENTS.md#14-2026-10-13-사용자-확정-학습과-실측-추론-분리)로 대체한다. 학습·공식 test 결과·가중치는 유지하며 [지정 snapshot 실측 재추론](../artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422_patch20_20261013/index.html)은 별도 결과다.

2026-10-10 Luna max 코드·문헌 분석과 Sol/Astra 3라운드 council을 완료했다. [표현·역렌더링·일반 DL 비교 및 방향성](../artifacts/representation_inverse_council_20261010/COMPARATIVE_ANALYSIS_AND_DIRECTION_KO.md) · [진단 결정표](../artifacts/representation_inverse_council_20261010/DIAGNOSTIC_DECISION_TABLE_KO.md)를 따른다. 기존292평가점의 성분별 상충·보존 가중치 감사만 추가했으며 새 특징 계산·추론·학습은 실행하지 않았다. 감사 순서에 합의했지만 원인과 다음 학습 개입은 미확정이다. 21채널 축소·24crop 재시도·용량 확대·normal 가중치 변경은 자동 실행안이 아니다.

2026-10-10 사용자 승인 후보 비교·실측 통제 추론을 완료했다. [7후보 비교표·방향성](../artifacts/robustness_suite_seed19_20261009/CANDIDATE_COMPARISON_KO.md)에서 extra_crop7m_p01을 후속 검증 후보로 유지한다. [새 실측 5묶음 HTML](../artifacts/robustness_real_extra_crop_20261010/index.html) · [실측 해석](../artifacts/robustness_real_extra_crop_20261010/REAL_COMPARISON_KO.md) · [검증](../artifacts/robustness_real_extra_crop_20261010/verification.json). 같은 크기 baseline 통제와 FP32 새 계산했으며 관측 재구성 L1 개선은10개 묶음·거리 조합 중3개로 일관된 실측 우위는 확인하지 못했다. 실측 GT와 절대 기하·광도 대응이 미확정이므로 기본 모델 교체·새 학습은 개시하지 않았다.

2026-10-09 **추천대로 전부 개시** 승인으로 [새 suite 계약](ROBUSTNESS_SUITE.md)의 용량·dropout·증강·강화·강건성 조건을 구현·검증하고 실제 직렬 학습을 시작했다. 7개 조건 모두 준비 완료이며 baseline7m_p01과 capacity15m_p01은 각각 map7100/render7400에서 완료했고 map500을 선택했다. dropout7m_p05도 map6700/render7500에서 완료했고 render100을 선택했다. gt_aug7m_p01은 map6700/render9200에서 완료했고 map100 가중치(render0)를 선택했다. extra_crop7m_p01도 map6900/render7400에서 완료했고 map300 가중치(render0)를 선택했다. ema7m_p01도 map8200/render7700에서 완료했고 EMA render300 가중치를 선택했다. 마지막 adamw7m_p01도 map6900/render7500에서 운영 plateau 종료했고 map300 가중치(render0)를 선택했다. 7조건 모두 완료, 최종 손상 진단과 clean 재현 검증도 완료했다. [최종 추천·수치 근거](../artifacts/robustness_suite_seed19_20261009/FINAL_RECOMMENDATION_KO.md)를 따른다. [실시간 대기열·학습 지표](http://127.0.0.1:8782/) · [현재 대기열](../artifacts/robustness_suite_seed19_20261009/queue_state.json) · [검증](../artifacts/robustness_suite_seed19_20261009/verification.json). 활성 job 경로는 arms_final/이며 arms/는 trainer가 시작되지 않은 초기 준비 이력이다. 이전 완료 후속과 새 suite를 구분한다.

독립 세션은 [scheduler 정책](TRAINING_SCHEDULER_POLICY.md)과 [기계 판독 요약](TRAINING_POLICY_STATE.json)을 먼저 읽는다. 2026-10-09 Go로 [새 후속 계약](PLATEAU_FOLLOWUP.md)의 단일 seed19 7.61M D4+dropout FP32 학습을 완료했다. map7100/render7400에서 운영 plateau 종료했고 map500 가중치를 선택했다. [결론·권장안](../artifacts/plateau_followup_verification_20261009/FINAL_RECOMMENDATION_KO.md)을 따른다. [실제 상태](../artifacts/plateau_followup_7m_seed19_20261009/status.json) · [Live viewer](http://127.0.0.1:8781/) · [검증 근거](../artifacts/plateau_followup_verification_20261009/verification.json)를 따른다. 기존 전역 trainer 기본값과 과거 고정 LR 구현은 별도로 보존한다. 아래 완료 실험은 **고정 LR의 과거 lineage**이며 새 기본값이 아니다.

|범위|현재 판정|실제 근거|
|---|---|---|
|7.61M baseline pilot/extension|기존 seed7/19/31 완료 결과 보존|[extension 완료 보고](../artifacts/bottleneck_fc_baseline_extended_20261008/FINAL_REPORT.md)|
|7.61M D4/dropout|선택 seed7/19의6개 새 run 완료; 원래3-seed 계획은 미완료|[두 seed 결론](../artifacts/regularization_probe_20261009_checked/two_seed_conclusion.json), [선택 완료 범위](../artifacts/regularization_probe_20261009_checked/user_scope_completion.json)|
|제외된 seed31|가드 오류로224 updates 실행 후 중단·비교 제외|[incident](../artifacts/regularization_probe_20261009_checked/seed31_exclusion_incident.json)|
|15.09M D4-only·dropout0·seed19|map10k/render3k 완료; FP32 최종 export 복구 완료|[complete](../artifacts/intermediate_capacity_15m_seed19_20261009/complete.json), [matched comparison](../artifacts/intermediate_capacity_15m_seed19_20261009/comparison.json), [export recovery](../artifacts/intermediate_capacity_15m_seed19_20261009/export_recovery.json)|
|ReduceLROnPlateau 후속|Go 승인; scoped 구현·CPU51검사·GPU smoke 통과, 7.61M 단일seed19 완료; map500 가중치 선택, 새 best 없음|[새 계약](PLATEAU_FOLLOWUP.md), [실제 상태](../artifacts/plateau_followup_7m_seed19_20261009/status.json), [검증](../artifacts/plateau_followup_verification_20261009/verification.json)|

## 결과·보존·해석

15M의 primary control은 같은 seed19의7.61M D4-only다. 7.61M D4+dropout 최종 모델은 선택 seed19에서 좋은 비교 후보지만 다른 dropout 조건이므로15M과의 차이를 capacity 하나의 효과로 설명하지 않는다. 초기 best-map 수치는 validation 로그에만 있고 가중치는 보존되지 않았다. 최종 `last.pt`를 초기 best라고 이름 붙이지 않는다.

`failure.json`·옛 launch/transition 기록은 사건 이력이다. 최신 상태는 complete·comparison·복구 검사와 함께 판단한다. 경로 이름에 `pending`이 있어도 현재 실행 단계라고 추정하지 않는다. 원래9-run config, 과거 seed31 baseline과 제외된 seed31 기록은 역사적 사실로 보존한다. `comparison_complete=false`는 원래3-seed 비교 미완료이며 선택된 두 seed 결과 부재를 뜻하지 않는다.

[Artifact 인덱스](../artifacts/training_artifact_index.json)의 role·context를 확인한다. 각 root의 `artifact_context.json`은 현재 적용 범위와 정책 링크를 제공하며 당시 수치/config를 새 정책으로 변환하지 않는다. Source snapshot·과거 council·준비 config·smoke PASS는 실행 허가나 현재 정책의 근거가 아니다.

## 실행 경로

[정식 데이터·학습 기록](FABRIC_LEATHER_TRAINING.md), [7.61M 고정 진단](../experiments/regularization/README.md), [15M 고정 진단](../experiments/intermediate_capacity/README.md), [기존 FC 진단](../experiments/bottleneck_fc/README.md)을 구분한다. 이 문서에서 과거 run 명령을 새 학습 명령으로 제시하지 않는다. 새 기본 scheduler를 구현·검사하는 작업과 새 GPU 학습은 실제 계약에 따라 별도로 기록한다.


2026-10-10 PI 피드백에 따른 [LibRaw·실측 입력 감사](../artifacts/representation_inverse_council_20261010/LIBRAW_REAL_INPUT_AUDIT_KO.md)를 추가했다. 저장 5묶음의 near/far metadata는 일치하지만 물리 보정의 타당성을 증명하지 않는다. 색 행렬·WB 의미를 우선 확인하며 현재 synthetic 학습 및 동결 입력을 변경하지 않았다. 새 DNG 처리·추론은 미실행이다.


## 2026-10-10 승인된 15M·24crop 단일 후속 실행

사용자 “진행하고 병렬로 15m에서 시도 가능한 다른 변주들을 비판적 사고로 논의” 승인으로 [새 계약](../artifacts/capacity15m_extra24_seed19_20261010/EXPERIMENT_CONTRACT_KO.md)의 seed19·FP32·15,085,034 parameters·24crop·dropout .1 fresh 실행을 개시했다. 실제 경로는 `artifacts/capacity15m_extra24_seed19_20261010/run`이다. 이전 suite7조건은 완료·보존이며 재개하지 않는다. smoke 및 *_prepared_gate_v1은 학습 개시 전 검증 이력이다. [실제 상태](../artifacts/capacity15m_extra24_seed19_20261010/run/status.json) · [검증](../artifacts/capacity15m_extra24_seed19_20261010/launch_verification.json) · [Live viewer](http://127.0.0.1:8783/) · [Sol/Astra 조건부 후속변주](../artifacts/capacity15m_extra24_seed19_20261010/analysis/VARIANTS_DECISION_KO.md). 다른 변주는 분석만 승인되었으며 자동 학습하지 않는다. 10분 heartbeat `15m-24crop-fp32`가 진행·PID identity·checkpoint·디스크를 매회 보고한다. 시작·저장 전 disk reserve gate를 새 wrapper에 적용했고 공간 부족 시 committed checkpoint를 보존하며 자동 재시작하지 않는다. 이전 heartbeat `7-61m-plateau`는 PAUSED 유지다.


2026-10-10 사용자 명시 승인으로 완료7M robustness6조건의 full optimizer/재개checkpoint·중복best·준비/smoke를 영구 정리했다. [보존·삭제 기록](../artifacts/cleanup_7m_variants_20261010/REPORT_KO.md)을 따른다. selected·고유best·단계 종료 FP32 모델·최종 수치/로그는 보존하며 이6조건의 exact resume는 불가하다. 15M 동결 의존성과 활성 실행은 유지했다. 이전 checkpoint inventory는 정리 전 이력이며 현존 여부는 cleanup manifest/alias를 따른다.


2026-10-10 15M24 단일 승인 실행 완료: map7100/render7400, floor20 역사적 best 미갱신 운영 종료. render0(map500) 선택, NDR0.176380/render0.016901. CPU FP32 export 및 tensor 동일성 검증 완료. [최종 비교·추천](../artifacts/capacity15m_extra24_seed19_20261010/FINAL_RECOMMENDATION_KO.md) · [export 검증](../artifacts/capacity15m_extra24_seed19_20261010/export_verification.json). 새 학습은 개시하지 않는다. 완료 heartbeat는 최종 보고 후 PAUSED로 전환한다.


## 2026-10-10 승인된 7M24 factor0.8 fresh 실행
사용자 추가 예산 및 7M·24crop·dropout.1 선택으로 map update0부터 새 seed19 FP32 학습을 개시했다. [계약](../artifacts/plateau08_7m24_seed19_20261010/EXPERIMENT_CONTRACT_KO.md) · [검증](../artifacts/plateau08_7m24_seed19_20261010/verification.json) · [상태](../artifacts/plateau08_7m24_seed19_20261010/run/status.json) · [Live viewer](http://127.0.0.1:8784/). 실제 run/만 활성이고 smoke/는 CPU25검사 후 두 단계 FP32 GPU 구현검증 이력이다. 원래 7M24와 초기 가중치·24crop·scientific settings 동일; map/render factor.5→.8 및 render cap15000→25000만 변경했다. map cap50000·floor20·초기 LR·손실·선택 규칙은 유지하며 cap은 일시정지이고 수렴이 아니다. render-only 15M24 factor.8 준비안은 실제 개시하지 않고 superseded.json으로 대체를 기록했다. 기존15M24 및7조건 완료 lineage와 heartbeat는 PAUSED 유지한다. 이 실행 외 후보·재시작·추가 연장은 자동 실행하지 않는다.


2026-10-11 사용자 "종료하고 대기" 지시로 7M24 factor.8 trainer를 종료했다. map 관측15628, committed last15600/previous15500 CPU identity·optimizer/scheduler/RNG 필드 확인. render 미개시, 완료·수렴이 아닌 interrupted 상태다. 로그·best 가중치는 보존하고 자동 재개하지 않는다. [중단 기록](../artifacts/plateau08_7m24_seed19_20261010/user_stop.json).


## 2026-10-11 데이터 범위 확인·새 손실 확정

최근8재질 진단은 전체 Fabric+Leather 학습이 아니었다. 실제 manifest는 train566(Fabric358/Leather208), validation164(98/66)이며 전체 trainer는 모든 split row를 사용한다. 사용자는 normal 유지와 NDRS 평균 대신 합을 확정했다. [새 데이터·손실 계약](TRAINING_OBJECTIVE_POLICY.md)에 따라 전체 trainer의 unit-normal XYZ 유지·N/D/R/S 합·S=.04 감독·best-map NDRS 합 선택을 구현했다. 기존8재질 runner/config/checkpoint·수치는 historical lineage로 보존한다. 새 전체 GPU 학습은 시작하지 않았으며 전역 scheduler gate는 미해결이다.


## 2026-10-11 최근 경량화 결과의 적용 범위 정정

사용자는 최근 진단셋 학습 가중치 폐기와 수정 손실에 따른 재실험을 지시했다. 최근8재질 결과는 전체 Fabric+Leather 모델 선택 근거에서 제외한다. 당시 관측·로그·설정은 역사적 진단으로 유지한다. [재조회·과학적 판정](../artifacts/lightweight_invalidation_20261011/REPORT_KO.md)과 [파일 목록](../artifacts/lightweight_invalidation_20261011/inventory.json)을 따른다. 재실험 조건 범위와 extra-crop의 새 의미는 미확정이며 전역 scheduler gate도 남아 있다. 물리적 삭제·새 GPU 학습은 아직 미실행이다.


## 2026-10-11 최근9조건 전체 재실험 승인·최근 가중치 폐기

사용자 "최근9조건 모두 재실험"으로 범위를 확정했다. [9조건 준비 목록](../artifacts/full_lightweight_nine_20261011/plan.json)을 따른다. 최근 경량화 가중치·파생 배열599개9.76GiB를 개별 삭제하고 부재 검사했다. 로그·설정·수치 기록은 보존하며 최근 경량화의 exact resume와 가중치 기반 재평가는 불가하다. [삭제 기록](../artifacts/lightweight_invalidation_20261011/deletion_manifest.json) · [배열 삭제 기록](../artifacts/lightweight_invalidation_20261011/array_deletion_manifest.json). 이전 가중치 보존·후보 유지 설명은 당시 이력이며 이번 지시로 대체된다. 저자 원본과 이전 전체 학습 모델은 삭제 대상에서 제외했다. crop2/3의 전체566재질 적용 여부는 사용자 답변 대기 중이다. 전체 streaming runner 구현·검증 후 fresh FP32/NDRS 합으로 시작하며 GPU 학습은 아직 미개시다.


2026-10-11 crop 설명 정정: 원본 bank는 재질당16crop이며 원본 train shard normal header의 shape(16,256,256,3)을 확인했다. 전체 재학습은 기존 계약의566×16=9056crop을 유지한다. 직전2/3crop 유지 권장과 해당 선택 질문은 철회한다.2/3crop은 과거8재질 진단에서만 사용한 subset이다.9조건 승인은 유지하며 과거 extra-crop 조건을 전체16crop에서 어떻게 재설계할지는 별도 미확정이다.


## 2026-10-11 승인9조건 critical redteam 재판정

[예정9조건 목록·재판정](../artifacts/full_lightweight_nine_20261011/CRITICAL_REDTEAM_KO.md)을 따른다.9조건승인은유지하지만실제새queue는미등록·GPU미개시다. 전체16crop에서기존extra7M/extra15M은baseline7M/15M과중복이므로새crop정의가필요하다. 권장16→24crop/샘플은아직제안이며원본감사·생성미실행이다.평가100/floor20을그대로옮기면floor대기노출량0.8834회라scheduler/예산의전체모집단기준재검토가필요하다.관련세heartbeat는PAUSED다.현재fulltrainer의40조명평가와구suite의12novel조명차이·NDRmean engine선택과새NDRS합차이를실행전해소해야한다.이검토에서새학습·자동화개시는하지않았다.


## 2026-10-11 최신 승인: Extra-crop/EMA 제외,1–3 자동진행

[새3조건 계약](../artifacts/full_dataset_three_20261011/EXPERIMENT_CONTRACT_KO.md)과[실제대기열](../artifacts/full_dataset_three_20261011/queue_state.json)을현재기준으로사용한다. 이전9조건은최신범위로대체했다.7.610M/.1→15.085M/.1→7.610M/.5,각scratch/seed19/FP32,train566개샘플폴더×16crop=9056,validation164개폴더×4crop/novel12조명,NDRS합이다. Extra-crop/EMA와증강/AdamW/factor.8/추가seed/64M은자동queue에없다. 넉넉한운영상한map400k/render100k,평가1000,floor20으로적용하며cap은pause이고수렴/자동stage전환이아니다. 새scoped전체runner의scheduler/선택/streaming을완성했고CPU25검사·HTML검사·3arm실제FP32map/renderGPU smoke를통과했다. Controller를시작했으며실제optimizer진행은queue/각run로그를확인한다. 기존전역trainer의schedulergate는별개다. 기존heartbeat3개는PAUSED유지,새실행은독립직렬controller다. [검증](../artifacts/full_dataset_three_20261011/verification.json) · [7Mlive](http://127.0.0.1:8785/) · [15Mlive](http://127.0.0.1:8786/) · [dropout.5live](http://127.0.0.1:8787/).


2026-10-11 실제시작확인: 새7Mbaseline의전체164폴더×4crop/novel12초기validation을완료하고optimizerupdate를확인했다.15M/dropout.5는prepared직렬대기다. [실제시작검증](../artifacts/full_dataset_three_20261011/launch_verification.json)을따르며관측step은고정문서에반복복사하지않고livequeue/status를사용한다.3조건전체완료나수렴을뜻하지않는다.


## 2026-10-12 사용자 대기열 수정·현재 전체 7M 완료 후 실측 보고
15M은 사용자 지시로 보류하며 현재 baseline7m_p01 trainer PID217420은 중단·재시작 없이 유지했다. 기존 고정 3조건 제어기만 종료하고 별도 artifact 제어기로 교체했다. 순서: 현재7M Map+Render 완주 → 현재 모델 best-map/최종 선택의 5묶음 real-data FP32 보고 → dropout7m_p05 scratch 개시. 단일 GPU 순차 실행이며 dropout.5는 아직 대기다. 15M 자동 실행·재개는 하지 않는다. 과학 설정·학습 동결 소스·모든 가중치는 그대로 보존한다.
[실제 대기열](../artifacts/full_dataset_three_20261011/queue_state.json) · [변경 검증](../artifacts/full_dataset_three_20261011/queue_revision_verification_20261012.json) · [완료 후 실측 출력 위치](../artifacts/full_dataset_baseline7m_real_20261012/README.md). 학습 중단/실패·report 실패 시 attention_required이며 자동 재시작하지 않는다. 보고서는 아직 생성되지 않았다.


## 2026-10-12 정적 학습 snapshot + 공식 test 사용자 요청

현재 baseline7m_p01은 Map153000/Render39000(total192000), memory_paused이며 trainer는 종료했습니다. Controller는 attention_required이고 dropout.5는 미개시·15M은 보류입니다. 완료나 floor 수렴으로 간주하지 않으며 이번 보고 작업에서 학습을 재개하지 않았습니다.

사용자가 같은 contained static HTML에 test 정성·정량 기록을 요청하여 Validation 선정 best-map152000/best-render185000을 동결 복사한 뒤 공식test12폴더(Fabric5/Leather7)×16crop×30조명(seed10000)을 FP32로 평가했습니다. [self-contained HTML](../artifacts/full_dataset_static_test_20261012/index.html) · [평가 계약](../artifacts/full_dataset_static_test_20261012/evaluation_contract.json) · [검증](../artifacts/full_dataset_static_test_20261012/verification.json). Test NDRS합은 best-map0.11759634/best-render0.11526922, renderL1은0.003715686/0.003527352입니다. Test를 모델 선택에 사용하지 않았습니다. Test30조명과validation novel12 모집단 차이를 명시했습니다. 기존 live viewer·가중치·학습 로그·실측 report는 그대로이며 SHA 계산을 하지 않았습니다.


2026-10-12 추가 사용자 요청으로 같은 static HTML에 relation/log-relation·Kθ(9IF)·near/far·coefficient/denominator·mask/log입력의GT실제feature와예측재렌더진단, GT/render ΔE00 및map difference를추가했다. 전체공식test12×16×30의novelΔE00은best-map0.97119183/best-render0.92195620, near1.02002383/0.94237315, far0.33826438/0.31537957이다. 유효픽셀만집계하고crop별valid fraction을남겼다. 기존L1재현·CIEDE2000 tests10·집계/범위/1,032PNG/offlineJS검사를통과했다. [추가 검증](../artifacts/full_dataset_static_test_20261012/feature_delta_verification.json)을따른다. 기존training snapshot/가중치/정량L1은유지하며새학습·test기반선택·SHA계산은하지않았다.

2026-10-13 사용자 승인 실측 relation 평균 방식 비교: [기존 patch20 보고서](../artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422_patch20_20261013/index.html)에 5묶음×동결 best-map/best-render의 RGB mean(abs) 및 가중 grayscale abs(sum) 결과를 추가한다. gain은 두 방식 모두 중앙20×20 RGB 평균 비율이며 학습 gain9와 구분한다. relation뿐 아니라 최종 N/D/R/S와 물리 차이를 공통 표시 범위로 비교한다. 기본 모델·학습·checkpoint 선택은 변경하지 않는다. [실험 계약과 검증](../artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422_patch20_20261013/relation_comparison/verification.json), [계산 규약](CAPTURE_REQUIREMENTS.md)을 따른다.


## 2026-10-19 A/B/C 조건별 실측 보고 추가 승인
사용자가 각 조건 종료 후 동일 실측 5묶음 추론·별도 정성 보고서를 요청했다. artifacts/render_weight_followup_20261019/real_reports/contract.json에 따라 별도 CPU FP32 2thread watcher가 validation 선정 checkpoint와 공통 map135k를 비교한다. 조건별 runs/<arm>/real_report/index.html에 저장하며 기존 합성 report/index.html과 구분한다. 학습 runner/config/checkpoint는 변경하지 않는다. 실측 author_patch20·both·기존 정합/프레임 게이트를 유지하고 gain9를 실측에 적용하지 않는다. 학습 완료 및 메모리 여유를 기다리며 실패 시 자동 재시작하지 않는다. 실측 보고까지 모두 완료해야 heartbeat 완료 처리한다.
