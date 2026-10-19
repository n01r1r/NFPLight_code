# Render weight/LR 순차 실험

2026-10-19 사용자 승인: A(weight1/LR1e-4) → B(weight1/LR1e-5) → C(weight10/LR1e-5), 같은135k best-map·fresh Adam·공통 RNG에서 각10k update. 각 조건 학습 뒤 정적 학습+test 정성/정량·feature/ΔE00 보고서를 완성한 후 다음 조건으로 진행한다.

구현: `python -m experiments.render_weight_followup.run prepare`, CPU 검사 후 `smoke`, 검증된 준비 상태에서 `queue`. 새 실행 경로는 artifacts/render_weight_followup_20261019이다. 이미 실행 중인 queue를 복제하거나 정지/실패 run을 자동 재개하지 않는다. 상태는 queue.json, 각 run/status.json, report/verification.json을 확인한다. [계약](../../artifacts/render_weight_followup_20261019/EXPERIMENT_CONTRACT_KO.md).

기존 full_dataset_suite 코드와 relmap run은 유지한다. Weight는 별도 WeightedPipeline의 실제 objective에 반영한다. 10k 종료는 screening_complete이며 수렴 완료가 아니다. Test는 이전에 관측한 보고용 자료이고 선정 기준은 validation만 사용한다. 실행 실패·memory/disk pause·보고서 실패는 queue 중단 및 검토 대상이다.

검사: `python -m pytest tests/test_render_weight_followup.py tests/test_full_dataset_readiness.py -q` (21 passed), FP32 GPU weight10 render2 및 checkpoint 전체복원, 현재 공간보정 feature exact 검사 통과. GPU smoke는 구현 검증이며 성능 증거가 아니다.
