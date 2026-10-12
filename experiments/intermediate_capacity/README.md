# 15.09M 중간 용량 고정 진단

독립 세션은 [scheduler 정책](../../docs/TRAINING_SCHEDULER_POLICY.md)·[현재 상태](../../docs/CURRENT_TRAINING_STATE.md)를 먼저 읽는다. 사용자 승인 모델은 width32·encoder[2,2,4]·middle12·decoder[2,2,2],15,085,034parameters, seed19·D4 only·dropout0이다. 원래64.1M으로 복귀한 실험이 아니다.

[결과 root](../../artifacts/intermediate_capacity_15m_seed19_20261009/artifact_context.json)의고정 [plan](../../artifacts/intermediate_capacity_15m_seed19_20261009/plan.json)을 유지하고 map10k/render3k를 완료했다. FP32·AMP/TF32 off·batch4·같은 데이터·near4/far12/gain9·F0=.04·40train/12novel lights·constant map5e-4/render1e-4·기존early stop·stage별freshAdam을 사용했다. Scheduler 없는 **완료된 historical screen**이며 새 기본값 구현이 아니다.

[complete](../../artifacts/intermediate_capacity_15m_seed19_20261009/complete.json)·[comparison](../../artifacts/intermediate_capacity_15m_seed19_20261009/comparison.json)·[export recovery](../../artifacts/intermediate_capacity_15m_seed19_20261009/export_recovery.json)를 함께 확인한다. 실제 학습은3000render까지 끝났으며 arrays 디렉터리 누락으로 export가 실패했다. 보존 checkpoint에서 추가 학습 없이32FP32배열을 export했고 평가 차이0을 확인했다. `failure.json`은 복구 전 사건 기록이며 현재 실패 상태가 아니다.

첫 자동 controller는 seed31 가드 실패로 종료했다. [제외 incident](../../artifacts/regularization_probe_20261009_checked/seed31_exclusion_incident.json)의224updates를 비교에서 제외하고 수동 검증한 전환으로 학습했다. `wait-run`의 원래seed31_rows0 gate가 통과했다고 주장하지 않는다. 옛launch/transition 기록은 현재 실행 승인이나 대기 상태가 아니다.

Primary control은 같은seed19의7.61M D4-only다. 7.61M D4+dropout과 비교할 때 dropout 차이를 capacity효과로 섞지 않는다. 단일seed·8개validation진단만으로 일반화나 scheduler효과를 입증하지 않는다.

읽기 전용 상태 확인:

```powershell
python experiments/intermediate_capacity/run.py status
```

이 runner의 `prepare/run/wait-run`은기존고정계약용이다. 새 기본ReduceLROnPlateau 계약으로 사용하지 않는다. [후속 Council 제안](../../artifacts/scheduler_council_20261009/COUNCIL_VERDICT.md)을 확인하고미합의된세부안을확정사실로옮기지않는다. 이전README는 [doctor 보존 사본](../../artifacts/policy_doctor_20261009/before_documents/experiments/intermediate_capacity/README.md)에 있다.
