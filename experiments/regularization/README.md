# 7.61M D4·dropout 고정 진단

독립 세션은 [scheduler 정책](../../docs/TRAINING_SCHEDULER_POLICY.md)·[현재 상태](../../docs/CURRENT_TRAINING_STATE.md)를 먼저 읽는다. 이 패키지의 config와 runner는 **완료된 고정 LR 진단의 구현**이며 새 ReduceLROnPlateau 기본값 구현이 아니다.

실제 출력은 [regularization_probe_20261009_checked](../../artifacts/regularization_probe_20261009_checked/artifact_context.json)다. Seed7/19×D4 only/dropout only/D4+dropout의6개 새 run을 완료했다. 대조는 각 seed의기존 baseline extension이다. 원래 immutable config의9개 run·3개 seed를 선택 완료 범위와 혼동하지 않는다.

[두 seed 결과](../../artifacts/regularization_probe_20261009_checked/TWO_SEED_CONCLUSION.md), [선택 완료 범위](../../artifacts/regularization_probe_20261009_checked/user_scope_completion.json), [seed31 incident](../../artifacts/regularization_probe_20261009_checked/seed31_exclusion_incident.json)를 함께 확인한다. 가드의 PowerShell 시각 비교 오류로seed31이224updates 실행됐으며 중단·비교 제외했다. 자동 가드가 성공했고0update였다고 서술하지 않는다.

[고정 진단 규약](../../docs/REGULARIZATION_PROBE.md), [execution config](../../artifacts/regularization_probe_20261009_checked/execution_config.json), 조건별complete·validation·train 로그가 당시 실행 근거다. 기존config는 seed7/19/31·constant map5e-4/render1e-4·cap10k/3k·adjacent-evaluation early stop을 보존한다. 이 파일을 새 후속 default로 복사하지 않는다.

기본 `artifacts/regularization_probe_20261008` 준비본은 source identity가 오래됐으므로 실행에 재사용하지 않는다. `manage report`의 원래 full3-seed effect는 선택2-seed 결론을 대체하지 않는다.

읽기 전용 상태 확인:

```powershell
python -m experiments.regularization.manage status --out artifacts/regularization_probe_20261009_checked
```

15.09M D4-only·단일seed19는 [별도 screen](../intermediate_capacity/README.md)에서 완료했다. Scheduler 후속의fresh 재학습·monitor·best 복원·종료 규칙은 [Council 제안](../../artifacts/scheduler_council_20261009/COUNCIL_VERDICT.md)이며 세부조건을 승인 사실로 사용하지 않는다. 이전README는 [doctor 보존 사본](../../artifacts/policy_doctor_20261009/before_documents/experiments/regularization/README.md)에 있다.
