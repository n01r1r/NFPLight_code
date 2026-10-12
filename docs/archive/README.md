# 과거 문서와 정리 증거

Capture의 현재 실행 기준은 [Fabric 리포트 생성 규약](../CAPTURE_REQUIREMENTS.md), 학습은 [scheduler 정책](../TRAINING_SCHEDULER_POLICY.md)·[현재 상태](../CURRENT_TRAINING_STATE.md)다.
아래 문서는 작성 당시의 기록이다. 폐기된 조건, 이전 weight, 이전 표시 규칙이 포함될 수 있다.
문서 안의 옛 경로와 실행 명령은 당시 상태를 설명하며 현재 실행 지시가 아니다.

2026-10-03 정리에서는 중복 규약을 현재 문서로 통합하고 기존 문서 12개를 이 디렉터리로 이동했다.
이동 전후 SHA256이 같음을 확인했다. 원본 수치 artifact와 checkpoint는 변경하지 않았다.
이동 대응표와 hash는 [document_cleanup_20261003.json](document_cleanup_20261003.json)에 있다.

## 기록 문서

- [BOTH_ONLY_CAPTURE_CONTRACT_20261001.md](BOTH_ONLY_CAPTURE_CONTRACT_20261001.md)
- [CLEANUP.md](CLEANUP.md)
- [DNG_AUTHOR_REAL_PIPELINE_20261001.md](DNG_AUTHOR_REAL_PIPELINE_20261001.md)
- [DNG_PIPELINE_AUDIT_20261001.md](DNG_PIPELINE_AUDIT_20261001.md)
- [NFPLight_fabric_scratch_plan.md](NFPLight_fabric_scratch_plan.md)
- [NFPLight_matsynth_scratch_plan.md](NFPLight_matsynth_scratch_plan.md)
- [NFPLight_redteam.md](NFPLight_redteam.md)
- [PAPER_EQUATION_COMPARISON_20261002.md](PAPER_EQUATION_COMPARISON_20261002.md)
- [PSEUDO_FAR_INVERSE_SQUARE_20261002.md](PSEUDO_FAR_INVERSE_SQUARE_20261002.md)
- [REPORT_NOTATION_DISPLAY_20261002.md](REPORT_NOTATION_DISPLAY_20261002.md)

## JSON 증거

- [capture_10cm30cm_burst_checks.json](capture_10cm30cm_burst_checks.json)
- [capture_10cm30cm_inventory.json](capture_10cm30cm_inventory.json)
- [cleanup_removed.json](cleanup_removed.json)
- [document_cleanup_20261003.json](document_cleanup_20261003.json)
- [precision_reset.json](precision_reset.json)
- [preservation_verification.json](preservation_verification.json)

2026-10-09 doctor는이동뒤깨진navigationlink를보정하고실제없는역사산출물은없음으로표시했다. 위2026-10-03 hash확인은당시이동이력이며이번수정뒤byte동일성주장이아니다. 이번작업은SHA-256을실행하지않았다.

## 완료된 refinement 코드 최소화

[원본 ZIP](completed_refinement_sources.zip)에 root 스크립트 23개, 전용 모듈 3개, 테스트 9개와 축소 전 physical-map 모듈·템플릿·README를 보존했다. 압축 후 원본 byte와 직접 비교하고 Python 구문을 확인한 뒤 활성 파일 35개를 제거했다. `rendering_loss_analysis.py`는 현재 report가 쓰는 FP32 검증·physical-map decode 두 함수만 유지한다. 저장된 수치·checkpoint와 학습 source snapshot은 변경하지 않는다. 재현이 필요하면 ZIP에서 필요한 원래 상대 경로와 의존 파일을 저장소에 복구한다. 과거 실행 계약을 새 실행 승인으로 간주하지 않는다.

정리 후 관련 CPU 테스트 51개, 현재 tests/ 수집, doctor(문서62·링크426·artifact22·보존파일677)와 diff 형식 검사를 확인했다. `pytest.ini`는 과거 artifact 테스트의 기본 수집을 제외한다. GPU 학습과 SHA-256 검사는 실행하지 않았다.
