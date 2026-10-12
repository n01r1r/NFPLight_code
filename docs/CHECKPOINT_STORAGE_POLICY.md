# 체크포인트 저장 공간 운영 정책

현재 승인 suite의 trainer/source/input은 동결 상태다. 저장 개수는 update에 비례해 늘지 않으며 `last.pt`와 `previous.pt`를 순환하고 best·selected를 덮어쓴다. Map/Render 종료 상태는 각각 하나를 보존한다. 완료 arm은 7개 `.pt`와 FP32 export 1개, 최대 8개 정상 파일을 유지한다. 진행 중에는 아직 생성되지 않은 파일이 있을 수 있다. 복원·선택·단계 종료 증거는 자동 삭제하지 않는다.

10분 heartbeat마다 다음 읽기 전용 검사를 실행한다.

`python -m artifacts.robustness_suite_seed19_20261009.checkpoint_storage_guard`

[현재 inventory](../artifacts/robustness_suite_seed19_20261009/checkpoint_storage_state.json)는 고정 위치에 덮어써 감시 기록 자체가 누적되지 않게 한다. arm별 이름·개수·bytes, 디스크 여유, `.pt.tmp`/`.pth.tmp` 크기·mtime을 검사한다. 임시 파일은 정상 개수에서 제외한다. 두 예약 점검에서 임시파일 크기·mtime이 같으면 경고하며, step·last 갱신 기록과 함께 해석한다. 짧은 저장 중 존재할 수 있으며 임시파일만으로 학습 정지를 단정하지 않는다.

계획용 W=max(parameters×4, 관측 선택 가중치 최대bytes), F=max(10W, 관측 full checkpoint 최대bytes)다. arm별 envelope=4F+4W+64MiB, 미완료 arm별 max(0,envelope−현재 정상checkpoint bytes)를 합산한다. 이는 EMA와 optimizer·선택 상태를 고려한 보수 운영 추정이며 확정 상한이나 안전 보장이 아니다. 일시 저장 공간은 별도 여유로 둔다.

- 경고: 여유 < max(15GiB, 남은 checkpoint 추정 + 5GiB, 긴급 기준).
- 긴급: 여유 < max(5GiB, 최대 F×2).
- 예상 밖 checkpoint 이름, 8개 초과, envelope 초과, 여러 임시 checkpoint도 경고한다.
- 매 점검 디스크 여유와 storage level을 사용자에게 보고한다. 부족하면 원인·재생성 가능한 공간 정리안·안전한 중단/재개안을 구체화한다. 자동 checkpoint 삭제·trainer 중단·재시작은 하지 않는다.

Astra low의 읽기 검토를 반영했다. 현재 controller는 GPU launch gate만 있고 디스크 gate는 없으며 engine은 디스크 부족을 사전 일시정지하지 않는다. 지금은 외부 감시만 추가했다. 다음 새 lineage를 구현할 때 controller launch 전과 atomic_save 전에 disk reserve gate를 넣고, 기존 committed checkpoint 보존·저장 실패 후 재저장 루프 방지·복원 경로를 검사한다. 현재 suite의 동결 소스를 변경하지 않는다.

2026-10-10 초기 검증: threshold 경계 계산, 실제 7-arm inventory, 격리 fixture에서 임시파일 unchanged 경고와 정상 개수 제외 검사를 통과했다. checkpoint 로드·SHA-256·GPU 진단은 사용하지 않았다.

과거 capture source_frames 중 바이트가 같은 100개는 NTFS 하드링크로 통합했다. [정리 기록](../artifacts/disk_cleanup_20261010/verification.json)을 보존하며 해당 과거 배열을 직접 덮어쓰지 않는다.


2026-10-10 사용자 명시 승인으로 완료7M robustness6조건의 full optimizer/재개checkpoint·중복best·준비/smoke를 영구 정리했다. [보존·삭제 기록](../artifacts/cleanup_7m_variants_20261010/REPORT_KO.md)을 따른다. selected·고유best·단계 종료 FP32 모델·최종 수치/로그는 보존하며 이6조건의 exact resume는 불가하다. 15M 동결 의존성과 활성 실행은 유지했다. 이전 checkpoint inventory는 정리 전 이력이며 현존 여부는 cleanup manifest/alias를 따른다.
