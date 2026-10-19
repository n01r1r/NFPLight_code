# 전체 데이터 재학습 준비

2026-10-16 거리비율 보정 기준은 `spatial_inverse_square_v1`이다. 현재 준비한 조건은 `baseline7m_p01` (7.61M, dropout 0.1, seed19)이며 FP32·gain9·near4/far12와 기존 scheduler 설정을 유지한다. 15M은 보류 상태다.

아래는 2026-10-16 준비 당시 기록이다. 이후 7M 재학습·A–C 완료 상태는 [현재 상태](../../docs/CURRENT_TRAINING_STATE.md), 전달할 weight는 [전달 점검](../../docs/WEIGHT_HANDOFF_20261019.md)을 따른다. 준비 검증은 [준비 보고](../../artifacts/spatial_retraining_20261016/README.md)에 보존한다.

저장소 루트에서 준비 설정을 재확인한다.

```powershell
python -m experiments.full_dataset_suite.run prepare --arm baseline7m_p01 --out artifacts/spatial_retraining_20261016/runs/baseline7m_p01
```

향후 본 학습 실행 명령은 다음과 같다. 이 문서 작성 과정에서는 실행하지 않았다.

```powershell
python -m experiments.full_dataset_suite.run run --arm baseline7m_p01 --out artifacts/spatial_retraining_20261016/runs/baseline7m_p01
```

검증에 사용한 smoke는 별도 `smoke_verified/baseline7m_p01` 폴더에 저장했다. smoke checkpoint는 본 학습에 사용할 수 없다. 과거 relation 버전 checkpoint의 학습 resume도 거부한다. 새 학습은 새 출력 폴더에서 시작한다.

준비 후 코드·입력 파일의 크기 또는 수정시각이 바뀌면 immutable 계약 검사가 거부한다. 기존 증거를 보존하고 새 폴더에서 prepare와 smoke를 다시 수행한다. 전체 shard는 이름·shape·dtype 헤더를 확인하고 실제 pixel 로딩은 제한된 smoke에서 검증한다.

과거 3조건 controller는 이번 baseline 준비에 사용하지 않는다. controller의 `--root`는 명시해야 하며 이 문서는 과거 대기열의 재실행 승인이 아니다.
