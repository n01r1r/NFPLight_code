# Bottleneck FC 비교 실험

**현재성 구분:** [scheduler 정책](../../docs/TRAINING_SCHEDULER_POLICY.md)·[현재 학습 상태](../../docs/CURRENT_TRAINING_STATE.md)를 먼저 읽는다. 아래 config·고정 LR·seed7/19/31·run/extension 명령은2026-10-08 고정 진단 계약의 이력이다. 새 기본ReduceLROnPlateau 구현이나새실행승인으로취급하지않는다. 원래3-arm 계획과baseline-only완료를구분한다.

세 조건의 표시 이름은 `baseline`, `baseline+Bottleneck_FC`, `baseline+Pixel_MLP`로 고정합니다. 파일·설정의 ID는 각각 `baseline`, `bottleneck_fc`, `pixel_mlp`입니다.

`baseline`은 7,609,834개 파라미터를 가진 TwoBranchNet입니다. 두 추가 조건은 각각 7,741,932개입니다. 연산자의 정의는 다음과 같습니다.

- **Bottleneck_FC:** 두 branch 각각의 256채널, 32×32 bottleneck에 채널 간 공유 공간 연산 `Linear(1024,32) → GELU → Linear(32,1024)`를 적용합니다. Bias는 없으며 모든 공간 위치를 연결합니다. 채널과 공간을 함께 flatten한 거대 FC와는 다른 연산입니다.
- **Pixel_MLP:** 각 공간 위치에서 독립적으로 `Linear(256,128) → GELU → Linear(128,256)`을 적용합니다. Bias는 없으며 공간 위치 간 직접 연결은 없습니다.
- 두 추가 연산자는 동일한 `LayerNorm2d(256)`과 초기값 0인 학습 가능한 scalar residual gate를 사용합니다. 같은 seed에서 공통 backbone 가중치와 초기 출력이 같습니다.

이 실험은 고정된 소규모 데이터에서 학습 동역학과 오차 변화를 비교합니다. 과적합 성공만으로 FC의 필요성, 물리적 타당성 또는 일반화를 입증하지는 않습니다.

## 설정과 준비

과학적 설정의 원본은 이 폴더의 [config.json](config.json)입니다. FP32 `raw_calibrated_gain9_v3`, near/far/gain=4/12/9, F0=0.04, 고정 material·crop 선택, seed 7/19/31, map 1,000회와 render 300회, Adam·loss 및 train/novel 조명 조건을 기록합니다. 알 수 없는 필드와 고정 구조·표본 선택·조명 계약의 변경은 거부합니다.

저장소 루트에서 준비와 상태 확인:

```powershell
python -m experiments.bottleneck_fc.manage prepare
python -m experiments.bottleneck_fc.manage status
```

`prepare`는 manifest와 소스·데이터의 크기/수정 시각을 확인하고 실행용 설정 사본을 출력 폴더의 `config.json`에 저장합니다. `metadata.json`과 `status.json`은 별도 관리합니다. CUDA 초기화나 학습 실행은 하지 않습니다. 기본 출력은 `artifacts/bottleneck_fc_probe`이며 `--out`은 절대 경로와 저장소 기준 상대 경로를 받습니다.

다른 작업 폴더에서도 절대 경로로 실행할 수 있습니다.

```powershell
python C:\workspace\NFPLight_code\experiments\bottleneck_fc\manage.py prepare
python C:\workspace\NFPLight_code\experiments\bottleneck_fc\manage.py status --out C:\workspace\NFPLight_code\artifacts\bottleneck_fc_probe
```

## 실행·재개·리포트·뷰어

```powershell
python -m experiments.bottleneck_fc.manage run
python -m experiments.bottleneck_fc.manage run --resume
# baseline arm only across all three seeds
python -m experiments.bottleneck_fc.manage run --arm baseline
python -m experiments.bottleneck_fc.manage run --arm baseline --resume
python -m experiments.bottleneck_fc.manage report --run-dir artifacts/bottleneck_fc_probe
# 조건별 터미널에서 실행
python live_training_plot.py --run-dir artifacts/bottleneck_fc_probe/seed7_baseline --port 8767
python live_training_plot.py --run-dir artifacts/bottleneck_fc_probe/seed7_bottleneck_fc --port 8768
python live_training_plot.py --run-dir artifacts/bottleneck_fc_probe/seed7_pixel_mlp --port 8769
```

`run`은 설정된 학습을 실제로 시작합니다. `--arm baseline`은 과학 설정의 세 arm을 바꾸지 않고 seed 7/19/31의 baseline만 예약합니다. 이 선택은 별도 `run_selection.json`에 고정되며, 재개할 때도 같은 `--arm baseline`을 지정해야 합니다. arm 옵션이 없는 기존 명령은 전체 세 조건을 실행합니다. `--resume`은 저장 설정, 소스·데이터 정보, 실행 환경과 checkpoint가 일치해야 합니다. 저장 중 실패해도 마지막 atomic checkpoint를 기준으로 optimizer와 표본 순서를 복구하며, 그 이후 또는 불완전한 로그 행은 제거합니다.

각 조건의 `train.jsonl`, `validation.json`, `status.json`은 조건별 폴더에 있습니다. `step`은 현재 단계 내 횟수이고 `total_step`은 전체 횟수입니다. 루트 status는 현재 조건을 추적합니다. 뷰어는 읽기 전용이며 Ctrl+C로 종료합니다.

짧은 실행 검사는 seed 7, map 2회, render 1회인 **별도 설정 파일**을 `--config`로 지정하고 새 `--out`을 사용합니다. 세 조건과 나머지 계약은 유지합니다. 해당 실행의 설정 사본도 출력 폴더에 보존합니다.

리포트는 저장된 로그·평가·최종 요약·FP32 prediction 배열만 읽습니다. Normal 차이는 각도(도), roughness는 prediction−target의 부호 있는 차이, near re-render 차이는 CIEDE2000(ΔE00)으로 표시합니다. 모든 지정 seed와 조건이 완료되어야 paired effect를 보고하며 material bootstrap의 한계도 표시합니다. `map_difference`는 수치 배열을 직접 반환합니다.

## 과거 실행 보존

이전 ID `none`, `spatial_fc`, `local_mlp`는 과거 구현에만 해당합니다. 소스 사본은 `artifacts/bottleneck_fc_overfit_20261008/legacy_code/`에 보존합니다. 기존 수치·로그·checkpoint·status는 변경하지 않으며 새 코드가 자동 변환하지 않습니다.

## Baseline 연장 컨트롤러

세 baseline seed의 기존 1,000 map + 300 render 실행과 map 경계 checkpoint archive가 모두 완료된 뒤 조건별 plateau를 판정하고, 미수렴 seed에만 연장을 적용합니다. 기존 실행 상태와 checkpoint는 읽기 전용으로 확인하며, 출력은 `artifacts/bottleneck_fc_baseline_extended_20261008`에 분리합니다. Controller 실행은 foreground에서 대기하고 원본 run이 완료되기 전에는 extension 학습을 시작하지 않습니다.

저장소 루트 또는 다른 작업 폴더에서 다음 명령을 실행합니다.

```powershell
python -m experiments.bottleneck_fc.extension_controller
python C:\workspace\NFPLight_code\experiments\bottleneck_fc\extension_controller.py --poll-seconds 10
```

완료 여부, 원본 process 종료, archive 상태, 설정·소스·데이터·runtime identity를 확인한 뒤 seed 7/19/31만 순차 처리합니다. `extension_policy.json`은 baseline 외 arm을 예약하지 않으며, 전체 3-arm 비교 상태를 `partial_baseline_extension`으로 남깁니다. `launch.json`은 부모가 기록하는 controller 프로세스 정보이며 실행 조건으로 해석하지 않습니다.

Plateau는 각 단계의 heldout `ndr_mean`(map) 또는 heldout `render_l1`(render)에서 `(previous-current)/max(abs(previous),1e-12) < 0.005`가 10개 validation interval 연속일 때만 인정합니다. Map 미수렴 seed는 보존된 map 1,000 checkpoint와 Adam 상태에서 이어서 누적 10,000회까지 학습하고, plateau 또는 cap에 도달하면 실제 map 종료 step을 경계로 새 Adam render 단계로 넘어갑니다. Map이 이미 수렴한 seed는 pilot render 300의 model·Adam 상태를 이어가며 render 누적 3,000회까지 진행합니다. 단계별 조기 종료는 현재 checkpoint를 보존하며 best checkpoint 복원이나 learning-rate 변경은 하지 않습니다.

같은 controller 명령을 다시 실행하면 extension의 immutable manifest와 checkpoint identity를 확인한 뒤 마지막 atomic checkpoint, optimizer, RNG 상태 및 고정 sample-order prefix에서 재개합니다. `status.json`과 조건별 `status.json`, `train.jsonl`, `validation.json`은 viewer와 호환됩니다. 실패·중단은 상태에 기록되고 원본 실행 파일과 숫자 결과는 수정하지 않습니다. 연장 완료만으로 수렴의 일반화나 세 arm 비교 완료를 뜻하지 않습니다.