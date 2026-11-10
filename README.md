# NFPLight

| Colab | 모델 |
|---|---|
| [NFPLight](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/nfplight.ipynb) | 원 additive U-Net, **64,102,378 parameters** |
| [U-Net++](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/unetpp.ipynb) | NAFBlock U-Net++, DS OFF |

Setup → Drive → local data → Train → Live view. Drive에서 train/val 데이터를 세션 로컬 저장소로 먼저 복사하고 학습·검증은 로컬 파일을 읽는다. 각 notebook은 모델 하나로 고정한다. 병렬 실험은 별도 GPU runtime에서 실행한다. FP32·무양자화·gain9·seed19·micro2×accum2(유효 batch4)이며 fresh는 pretrained weight를 읽지 않는다.

Validation은 batch16, CUDA OOM이면 validation만 절반으로 줄여 재시도한다. Crop별 loss와 기존 재질 평균·FP32를 유지하며 train 설정은 변경하지 않는다.

`DRIVE_FOLDER`와 새 `RUN_ROOT`를 지정하고 Train 셀을 실행한다. Train step과 **1×2 train/validation loss(log Y축)**를 표시한다. Diagram·모델 소스 출력·HTML·iframe은 없다.

각 checkpoint 저장 후 `MyDrive/NFPLight_runs/<run 이름>/`에 **last.pt(전체 재개 상태)**와 **best.pt(선정 모델 weight)**를 보존한다. 기존 backup 폴더는 덮어쓰지 않는다.

[Colab details](docs/COLAB.md) · [DNG / inference](docs/USAGE_KO.md) · [Numerical contract](docs/CLEAN_CORE_REWRITE.md)

데이터·연구 weight·checkpoint는 배포하지 않는다. Baseline은 무양자화 원 NFPLight를 MatSynth fabric/leather subset으로 학습한 weight만 뜻한다. 저자 원본 inference weight와 과거 양자화 실험을 baseline으로 취급하지 않는다.
