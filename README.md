# NFPLight

| Colab | 모델 |
|---|---|
| [NFPLight](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/nfplight.ipynb) | 원 additive U-Net, **64,102,378 parameters** |
| [U-Net++](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/unetpp.ipynb) | NAFBlock U-Net++, DS OFF |

Setup → Drive data → Train → Live view. 각 notebook은 모델 하나로 고정한다. 병렬 실험은 별도 GPU runtime에서 실행한다. FP32·무양자화·gain9·seed19·micro1×accum4이며 fresh는 pretrained weight를 읽지 않는다.

`DRIVE_FOLDER`와 새 `RUN_ROOT`를 지정하고 Train 셀을 실행한다. Train step과 **1×2 train/validation loss(log Y축)**를 표시한다. Diagram·모델 소스 출력·HTML·iframe은 없다.

[Colab details](docs/COLAB.md) · [DNG / inference](docs/USAGE_KO.md) · [Numerical contract](docs/CLEAN_CORE_REWRITE.md)

데이터·연구 weight·checkpoint는 배포하지 않는다. Baseline은 무양자화 원 NFPLight를 MatSynth fabric/leather subset으로 학습한 weight만 뜻한다. 저자 원본 inference weight와 과거 양자화 실험을 baseline으로 취급하지 않는다.
