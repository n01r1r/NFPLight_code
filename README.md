# NFPLight

선형 near/far RGB로 normal(N), diffuse(D), roughness(R), specular(S)를 추정한다. 원 NFPLight와 NAFBlock U-Net++의 공통 CUDA FP32 학습 코어, 명시적 profile, DNG 전처리와 추론을 제공한다.

[Colab에서 학습](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/train.ipynb) · [Colab·Drive·데이터·비교 조건](docs/COLAB.md) · [DNG·추론](docs/USAGE_KO.md) · [수치 계약](docs/CLEAN_CORE_REWRITE.md)

```bash
git clone --depth 1 --branch codex/colab-experiments https://github.com/n01r1r/NFPLight_code.git
cd NFPLight_code
python -m pip install -r requirements-colab.txt
```

Notebook을 위에서부터 실행하고 `MODEL='nfplight'` 또는 `'naf_unetpp'`를 선택한다. `DRIVE_DATA_FOLDER`에 Drive 데이터 폴더를 지정하고 mount한 뒤 `RUN_TRAINING=True`로 시작한다. 학습 코드는 프로젝트 상대 경로로 데이터를 읽는다. 각 모델을 별도 Colab GPU 런타임에서 같은 commit·데이터·조건으로 비교할 수 있다. tqdm·선택 matplotlib plot을 셀에서 갱신한다. HTML plot·iframe·별도 서버는 사용하지 않는다.

기본 모델은 암묵적으로 선택하지 않는다. 학습은 무양자화 관측·FP32·gain9·seed19·micro1×accum4다. Baseline은 원 NFPLight를 MatSynth fabric/leather subset으로 학습한 weight만 뜻한다. Fresh는 pretrained 없이 시작하며 baseline 결과 weight를 동봉하지 않는다. 데이터·보고서·resume checkpoint는 Git에 포함하지 않는다.

Local CLI도 명시적 profile을 사용한다. 아래 실행 전 데이터 셀로 portable manifest를 생성한다. 출력 폴더는 새 이름이어야 한다.

```bash
python train_colab.py --config configs/nfplight_32_noquant.json --manifest work/train_val_manifest.jsonl --run-root runs/nfplight_fresh_01
python train_colab.py --config configs/nested_32_noquant.json --manifest work/train_val_manifest.jsonl --run-root runs/naf_unetpp_fresh_01
```

학습 의존성은 `requirements-colab.txt`, DNG까지 사용할 때는 환경에 맞는 PyTorch 설치 후 `requirements.txt`를 사용한다. Notebook 검사에는 `pytest nbformat nbclient ipykernel`이 추가로 필요하다.

```bash
python -m pytest tests/test_colab.py -q
```

저자 원본: [CGLiWang/NFPLight_code](https://github.com/CGLiWang/NFPLight_code). 저자 real/syn inference weight는 원본에서 별도로 확보하고 `configs/author_real.json`·`author_syn.json`을 명시한다. Denoiser는 사용하지 않는다.
