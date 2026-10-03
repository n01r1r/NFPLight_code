# NFPLight

*NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting* (SIGGRAPH Asia 2024)의 fork다.
저자 checkpoint와 호환되는 estimator 및 Fabric 학습 코드를 유지한다.

## 현재 Fabric 리포트

[생성 규약과 상세 데이터 흐름](docs/CAPTURE_REQUIREMENTS.md)을 기준으로 실행한다.
문서는 한국어로 작성했으며 ISO 24495-1:2023의 공개된 간결한 언어 원칙을 적용했다.
입력 처리, 공통 정합, 평균 조건, 광도·색 변환, 33채널 특징, 추론과 표시를 단계별로 설명한다.

현재 경로는 5개 DNG 묶음을 독립 처리한다. 입력 처리와 추론은 FP32이며 원본 `net_g_real.pth`를 사용한다.
광도 조건은 `both=(D-B)/(W-B)` 하나다. denoiser는 사용하지 않는다.
양쪽 5장 모두 수치 검사와 무늬 정합 검토를 통과하면 side별로 평균한다.
실패하면 같은 묶음의 near_00/far_00을 사용하고 이유를 기록한다.

새 실행에는 새 artifact 디렉터리를 지정한다. 아래 경로는 예시다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepare-only --precision fp32 --out artifacts/fabric_capture_20261003_run01
```

각 묶음의 `prepare_review.html`을 확인하고 실제 판단을 `root_review.json`에 기록한다.
형식과 승인 조건은 [실행 절차](docs/CAPTURE_REQUIREMENTS.md#12-실행-절차)를 따른다.
그 다음 같은 prepared 경로에서 추론한다.

```powershell
python run_fabric_capture.py --source data/260930_152732_008 --all-groups --prepared artifacts/fabric_capture_20261003_run01 --precision fp32
```

묶음별 `report.html`과 전체 `index.html`을 확인한다.
별도 `paper_comparison`과 `--paper-equations` 옵션은 폐지했다.
기존 best_render 수치 결과는 보존하지만 현재 비교에서 제외한다.
표시 gamma는 기본 off이며 입력·예측 배열을 바꾸지 않는다.
원본 DNG 폴더에 입력 PNG를 다시 만들지 않는다.

## 설치와 검증

Python 3.10/3.11과 장치에 맞는 PyTorch를 설치한 뒤 의존성을 설치한다.
DNG 처리는 rawpy가 필요하다. 모델의 CPU 실행은 `--device cpu`로 지정한다.

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q -p no:cacheprovider
```

## Fabric 학습

학습 entry point는 `train_fabric.py`다. 데이터 준비는 `python -m data.fabric --help`로 확인한다.
학습에서 만든 custom weight는 현재 DNG 리포트의 허용 추론 weight가 아니다.

```powershell
python train_fabric.py --manifest data/fabric_native256_v1/manifest.json --out checkpoints/fabric_scratch
```

이전 source 계약을 기록한 checkpoint의 exact optimizer resume는 거부한다.
정리 과정에서 migration whitelist를 늘리지 않았다.
학습과 재실행에는 해당 dataset과 로컬 checkpoint가 필요하다.

## 문서와 자료 보존

현재 규약은 [CAPTURE_REQUIREMENTS.md](docs/CAPTURE_REQUIREMENTS.md) 한 곳에서 관리한다.
[과거 문서와 정리 증거](docs/archive/README.md)는 당시 판단과 실행 이력을 보존한다.
과거 문서를 현재 실행 지시로 사용하지 않는다.

capture data, checkpoint, cache, artifact는 로컬 자료다.
`.git`, `checkpoints`, `results`, dataset junction의 외부 대상을 재귀 삭제하거나 이동하지 않는다.

## 인용

```bibtex
@article{10.1145/3687978,
  author = {Wang, Li and Zhang, Lianghao and Gao, Fangzhou and Kang, Yuzhen and Zhang, Jiawan},
  title = {NFPLight: Deep SVBRDF Estimation via the Combination of Near and Far Field Point Lighting},
  journal = {ACM Transactions on Graphics},
  volume = {43}, number = {6}, articleno = {274}, year = {2024},
  doi = {10.1145/3687978}
}
```
