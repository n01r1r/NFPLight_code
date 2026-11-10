# Colab 학습

[Notebook 열기](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/train.ipynb). Notebook은 GitHub 코드를 clone하고 공통 모듈을 직접 사용한다. 섹션 제목과 실행 셀만 제공한다.

1. GPU 런타임에서 셀을 위에서부터 실행한다. 첫 셀이 `NFPLight_code`를 clone하고 그 폴더로 이동한다. 기존 clone은 자동 업데이트하거나 덮어쓰지 않는다.
2. Controls에서 `DRIVE_DATA_FOLDER`를 Drive mount 기준 상대 경로로 지정한다. 기본은 `MyDrive/MatSynth_PRIMA_Fabric`이다. 그 폴더에 `manifests/fabric_leather_training_manifest.jsonl`과 `source_shards/`가 있어야 한다.
3. Drive 셀에서 Google 인증을 진행한다. `/content/drive`를 mount하고 `data/MatSynth_PRIMA_Fabric` 상대 경로를 해당 Drive 폴더로 연결한다. 원본 파일을 복사·수정·삭제하지 않는다. 링크 대상이 다르면 덮어쓰지 않고 거부한다. Shared Drive라면 mount 아래 실제 상대 폴더를 지정한다.
4. `MODEL = 'nfplight'` 또는 `'naf_unetpp'`를 선택하고 새 `RUN_NAME`을 사용한다. 설정·구조·입력·loss와 pipeline plot을 확인한다. `RUN_TRAINING = True`로 설정한 뒤 학습 셀을 실행한다.
5. `SHOW_PLOTS = False`에서도 tqdm과 JSON snapshot 저장은 동작한다. Plot은 matplotlib와 IPython clear_output/display로 셀 안에서 갱신한다. HTML·iframe·웹 서버는 사용하지 않는다. Plot을 켜면 PNG snapshot도 저장한다. `RUN_PIPELINE_PREVIEW`는 별도 CUDA forward 검사이며 학습 결과 평가가 아니다.
6. 병렬 비교는 서로 다른 GPU 런타임에서 같은 notebook·commit·Drive 데이터 폴더를 사용하고 MODEL만 바꾼다. 동시 GPU 할당·수명은 Colab 계정과 가용 자원에 따라 달라진다. 같은 GPU에서 두 trainer를 실행하는 스케줄러는 제공하지 않는다.

| 선택 | 구조 | 공통 구조 설정 |
|---|---|---|
| `nfplight` | 원 NFPLight dual NAF encoder, additive U-Net decoder | dim32, encoder [2,2,4,8], middle12, decoder [2,2,2,2] |
| `naf_unetpp` | dual NAF encoder, dense nested U-Net++ decoder, DS OFF | 같은 dim/encoder/middle, nested node당 NAFBlock2 |

두 profile은 구조와 이름만 다르다. 21ch, near4/far12, gain9, unit XYZ, 무양자화 관측·FP32, AMP/TF32 OFF, seed19, micro1×accum4, 같은 loss·Adam·scheduler·데이터·평가 조건을 사용한다. 같은 seed는 구조 간 모든 초기 weight가 같다는 뜻이 아니다. 구조별 메모리·처리량이 다르며 이 배포는 성능 우열을 주장하지 않는다.

Baseline은 **무양자화 원 NFPLight를 MatSynth fabric/leather subset으로 학습한 weight**다. 저장소는 학습 profile을 제공하며 baseline weight를 생성하거나 동봉하지 않는다. 저자 원본 real/syn weight와 과거 linear16 weight를 이 baseline으로 취급하지 않는다. Fresh는 pretrained 없이 초기화한다. 다른 구조는 `nfplight/checkpoint.py`의 `build_model`과 명시적 profile에 추가하고 공통 trainer를 재사용한다.

데이터는 train566(Fabric358/Leather208), val164(Fabric98/Leather66), 합계730 NPZ·6,284,810,295 bytes다. 16×256² crop/material이며 train9056 slots, val crop0/4/8/12로656 crop/evaluation이다. 전체 원 폴더는6,345,669,728 bytes다. 12개 official test는 portable manifest에서 제외하며 payload를 읽지 않는다. 저장 N/D/R/S map의 uint8과 합성 관측의 무양자화는 다른 계약이다. 원 manifest를 수정하지 않고 Windows shard 경로를 상대 경로로 재작성한다. 같은 portable manifest의 재실행은 mtime을 유지한다. 기대 개수·중복·그룹 누출·누락·경로 이탈은 거부한다.

Drive에서 NPZ를 직접 읽으며 기존 MaterialBank cache를 사용한다. GPU별 속도와 Drive I/O 병목은 여기서 측정하지 않았다. 병목이 관측되면 모델 수치를 바꾸지 않는 로컬 staging 등을 별도로 검토한다. 자동 staging은 하지 않는다.

Map→render 정책은 그대로다. Map400k/render100k cap은 pause이며 강제 전환·수렴 판정이 아니다. LR floor의 의미 있는 개선 없음20회를 기존 trainer가 판정한다. tqdm은 총량 없는 optimizer update 계수다. Plot은 단계별 raw/100-update 평균, validation overall/Fabric/Leather, LR·gradient·소요시간·floor 상태를 표시한다. Train은 유효 픽셀 가중 minibatch, validation은 crop 평균 후 재질 동일 가중 평균이다. 8bit 표시로 artifact/degradation을 판정하지 않는다.

`runs/<RUN_NAME>/launch.json`에 commit·dirty 상태·설정 identity를 기록한다. Full `run/last.pt`만 exact resume에 사용한다. 모델/입력/decoder/Adam/scheduler/전체 RNG·Python/PyTorch/CUDA/GPU·데이터 절대 경로/size/mtime 검사를 완화하지 않는다. Colab reset 후 재업로드·다른 GPU·Windows→Linux에서는 exact resume이 거부될 수 있다. `RESUME`도 새 RUN_NAME에 기록하며 자동 재개하지 않는다. 마지막 저장 checkpoint 이후 미저장 update는 복구하지 않는다. 출력/checkpoint는 런타임 로컬이며 종료 전 사용자가 보존해야 한다. 데이터 mount는 결과의 Drive 백업을 의미하지 않는다.

CUDA PyTorch는 Colab 기본 설치를 유지하고 학습 의존성만 `requirements-colab.txt`로 설치한다. DNG용 rawpy/OpenCV는 학습에 필요하지 않다. 이미 import한 패키지 버전이 설치 중 바뀌면 런타임 재시작을 요구한다. GPU free12GiB/host available8GiB/disk free12GiB를 시작 조건으로 검사한다. 하한 admission이며 모든 GPU의 성공을 보장하지 않는다. 모델은 CUDA에서 초기화하고 FP32로 계산한다.

Clone 방식은 모델·trainer 복제본을 줄이고 commit으로 코드를 고정할 수 있다. 단일 self-contained ipynb는 저장소 없이 전달할 수 있다는 장점이 있지만 코드·설정이 분기될 수 있다. 새 설계는 하나의 코어와 명시적 profile, runtime별 trainer 하나를 사용한다. Notebook의 코드 출력은 실제 클래스를 inspect로 읽는다. 별도 모델 registry framework나 원격 job 관리기를 추가하지 않는다.

설계 흐름 참조: [HoViT](https://github.com/hun9008/HoViT). 환경→설정→데이터→모델→학습→plot 흐름을 참고했다. Colab CLI 설치·인증 자동화는 포함하지 않는다. Drive mount는 [공식 API](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)를 사용하며 [Colab 자원 제한](https://research.google.com/colaboratory/faq.html)을 따른다.
