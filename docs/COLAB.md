# Colab

[NFPLight 64M original](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/nfplight.ipynb) · [NAF U-Net++ DS OFF](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/unetpp.ipynb)

1. 새 GPU 런타임에서 Setup을 실행한다. 기존 clone이 오래됐으면 자동 덮어쓰기하지 않고 알려준다. 실행 중인 trainer는 유지한다.
2. Drive data의 `DRIVE_FOLDER`를 지정한다. 기본은 `MyDrive/MatSynth_PRIMA_Fabric`이며 `manifests/fabric_leather_training_manifest.jsonl`과 `source_shards/`를 포함해야 한다. Drive 원본은 수정하지 않고 프로젝트 상대 경로로 연결한다.
3. Train에서 새 `RUN_ROOT`를 지정하고 셀을 실행한다. 각 notebook의 모델은 고정이다. 별도 모델 선택·정밀도·구조 설정 셀은 없다.
4. 뷰어만 인터럽트한 뒤에는 Live view로 재연결한다. 표시 중단은 코드상 trainer 중단을 요청하지 않는다. 같은 runtime의 중복 trainer 시작은 거부한다.

NFPLight는 원 synthetic profile과 동일한 dim32, encoder[2,2,4,8], middle12, decoder[2,2,2,2], appearance18의 additive 모델이다. 저자 synthetic weight의 tensor shape metadata는64,102,378 elements이며 학습 진입 시 실제 모델 parameter count도 이 값으로 검사한다. U-Net++는 별도 `nested_model.py`의 NAF dense nested decoder, DS OFF다. 축소7M profile은 배포 설정에서 제거했다.

공통 수치 조건은 무양자화·FP32·gain9·seed19·micro1×accum4이며 [고정 학습 정책](../configs/training_noquant.json)과 [runtime](../configs/runtime_accum4.json)을 사용한다. Fresh는 pretrained 없이 시작한다. 데이터는 train566/val164,730 NPZ 약6.285GB다. Val은 재질당4crop, 합계656crop이며 공식 test payload는 읽지 않는다. Drive 직접 I/O 속도는 아직 측정하지 않았다.

화면 위에는 phase·PID·train step·최근 loss/LR가 나온다. tqdm의 train step 총량은 현재 stage의 **cap**이며 수렴 ETA가 아니다. Val은 완료 crop/전체만 표시한다. 초기 validation이 끝나기 전에는 Val loss가 없고, 첫 optimizer update 전에는 Train loss가 없다. 이때 빈 plot에 대기 이유가 표시된다. 이전 버전에서 시작한 trainer는 crop 진행 정보를 기록하지 않아 새 뷰어도 이를 복원할 수 없다.

Figure는 Train/Validation loss의1×2이며 Y축은 log다. Train은 실제 objective와 단계별100-update 평균이다. Validation은 map 단계에서 NDRS L1 sum, render 단계에서 NDRS sum+render_weight×novel-light render L1을 표시한다. 이는 **표시 계산만** 바꾸며 checkpoint 선정·scheduler 지표를 변경하지 않는다. Val은 crop 평균 후 재질 동일 가중 평균이며 Train minibatch와 광원·집계가 다르다. Map/render는 분리해 표시한다. 0은 log 축에 그릴 수 없다는 문구를 표시하며 작은 양수로 바꾸지 않는다.

실행 source와 numerical config는 공통 코어에 유지하고 notebook에는 복제하지 않는다. Loss·모델 연산·Adam·scheduler·sampling·RNG·checkpoint 계약은 유지하며 `pipeline.evaluate`에 optional progress callback만 추가했다. 진행 기록은 runtime 로컬 `activity.json`에 최대 약1초 간격으로 쓴다. 기존 로컬 frozen 실행 source는 변경하지 않았다.

출력/checkpoint는 `RUN_ROOT`의 runtime 로컬 파일이다. Drive 자동 백업은 아직 포함하지 않는다. Full `run/last.pt`의 exact resume는 모델/설정/Adam/scheduler/전체 RNG, Python/PyTorch/CUDA/GPU, 데이터 절대 경로·size·mtime가 같아야 한다. Colab reset·재업로드·다른 GPU·Windows checkpoint 이관은 거부될 수 있다. 마지막 저장 이후 미저장 update는 복구하지 않는다.

현재 테스트 실행은 사용자 지시로 보류했다. 실제 Colab/Drive 인증·GPU 학습의 표시 결과는 아직 확인하지 않았다. Colab 자원 가용성과 병렬 실행 제한은 [공식 FAQ](https://research.google.com/colaboratory/faq.html)를 따른다.
