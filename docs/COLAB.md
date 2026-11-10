# Colab

[NFPLight 64M original](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/nfplight.ipynb) · [NAF U-Net++ DS OFF](https://colab.research.google.com/github/n01r1r/NFPLight_code/blob/codex/colab-experiments/notebooks/unetpp.ipynb)

1. 새 GPU 런타임에서 Setup을 실행한다. 기존 clone이 오래됐으면 자동 덮어쓰기하지 않고 알려준다. 실행 중인 trainer는 유지한다.
2. Drive → local data의 `DRIVE_FOLDER`를 지정한다. 기본은 `MyDrive/MatSynth_PRIMA_Fabric`이며 `manifests/fabric_leather_training_manifest.jsonl`과 `source_shards/`를 포함해야 한다. Train/val shard를 프로젝트 상대 경로 `data/MatSynth_PRIMA_Fabric_local/`에 실제 복사한다. 복사 byte 진행은 tqdm에 표시하며 완료 후 로컬 경로를 출력한다. Drive 원본은 수정하지 않는다.
3. Train에서 새 `RUN_ROOT`를 지정하고 셀을 실행한다. 각 notebook의 모델은 고정이다. 별도 모델 선택·정밀도·구조 설정 셀은 없다.
4. 뷰어만 인터럽트한 뒤에는 Live view로 재연결한다. 표시 중단은 코드상 trainer 중단을 요청하지 않는다. 같은 runtime의 중복 trainer 시작은 거부한다.

NFPLight는 원 synthetic profile과 동일한 dim32, encoder[2,2,4,8], middle12, decoder[2,2,2,2], appearance18의 additive 모델이다. 저자 synthetic weight의 tensor shape metadata는64,102,378 elements이며 학습 진입 시 실제 모델 parameter count도 이 값으로 검사한다. U-Net++는 별도 `nested_model.py`의 NAF dense nested decoder, DS OFF다. 축소7M profile은 배포 설정에서 제거했다.

두 profile과 모델 생성 코드를 확인하면 dim32, encoder depth[2,2,4,8], middle12, decoder depth[2,2,2,2]가 일치한다. 각 encoder의 채널은32→64→128→256→512이며 fused N/R decoder의 채널은1024→512→256→128→64다. U-Net++의 decoder depth2는 nested node마다 적용된다. 따라서 decoder 하나당10node×2=20 NAFBlock이며, 원 additive decoder의4stage×2=8과 전체 block 수/parameter 수가 같다는 뜻은 아니다. 전체 NAFBlock은 NFPLight72개, U-Net++96개다. U-Net++도 원본 폭을 유지한 dim32이며 축소 모델이 아니다.

공통 수치 조건은 무양자화·FP32·gain9·seed19·micro2×accum2(유효 batch4)이며 [고정 학습 정책](../configs/training_noquant.json)과 [runtime](../configs/runtime_accum4.json)을 사용한다. Runtime 파일명은 기존 호출과의 호환성을 위해 유지했다. Fresh는 pretrained 없이 시작한다. 데이터는 train566/val164,730 NPZ 약6.285GB다. Val은 재질당4crop, 합계656crop이며 공식 test payload는 복사하거나 읽지 않는다. Colab validation은 A100을 대상으로 batch16을 기본값으로 사용한다. 실제 A100 속도 및 최적 batch/VRAM 안전성은 아직 측정하지 않았다.

Validation은 원래 material→crop 순서대로 묶어 FP32 추론·near/far 관측·12광원 정답/예측 렌더링을 수행한다. 각 crop마다 mask의 유효 픽셀×채널×광원 분모로6개 지표를 따로 계산하고, 기존과 같이 CPU float64에서 crop→재질→category/source-group 평균을 구한다. 전체 batch의 mask를 합쳐 평균하지 않는다. 마지막 짧은 batch도 처리하며 population과 평가 간격을 유지한다. CUDA OOM일 때만 같은 미처리 crop부터 batch를 절반으로 줄여 재시도하고 batch1 실패는 전파한다. 낮춘 상한은 해당 프로세스의 다음 validation에도 재사용한다. 요청 batch·사용 batch 크기·OOM 횟수는 evaluation heldout의 `validation_batch_size_requested`, `validation_batch_size_final`, `validation_batch_sizes_used`, `validation_oom_retries`에 기록하고 진행 화면에도 batch/OOM 횟수를 표시한다.

사용자 승인으로 train 분할을 micro1×accum4에서 micro2×accum2로 변경했다. 유효 batch4·유효 픽셀 가중 objective·렌더 loss·Adam·scheduler 정책·모델·RNG 복원 경로·FP32/AMP/TF32 설정은 유지한다. Validation batching은 sample 단위 feature normalization에서만 허용한다. Batched CUDA kernel과 지표 축약 순서 때문에 FP32의 작은 차이는 가능하며 bitwise 동일성을 보장하지 않는다. Scheduler/best 판정의 정의는 같지만 경계의 미세 차이가 학습 이력에 영향을 줄 수 있다. Micro 분할 또는 validation 설정이 다른 기존 checkpoint와 새 runtime을 exact resume로 자동 호환시키지 않는다. 실행 중인 trainer와 이미 저장된 run의 설정에는 소급 적용하지 않는다.

데이터 복사 전에 필요한 추가 용량과 학습용12GiB 여유를 확인한다. 파일은 임시 복사 후 크기와 source size/mtime가 유지됐는지 확인하고 교체하며, 복사 실패는 학습 시작 전에 오류로 표시한다. 재실행은 source와 size/mtime가 일치하는 로컬 파일을 재사용한다. 내용 hash 검사는 수행하지 않는다. 다른 로컬 파일이나 symlink는 덮어쓰지 않는다. Runtime reset으로 로컬 데이터가 사라지면 다시 복사한다. 기존 버전의 Drive symlink와 다른 디렉터리를 사용하며 이미 실행 중인 trainer의 데이터 경로를 변경하지 않는다.

화면 위에는 phase·PID·train step·최근 loss/LR가 나온다. tqdm의 train step 총량은 현재 stage의 **cap**이며 수렴 ETA가 아니다. Val은 완료 crop/전체만 표시한다. 초기 validation이 끝나기 전에는 Val loss가 없고, 첫 optimizer update 전에는 Train loss가 없다. 이때 빈 plot에 대기 이유가 표시된다. 이전 버전에서 시작한 trainer는 crop 진행 정보를 기록하지 않아 새 뷰어도 이를 복원할 수 없다.

Figure는 Train/Validation loss의1×2이며 Y축은 log다. Train은 실제 objective와 단계별100-update 평균이다. Validation은 map 단계에서 NDRS L1 sum, render 단계에서 NDRS sum+render_weight×novel-light render L1을 표시한다. 이는 **표시 계산만** 바꾸며 checkpoint 선정·scheduler 지표를 변경하지 않는다. Val은 crop 평균 후 재질 동일 가중 평균이며 Train minibatch와 광원·집계가 다르다. Map/render는 분리해 표시한다. 0은 log 축에 그릴 수 없다는 문구를 표시하며 작은 양수로 바꾸지 않는다.

Live view는 전체 출력을 지우지 않고 [IPython display handle](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.DisplayHandle)로 고정된 상태/figure 영역을 각각 갱신한다. 상태는5초마다 확인하고 figure는 새 Train loss 또는 완료된 Val 기록이 생겼을 때만 교체한다. 갱신 사이에 figure가 빈 화면으로 사라지는 경로를 제거했다. 실제 Colab frontend의 시각적 깜빡임 감소는 아직 확인하지 않았다.

실행 source와 numerical config는 공통 코어에 유지하고 notebook에는 복제하지 않는다. 학습 코어·모델·Adam·scheduler·sampling·RNG·checkpoint 계약은 유지하며 validation만 batching과 optional progress callback을 사용한다. 진행 기록은 runtime 로컬 `activity.json`에 최대 약1초 간격으로 쓴다. 기존 로컬 frozen 실행 source는 변경하지 않았다.

사용자 추가 승인으로 local checkpoint가 정상 저장된 후 `CHECKPOINT_DIR`에 `last.pt`와 `best.pt`를 복사한다. 기본은 `MyDrive/NFPLight_runs/<RUN_ROOT 이름>/`이다. 별도 cell은 늘리지 않았다. 기존 Drive 폴더는 거부하므로 새 run 이름을 사용한다. Local `run/best.pt`는 현재 선정된 eligible best_render가 있으면 그것을, 없으면 best_map을 내보낸다. 선정 조건은 기존 trainer를 그대로 따른다. Best가 갱신될 때만 best를 다시 쓰고 last는 checkpoint 저장마다 갱신한다. 초기 validation 후의 첫 full checkpoint에도 적용한다.

`last.pt`는 model·Adam·scheduler·전체 RNG·단계·best 이력을 포함한 재개용 전체 상태다. `best.pt`는 inference용 model-only weight이며 exact resume에 쓸 수 없다. 두 파일은 각각 임시 파일 쓰기 후 교체한다. Full last에도 best state가 포함되므로 Drive best 파일 갱신 중 실패해도 성공한 last에서 선정 모델을 복원할 수 있다. Drive 복사 실패는 local checkpoint를 보존하고 화면에 표시하며 다음 checkpoint 때 다시 시도한다. 복사 중에는 phase가 `Drive checkpoint backup`으로 표시된다.

Full `run/last.pt`의 exact resume는 모델/설정/Adam/scheduler/전체 RNG, Python/PyTorch/CUDA/GPU, 데이터 절대 경로·size·mtime가 같아야 한다. Drive 보존이 이 조건을 완화하지 않는다. Colab reset·재업로드·다른 GPU·Windows checkpoint 이관은 거부될 수 있다. 마지막 저장 이후 미저장 update는 복구하지 않는다. 기존 버전으로 이미 실행 중인 trainer에는 새 자동 backup이 소급 적용되지 않는다.

현재 테스트 실행은 사용자 지시로 보류했다. 실제 Colab/Drive 인증·GPU 학습의 표시 결과는 아직 확인하지 않았다. Colab 자원 가용성과 병렬 실행 제한은 [공식 FAQ](https://research.google.com/colaboratory/faq.html)를 따른다.
