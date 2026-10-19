# 감사와 검증

이 저장소는 near/far 선형 RGB로 SVBRDF를 추정하고 Fabric/Leather 데이터를 학습한다. 단일 DNG 전처리 진입점은 센서 입력을 모델 입력으로 만든다. 예상 부하는 한 사용자의 단일 GPU 학습 또는 CPU 추론이다.

## 기존 코드 감사와 조치

1. **7M 로딩에 실험 runner가 필요했다.**
   - 역할: checkpoint 구조를 만들고 weight를 로드한다.
   - 문제: 기본 capture 모델은 64M이며 7M factory는 robustness 실험 패키지에 있었다. 기본 로더로 전달용 7M을 로드하면 shape가 맞지 않았다.
   - 조치: `nfplight/network.py`와 profile config로 통합하고 기본을 7M으로 정했다. Metadata 충돌·누락한 명시 profile·strict state 불일치를 검사한다.
   - 방치 영향: 전달받은 weight가 실행 경로 선택에 따라 로드되지 않는다.

2. **기존 weight에도 현재 relmap을 적용할 수 있었다.**
   - 역할: near/far 쌍에서 relation 특징을 만든다.
   - 문제: 이전 weight를 로드해도 새 공간 보정 경로로 계산하면 과거 결과가 재현되지 않는다.
   - 조치: Weight별 config에 relation·geometry·normal decoder를 명시한다. 7M original과 spatial/C를 분리하고 학습 gain9와 실측 patch20도 별도로 계산한다.
   - 방치 영향: 가중치가 같아도 특징과 예측이 달라진다.

3. **C 이어 학습이 여러 과거 실험 패키지에 연결돼 있었다.**
   - 역할: loss·평가·Adam·scheduler·sampler·RNG를 유지한다.
   - 문제: 코드를 단순 이동하거나 model state만 로드하면 dropout·샘플 선택·Adam moments·종료 상태가 달라진다.
   - 조치: 독립 TrainingPipeline/Trainer로 재작성했다. Parameter 이름·순서, 전체 상태와 역사적 identity를 유지하고 새 implementation provenance를 따로 기록한다. Runtime·환경·데이터 identity 불일치와 smoke 재개를 거부한다.
   - 방치 영향: 이어 학습이라고 표시하면서 실제로는 다른 학습 경로를 실행한다.

4. **DNG 전처리가 추론·리포트와 결합돼 있었다.**
   - 역할: 센서 연산·공통 정합·평균 선택과 검토를 수행한다.
   - 문제: 모델을 사용하려면 보고서 실행기와 역사적 hash 경로까지 따라가야 했다.
   - 조치: prepare/finalize를 단일 preprocess_dng.py로 분리했다. 검토 전 평균을 금지하고 수치/무늬 실패 시 양쪽 frame00을 사용한다. AHD·FP32 정합·광도 계산은 검증된 수식을 보존해 재배치했다. Hash를 실행하지 않고 path/size/mtime provenance임을 명시한다.
   - 방치 영향: 추론 API와 입력 규약이 보고서 변경에 함께 영향을 받는다.

## 수행한 검사

- 실제 6개 checkpoint: 전달용 7M 3개, 현재 C last, 저자 real/syn. Strict load·parameter 순서·256×256 특징·raw/decoded 출력이 기존 CPU FP32 경로와 비트 단위 일치했다. `compatibility.json`의 path/size/mtime는 검사 당시 읽은 파일이다. C last는 이후 학습으로 변경될 수 있다.
- 기존 엔진과 새 Trainer의 동일 다음 batch/dropout/Adam update 및 평가·scheduler·전체 RNG 상태를 작은 결정적 CPU 사례에서 비교해 일치했다.
- 실제 7M C weight로 CPU backward·gradient clipping·Adam weight/moments를 비교해 일치했다. 실제 train material 한 crop의 D4 8종·safe target·mask·합성 특징도 일치했다.
- 입력 shape/range/nonfinite/mask, gain 분리, 공간 계수, normal decoder, checkpoint 충돌, evaluation RNG 보존, budget pause, atomic write 실패, 출력 lock, DNG 평균/fallback, AHD 센서 규약을 pytest로 검사했다.
- 단일 DNG 스크립트의 prepare→검토→finalize는 가짜 sensor IO와 실제 FP32 정합·crop·photometry를 사용한 통합 fixture로 검사했다.
- CLI 추론과 Python `-O`에서 nonfinite 거부, 합성 fixture 한 update 학습·smoke 표시를 검사했다. 실제 연구 학습과 구분한다.
- 현재 C의 checkpoint/config 및 전체 데이터 identity가 새 resume gate를 통과함을 CUDA 초기화 없이 확인했다.
- 새 코어와 전처리 스크립트에서 과거 model/network/experiments/capture_processing import가 없음을 검사했다. Formatter·자동 인덱싱·SHA-256은 실행하지 않았다.

## 보존과 범위

새 브랜치에서 교체가 확인된 과거 파일 168개를 개별 제거했다. 원래 checkout은 수정하지 않았다. 기존 파일의 Git 일치와 별도 ZIP의 byte 일치를 확인했으며, `.git`·checkpoint·결과·dataset 외부 대상에 재귀 삭제를 수행하지 않았다. 상세 목록은 `removed_sources.json`이다.
기존 DNG·weight·결과·예제 input_data는 복사하거나 삭제하지 않았다. 새 worktree에서 필요 시 원래 파일의 절대 경로를 명시한다.

## 추가 전면 검사와 수정

- CUDA 장치 문자열 `cuda`와 tensor 장치 `cuda:0` 비교 때문에 정상 입력을 거부하던 오류를 수정했다. 실제 할당된 특징 tensor의 장치를 기준으로 비교한다.
- exact resume가 norm epsilon·특징 normalization 등의 변경을 놓치던 검사를 수정했다. 새 snapshot은 전체 model/feature profile을 보존하고, 재개 시 수치 설정과 CPU thread 수까지 일치해야 한다.
- 실제 6개 weight의 parameter 순서·strict load·256px 특징·raw/decoded 출력과 renderer가 CPU 및 CUDA에서 기존 구현과 비트 단위 일치했다.
- 현재 C의 실제 Adam moments·CUDA RNG를 복원한 7M 모델의 다음 backward·clipping·Adam update가 일치했다. 이 검사의 입력은 1×21×32×32이며 생산 batch의 연구 update를 실행한 것이 아니다.
- 실제 train crop과 D4 8종, CPU/CUDA 합성 특징이 일치했다. CPU에서 256px·40광원·weight10 전체 C loss와 prediction gradient도 일치했다.
- 작은 결정적 모델로 map→render 전환, selected weights, floor 종료, scheduler·전체 snapshot 및 map_stage_end를 기존 엔진과 비교했다.
- 현재 C의 실제 데이터 identity와 Python/PyTorch/CUDA/GPU 환경이 새 재개 gate를 통과했다. 실제 Trainer로 전체 checkpoint를 복원한 snapshot의 모든 상태도 원본과 일치했다. NaN optimizer/scheduler와 잘못된 step counter는 거부한다.
- 기존 입력 경로로 DNG 5묶음 50장을 재처리했다. 과거 실제 산출물의 모든 geometry·alignment·crop support와 일치했다.

## 사용자가 승인한 새 입력 영역

새 기본은 `robust_target`이다. 흰 종이와 구분되는 target 경계를 edge·직선 segment로 찾고 같은 near_00 사각형을 모든 near/far에 적용한다. 기존 crop46은 `capture_legacy.json`에 보존했다. 새 입력 시야와 예측값은 기존 결과와 같다고 주장하지 않는다.

최소 3개 공통 태그와 양 축의 분포를 요구한다. 전체 태그로 먼저 맞추고, 필요 시 결정적인 부분집합 검사로 일관된 최대 태그 집합을 찾는다. 사용·제외 ID, 전체 잔차와 최종 RMS를 기록한다. 일관된 집합이 부족하더라도 전체 태그의 기하가 유효하면 평균 불가로 기록하고 frame00 fallback을 유지한다. 공간 분포가 부족하면 거부한다. 1px 기준은 완화하지 않았다.

경계 검출에 사용한 표시 영상은 측정 RGB로 사용하지 않는다. 원본 FP32 AHD에서 한 번 warp하고 area resize한다. 새 입력의 finalize에는 경계와 무늬 정합 검토가 필요하다. 자동 검출 결과를 사람의 검토로 기록하지 않는다.

태그 3개 누락 사례, 잘못된 태그 1개 배제, 한쪽에 몰린 태그 거부, blank paper·nonfinite 설정 거부, 경계 미검토 거부, 두 입력 방식의 prepare/finalize를 검사했다. 실제 5묶음 좌표에서 3개 태그만 남긴 알려진 이동 복원도 검사했다. 이는 실제 가림 촬영을 대신하는 연구 증거가 아니다.

최종 자동 검사: 전체 pytest 54 passed (26.65초). 기존/새 기본 경로 모두 실제 DNG 50장 전처리 오류가 없었고 모든 frame00이 유효했다. 새 기본 경로의 정합은 최종 코드 재계산과 일치했다. 5묶음의 평균 수치 게이트 통과 프레임 수는 각각 5/10, 10/10, 9/10, 10/10, 10/10이며, 평균 여부는 이후 사람의 검토까지 포함해 결정한다. 상세 결과와 최종 정적 검사는 validation.json에 기록한다.

## 제한

실행 중 연구 학습은 변경하지 않았다. 생산 batch 크기의 GPU 연구 update, 새 가림 촬영, 사람의 최종 target 경계·무늬 검토, 모든 지원 버전의 설치·성능 benchmark는 수행하지 않았다. 이전 보고서의 과학적 결론을 재검증한 작업이 아니다. Formatter·자동 인덱싱·SHA-256은 실행하지 않았다.

판정: 기존 profile의 weight·수치·재개 호환성과 새 기본 전처리의 자동 검증을 통과했다. 실제 최종 입력 생성에는 경계·무늬 검토가 남아 있다.
