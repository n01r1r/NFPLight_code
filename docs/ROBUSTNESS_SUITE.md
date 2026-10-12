# 7.61M 후속 증강·용량·강건성 실험

2026-10-09 사용자가 Council 논의 뒤 **추천대로 전부 개시**를 지시했다. 이전 초안·Council 권고는 그대로 보존하고 이 문서를 실제 실행 계약으로 사용한다. 목표는 validation NDR best map, 단일seed19 선별이다. FP32, AMP/TF32 off, near4/far12/gain9/F0=.04, raw_calibrated_gain9_v3, denoiser금지, 공식test·실측DNG평가범위제외를 유지한다.

## 승인된 실행 범위와 운영 선택

|순서|조건|변경 한 가지|
|---|---|---|
|1|baseline7m_p01|새 패키지 공통대조, 기존7.61M/.1 정책|
|2|capacity15m_p01|middle4→12, 15,085,034 parameters|
|3|dropout7m_p05|같은NAF residual 위치의원소별p.5|
|4|gt_aug7m_p01|GT diffuse scalar×U(.8,1.2), roughness+[−.05,.05], 물리범위clip후고정기하재렌더|
|5|extra_crop7m_p01|각훈련재질의증명된비중첩crop1개씩추가, 원래16+8=24slots|
|6|ema7m_p01|EMA CPU shadow decay.999 peroptimizerupdate, EMA 평가·선택, onlineAdam·EMA동시복원|
|7|adamw7m_p01|AdamW, decoupled weight decay1e−4|

입력손상평가·두기존후보의불일치·성분gradient진단도개시한다. Noise 학습, 더많은모델규모·추가seed·큰factorial grid는 현재기본queue에포함하지않는다. 추천에서대안으로남긴것과직접비교조건을구분한다. 새재질·독립평가군 확보는현재 bank진단의한계를보고한후별도데이터조건으로구체화한다. 임의외부다운로드나공식test개봉은없다.

수치는이번운영설정이며최적값이나성능향상주장이아니다. Crop추가는정답원본영역의좌표·증거를먼저확인한다. 기존traincrop와unseen crop의중첩이발견되면이력을보존하고같은재질crop진단의한계로보고한다. Extra arm은모든8재질에동일한추가slot수를확보해야하며불가하면이유를기록하고나머지조건을진행한다.

## 공통 학습·비교 계약

기존[Plateau 후속](PLATEAU_FOLLOWUP.md)의map5e−4→1e−5/render1e−4→1e−6, ReduceLROnPlateaufactor.5/patience5/thresholdrel.005/cooldown2, eval·checkpoint100, floor실제학습평가20, map50k/render15k상한을유지한다. Mapbest 복원→freshrenderAdam,rendermeanNDR≤bestmap×1.02 후보중최저novelrender선택,step0포함규칙을유지한다. Cap은일시정지이며수렴판정이아니다. 초반arm부진만으로자동기각하지않는다.

MapNDR의normal은unitXYZ, diffuse/roughness는packed2v−1 L1이며성분동일평균이다. GT증강·입력손상은관측을다시생성하되평가GT·관측·12novel조명은고정한다. 분포를보고사후목표를바꾸지않는다. EMAarm의optimizer는online모델을학습하며EMA선택이개입임을명시한다. 원본상태와선택상태는분리저장한다.

모델 초기화·표본(seed+10000)·D4(+20000)·GT증강(+30000)·입력noise(+40000)의난수상태를복원가능하게보존한다. 같은seed라도다른depth/dropout이완전같은난수소비·궤적을뜻하지않는다. 동일update상한은동일compute/time이아니므로학습비용을기록한다. baseline경로의처음평가/훈련prefix재현을검증한다.

## 구현·실행·저장

새패키지 experiments/robustness_suite에서구현하고기존plateau_followup/regularization/helper소스·완료checkpoint는변경하지않는다. 경로는 artifacts/robustness_suite_seed19_20261009이며job별새lineage를쓴다. 고정source/inputidentity는path/size/mtime이고SHA-256을실행하지않는다. 메모리제한·OOMcommittedrollback·단계/Adam/scheduler/RNG/EMA상태·유한값검사·CPUcheckpoint저장·복원과유한checkpoint개수규칙을유지한다.

단일GPUqueue로직렬화한다. CPU검사후각변경경로의실제GPUmap+render smoke를분리artifact에서통과시키고학습을개시한다. CUDA작업중복은PID생성identity·명령·GPU상태로검사한다. 실패·메모리/예산일시정지는checkpoint를보존하고queue중단원인을보고한다. 수치실패를자동재시도로덮어쓰지않는다.

## 진단 범위

성분별gradientnorm및map/render 방향은지정update에서만측정하며추가optimizerupdate를만들지않는다. 입력손상은합성linear관측에서Gaussianσ.001/.005/.01, 픽셀결측1%, saturation×1.2의각각독립stress조건으로평가한다. 이크기는실측sensor모델로calibrate된것이아니다. 비교target·유효mask와기본gain/기하를유지하고noise처리clip을기록한다. 두모델의map불일치는기술통계이며보정된불확실성이나OOD탐지정확도가아니다.

## 결과 해석

재사용heldout8재질, 같은훈련재질unseen crop, category및성분별결과를분리한다. 단일seed선별·깊이family·한축대조라는주장범위를유지한다. 3기본조건으로capacity×dropout상호작용을주장하지않는다. 검증한구현·smoke·실제job진행·학습효과는서로다른근거다.

체크포인트 과밀 예방과 10분 외부 감시는 [저장 공간 운영 정책](CHECKPOINT_STORAGE_POLICY.md)을 따른다. 실행 중 동결 trainer는 변경하지 않는다.
