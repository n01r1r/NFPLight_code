# Fabric + Leather subset 확장 및 학습 노출량 점검

학습 및 합성 validation/test의 고정 gain9는 실측 inference 계약이 아니다. 실측은 중앙20×20 RGB 평균 비율을 사용하며 [CAPTURE_REQUIREMENTS §1.4](CAPTURE_REQUIREMENTS.md#14-2026-10-13-사용자-확정-학습과-실측-추론-분리)를 따른다.

## 독립 세션의 현재 기준

[학습 scheduler 정책](TRAINING_SCHEDULER_POLICY.md) → [현재 학습 상태](CURRENT_TRAINING_STATE.md) → [artifact 인덱스](../artifacts/training_artifact_index.json)를 먼저 읽는다. 아래 데이터 계약·구현 설명·날짜별 실행은 해당 범위의 기록이다. 새 default는ReduceLROnPlateau지만기존trainer는cosine/고정LR이며새default구현은미완료다. 옛예산·seed·실행명령을새후속계약으로복사하지않는다. 2026-10-09 Go 후 현재 단일seed19의 validation best·종료·전환·best복원·넉넉한 연장 예산은 [후속 계약](PLATEAU_FOLLOWUP.md)을 적용하며 별도 runner의 구현·검사를 마쳐 7.61M seed19 학습을 시작했다. 실제 진행과 예산은 현재 상태 문서의 run 기록을 따른다. 현재완료된7.61M/15.09M을재개대기라고해석하지않는다.

2026-10-09 최신 사용자 지시: 새 학습의 기본 scheduler는 ReduceLROnPlateau다. [정책과 적용 상태](TRAINING_SCHEDULER_POLICY.md)를 따른다. 아래 cosine·고정 LR 관련 기록은 기존 lineage의 이력이며, 실행 중인 고정 LR 실험에는 소급 적용하지 않는다. Map→render 전환·best 선택·재개 상태의 새 계약은 별도 검토 후 기록한다.

현재 HTML 표시·집계·디자인 규약은 [TRAINING_HTML_REPORT_CONTRACT.md](TRAINING_HTML_REPORT_CONTRACT.md)가 기준이다. 아래 초기 plot 조사/구현 기록의 이전 표시 방식은 해당 문서의 대체 사항을 따른다.

2026-10-08 후속 확정: small baseline의 bottleneck FC 비교는 [독립 실험 폴더](../experiments/bottleneck_fc/README.md)의 checked-in config와 관리 명령으로 준비한다. 조건은 `baseline`, `baseline+Bottleneck_FC`, `baseline+Pixel_MLP` 세 개다. 기존 64M trainer 및 정식 학습 budget은 이번 refactor에서 변경하지 않는다. 이전 exploratory 결과는 보존하고 새 규약 실험의 결과와 구분한다. 후속 사용자 지시로 2026-10-08 20:00 baseline 진단 학습을 시작했다. 출력 `artifacts/bottleneck_fc_baseline_20261008`, seed 7/19/31, 각각 map 1,000 + render 300 update이며 FP32/near4/far12/gain9를 유지한다. 명령은 `python -m experiments.bottleneck_fc.manage run --arm baseline --out artifacts/bottleneck_fc_baseline_20261008`이다. 재개 시 동일 명령에 `--resume`을 추가한다. 기존 viewer는 조건별 `seed7_baseline`/`seed19_baseline`/`seed31_baseline` 폴더를 읽고 포트 8767/8768/8769에서 실행한다.


2026-10-08 후속 승인: baseline 세 seed의 초기 1k map + 300 render 실행이 모두 완료된 후 단계별 수렴을 판정한다. 미수렴은 누적 map 10k / render 3k를 상한으로 연장한다. Early stop은 Validation 10회 연속 구간의 상대 개선 <0.5%이며 map=heldout NDR mean, render=heldout render L1이다. Map을 연장하면 map 종료 checkpoint·Adam에서 이어간 뒤 fresh Adam으로 render를 시작한다. 연장 결과는 `artifacts/bottleneck_fc_baseline_extended_20261008`에 분리하며 viewer 포트는 seed 7/19/31 순서 8777/8778/8779다. 기존 학습 프로세스와 과학 설정은 변경하지 않는다.

2026-10-08 이전 전체 데이터 학습 점검 상태: 이전 정식 run은 현재 경로에서 확인되지 않아 **새 학습 시작을 전제로 세팅 확인 후 대기**한다. 당시 학습·viewer는 실행하지 않았다. 현재 구현/default와 실제 전체 bank 검사 결과는 [학습 설정 점검](../artifacts/fabric_leather_contract_audit_20261008/SETUP.md)과 [검사 수치](../artifacts/fabric_leather_contract_audit_20261008/audit.json)를 참조한다. 아래 날짜별 내용은 당시 기록이며 21ch scratch 구현과 33ch pretrained 제안은 구분한다.

- DeepMaterials는 학습 대상에서 제외하고 노출량·분포 비교에만 사용한다.
- 기존 C:/MatSynth_PRIMA_Fabric의 Fabric shards와 frozen manifest는 보존한다.
- Leather 원본은 D:/MatSynth_preprocessed이며 새 shard만 C: bank에 추가한다.
- 전체 준비 manifest는 manifests/fabric_leather_split_manifest.jsonl로 보존한다. 실제 학습 선택 manifest는 manifests/fabric_leather_training_manifest.jsonl이며 사용자가 제외한 5개 Leather를 포함하지 않는다. Fabric 전용 소비자는 기존 manifest를 계속 사용한다.
- Leather는 native 256 crop 16개, opacity 사용 없이 전체 native window에서 추출, seed7, 개별 asset source link(홈페이지 공통값 제외)·반복 Deschaintre descriptor 그룹 기준 val20%를 사용한다.
- 새 Leather는 네 map만 저장하며 opacity를 읽거나 선별·mask에 쓰지 않는다. RGBA diffuse는 alpha를 버리고 RGB만 보존한다. uint16 roughness는 round(value/257)로 기존 uint8 표현에 맞춘다. n/d/r/s map은 spatial resize하지 않는다.
- Luna(max) 독립 감사로 기존 Fabric 456 train/val shard의 네 map 형식과 표본 448 map crop의 원본 일치를 확인했다. 표본 112 valid_mask는 opacity==255와 일치하므로 과거 opacity 선별 이력은 남아 있다. 새 Leather와 네 map 표현은 같지만 crop 모집단의 준비 규약은 완전히 같지 않다. 기존 bank를 보존하고 auxiliary mask는 학습 입력·loss에 사용하지 않는 계약을 유지한다.
- 원본 specular는 보존한다. 입력 생성·loss의 F0=.04는 trainer 조건이며 저장 데이터의 물리적 정답으로 주장하지 않는다.
- 생성 검증: map 정렬, distinct origins, RGB/roughness bit-depth 규약, decoded source→NPZ 배열 동일성, 기존 bank size/mtime·Fabric manifest bytes 보존, group split 교집합.
- SHA-256 계산·검증은 실행하지 않는다. 신규 provenance는 source size/mtime와 crop 좌표를 기록한다.
- 기존 Fabric crop RNG는 재구성하지 않았으며 Leather RNG를 명시한다. 별도 link를 가진 시각적 유사 재질까지 완전히 배제했다는 주장은 하지 않는다.

학습 합의: 무작위 초기화, denoiser 제외, n/d/r map별 동일 평균 L1, specular head supervision 제외, 입력 및 양측 rendering에 RGB F0=.04, 400K map stage +100K rendering stage, cosine LR 5e-4→1e-5 /1e-4→1e-5, 잠정 rendering weight1, R4·theta5/10/20/30/45×phi45도 간격의 고정40방향 linear-HDR L1. Effective batch는 4로 확정했다(출발점 microbatch 2 × accumulation 2). 전체 dataset loader 연결은 아직 구현하지 않았다.

노출량은 image 수·독립 재질·crop 반복·optimizer update를 구분한다. 동일 iteration×effective batch는 동일 샘플 presentation 수이지 동일 다양성을 의미하지 않는다. 분포 비교는 train만 사용하고 test 성능은 계산하지 않는다. 수치 근거는 artifacts/fabric_leather_data_audit_20261007/audit.json에 기록한다.

현재 bit-depth 제한: 새 Leather roughness가 uint16이면 normalized 범위를 보존하며 uint8로 양자화한다. 최대 양자화 오차는 약 0.5/255이다. Spatial resolution은 바꾸지 않는다. RGBA diffuse의 alpha는 무시한다. 이 형식은 네 map 전용 사용 규약이며 원본 16-bit 정밀도 보존은 아니다.


## 2026-10-07 사용자 확정

목표는 fabric/leather 계열의 재질 복원 성능이다. 범용 재질 일반화나 DeepMaterials와 동일한 crop 반복량은 목표로 삼지 않는다. Val은 해당 계열의 held-out 재질 성능과 과적합을 확인하기 위해 유지한다.

- 주 baseline: XYZ normal head, 단위 normal의 성분 평균 XYZ L1. Diffuse/roughness와 함께 map별 동일 평균을 사용한다. Specular head는 supervision에서 제외한다.
- Cosine normal loss는 baseline 이후 동일 초기화·effective batch·augmentation·presentation 조건의 별도 비교 실험으로 둔다. 최초 학습에 L1+cosine 혼합 loss는 추가하지 않는다. Cosine으로 변경할 때 loss scale과 map 간 비중 차이를 기록한다.
- Batch: effective batch 4. 초기 microbatch 2 × accumulation 2이며 GPU 측정 후 effective batch를 유지하면서 microbatch를 조정할 수 있다. 측정 전에는 40방향 rendering의 처리량이나 메모리 적합성을 주장하지 않는다.
- Augmentation: train에서 0/90/180/270도 회전 × horizontal flip 여부의 D4 8종을 균등 sampling한다. 네 map의 공간 변환을 일치시키고 normal x/y 성분을 함께 변환한다. 변환한 map으로 near/far 입력을 합성한다. 보간 기반 임의각 회전은 추가하지 않는다.
- Val: crop·조명·입력을 고정한다. 두 normal loss 실험 모두 평균/중앙값/P95 각도 오차와 rendering 오차를 기록하고 Fabric/Leather별 결과를 구분한다.
- 400K map +100K render optimizer update는 잠정 예산 상한이다. Val은 1,000 optimizer update마다 평가한다. 10회 연속 평가 구간의 상대 개선이 각각 0.5% 미만이면 plateau로 표시하며 자동 종료하거나 stage를 전환하지 않는다. 상대 개선은 (previous-current)/max(abs(previous),1e-12)로 기록하고 악화는 음수로 표시한다. Stage가 바뀌면 관찰 이력을 구분한다. Plateau 주 지표는 map stage에서 n/d/r 평균 L1, render stage에서 rendering L1이다. 다른 map 성분도 함께 관찰한다. 기존 map_tolerance=0.02는 rendering checkpoint 선택 조건이며 종료 기준이 아니다.

위 항목은 학습 설계의 확정 사항이며 현재 trainer 구현·실행 완료를 의미하지 않는다.


## 기존 plot 구현 조사

- C:/workspace/PRIMA/src/loss_study_plot.py와 PRIMA/README.md의 loss-study plot 규약을 확인했다. 매 optimizer update의 pre-update minibatch objective와 checkpoint cadence의 평가값을 분리한다. Raw train trace와 100-update trailing mean을 표시하고 x축은 정확한 optimizer update를 사용한다. 미기록 val 값을 보간하지 않는다.
- 해당 plotter는 30-cell 공통 완료 stage 이후 명령을 다시 실행해 PNG/PDF를 생성하는 방식이다. 자동 polling/live refresh 구현은 아니다. PRIMA의 고유 schema, 공통 stage와 hash 검증에 의존하므로 NFPLight 로그에 그대로 실행할 수 없다. 조사 중 hash 계산은 실행하지 않았다.
- NFPLight train_fabric.py는 매 update train.jsonl을 기록하고 기본 100 update마다 flush한다. validation.json은 평가 시 history/latest를 임시 파일 후 교체하여 갱신한다. log_every=100은 console 출력 간격이며 JSONL 기록 간격이 아니다.
- 현재 검색한 NFPLight 저장소(ignored 파일 포함)에서는 live plot 전용 script를 찾지 못했다. 인접 NeF_SVBRDF_readable의 live.py는 다른 JAX fitting 실험의 진행 기록 구현이며 이 trainer 전용 plotter가 아니다.
- NFPLight live viewer는 live_training_plot.py로 구현했다. 화면은 10초마다 갱신하며 train raw/100-update trailing mean과 기록된 val 점을 분리한다. Train x축은 total_step, val은 로그에 기록된 stage-local step이며 global offset을 임의 추정하지 않는다. Browser polling은 val 평가 간격을 바꾸지 않는다.
- 실행: python live_training_plot.py --run-dir artifacts/fabric_leather_training_gain9_20261008 --port 8765. http://127.0.0.1:8765에서 확인한다. 해당 경로는 새 학습을 위한 예정 출력 경로이며 viewer 실행만으로 학습은 시작되지 않는다.
- Viewer는 로그를 읽기만 하고 missing/invalid metric을 표시한다. 기존 trainer의 map_loss는 4-map 합이므로 n/d/r 평균으로 재해석하지 않고 원래 이름으로 표시한다. Train diffuse/roughness 개별 지표가 없으면 표시하지 않는다. Val n/d/r 평균은 명시된 세 성분의 평균으로 계산한다.
- Train 표시점이 6,000개를 넘으면 화면용 downsampling을 수행한다. 원본 train.jsonl은 보존하고 trailing mean은 원본 순서의 100개 값으로 계산한다. 빠른 갱신을 위해 JSONL을 증분으로 읽고 부분행은 다음 갱신까지 기다리며 로그 교체/재작성 시 cache를 초기화한다.
- 테스트: tests/test_live_training_plot.py의 세 검사가 stage 분리·상대 개선량·부분행·로그 재작성·val 미기록값 보존을 확인한다. Plateau는 화면 표시뿐이며 trainer 자동 종료 기능은 추가하지 않았다.


## F0 검토

2026-10-07 scientific-critical-thinking/council/antislop 검토는 기존 fixed F0=.04 baseline을 유지하도록 권고했다(3명 모두 중간 확신, 경험적 우열 미확인). 입력·양측 loss의 F0 고정은 모든 fabric/leather의 물리적 F0가 .04라는 주장이 아니다. 원본 specular 입력 + 양측 fixed-F0 재렌더링 loss도 정답 n/d/r에서 zero loss이므로 목적함수 모순으로 배제하지 않는다. 남는 차이는 입력 F0 변동에 대한 식별 가능성과 실제 성능이며 원본 인코딩 확인 후 별도 비교해야 한다. 상세 근거와 반론: artifacts/fabric_leather_data_audit_20261007/f0_council.md.


## 2026-10-07 speed/sanity 결과

RTX3090 FP32 speed probe는 artifacts/fabric_leather_speed_20261007_valid_probe/report.md와 summary.json에 기록했다. Effective batch4에서 2×2는 map .699s/render40 .928s, 4×1 후보는 .526s/.605s였고 후자는 allocated15.71GiB에서 OOM 없이 통과했다. 40방향은 동일 warm-cache 표본의10방향보다 약8.3% 느려 유지 권고다. 24-update 고정 Fabric/Leather 반복에서 loss가 감소하고 finite nonzero gradient 및 specular output gradient0을 확인했다. 이는 수렴/실측 복원 성능의 증거가 아니다.

정식 학습 전 blocker: train Leather 1652–1656_fur_cow_long의 원본 및 bank normal이 공간 상수·blue54이며 모든80 crop의 앞면 normal support가0이다. Train/val735 shard 검사로 확인했고 source→crop 대조도 일치했다. 기존 bank·manifest는 보존했다. 후속 speed 측정은 이5개를 명시적으로 제외한 in-memory population으로만 수행했다. 사용자가 5개 제외를 승인했다. 원래 전체 준비 manifest의 게이트 실패 이력은 보존하고, 새 학습 선택 manifest에서 5개를 제외한다.

400K+100K 및1000-step val 기준 ETA는 2×2 약113h, 4×1 후보 약85h이다. Val 비용은8재질×4crop timing을164재질로 외삽했다. 짧은 warm-cache probe 외삽이고 checkpoint/startup시간은 제외하며 ±20%는 통계적 신뢰구간이 아닌 운영 여유다. 정식 trainer 통합 및 optimizer reduction 비교는 별도로 필요하다. Speed test만 실행했고 정식 학습은 시작하지 않았다.


## 2026-10-07 제외 승인 반영

사용자의 “제외해” 지시에 따라 1652–1656_fur_cow_long 5개 train Leather를 새 fabric_leather_training_manifest.jsonl에서 제외했다. 기존 split manifest와 bank payload는 그대로 보존한다. 최종 선택은 Fabric358/98/5, Leather208/66/7, 합계 train566/val164/test12, 저장 train crop9056개다. 제외 근거 및 검증은 manifests/fabric_leather_training_summary.json과 artifacts/fabric_leather_speed_20261007_valid_probe/authorized_exclusion.json에 기록했다. 새 manifest의 재질ID 유일성과 split group 교집합0, 기존 val/test metadata 행 보존을 확인했다. 공식 test payload와 SHA 검증은 실행하지 않았다.

이로써 발견된 zero-front-facing-support 재질 문제는 학습 선택에서 해결했다. Benchmark 기본 manifest를 새 선택 경로로 바꿨다. 기존 production trainer의 dataset loader/loss/rendering 계약 통합은 아직 완료하지 않았고 정식 학습은 시작하지 않았다. 이전 speed 보고서의 실패/임시 제외 기록은 당시 실행 근거로 보존한다.


## 2026-10-08 입력 계약 교정 및 재개 대기

사용자 확정: 입력 합성의 colocated light/camera는 near=4, far=12 모델 좌표, surface 범위 [-1,1], lamp intensity=16이다. 실측 10/30cm의 비율을 유지하며 physical crop 크기와의 절대 좌표 대응을 주장하지 않는다. Far RGB 배율은 고정 9=(12/4)^2이며 중앙 픽셀로 추정하지 않는다. Linear FP32, 입력 gamma/8bit quantization/denoiser 없음. 새 feature 버전은 raw_calibrated_gain9_v3이다. Rendering loss의 R4 40방향 및 view_z=2.75는 유지한다.

2026-10-08 최신 사용자 지시: 이전 학습 run의 폐기·재개 금지 원칙을 철회한다. 남아 있는 프로세스·기록·웨이트·감시 설정은 보존하고 재개 지시를 기다린다. 현재 지시는 실제 resume 실행이나 감시 재활성화를 허용하지 않는다. 재개 시 실제 checkpoint 존재와 optimizer·step·LR·RNG 상태, 입력 기하·gain 및 코드 계약의 호환성을 확인한다. 계약 차이는 정책상 금지가 아니라 기술적으로 해결할 항목이며, 이번 철회만으로 loader의 호환성 검사를 우회하거나 이미 제거된 artifact가 복구되었다고 간주하지 않는다. 새 출력 기본값은 artifacts/fabric_leather_training_gain9_20261008이며 새 학습은 아직 시작하지 않았다. 기존 400k/449k/500k 비교와 저자 원본 웨이트는 당시 입력 계약을 보존한다. 이전 speed/smoke 수치는 교정 전 기하의 구현 점검이며 gain9 성능 증거가 아니다.

## Live viewer 통합 그래프

사용자 요청으로 train/val 각각 하나의 큰 canvas에 normal/diffuse/roughness/rendering/NDR mean을 색상으로 구분해 표시한다. Train total objective도 구분하고 raw는 옅게, 원본100-update trailing mean은 진하게 표시한다. 모든 loss curve는 전체 train 및 모든 val category 데이터에서 산출한 공통 X/Y 범위를 사용한다. Val Fabric/Leather 선택으로 범위를 바꾸지 않는다. Linear/log10 toggle을 지원하며 log에서0이하 값은 표시하지 않고 명시적으로 안내한다. 각도(도)·learning rate는 다른 단위이므로 같은 loss 축에 섞지 않고 수치로 표시한다. Val은 기록된 평가점을 직선으로 연결하며 marker를 유지한다. 중간 update 평가값을 생성하지 않는다. 동일 범위는 시각적 비교를 위한 것으로 서로 다른 objective 의미의 동등성을 주장하지 않는다.

HTML은 live_training_plot.html에서 요청마다 읽어 UI 수정 시 학습이나 reader code를 다시 시작할 필요가 없다. Viewer는 독립 process로 실행할 수 있다. JS 검사 tests/check_live_training_plot.cjs는 공통 범위·log zero 처리·category range 불변성과 canvas 실행을 확인하며 관련 Python5개 검사도 통과했다. HTTP HTML/API도 확인했다. 브라우저 자동 화면 검사는 sandbox helper 오류로 수행하지 못했다. 기존 열린 탭은 새로고침하면 새 HTML이 적용된다. 이전 run의 viewer·한 시간 자동화 제거 원칙도 최신 재개 대기 지시로 철회한다. 현재 대기 중에는 viewer·감시를 시작하거나 중지하지 않는다.


## 2026-10-11 사용자 확정: normal 유지·NDRS 합

[새 데이터·손실 계약](TRAINING_OBJECTIVE_POLICY.md)이 앞으로 전체 학습의 손실 기준이다. Unit-normal XYZ 처리 유지, N/D/R/S 성분 L1의 합, S는기존 F0=.04 정답 감독, 입력/양측 render F0 고정 유지다. 과거 NDR 평균·specular 감독 제외 서술은 당시 lineage의 기록이며 새 기본 손실로 사용하지 않는다. 전체 데이터는 train566/val164이고 최근 train8/heldout8 고정 bank는 별도 진단이다. 전체 trainer 코드·저장 계약·로그·선택에 적용했으나 GPU 학습은 미개시이며 scheduler 구현 gate는 남아 있다.
