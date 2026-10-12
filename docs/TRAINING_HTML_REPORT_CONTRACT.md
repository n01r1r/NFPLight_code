# 학습 HTML viewer/report 규약

현재 규약: 2026-10-08 사용자 승인 내용. 이 문서는 NFPLight 학습 HTML의 표시·집계·디자인 기준이다. 다른 목적의 capture/cue report에 자동 적용하지 않는다. 이 문서와 예전 세션 기록이 충돌하면 현재 사용자 지시와 실제 구현을 확인하고, 아래의 이전 규약 대체 사항을 따른다.

## 독립 세션 시작 경로

[저장소 안내](../README.md) → [scheduler 정책](TRAINING_SCHEDULER_POLICY.md) → [현재 학습 상태](CURRENT_TRAINING_STATE.md) → 이 문서 → [학습 계약](FABRIC_LEATHER_TRAINING.md). 구현은 [HTML 템플릿](../live_training_plot.html), [읽기 전용 서버](../live_training_plot.py), [정식 trainer](../train_fabric.py)·[독립 Fabric/Leather trainer](../train_fabric_leather.py)이다. 새로운 HTML을 임의 디자인으로 다시 만들지 말고 기존 템플릿을 재사용한다.

```powershell
python live_training_plot.py --run-dir artifacts/fabric_leather_training_gain9_20261008 --port 8767
```

주소: http://127.0.0.1:8767/ . 다른 run은 `--run-dir`만 변경한다. Viewer는 학습을 시작하거나 재평가하지 않는다. HTML은 요청마다 읽으므로 UI 수정은 새로고침으로 적용한다. Python reader 변경은 viewer만 재시작한다. 학습 프로세스를 중단하지 않는다. 포트의 프로세스 command line이 live_training_plot.py인지 확인한 뒤 viewer를 재시작한다.

Live viewer는 서버 기반이며, 사용자 요청 시 기존 템플릿으로 self-contained 정적 snapshot을 보존한다. 2026-10-12 exporter는 `artifacts/full_dataset_static_test_20261012/build_report.py`이다. Snapshot 시각·run 경로를 표시하고 자동 polling을 제거한다. JSON·이미지는 HTML 내부에 포함하여 서버·외부 asset 없이 열리게 한다.

## 화면과 디자인

순서: 제목·상태 → 세로축 선택/Val 재질 선택 → 범위·legend·plateau → Train 큰 canvas → Validation 큰 canvas → Metrics · Status 표 → 데이터 처리 안내.

- Component마다 작은 subplot을 만들지 않는다. Train/Val은 별도의 큰 통합 plot이다.
- 밝은 배경 #f5f5f5, 흰 section, system-ui 15px, main 최대 1600px, section padding20px/radius8px, canvas 높이440px. 정확한 CSS와 간격은 HTML 템플릿을 기준으로 한다.
- 색상 고정: Normal #0072b2, Diffuse #e69f00, Roughness #009e73, Rendering #cc79a7, N/D/R mean #333333, Total objective #d55e00.
- Train raw: opacity .15/선폭1. 100-update 이동평균: opacity1/선폭1.8. Total objective는 점선.
- Val 평가점 사이를 직선으로 연결하고 marker도 유지한다. 전체 ◆, Fabric ●, Leather ■; Leather 연결선은 점선. 연결선은 추세 안내이며 중간 update의 평가값을 생성하지 않는다.
- 표는 native HTML table, 우측 정렬 숫자·tabular-nums, 좌측 지표명, 얇은 행 구분선, 충분한 cell 간격. 좁은 화면은 표 내부 가로 scroll로 겹침을 방지한다.

## 축과 데이터 표시

- X는 **항상 선형 optimizer update**. Log/Linear 선택은 **Y에만** 적용한다.
- Train은 total_step. Val은 기록된 total_step, legacy에 없으면 기록된 stage-local step만 사용한다. 임의 global offset을 만들지 않는다. Legacy 혼용이면 명시한다.
- Train/Val 및 모든 Val category에서 공통 X/Y 범위를 산출한다. 재질 선택에 따라 범위를 다시 맞추지 않는다.
- X는 0부터 최대 기록 step까지. Linear Y는 0부터 전체 최대 loss ×1.05까지. Log Y는 전체 양수 loss의 범위를 포함하는 정수 10제곱 경계.
- Log Y 눈금은 10ⁿ만 표시한다. 시작점 데이터는 유지한다. Linear Y는 선형 눈금이다.
- Log에서 0 이하 값은 표시하지 않고 안내한다. epsilon을 더해 가짜 양수 값을 만들지 않는다. 결측·비유한값을 가짜 값으로 채우지 않는다.
- 각도(°), LR, 시간은 loss plot에 혼합하지 않고 하단 표에 표시한다. 같은 축 범위는 loss 의미나 단위의 동등성을 보장하지 않는다.
- Train 100-update 평균은 원본 update 순서의 trailing mean이다. 초반 100개 미만이면 존재하는 값만 평균한다. Stage별 window를 분리한다.
- 전체 reader history를 보존하며 응답에서 raw(column1)와 평균(column2)을 각각 독립 축약한다. 6000점 초과 시 균등 bucket의 시작·끝·해당 곡선 최소·최대 점을 선택한다. 이미 축약된 초반 데이터를 반복 축약하지 않는다. API `train`과 `train_smooth`가 각 곡선용 데이터이며 양쪽 모두 축 범위에 포함한다. 원본 train.jsonl은 변경하지 않는다.

## 정량 표와 metric 의미

표는 10초 polling으로 갱신한다. Polling은 trainer의 평가/flush 간격을 바꾸지 않는다. Train과 Val은 서로 다른 step에서 얻은 값이므로 각각 step을 표시한다.

1. 실행 상태: status, stage, train total/stage step, Val step/stage, LR, update seconds, invocation elapsed hours, PID, train/status/validation 파일 수정 시각, viewer 갱신 시각, invalid log rows, validation read error, stage별 plateau.
2. Train: total objective, N/D/R mean, Normal/Diffuse/Roughness/Rendering L1의 현재 값과 100-update 평균. Clip 전 gradient norm도 현재 값으로 표시하고 미기록 평균은 —.
3. Val: 최근 평가의 N/D/R mean, Normal/Diffuse/Roughness/Rendering L1, Normal angle. 열은 전체/Fabric/Leather이며 선택된 plot category와 무관하게 모두 보여준다.

Normal loss는 unit normal XYZ의 성분별 absolute error 평균이며 무차원이다. 0.04를 4% 또는 특정 각도라고 해석하지 않는다. Val map/render 지표는 crop 평균 후 재질 동일 가중치 평균이며 category도 같은 규칙이다. Train은 현재 minibatch objective이므로 Val macro 평균과 동일 모집단 집계라고 설명하지 않는다.

Normal angle 표시에는 **재질별 동일 가중치 평균 각도 오차(°) 하나만** 사용한다: 각 재질의 지정 crop 유효 픽셀 평균을 구한 뒤 재질 평균. 전체/Fabric/Leather에 같은 규칙을 적용한다. Pixel-pooled median/P95는 집계 방식이 다르므로 viewer 표에서 제외한다. 기존 로그에 저장된 값은 삭제하지 않는다.

시각은 Asia/Seoul로 날짜·시간을 표시한다. 개별 update timestamp가 없으므로 파일 mtime은 반드시 ‘파일 수정 시각’이라고 표시한다. Browser 갱신 시각을 학습 발생 시각으로 쓰지 않는다. 누락 값은 —, 실제 0은 0. Stage 전환 후 과거 Val은 그 stage/step을 표시하며 현재 train과 같은 시점인 것처럼 보이지 않게 한다.

Plateau는 stage별 10회 연속 validation interval에서 상대 개선 <0.5%인 경우 표시한다. map은 NDR mean, render는 render L1을 사용한다. 화면 지표일 뿐 자동 종료하지 않는다. 자세한 과학 계약은 [학습 문서](FABRIC_LEATHER_TRAINING.md)를 따른다.

## 입력·출력과 검증

읽기 전용 입력: run/train.jsonl, validation.json(history 및 category), status.json. `GET /`는 HTML, `GET /api`는 train/train_smooth/val, latest, latest_validation, status, timestamps, plateau 등 현재 snapshot. 원본 파일 수정·dataset 변경·checkpoint 재추론·SHA 계산은 viewer 작업에 포함하지 않는다.

```powershell
python -m pytest tests/test_live_training_plot.py -q
node tests/check_live_training_plot.cjs
```

[Python 검사](../tests/test_live_training_plot.py): incremental partial/rewrite, missing metrics, independent raw/mean reduction, 최신 Val 및 timestamp, plateau. [JS 검사](../tests/check_live_training_plot.cjs): 공통 범위, 항상 선형 X, log zero/10제곱 눈금, category range 불변, canvas/table 실행, Normal angle 평균만 표시. 변경 후 HTML/API HTTP 응답도 확인한다. DOM stub 검증은 실제 브라우저 visual 검증과 다르며 미수행 visual 검증을 완료라고 기록하지 않는다.

## 이전 규약 대체 사항

- ‘Val 점만 표시’ → 실제 평가점을 직선 연결 + marker 유지.
- 반복 history 축약·raw 극값으로 평균도 축약 → 전체 history에서 두 곡선 독립 축약.
- 하단 angle/LR 문자열 → 자동 갱신 status/train/val table.
- Mean/median/P95 혼합 표시 → 재질 macro mean angle 하나만 표시.
- 포트8765와 예정 run 설명 → 실행 시 지정한 run/port를 사용하며 이 세션 viewer 주소는8767.

Backlinks: [AGENTS.md](../AGENTS.md), [README](../README.md), [학습 계약](FABRIC_LEATHER_TRAINING.md), [HTML](../live_training_plot.html), [서버](../live_training_plot.py).


## 2026-10-11 NDRS합 전체학습 표시 확장

새전체run에서map_aggregation=ndrs_sum이면map_loss를N/D/Rmean으로표시하지않는다. NDRS sum(#9467bd)과Specular(#8c564b)를추가하고NDRmean은별도diagnostic이다. 기존run의지표/색/템플릿은유지한다. Plot의plateau는기존NDR진단표시이고새전체학습종료는실제run status의scheduler/convergence기록을따른다.


## 2026-10-12 정적 학습 + 공식 test 보고

사용자가 같은 contained static HTML 안에 test 정성·정량 기록을 명시 요청했다. 기존 통합 Train/Validation canvas·축·100-update 평균·status 표를 유지하고, 뒤에 test 전체/Fabric/Leather 및 폴더별 native table과 정성 gallery를 붙인다. Live viewer의 동작을 변경하지 않는다.

현재 baseline7m_p01은 memory_paused(total192000)이며 완료·수렴으로 표시하지 않는다. Validation으로 이미 선정된 best-map152000과 best-render185000을 평가 전에 복사해 고정한다. Test 결과로 checkpoint·scheduler·학습 조건을 고르지 않는다. 이후 test를 보고 튜닝하면 이 test는 더 이상 독립 최종 확인 자료가 아님을 다음 보고에 명시해야 한다.

공식 test는 Fabric5/Leather7 폴더 모두와 각 폴더의16crop을 사용한다. 정식 evaluator 기본30조명·seed10000·top-view2.75를 재사용하고 입력near4/far12·gain9·F0.04·unit XYZ·FP32/AMP·TF32 off를 유지한다. Validation novel12와 test30 조명 모집단 차이를 표시한다. Metric은 유효 픽셀 평균→crop 평균→폴더 동일 가중치 평균이며 NDRS합·네 성분 L1·linear-HDR render L1·평균 normal angle(°)을 기록한다. 누락·실패 샘플을 조용히 제외하지 않는다.

정성 표본은 결과와 무관하게 모든12폴더의crop0/고정test조명0이다. GT/best-map/best-render를 같은 범위로 나란히 표시한다. Normal은(XYZ+1)/2, packed map은(x+1)/2이며 render clipping은 표시만 적용한다. S head 표시와 render F0고정을 구분한다. PNG는 data URI로 포함한다. Snapshot JSON·평가 계약·per-crop metric·동결 checkpoint와 검증 기록은 같은 artifact 디렉터리에 별도 보존하지만 HTML 실행에 필요하지 않다. SHA 계산은 하지 않는다.


### 2026-10-12 feature·GT/render ΔE00 확장
사용자 요청으로 같은 정적 HTML에 각 test폴더 crop0의 GT실제입력과 두 예측map의 재렌더 feature trace를 추가한다. GT feature는 기존 prepare/build_features 결과와 exact equality를 확인하며 예측 relation은 네트워크 직접출력이 아닌 재렌더 진단으로 명시한다. Near/far·9IF·Kθ(9IF)의 raw/clip·signed near차이·Kθ/c_raw/c/denominator·q/logq·relation/log-relation·non-saturated/valid mask·log appearance를 표시한다. K_L=9는 숫자로 표시하며 기존 source·거리·손실은 변경하지 않는다. RGB [0,1], relation/mask gray [0,1], scalar raw min/max와 눈금, feature [-1,1] encoding을 명시한다.

GT/예측 render ΔE00은 공식test12폴더×16crop×30고정조명 전체에서 계산한다. Near/far colocated4/12의 GT/예측 재렌더 ΔE00도 별도 집계한다. 기존 capture_processing/difference_visualization.py를 재사용하며 FP32 unclipped linear RGB→CPU FP64 XYZ D65/Lab2°/CIEDE2000(kL=kC=kH=1), gamma/노출/PNG재읽기 없이 계산한다. 유효 픽셀→조명→crop→폴더macro를 사용하고 기존 L1과 대체하지 않는다. 대표crop0 heatmap은 sample·지표별두checkpoint 공통range, P98고립꼬리 표시만clip, colorbar/단위/범위/clipping비율을 포함한다. Normal angle·diffuse/F0 physicalRGB RMS·roughness signed차이도 같은규칙이다. 다른sample의색강도를절대비교하지않는다. 새 script는 artifacts/full_dataset_static_test_20261012/extend_features.py이며 기존동결weights만읽고새학습/새선택은하지않는다.

유효 영역 밖 픽셀은 relation 검증에서 실제 feature의 mask 연산을 그대로 적용하고 ΔE00·map error 집계에서 제외한다. crop별 valid fraction은 JSON에 남기고 유효영역 mask도 gallery에 표시한다. 색차 계산은 같은 CPU FP64 함수를2조명씩4worker 이내로 실행하며 기존serial 결과와 일치성을 검사했다. 폴더 단위progress는 재개 전후 첫crop의색차 일치를 확인한다.
