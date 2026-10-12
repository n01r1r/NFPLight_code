# 7.61M ReduceLROnPlateau 후속 계약

[Scheduler 정책](TRAINING_SCHEDULER_POLICY.md) · [현재 상태](CURRENT_TRAINING_STATE.md) · [새 구현](../experiments/plateau_followup/README.md)

## 승인과 목적

2026-10-09 사용자는 Astra low의 권고 뒤 `Go`를 지시했다. 목표는 validation NDR이 가장 좋은 map 확보다. 단일 seed19, 7.61M D4+dropout p=.1을 처음부터 학습하고 기존 동일 조건과 비교한다. 사용자는 예산을 넉넉하게 두고 메모리를 채우지 않는 범위에서 체크포인트로 연장할 수 있도록 요청했다. 15.09M 자동 학습은 예약하지 않는다.

이 문서의 숫자는 승인된 방향을 구현하는 운영 설정이다. Council 숫자의 최적성이나 scheduler 개선 효과가 검증됐다는 뜻은 아니다. 기존 Council의 20k/5k 예산, floor10/20 논의와 과거 고정 LR 결과는 이력으로 보존한다.

## 고정 비교 조건

- 기존 `artifacts/regularization_probe_20261009_checked/seed19_d4_dropout`이 같은 조건의 대조다. 기존 baseline extension과 15M은 참고이며 dropout·용량 차이를 섞지 않는다.
- TwoBranchNet 7,609,834 parameters, width32/encoder[2,2,4]/middle4/decoder[2,2,2], seed19, scratch, D4 true, NAFBlock dropout .1, batch4.
- 기존 material·crop·manifest 선택, FP32, AMP/TF32 off, raw_calibrated_gain9_v3, near4/far12/gain9/F0=.04, train40/novel12 조명과 L1 목적함수를 유지한다. denoiser·공식 test는 사용하지 않는다.
- map은 N/D/R 동일 평균 L1, render 단계는 map loss + 1×render loss다. validation은 dropout off이며 평가 전후 train RNG·mode를 복원한다.
- sample/D4 RNG는 스트림으로 유지한다. map 예산이 달라지면 render의 표본 순서 시작점도 달라지므로 과거 run과 render minibatch까지 짝지었다고 주장하지 않는다.

## Scheduler와 단계 선택

|항목|map|map + render|
|---|---|---|
|monitor|heldout validation NDR mean|heldout novel-light render L1|
|초기 LR / 최저 LR|5e-4 / 1e-5|1e-4 / 1e-6|
|최초 예산|50,000 updates|15,000 updates|
|평가 / 체크포인트|100 updates|100 updates|
|scheduler|mode=min, factor=.5, patience=5, threshold=.005, threshold_mode=rel, cooldown=2, eps=1e-8|동일|
|종료 운영 기준|LR floor 도달 뒤 실제 floor LR에서 20회 평가 동안 의미 개선 없음|동일|

patience5는 기준 best 설정 후 bad observation 6회에서 LR을 낮춘다. LR을 낮출 때 단계 내부 Adam을 재생성하지 않는다. floor에 처음 도달한 평가는 대기 counter0이며 이후 floor LR로 학습한 평가부터 센다. meaningful best는 마지막 상대 .5% 개선값 기준이며 raw best와 분리한다. 작은 개선이 누적돼 .5%를 넘으면 대기 counter를 초기화한다. 20회는 2,000 updates의 운영 예산이며 수렴 증명이 아니다.

Map의 raw 최저 NDR마다 best_map을 저장한다. map 종료 시 마지막 map의 model/Adam/scheduler/RNG·counter를 map_stage_end로 보존한 뒤 best_map 가중치를 복원하여 fresh Adam·fresh scheduler로 render를 시작한다. 선택용 best-map 가중치와 이어 학습할 마지막 map 상태는 다르다.

Render 선택은 best-map NDR 대비 상대 악화≤2%를 만족하는 최저 novel-light L1 checkpoint다. 허용 후보가 없으면 best_map을 사용한다. 2%는 운영상 보호 폭이지 물리 정확도 기준이 아니다. 최종 선택 가중치와 마지막 학습 상태를 별도 보존한다.

## 예산과 연장

상한 도달은 `budget_paused`이며 map→render 전환이나 수렴 판정으로 쓰지 않는다. `run --resume --extend-map-to N` 또는 `--extend-render-to N`으로 상한을 단조 증가한다. base config·identity를 바꾸지 않고 연장 ledger를 별도 기록한다.

Plateau 종료 후 추가 탐색은 보존된 stage-end 전체 상태에서 `--fork-map-from` 또는 `--fork-render-from`으로 새 출력 lineage를 만든다. floor 대기 counter를0으로 재개해 추가 평가 여유를 제공하며 optimizer·scheduler·RNG와 단계 내 update를 이어간다. 이미 render에 진입한 뒤 map을 연장하면 원래 render 결과는 보존하고 새 lineage에서 변경된 best-map으로 render를 다시 시작한다. 기존 결과를 덮어쓰지 않는다.

## 저장·메모리·재현성

last/previous, best_map/best_render, stage_end처럼 고정 개수의 체크포인트를 유지한다. 임시 파일 작성·원자 교체로 마지막 committed 상태를 보존한다. 평가마다 checkpoint 파일을 무한히 추가하지 않는다. CPU로 가중치를 저장·로드하여 두 GPU 모델을 보유하지 않는다.

model, optimizer, scheduler, torch CPU/CUDA·numpy·python·sample·D4 RNG, phase, stage/total steps, 평가·종료 counter, raw/meaningful best, 선택 참조를 저장한다. own checkpoint는 weights_only=True로 읽을 수 있는 tensor·primitive 상태로 저장하고 과거 checkpoint pickle은 로드하지 않는다. identity는 path/size/mtime_ns이며 SHA-256은 실행하지 않는다.

GPU 평가 cache는 CPU로 두고 한 slot씩 장치로 옮긴다. Bank cache도 유한 크기로 유지한다. 로그는 파일에 기록하며 GPU tensor나 모든 미래 sample order를 누적 보유하지 않는다. 실제 GPU smoke에서 CPU/RAM/GPU 사용량과 map·render 양쪽 경로를 확인한다.

## 검증과 결과 주장

작은 결정적 CPU 검사로 LR 감소·resume 연속성, dropout off·평가 RNG 보존, best-map 복원, render guard/fallback, budget pause/extension, 단계 종료 후 fork, 유한값 거부, 제한된 checkpoint 개수를 검사한다. 실제 학습은 검증 뒤 새 artifact에서 시작한다. 검증 결과와 실행 기록은 해당 artifact로 연결한다.

학습 정책 전체의 비교이며 scheduler 하나의 인과 효과나 일반화 개선으로 해석하지 않는다. 기존8개 heldout 재질은 validation이며 독립 test가 아니다. 상한·운영 plateau 종료는 물리적 수렴 증명이 아니다.
