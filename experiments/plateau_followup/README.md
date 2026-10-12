# Seed19 7.61M ReduceLROnPlateau follow-up

먼저 [scheduler 정책](../../docs/TRAINING_SCHEDULER_POLICY.md), [현재 상태](../../docs/CURRENT_TRAINING_STATE.md), [승인된 후속 계약](../../docs/PLATEAU_FOLLOWUP.md)을 읽는다. 과거 regularization lineage는 고정 LR 이력이며 이 패키지는 새 scratch 실행만 준비한다. SHA-256 또는 역사적 checkpoint pickle 로드는 실행하지 않는다.

```powershell
python -m experiments.plateau_followup.run prepare
python -m experiments.plateau_followup.run smoke --out artifacts/plateau_followup_smoke_20261009
python -m experiments.plateau_followup.run run
python -m experiments.plateau_followup.run status
python -m experiments.plateau_followup.run run --resume
python -m experiments.plateau_followup.run run --resume --extend-map-to 60000
```

고정 seed19, 7,609,834 parameters, D4+dropout0.1, batch4, FP32·AMP/TF32 off. `regularization_probe_20261009_checked/seed19_d4_dropout`의 16 train/8 unseen-crop/8 heldout slots, 원래 Adam/L1/renderer/40 train lights/12 novel lights를 유지한다. Map validation NDR, render validation novel-render L1을 각각 관측한다. 8개 heldout 재질은 반복 관측한 validation이며 독립 test가 아니다.

Scheduler: factor.5, patience5, threshold rel.005, cooldown2, eps1e-8. Map LR5e-4→1e-5, render LR1e-4→1e-6. Floor 최초 도달 eval은0이며 이후 floor LR로 학습한 eval에서 meaningful best 대비 누적 상대.005 개선이 없는20회를 센다. 이는 운영 종료 기준이며 수학적 수렴 증명이 아니다. Raw minimum은 scheduler/meaningful threshold와 별도로 선택한다. Map 종료 때 raw best-map 가중치를 복원하고 render Adam/scheduler를 새로 만든다. Render stage0 baseline도 후보이며 NDR≤1.02×best-map NDR를 만족하는 raw lowest render L1을 선택한다. 적격 후보가 없으면 best-map이다.

초기 map50k/render15k cap은 자동 단계 전환·수렴 선언 없이 `budget_paused`로 멈춘다. `--resume --extend-map-to`/`--extend-render-to`는100단위로 cap을 엄격히 늘리고 `budget_ledger.json`에 append한다. 실행 config는 수정하지 않는다. RAM 여유1GiB 미만 또는 CUDA 사용 가능 여유512MiB 미만이면 마지막 commit으로 rollback하고 `memory_paused`로 멈춘다. 실제 OOM도 마지막 commit과 train log를 복구한다. 재개 전 다른 작업의 메모리 사용을 확인한다.

Plateau 종료 뒤 추가 탐색은 원본 완료 기록을 유지하고 새 output lineage로 fork한다:

```powershell
python -m experiments.plateau_followup.run run --out artifacts/plateau_followup_map_extension --fork-map-from artifacts/plateau_followup_7m_seed19_20261009/map_stage_end.pt --extend-map-to 60000
python -m experiments.plateau_followup.run run --out artifacts/plateau_followup_render_extension --fork-render-from artifacts/plateau_followup_7m_seed19_20261009/render_stage_end.pt --extend-render-to 20000
```

Fork는 원본 source/data/runtime 계약을 검사하며 마지막 model/Adam/scheduler/모든 RNG/raw·meaningful best를 CPU에서 복구한다. Floor 대기 counter만0으로 초기화하여 추가20개 floor eval을 허용한다. Map fork는 마지막 map에서 재개하고 다시 best-map을 선택한 뒤 render를 새로 시작한다. 원본 render/완료 파일은 변경하지 않는다. Render fork는 `render_stage_end.pt` 또는 이 패키지의 render 단계 `last.pt`를 받는다. 원본 checkpoint가 cap 이상의 step이면 cap을 명시적으로 늘린다.

`last.pt`는 마지막 학습 상태이며 `previous.pt`는 바로 이전 commit이다. `best_map.pt`/`best_render.pt`/`selected.pt`는 선택 가중치다. `map_stage_end.pt`는 best 복원 전 마지막 map/Adam 상태를 보존한다. `render_stage_end.pt`는 마지막 render 상태다. Per-eval checkpoint는 생성하지 않는다. 별도 CPU 평가 cache를 보유하고 GPU에는 한 평가 slot만 올린다. Train target은 CPU에 보관하며 sample stream은 cap 크기의 배열을 만들지 않는다. 초기 map sample/D4 prefix는 기존 seed19와 같다. Render sample offset은 실제 map 종료에 따라 바뀌므로 과거 run과 paired render sample이라고 주장하지 않는다.

`evaluation.json`, 기존 viewer 호환 `validation.json`, `train.jsonl`, `status.json`, `complete.json`을 기록한다. Status의 PID 생성 identity와 exclusive lock은 중복 실행을 막는다. `execution_config.json`은 canonical path/size/mtime identity와 input scope를 보존한다. Own checkpoint만 `weights_only=True`로 CPU 로드한다. 완료 직후 중단으로 파생 완료 파일이 빠져도 `--resume`이 추가 update 없이 복구한다.

```powershell
python -m pytest tests/test_plateau_followup.py -q
```

CPU tiny-engine 검증은 exact resume/LR 감소/단계 경계/RNG neutral eval/dropout off/raw-best 복원/guard·fallback/상한·monotone extension/별도 fork/완료 파일 복구/메모리 pause/checkpoint 수 상한을 확인한다. GPU smoke는 각 stage1 update와 고정32 slots 평가만 확인하며 성능 개선의 증거가 아니다.
