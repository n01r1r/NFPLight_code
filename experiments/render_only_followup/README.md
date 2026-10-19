# Render-only 후속

[승인 계약](../../artifacts/render_only_followup_20261019/EXPERIMENT_CONTRACT_KO.md)에 따라 기존 A–C를 유지하고 C의 학습·합성·실측 보고 완료 뒤 D를 실행한다. 기존 trainer를 수정하지 않고 Pipeline/Engine/초기화/평가/보고기를 재사용한다.

```powershell
python -m experiments.render_only_followup.run prepare
python -m experiments.render_only_followup.run watch
```

watch는 새 학습 허가를 받기 위한 명령이 아니며 이미 승인된 한 번의 후속 queue다. 중복 실행·실패한 학습 자동 재개를 금지한다. prepare 후 source identity를 고정한다. smoke2→train10k→공동 비교→합성 report→CPU 실측 순서이며, 가로 비교 HTML은 완료한 조건만 갱신한다. 대기 중 CUDA를 초기화하지 않는다.

출력은 `artifacts/render_only_followup_20261019/{queue.json,run/,COMPARISON_KO.md}`이며 C/D 완료 전에는 성능 결과가 없다. 기존 실측 비교는 `artifacts/relmap_three_way_real_20261019/index.html`이다.

```powershell
python -m pytest tests/test_render_only_followup.py tests/test_render_weight_followup.py -q
```
