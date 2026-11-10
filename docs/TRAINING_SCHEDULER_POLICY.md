# 학습 scheduler와 종료

현재 정책은 [training_noquant.json](../configs/training_noquant.json)과 [공통 trainer](../nfplight/training.py)가 결정한다. [Colab 안내](COLAB.md)를 따른다.

Map은 Adam LR5e-4/floor1e-5, render는 새 Adam LR1e-4/floor1e-6을 사용한다. ReduceLROnPlateau는 factor0.5, patience5, relative threshold0.005, cooldown2, eval1000이다. Map은 heldout NDRS sum, render는 heldout novel-light render L1을 감시한다. LR floor의 의미 있는 개선 없음20회로 단계를 종료한다. Map 종료 시 best-map을 읽고 render optimizer/scheduler를 새로 만든다.

Map400k/render100k cap은 pause이며 강제 단계 전환이나 수렴 증명이 아니다. Validation 선정과 render NDR guard1.02를 유지한다. Train과 validation의 집계는 다르며 plot은 단계별 raw와100-update 평균을 따로 표시한다.

Exact resume는 full last checkpoint·일치하는 profile/runtime/환경/입력 identity를 요구한다. Model-only best, 새 seed, 수동 LR 변경, smoke의 연구 학습 전환, 자동 재시작은 허용하지 않는다. 마지막 저장 이후의 미저장 update는 복구하지 않는다.
