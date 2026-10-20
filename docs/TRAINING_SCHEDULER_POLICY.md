# 학습 scheduler와 운영 종료

현재 C 연장은 완료됐다. 이 문서는 실행을 재개하는 지시가 아니라 보존된 정책이다. [현재 상태와 결과](CURRENT_TRAINING_STATE.md)를 먼저 확인한다.

승인된 C exact-state 연장은 render10k/total145k의 weights·Adam·scheduler·전체 RNG·best·floor counter를 그대로 복원했다. ReduceLROnPlateau 설정은 initial LR1e-5, floor1e-6, factor0.5, patience5, relative threshold0.005, cooldown2, eval1000이다. Floor에서 의미 있는 개선 없는20회 평가를 운영 종료 조건으로 사용했고 유한 학습 cap은 없었다.

Render65k/total200k에서 운영 종료했으며 선정 best는64k, 마지막 LR은1e-6이다. 이 운영 종료를 수렴 증명으로 해석하지 않는다. 기존 NDRS합×1.02 선정보호와 validation 선정 규약을 유지했다. Test·실측으로 checkpoint를 선택하거나 임의 성분 허용폭을 추가하지 않았다.

새 실행은 별도 사용자 승인과 새 출력 경로가 필요하다. Exact resume는 model-only best가 아니라 전체 last 상태와 일치하는 runtime·환경·입력 identity를 요구한다. 수동 LR 변경·새 seed·smoke의 연구 학습 전환·자동 재시작을 허용하지 않는다. Memory/disk/error pause는 상태를 보존하고 조사한다.

원본 [실험 계약](../artifacts/C_convergence_followup_20261019/EXPERIMENT_CONTRACT_KO.md)과 [후처리 계약](../artifacts/C_convergence_followup_20261019/POSTPROCESS_CONTRACT_KO.md)을 보존했다. 과거 전용 runner는 archive에 있으며 활성 CLI의 재개 조건은 [사용법](USAGE_KO.md)을 따른다.
