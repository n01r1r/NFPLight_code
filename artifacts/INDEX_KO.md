# 로컬 산출물 안내

2026-10-20 기준 최신 안내다. 이 안내만 Git에 기록하며 나머지 산출물은 Git에서 제외한다. 실제 데이터·weight·수치 JSON·HTML·과거 보고서 원본을 보존한다. 이전 날짜의 status/queue/문서는 당시 실행 기록이다. 현재 코드·운영 상태는 [README](../README.md)와 [현재 상태](../docs/CURRENT_TRAINING_STATE.md)를 따른다. 새 clone에서는 로컬 산출물을 별도로 준비해야 아래 링크를 열 수 있다.

|목적|현재 참조할 결과|
|---|---|
|C10k 대비 연장64k 유효성|[최종 비교](C_convergence_followup_20261019/FINAL_COMPARISON_KO.md) / [JSON](C_convergence_followup_20261019/FINAL_COMPARISON.json)|
|C10k·연장 best 실측 나란히 비교|[통합 HTML](relmap_three_way_real_20261019/index.html)|
|연장 best 합성·실측 보고|[합성](C_convergence_followup_20261019/run/report/index.html) / [실측](C_convergence_followup_20261019/run/real_report/index.html)|
|사용할 C64k weight|[FP32 export](C_convergence_followup_20261019/run/net_g_selected_fp32.pth) / [best checkpoint](C_convergence_followup_20261019/run/best_render.pt)|
|기존 코드와 최종 weight 호환성|[검증 JSON](C_best64k_clean_main_20261020/verification.json) / [당시 결과](C_best64k_clean_main_20261020/RESULTS_KO.md)|
|현재 main·branch 정리·보존 확인|[최신 전환 결과](clean_main_cleanup_20261020/RESULTS_KO.md) / [검증 JSON](clean_main_cleanup_20261020/verification.json)|
|미커밋 변경과 이력 복구|[복구 방법](../docs/C_BEST64K_MAIN_HANDOFF.md) / [최신 bundle](clean_main_cleanup_20261020/repository_with_latest_stash.bundle) / [원본13파일 ZIP](clean_main_cleanup_20261020/latest_worktree.zip)|

C 연장은65k 운영 종료, validation best64k다. 전체 평균에서 render1.67% 개선과 모든 N/D/R/S 비악화를 확인했지만 Fabric N/R 및 일부 source group 손해는 남아 있다. 실측은 GT 없는 관측 재구성·정성 비교다. 과거 보고서 생성 오류는 학습 완료 뒤의 오류였고 adapter가 보고만 완료했다.

기존 C_best64k 전환 보고의 “C checkout은 legacy branch” 상태는 이후 main 통일로 대체됐다. 과거 보고서·script는 감사용이며 현재 main에서 그대로 재실행하는 사용법이 아니다. 다른 날짜의 결과를 새 입력 robust_target과 같은 조건으로 간주하지 않는다. 원본 artifact를 고치거나 삭제해 최신 상태처럼 보이게 만들지 않는다.
