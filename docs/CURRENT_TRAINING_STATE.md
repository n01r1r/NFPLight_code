# 현재 모델과 실험 상태

2026-10-20. 활성 코드는 **main**, 작업 폴더는 **C:/workspace/NFPLight_code**다. C 연장은 **render65k / total200k에서 정상 운영 종료**했고 validation 선정 best는 **render64k**다. 학습을 재개하지 않는다. 감시 heartbeat `c-optimizer-rng`는 PAUSED다. A/B는 현재 후보에서 제외했고 D는 failed/stopped 상태를 보존한다.

## C 연장의 실제 결과

동일164재질·novel12 validation에서 보존 C10k best와 연장64k best를 비교했다. 아래 값은 낮을수록 좋다.

|지표|C10k best|C64k best|상대 변화|
|---|---:|---:|---:|
|Render L1|0.003274934045|0.003220241612|-1.67003%|
|N|0.03933809901|0.03928391212|-0.13775%|
|D|0.02059376114|0.02043174751|-0.78671%|
|R|0.06067602248|0.06060226958|-0.12155%|
|S|0.0001905174376|0.0001797945457|-5.62830%|
|NDRS합|0.1207984001|0.1204977238|-0.24891%|

판정은 **이 validation 전체 평균에서 render 개선과 모든 N/D/R/S 비악화**다. Fabric N(+0.10947%)·R(+0.08671%)은 악화했고 Leather는 모든 성분이 감소했다. Source group의 render는89개 개선·1개 악화, NDRS합은66개 개선·24개 악화다. Clean map135k 대비 render는5.43941% 감소했으나 R은0.23878% 증가했다. 마지막65k endpoint와 선정64k를 구분한다.

GT 없는 실측5묶음·near/far10case는 같은 입력·정합·색변환·표시범위, both/author_patch20/CPU FP32로 비교했다. 관측 재구성 L1은6/10, ΔE00은7/10 개선했고 평균은 각각1.75256%,0.65953% 감소했다. Near는 모두 개선, far는 혼재했다. 물성 GT 정확도·일괄 정성 우위·통계적 유의성·독립 일반화·수렴 증명으로 해석하지 않는다.

## 결과와 사용할 weight

- [전체 비교·절대 변화·범주·endpoint 해석](../artifacts/C_convergence_followup_20261019/FINAL_COMPARISON_KO.md) / [비교 JSON](../artifacts/C_convergence_followup_20261019/FINAL_COMPARISON.json)
- [실측 통합 HTML: C10k best와 C 연장 best 인접 열](../artifacts/relmap_three_way_real_20261019/index.html)
- [합성 보고](../artifacts/C_convergence_followup_20261019/run/report/index.html) / [실측 보고](../artifacts/C_convergence_followup_20261019/run/real_report/index.html)
- 최종 weight: [net_g_selected_fp32.pth](../artifacts/C_convergence_followup_20261019/run/net_g_selected_fp32.pth), 명시 profile [7m_c.json](../configs/7m_c.json)
- [CPU/CUDA 최종 호환성](C_BEST64K_MAIN_HANDOFF.md) / [모든 로컬 산출물 안내](../artifacts/INDEX_KO.md)

원래 worker는 학습 완료 뒤 보고서 API의 status key 누락으로 실패했다. 학습 실패와 구분하며 adapter가 원래 JSON·오류·부분 보고를 보존하고 보고만 완료했다. 원본 C10k best와 연장 best를 모두 보존했다. Browser file URL 정책으로 실제 HTML 브라우저 배치는 확인하지 못했으며379자산·366PNG·JS syntax 검사는 통과했다.
