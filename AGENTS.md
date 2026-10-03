# 작업 지침
사용자 설명은 한국어로 간결하게 작성하고 `ponytail:ponytail` 지침을 항상 적용한다.
불확실한 결정은 임의 판단하지 않고 `C:/Users/hdy/.agents/skills/grill-me/SKILL.md`에 따라 권장안을 포함해 한 번에 하나씩 질문한다. 코드 탐색으로 확인할 사실은 먼저 조사한다.
연구·수치·ML 변경은 `C:/Users/hdy/.codex/policies/research-code.md`와 `working-preferences.md`를 따른다.
실제 fabric DNG 비교는 [요구사항](docs/CAPTURE_REQUIREMENTS.md)과 `run_fabric_capture.py`를 사용하며 FP32만 허용하고 기본값·현재 실험도 FP32다.
현재 실험은 새 data의 5개 10/30cm 묶음이며 활성 photometry 조건은 `both=(D-B)/(W-B)` 하나다. `white_only`·`neither`는 폐지했고 입력을 거부한다. 공통 마커 RMS·crop·무늬 정합 게이트는 통과 시 5장 평균, 실패 시 near_00/far_00 사용이다. 저자 원본 net_g_real.pth·net_g_syn.pth만 추론에 허용하며 denoiser·denoising image 사용을 금지한다.
과거 PNG/FP32 실험·report 파생 스크립트를 활성 코드에 다시 추가하지 않는다.
새 추론은 지정 artifact 디렉터리에 FP32로 새 계산한다. 기존 best_render.pth 수치 결과는 보존하되 현재 report 비교에서 제외하고 재추론하지 않는다.
`.git`·`checkpoints`·`results`·dataset junction의 외부 대상을 재귀 삭제하거나 이동하지 않는다.
