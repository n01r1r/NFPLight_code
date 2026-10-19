# 작업 지침
사용자 설명은 한국어로 간결하게 작성하고 `ponytail:ponytail`을 항상 적용한다. 독립 학습 세션은 [scheduler 정책](docs/TRAINING_SCHEDULER_POLICY.md)·[현재 상태](docs/CURRENT_TRAINING_STATE.md)를 먼저 읽어 확정·제안·이력을 구분한다. 학습 HTML은 [표시 규약](docs/TRAINING_HTML_REPORT_CONTRACT.md)을 먼저 읽고 기존 템플릿을 재사용한다.
불확실한 결정은 임의 판단하지 않고 `C:/Users/hdy/.agents/skills/grill-me/SKILL.md`에 따라 권장안을 포함해 한 번에 하나씩 질문한다. 코드 탐색으로 확인할 사실은 먼저 조사한다.
연구·수치·ML 변경은 `C:/Users/hdy/.codex/policies/research-code.md`와 `working-preferences.md`를 따른다.
실제 fabric DNG 비교는 [요구사항](docs/CAPTURE_REQUIREMENTS.md)과 `run_fabric_capture.py`를 사용하며 FP32만 허용하고 기본값·현재 실험도 FP32다.
실측 capture 실험은 새 data의 5개 10/30cm 묶음이며 활성 photometry 조건은 `both=(D-B)/(W-B)` 하나다. `white_only`·`neither`는 폐지했고 입력을 거부한다. 공통 마커 RMS·crop·무늬 정합 게이트는 통과 시 5장 평균, 실패 시 near_00/far_00 사용이다. 기본 경로는 저자 원본 net_g_real.pth·net_g_syn.pth만 허용하며, 사용자 승인 Fabric 재비교는 요구사항 §1.1의 별도 경로를 사용한다. denoiser·denoising image 사용을 금지한다.
과거 PNG/FP32 실험·report 파생 스크립트를 활성 코드에 다시 추가하지 않는다.
학습·합성 validation/test는 raw_calibrated_gain9_v3·near4/far12·고정 gain9이며 실측 inference와 엄격히 구분한다. 실측 inference는 중앙20×20 RGB 평균 near/far 비율(author_patch20)만 사용하고 학습 gain을 적용하지 않는다. 새 FP32 추론·보고와 과거 결과 보존은 요구사항 §1.4를 따른다.
`.git`·`checkpoints`·`results`·dataset junction의 외부 대상을 재귀 삭제하거나 이동하지 않는다.
