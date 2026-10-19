# 작업 지침
한국어로 간결하게 설명하고 ponytail을 적용한다. 범위·호환성·스타일은 [재작성 계약](docs/CLEAN_CORE_REWRITE.md)을 따른다.
연구·수치 변경은 C:/Users/hdy/.codex/policies/research-code.md와 working-preferences.md를 따른다. 불확실한 과학적 결정만 grill-me로 한 번에 하나씩 확인한다.
기본 구조는 configs/7m_spatial.json이며 원본·7M profile을 명시적으로 구분한다. Weight와 입력·decoder·Adam·scheduler·RNG의 호환성을 함께 검증한다.
FP32·학습 gain9·실측 patch20·both·단일 crop·검토 후 평균/첫 프레임·denoiser 금지를 유지한다.
Formatter·자동 인덱싱·SHA-256 검증을 실행하지 않는다. Python 검증 CLI는 assert에 의존하지 않는다.
실제 연구 학습 개시·기존 학습 중단·산출물 덮어쓰기는 별도 사용자 권한이 필요하다. 작은 CPU fixture와 연구 학습을 구분한다.
`.git`·`checkpoints`·`results`·dataset junction 외부 대상을 재귀 삭제하거나 이동하지 않는다.
