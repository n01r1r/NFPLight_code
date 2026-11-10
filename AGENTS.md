# 작업 지침
한국어로 간결하게 설명하고 ponytail을 적용한다. 범위·호환성·스타일은 docs/CLEAN_CORE_REWRITE.md를 따른다.
기본 모델은 없으며 weight별 profile을 명시한다. Colab clone·Drive·NFPLight/NAF U-Net++는 docs/COLAB.md를 따른다. 공통 코어는 nfplight/에 유지한다.
FP32·학습 gain9·실측 author_patch20·both·검토 후 평균/첫 프레임·denoiser 금지를 유지한다. 모델 연산은 GPU로 수행한다.
연구 목적과 수치 계약을 임의 변경하지 않는다. 새 연구 학습·기존 실행 중단·산출물 덮어쓰기는 사용자 승인이 필요하다.
Weight와 입력·decoder·Adam·scheduler·전체 RNG의 호환성을 검증한다. 검증 CLI는 assert에 의존하지 않는다.
Formatter·자동 인덱싱·SHA-256 검증을 실행하지 않는다.
데이터·보고서·archive·resume checkpoint는 배포하지 않고 저자 원본 inference weight2개 외 연구 weight는 포함하지 않는다.
.git·checkpoints·results·dataset junction 외부 대상을 재귀 삭제하거나 이동하지 않는다.
