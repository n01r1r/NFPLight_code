# Clean core 계약

`nfplight/`가 모델·입력 특징·렌더링·데이터·학습·checkpoint의 단일 구현이다. Notebook은 clone한 같은 모듈을 읽고 호출한다. 기본 모델은 없으며 weight와 실험마다 profile을 명시한다. [Colab 계약](COLAB.md)을 따른다.

공개 입력은 교정 완료 `[B,6,256,256]` 선형 RGB FP32와 bool mask다. 특징21ch, 출력 N3/D3/R1/S3 raw tanh를 profile에 따라 unit XYZ normal로 decode한다. 저자 real33ch/syn21ch inference profile과 fresh 학습 profile을 구분한다. Synthetic 학습은 near4/far12·gain9·무양자화 관측·FP32·AMP/TF32 OFF다. Map은 unit XYZ N/D/R/S L1 sum, render는 map sum+render L1이다. Specular target/render F0는0.04다. 데이터 구성·집계·seed·Adam/scheduler·sampling·augmentation을 임의로 바꾸지 않는다.

원 NFPLight의 NAFBlock·4단 stride2 downsampling·encoder[2,2,4,8]·middle12·decoder[2,2,2,2]·parameter 등록 순서를 유지한다. Nested 후보는 명시적 architecture=nested로 선택하며 최종 full-resolution 출력만 supervision한다. DS OFF다. Baseline은 원 NFPLight+MatSynth fabric/leather subset 무양자화 학습 weight만 뜻한다.

Full checkpoint의 model·Adam·scheduler·모든 RNG·sampler·단계·best·floor counter를 함께 복원한다. Model-only weight나 smoke checkpoint는 연구 exact resume 입력이 아니다. Profile/runtime/환경/데이터 identity가 다르면 거부한다. Cap은 pause이며 자동 재개·강제 단계 전환하지 않는다. 새 학습은 사용자 실행과 새 출력 폴더를 요구한다.

실측은 black/white both, 중앙20×20 author_patch20 gain, 검토 후 평균/첫 프레임 선택을 유지한다. Denoiser는 사용하지 않는다. 표시 PNG를 FP32 수치 평가 입력으로 사용하거나 8bit 표시만으로 artifact/degradation을 주장하지 않는다.

기존 Module/ModuleList·명시적 residual·dim/depth 이름을 따른다. 잘못된 입력/nonfinite/호환성 충돌은 예외로 거부한다. 검증 CLI는 assert에 의존하지 않는다. Formatter·자동 인덱싱·SHA-256은 실행하지 않는다. 데이터·보고서·archive·resume checkpoint를 배포하지 않으며 저자 원본 inference weight 외 연구 weight는 포함하지 않는다. .git·checkpoints·results·dataset junction 외부 대상은 재귀 삭제하거나 이동하지 않는다.
