# Clean core rewrite 계약

## 확정 범위

- 브랜치: codex/clean-core-rewrite. 별도 worktree에서 작업하며 원래 작업 폴더와 미커밋 변경은 보존한다.
- 모델·입력 처리·추론·학습 코어를 재작성한다. 의미별 모듈과 최소 깊이의 폴더를 사용한다.
- 새 API·CLI로 통합한다. 기존 import·CLI 호환 facade는 남기지 않고 전환 방법을 문서화한다.
- 과거 실험·보고서 생성 코드는 Git 이력과 원래 작업 폴더에 보존하고 새 브랜치에서 제거한다. 실제 데이터·weight·결과 및 junction 외부 대상은 삭제하거나 이동하지 않는다.
- 필수 weight: 전달용 7M relmap 수정 전·수정 후·C와 현재 C 후속. 저자 원본 real/syn도 지원 범위에 포함한다. 그 밖의 실험 weight는 제외한다.
- Weight별 원래 feature/relmap 규약을 config에 명시한다. 부족하거나 충돌하는 metadata를 추측하지 않고 거부한다. 학습 gain9와 실측 author_patch20을 분리한다.
- 기존 재개 가능 checkpoint의 model·Adam·scheduler·전체 RNG·sampler·학습 단계·평가 및 종료 상태를 보존한다. 기존 규약상 재개 금지는 유지한다.
- 이번 재작성은 실제 연구 학습 개시·현재 학습 중단·기존 결과 덮어쓰기 권한을 포함하지 않는다.

## 코드 스타일

- 기존 lucidrains 관례. 의미별 모듈, 읽기 쉬운 함수, 명시적 실패 계약. 실제 필요가 없는 추상화·설정·의존성은 추가하지 않는다.
- 모델은 from torch.nn import Module, ModuleList, 명시적 residual 합, dim/depth/to_* 관례를 따른다. 기존 state_dict의 이름·등록 순서는 로더 및 optimizer 호환성 검증 대상이다.
- Tensor layout 변경은 einops rearrange/einsum을 사용하고 shape·단위·순서를 명시한다. 수치 의미를 바꾸는 broadcasting을 숨기지 않는다.
- 기본은 동적 typing. Typing 사용 시 future annotations와 X | None을 따른다.
- Kwarg는 name = value, 문자열은 작은따옴표. 주석은 필요한 이유·계약만 기록한다. 범위 밖 스타일 정리를 하지 않는다.
- Import 그룹: stdlib / torch / einops / 기타 / 프로젝트. 패키지 내부 import는 absolute. __init__.py는 명시적 재노출만 한다.
- 연구 입력·nonfinite·영속 데이터 오류는 ValueError 등으로 거부한다. 검증 CLI는 assert에 의존하지 않는다.
- Formatter·자동 인덱싱·SHA-256 검증을 실행하지 않는다. 과거 외부 저장소 관찰·추론·제안 목록을 현재 지침에 포함하지 않는다.

## 수용 검사

- 실제 대상 checkpoint strict load, key/shape/dtype와 모든 tensor 대응 검증.
- 동일 입력·동일 FP32 환경에서 기존 기준과 feature·raw prediction·decoded prediction 비교. 비교한 profile과 환경 및 오차를 기록한다.
- 작은 결정적 pytest로 입력 shape/range/mask/nonfinite, 채널 순서, gain 분리, 공간 보정과 clipping 순서, normal decoder 불변식을 검사한다.
- 재개 가능한 checkpoint를 복원하고 동일 다음 batch·증강·RNG에서 loss·gradient·Adam update·scheduler·단계 상태를 비교한다. CPU fixture와 실제 연구 데이터 smoke를 구분한다.
- 기존 source identity를 거짓으로 일치시켜 재개하지 않는다. 원본 provenance와 새 코드 provenance를 구분하며 검증된 전환을 기록한다.
- 설정은 실행·과학 조건과 경로를 분리하되 tensor 채널 순서 같은 고정 형식 계약을 임의 변경 옵션으로 만들지 않는다.
- 새 코어가 날짜별 experiments·기존 model/network 경로를 참조하지 않는지 검사한다.

## 조사 결과

- 7M network factory가 experiments/robustness_suite/run.py에 있으며 C objective는 render_weight_followup → full_dataset_suite → benchmark/train_fabric에 의존한다.
- C 재개 상태에는 global RNG 외 sample_rng, d4_rng, augmentation_rng, noise_rng 및 convergence/history가 있다. Weight만 복원하면 exact resume가 아니다.
- 기본 MatSynth capture 모델은 64M 구조여서 7M을 위한 별도 구성이 필요하다.
- 현재 DNG 처리 진입점은 정합·검토·수치 입력 준비·HTML 출력을 결합한다. 공개 모델 입력은 교정 완료 FP32 RGB 쌍이며, 독립 단일 preprocess_dng.py가 DNG 준비·검토·입력 저장을 담당하도록 확정했다.
- 구현 및 수행한 검증 범위는 [감사와 검증](AUDIT.md)에 기록했다. 생산 batch 규모의 GPU 연구 next-step과 사람의 최종 경계·무늬 검토는 구현 검사와 구분한다. 실제 CPU/CUDA 및 DNG 검증 결과는 감사 문서를 따른다.

- 구조의 dim·encoder/decoder layer 수·중간 layer 수·appearance channel은 config로 구성하며 CLI 기본은 현재 7M spatial profile이다.

## 2026-10-19 입력 영역 변경 승인

사용자는 일부 AprilTag 누락에도 검출 태그를 최대한 이용하고, 흰 종이와 경계가 명확한 target 사각형 전체를 입력에 포함하는 방식을 새 기본값으로 선택했다. 기존 512/crop46 입력은 capture_legacy.json으로 보존한다. 새 기본값의 시야·특징·예측은 과거 값과 같다고 주장하지 않는다. Weight 구조와 FP32 RGB tensor 규약은 유지한다. 경계 검출은 좌표용이며 측정 RGB는 원본 FP32 AHD에서 한 번만 resampling한다. 기준 near_00에서 결정한 한 사각형을 전체 burst에 공통 적용한다. 불확실한 경계·부족한 태그 분포는 거부하고, 사용 태그·제외 태그·잔차·crop 범위를 기록한다. 최종 입력 생성에는 경계와 무늬 정합 검토가 필요하다.
