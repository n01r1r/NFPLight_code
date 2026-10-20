# Clean core 계약

## 현재 기준과 범위

작업 기준은 main, 기본 경로는 C:/workspace/NFPLight_code다. [사용법](USAGE_KO.md), [현재 상태](CURRENT_TRAINING_STATE.md), [검증 근거](AUDIT.md), [archive와 복구](C_BEST64K_MAIN_HANDOFF.md)를 따른다. 날짜별 실험·보고서 생성기는 활성 코드에 다시 추가하지 않는다.

`nfplight/`는 모델·특징·추론·학습·센서 수치를 담당하고 `preprocess_dng.py`는 prepare/검토/finalize의 단일 진입점이다. 과거 import/CLI facade는 유지하지 않는다. 실제 데이터·weight·결과와 junction 대상은 기존 위치에 보존한다.

지원 weight는 전달용7M original/spatial/C와 C 후속, 저자 real/syn이다. 기본 구조는 `configs/7m_spatial.json`이며 구조·특징·geometry·relmap·normal decoder를 config로 명시한다. Metadata가 부족하면 profile을 명시하고 충돌하면 거부한다. State key와 parameter 등록 순서는 optimizer 호환성에 포함한다.

## 수치와 입력

- FP32, AMP/TF32 off. 학습·합성7M spatial/C는 near4/far12·고정 gain9, 실측은 both·중앙20×20 RGB 평균 near/far 비율(author_patch20)이다. 저자 profile geometry는 해당 config를 따른다. Denoiser를 사용하지 않는다.
- 공개 모델 입력은 교정 완료 `[1,6,256,256]` 선형 RGB와 bool mask다. 출력은 N3/D3/R1/S3이며 profile에 맞게 decode한다.
- 새 DNG 기본 robust_target은 near_00의 target 사각형 하나를 전체 burst에 적용한다. 공통 태그 최소3개·양 축 분포·RMS1px·전체 crop support를 검사하고 사용/제외 태그와 잔차를 기록한다.
- 원본 FP32 AHD에서 warp512·area256을 수행한다. 경계 검출용 표시 영상을 측정 RGB로 사용하지 않는다. 모든 수치 조건과 사람의 경계·무늬 검토 통과 시5평균, 실패하면 유효한 near_00/far_00, 기하가 불충분하면 거부한다.
- 기존 crop46/420은 `configs/capture_legacy.json`으로 보존한다. 새 기본과 과거 입력·예측의 동일성을 주장하지 않는다.

## 학습과 복원

Model-only best는 exact resume 입력이 아니다. 전체 last checkpoint의 Adam·scheduler·전체 RNG·sampler·단계·평가·best·floor counter·종료 상태를 보존한다. Runtime·환경·입력 identity·수치 설정이 다르거나 smoke를 연구 학습으로 전환하면 거부한다. 과거 provenance와 새 코드 provenance를 별도로 기록한다.

완료한 C는 재개하지 않는다. 새 연구 학습·기존 학습 중단·원본 산출물 덮어쓰기는 별도 사용자 권한이 필요하다. 승인된 실행도 새 출력 폴더를 사용한다. Pause·failure를 자동 재개하지 않으며 유한 cap 도달을 수렴으로 해석하지 않는다. [Scheduler와 운영 종료 정책](TRAINING_SCHEDULER_POLICY.md)을 따른다.

## 구현과 검증

의미별 모듈과 명시적 실패 계약을 유지한다. 기존 lucidrains 관례, `Module`/`ModuleList`, 명시적 residual 합, dim/depth/to_* 이름을 따른다. Tensor layout은 einops rearrange/einsum으로 shape·단위·순서를 드러낸다. 불필요한 추상화·옵션·의존성을 추가하지 않는다.

기본은 동적 typing이며 필요 시 future annotations와 `X | None`을 쓴다. Kwarg는 `name = value`, 문자열은 작은따옴표, 내부 import는 absolute다. Import는 stdlib / torch / einops / 기타 / 프로젝트 순서다. 잘못된 연구 입력·nonfinite·영속 데이터는 ValueError 등으로 거부하고 검증 CLI는 assert에 의존하지 않는다.

변경 범위에 맞게 strict weight·feature/raw/decoded·재개 상태와 작은 결정적 검사를 수행한다. 실제 연구 update, 구현 fixture, 사람이 검토할 입력을 구분한다. Formatter·자동 인덱싱·SHA-256은 실행하지 않는다. `.git`·checkpoints·results·dataset junction 외부 대상을 재귀 삭제하거나 이동하지 않는다.
