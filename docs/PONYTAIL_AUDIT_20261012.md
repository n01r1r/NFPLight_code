# Push 전 ponytail 감사 · 2026-10-12

이 저장소는 near/far RAW 촬영의 FP32 처리와 SVBRDF 추론, Fabric/Leather 학습 및 로컬 리포트를 제공한다. 예상 부하는 한 사용자의 단일 GPU 학습과 소수의 로컬 viewer 요청이다. 공개 서버나 여러 GPU의 동시 실행을 가정하지 않았다.

감사는 ponytail-audit의 읽기 전용 검토를 먼저 수행했다. 아래 수정은 사용자가 승인한 push 준비 작업으로 별도 수행했다. 기존 로컬 결과·checkpoint와 학습 실행은 변경하지 않았다.

## 먼저 고쳐야 했던 문제 — 수정 완료

1. **학습 checkpoint와 capture decoder의 계약 불일치** (`model/nfplight_matsynth_capture_model.py`, `model/nfplight_model.py`).
   - 역할: 저장 모델을 불러와 실제 선형 RGB 쌍을 추론한다.
   - 문제: export의 `normal_decoder=unit XYZ`가 인식되지 않아 normal 정규화가 생략됐다. full_dataset_suite의 best/selected checkpoint는 feature version을 identity 안에 기록하므로 기존 loader는 과거 geometry로 fallback했다. F0 계약이 있는 checkpoint는 raw specular head까지 덮어썼다.
   - 수정: 명시된 decoder 표기 및 suite rendering identity를 읽고, suite feature metadata 누락은 거부한다. raw specular head를 보존하고 F0는 rendering 단계의 계약으로 남겼다. 세 checkpoint 형식의 정상화·원시 specular 보존을 회귀 검사했다.
   - 미수정 시: 가중치는 같아도 입력 기하와 출력 normal/specular가 학습·보고 계약과 달라진다.

2. **고정 gain을 중앙 패치 추정으로 잘못 안내** (`capture_processing/report.py`).
   - 역할: 저장 수치의 특징·수식·표시를 설명한다.
   - 문제: gain9에도 저자 중앙 20×20 추정 설명을 붙였으며, gain 수치가 없는 pseudo 단계에도 그 안내가 표시됐다. gain9 특징 버전은 의미 있는 채널 이름 대신 일반 채널로 표시됐다.
   - 수정: 고정 checkpoint gain과 patch20을 계약으로 분기하고, gain이 없는 단계에서는 안내를 생략한다. v3에 동일한 RAW 21채널 의미를 적용했다.
   - 미수정 시: 사용자에게 계산하지 않은 밝기 추정과 잘못된 전처리 의미를 설명한다.

3. **새 실행 관리 코드의 설치 의존성 누락** (`requirements.txt`, `README.md`).
   - 역할: 독립 환경의 설치와 검사 절차다.
   - 문제: queue/process 관리가 psutil을 import하지만 설치 목록에 없었다. README의 pytest 실행 전 설치도 명시되지 않았다.
   - 수정: psutil 의존성을 추가하고 pytest 설치 명령을 기록했다.
   - 미수정 시: 새 환경에서 controller import 또는 검사 명령이 실패한다.

4. **로컬 생성물이 공개 후보에 섞임** (`.gitignore`).
   - 역할: source와 로컬 실행 결과를 분리한다.
   - 문제: 브라우저 프로필, 임시 수치, 출력 PDF·graph 캐시가 미추적 파일로 노출돼 전체 add에 포함될 수 있었다.
   - 수정: `_workspace/`, `tmp/`, `output/` 및 graphify 출력 폴더를 제외했다. 데이터·모델·artifacts의 기존 제외도 유지했다. 파일을 삭제하지 않았다.
   - 미수정 시: 코드 push에 개인 브라우저 상태와 대량 임시 결과가 함께 들어간다.

5. **회귀 검사가 과거 로컬 파일 및 검사 순서에 의존** (`tests/test_regularization_probe.py`, `tests/test_render_factor_followup.py`, `tests/test_training_policy_doctor.py`).
   - 역할: prepare·gate·identity·보존 동작을 검사한다.
   - 문제: 승인된 cleanup으로 삭제된 checkpoint를 unit fixture로 읽었으며, 이미 다른 검사에서 초기화된 CUDA를 반드시 미초기화 상태라고 가정했다. 정책 doctor 검사도 변경되는 실제 로컬 이력 전체가 무조건 PASS라고 가정했다.
   - 수정: 변경 계약을 검사하는 작은 identity/audit fixture를 사용했다. CUDA 상태가 호출 전후 유지되는지를 검사한다. doctor는 audit 자체가 보존 파일 변경을 감지하는 경우를 검사한다.
   - 미수정 시: 정상 코드도 로컬 이력 또는 검사 순서만으로 실패하고 보존 실패의 실제 원인이 가려진다.

## 별도로 정리할 로컬 이력

6. **과거 artifact doctor의 실제 로컬 이력은 아직 PASS가 아니다** (`doctor_training_policy.py`, 로컬 `artifacts/`).
   - 역할: 과거 결과의 링크와 기록된 size/mtime를 엄격히 대조한다.
   - 문제: 기존 cleanup 및 이후 코드 수정을 반영하지 않은 링크·동결 기록이 있다. 첫 검사에서 실제 로컬 이력을 참조하던 테스트는 576개 오류를 반환했다.
   - 처리: 이 실패를 숨기거나 원래 결과/identity를 덮어쓰지 않았다. 공개 코드의 fixture 검사와 실제 과거자료 정합 판정은 분리했다. 정리하려면 해당 cleanup·source repair 이력을 기준으로 로컬 기록을 별도 대조해야 한다.
   - 미정리 시: 과거 이력 전체를 대상으로 하는 doctor는 실패를 계속 보고한다. 이번 코드 감사 통과는 이 doctor PASS를 뜻하지 않는다.

## 검사와 공개 범위

- 전체 회귀: **347 passed, 42 skipped**, 45.79초. 42개 생략은 모두 사용자의 SHA-256 실행 금지 때문이다. 해당 경로에 들어가면 skip하도록 보호한 별도 검사 launcher를 사용했다.
- JavaScript: live training plot의 shared range·log zero·category·canvas 검사가 PASS. difference_visualization.js의 Node syntax 검사 통과.
- 정적 검사: Python 104파일 syntax parse, 고신뢰 credential 패턴 일치 없음, git diff --check 통과. 이 패턴 검사는 포괄적 보안 인증이 아니다.
- 크롭 입력: inset0 → warp512 → 단일 crop46/420 → area256, 종이를 제외한 샘플 5cm. 관련 회귀 및 5묶음 첫 near/far 쌍의 실제 ROI 검토는 CAPTURE_REQUIREMENTS의 검증 기록을 따른다.
- 포함: capture/model/training 구현, 실험 source/config, 관련 테스트, 정책·문서, 기존 정리본을 복구하는 작은 source ZIP.
- 제외: browser profile, 임시 결과, 생성 PDF/graph 출력, RAW 촬영·dataset·checkpoint·artifacts. 기존 로컬 파일은 보존한다.

판정: 공개할 코드·문서의 위 blocker를 수정했고 회귀 검사는 통과했다. 현재 브랜치의 commit/push를 진행할 수 있다.

미검사: 새 환경의 완전 설치, GPU 전체 학습 및 재추론, 모든 과거 artifact의 과학적 타당성, 브라우저 전체 상호작용. 변경 규모가 커서 frozen/archived source와 모든 실험 코드를 줄 단위로 전부 읽지는 않았다. 입력·모델·저장·resume/lock·로컬 viewer의 주요 경로를 우선 감사했다. 현재 진행/보존 기록을 이 감사만으로 갱신하지 않는다.
