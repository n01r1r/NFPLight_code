# 검증 범위와 근거

2026-10-20. 활성 구현은 main의 `nfplight/`와 `preprocess_dng.py`다. 기존 runner 의존성을 제거하면서 weight·수치·재개 상태의 호환성을 검사했다. 생성된 JSON은 검사 당시의 입력과 환경을 기록한 원본 증거로 유지한다.

## 수행한 검증

|범위|실제 검사와 근거|
|---|---|
|지원 모델6개|7M profile3개·당시 C last·저자 real/syn의 strict load, parameter 순서, CPU/CUDA FP32 특징·raw·decoded·renderer 비교. [compatibility.json](compatibility.json)|
|학습 상태와 수치|동일 batch·증강·dropout·RNG·loss·gradient·clipping·Adam moments·scheduler·단계/종료 상태 비교. 실제 C snapshot 복원 및 환경·입력 gate 검사. [training_compatibility.json](training_compatibility.json)|
|DNG 입력|실제5묶음50장, 기존 crop의 geometry/alignment/support 재현 및 새 robust_target 전처리. 태그 누락·오검출·분포 부족·경계 미검토 거부 포함. [validation.json](validation.json)|
|최종 C best64k|동일 장치의 구/신 구현14case 출력 비트 일치, 세 weight/export 일치, 공개 CLI 검사. [C_BEST64K_VERIFICATION.json](C_BEST64K_VERIFICATION.json)|
|C 작업 폴더 통일|전체54tests, C 경로에서 best64k CLI 재실행 및 이전 저장 출력 비교, bundle·stash·보존 항목 검사. [전환 검증](../artifacts/clean_main_cleanup_20261020/verification.json)|
|과거 source 정리|교체된168개 파일 목록은 [removed_sources.json](removed_sources.json). Git 이력·archive에 보존|

단위 검사는 입력 shape/range/mask/nonfinite, gain 분리, spatial relation과 normal decoder, metadata 충돌, evaluation RNG 보존, budget pause, atomic write 실패, 출력 lock과 sensor 계약을 포함한다. 검증 CLI는 Python `-O`에서도 잘못된 입력을 거부한다.

## 확인하고 수정한 주요 문제

- 7M 구조·relmap·normal decoder가 실행 경로에 따라 달라지던 문제를 profile config와 strict loader로 통합했다.
- C 이어 학습에서 model state만 복원하거나 변경된 수치 설정·입력·환경을 허용하던 경로를 막았다. 역사적 source identity와 새 implementation provenance를 구분한다.
- `cuda`/`cuda:0` 장치 비교 오류를 수정했다. 실제 할당 장치를 기준으로 검사한다.
- DNG 센서 처리와 보고서를 분리했다. 경계·무늬 정합 검토 전 평균을 확정하지 않고, 수치/검토 실패 시 유효한 frame00을 사용한다.

## 해석의 한계

실제 C Adam/CUDA RNG를 사용한 다음 update 비교는 작은32px 입력 검사다. 실제 train crop·D4 8종과 CPU256px·40광원 C loss/gradient 검사는 별도로 수행했다. 이들을 생산 batch 크기의 GPU 연구 학습이나 수렴 증거로 해석하지 않는다.

기존 crop46/420의 동일성 검증과 새 robust_target 기본 입력 검증은 다르다. 새 영역은 시야와 예측이 달라질 수 있다. 자동 전처리 통과는 사람의 최종 경계·무늬 검토를 대체하지 않는다. 실제 가림 촬영, 모든 지원 버전의 설치·성능 benchmark, 새 연구 학습은 수행하지 않았다.

전체 validation 및 실측 비교의 과학적 결론은 [현재 상태](CURRENT_TRAINING_STATE.md)와 연결된 원본 보고서를 따른다. 이번 코드 검증은 그 결론의 독립 재현 실험이 아니다. 기존 데이터·weight·보고서를 유지했고 SHA-256·formatter·자동 인덱싱은 실행하지 않았다.
