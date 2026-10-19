# 7M weight 전달과 코드 점검

2026-10-19 origin/codex/fork-portability를 fetch한 뒤 비교했다. 점검 시작 시 HEAD와 remote는 `4b07b43`으로 같았으나, 아래 공통 relmap·gain 분리·runner·보고 변경은 로컬에만 있었다.

## 전달할 파일

저장소 루트 `C:/workspace/NFPLight_code` 기준이며 weight와 artifact는 Git에서 제외된다. 아래 세 파일은 별도로 전달해야 한다.

|구분|파일|선택 시점|relation|
|---|---|---|---|
|수정 전 7M|`artifacts/full_dataset_baseline7m_best_real_20261012/snapshot_20261012_090422/snapshot/best_render.pt`|render32k, 기록상 누적185k|legacy_no_spatial|
|수정 후 7M|`artifacts/spatial_7m_real_20261019/snapshot/best_render.pt`|render49k, 기록상 누적204k|spatial_inverse_square_v1|
|C|`artifacts/render_weight_followup_20261019/runs/C_weight10_lr1e5/best_render.pt`|map135k에서 render10k, 누적145k|spatial_inverse_square_v1|

세 파일 모두 7,609,834 parameters의 동일 구조로 strict load했고 모든 state tensor의 FP32·유한값을 확인했다. 세 checkpoint의 train/validation 행도 정확히 같다. C의 기존 `net_g_selected_fp32.pth` tensor는 `best_render.pt`와 정확히 같다. 다만 기존 간이 export에는 `feature_version`과 backbone 정보가 없으므로 위 `.pt`를 전달한다. 기존 파일은 덮어쓰지 않았다. 향후 export는 두 metadata를 보존하도록 수정했다.

C는 render weight10·초기 LR1e-5의 validation 선택 render10k다. 10k 선별 예산 종료이며 수렴 또는 통계적 유의성 판정이 아니다. 사용자의 정성 선호와 별도로 validation NDRS합=0.1207984001, render L1=0.0032749340이다. 기존7M은 memory_paused 실행에서 보존한 선택 weight로 완주 모델이 아니다.

## 수신 측 inference·validation 계약

- `.pt`의 `net_g`를 parameter mapping으로 사용하고, `identity.policy`와 `identity.scientific_settings.backbone`으로 `experiments.robustness_suite.run.create_network`를 구성한다. 기본 `MatSynthCaptureModel`은 64M 구조를 만들므로 7M weight를 그대로 넣는 CLI로 안내하지 않는다. 네트워크는 eval 모드로 실행한다.
- 두7M과 C는 `raw_calibrated_gain9_v3`, near4/far12, unit XYZ normal decoder다. 추론 뒤 normal XYZ를 단위 정규화한다. Specular head는 보존하며 재렌더링에서 F0=.04를 사용한다.
- 학습·합성 validation은 `build_features`의 gain9, 실측은 `build_capture_features`의 중앙20×20 RGB 평균 비율을 사용한다. FP32, AMP/TF32 off, 입력 선형 RGB, denoiser 없음이다. DNG 처리는 기존 capture 정합·crop·both 정책을 따른다.
- 수정 전 weight의 과거 특징을 재현하려면 공간 계수를 적용하지 않아야 한다. 현행 모델은 checkpoint의 옛 relation metadata만으로 자동 전환하지 않는다. 과거21채널 경로는 `spatial_attenuation_factor`를 ones로 설정하고 `relation_version=legacy_no_spatial`로 명시해야 한다. 기존3조건 비교는 보존 당시 relation 함수도 연결해 전체 특징의 정확 일치를 확인한 별도 경로다.
- 수정 후/C는 `spatial_inverse_square_v1`을 사용한다. 학습과 eval·실측 모두 공간 거리 계수와 곱셈/clip 순서를 공유한다. 보정 전 optimizer의 현행 학습 resume은 거부한다.
- `.pt`의 identity에 있는 로컬 데이터 경로·mtime는 학습 provenance다. 수신 환경의 재개 권한이나 데이터 설치 경로로 해석하지 않는다. 원본 snapshot의 전체 CPU 실측 비교 runner는 로컬 artifact에 있으므로 보고서 재생성까지 필요하면 해당 artifact도 별도 전달해야 한다.

## 변경·검증 범위

공통 공간 계수, gain 분리, checkpoint 버전 검사, 전체 데이터 준비/smoke 검사, A–C/D runner와 보고·표시 변경을 점검했다. 구조·데이터 population·gain9/patch20·FP32 계약은 유지된다. 현행 코드로 옛 weight를 실행하면 공간 보정 때문에 과거 추론과 달라지는 점은 의도된 호환성 제한이다.

간이 export metadata 누락을 수정하고 세 runner에서 같은 payload 함수를 사용한다. D worker의 실제 실패 기록에 있던 Windows `psutil.disk_usage(Path)` 호출도 문자열 경로로 수정했다. 실패한 queue를 재개하거나 새 학습을 시작하지 않았다. source identity가 바뀌므로 기존 immutable 실행을 그대로 재개할 수 있다고 주장하지 않는다.

CPU 전체 검사: 405 passed / 42 skipped. SHA-256 호출을 skip으로 막았고 CUDA를 숨겨 GPU 검사를 제외했다. export 수정 뒤 관련 검사 45 passed. 세 실제 weight의 strict load·FP32·유한값·C export tensor 일치 검사를 별도로 통과했다. 새 GPU 학습·실제 DNG 재추론·성능 검증·SHA-256은 실행하지 않았다.

로컬 `.claude_resources.json`과 과거 수식 진단 스크립트 `audit_actual_relation_albedo.py`, `verify_diffuse_gain_relation.py`, `visualize_relation_equations.py`는 이번 전달 commit에서 제외한다. 기존 artifact·weight는 보존한다.
