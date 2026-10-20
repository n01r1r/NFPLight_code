# NFPLight

Near/far 선형 RGB 쌍으로 N/D/R/S를 추정하는 FP32 모델·추론·학습 코어다. 작업 기준은 **main**, 기본 작업 폴더는 **C:/workspace/NFPLight_code**다. 기존 데이터·weight·보고서는 같은 위치에 보존했다.

## 시작

환경에 맞는 PyTorch CPU/CUDA 배포판을 설치한 뒤 실행한다.

```powershell
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

기본 모델은 7.61M, width32, encoder [2,2,4], middle4, decoder [2,2,2], dropout .1이다. 기본 config는 `configs/7m_spatial.json`이다. Weight에 맞는 profile을 명시하며 metadata 충돌은 거부한다.

## C best64k 추론

C는 render65k에서 운영 종료했고 validation 선정 best는64k다. 기존 구현과 CPU/CUDA 각각의 FP32 추론이 일치했다. 이 weight를 사용할 때는 C profile을 지정한다.

```powershell
python -m nfplight infer --config configs/7m_c.json --checkpoint artifacts/C_convergence_followup_20261019/run/net_g_selected_fp32.pth --input <reviewed_pair.pt> --out <new_prediction.pt> --device cuda
```

입력은 검토 완료한 `[1,6,256,256]` FP32 선형 RGB 쌍과 bool mask다. 실측은 `both=(D-B)/(W-B)`와 `author_patch20`을 사용한다. 학습 gain9를 실측에 적용하지 않는다. 출력은 N(3)/D(3)/R(1)/S(3) 순서다.

## DNG 입력 준비

```powershell
python preprocess_dng.py prepare --source <촬영묶음> --out <새준비폴더>
python preprocess_dng.py finalize --prepared <준비폴더> --review <review.json> --out <새pair.pt>
```

새 기본은 `robust_target`이다. 준비 영상의 target 경계와 무늬 정합을 사람이 검토한 뒤 finalize한다. 수치·검토 게이트를 모두 통과하면 5장 평균, 실패하면 기하가 유효한 frame00을 사용한다. 과거 crop46/420 재현은 `prepare --config configs/capture_legacy.json`을 사용한다. 두 입력 영역의 예측이 같다고 주장하지 않는다.

## 코드와 문서

|찾는 내용|위치|
|---|---|
|모델·특징·checkpoint·추론·학습|[nfplight](nfplight/)|
|모델 및 실행 설정|[configs](configs/)|
|상세 사용법·profile·재개 조건|[사용법](docs/USAGE_KO.md)|
|현재 모델·실험 결론|[현재 상태](docs/CURRENT_TRAINING_STATE.md)|
|수치·입력·코드 변경 계약|[코어 계약](docs/CLEAN_CORE_REWRITE.md)|
|검증 범위와 한계|[감사와 검증](docs/AUDIT.md)|
|main 전환·branch 정리·복구|[전환 기록](docs/C_BEST64K_MAIN_HANDOFF.md)|
|실험 보고서·weight·archive|[로컬 산출물 안내](artifacts/INDEX_KO.md)|

`artifacts/INDEX_KO.md`는 Git에 기록하는 안내다. 실제 artifact·데이터·weight는 로컬 보존 대상이며 Git에 업로드하지 않는다. 새 clone에는 별도로 준비해야 한다. 과거 실험 runner와 보고서 생성기는 archive/Git 이력에 보존했으며 현재 실행 진입점은 `nfplight`와 `preprocess_dng.py`다. 실제 연구 학습과 완료한 C 학습의 재개는 별도 사용자 승인이 필요하다.
