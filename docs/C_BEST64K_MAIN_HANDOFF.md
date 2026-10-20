# C best64k 최종 호환성과 main 전환

2026-10-20. 기존 C 연장은 render65k에서 운영 종료했고 validation 선정 best는64k다. clean core `cc4b0f0`와 기존 구현을 같은 환경에서 비교했다.

## 실제 검사

- `best_render.pt`, `selected.pt`, `net_g_selected_fp32.pth`의 모든 weight tensor 및 key가 일치하고 strict load·parameter 등록 순서 검증을 통과했다.
- CPU와 RTX3090 CUDA 각각에서 결정적 capture/synthetic 입력2개와 기존 실측5묶음, 총14case의 특징·raw 예측·unit XYZ decoded N/D/R/S가 비트 일치했다. 최대 절대 오차0이다.
- 실제 실측5묶음은 기존 검토 기록과 fresh registration이 정확히 같을 때만 재사용했다. both·author_patch20·기존 crop46/420·동일 색변환과5평균/첫프레임 gate를 유지했다.
- 공개 `nfplight.cli.infer` 진입점으로 실제 입력 한 묶음을 저장/추론하고 같은 decoded 출력임을 확인했다.
- Python3.11.0rc2/PyTorch2.5.1+cu121/CUDA12.1, FP32, AMP/TF32 off. [검증 JSON](C_BEST64K_VERIFICATION.json).
- 현재 clean core 코드가 이전 전체54tests 검사 때와 같아 전체 테스트를 반복하지 않았다. 이번 검사는 최종64k weight와 입력/추론 경로에 집중했다.

CPU와 CUDA 사이의 출력 비트 일치는 주장하지 않는다. 각 장치에서 구/신 코드끼리 비교했다. 새 DNG 기본 robust_target crop은 기존 crop과 입력 영역이 달라 이 동일성 검증에 포함하지 않았다. 실제 연구 학습·optimizer update는 실행하지 않았고 원본 weight와 보고를 보존했다. SHA-256은 실행하지 않았다.

## 사용할 코드와 weight

현재 `main`은 이 clean core다. clean 작업 폴더는 `C:/Users/hdy/.codex/worktrees/clean-core-rewrite/NFPLight_code`다. 기존 `C:/workspace/NFPLight_code`는 미커밋 연구 코드·데이터·산출물을 보존한 checkout이며 아직 codex/fork-portability에 있다.

C 추론에는 `configs/7m_c.json`을 명시한다. 최종 weight는 `C:/workspace/NFPLight_code/artifacts/C_convergence_followup_20261019/run/net_g_selected_fp32.pth` 또는 같은 폴더 best_render.pt다. 입력은 검토 완료 FP32 pair만 허용한다. 기본 저자 inference profile을 C로 자동 교체하지 않았다.

```powershell
python -m nfplight infer --config configs/7m_c.json --checkpoint C:/workspace/NFPLight_code/artifacts/C_convergence_followup_20261019/run/net_g_selected_fp32.pth --input <reviewed_pair.pt> --out <new_prediction.pt> --device cuda
```

원본 전체 결과·검증 script/log·실제 준비 입력은 `C:/workspace/NFPLight_code/artifacts/C_best64k_clean_main_20261020`에 있다. 최종 모델 비교 HTML은 `C:/workspace/NFPLight_code/artifacts/relmap_three_way_real_20261019/index.html`이다.

## Archive와 다른 branch 정리 조건

같은 audit 폴더의 `archive/repository.bundle`은 모든133개 refs와 완전한 Git history를 담고 `git bundle verify`를 통과했다. refs-before.txt/worktrees-before.txt가 전환 전 위치를 기록한다. 미커밋 tracked4개는 binary patch와 원본 byte ZIP, untracked9개는 ZIP으로 보존했고 byte 검사를 통과했다. ignored 데이터·checkpoint·results·artifacts는 bundle/ZIP에 포함하지 않고 원래 위치에 보존한다. 이 보존 폴더를 지우면 안 된다.

|branch|정리 상태|
|---|---|
|codex/clean-core-rewrite|main이 동일 코어를 포함하고 worktree는 main으로 전환했다. 기존 ref는 bundle에 보존됐으며 branch ref 정리 가능|
|claude/reverent-bell-774205|main의 선조이며 bundle에 보존. branch ref 정리 가능; detached worktree의 파일 삭제는 별도|
|codex/fork-portability|커밋은 main의 선조이고 bundle에 보존. 현재 checkout이 dirty이므로 로컬 branch/worktree 삭제 보류. 원격 ref 정리는 가능하지만 이번에 삭제하지 않음|

다른 branch ref는 실제 삭제하지 않았다. 복구 자료와 명확한 경계를 남겨 후속 delete/archive가 가능하도록 했다. `.git`, checkpoints/results, dataset junction 외부 대상과 기존 산출물은 이동·삭제하지 않는다.

복구는 기존 checkout을 덮어쓰지 않고 새 폴더에서 bundle을 clone한 뒤 codex/fork-portability를 checkout하고 staged.patch/unstaged.patch와 untracked.zip을 적용한다. 정확한 수정 파일 원본은 modified_worktree_bytes.zip에도 있다. Git bundle 검증은 SHA-256 검증이 아니다.
