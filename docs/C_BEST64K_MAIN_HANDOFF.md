# C best64k 검증과 clean main 전환

2026-10-20 기준 작업 폴더는 **C:/workspace/NFPLight_code**, branch는 **main**이다. Clean core를 승격한 뒤 이 경로에서도 실제 C best64k 추론을 확인했다. 별도 clean worktree는 detached 상태로 보존했다.

## 최종 weight 호환성

`best_render.pt`, `selected.pt`, `net_g_selected_fp32.pth`의 모든 weight tensor와 key가 일치했다. Strict load와 parameter 등록 순서도 통과했다. CPU와 RTX3090 CUDA 각각에서 결정적 입력2개·실측5묶음, 총14case의 특징·raw·decoded 출력이 기존 코드와 비트 일치했고 최대 절대 오차는0이었다. 공개 CLI의 실제 입력 저장·추론도 통과했다.

환경은 Python3.11.0rc2, PyTorch2.5.1+cu121, CUDA12.1, FP32, AMP/TF32 off였다. [원본 검증 JSON](C_BEST64K_VERIFICATION.json)은 변경하지 않았다. CPU와 CUDA 사이의 비트 일치나 새 robust_target 영역과 과거 crop46/420의 동일성을 주장하지 않는다.

현재 작업 폴더에서 전체54tests를 다시 통과했고, 기존 검토 입력과 best64k를 공개 CLI로 추론해 이전 CLI의 특징·raw·decoded 출력과 비교했다. 이번 전환의 [검증 기록](../artifacts/clean_main_cleanup_20261020/verification.json)을 따른다. 실제 연구 학습이나 optimizer update는 실행하지 않았다.

## 정리한 branch

|대상|결과|
|---|---|
|로컬 main / origin/main|현재 clean core와 최신 문서의 기준|
|로컬 codex/clean-core-rewrite|main의 선조임을 확인하고 삭제|
|로컬 codex/fork-portability|최신 미커밋 변경을 보존하고 선조 관계 확인 후 삭제|
|로컬 claude/reverent-bell-774205|main의 선조임을 확인하고 삭제|
|origin의 clean-core-rewrite / fork-portability|두 원격 branch 삭제|
|기존 detached worktree|파일을 유지; 디렉터리 삭제하지 않음|

GitHub 기본 branch는 main이다. `upstream` 원본 저장소는 출처로 유지한다. Git metadata는 기존 `D:/NFPLight_runs/NFPLight_code/.git`에 있고 C 경로의 `.git` 파일과 worktree 설정으로 연결된다. D 저장소와 junction 외부 대상을 이동하거나 삭제하지 않았다. 따라서 C 폴더만 단독 복사하면 Git 저장소 전체를 복사한 것이 아니다.

## 보존과 복구

현재 archive는 [clean_main_cleanup_20261020](../artifacts/clean_main_cleanup_20261020/RESULTS_KO.md)다.

- `repository_with_latest_stash.bundle`: 모든 refs와 최신 stash를 포함한다. `git bundle verify`를 통과했다.
- `latest_worktree.zip`: 전환 직전 수정4개·미추적9개, 총13개 파일의 원본 byte를 보존했다. Binary patch와 경로 목록도 함께 있다.
- Stash commit `6a47a414cc0f27458fb3e4fdb990f2a5166c7f81`을 보존했다. 정리된 main에 바로 apply하지 않는다.
- `docs_before.zip`: 이번 문서 정리 전 README와 Markdown 문서3개의 원본이다. 과거 수치 JSON·HTML·보고서 원본은 그대로 유지했다.
- 데이터·weight·보고서 등33,979개 기존 로컬 항목의 경로·크기·mtime와 junction target을 전후 비교했다. 암호학적 내용 검증은 수행하지 않았다.

복구는 **새 폴더**에서 수행한다. Bundle을 clone한 뒤 stash의 첫 부모를 기준으로 별도 복구 branch를 만들고 stash commit을 적용한다. 충돌 시 ZIP 원본을 사용한다. 예:

```powershell
git clone C:/workspace/NFPLight_code/artifacts/clean_main_cleanup_20261020/repository_with_latest_stash.bundle <새복구폴더>
git -C <새복구폴더> fetch C:/workspace/NFPLight_code/artifacts/clean_main_cleanup_20261020/repository_with_latest_stash.bundle refs/stash:refs/archive/legacy-stash
git -C <새복구폴더> switch -c codex/restore-legacy '6a47a414cc0f27458fb3e4fdb990f2a5166c7f81^1'
git -C <새복구폴더> stash apply --index 6a47a414cc0f27458fb3e4fdb990f2a5166c7f81
```

Bundle에는 ignored 데이터·weight·보고서가 포함되지 않는다. 이들은 기존 C 경로와 D junction 대상에 보존돼 있다. 초기 전환의 [별도 archive](../artifacts/C_best64k_clean_main_20261020/archive/manifest.json)도 유지했다. 과거 문서의 “C 경로는 아직 legacy branch”라는 내용은 당시 기록이며 이 문서로 대체한다.
