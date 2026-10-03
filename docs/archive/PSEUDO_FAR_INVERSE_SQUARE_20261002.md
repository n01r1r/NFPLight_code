# far_pseudo 역제곱 밝기 감쇠 대조

사용자가 요청한 scaling은 공간 배율이 아니라 밝기 감쇠다. 현재 real 저자 모델의
정규화 거리 4/12와 실제 촬영 10/30cm는 거리비 3을 공유하므로, 정렬 후 clipping된
near 입력을 그대로 유지하고 `far_pseudo = near * float32(1/9)`로 바꾼다.
공간 리샘플링, 각도 역보정, denoiser, 재학습은 하지 않는다. synthetic 학습의
2.414/10 설정을 모든 real 학습/추론에 일반화하지 않는다.

두 비교 분기는 같은 원본 `net_g_real.pth` SHA256 allowlist와 FP32 추론을 사용한다.
저자 코드 분기 및 scalar 논문 수식+author33 적응 분기를 각각 actual far와 비교한다.
각 입력의 gain을 독립적으로 다시 계산한다. pseudo의 논문 중앙 gain은 약 9다.
따라서 global 밝기 감쇠는 `K_L`에 의해 상쇄되고 분자는 FP32 오차 내에서
`abs(I_N * (1 - cos(theta_N)/cos(theta_F)))`로 환원된다. RM 변화는 단순 밝기 변화
자체가 아니라 actual far의 공간/색/반사 차이를 copied-near로 제거한 결과다.
이를 분모로 나누면 `raw RM ≈ I_N/(cos(theta_N)*cos(theta_F))`이므로 pseudo raw
RM은 near 밝기 무늬를 주로 반영한다. 중앙 작은 분모의 FP32 상쇄 증폭을 포함한
`4e-7/C_M + 5e-5*abs(expected)` 오차 상한으로 이 항등식을 추가 검증한다.

이 대조는 거리 변화의 BRDF 효과를 포함하지 않아 optics 원인을 단독 식별할 수 없다.
조명, 센서, photometry 및 실제 표면 반사 변화도 경쟁 설명이다. GT가 없으므로
출력 차이는 정확도 향상의 증거가 아니다. 이미 clipping된 near는 복원하지 않는다.

입력 계약은 finite FP32 HWC `[256,256,6]`, linear RGB `[0,1]`이고 중앙 near RGB
평균은 양수여야 한다. raw RM, 적응 RM, log RM 지표는 각 분기 안에서 actual과
pseudo의 MAE/RMSE/max를 구하며 서로 다른 raw RM 단위를 섞지 않는다. 성분별
예측 지표는 raw tanh `[−1,1]` 단위다. RGB 공통 표시 범위는 `[0,1]`, raw RM은
각 분기의 actual/pseudo 공통 1–99 percentile, raw 차이는 abs p99 대칭 범위로 표시한다.
중앙 spike와 extrema는 원래 배열/수치에 보존한다. adapted/log는 공통 min/max와
최대 절댓값 대칭 범위다. gamma와 표시 범위는 수치 배열을 변경하지 않는다.

```powershell
python -m pytest tests/test_pseudo_far.py -q
python -m capture_processing.pseudo_far --capture-root artifacts/fabric_capture_20261002_aligned_paper_equations
```

baseline의 `both/arrays.npz`, `both/paper_arrays.npz`, `manifest.json` 해시를 실험 전후
검증한다. 새 결과만 `capture-root/pseudo_far/<group>/`에 저장한다. 입력/weight/code
해시, 선택 frame, dtype, 환경, seed, 수식 독립 검증 및 실제 지표를 provenance에 기록한다.
실험은 baseline이 완료된 그룹만 실행하고 모든 5그룹을 누락 없이 최종 확인한다.

## 실행 결과

5그룹 모두 author+paper 새 FP32 추론 완료. CUDA matmul/cudnn TF32와 autocast를
명시적으로 끄고 실행했다. 첫 그룹의 TF32 플래그 미설정 사전 결과는
`pseudo_far_unverified_tf32/`에 보존하며 아래 결과와 현재 index에서 제외했다.
baseline/code/checkpoint SHA256 전후 동일, archive roundtrip exact equality,
논문 독립 검증 및 copied-near analytic identity의 FP32 오차 상한 검증을 모두 통과했다.

아래는 논문 수식 분기의 actual far 대비 pseudo far MAE다. gain은 actual→pseudo,
적응 RM/log는 각 저장 feature 단위이고 성분별 출력은 raw tanh `[−1,1]` 단위다.

| 그룹 | gain | 적응 RM | log RM | normal | diffuse | roughness | specular |
|---|---:|---:|---:|---:|---:|---:|---:|
| 260818_174845_478 | 6.62572→9 | 0.391310 | 0.116586 | 0.104844 | 0.325319 | 0.143914 | 0.150535 |
| 260818_175005_078 | 6.13618→9 | 0.226106 | 0.163557 | 0.074336 | 0.031076 | 0.337628 | 0.018961 |
| 260923_141241_844 | 8.36219→9 | 0.074111 | 0.159197 | 0.044464 | 0.114011 | 0.590709 | 0.046276 |
| 260923_161825_503 | 12.4616→9 | 0.098479 | 0.181871 | 0.065983 | 0.077410 | 0.623427 | 0.149256 |
| 260930_152732_008 | 5.42901→9 | 0.286497 | 0.174314 | 0.075925 | 0.084703 | 0.617613 | 0.020195 |

첫 그룹과 마지막 그룹의 실제 이미지 검토에서는 actual raw RM의 중앙 residual이
pseudo에서 크게 줄고, pseudo 적응 RM/log가 near의 직물 무늬를 보이는 양상을 확인했다.
이는 `raw RM≈I_N/(cosθ_N*cosθ_F)` 항등식과 일치한다. 입력마다 min–max 적응을
다시 계산하므로 raw의 작은 범위가 적응 RM의 넓은 대비로 변할 수 있다.
실제 gain이 9보다 크거나 작은 사실만으로 optics 오류를 판정할 수 없다.
출력 변화가 있으나 GT 없는 이 대조로 개선이나 optics 단독 원인을 주장하지 않는다.
# 컴포넌트 report 표시 갱신

`pseudo_far/index.html`에서 5개 묶음의 저자 코드 단계와 논문 수식 단계를 각각 선택한다. 기존 capture report 렌더러를 재사용하여 입력, gain·각도 보정, near-minus-scaled-far, 절댓값, 분모, RM, 33개 feature 채널과 normal/diffuse/roughness/specular를 선택해 볼 수 있다. 논문 경로는 RGB 평균 후 스칼라 차이와 절댓값을 계산한 저장값을 표시하며, 저자 경로의 반대 부호 trace는 표시만 반전한다. 실제 far 대조와 기존 변화 지표는 `comparison.html`과 `provenance.json`에서 유지한다.

표시만 갱신하는 명령:

```powershell
python -m capture_processing.pseudo_far --capture-root artifacts/fabric_capture_20261002_aligned_paper_equations --report-only
```

이 명령은 모델을 생성하거나 재추론하지 않는다. 기존 baseline/pseudo NPZ, 추론 manifest, provenance, index.json의 SHA-256을 전후 비교하고 별도 `display_verification.json`에 기록한다. 기존 provenance에 기록된 `report.html` 해시는 새 표시로 대체되며 원래 provenance는 수정하지 않는다. 원래 수치 결과·가중치·FP32·both·denoiser 제외 규정은 유지한다. 10개 report의 550개 선택 단계 파일과 모든 이미지·링크 대상을 확인했으며, 관련 테스트 18개가 통과했다. 브라우저에서 논문/저자 차이의 부호, feature 채널 선택, 출력 통계와 gamma 전환을 확인했다. 감마는 diffuse/specular와 입력 RGB 표시만 바꾸고 normal/roughness·스칼라 feature는 바꾸지 않는다.
