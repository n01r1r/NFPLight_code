> Historical research record. Current capture contract: [CAPTURE_REQUIREMENTS.md](../CAPTURE_REQUIREMENTS.md). Current commands: [README.md](../../README.md). Code citations and run status below describe their recorded version; removed code and old line numbers are not current implementation claims.

# NFPLight 아키텍처 레드팀

**프레임.** SVBRDF 추정의 전 과정 — 촬영·optics·forward model·전처리·network·출력 — 이 단 하나의 물리 단서에 의존한다: near/far 두 샷의 공간 광량 falloff 차이. 이 문서는 그 단서가 어디서 오염되고 어디서 사라지는지를 심각도 순으로 정리한다.

정상(n)과 거칠기(r)는 전처리에서 유도한 relation·mask 3채널에서만 나온다([nfplight_net.py:326](../../network/nfplight_net.py:326), [:361](../../network/nfplight_net.py:361)). 따라서 falloff 단서의 모든 약점은 곧 n, r의 오차다.

## 치명 — 구조적

**1. 단서와 노이즈가 같은 부분공간에 있다.**
disambiguation은 `relation_map = (time_co·aligned_far − near) / coefficient`로 유도된다([nfplight_real_model.py:94](../../model/nfplight_real_model.py:94)). vignetting, flash 반사판 불균일, board tilt, ambient, inter-reflection은 모두 near-field falloff와 같은 저주파 radial 패턴이라 물리적으로 단서에서 분리할 수 없다. white/gray 기준판, geometry용 checkerboard, 측정 광량 같은 per-capture calibration이 전혀 없어 redundancy가 0이다.

**2. Vignetting은 상쇄되지 않고 near/far에서 다르게 나타난다.**
near 샷은 board가 프레임을 채우고 far 샷은 board가 중앙 소영역만 차지한다. resize 후 두 샷의 vignette 프로파일이 다르다. relation 식에서 렌즈 vignette V(x)는 곱으로 남아 나눠지지 않으므로 radial 체계 편향이 n, r 단서에 그대로 주입된다. 학습은 vignette 없는 synthetic이라 net이 이 편향을 흡수하지 못한다.

**3. Colocation(V=L) 가정 vs 실제 flash–lens baseline.**
학습 데이터는 완전 colocated 렌더다(l, v 모두 `near_light_dir`, [nfplight_model.py:69](../../model/nfplight_model.py:69)). 실제 flash는 렌즈에서 수 cm 떨어져 있고 각오차 ≈ baseline/거리이므로, 신호가 강한 near 샷(dist 4)에서 far보다 오차가 크다. highlight가 가정 위치에서 밀려 normal 단서가 편향된다 — 앞서 관측한 hotspot off-center(0.48, 0.54)와 같은 병증이다.

**4. Roughness는 구조적으로 대부분 미관측이다.**
colocated에서 H=L=V이므로 픽셀당 specular lobe를 단 하나의 configuration으로만 샘플한다([render_util.py:326](../../utils/render_util.py:326)). matte·평탄 normal 영역에는 highlight가 없어 roughness 렌더 민감도가 2% 미만이다(선행 실험). 출력 r은 측정이 아니라 prior의 산물이며, confidence 출력이 없어 측정값과 구분되지 않는다.

**5. Geometry 상수 하드코딩이 rig에 엄격한 calibration을 요구한다.**
real은 near=4 / far=12([nfplight_real_model.py:23](../../model/nfplight_real_model.py:23)), syn은 2.414 / 10이다. "distance 4"는 카메라가 board 반폭의 4배 거리·축상·정중앙이라는 뜻이고 coefficient map이 이 값에 의존한다. per-capture 보정 입력이 없으므로 실측 rig가 4× / 12×를 정확히 재현하지 못하면 단서 자체가 miscalibrated된다. 30 cm board 기준 60 cm / 180 cm 축상 중앙 — 재현이 까다롭고 검증 수단이 없다.

## 심각

- **scale이 중앙 20×20 단일 스칼라다**([nfplight_real_model.py:108](../../model/nfplight_real_model.py:108)). 중앙이 board 위·비포화·동일재질이어야 한다. 하나라도 깨지면 far 정렬과 relation 전체가 무너진다.
- **coefficient가 중앙에서 0이라 0/0 증폭이 일어난다**([nfplight_real_model.py:97](../../model/nfplight_real_model.py:97)). 단서는 중앙에서 소멸하고 corner에서 최대인데, corner는 vignette·perspective·board edge·수차가 가장 나쁜 곳이다. 단서가 optics를 가장 못 믿을 곳에만 산다.
- **ClipToOne이 전역 min/max 정규화다**([nfplight_real_model.py:88](../../model/nfplight_real_model.py:88)). dust나 saturated 반점 하나가 map 전체를 재스케일한다.
- **2샷 공간 정합이 없다.** 정렬은 전역 스칼라 scale과 per-pixel time_coeff뿐이다. lateral shift나 재프레이밍이 있으면 relation이 망가진다.
- **ambient는 가법항이라 1/d²를 따르지 않는다.** 실내 조명이 relation 단서를 직접 오염시킨다. 암실 강제나 보정이 없다.
- **denoiser 도메인갭.** synthetic noise로 학습한 첫 단이라([nfplight_net.py:374](../../network/nfplight_net.py:374)) 실측 sensor 노이즈와 어긋나면 hallucination이 하류로 전파된다.
- **LDR clip·8 bit·미보정 WB/tone.** 포화 highlight의 spec 정보가 소실되고(>0.95는 mask로 폐기), 순수 gamma 2.2 가정과 미보정 white balance가 d, s 색에 baked-in된다.
- **absolute albedo 미교정.** 학습 `lampIntensity=16` 고정, 실측 노출·광량 미지 → 출력 diffuse는 상대값이다.

## 경미

단일 lobe GGX+Smith+Schlick(anisotropy·clearcoat·subsurface 없음, diffuse는 Lambertian만) · d와 s 독립이라 energy conservation 미강제 · roughness 0.05 floor · 256 고정 INTER_AREA resize로 미세 디테일 소실 · 유한 flash 광원의 highlight 연화(roughness 과대 방향) · flash TTL 변동 시 공간 성분 미보정.

## 반증 테스트 — 싸고 결정적

| # | 대상 | 방법 | 확정 조건 |
|---|------|------|----------|
| 1 | 치명2 | 백색 무광판 near/far flat-field | 잔차가 radial이면 vignette 오염 |
| 2 | 치명1 | 암실 vs 실내조명 동일 샘플 | n, r 변화 크면 단서가 광량 구성에 종속 |
| 3 | 치명3 | baseline b 측정, b/Z 각오차 vs hotspot 변위 | 일치하면 normal 편향 원인 |
| 4 | 치명4 | 알려진 무광/유광 평판의 r 출력 | GT 미추종이면 prior hallucination |
| 5 | 치명5 | rig 거리 ±10 % 재촬영 | 출력 drift 크면 4/12가 knee |

## 결론

측정 도구가 아니라 단일 저주파 단서와 학습 prior의 결합체다. 단서가 강한 corner에서는 optics 오염이 최대이고(치명2), 단서가 약한 평탄부에서는 prior가 대체한다(치명4). 두 실패 모드 모두 결과가 그럴듯하고 confident해서 육안으로 구분되지 않는다.

정확도 상한을 올리려면 알고리즘이 아니라 계측이 필요하다: per-capture calibration frame, 암실, flash–lens colocation 확보 또는 baseline 모델링, vignette flat-field, rig 거리 실측 입력. 이 계측 없이는 개선의 상한이 물리적으로 막힌다 — TTO 폐기 때와 같은 벽이다.
