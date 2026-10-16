# 원본 real 모델의 위치별 거리 감쇠

`model/nfplight_real_model.py`의 `NFPLightModel`은 초기화와 `test()`에서 새 함수를 사용한다.
이 파일은 대상 브랜치에 없었으므로 로컬 main의 원본 real 구현을 기반으로 추가했다.
현재 DNG용 `AuthorRealCaptureAdapter`, 학습 경로와 runner는 수정하지 않았다.

- `spatialAttenuationFactor(r_near_squared, r_far_squared)`:
  FP32 `[B,1,H,W]` 거리 제곱에서 `ratio / ratio(center)`를 계산한다.
- `coefficientGenerationSpatialAttenuation()`:
  기존 각도 계수와 분모를 유지하고, real 모델의 pixel-center grid와 near=4/far=12로 거리 지도를 만든다.
- `relationmapExtractionSpatialAttenuation(near, aligned_far)`:
  중심 밝기 gain이 이미 적용된 far에 공간 계수를 한 번 곱하고 기존 relation 함수에 전달한다.

수식은 `20261014_s34_normal_s15_spatial_attenuation` 리포트의
`factor(x) = [r_far²(x)/r_near²(x)] / [r_far²(center)/r_near²(center)]`이다.
256 입력의 기준 픽셀은 리포트와 같은 `[127,127]`이다.
리포트의 합성 geometry(2.414/10, renderer endpoint grid)를 real 모델에 복사하지 않는다.
실제 물리 거리나 crop 범위를 추정하는 기능은 추가하지 않았다.

기존 `coefficientGeneration()`, `coefficientGenerationPaper()`, `relationmapExtraction()`은 보존했다.
기존 각도 계수 setup의 NumPy FP64 계산과 FP32 buffer 변환도 그대로 유지한다.
추가된 거리 계산, 공간 계수, relation과 estimator 입력은 FP32다.
원래 `gain * angular * far` clipping 위치를 유지하면서 공간 계수를 함께 곱한다.
즉 `clamp(gain * angular * factor * far, 0, 1)`이다.
원본 리포트는 clipping이 없는 조건에서 비교했으므로 포화 입력까지 동일한 결과를 보장하지 않는다.

원본 파일의 제거된 network/utility import는 현재 브랜치에서 사용 가능한 동일한 real33 구조와 utility로 연결했다.
저장소 규약에 따라 denoiser를 생성하거나 로드하지 않고 원본 입력을 copy 슬롯에 넣는다.
과거 호환 속성 `pred_denoise_img`도 이 identity copy를 담는다.
33채널 구조와 appearance 슬롯은 유지하고 relation 두 채널만 거리 보정을 적용한다.
호출자는 `NFPLightModel(args)`에 원본 `net_g_real.pth`를 지정하고 `feed_data()`, `test()`를 호출한다.
`args.device`를 생략하면 기존처럼 CUDA를 사용한다.

검증:

```sh
python -m pytest -q tests/test_real_spatial_attenuation.py tests/test_author_real_capture.py tests/test_paper_equations.py
```

CPU 검증은 합성 Lambertian 거리 잔차 소거와 새 함수 호출, FP32, 33채널 구조 및 기존 adapter를 검사한다.
원본 checkpoint 추론이나 실제 DNG 정확도 개선은 검증하지 않았다.
