# 새 학습의 데이터 범위와 NDRS 합 손실

2026-10-11 사용자 지시: “normal 유지, ndrs 평균 아니라 합으로 고정하자.”

## 확정 범위와 과거 실험의 구분

Fabric+Leather 전체 manifest는 train566재질(Fabric358/Leather208), validation164재질(Fabric98/Leather66)이다. Train은 재질당16crop, 총9056crop을 유지한다. 제외 승인된5개 zero-support Leather는 이미 manifest에서 빠져 있다. 공식 test12재질은 학습과 validation에 사용하지 않는다.

최근 bottleneck/regularization/plateau/suite 실행은 manifest 전체를 사용하는 정식 학습이 아니라, `materials_per_category=4`로 선택한 train8재질(Fabric4+Leather4)·heldout8재질의 고정 bank 진단이다. Train16/24slots는8재질×2/3crop이다. 준비된 전체 manifest와 실제 사용한 slots는 다르다. 이 진단 결과를 전체 Fabric+Leather 학습의 결과나 모델 용량의 충분성으로 확대 해석하지 않는다.

`train_fabric_leather.py`의 Bank는 train/val split의 모든 row를 유지한다. 정식 경로는 train566/val164를 확인하고 validation 제한은 짧은 smoke에서만 허용한다. 기존8재질 runner의 immutable config/source/checkpoint와 수치 정의는 과거 lineage로 보존한다.

## 새 목적함수

- 예측 normal은 기존처럼 unit 정규화 후 XYZ L1로 감독한다. Raw XYZ 또는 다른 각도 손실로 변경하지 않는다.
- 각 성분의 L1은 valid pixel과 해당 성분 채널에서 평균한다. **성분 사이에는 평균을 내지 않는다.**
- `L_map = L_normal + L_diffuse + L_roughness + L_specular`이며 네 성분의 계수는 각각1이다.
- S는 현재 fixed-F0 계약의 `.04`(packed `-.92`)를 정답으로 감독한다. 원본 bank specular를 decode하여 감독하는 변경은 이번 적용에 포함하지 않는다.
- 입력 합성과 GT/prediction 양측 rendering의 F0는 `.04`로 고정한다. S head의 map gradient는 허용되지만 rendering에서 S head gradient는 고정 F0 치환으로 제거된다.
- Map 단계는 NDRS 합, Render 단계는 NDRS 합+기존 계수1의render L1이다. Sum 변경에 맞춰 render 계수나 LR을 자동 재조정하지 않는다.
- `map_loss`와 `ndrs_sum`은 네 성분의 합이다. `normal_loss/diffuse_loss/roughness_loss/specular_loss`와 validation의 `map_normal/map_diffuse/map_roughness/map_specular`를 각각 기록한다.
- Best-map 선택 지표는 `ndrs_sum`이다. `ndr_mean`은 과거 그래프와 비교를 위한 별도 기술 지표이며 최적화·best-map 선택에 사용하지 않는다. 기존 viewer의 plateau 표시는 NDR mean의 진단 표시이며 새 종료 정책으로 사용하지 않는다.

계약과 checkpoint format은 v2다. 과거 평균/no-S 실행을 새 sum/S 계약으로 이어 학습하지 않는다. 기존 checkpoint·로그·수치를 덮어쓰거나 변환하지 않는다. 변경 전 전체 trainer/probe 소스는 `artifacts/objective_ndrs_sum_20261011/before_sources/`에 보존했다.

## 구현과 검증·실행 상태

`benchmark_fabric_leather.py:ndrs_loss/update`, `train_fabric_leather.py:evaluate/contract_for/best-map`에 적용했다. 기존 `ndr_loss`와 역사적8재질 pipeline은 보존했다. 고정 F0 렌더 경로는 유지된다.

CPU 테스트는 합·S gradient·invalid support/NaN 거부·normal 유지·실제 optimizer update·평가 합·Bank 전체 선택·v2 계약을 검사한다. GPU 학습·실측 추론·DNG 처리·SHA 검증은 실행하지 않는다.

전체 trainer의 ReduceLROnPlateau 전역 구현 gate는 여전히 남아 있다. 이번 작업은 사용자 손실 계약 적용이며 scheduler 구현·전체 학습 시작 완료를 뜻하지 않는다. 기존 정지 실행을 자동 재개하지 않는다. 새 전체 학습에는 현재 [scheduler 정책](TRAINING_SCHEDULER_POLICY.md)의 구현 gate를 먼저 충족해야 한다.


## 사용자 설명에 따른 샘플 단위 명확화

사용자의 여러 crop 지시는 같은 계열 이름의 여러 폴더에서 각 폴더의 각 샘플마다 여러 crop을 활용한다는 뜻이다. 현재 manifest의 material_id는 원본 샘플 폴더 식별자이며 고유 물질 종류 수를 뜻하지 않는다. 같은 family/group_id의 폴더들도 각각 학습 대상이고 대표 하나로 합치지 않는다. Group은 split 분리에만 사용한다. train566개 원본 샘플 폴더 각각의 N/D/R/S 배열 헤더에서16crop을 확인했다(총9056crop). 검사는 헤더만 읽었으며 픽셀·공식 test·SHA는 검사하지 않았다. [폴더별 검사 기록](../artifacts/full_lightweight_nine_20261011/sample_folder_crop_audit.json)을 따른다.
