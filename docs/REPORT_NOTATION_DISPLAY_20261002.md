# 보고서 표기 및 흑백 표시

대상: `artifacts/fabric_capture_20261001_author_original/260930_152732_008/report.html`.
논문 출처: https://cgliwang.github.io/NFPLight/pdf/NFPLight.pdf, Fig. 3, 식 (4)–(5), §3.2.2.

내부 배열 이름과 계산 결과는 보존하고, 선택 메뉴·버튼·영상 제목·계산 설명에
`I_N`, `I_F`, `K_L`, `K_θ`, `K`, `C_M`, `R_M`, `M_R`, `n`, `d`, `r`, `s`를 사용한다.
논문에 없는 센서 복원·정렬·크기 변경 단계는 촬영 거리·순번·작업·해상도로 설명한다.
`time_co_map`은 시간 계수가 아니라 입사각 코사인 비율 `K_θ`다.

논문 분모 `C_M = cos θ_N |cos θ_N − cos θ_F|`와 저장된 저자 코드 분모는 다르다.
화면은 대응 단계를 표시하되 같은 식이라고 주장하지 않는다. 현재 실험의 denoiser 제외,
중앙 20×20 평균 밝기 배율, 관계값의 정규화·반전도 명시한다.

모든 scalar 표시는 검정 0 → 흰색 1이며 감마를 적용하지 않는다.
원래 값 범위 `[a,b]`의 표시값은 `clip((x-a)/(b-a),0,1)`이다.
관계 맵·거칠기·마스크는 0–1, 저장된 신경망 입력 채널은 −1–1을 표시 범위로 사용한다.
음수·1 초과 중간값은 원래 범위를 함께 표시한다. 실제 수치·통계는 바꾸지 않는다.

기존 결과의 표시는 `capture_processing.report.regenerate_scalar_display(output)`로 갱신한다.
이 함수는 저장된 NPZ를 읽고 `scalar_display/`에 표시용 PNG를 생성하며 추론하지 않는다.
수치 NPZ와 실행 manifest의 갱신 전후 SHA256은 `scalar_display_generation.json`에 기록한다.
기존 단계 PNG는 보존하고 활성 갤러리와 비교 표시만 새 흑백 렌더로 갱신한다.
