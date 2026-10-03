> Historical research record. Current capture contract: [CAPTURE_REQUIREMENTS.md](../CAPTURE_REQUIREMENTS.md). Current commands: [README.md](../../README.md). Code citations and run status below describe their recorded version; removed code and old line numbers are not current implementation claims.

# NFPLight — MatSynth From-Scratch 학습 데이터 구성 초안

> **2026-09-12 범위 수정:** 현재 사용자 목표는 **Fabric만 사용하는 from-scratch estimator 학습**이며 DenoiseNet은 명시적으로 제외한다. 최신 설계와 코드 점검은 [Fabric 전용 학습 설계](NFPLight_fabric_scratch_plan.md)를 따른다. 아래 내용은 이전 전체 category 실험 기록으로 보존한다. 특히 Metal 중심 평가, 고정 LR에 대한 논문 해석, LDR/log1p/off-colocated 설명을 현행 구현의 사실로 인용하지 않는다.

**범위.** 파이프라인은 두 학습 단으로 갈린다: `DenoiseNet`(첫 단, noise 부분, [nfplight_net.py:374](../../network/nfplight_net.py:374))과 SVBRDF 추정기 `TwoBranchNet`(후반부, 21-ch, [nfplight_net.py:69](../../network/nfplight_net.py:69)). 이 문서는 **noise 단을 범위에서 제외**하고, 후반부 추정기를 DeepMaterials가 아니라 MatSynth로 **처음부터** 학습시키기 위한 데이터 구성만 다룬다. 기존 합성 학습 경로 `SynDataset`([dataset.py:8](../../data/dataset.py:8))을 대체한다. Optics·capture 물리(vignette, flash baseline, rig)는 데이터로 못 고치므로 비목표 — 레드팀 치명 항목 참조([NFPLight_redteam.md](NFPLight_redteam.md)).

기존 v0 probe(`finetune_fabric.py`)와의 차이: v0은 **사전학습 net의 LoRA finetune + map-L1 only**. 이 초안은 **전 파라미터 from-scratch + render loss 포함**. loader·renderer·21-ch 조립은 v0에서 그대로 재사용한다([finetune_fabric.py:249](../../finetune_fabric.py:249)).

**데이터 소스(disk 확인).** `D:\MatSynth_cropped`는 없음. 실제 후보:
- `D:\MatSynth_preprocessed` — 재질별 full-res(1–2K) `diffuse/specular/normal/roughness/metallic/basecolor/opacity.png` + metadata. specular workflow **직접** 사용 가능(§2-B 첫 경로).
- `D:\MatSynth_crops256` — 256 npz, `train/test/<Category>/*.npz`, key = **basecolor/metallic/roughness/normal** (specular·diffuse **없음**). F0/diffuse를 metallic workflow로 유도해야 함(§2-B 둘째 경로). 이미 256 crop이라 학습 iteration은 가장 빠름.
- `D:\DeepMaterialsData_cropped` — 비교 baseline(Diffuse/Normal/Roughness/Specular 폴더). specular = 상수 0.043.

**확정: crops256 + on-the-fly F0/diffuse 유도**(소스-무관, sRGB-specular 함정 회피, 가장 빠름). preprocessed는 §5 native crop / 고해상 eval이 필요할 때만.

---

## 1. DeepMaterials 한계 → MatSynth 대응

| 한계 (DeepMaterials) | 원인 | MatSynth 대응 | NFPLight 접점 |
|---|---|---|---|
| specular 부정확 | DeepMaterials specular은 **constant ~0.043 linear, achromatic — dielectric 전용**. metal/high-F0를 표현 못함(측정 확인) | MatSynth specular은 dielectric **0.04** → metal **0.4+** F0 범위를 실제로 커버(steel 예: basecolor_lin 0.44, metallic 0.89 → F0 0.40) | `s`는 render에서 Fresnel F0로 직접 사용([render_util.py:344](../../utils/render_util.py:344)) — F0 범위가 맞아야 highlight 강도·색이 맞음 |
| distribution 제한 | 소수 절차적 재질군, category 편향 | 4069 재질 / 13 category(Ceramic…Wood), 물리 크기 metadata | category-balanced sampling, from-scratch broad coverage |
| resolution | 저해상 crop, thread detail 소실 | 1K–4K native | native-res crop 유지(다운샘플 금지) |

**측정 근거(disk 확인).** DeepMaterials specular = 상수 0.043(std≈0, chroma=0) — 전부 dielectric으로 고정. MatSynth specular = dielectric ~0.04, metal ~0.10–0.40+, `lerp(0.04, basecolor_lin, metallic)`이 MatSynth 자체 specular과 일치. 즉 specular F0 **범위·분포**가 이 전환의 핵심 실측 이득이다(개별 픽셀 색이 아니라 F0 span). 나머지(vignette 등)는 capture-side라 학습으로 못 건드린다.

> **경고 — 현재 loader는 이 이득을 못 살린다.** `_build_svbrdf`([finetune_fabric.py:176](../../finetune_fabric.py:176))는 MatSynth diffuse/specular를 **sRGB 그대로** `*2-1`한다. MatSynth map은 sRGB 인코딩(specular 0.247 sRGB = 0.045 linear)이라, decode 없이 쓰면 GT specular가 **5–6× 과대**, diffuse도 과대. §2-A에서 수정.

---

## 2. GT map → NFPLight svbrdf 변환 (crux)

목표 텐서: `ndrs` 순, `[-1,1]`, 10-ch (n3 d3 r1 s3). d·s는 **linear**여야 함(renderer가 `d*0.5+0.5`, `s*0.5+0.5`를 linear albedo/F0로 사용, [render_util.py:319](../../utils/render_util.py:319)). 검증해야 할 gotcha:

**A. sRGB→linear decode (필수, 현재 버그).** MatSynth `basecolor/diffuse/specular`는 sRGB. GT로 넣기 전 `^2.2`. 안 하면 specular 5–6×·diffuse ~2× 과대. `normal/roughness/metallic/height`는 linear라 decode 금지. (roughness는 macro-BRDF라 gamma를 걸지 않는 게 관례 — MatSynth도 linear.)

**B. 금속 F0 — workflow 두 경로 (등가, 소스에 따라 선택).**
   - `MatSynth_preprocessed`: `diffuse.png`+`specular.png`(specular-workflow) **직접 사용**(단 A의 decode 후). MatSynth가 이미 metallic을 접어둠.
   - `MatSynth_crops256`: npz에 **specular 없음**(basecolor/metallic/roughness/normal뿐). 직접 유도 —
     ```
     bl = basecolor**2.2                 # sRGB->linear
     F0      = 0.04*(1-metallic) + bl*metallic     # -> s
     diffuse = bl*(1-metallic)                     # -> d
     ```
     **측정(test_crops_loader.py [b]):** dielectric F0 오차 **0.003(≈정확)**, metal F0 오차 **0.05–0.08**. mean은 맞지만 metal per-pixel은 MatSynth specular.png(plain lerp 아님)와 drift. Metal은 train의 12.5%(716/5699). metal 정밀도가 필요하면 **Metal category만 preprocessed specular.png/diffuse.png 사용**(dielectric은 유도로 충분). clip 없이 raw F0 보존.
     주의: preprocessed 맵은 **type별 해상도가 다를 수 있음**(basecolor 2048 vs metallic 4096) — 교차검증 시 공통 res로 resize. crops256은 전부 256이라 무관.

**C. normal Y 규약 — 측정 완료.** height→normal 상관(test [c]): corr(nx,dH/dx)=−0.95, corr(ny,dH/drow)=+0.92 → MatSynth normal은 real heightfield-consistent. 절대 OpenGL/DirectX는 colocated render가 Y-무감(치명4)이라 이걸로 확정 못함. **결정적 증거는 test [d]**: net_g_syn(DeepMaterials 학습)의 MatSynth GT normal L1=**0.039**(낮음) → DeepMaterials·MatSynth가 **같은 규약** → **base Y-flip 불필요**. vflip aug의 `n[...,1]` 부호 반전은 규약과 무관하게 맞음. (real 배포 규약은 실측 1장으로 최종 확인.)

**D. roughness.** GGX `alpha=r*r`([render_util.py:301](../../utils/render_util.py:301)) — perceptual roughness. MatSynth roughness도 perceptual/linear라 **직접 매칭**(α 저장이면 √). decode 안 함.

**E. opacity/cutout.** opacity<1 재질은 alpha hole이 diffuse에 구멍 — §3에서 제외 또는 composite.

---

## 3. 데이터 위생 — 제외할 재질 ("noise" 재질)

후반부 단서(near/far falloff)가 물리적으로 존재하는 재질만 남긴다. from-scratch 라벨 품질이 그대로 상한이 됨.

- **opacity<1** (cutout/투명) — 제외.
- **degenerate**: flat normal + near-uniform albedo → falloff 외 단서 0, 학습에 기여 없음. normal 분산·albedo 분산 threshold로 필터.
- **순수 절차적 noise 텍스처**: 구조 없는 high-freq noise map은 crop마다 label이 무의미 → 제외 또는 저비중.
- **saturated F0 오류**: specular map이 전면 1.0인 잘못 태깅 재질 검출.

산출: `keep_list.txt`(재질 경로 + category + 통과 사유). 무엇을 얼마나 버렸는지 로그로 남길 것(silent truncation 금지).

---

## 4. 구성 / 샘플링 (distribution 수정)

- **category-balanced.** v0의 `fabric_tilt`는 from-scratch에선 끈다(=1.0). 13 category 균등 기대질량([finetune_fabric.py:230](../../finetune_fabric.py:230)의 `balanced_weights` 재사용, tilt 제거).
- **material-level split.** crop leakage 방지([finetune_fabric.py:214](../../finetune_fabric.py:214)). MatSynth official train/test 우선, val은 train에서 category별로.
- **crop budget.** 재질당 crop 수를 category 크기에 반비례로 상한(대형 category가 step을 독점하지 않게).

---

## 5. crop & augmentation (resolution 수정, noise 배제)

- **native-res random crop, 다운샘플 절대 금지.** uint8 읽고 crop window slice 후 float32([finetune_fabric.py:118](../../finetune_fabric.py:118)) — 4K resize OOM 회피 + thread detail 보존.
- **multi-scale는 기본 OFF.** 큰 window를 256으로 줄이면 그게 곧 다운샘플 = thread detail 소실(다운샘플 금지 규칙 위반). crops256은 이미 native 256이라 애초에 큰 window가 없음. scale-invariance가 실측에서 부족하다고 확인될 때만 preprocessed에서 native crop 크기를 바꿔 켠다.
- **flip only** (h/v), normal 부호 동반 반전. rotation은 normal 재정의 필요하니 v0에선 생략.
- **photometric noise 없음.** denoise 단 제외이므로 sensor noise·exposure jitter를 추정기 입력에 **넣지 않는다**. LDR 8-bit 양자화만 유지([nfplight_model.py:75](../../model/nfplight_model.py:75)). 실측 시 별도 학습된 denoiser가 앞단에서 정리 — 그 domain-gap은 noise 단의 책임(범위 밖).

---

## 6. 입력 렌더링

추론과 **동일 조립**을 재사용해 train/infer skew 0:

- `render_input_images(svbrdf, toLDR=True)` — colocated GGX near/far 렌더 + LDR 양자화([nfplight_model.py:67](../../model/nfplight_model.py:67)).
- `build_net_input` — scale-align → relation/log_relation → mask → 21-ch(18 albedo + 3 specular)([finetune_fabric.py:249](../../finetune_fabric.py:249)). grad 유지.
- syn geometry near=2.414 / far=10 사용(추정기가 syn net이므로). real geometry(4/12)는 real net 경로라 별건.

즉 데이터 파이프라인 = **MatSynth GT 추출(§2·3) + on-the-fly colocated 렌더(§6)**. 저장은 GT map만; 입력은 매 step 렌더(디스크 절약 + geometry 변경 자유).

---

## 7. Loss (from-scratch — v0의 map-L1-only를 넘어서) — ✅ 구현: [train_scratch.py](../../train_scratch.py)

v0는 "map-space L1 only, render loss는 설계상 제외"였다([finetune_fabric.py:6](../../finetune_fabric.py:6)). from-scratch는 render loss가 있어야 spec/rough가 잡힌다. `loss = map_loss + render_w·novel_light_render_loss`.

- **map loss**: per-map L1(n,d,r,s), normal은 unit-vector 공간에서([finetune_fabric.py:265](../../finetune_fabric.py:265) 재사용).
- **novel-light render loss**: 예측·GT svbrdf를 **입력엔 없던 랜덤 광원 K개**로 재렌더 후 비교(`render_colocated`, log1p tonemap으로 highlight 압축). colocated near/far 1쌍으론 미관측인 specular lobe·roughness를 **다른 조명에서 일치하도록** 강제. **repeat-capture 일관성(§9)과 같은 의도** — "조명 바꿔도 같은 재질". 추론은 여전히 near/far colocated이므로 이건 측정이 아니라 **prior 강화**.
- **specular(F0)**: map loss의 `s` 항이 담당. MatSynth F0 **범위**(dielectric 0.04 ↔ metal 0.4+)를 감독 — 이득은 hue 아니라 **F0 magnitude 분포**(DeepMaterials 0.043 고정 대비). 유색 금속엔 hue 항 추가 여지.
- **입력 열화 augmentation**은 아직 미배선(§10 syn-real gap lever) — `train_scratch.py`의 aug hook에 colored flash·vignette·LDR를 붙이는 자리 표시만.

## 8. 학습 프로토콜

- 전 파라미터 from-scratch(LoRA 아님). `set_trainable`/`inject_lora` 미사용.
- AdamW, cosine, warmup. batch·steps는 category 균등 sampler로.
- seed 고정, material-level split. checkpoint는 `checkpoints/`(로컬).

## 9. 검증 / gate

**진짜 문제 = syn-real gap.** syn map L1은 saturate(test [d]가 낮음 = 변별력 없음), real엔 GT 없음. 그래서 두 층으로:

*syn 층 (GT 있음, prior 품질):*
- **stratified 필수.** test [d]: aggregate map L1이 낮아 DeepMaterials 한계가 안 보임. gate는 **Metal 분리 + dielectric/metal F0 stratum**으로. (train_scratch.py eval이 per-category + Metal specular 출력.)
- per-category map error + render loss(held-out category 포함).

*real 층 (GT 없음, gap 측정):*
- **repeat-capture 일관성 (핵심 GT-free 지표).** 같은 재질을 노출·각도·조명 바꿔 N번 촬영 → 출력 SVBRDF **분산**. 낮아야 좋음. **§7 novel-light render loss의 real 판(같은 의도)** — train은 synthetic 조명 일치를 감독, eval은 real 조명 일치를 측정.
- **specular chroma proxy**: 알려진 비금속 real에서 예측 specular 색도 → 낮아야 정상(colored-specular 실패 직접 정량).
- **roughness sanity**: 알려진 matte/glossy 평판 r이 GT를 따르는지(레드팀 test 4).

real을 학습·supervision 소스로는 쓰지 않음(GT 없음). aug fidelity는 원리상 검증 불가 — repeat-capture 일관성만 GT 없이 측정 가능.

## 10. 정직한 한계 (레드팀 연결)

- **roughness 미관측(치명4)**: colocated 단일 config는 flat matte에서 r을 못 본다. MatSynth는 **r prior 품질**만 올림 — 측정으로 바꾸지 못함. render loss의 off-colocated 표본도 prior일 뿐.
- **specular F0 범위(진짜 이득, 단 aggregate엔 안 보임)**: MatSynth는 dielectric~metal F0 span을 담아 metal highlight가 물리적으로 맞아짐 — DeepMaterials(0.043 고정) 대비 실측 개선. **하지만 test [d]에서 aggregate map L1은 이미 낮음** — 이득은 평균이 아니라 **metal/high-F0 stratum + distribution 폭 + resolution**에서 나온다. 논문 주장·gate를 여기에 맞춰라(§9). §2-A decode 빠뜨리면 이 이득 통째로 뒤집힘(과대 specular).
- capture-side(vignette·baseline·rig)는 전부 범위 밖 — 데이터가 아니라 계측 문제.

## 11. 다음 한 걸음

1. ✅ **완료·검증**: crops256 loader [data/matsynth_crops.py](../../data/matsynth_crops.py) + 사전점검 [test_crops_loader.py](../../test_crops_loader.py). 4-check ALL PASS(가마 decode·유도·normal-Y·round-trip). full train 전 재실행:
   ```
   python test_crops_loader.py --crops D:/MatSynth_crops256 --prep D:/MatSynth_preprocessed --loadpath checkpoints/net_g_syn.pth
   ```
2. §3 재질 필터로 `keep_list.txt`(opacity<1·degenerate·noise 재질 제외, 버린 수 로그). crops256는 opacity 키가 없으니 preprocessed opacity.png 참조 또는 생략.
3. ✅ **완료·smoke PASS**: from-scratch 학습 [train_scratch.py](../../train_scratch.py) — `ScratchModel`(random-init net, 렌더 machinery 재사용) + map loss + **novel-light render loss**(§7) + category-balanced(no tilt) + material-level split + stratified eval(Metal 분리). 2-step smoke: loss 2.48→2.22 finite. 실행:
   ```
   python train_scratch.py --crops D:/MatSynth_crops256 --out checkpoints/net_g_matsynth.pth --steps 60000 --batch 8
   ```
4. **아직 안 됨**: (a) §3 재질 필터 `keep_list.txt`, (b) 입력 열화 augmentation hook(colored flash·vignette·LDR — colored-specular 잡는 lever), (c) real **repeat-capture 일관성** eval 스크립트(§9 real 층), (d) `harness/contracts/` train contract.
