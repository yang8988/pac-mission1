# IRAP: 재고 인지형 강건 적응 팔레타이징 알고리즘

**IRAP** (Inventory-aware Robust Adaptive Palletizing)
PAC 2026 HD현대로보틱스 과제 1 「불확정 투입 순서에 대응하는 Mixed Palletizing 적재 패턴 생성」 · 팀 LYL

---

## 0. 한눈에 보기

> **"무엇이 안전한가"는 규칙이 정하고, "안전한 것 중 무엇이 미래에 유리한가"는 학습과 탐색이 정한다.**

박스가 하나 들어올 때마다 아래 순서로 결정합니다.

1. 후보 위치를 **규칙으로 생성**합니다(Extreme Point).
2. 하드 제약으로 **마스크**를 만들어 불가능한 후보를 제거합니다.
3. **즉시 점수**와 **정책망**으로 유망한 후보만 남깁니다.
4. **남은 재고로 만든 여러 투입 순서 시나리오**에서 롤아웃해 **평균과 최악 성능**을 함께 평가합니다.
5. **로봇 실행 가능성**을 검증한 뒤 **한 개만 실행**하고, 실제 상태를 다시 관측합니다.

학습 단계에서는 이 롤아웃 플래너를 "교사"로, 신경망을 "학생"으로 삼아 반복적으로 함께 개선합니다(PPO 사전학습 → Expert Iteration).

### 설계 원칙

| # | 원칙 | 구현 |
|---|---|---|
| P1 | 안전은 학습에 맡기지 않는다 | 규칙 기반 하드 제약 → 행동 마스크 (위반 확률 0) |
| P2 | 알고 있는 정보는 모두 쓴다 | 관측된 가까운 미래는 그대로, 먼 미래는 **잔여 재고**로 시나리오 생성 |
| P3 | 평균만이 아니라 최악도 본다 | 시나리오 CVaR + 막힘 확률을 평가에 포함 |
| P4 | 항상 제시간에 답한다 | 시간 예산 기반 anytime 탐색 (Successive Halving) |
| P5 | 한 번에 하나만 확정한다 | receding horizon: 1개 실행 → 실측 보정 → 재결정 |
| P6 | 재계획은 필요한 만큼만 | 지지 그래프 기반 영향 범위 분석 |
| P7 | 코어는 ROS와 분리한다 | `palletizing_core`(순수 Python) + ROS2 래퍼 |

---

## 1. 문제 정의와 가정

### 1.1 입력
- 팔레트: 길이 `L`, 폭 `W`, 최대 적재 높이 `H_max`, 최대 적재 중량 `M_max`
- 박스 종류 `t ∈ T`: 치수 `(w_t, d_t, h_t)`, 공칭 무게 `m_t`, 허용 상부하중 `F_t`, 허용 자세 집합 `O_t`, 수량 `n_t`
- 로봇: 베이스 위치, 가반하중, 그리퍼(흡착 패드) 크기, 픽업 위치
- (선택) 상류 카메라로 관측한 다음 `k`개 박스 (`k = 0`도 지원)

### 1.2 모르는 것
- 실제 투입 순서 σ (종류와 수량은 알지만 순서는 모름)

### 1.3 결정 변수 (박스 한 개당)
`a = (x, y, o)`: 바닥 기준 xy 위치와 자세. `z`는 높이맵에서 자동으로 결정됩니다(위에서 내려놓기).
또는 예외 행동 `{REJECT, HOLD}`.

### 1.4 기본 가정 (설정으로 변경 가능)
- 로봇이 위에서 내려놓는 top-down 적재, 흡착 그리퍼
- 자세: z축 기준 0°/90° 회전. 눕히기는 박스 종류별로 허용 여부 지정
- 임시 보류 구역(HOLD)은 기본값 0칸. 현장에 있으면 1~2칸 사용
- 단위: mm, kg, N

### 1.5 목표
모든 단계에서 하드 제약을 만족하면서, 아래 항목을 종합적으로 최적화합니다.
**최종 적재율 ↑, 최악 순서에서의 적재율 ↑, 안정성 여유 ↑, 무게중심 편차 ↓, 로봇 작업시간 ↓, 박스당 결정 시간 ≤ 예산**

---

## 2. 전체 구조

```
                     ┌──────────────────── 오프라인 학습 (§6) ───────────────────┐
                     │  Gym 환경 ─▶ PPO 사전학습 ─▶ Expert Iteration 반복        │
                     │       (π_θ 정책망, V_θ 가치망) ──────────────┐            │
                     └──────────────────────────────────────────────┼───────────┘
                                                                    ▼ 모델 파일
 ┌─────────── 온라인 의사결정 (박스 1개마다) ───────────────────────────────────────┐
 │                                                                                │
 │ [M1 관측·재고] ─ 이상? ─▶ [M9 예외 처리]                                         │
 │      │ 정상                                                                     │
 │      ▼                                                                         │
 │ [M2 후보 생성: EP × 자세] ─▶ [M3 하드 제약 마스크 (Tier 1)]                      │
 │      ▼                                                                         │
 │ [M4 즉시 점수 Q_now] + [M5 정책망 prior π_θ] ─▶ 상위 K0개                        │
 │      ▼                                                                         │
 │ [M6 강건 미래 평가: 재고 시나리오 × 롤아웃(+V_θ) → 평균/CVaR/막힘]               │
 │      ▼  (시간 예산 내 Successive Halving)                                       │
 │ [M7 최종 점수 → 순위 목록]                                                      │
 │      ▼                                                                         │
 │ [M8 로봇 검증 (Tier 2: IK·충돌·그리퍼)] ─ 첫 통과 후보 실행                      │
 │      ▼                                                                         │
 │ [M9 실측 보정 · 영향 범위 재검증 · 부분 재계획] ─▶ 다음 박스                      │
 └────────────────────────────────────────────────────────────────────────────────┘
   ※ 안전 정지(사람 접근·설비 이상)는 모든 단계보다 우선 (L0 Safety)
```

---

## 3. 상태 표현

| 이름 | 내용 | 용도 |
|---|---|---|
| `H[i,j]` | 높이맵, 격자 `g` = 10 mm | 충돌·지지·후보 계산, 신경망 입력 |
| `placed[]` | 놓인 박스: AABB, 실측 무게, 허용하중 `F`, 현재 받는 하중 `load` | 하중 전파, 재계획 |
| `G_sup` | 지지 그래프: 간선 `(아래 → 위, 접촉면적)` | 하중 전파, 영향 범위 분석 |
| `EP` | Extreme Point 집합 | 후보 생성 |
| `inv` | 종류별 잔여 수량 벡터 `n_t` | 시나리오 생성, 신경망 입력 |
| `queue` | 관측된 다음 박스 목록 (길이 `k` ≥ 0) | 가까운 미래 |
| `COG`, `M` | 팔레트 전체 무게중심, 총 중량 | 안정성 제약 |

신경망 입력 텐서:
- 높이맵 `H / H_max` (1채널)
- 현재 박스 특징 `(w, d, h, m, F)` 정규화값
- 재고 벡터 `inv / n_total`, 진행률 `ρ = 배치 완료 수 / 전체 수`
- 후보별 특징 벡터 (§4.4의 `f1..f8` + 위치·자세)

---

## 4. 온라인 모듈 상세

### M1. 관측과 재고 갱신
```
측정 → 종류 분류 → 치수·무게 실측값 비교
  |Δ치수| > tol_dim  또는  |Δ무게| > tol_m  또는  변형·파손 감지  → M9 예외
  정상: inv[t] -= 1, queue 갱신, 실측 무게로 박스 정보 갱신
```

### M2. 후보 생성
1. **Extreme Point**: 박스를 놓을 때마다 그 박스 모서리 `(x+w, y)`, `(x, y+d)`를 x·y 방향으로 투영해 EP를 추가합니다. 덮인 EP는 제거합니다.
2. **보조 후보**: 팔레트 네 모서리 정렬점, 높이맵 단차 경계점, 놓인 박스 윗면 모서리
3. **자세 확장**: `C = { (x, y, o) | (x,y) ∈ EP ∪ 보조점, o ∈ O_t }`. 각 자세에서 박스 모서리가 기준점에 붙도록 네 방향 앵커를 시도합니다(OPAL의 다중 앵커 아이디어).
4. **z 결정**: `z = max(H[footprint])` (내려놓기)

전형적인 후보 수는 50~300개입니다.

### M3. 하드 제약 마스크 (Tier 1, 모든 후보)
계산이 싼 검사부터 순서대로 수행해 빨리 탈락시킵니다. 하나라도 실패하면 `mask = 0`입니다.

| ID | 제약 | 판정식 |
|---|---|---|
| H1 | 경계·높이 | `x≥0, y≥0, x+w≤L, y+d≤W, z+h≤H_max` |
| H2 | 관통 없음 | `z = max(H[fp])`이므로 자동 만족 (재보정 후에는 AABB 교차로 재확인) |
| H3a | 지지면적 | `s = |{c∈fp : H[c] ≥ z − ε}| / |fp| ≥ θ_s` (z=0이면 s=1) |
| H3b | 무게중심 지지 | 박스 COG의 xy 투영 ∈ 지지 셀의 볼록 껍질을 `δ_cog`만큼 안쪽으로 줄인 영역 |
| H4 | 하중 전달 | 아래 하중 전파 결과 모든 박스에서 `load + Δ ≤ F` |
| H5 | 팔레트 COG | `‖COG_xy − 중심‖ ≤ r_max(ρ)`, `r_max`는 진행률 ρ에 따라 점점 줄어듦 |
| H6 | 총 중량 | `M + m ≤ M_max` |
| H7 | 간이 도달성 | 사전 계산한 도달 가능 영역 맵 `R(x,y,z,o) = 1` |
| H8 | 그리퍼 접근 여유 | 박스 footprint를 그리퍼 여유만큼 넓힌 영역에서 `max H ≤ z + h + c_app` |

**하중 전파 (H4)**: 접촉면적 비례 분배로 아래로 재귀 전파합니다(보수적 근사).
```python
def propagate(box, F, delta):           # F: box가 아래로 전달하는 힘
    total = sum(area for _, area in G_sup.below(box))
    for below, area in G_sup.below(box):
        f = F * area / total
        delta[below] = delta.get(below, 0) + f
        if below.load + delta[below] > below.F:  return False
        if not propagate(below, f, delta):        return False
    return True
# 새 박스: propagate(new_box, m_new * g, {})
```
"무거운 박스를 가벼운 박스 위에 올리지 않기"는 이 규칙 안에 자연스럽게 포함됩니다. 허용하중 데이터가 없으면 `F_t = κ · m_t · g`(예: κ = 3)를 기본값으로 씁니다.

### M4. 즉시 점수 `Q_now`
후보별 특징(모두 0~1로 정규화):

| 특징 | 의미 | 방향 |
|---|---|---|
| f1 지지율 | `s` (H3a) | + |
| f2 접촉률 | 측면이 벽·이웃과 닿는 둘레 비율 | + |
| f3 높이 증가 | `(max H 이후 − max H 이전) / H_max` | − |
| f4 요철도 | 배치 후 높이맵 인접 셀 높이차 합의 증가량 | − |
| f5 COG 편차 | 배치 후 팔레트 COG 편차 / r_max | − |
| f6 작업시간 | 픽→플레이스 예상 시간(거리, 회전) / t_ref | − |
| f7 낮은 위치 | `z / H_max` | − |
| f8 접근 순서 | 로봇에서 먼 곳부터 채우는 정도 (가까운 곳을 먼저 막으면 감점) | + |

```
Q_now(a) = Σ_i w_i · f_i(a)        (부호는 위 표의 방향 반영)
```
- 가중치 `w`는 벤치마크로 튜닝합니다(Optuna 등).
- `Q_now`는 ① 1차 거르기, ② 학습 0회차의 롤아웃 정책, ③ 신경망 입력 특징으로 쓰입니다.

### M5. 신경망: 정책망 π_θ와 가치망 V_θ
하나의 네트워크가 두 출력을 냅니다(Actor-Critic).
```
높이맵 ──CNN──┐
박스·재고·ρ ─MLP─┼─▶ 전역 임베딩 g ──▶ V_θ(s)          (상태 가치: 최종 적재율 예측)
후보 특징 ─MLP─▶ 후보 임베딩 c_j ─(g와 cross-attention)─▶ logit_j ─마스크─▶ softmax = π_θ(a|s)
```
- 마스크: `logit_j = −∞ if mask_j = 0`
- 후보 수가 바뀌어도 처리되도록 패딩 + 마스크를 씁니다.
- 온라인에서의 역할:
  - **prior**: `Q_now + η·log π_θ`로 상위 K0개를 선정합니다.
  - **롤아웃 정책**: 롤아웃할 때 π_θ의 argmax로 빠르게 가상 적재합니다.
  - **잎 평가**: 롤아웃을 깊이 D에서 끊고 나머지는 V_θ로 평가합니다.
- 모델이 없을 때(학습 전)는 `π_θ` 대신 `softmax(Q_now / τ)`, `V_θ` 대신 남은 부피 기반 단순 추정치를 쓰도록 대체 경로를 둡니다.

### M6. 강건 미래 평가 (핵심)

**(1) 시나리오 생성: 재고 인지형**
```
σ_k = queue (관측 순서 고정)  ⊕  shuffle_k( inv − queue )      k = 1..K
```
- 모든 후보가 **같은 시나리오 집합**을 공유합니다(common random numbers). 후보 간 비교의 분산이 줄어듭니다.
- 순서 분포에 대한 사전 지식(같은 종류가 몰려서 들어오는 경향 등)이 있으면 그 분포에서 샘플링합니다. 없으면 균등 셔플을 씁니다.
- 평가 견고성을 높이려면 시나리오의 일부(예: 20%)를 **불리한 순서**(큰 박스가 늦게 오는 등)로 채웁니다. AR2L의 순서 공격자 아이디어를 가져온 것입니다.

**(2) 롤아웃**
```python
def rollout(state, a, sigma, D):
    s = apply(state, a)
    blocked, first_block = 0, None
    for step, box in enumerate(sigma[:D]):
        cands = mask(generate(s, box))                 # M2+M3 (Tier 1만)
        if not cands:
            blocked += 1; first_block = first_block or step; continue
        s = apply(s, best(cands, policy=π_θ or Q_now))
    tail = V_θ(s) if len(sigma) > D else util(s)        # 잎 평가
    return tail, blocked, first_block
```

**(3) 후보의 미래 가치**
```
U_k        = 롤아웃 k의 최종 적재율 (또는 V_θ 추정치)
V_mean(a)  = mean_k U_k
V_cvar(a)  = 하위 α 비율 시나리오들의 U_k 평균          (예: α = 0.2)
P_block(a) = mean_k 1[blocked_k > 0]

V_future(a) = (1−λ)·V_mean + λ·V_cvar − μ·P_block
```

**(4) 시간 예산 안에서 계산: Successive Halving**
```
후보 = 상위 K0개 (예: 16)
while 시간 남음 and 후보 > 1:
    각 후보에 시나리오 K_r개 추가 롤아웃 (공유 시나리오, 병렬)
    Score 하위 절반 제거
반환: 현재 Score 기준 순위 목록 (시간이 끊겨도 항상 답이 있음)
```

### M7. 최종 점수와 순위
```
Score(a) = Q_now(a) + β(ρ) · V_future(a)
β(ρ) = β0 · (1 − ρ) + β_min        # 초반엔 미래 비중 ↑, 막바지엔 ↓
```
결과는 **순위 목록**으로 반환합니다. 1순위 후보가 M8에서 탈락하면 다시 계획하지 않고 바로 2순위를 검사합니다.

### M8. 로봇 실행 가능성 검증 (Tier 2, 상위 후보만)
순위대로 검사하다 처음 통과한 후보를 실행합니다.
1. **IK**: approach(위 c_app) → place → retreat 세 자세 모두 해가 존재하고 관절 한계 이내일 것
2. **충돌**: 접근·하강·후퇴 경로에서 그리퍼+박스가 쓸고 지나가는 공간과 놓인 박스·팔레트·설비 사이 충돌 없음 (MoveIt2 planning scene)
3. **그리퍼 이탈**
   - 흡착 패드가 박스 윗면을 덮는 비율 ≥ θ_pad
   - 편심 모멘트 `m·g·e` + 관성 `m·a_max` ≤ 흡착 한계 / 안전계수
   - 가반하중 이내
4. 모두 실패하면 → HOLD(보류 구역이 있으면) → 없으면 REJECT, 그리고 실패 사유를 기록

### M9. 실행 후 보정, 예외 처리, 부분 재계획
**실측 보정**: 배치 후 비전으로 실제 자세를 측정하고 `placed`, `H`, `G_sup`, `EP`를 갱신합니다.

**영향 범위 분석**: 문제 박스 b에 대해 `Affected = {b} ∪ G_sup에서 b 위로 이어진 모든 박스`. 이 집합에만 H3, H4를 다시 검사합니다.

| 상황 | 판정 | 대응 |
|---|---|---|
| 심한 파손·변형 | M1 | REJECT (리젝트 라인). `inv` 갱신 |
| 경미한 변형 | M1 | `F = 0`으로 설정 → 위에 아무것도 올리지 않음 → 상단 배치로 자연스럽게 유도 |
| 규격 오인식 (배치 전) | M1/M8 | 실측 치수로 M2부터 다시 실행 |
| 규격 오인식 (배치 후) | M9 | 높이맵 보정 → Affected 재검증 |
| 위치·자세 오차 ≤ tol | M9 | 상태만 보정 |
| 위치·자세 오차 > tol, 또는 재검증 실패 | M9 | 재파지 후 재배치 (그 박스만 다시 결정) |
| 집기 실패·낙하 | M8/M9 | 재시도 1회 → 실패 시 REJECT, `inv` 보정 |
| 박스 누락·예상 밖 박스 | M1 | `inv` 갱신 (시나리오는 다음 결정부터 자동 반영) |
| 팔레트 규격·박스 구성 변경 | 설정 | 설정만 바꿔 재시작. 알고리즘 수정 없음 |
| 사람 접근·설비 이상 | L0 | **즉시 안전 정지**. 재계획보다 우선 |

결정을 매번 새로 하는 구조이므로, "부분 재계획"은 사실상 **상태 보정 + 영향 범위 재검증 + 해당 박스만 재결정**으로 끝납니다. 전체를 다시 계산하지 않습니다.

---

## 5. 온라인 메인 루프

```python
def on_box(S, obs, budget):
    b = classify_and_measure(obs)                         # M1
    if abnormal(b):  return handle_exception(S, b)        # M9
    S.inv.consume(b.type)

    C = generate_candidates(S, b)                         # M2
    C = [c for c in C if tier1_feasible(S, b, c)]         # M3
    if not C:  return hold_or_reject(S, b)

    for c in C:
        c.q = Q_now(S, b, c)                              # M4
        c.prior = log_pi(S, b, C, c)                      # M5
    C = top_k(C, key=lambda c: c.q + η * c.prior, k=K0)

    scen = sample_scenarios(S.queue, S.inv, K)            # M6-1
    ranked = successive_halving(S, C, scen, budget)       # M6-2~4, M7

    for c in ranked:                                      # M8
        if tier2_feasible(S, b, c):
            execute(c)
            S = observe_and_correct(S, c)                 # M9
            revalidate_affected(S, c)
            log_sample(S, C, ranked)                      # 학습 데이터 축적
            return
    hold_or_reject(S, b)
```

---

## 6. 학습 파이프라인

### 6.1 환경 (`PalletEnv`, Gymnasium)
- M1~M4를 그대로 재사용합니다. ROS 없이 병렬 실행합니다.
- **에피소드**: 박스셋 생성기로 종류와 수량을 샘플링하고 → 투입 순서를 섞고 → 모든 박스를 처리하면 종료합니다.
- **관측**: §3의 신경망 입력 / **행동**: 마스크된 후보 인덱스 (+ REJECT)
- **보상**
  ```
  r_t = v_b / V_pallet                       # 배치한 박스 부피 비율 (주 보상)
      + c1·f1 + c2·f2 − c3·f6               # 약한 보조 보상 (지지·접촉·시간)
      − c_rej · 1[REJECT 또는 배치 불가]
  종료 시: − c_cog · COG 편차
  ```
  하드 제약은 마스크로 강제하므로 "불가능한 행동 페널티"는 필요 없습니다.

### 6.2 1단계: PPO 사전학습
- **MaskablePPO** (sb3-contrib) 또는 직접 구현한 PPO
- 안정적인 on-policy 학습. 시뮬레이터가 싸서 샘플 효율보다 안정성이 중요합니다.
- 커리큘럼: 종류 3개 → 5개 → 10개 이상, 작은 팔레트 → 실제 규격
- 데이터 증강: 높이맵 좌우·상하 반전 (로봇 위치 특징도 함께 반전)
- 산출물: π_θ⁰, V_θ⁰

### 6.3 2단계: Expert Iteration (반복 개선)
```
for i in 0..N:
    Expert  = M6 롤아웃 플래너 (롤아웃 정책·prior = π_θⁱ, 잎 평가 = V_θⁱ)
    데이터   = Expert로 수천 에피소드 적재 → (상태, 후보, Expert 순위, 롤아웃 가치)
    π_θⁱ⁺¹  ← 교차엔트로피(Expert 선택) [+ 순위 손실]
    V_θⁱ⁺¹  ← MSE(롤아웃 가치 또는 최종 적재율)
    평가     = 고정 테스트셋에서 Expert와 π_θ 단독 성능 기록
```
- 롤아웃 플래너는 기반 정책보다 성능이 좋으므로(정책 개선), 반복할수록 학생과 교사가 함께 좋아집니다(AlphaZero와 같은 원리).
- 회차별 성능 곡선이 **강화학습 기여의 근거**가 됩니다.

### 6.4 배포
- 학습된 `π_θ`, `V_θ`를 TorchScript나 ONNX로 내보내고 `placement_planner` 노드가 불러옵니다.
- 모델이 없거나 추론에 실패하면 M5의 대체 경로(Q_now 기반)로 자동 전환합니다.

---

## 7. 실시간 예산 (박스당 목표)

| 단계 | 목표 시간 | 비고 |
|---|---|---|
| M1 관측·분류 | 비전 시스템 의존 | 컨베이어 이동과 병렬 |
| M2+M3 | ≤ 20 ms | NumPy 벡터화 |
| M4+M5 | ≤ 20 ms | 신경망 추론 1회 (배치 처리) |
| M6+M7 | **예산 T = 300~800 ms** | anytime, 병렬 롤아웃 |
| M8 | ≤ 200 ms | 상위 후보부터 차례로 IK·충돌 검사 |
| **합계** | **≤ 1 s** | 로봇 1 사이클(수 초)보다 짧아 이전 박스 동작 중에 다음 결정 가능 |

가속 수단: 높이맵 연산 NumPy·Numba, 롤아웃은 `multiprocessing`으로 시나리오 병렬, V_θ로 롤아웃 깊이 단축.

---

## 8. 주요 파라미터 (초기값)

| 파라미터 | 의미 | 초기값 |
|---|---|---|
| g | 높이맵 격자 | 10 mm |
| θ_s | 최소 지지면적 비율 | 0.75 |
| δ_cog | COG 지지 영역 안쪽 여유 | 10 mm |
| r_max(ρ) | 팔레트 COG 허용 반경 | 0.25·min(L,W) → 0.10·min(L,W) |
| c_app | 그리퍼 접근 여유 높이 | 50 mm |
| K0 | 롤아웃 대상 후보 수 | 16 |
| K | 시나리오 수 (총합) | 32~64 |
| D | 롤아웃 깊이 | 15 (이후 V_θ) |
| α, λ, μ | CVaR 비율, 위험 비중, 막힘 페널티 | 0.2, 0.4, 0.3 |
| β0, β_min | 미래 가치 비중 | 1.0, 0.2 |
| η | 정책망 prior 비중 | 0.5 |
| T | 결정 시간 예산 | 0.5 s |
| w1~w8 | 즉시 점수 가중치 | 벤치마크로 튜닝 |

모든 값은 `config/*.yaml`로 관리합니다.

---

## 9. 검증 계획

### 9.1 시나리오
- **박스셋**: 종류 수 {3, 5, 10, 20} × 크기 편차 {작음, 큼} × 무게 편차. 가능하면 실제 물류 데이터(BED-BPP, RoboBPP 데이터셋)를 포함합니다.
- **투입 순서**: 무작위 / 같은 종류가 몰려서 들어옴 / **불리한 순서**(작은 것 먼저, 무거운 것 나중) / 학습된 순서 공격자(AR2L 방식)
- **미리보기**: k = 0, 1, 3
- **예외 주입**: 변형 박스, 치수 오인식, 위치 오차, 집기 실패를 일정 확률로 발생

### 9.2 비교 대상 (같은 박스셋, 같은 시간 예산)
1. 휴리스틱: DBL, 높이맵 최소화(HM), Best-Fit
2. `Q_now` 탐욕 (IRAP의 M4만 사용)
3. PPO 정책 단독
4. 롤아웃 플래너 (π 없이)
5. **IRAP 전체** (Expert Iteration N회차 π_θ, V_θ + 롤아웃)
6. (참고) PCT 공개 코드

### 9.3 지표 (RoboBPP 지표 체계 참고)
| 분류 | 지표 |
|---|---|
| 조밀도 | 적재율 평균, **하위 20% 평균(CVaR)**, 적재 개수, REJECT 수 |
| 안정성 | 최소·평균 지지율, 최소 하중 여유 `min(F − load)`, COG 편차, **PyBullet 흔들기 시험 후 박스 변위** |
| 실행성 | M8 통과율, 예상 사이클 타임 |
| 실시간성 | 박스당 결정 시간 (평균·최대) |
| 강건성 | 무작위 순서 대비 불리한 순서에서의 성능 하락 폭 |

### 9.4 Ablation (각 요소의 기여 측정)
- 재고 기반 시나리오 vs 무작위 박스 시나리오 (재고를 모른다고 가정)
- CVaR 포함 vs 평균만
- V_θ 잎 평가 유무, π_θ prior 유무
- Expert Iteration 회차별 성능 곡선
- 시간 예산 T별 성능 곡선 (anytime 특성)

---

## 10. ROS2 매핑 요약

| 노드 | 담당 모듈 | 인터페이스 |
|---|---|---|
| `box_observer` | M1 | pub `/box_queue` |
| `pallet_state` | 상태, M9 보정 | pub `/pallet_state`, srv `UpdatePlacement` |
| `placement_planner` | M2~M7 (`palletizing_core` 호출) | action `PlanPlacement` (feedback: 중간 최선 후보) |
| `robot_feasibility` | M8 (MoveIt2) | srv `CheckFeasibility` |
| `task_executor` | 실행 | action `ExecutePlacement` |
| `exception_manager` | M9, L0 연동 | srv `Replan`, 이벤트 구독 |
| `visualizer` | RViz 3D 시각화 | MarkerArray |

---

## 11. 개발 로드맵

| 단계 | 내용 | 산출물 |
|---|---|---|
| 1 | 데이터 모델, M2·M3·M4, 탐욕 적재, matplotlib 3D 시각화, pytest | 기준선 + 시각화 |
| 2 | 박스셋·순서 생성기, 휴리스틱 기준선, 벤치마크 스크립트 | 비교표 v1 |
| 3 | M6 시나리오 롤아웃 + Successive Halving | 강건성 향상 입증 |
| 4 | `PalletEnv` + MaskablePPO 사전학습 | π_θ⁰, V_θ⁰ |
| 5 | Expert Iteration 반복 | 회차별 성능 곡선 |
| 6 | ROS2 래핑 + RViz + MoveIt2 검증(M8) + 예외 처리(M9) | 시뮬 데모 |
| 7 | PyBullet 안정성 검증, ablation, 최종 리포트 | 제출 자료 |

---

## 12. 참고 연구

- Zhao et al., *Online 3D Bin Packing with Constrained Deep Reinforcement Learning*, AAAI 2021 — 실행 가능성 마스크, 미리보기
- Zhao et al., *Learning Efficient Online 3D Bin Packing on Packing Configuration Trees (PCT)*, ICLR 2022 — 후보 기반 DRL
- Zhao et al., *Deliberate Planning of 3D Bin Packing on Packing Configuration Trees*, IJRR 2025 — 계획 통합, 실제 로봇
- Pan et al., *Adjustable Robust Reinforcement Learning for Online 3D Bin Packing (AR2L)*, NeurIPS 2023 — 투입 순서 강건성
- *Physics-Aware Robotic Palletization with Online Masking Inference*, ICRA 2025 — 물성(밀도·강성) 반영 마스크
- *Efficient RL of Task Planners for Robotic Palletization through Iterative Action Masking Learning*, 2024
- *Effective Online 3D Bin Packing with Lookahead Parcels Using MCTS*, 2026 — 미리보기 + MCTS
- *OPAL: Operationally Guided Placement-Aware Learning for Industrial Online 3D BPP*, 2026 — 운영 기준 후보 생성 + PPO 순위
- *RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation*, 2025 — 평가 체계
