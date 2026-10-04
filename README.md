# PAC 2026 · HD현대로보틱스 과제 1 (팀 LYL)

불확정 투입 순서에 대응하는 Mixed Palletizing 적재 패턴 생성.

- 알고리즘 설계: [docs/ALGORITHM.md](docs/ALGORITHM.md)
- 구현 스택: Python, ROS2 (코어는 ROS 비의존 라이브러리)

## 구조

```
src/
└── palletizing_core/          # ROS 비의존 플래너 코어 (ament_python 패키지로도 빌드 가능)
    ├── palletizing_core/
    │   ├── config.py          # 팔레트·로봇·제약·가중치 설정 (YAML 로드)
    │   ├── model.py           # BoxType, Box, PalletState (높이맵, 지지 그래프, Extreme Point)
    │   ├── candidates.py      # M2 후보 생성 (EP × 자세 × 앵커)
    │   ├── constraints.py     # M3 하드 제약 H1~H8 (마스크)
    │   ├── scoring.py         # M4 즉시 점수 Q_now (8개 특징)
    │   ├── planner.py         # 탐욕 플래너: irap / dbl / hm
    │   ├── lookahead.py       # M6·M7 재고 기반 시나리오 롤아웃 + Successive Halving (irap_la)
    │   ├── rl/                # 4·5단계: PalletEnv, Actor-Critic, BC+PPO, Expert Iteration, RLPlanner (irap_rl)
    │   ├── generator.py       # 박스셋·투입 순서 생성기
    │   ├── simulate.py        # 에피소드 실행, 지표, 독립 검증기
    │   ├── viz.py             # 3D 적재 + 높이맵 시각화
    │   ├── demo.py            # 단일 에피소드 실행
    │   └── benchmark.py       # 전략 × 순서 × 시드 비교
    ├── config/default.yaml
    ├── models/                # 학습된 정책 (ppo.pt, ppo_bc.pt, ei/ei_3.pt) + 학습 로그
    └── test/
```

## 실행

```bash
pip install numpy matplotlib pyyaml pytest
pip install torch          # 4단계(RL)에만 필요
cd src/palletizing_core

python3 -m pytest -q test                                    # 테스트
python3 -m palletizing_core.demo --order random --seed 0     # out/demo/pallet.png, result.json
python3 -m palletizing_core.demo --strategy irap_la          # 3단계 미래 평가 플래너
python3 -m palletizing_core.benchmark --seeds 10 --jobs 4    # 전략 비교표 + out/benchmark.csv
python3 -m palletizing_core.rl.train --out models/ppo.pt     # 4단계 학습 (BC + PPO, 약 25분)
python3 -m palletizing_core.benchmark --strategies irap irap_rl --model models/ppo.pt --seeds 10 --jobs 4
python3 -m palletizing_core.rl.expert --init models/ppo.pt --iters 3 --episodes 64   # 5단계 (약 20분)
python3 -m palletizing_core.benchmark --strategies irap_ei --model models/ei/ei_3.pt --seeds 10 --jobs 4
```

주요 옵션: `--types`(박스 종류 수), `--fill`(전체 박스 부피 / 팔레트 부피), `--order {random,small_first,large_first,clustered}`, `--strategy {irap,dbl,hm,irap_la,irap_rl,irap_ei}`, `--model`(irap_rl·irap_ei용 체크포인트), `--value-leaf`(irap_ei에서 V_θ 잎 평가 사용), `--max-rollouts`(결정당 롤아웃 수, 기본 32) 또는 `--time-budget`(결정당 초), `--lookahead-k`(상류에서 미리 본 박스 수), `--config config/default.yaml`.

## 현재 성능 (박스 종류 6개, 총부피 = 팔레트의 80%, 4개 순서 × 시드 10개)

| 투입 순서 | DBL | IRAP 탐욕 (1단계) | **+ 미래 평가 (3단계)** | RL 정책 (4단계) | EI 학생 (5단계) | EI 교사 + V_θ (5단계) |
|---|---|---|---|---|---|---|
| random | 58.3% | 66.2% | **69.5%** | 63.9% | 65.8% | 67.9% |
| small_first (불리한 순서) | 49.9% | 49.4% | **54.6%** | 51.4% | 48.5% | 52.7% |
| large_first | 67.2% | 69.6% | **69.9%** | 68.3% | 69.5% | 67.9% |
| clustered | 55.9% | 57.6% | **63.5%** | 59.1% | 57.4% | 60.9% |
| **평균** | 57.8% | 60.7% | **64.4%** | 60.7% | 60.3% | 62.4% |

- 수치는 팔레트 부피 대비 적재율(상한 80%)이며, 모든 실행에서 제약 위반 0건입니다.
- **현재 최선은 3단계(`irap_la`)** 입니다. 결정당 롤아웃 32회, 결정 시간은 약 0.37초입니다. RL 정책은 결정당 약 3.4ms입니다.
- 4·5단계의 학습 요소는 아직 3단계를 넘지 못했습니다. 원인과 다음 시도는 [설계 문서 §6.3 구현 메모](docs/ALGORITHM.md)에 정리했습니다.
- 재현: `python3 -m palletizing_core.benchmark --seeds 10 --strategies irap dbl irap_la --jobs 4`

| 4단계 학습 곡선 | 5단계 Expert Iteration |
|---|---|
| ![4단계](docs/images/stage4_training.png) | ![5단계](docs/images/stage5_expert_iteration.png) |

| 1단계 IRAP 탐욕 (69.6%) | 3단계 미래 평가 (74.2%) |
|---|---|
| ![1단계](docs/images/stage1_demo.png) | ![3단계](docs/images/stage3_demo.png) |

## 진행 현황

- [x] 1단계: 데이터 모델, 후보 생성, 하드 제약 마스크, 즉시 점수, 탐욕 적재, 시각화, 테스트
- [ ] 2단계: 벤치마크 확장 (실제 물류 데이터셋, 지표 보강)
- [x] 3단계: 재고 기반 시나리오 롤아웃 + Successive Halving
- [x] 4단계: Gym 환경 + BC 사전학습 + 마스크 PPO (평균 성능은 탐욕과 동일, 불리한 순서에서 우세)
- [x] 5단계: Expert Iteration 구현 (3회차; 최종 플래너 개선 효과는 아직 없음, 원인 분석 완료)
- [ ] 6~7단계: ROS2 래핑, MoveIt2 검증, 예외 처리, PyBullet 안정성 검증
