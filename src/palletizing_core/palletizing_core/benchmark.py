"""Compare strategies over arrival orders and seeds (same box sets for every strategy).

    python -m palletizing_core.benchmark --seeds 10 --types 6 --jobs 4
    python -m palletizing_core.benchmark --strategies irap irap_la --max-rollouts 32 --lookahead-k 0

`irap_la` is the stage-3 lookahead planner. By default it uses a fixed rollout budget per
decision (`--max-rollouts`) so results are deterministic and machine independent; pass
`--time-budget` to use the wall-clock budget instead.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import Config, load_config
from .generator import ORDERS, counts_for_fill, make_sequence, random_box_types
from .lookahead import LookaheadPlanner
from .planner import STRATEGIES, GreedyPlanner
from .simulate import run_episode, validate

ALL_STRATEGIES = STRATEGIES + ("irap_la",)
COLUMNS = ("utilization", "placed_ratio", "min_support", "min_load_margin", "cog_offset", "decision_ms_mean", "robot_time_s")


def make_planner(strategy: str):
    return LookaheadPlanner() if strategy == "irap_la" else GreedyPlanner(strategy)


def run_one(task) -> Dict[str, float]:
    cfg, order, seed, strategy, n_types, fill, lookahead_k = task
    rng = np.random.default_rng(seed)
    types = random_box_types(rng, n_types)
    counts = counts_for_fill(rng, types, cfg, fill)
    seq = make_sequence(rng, types, counts, order, cfg)
    res = run_episode(cfg, types, counts, seq, make_planner(strategy), lookahead_k)
    row = {"order": order, "seed": seed, "strategy": strategy, **res.metrics()}
    row["violations"] = len(validate(res.state))
    return row


def build_config(path: Optional[str], max_rollouts: Optional[int], time_budget: Optional[float]) -> Config:
    cfg = load_config(path) if path else Config()
    la = cfg.lookahead
    if time_budget is not None:
        cfg.lookahead = dataclasses.replace(la, time_budget=time_budget, max_rollouts=None)
    else:
        cfg.lookahead = dataclasses.replace(la, time_budget=None, max_rollouts=max_rollouts)
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--fill", type=float, default=0.8)
    ap.add_argument("--orders", nargs="+", default=list(ORDERS), choices=ORDERS)
    ap.add_argument("--strategies", nargs="+", default=list(ALL_STRATEGIES), choices=ALL_STRATEGIES)
    ap.add_argument("--max-rollouts", type=int, default=32, help="rollouts per decision for irap_la")
    ap.add_argument("--time-budget", type=float, help="seconds per decision for irap_la (overrides --max-rollouts)")
    ap.add_argument("--lookahead-k", type=int, default=0, help="observed upstream boxes passed to the planner")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--csv", default="out/benchmark.csv")
    args = ap.parse_args(argv)

    cfg = build_config(args.config, args.max_rollouts, args.time_budget)
    tasks = [
        (cfg, order, seed, strat, args.types, args.fill, args.lookahead_k)
        for order in args.orders
        for seed in range(args.seeds)
        for strat in args.strategies
    ]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            rows = pool.map(run_one, tasks, chunksize=1)
    else:
        rows = [run_one(t) for t in tasks]
    violations = sum(r["violations"] for r in rows)

    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    head = f"{'order':<12}{'strategy':<9}" + "".join(f"{c:>17}" for c in COLUMNS)
    print(head)
    print("-" * len(head))
    for order in args.orders:
        for strat in args.strategies:
            sel = [r for r in rows if r["order"] == order and r["strategy"] == strat]
            vals = "".join(f"{np.mean([r[c] for r in sel]):>17.3f}" for c in COLUMNS)
            print(f"{order:<12}{strat:<9}{vals}")
    print(f"\nconstraint violations (validator): {violations}")
    print(f"per-episode rows saved to {args.csv}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
