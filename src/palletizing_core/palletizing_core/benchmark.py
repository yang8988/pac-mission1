"""Compare strategies over arrival orders and seeds (same box sets for every strategy).

    python -m palletizing_core.benchmark --seeds 10 --types 6
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .config import Config
from .generator import ORDERS, counts_for_fill, make_sequence, random_box_types
from .planner import STRATEGIES, GreedyPlanner
from .simulate import run_episode, validate

COLUMNS = ("utilization", "placed_ratio", "min_support", "min_load_margin", "cog_offset", "decision_ms_mean", "robot_time_s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--fill", type=float, default=0.8)
    ap.add_argument("--orders", nargs="+", default=list(ORDERS), choices=ORDERS)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--csv", default="out/benchmark.csv")
    args = ap.parse_args(argv)

    cfg = Config()
    rows = []
    violations = 0
    for order in args.orders:
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            types = random_box_types(rng, args.types)
            counts = counts_for_fill(rng, types, cfg, args.fill)
            seq = make_sequence(rng, types, counts, order, cfg)
            for strat in args.strategies:
                res = run_episode(cfg, types, counts, seq, GreedyPlanner(strat))
                violations += len(validate(res.state))
                rows.append({"order": order, "seed": seed, "strategy": strat, **res.metrics()})

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
