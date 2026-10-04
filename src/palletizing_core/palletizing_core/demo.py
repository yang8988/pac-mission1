"""Run one episode and save metrics + a 3D image.

    python -m palletizing_core.demo --types 6 --order random --strategy irap --seed 0 --out out/demo
    python -m palletizing_core.demo --strategy irap_la --max-rollouts 32 --lookahead-k 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .generator import ORDERS, counts_for_fill, make_sequence, random_box_types
from .benchmark import ALL_STRATEGIES, build_config, make_planner
from .simulate import run_episode, validate
from .viz import plot_pallet


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config file")
    ap.add_argument("--types", type=int, default=6)
    ap.add_argument("--fill", type=float, default=0.8, help="total box volume / pallet volume")
    ap.add_argument("--order", choices=ORDERS, default="random")
    ap.add_argument("--strategy", choices=ALL_STRATEGIES, default="irap")
    ap.add_argument("--max-rollouts", type=int, default=32, help="rollouts per decision for irap_la")
    ap.add_argument("--time-budget", type=float, help="seconds per decision for irap_la (overrides --max-rollouts)")
    ap.add_argument("--lookahead-k", type=int, default=0, help="observed upstream boxes passed to the planner")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/demo")
    args = ap.parse_args(argv)

    cfg = build_config(args.config, args.max_rollouts, args.time_budget)
    rng = np.random.default_rng(args.seed)
    types = random_box_types(rng, args.types)
    counts = counts_for_fill(rng, types, cfg, args.fill)
    seq = make_sequence(rng, types, counts, args.order, cfg)

    res = run_episode(cfg, types, counts, seq, make_planner(args.strategy), args.lookahead_k)
    m = res.metrics()
    errors = validate(res.state)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    title = f"{args.strategy} | {args.order} | util {m['utilization']:.1%} | placed {m['placed']}/{len(seq)}"
    plot_pallet(res.state, out / "pallet.png", title)
    placements = [
        {
            "box_id": pb.box.box_id,
            "type": pb.box.type_id,
            "x": pb.x, "y": pb.y, "z": pb.z,
            "w": pb.w, "d": pb.d, "h": pb.h,
            "orientation": pb.orientation,
            "mass": pb.box.mass,
            "load": round(pb.load, 1),
            "max_load": round(pb.box.max_load, 1),
        }  # fmt: skip
        for pb in res.state.placed
    ]
    (out / "result.json").write_text(
        json.dumps(
            {"args": vars(args), "metrics": m, "violations": errors, "placements": placements,
             "rejected": [b.box_id for b in res.state.rejected]},  # fmt: skip
            indent=2,
        )
    )
    for k, v in m.items():
        print(f"{k:>18}: {v:.3f}" if isinstance(v, float) else f"{k:>18}: {v}")
    print(f"{'violations':>18}: {len(errors)}")
    print(f"saved {out / 'pallet.png'} and {out / 'result.json'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
