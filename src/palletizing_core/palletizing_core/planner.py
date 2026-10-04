"""Greedy single-step planner (stage-1 baseline) and classic heuristic baselines.

Strategies:
  - "irap": Q_now weighted score (M4), the baseline IRAP builds on
  - "dbl":  deepest-bottom-left (min z, then x, then y)
  - "hm":   heightmap minimisation (min sum of the heightmap after placement)
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import candidates as cgen
from .constraints import Candidate, evaluate
from .model import Box, PalletState
from .scoring import features, q_now

PLACE = "place"
REJECT = "reject"

STRATEGIES = ("irap", "dbl", "hm")


@dataclass
class Decision:
    box: Box
    action: str
    best: Optional[Candidate] = None
    ranked: List[Candidate] = field(default_factory=list)
    reason: str = ""
    n_generated: int = 0
    n_feasible: int = 0
    mask_stats: Dict[str, int] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class GreedyPlanner:
    def __init__(self, strategy: str = "irap", top_n: int = 10) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")
        self.strategy = strategy
        self.top_n = top_n

    def feasible_candidates(self, state: PalletState, box: Box) -> tuple[List[Candidate], int, Counter]:
        actions = cgen.generate(state, box)
        stats: Counter = Counter()
        feasible: List[Candidate] = []
        for i, j, o in actions:
            cand, reason = evaluate(state, box, i, j, o)
            if cand is None:
                stats[reason] += 1
            else:
                feasible.append(cand)
        return feasible, len(actions), stats

    def decide(self, state: PalletState, box: Box) -> Decision:
        """Choose a placement for `box` without mutating `state`."""
        t0 = time.perf_counter()
        rob = state.cfg.robot
        if box.damaged:
            return Decision(box, REJECT, reason="damaged", elapsed_ms=_ms(t0))
        if box.mass > rob.payload:
            return Decision(box, REJECT, reason="payload", elapsed_ms=_ms(t0))

        feasible, n_gen, stats = self.feasible_candidates(state, box)
        if not feasible:
            return Decision(
                box, REJECT, reason="no_feasible", n_generated=n_gen, mask_stats=dict(stats), elapsed_ms=_ms(t0)
            )

        self._rank(state, feasible)
        return Decision(
            box,
            PLACE,
            best=feasible[0],
            ranked=feasible[: self.top_n],
            n_generated=n_gen,
            n_feasible=len(feasible),
            mask_stats=dict(stats),
            elapsed_ms=_ms(t0),
        )

    def _rank(self, state: PalletState, cands: List[Candidate]) -> None:
        g = state.cfg.pallet.grid
        if self.strategy == "irap":
            for c in cands:
                q_now(state, c)
            cands.sort(key=lambda c: (-c.score, c.z, c.i, c.j, c.o))
            return
        for c in cands:
            c.features = features(state, c)
        if self.strategy == "dbl":
            for c in cands:
                c.score = -c.z
            cands.sort(key=lambda c: (c.z, c.i * g, c.j * g, c.o))
        else:  # "hm": sum of heightmap increase over the footprint, ties -> lower, then DBL
            for c in cands:
                region = state.H[c.i : c.i + c.ni, c.j : c.j + c.nj]
                c.score = -float(((c.z + c.h) - region).sum())
            cands.sort(key=lambda c: (-c.score, c.z, c.i, c.j, c.o))


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0
