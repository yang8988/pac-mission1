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
from typing import Dict, List, Optional, Sequence

from . import candidates as cgen
from .constraints import Candidate, evaluate_many
from .model import Box, PalletState
from .scoring import FeatureContext, features, q_now

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
    info: Dict[str, object] = field(default_factory=dict)


class GreedyPlanner:
    def __init__(self, strategy: str = "irap", top_n: int = 10, limit: Optional[int] = None) -> None:
        """`limit` caps the fully checked candidates (see constraints.evaluate_many); for rollouts."""
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy '{strategy}', expected one of {STRATEGIES}")
        self.strategy = strategy
        self.top_n = top_n
        self.limit = limit

    def feasible_candidates(self, state: PalletState, box: Box) -> tuple[List[Candidate], int, Counter]:
        actions = cgen.generate(state, box)
        feasible, stats = evaluate_many(state, box, actions, self.limit)
        return feasible, len(actions), stats

    def decide(self, state: PalletState, box: Box, queue: Sequence[str] = ()) -> Decision:
        """Choose a placement for `box` without mutating `state`.

        `queue` holds the type ids of boxes already observed upstream; the greedy planner ignores it.
        """
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
        ctx = FeatureContext(state)
        if self.strategy == "irap":
            for c in cands:
                q_now(state, c, ctx)
            cands.sort(key=lambda c: (-c.score, c.z, c.i, c.j, c.o))
            return
        for c in cands:
            c.features = features(state, c, ctx)
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
