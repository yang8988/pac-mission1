"""PalletEnv: one step = one placement decision for the current box (stage 4, section 6.1).

The action is a candidate slot; slots that did not pass the tier-1 hard constraints are
masked, so the agent can never choose an infeasible placement. Boxes without any feasible
candidate are rejected automatically (their volume is simply never rewarded).

Reward: 10 * placed box volume / pallet volume, so the episode return is 10 * utilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import Config
from ..generator import counts_for_fill, make_sequence, random_box_types
from ..model import Box, BoxType, PalletState
from ..planner import PLACE, GreedyPlanner
from .obs import N_CANDS, encode

REWARD_SCALE = 10.0


@dataclass
class EpisodeSpec:
    types: List[BoxType]
    counts: Dict[str, int]
    sequence: List[Box]
    order: str


@dataclass
class TaskDistribution:
    """Random episodes for training: box types, fill ratio and arrival order."""

    n_types: Tuple[int, int] = (3, 10)
    fill: Tuple[float, float] = (0.6, 0.9)
    orders: Tuple[str, ...] = ("random", "small_first", "large_first", "clustered")
    order_probs: Tuple[float, ...] = (0.5, 0.15, 0.1, 0.25)

    def sample(self, rng: np.random.Generator, cfg: Config) -> EpisodeSpec:
        n = int(rng.integers(self.n_types[0], self.n_types[1] + 1))
        types = random_box_types(rng, n)
        counts = counts_for_fill(rng, types, cfg, float(rng.uniform(*self.fill)))
        order = str(rng.choice(self.orders, p=self.order_probs))
        return EpisodeSpec(types, counts, make_sequence(rng, types, counts, order, cfg), order)


class PalletEnv:
    """Gymnasium-style API: reset() -> (obs, info), step(a) -> (obs, r, terminated, truncated, info)."""

    def __init__(
        self,
        cfg: Optional[Config] = None,
        tasks: Optional[TaskDistribution] = None,
        seed: int = 0,
        shuffle_slots: bool = True,
    ) -> None:
        self.cfg = cfg or Config()
        self.tasks = tasks or TaskDistribution()
        self.rng = np.random.default_rng(seed)
        self.shuffle_slots = shuffle_slots
        self.ranker = GreedyPlanner("irap", top_n=N_CANDS)
        self.state: Optional[PalletState] = None
        self.spec: Optional[EpisodeSpec] = None
        self.t = 0
        self.slots: List = []
        self.obs: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------ API
    def reset(self, spec: Optional[EpisodeSpec] = None) -> Tuple[Dict[str, np.ndarray], dict]:
        self.spec = spec or self.tasks.sample(self.rng, self.cfg)
        self.state = PalletState.new(self.cfg, self.spec.types, self.spec.counts)
        self.t = 0
        done = self._advance()
        assert not done or self.obs is None
        return self.obs, {"done": done}

    def action_masks(self) -> np.ndarray:
        return self.obs["mask"].copy() if self.obs is not None else np.zeros(N_CANDS, dtype=bool)

    def step(self, action: int):
        cand = self.slots[int(action)]
        if cand is None:
            raise ValueError(f"action {action} is masked")
        self.state.commit(cand)
        reward = REWARD_SCALE * cand.box.volume / self._pallet_volume()
        self.t += 1
        terminated = self._advance()
        info = {}
        if terminated:
            info["utilization"] = self.state.utilization()
            info["placed"] = len(self.state.placed)
            info["rejected"] = len(self.state.rejected)
        return self.obs, reward, terminated, False, info

    # ------------------------------------------------------------------ internals
    def _pallet_volume(self) -> float:
        p = self.cfg.pallet
        return p.length * p.width * p.max_height

    def _advance(self) -> bool:
        """Move to the next box that has a feasible placement. Returns True when the episode ends."""
        seq = self.spec.sequence
        while self.t < len(seq):
            box = seq[self.t]
            self.state.consume(box.type_id)
            dec = self.ranker.decide(self.state, box)
            if dec.action == PLACE:
                cands = dec.ranked  # ranked by Q_now, features filled
                self.obs, self.slots = encode(self.state, box, cands, self.rng if self.shuffle_slots else None)
                self.current_box = box
                return False
            self.state.reject(box)
            self.t += 1
        self.obs, self.slots = None, []
        return True


def greedy_slot(slots: Sequence) -> int:
    """Slot holding the best candidate by Q_now (the greedy teacher's choice)."""
    best, best_k = -np.inf, -1
    for k, c in enumerate(slots):
        if c is not None and c.score > best:
            best, best_k = c.score, k
    return best_k
