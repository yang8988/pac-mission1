"""M6/M7: robust future evaluation with inventory-based scenarios.

For the current box, the best K0 candidates by Q_now are re-ranked by simulating the near
future. Scenarios keep the observed upstream queue as-is and fill the rest with random
permutations of the remaining inventory (a share of them in an unfavourable small-first
order). Every candidate is rolled out on the *same* scenarios (common random numbers), and
Successive Halving spends the time budget on the promising candidates. The final score
combines Q_now with the mean, the CVaR tail and the blocking probability of the rollouts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import LookaheadParams
from .constraints import Candidate
from .model import Box, PalletState
from .planner import PLACE, Decision, GreedyPlanner

REWARD_SCALE = 10.0  # rl.env: episode return = 10 x utilisation

# Lower bounds on the spread used to standardise scores, so that differences far below
# these scales (noise) are not blown up into a full standard deviation.
Q_STD_FLOOR = 0.05
V_STD_FLOOR = 0.02


@dataclass
class RolloutStats:
    values: List[float] = field(default_factory=list)
    blocked: List[bool] = field(default_factory=list)

    def summary(self, m: int, alpha: float) -> Tuple[float, float, float]:
        """(mean, CVaR_alpha, blocking probability) over the first m scenarios."""
        v = np.asarray(self.values[:m])
        tail = np.sort(v)[: max(1, math.ceil(alpha * m))]
        return float(v.mean()), float(tail.mean()), float(np.mean(self.blocked[:m]))


def sample_scenarios(
    state: PalletState, queue: Sequence[str], params: LookaheadParams, rng: np.random.Generator, extra: int = 0
) -> List[List[str]]:
    """Future arrival sequences (type ids), truncated to the rollout depth (+ `extra` boxes).

    The current box is assumed to be consumed from `state.inventory` already; boxes in `queue`
    have been observed but not consumed, so they are removed from the random part.
    """
    remaining = dict(state.inventory)
    for tid in queue:
        if remaining.get(tid, 0) > 0:
            remaining[tid] -= 1
    pool = np.array([tid for tid, n in sorted(remaining.items()) for _ in range(n)], dtype=object)
    vol = np.array([state.types[t].volume for t in pool], dtype=float) if pool.size else np.zeros(0)
    every = round(1.0 / params.adversarial_frac) if params.adversarial_frac > 0 else 0
    scenarios = []
    for s in range(params.max_scenarios):
        perm = rng.permutation(pool.size)
        if every and s % every == 1:  # unfavourable: small boxes first (noisy), big ones late
            perm = perm[np.argsort(vol[perm] * rng.uniform(0.8, 1.2, perm.size), kind="stable")]
        seq = list(queue) + [str(t) for t in pool[perm]]
        scenarios.append(seq[: params.depth + extra])
    return scenarios


class LookaheadPlanner:
    """Greedy candidate generation + rollout-based robust re-ranking (IRAP stage 3).

    With a trained `model` (stage 5) two things change:
      - prior: the K0 candidates to roll out are the best by Q_now + prior_weight * log pi(a|s);
      - leaf value: a rollout ends with V_theta of the next scenario box, so its value is the
        estimated final utilisation  util(after rollout) + V / 10  instead of the proxy.
    """

    def __init__(self, params: Optional[LookaheadParams] = None, top_n: int = 10, model=None) -> None:
        self.params = params
        self.top_n = top_n
        self.base = GreedyPlanner("irap", top_n=10**9)
        self._policies: Dict[Optional[int], GreedyPlanner] = {}
        self.model = None
        if model is not None:
            from .rl.train import load_model  # needs torch

            self.model = load_model(model) if isinstance(model, (str, Path)) else model
            self.model.eval()

    def _policy(self, limit: Optional[int]) -> GreedyPlanner:
        if limit not in self._policies:
            self._policies[limit] = GreedyPlanner("irap", top_n=1, limit=limit)
        return self._policies[limit]

    def decide(self, state: PalletState, box: Box, queue: Sequence[str] = ()) -> Decision:
        t0 = time.perf_counter()
        P = self.params or state.cfg.lookahead
        dec = self.base.decide(state, box)
        if dec.action != PLACE or dec.n_feasible <= 1 or P.depth <= 0 or P.k0 <= 1:
            dec.ranked = dec.ranked[: self.top_n]
            return dec

        info: Dict[str, object] = {}
        if self.model is not None:
            ranked, value_now = self._apply_prior(state, box, dec.ranked, P)
            dec.ranked = ranked
            info["value"] = value_now

        rng = np.random.default_rng([P.seed, state.processed])
        use_leaf = self.model is not None and P.value_leaf
        scenarios = sample_scenarios(state, queue, P, rng, extra=1 if use_leaf else 0)
        if not scenarios or not scenarios[0]:
            dec.ranked = dec.ranked[: self.top_n]
            dec.info.update(info)
            return dec

        cands = dec.ranked[: P.k0]
        rest = dec.ranked[P.k0 :]
        stats = [RolloutStats() for _ in cands]
        deadline = None if P.time_budget is None else t0 + P.time_budget
        n_rollouts = 0

        def out_of_budget() -> bool:
            if P.max_rollouts is not None and n_rollouts >= P.max_rollouts:
                return True
            return deadline is not None and time.perf_counter() >= deadline

        alive = list(range(len(cands)))
        eliminated: List[List[int]] = []  # per round, best first
        target = min(P.first_round, len(scenarios))
        rounds = 0
        stopped = False
        while True:
            rounds += 1
            for c in alive:
                while len(stats[c].values) < target:
                    if out_of_budget():
                        stopped = True
                        break
                    value, blocked = self._rollout(state, cands[c], scenarios[len(stats[c].values)], P)
                    stats[c].values.append(value)
                    stats[c].blocked.append(blocked)
                    n_rollouts += 1
                if stopped:
                    break
            if stopped or len(alive) == 1 or target >= len(scenarios):
                break
            order = self._rank(cands, stats, alive, state, P)
            keep = math.ceil(len(order) / 2)
            alive, dropped = order[:keep], order[keep:]
            eliminated.append(dropped)
            target = min(target * 2, len(scenarios))

        ranked_idx = self._rank(cands, stats, alive, state, P)
        for dropped in reversed(eliminated):
            ranked_idx += dropped
        ranked = [cands[k] for k in ranked_idx] + rest

        dec.best = ranked[0]
        dec.ranked = ranked[: self.top_n]
        dec.info = {
            **info,
            "rollouts": n_rollouts,
            "rounds": rounds,
            "scenarios": len(scenarios),
            "budget_hit": stopped,
            "changed_choice": ranked_idx[0] != 0,
        }
        dec.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return dec

    # ------------------------------------------------------------------ internals
    def _apply_prior(self, state: PalletState, box: Box, ranked: List[Candidate], P: LookaheadParams):
        """Re-order the Q_now-ranked candidates by Q_now + prior_weight * log pi. Returns (ranked, V(s))."""
        import torch

        from .rl.obs import N_CANDS, encode
        from .rl.policy import to_tensors

        head, tail = ranked[:N_CANDS], ranked[N_CANDS:]
        obs, slots = encode(state, box, head)
        with torch.no_grad():
            logits, value = self.model(to_tensors([obs]))
            logp = torch.log_softmax(logits[0], -1).numpy()
        for k, c in enumerate(slots):
            if c is not None:
                c.features["log_pi"] = float(logp[k])
        head = sorted(head, key=lambda c: -(c.score + P.prior_weight * c.features["log_pi"]))
        return head + tail, float(value[0])

    def _leaf_value(self, s: PalletState, tid: str) -> Optional[float]:
        """V_theta for the state where box type `tid` arrives next (None if it cannot be placed)."""
        import torch

        from .rl.obs import encode
        from .rl.policy import to_tensors

        cp = s.cfg.constraints
        b = Box.from_type("leaf", s.types[tid], cp.default_load_factor, None, cp.area_load_capacity)
        s.consume(tid)
        d = self._leaf_ranker().decide(s, b)
        if d.action != PLACE:
            return None
        obs, _ = encode(s, b, d.ranked)
        with torch.no_grad():
            _, value = self.model(to_tensors([obs]))
        return float(value[0])

    def _leaf_ranker(self) -> GreedyPlanner:
        if not hasattr(self, "_leaf_planner"):
            from .rl.obs import N_CANDS

            self._leaf_planner = GreedyPlanner("irap", top_n=N_CANDS)
        return self._leaf_planner

    def _rank(
        self, cands: List[Candidate], stats: List[RolloutStats], idx: List[int], state: PalletState, P: LookaheadParams
    ) -> List[int]:
        """Order `idx` by Score = z(Q_now) + beta(rho) * z(V_future), on the common scenarios."""
        m = min(len(stats[k].values) for k in idx)
        q = np.array([cands[k].score for k in idx])
        zq = (q - q.mean()) / max(q.std(), Q_STD_FLOOR)
        if m == 0:
            score = zq
        else:
            vf = []
            for k in idx:
                mean, cvar, pblock = stats[k].summary(m, P.alpha)
                v = (1 - P.lam) * mean + P.lam * cvar - P.mu * pblock
                cands[k].features.update({"v_mean": mean, "v_cvar": cvar, "p_block": pblock, "v_future": v})
                vf.append(v)
            vf = np.array(vf)
            zv = (vf - vf.mean()) / max(vf.std(), V_STD_FLOOR)
            rho = state.progress
            beta = P.beta0 * (1 - rho) + P.beta_min * rho
            score = zq + beta * zv
        for k, sc in zip(idx, score):
            cands[k].features["robust_score"] = float(sc)
        order = sorted(range(len(idx)), key=lambda a: (-score[a], idx[a]))
        return [idx[a] for a in order]

    def _rollout(self, state: PalletState, cand: Candidate, seq: Sequence[str], P: LookaheadParams) -> Tuple[float, bool]:
        """Place `cand`, then greedily place the scenario boxes. Returns (value, blocked)."""
        cp, pal = state.cfg.constraints, state.cfg.pallet
        s = state.copy()
        s.commit(cand)
        vol_total = vol_placed = cand.box.volume
        blocked = False
        policy = self._policy(P.rollout_limit)
        use_leaf = self.model is not None and P.value_leaf
        leaf_tid = seq[P.depth] if use_leaf and len(seq) > P.depth else None
        for tid in seq[: P.depth]:
            t = s.types[tid]
            b = Box.from_type("sim", t, cp.default_load_factor, None, cp.area_load_capacity)
            s.consume(tid)
            d = policy.decide(s, b)
            vol_total += b.volume
            if d.action == PLACE:
                s.commit(d.best)
                vol_placed += b.volume
            else:
                s.reject(b)
                blocked = True
        if use_leaf:
            # Estimated final utilisation: what is on the pallet now + V_theta of the rest.
            util = s.utilization()
            if leaf_tid is not None:
                v = self._leaf_value(s, leaf_tid)
                if v is not None:
                    util += max(0.0, v) / REWARD_SCALE
            return util, blocked
        fill = vol_placed / vol_total
        top = float(s.H.max())
        compact = sum(pb.box.volume for pb in s.placed) / (pal.length * pal.width * top) if top > 0 else 0.0
        return (1 - P.compactness) * fill + P.compactness * compact, blocked
