"""RLPlanner: the trained policy as a drop-in planner (strategy "irap_rl")."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence, Union

import torch

from ..model import Box, PalletState
from ..planner import PLACE, Decision, GreedyPlanner
from .obs import N_CANDS, encode
from .policy import ActorCritic, to_tensors


class RLPlanner:
    """Ranks the feasible candidates (best N_CANDS by Q_now) by the policy logits.

    Hard constraints are untouched: the policy only reorders candidates that passed the mask.
    `info["value"]` carries V_theta(s), the predicted remaining return (10 x future utilisation).
    """

    def __init__(self, model: Union[ActorCritic, str, Path], top_n: int = 10) -> None:
        if not isinstance(model, ActorCritic):
            from .train import load_model

            model = load_model(model)
        self.model = model.eval()
        self.top_n = top_n
        self.ranker = GreedyPlanner("irap", top_n=N_CANDS)

    def decide(self, state: PalletState, box: Box, queue: Sequence[str] = ()) -> Decision:
        t0 = time.perf_counter()
        dec = self.ranker.decide(state, box)
        if dec.action != PLACE:
            return dec
        obs, slots = encode(state, box, dec.ranked)
        with torch.no_grad():
            logits, value = self.model(to_tensors([obs]))
        order = torch.argsort(logits[0], descending=True).tolist()
        ranked = [slots[k] for k in order if slots[k] is not None]
        dec.best = ranked[0]
        dec.ranked = ranked[: self.top_n]
        dec.info = {"value": float(value[0])}
        dec.elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return dec
