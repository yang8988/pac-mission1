"""M4: immediate score Q_now from eight normalised features."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .constraints import Candidate
from .model import PalletState

FEATURES = ("support", "contact", "height_gain", "roughness", "cog", "time", "low_z", "far_first")
# +1 rewards the feature, -1 penalises it.
SIGNS = {
    "support": 1.0,
    "contact": 1.0,
    "height_gain": -1.0,
    "roughness": -1.0,
    "cog": -1.0,
    "time": -1.0,
    "low_z": -1.0,
    "far_first": 1.0,
}


class FeatureContext:
    """Per-decision precomputation shared by all candidates of one state."""

    def __init__(self, state: PalletState) -> None:
        pal, rob = state.cfg.pallet, state.cfg.robot
        H = state.H
        self.h_max = float(H.max())
        # 2D prefix sums of |neighbour height steps| along x and y, for O(1) window sums.
        self.sx = _prefix(np.abs(np.diff(H, axis=0)))
        self.sy = _prefix(np.abs(np.diff(H, axis=1)))
        bx, by, _ = rob.base
        self.far_max = max(math.hypot(px - bx, py - by) for px in (0, pal.length) for py in (0, pal.width))
        self.t_ref = rob.fixed_time + math.hypot(pal.length, pal.width) * 2 / rob.lin_speed + rob.rot_time


def robot_time(state: PalletState, cand: Candidate) -> float:
    """Rough pick-to-place cycle time estimate in seconds."""
    rob, g = state.cfg.robot, state.cfg.pallet.grid
    cx, cy, _ = cand.center(g)
    px, py, pz = rob.pick
    dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2 + (cand.z + cand.h + rob.approach_clearance - pz) ** 2)
    rot = rob.rot_time if cand.o != 0 else 0.0
    return rob.fixed_time + dist / rob.lin_speed + rot


def features(state: PalletState, cand: Candidate, ctx: FeatureContext | None = None) -> Dict[str, float]:
    ctx = ctx or FeatureContext(state)
    pal, rob = state.cfg.pallet, state.cfg.robot
    g, H = pal.grid, state.H
    i, j, ni, nj = cand.i, cand.j, cand.ni, cand.nj
    z, top = cand.z, cand.z + cand.h
    nx, ny = H.shape

    # Neighbour strips around the footprint (None at a pallet wall).
    strips = (
        H[i - 1, j : j + nj] if i > 0 else None,
        H[i + ni, j : j + nj] if i + ni < nx else None,
        H[i : i + ni, j - 1] if j > 0 else None,
        H[i : i + ni, j + nj] if j + nj < ny else None,
    )
    lengths = (nj, nj, ni, ni)

    # f2: share of the perimeter touching a wall or a neighbour, weighted by touching height.
    touch = 0.0
    new_steps = 0.0
    for strip, n in zip(strips, lengths):
        if strip is None:
            touch += n
        else:
            touch += float(np.maximum(np.minimum(strip, top) - z, 0.0).sum()) / cand.h
            new_steps += float(np.abs(strip - top).sum())
    contact = touch / (2 * (ni + nj))

    # f4: change of roughness (sum of neighbour height steps). Only steps touching the footprint
    # change: inside they become 0, on the boundary they become |top - neighbour|.
    old_steps = _rect_sum(ctx.sx, i - 1, i + ni, j, j + nj) + _rect_sum(ctx.sy, i, i + ni, j - 1, j + nj)
    rough = (new_steps - old_steps) / (2 * (ni + nj) * pal.max_height)

    cx, cy, _ = cand.center(g)
    bx, by, _ = rob.base
    t = robot_time(state, cand)

    return {
        "support": cand.support_ratio,
        "contact": contact,
        "height_gain": max(0.0, top - ctx.h_max) / pal.max_height,
        "roughness": min(1.0, max(-1.0, rough)),
        "cog": math.hypot(cand.cog_after[0] - pal.length / 2, cand.cog_after[1] - pal.width / 2)
        / (0.5 * min(pal.length, pal.width)),
        "time": t / ctx.t_ref,
        "low_z": z / pal.max_height,
        "far_first": math.hypot(cx - bx, cy - by) / ctx.far_max,
        "t_robot": t,  # raw seconds, informational (not weighted)
    }


def q_now(state: PalletState, cand: Candidate, ctx: FeatureContext | None = None) -> float:
    f = features(state, cand, ctx)
    cand.features = f
    w = state.cfg.weights
    cand.score = sum(SIGNS[k] * getattr(w, k) * f[k] for k in FEATURES)
    return cand.score


def _prefix(a: np.ndarray) -> np.ndarray:
    p = np.zeros((a.shape[0] + 1, a.shape[1] + 1))
    p[1:, 1:] = a.cumsum(0).cumsum(1)
    return p


def _rect_sum(p: np.ndarray, a0: int, a1: int, b0: int, b1: int) -> float:
    """Sum of the underlying array over rows [a0, a1) and cols [b0, b1), clipped to its extent."""
    n, m = p.shape[0] - 1, p.shape[1] - 1
    a0, a1, b0, b1 = max(0, a0), min(n, a1), max(0, b0), min(m, b1)
    if a1 <= a0 or b1 <= b0:
        return 0.0
    return float(p[a1, b1] - p[a0, b1] - p[a1, b0] + p[a0, b0])
