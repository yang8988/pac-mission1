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


def robot_time(state: PalletState, cand: Candidate) -> float:
    """Rough pick-to-place cycle time estimate in seconds."""
    rob, g = state.cfg.robot, state.cfg.pallet.grid
    cx, cy, _ = cand.center(g)
    target = np.array([cx, cy, cand.z + cand.h + rob.approach_clearance])
    dist = float(np.linalg.norm(target - np.array(rob.pick)))
    rot = rob.rot_time if cand.o != 0 else 0.0
    return rob.fixed_time + dist / rob.lin_speed + rot


def features(state: PalletState, cand: Candidate) -> Dict[str, float]:
    pal, rob = state.cfg.pallet, state.cfg.robot
    g, H = pal.grid, state.H
    i, j, ni, nj = cand.i, cand.j, cand.ni, cand.nj
    z, top = cand.z, cand.z + cand.h
    nx, ny = H.shape

    # f2: share of the perimeter touching a wall or a neighbour, weighted by touching height.
    sides = []
    for strip in (
        H[i - 1, j : j + nj] if i > 0 else None,
        H[i + ni, j : j + nj] if i + ni < nx else None,
        H[i : i + ni, j - 1] if j > 0 else None,
        H[i : i + ni, j + nj] if j + nj < ny else None,
    ):
        sides.append(None if strip is None else np.clip((np.minimum(strip, top) - z) / cand.h, 0.0, 1.0))
    lengths = (nj, nj, ni, ni)
    contact = sum(n if s is None else float(s.sum()) for s, n in zip(sides, lengths)) / (2 * (ni + nj))

    # f4: change of local roughness (sum of neighbour height steps) around the footprint.
    a0, a1, b0, b1 = max(0, i - 1), min(nx, i + ni + 1), max(0, j - 1), min(ny, j + nj + 1)
    before = H[a0:a1, b0:b1]
    after = before.copy()
    after[i - a0 : i - a0 + ni, j - b0 : j - b0 + nj] = top
    rough = (_roughness(after) - _roughness(before)) / (2 * (ni + nj) * pal.max_height)

    cx, cy, _ = cand.center(g)
    bx, by, _ = rob.base
    far_max = max(math.hypot(px - bx, py - by) for px in (0, pal.length) for py in (0, pal.width))
    t = robot_time(state, cand)
    t_ref = rob.fixed_time + math.hypot(pal.length, pal.width) * 2 / rob.lin_speed + rob.rot_time

    return {
        "support": cand.support_ratio,
        "contact": float(contact),
        "height_gain": max(0.0, top - float(H.max())) / pal.max_height,
        "roughness": float(np.clip(rough, -1.0, 1.0)),
        "cog": math.hypot(cand.cog_after[0] - pal.length / 2, cand.cog_after[1] - pal.width / 2)
        / (0.5 * min(pal.length, pal.width)),
        "time": t / t_ref,
        "low_z": z / pal.max_height,
        "far_first": math.hypot(cx - bx, cy - by) / far_max,
        "t_robot": t,  # raw seconds, informational (not weighted)
    }


def q_now(state: PalletState, cand: Candidate) -> float:
    f = features(state, cand)
    cand.features = f
    w = state.cfg.weights
    cand.score = sum(SIGNS[k] * getattr(w, k) * f[k] for k in FEATURES)
    return cand.score


def _roughness(a: np.ndarray) -> float:
    return float(np.abs(np.diff(a, axis=0)).sum() + np.abs(np.diff(a, axis=1)).sum())
