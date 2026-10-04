"""Observation encoding shared by the RL environment and the RL planner (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..config import GRAVITY
from ..constraints import Candidate
from ..model import Box, PalletState
from ..scoring import FEATURES

N_CANDS = 48  # candidate slots (best feasible candidates by Q_now)
N_TYPES = 12  # inventory slots (box types, largest volume first)
POOL = 5  # heightmap max-pooling factor (10 mm cells -> 50 mm)
CAND_DIM = len(FEATURES) + 10
GLOB_DIM = 10
TYPE_DIM = 5


@dataclass
class ObsSpec:
    hm_shape: tuple
    n_cands: int = N_CANDS
    cand_dim: int = CAND_DIM
    glob_dim: int = GLOB_DIM
    inv_dim: int = N_TYPES * TYPE_DIM


def obs_spec(state: PalletState) -> ObsSpec:
    nx, ny = state.H.shape
    return ObsSpec(hm_shape=(1, -(-nx // POOL), -(-ny // POOL)))


def heightmap(state: PalletState) -> np.ndarray:
    H = state.H / state.cfg.pallet.max_height
    nx, ny = H.shape
    px, py = -(-nx // POOL) * POOL, -(-ny // POOL) * POOL
    pad = np.ones((px, py))  # outside the pallet counts as full
    pad[:nx, :ny] = H
    return pad.reshape(px // POOL, POOL, py // POOL, POOL).max(axis=(1, 3))[None].astype(np.float32)


def global_features(state: PalletState, box: Box) -> np.ndarray:
    pal, rob = state.cfg.pallet, state.cfg.robot
    size = max(pal.length, pal.width)
    v_pal = pal.length * pal.width * pal.max_height
    remaining = sum(state.types[t].volume * n for t, n in state.inventory.items()) / v_pal
    return np.array(
        [
            box.w / size,
            box.d / size,
            box.h / pal.max_height,
            box.mass / rob.payload,
            box.max_load / (rob.payload * GRAVITY * 10),
            box.volume / v_pal * 20,
            state.progress,
            state.utilization(),
            remaining,
            float(state.H.max()) / pal.max_height,
        ],
        dtype=np.float32,
    )


def inventory_features(state: PalletState) -> np.ndarray:
    pal, rob = state.cfg.pallet, state.cfg.robot
    size = max(pal.length, pal.width)
    out = np.zeros((N_TYPES, TYPE_DIM), dtype=np.float32)
    types = sorted(state.types.values(), key=lambda t: -t.volume)[:N_TYPES]
    for k, t in enumerate(types):
        n = state.inventory.get(t.type_id, 0)
        out[k] = (t.w / size, t.d / size, t.h / pal.max_height, t.mass / rob.payload, n / max(1, state.total_boxes))
    return out.reshape(-1)


def candidate_features(state: PalletState, cand: Candidate) -> np.ndarray:
    pal = state.cfg.pallet
    nx, ny = state.H.shape
    f = cand.features
    return np.array(
        [f[k] for k in FEATURES]
        + [
            cand.score / 5.0,
            cand.i / nx,
            cand.j / ny,
            (cand.i + cand.ni) / nx,
            (cand.j + cand.nj) / ny,
            cand.z / pal.max_height,
            (cand.z + cand.h) / pal.max_height,
            float(cand.o != 0),
            cand.w * cand.d / (pal.length * pal.width),
            len(cand.supporters) / 4.0,
        ],
        dtype=np.float32,
    )


def encode(
    state: PalletState, box: Box, cands: List[Candidate], rng: Optional[np.random.Generator] = None
) -> tuple[Dict[str, np.ndarray], List[Optional[Candidate]]]:
    """Encode the decision. `cands` must be ranked by Q_now with features filled in.

    Keeps the best N_CANDS and, when `rng` is given, shuffles them into the slots so the policy
    has to judge candidates by their features rather than by slot position. Returns the
    observation and the slot -> candidate list (None for empty slots).
    """
    kept = cands[:N_CANDS]
    slots: List[Optional[Candidate]] = list(kept) + [None] * (N_CANDS - len(kept))
    if rng is not None:
        order = rng.permutation(N_CANDS)
        slots = [slots[k] for k in order]
    feats = np.zeros((N_CANDS, CAND_DIM), dtype=np.float32)
    mask = np.zeros(N_CANDS, dtype=bool)
    for k, c in enumerate(slots):
        if c is not None:
            feats[k] = candidate_features(state, c)
            mask[k] = True
    obs = {
        "hm": heightmap(state),
        "glob": global_features(state, box),
        "inv": inventory_features(state),
        "cands": feats,
        "mask": mask,
    }
    return obs, slots
