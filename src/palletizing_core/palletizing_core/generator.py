"""Box-set and arrival-order generators for experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import Config
from .model import Box, BoxType

ORDERS = ("random", "small_first", "large_first", "clustered")


def random_box_types(
    rng: np.random.Generator,
    n_types: int,
    size_range: Tuple[float, float] = (200.0, 600.0),
    height_range: Tuple[float, float] = (150.0, 400.0),
    step: float = 50.0,
    density_range: Tuple[float, float] = (80.0, 300.0),  # kg/m^3
    lay_down_prob: float = 0.0,
) -> List[BoxType]:
    types = []
    for k in range(n_types):
        w, d = (float(rng.choice(np.arange(size_range[0], size_range[1] + 1, step))) for _ in range(2))
        h = float(rng.choice(np.arange(height_range[0], height_range[1] + 1, step)))
        rho = float(rng.uniform(*density_range))
        mass = round(rho * w * d * h * 1e-9, 2)
        orients = (0, 1, 2, 3, 4, 5) if rng.random() < lay_down_prob else (0, 1)
        types.append(BoxType(f"T{k:02d}", w, d, h, max(mass, 0.5), None, orients))
    return types


def counts_for_fill(
    rng: np.random.Generator, types: Sequence[BoxType], cfg: Config, fill: float = 0.8
) -> Dict[str, int]:
    """Random counts per type whose total volume is about `fill` of the pallet volume."""
    p = cfg.pallet
    target = fill * p.length * p.width * p.max_height
    share = rng.dirichlet(np.ones(len(types)))
    counts = {t.type_id: max(1, int(round(s * target / t.volume))) for t, s in zip(types, share)}
    return counts


def make_sequence(
    rng: np.random.Generator,
    types: Sequence[BoxType],
    counts: Dict[str, int],
    order: str = "random",
    cfg: Config | None = None,
    mass_noise: float = 0.05,
) -> List[Box]:
    """Arrival sequence of concrete boxes (measured mass jittered around the nominal mass)."""
    if order not in ORDERS:
        raise ValueError(f"unknown order '{order}', expected one of {ORDERS}")
    load_factor = cfg.constraints.default_load_factor if cfg else 3.0
    area_capacity = cfg.constraints.area_load_capacity if cfg else 0.0
    by_id = {t.type_id: t for t in types}
    ids = [tid for tid, n in counts.items() for _ in range(n)]
    if order == "random":
        rng.shuffle(ids)
    elif order == "small_first":
        ids.sort(key=lambda tid: (by_id[tid].volume, tid))
    elif order == "large_first":
        ids.sort(key=lambda tid: (-by_id[tid].volume, tid))
    else:  # clustered: boxes of one type arrive together, type order random
        tids = list(counts)
        rng.shuffle(tids)
        ids = [tid for tid in tids for _ in range(counts[tid])]
    boxes = []
    for k, tid in enumerate(ids):
        t = by_id[tid]
        m = t.mass * float(1.0 + rng.uniform(-mass_noise, mass_noise))
        boxes.append(Box.from_type(f"B{k:03d}", t, load_factor, round(m, 3), area_capacity))
    return boxes
