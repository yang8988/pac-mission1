"""Episode runner, metrics and an independent validator of the final pallet."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Union

import numpy as np

from .config import GRAVITY, Config
from .constraints import convex_hull, point_in_convex
from .model import Box, BoxType, PalletState
from .planner import PLACE, Decision, GreedyPlanner

Planner = Union[GreedyPlanner, "LookaheadPlanner"]  # noqa: F821


@dataclass
class EpisodeResult:
    state: PalletState
    decisions: List[Decision] = field(default_factory=list)

    def metrics(self) -> Dict[str, float]:
        return compute_metrics(self)


def run_episode(
    cfg: Config,
    box_types: Sequence[BoxType],
    counts: Dict[str, int],
    sequence: Sequence[Box],
    planner: Planner,
    lookahead_k: int = 0,
) -> EpisodeResult:
    """Feed `sequence` to `planner` one box at a time.

    `lookahead_k` boxes after the current one are passed as the observed upstream queue.
    """
    state = PalletState.new(cfg, box_types, counts)
    result = EpisodeResult(state)
    for n, box in enumerate(sequence):
        state.consume(box.type_id)
        queue = [b.type_id for b in sequence[n + 1 : n + 1 + lookahead_k]]
        dec = planner.decide(state, box, queue)
        if dec.action == PLACE:
            state.commit(dec.best)
        else:
            state.reject(box)
        result.decisions.append(dec)
    return result


def compute_metrics(res: EpisodeResult) -> Dict[str, float]:
    s = res.state
    pal = s.cfg.pallet
    placed = s.placed
    times = [d.elapsed_ms for d in res.decisions] or [0.0]
    supports = [d.best.support_ratio for d in res.decisions if d.action == PLACE]
    margins = [(pb.box.max_load - pb.load) / pb.box.max_load for pb in placed if pb.box.max_load > 0]
    cog = s.cog_xy()
    robot_time = sum(d.best.features.get("t_robot", 0.0) for d in res.decisions if d.action == PLACE)
    return {
        "utilization": s.utilization(),
        "placed": len(placed),
        "rejected": len(s.rejected),
        "placed_ratio": len(placed) / max(1, s.total_boxes),
        "max_height": float(s.H.max()),
        "min_support": min(supports) if supports else 1.0,
        "mean_support": float(np.mean(supports)) if supports else 1.0,
        "min_load_margin": min(margins) if margins else 1.0,
        "cog_offset": 0.0 if cog is None else math.hypot(cog[0] - pal.length / 2, cog[1] - pal.width / 2),
        "mass": s.mass,
        "decision_ms_mean": float(np.mean(times)),
        "decision_ms_max": float(np.max(times)),
        "robot_time_s": robot_time,
    }


def validate(state: PalletState) -> List[str]:
    """Re-check the final pallet from the box list alone (independent of the heightmap).

    Uses the cell-rounded footprints the planner works with. Returns a list of violations.
    """
    cfg: Config = state.cfg
    pal, cp = cfg.pallet, cfg.constraints
    g = pal.grid
    errors: List[str] = []
    boxes = state.placed
    for k, a in enumerate(boxes):
        if a.i0 < 0 or a.j0 < 0 or (a.i0 + a.ni) * g > pal.length + 1e-6 or (a.j0 + a.nj) * g > pal.width + 1e-6:
            errors.append(f"{a.box.box_id}: outside pallet")
        if a.top > pal.max_height + 1e-6:
            errors.append(f"{a.box.box_id}: above height limit")
        for b in boxes[k + 1 :]:
            oi = min(a.i0 + a.ni, b.i0 + b.ni) - max(a.i0, b.i0)
            oj = min(a.j0 + a.nj, b.j0 + b.nj) - max(a.j0, b.j0)
            oz = min(a.top, b.top) - max(a.z, b.z)
            if oi > 0 and oj > 0 and oz > 1e-6:
                errors.append(f"{a.box.box_id} and {b.box.box_id} overlap")

    # Support ratio, COG-in-support-polygon and loads, recomputed in placement order.
    loads = [0.0] * len(boxes)
    for k, a in enumerate(boxes):
        if a.z <= 0:
            continue
        below = []
        cells = np.zeros((a.ni, a.nj), dtype=bool)
        for m, b in enumerate(boxes[:k]):
            if abs(b.top - a.z) > cp.height_eps:
                continue
            i0, i1 = max(a.i0, b.i0), min(a.i0 + a.ni, b.i0 + b.ni)
            j0, j1 = max(a.j0, b.j0), min(a.j0 + a.nj, b.j0 + b.nj)
            if i1 > i0 and j1 > j0:
                below.append((m, (i1 - i0) * (j1 - j0)))
                cells[i0 - a.i0 : i1 - a.i0, j0 - a.j0 : j1 - a.j0] = True
        ratio = cells.mean()
        if ratio < cp.min_support_ratio - 1e-9:
            errors.append(f"{a.box.box_id}: support ratio {ratio:.2f}")
        pts = []
        for r in range(a.ni):
            cols = np.flatnonzero(cells[r])
            if cols.size:
                xa, xb = (a.i0 + r) * g, (a.i0 + r + 1) * g
                ya, yb = (a.j0 + cols[0]) * g, (a.j0 + cols[-1] + 1) * g
                pts += [(xa, ya), (xb, ya), (xa, yb), (xb, yb)]
        if not point_in_convex(convex_hull(pts), (a.x + a.w / 2, a.y + a.d / 2), cp.cog_margin):
            errors.append(f"{a.box.box_id}: COG outside support polygon")
        _push(state, below, a.box.mass * GRAVITY, loads)
    for k, b in enumerate(boxes):
        if loads[k] > b.box.max_load + 1e-6:
            errors.append(f"{b.box.box_id}: overloaded {loads[k]:.0f} N > {b.box.max_load:.0f} N")
    return errors


def _push(state: PalletState, below, force: float, loads: List[float]) -> None:
    total = sum(a for _, a in below)
    for m, area in below:
        f = force * area / total
        loads[m] += f
        if state.supports[m]:
            _push(state, state.supports[m], f, loads)
