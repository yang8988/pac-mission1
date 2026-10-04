"""Tier-1 hard constraints (H1-H8). A candidate that fails any check is masked out."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import GRAVITY
from .model import Box, PalletState

# Reason codes for rejected candidates (useful for statistics and debugging).
OUT_OF_BOUNDS = "H1_bounds"
TOO_HIGH = "H1_height"
SUPPORT_AREA = "H3a_support_area"
SUPPORT_COG = "H3b_support_cog"
OVERLOAD = "H4_overload"
PALLET_COG = "H5_pallet_cog"
PALLET_MASS = "H6_pallet_mass"
UNREACHABLE = "H7_reach"
GRIPPER_CLEARANCE = "H8_gripper"


@dataclass
class Candidate:
    box: Box
    i: int
    j: int
    o: int
    w: float
    d: float
    h: float
    ni: int
    nj: int
    z: float = 0.0
    support_ratio: float = 1.0
    supporters: List[Tuple[int, float]] = field(default_factory=list)
    load_delta: Dict[int, float] = field(default_factory=dict)
    cog_after: Tuple[float, float] = (0.0, 0.0)
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0

    def x(self, grid: float) -> float:
        return self.i * grid

    def y(self, grid: float) -> float:
        return self.j * grid

    def center(self, grid: float) -> Tuple[float, float, float]:
        return self.i * grid + self.w / 2, self.j * grid + self.d / 2, self.z + self.h / 2


def evaluate(state: PalletState, box: Box, i: int, j: int, o: int) -> Tuple[Optional[Candidate], str]:
    """Run all tier-1 checks for placing `box` with orientation `o` at cell (i, j).

    Returns (candidate, "") when feasible, otherwise (None, reason_code).
    """
    cands, stats = evaluate_many(state, box, [(i, j, o)])
    if cands:
        return cands[0], ""
    return None, next(r for r, n in stats.items() if n > 0)


def evaluate_many(
    state: PalletState, box: Box, actions: Sequence[Tuple[int, int, int]], limit: Optional[int] = None
) -> Tuple[List[Candidate], Counter]:
    """Tier-1 checks for many (i, j, o) actions of one box.

    The cheap checks (bounds, height, support area) run vectorised per orientation; the rest
    run per surviving candidate. Returns the feasible candidates and a count of reject reasons.

    With `limit`, survivors of the cheap checks are visited lowest-top / best-supported first and
    the search stops after `limit` feasible candidates (fast rollout policy).
    """
    cfg = state.cfg
    pal, cp = cfg.pallet, cfg.constraints
    stats: Counter = Counter()
    out: List[Candidate] = []
    if not actions:
        return out, stats
    # H6: pallet mass (independent of the position).
    if state.mass + box.mass > pal.max_mass + 1e-9:
        stats[PALLET_MASS] += len(actions)
        return out, stats

    nx, ny = state.H.shape
    pending: list = []
    by_o: Dict[int, List[Tuple[int, int]]] = {}
    for i, j, o in actions:
        by_o.setdefault(o, []).append((i, j))
    for o, ij in by_o.items():
        w, d, h = box.dims(o)
        ni, nj = state.cells(w), state.cells(d)
        arr = np.asarray(ij, dtype=np.int64)
        I, J = arr[:, 0], arr[:, 1]
        # H1: inside the pallet footprint.
        inside = (I >= 0) & (J >= 0) & (I + ni <= nx) & (J + nj <= ny)
        stats[OUT_OF_BOUNDS] += int((~inside).sum())
        I, J = I[inside], J[inside]
        if I.size == 0:
            continue
        regions = sliding_window_view(state.H, (ni, nj))[I, J]  # (n, ni, nj)
        Z = regions.max(axis=(1, 2))
        # H1: height limit.
        ok_h = Z + h <= pal.max_height + 1e-6
        stats[TOO_HIGH] += int((~ok_h).sum())
        # H3a: supported area ratio. (H2, no interpenetration, holds by construction: z = max height.)
        ratio = (regions >= (Z - cp.height_eps)[:, None, None]).mean(axis=(1, 2))
        ok_s = ratio >= cp.min_support_ratio - 1e-9
        stats[SUPPORT_AREA] += int((ok_h & ~ok_s).sum())
        for k in np.flatnonzero(ok_h & ok_s):
            pending.append((float(Z[k]) + h, -float(ratio[k]), int(I[k]), int(J[k]), o, w, d, h, ni, nj, float(Z[k])))
    if limit is not None:
        pending.sort(key=lambda p: p[:5])
    for top, neg_ratio, i, j, o, w, d, h, ni, nj, z in pending:
        cand, reason = _check_rest(state, box, i, j, o, w, d, h, ni, nj, z, -neg_ratio)
        if cand is None:
            stats[reason] += 1
        else:
            out.append(cand)
            if limit is not None and len(out) >= limit:
                break
    return out, stats


def _check_rest(
    state: PalletState, box: Box, i: int, j: int, o: int, w: float, d: float, h: float, ni: int, nj: int, z: float,
    ratio: float,
) -> Tuple[Optional[Candidate], str]:  # fmt: skip
    cfg = state.cfg
    pal, cp, rob = cfg.pallet, cfg.constraints, cfg.robot
    g = pal.grid
    x0, y0 = i * g, j * g
    cx, cy = x0 + w / 2, y0 + d / 2

    # H7: simple reachability model (horizontal reach + tool height).
    bx, by, _ = rob.base
    if math.hypot(cx - bx, cy - by) > rob.reach or z + h + rob.approach_clearance > rob.max_tool_height:
        return None, UNREACHABLE

    # H8: the vacuum pad may overhang the box; neighbours under the overhang must not be higher
    # than the box top, otherwise the gripper collides while releasing.
    if not _gripper_clear(state, i, j, ni, nj, w, d, z + h):
        return None, GRIPPER_CLEARANCE

    # Supporting boxes: placed boxes whose top is at z and that overlap the footprint. Their
    # clipped overlap rectangles are exactly the supported cells of the heightmap.
    supporters: List[Tuple[int, float]] = []
    corners: List[Tuple[float, float]] = []
    if z > 0:
        r = state.rects
        a0 = np.maximum(r[:, 0], i)
        a1 = np.minimum(r[:, 2], i + ni)
        b0 = np.maximum(r[:, 1], j)
        b1 = np.minimum(r[:, 3], j + nj)
        hit = (np.abs(r[:, 4] - z) <= cp.height_eps) & (a1 > a0) & (b1 > b0)
        for k in np.flatnonzero(hit):
            x_a, x_b, y_a, y_b = a0[k] * g, a1[k] * g, b0[k] * g, b1[k] * g
            supporters.append((int(k), float((x_b - x_a) * (y_b - y_a))))
            corners += [(x_a, y_a), (x_b, y_a), (x_a, y_b), (x_b, y_b)]
        # H3b: the box COG must lie inside the support polygon (shrunk by a margin).
        if not point_in_convex(convex_hull(corners), (cx, cy), cp.cog_margin):
            return None, SUPPORT_COG

    # H4: propagate the new weight down the support graph and check allowable loads.
    delta: Dict[int, float] = {}
    if supporters and not propagate_load(state, supporters, box.mass * GRAVITY, delta):
        return None, OVERLOAD

    # H5: pallet COG radius, shrinking as the pallet fills up.
    m_after = state.mass + box.mass
    cogx = (state.moment[0] + box.mass * cx) / m_after
    cogy = (state.moment[1] + box.mass * cy) / m_after
    if math.hypot(cogx - pal.length / 2, cogy - pal.width / 2) > cog_radius(state, 1):
        return None, PALLET_COG

    return Candidate(box, i, j, o, w, d, h, ni, nj, z, ratio, supporters, delta, (cogx, cogy)), ""


def cog_radius(state: PalletState, extra: int = 0) -> float:
    """Allowed distance of the pallet COG from the pallet center at the current progress."""
    cp, pal = state.cfg.constraints, state.cfg.pallet
    rho = min(1.0, (state.processed + extra) / max(1, state.total_boxes))
    frac = cp.cog_radius_end + (cp.cog_radius_start - cp.cog_radius_end) * (1.0 - rho) ** 2
    return frac * min(pal.length, pal.width)


def propagate_load(
    state: PalletState, supporters: Sequence[Tuple[int, float]], force: float, delta: Dict[int, float]
) -> bool:
    """Distribute `force` over `supporters` by contact area and push it down recursively.

    `delta` accumulates the extra load per placed box. Returns False if any box would exceed
    its allowable top load. Area-proportional splitting is a conservative approximation of the
    (statically indeterminate) true load distribution.
    """
    total = sum(a for _, a in supporters)
    if total <= 0:
        return True
    for k, area in supporters:
        f = force * area / total
        pb = state.placed[k]
        delta[k] = delta.get(k, 0.0) + f
        if pb.load + delta[k] > pb.box.max_load + 1e-9:
            return False
        below = state.supports[k]
        if below and not propagate_load(state, below, f, delta):
            return False
    return True


def convex_hull(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Andrew's monotone chain. Returns the hull in counter-clockwise order."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_convex(hull: Sequence[Tuple[float, float]], p: Tuple[float, float], margin: float = 0.0) -> bool:
    """True if p is inside the CCW convex polygon with at least `margin` distance to every edge."""
    if len(hull) < 3:
        return False
    n = len(hull)
    for k in range(n):
        ax, ay = hull[k]
        bx, by = hull[(k + 1) % n]
        ex, ey = bx - ax, by - ay
        length = math.hypot(ex, ey)
        # Signed distance, positive on the left (inside for CCW order).
        dist = (ex * (p[1] - ay) - ey * (p[0] - ax)) / length
        if dist < margin - 1e-9:
            return False
    return True


def _gripper_clear(state: PalletState, i: int, j: int, ni: int, nj: int, w: float, d: float, top: float) -> bool:
    gw, gd = state.cfg.robot.gripper_size
    # Align the pad's long side with the box's long side.
    if (w >= d) != (gw >= gd):
        gw, gd = gd, gw
    oi = state.cells(max(0.0, (gw - w) / 2))
    oj = state.cells(max(0.0, (gd - d) / 2))
    if oi == 0 and oj == 0:
        return True
    nx, ny = state.H.shape
    a0, a1 = max(0, i - oi), min(nx, i + ni + oi)
    b0, b1 = max(0, j - oj), min(ny, j + nj + oj)
    return float(state.H[a0:a1, b0:b1].max()) <= top + state.cfg.constraints.height_eps
