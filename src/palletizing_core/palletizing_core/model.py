"""Data model: box types, box instances, placed boxes and the pallet state (heightmap based)."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import GRAVITY, Config

# Axis permutations of (w, d, h). 0 and 1 keep the box upright (rotation about z only);
# 2..5 lay the box on a side and must be explicitly allowed per box type.
ORIENTATIONS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 0, 2),
    (0, 2, 1),
    (2, 0, 1),
    (1, 2, 0),
    (2, 1, 0),
)
UPRIGHT = (0, 1)


@dataclass(frozen=True)
class BoxType:
    type_id: str
    w: float
    d: float
    h: float
    mass: float  # nominal kg
    max_load: Optional[float] = None  # N the box can carry on its top; None -> derived from mass
    orientations: Tuple[int, ...] = UPRIGHT

    @property
    def volume(self) -> float:
        return self.w * self.d * self.h


@dataclass
class Box:
    """One physical box as observed on the conveyor (measured values may differ from the type)."""

    box_id: str
    type_id: str
    w: float
    d: float
    h: float
    mass: float
    max_load: float  # N
    orientations: Tuple[int, ...] = UPRIGHT
    damaged: bool = False

    @classmethod
    def from_type(
        cls,
        box_id: str,
        t: BoxType,
        load_factor: float = 3.0,
        mass: Optional[float] = None,
        area_capacity: float = 0.0,
    ) -> "Box":
        m = t.mass if mass is None else mass
        if t.max_load is not None:
            max_load = t.max_load
        else:
            max_load = load_factor * m * GRAVITY + area_capacity * t.w * t.d
        return cls(box_id, t.type_id, t.w, t.d, t.h, m, max_load, tuple(t.orientations))

    @property
    def volume(self) -> float:
        return self.w * self.d * self.h

    def dims(self, orientation: int) -> Tuple[float, float, float]:
        src = (self.w, self.d, self.h)
        p = ORIENTATIONS[orientation]
        return src[p[0]], src[p[1]], src[p[2]]


@dataclass
class PlacedBox:
    box: Box
    orientation: int
    x: float
    y: float
    z: float
    w: float
    d: float
    h: float
    i0: int
    j0: int
    ni: int
    nj: int
    load: float = 0.0  # N currently carried on top

    @property
    def top(self) -> float:
        return self.z + self.h

    @property
    def center(self) -> Tuple[float, float, float]:
        return self.x + self.w / 2, self.y + self.d / 2, self.z + self.h / 2


@dataclass
class PalletState:
    cfg: Config
    inventory: Dict[str, int]  # remaining (not yet arrived) count per type
    total_boxes: int
    H: np.ndarray = None  # heightmap [nx, ny], mm
    placed: List[PlacedBox] = field(default_factory=list)
    # supports[k] = [(index of box below, contact area mm^2), ...]
    supports: List[List[Tuple[int, float]]] = field(default_factory=list)
    ep: Set[Tuple[int, int]] = field(default_factory=set)  # extreme points in cell indices
    rejected: List[Box] = field(default_factory=list)
    types: Dict[str, BoxType] = field(default_factory=dict)  # known box types (shared, read-only)
    mass: float = 0.0
    moment: np.ndarray = None  # sum(m * center_xy)
    # Footprint and top of placed boxes as rows [i0, j0, i1, j1, top] for vectorised queries.
    rects: np.ndarray = None

    def __post_init__(self) -> None:
        p = self.cfg.pallet
        if self.H is None:
            self.H = np.zeros((p.nx, p.ny), dtype=np.float64)
        if self.moment is None:
            self.moment = np.zeros(2)
        if self.rects is None:
            self.rects = np.zeros((0, 5))
        if not self.ep:
            self.ep = {(0, 0)}

    @classmethod
    def new(cls, cfg: Config, box_types: Sequence[BoxType], counts: Dict[str, int]) -> "PalletState":
        inv = {t.type_id: int(counts.get(t.type_id, 0)) for t in box_types}
        return cls(cfg=cfg, inventory=inv, total_boxes=sum(inv.values()), types={t.type_id: t for t in box_types})

    # ------------------------------------------------------------------ geometry helpers
    def cells(self, length: float) -> int:
        return int(math.ceil(length / self.cfg.pallet.grid - 1e-9))

    @property
    def processed(self) -> int:
        return len(self.placed) + len(self.rejected)

    @property
    def progress(self) -> float:
        return self.processed / self.total_boxes if self.total_boxes else 1.0

    def cog_xy(self) -> Optional[np.ndarray]:
        return self.moment / self.mass if self.mass > 0 else None

    def utilization(self) -> float:
        p = self.cfg.pallet
        used = sum(pb.box.volume for pb in self.placed)
        return used / (p.length * p.width * p.max_height)

    def copy(self) -> "PalletState":
        new = copy.copy(self)
        new.H = self.H.copy()
        new.placed = [copy.copy(pb) for pb in self.placed]
        new.supports = [list(s) for s in self.supports]
        new.ep = set(self.ep)
        new.rejected = list(self.rejected)
        new.moment = self.moment.copy()
        new.rects = self.rects  # replaced (never mutated in place) on commit
        new.inventory = dict(self.inventory)
        return new

    # ------------------------------------------------------------------ mutation
    def consume(self, type_id: str) -> None:
        """Mark one box of this type as arrived (removed from the unknown future)."""
        if self.inventory.get(type_id, 0) > 0:
            self.inventory[type_id] -= 1

    def reject(self, box: Box) -> None:
        self.rejected.append(box)

    def commit(self, cand: "Candidate") -> PlacedBox:  # noqa: F821 (defined in constraints)
        g = self.cfg.pallet.grid
        pb = PlacedBox(
            box=cand.box,
            orientation=cand.o,
            x=cand.i * g,
            y=cand.j * g,
            z=cand.z,
            w=cand.w,
            d=cand.d,
            h=cand.h,
            i0=cand.i,
            j0=cand.j,
            ni=cand.ni,
            nj=cand.nj,
        )
        idx = len(self.placed)
        self.placed.append(pb)
        self.supports.append(list(cand.supporters))
        for k, df in cand.load_delta.items():
            self.placed[k].load += df
        self.H[pb.i0 : pb.i0 + pb.ni, pb.j0 : pb.j0 + pb.nj] = pb.top
        self.rects = np.vstack([self.rects, [pb.i0, pb.j0, pb.i0 + pb.ni, pb.j0 + pb.nj, pb.top]])
        self.mass += pb.box.mass
        cx, cy, _ = pb.center
        self.moment += pb.box.mass * np.array([cx, cy])
        self._update_extreme_points(idx)
        return pb

    def _update_extreme_points(self, idx: int) -> None:
        """Add extreme points generated by the newly placed box (in cell indices).

        New corner points (x+w, y) and (x, y+d) are added both as-is and projected towards
        the origin along the other axis until they hit a placed box or the pallet wall,
        which is the classic Extreme Point rule restricted to the drop-placement case.
        """
        pb = self.placed[idx]
        nx, ny = self.H.shape
        i_end, j_end = pb.i0 + pb.ni, pb.j0 + pb.nj
        new_pts = {(pb.i0, pb.j0), (i_end, pb.j0), (pb.i0, j_end), (i_end, j_end)}
        new_pts.add((i_end, self._project_j(i_end, pb.j0)))
        new_pts.add((self._project_i(pb.i0, j_end), j_end))
        for p in new_pts:
            if 0 <= p[0] < nx and 0 <= p[1] < ny:
                self.ep.add(p)

    def _project_j(self, i: int, j: int) -> int:
        best = 0
        for pb in self.placed:
            if pb.i0 <= i < pb.i0 + pb.ni and pb.j0 + pb.nj <= j:
                best = max(best, pb.j0 + pb.nj)
        return best

    def _project_i(self, i: int, j: int) -> int:
        best = 0
        for pb in self.placed:
            if pb.j0 <= j < pb.j0 + pb.nj and pb.i0 + pb.ni <= i:
                best = max(best, pb.i0 + pb.ni)
        return best
