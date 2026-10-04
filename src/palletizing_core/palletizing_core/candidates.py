"""M2: candidate generation from extreme points, orientations and anchors."""

from __future__ import annotations

from typing import List, Set, Tuple

from .model import Box, PalletState

Action = Tuple[int, int, int]  # (i, j, orientation)


def generate(state: PalletState, box: Box) -> List[Action]:
    """Candidate (i, j, o) triples for `box`.

    Every extreme point is tried with the box corner anchored to it from all four sides,
    plus alignments against the far pallet walls. Out-of-range anchors are dropped here;
    all other checks are left to the constraint module.
    """
    nx, ny = state.H.shape
    out: Set[Action] = set()
    for o in box.orientations:
        w, d, _ = box.dims(o)
        ni, nj = state.cells(w), state.cells(d)
        if ni > nx or nj > ny:
            continue
        imax, jmax = nx - ni, ny - nj
        anchors = {(0, 0), (imax, 0), (0, jmax), (imax, jmax)}
        for pi, pj in state.ep:
            anchors.update(
                {
                    (pi, pj),
                    (pi - ni, pj),
                    (pi, pj - nj),
                    (pi - ni, pj - nj),
                    (imax, pj),
                    (pi, jmax),
                }
            )
        for i, j in anchors:
            if 0 <= i <= imax and 0 <= j <= jmax:
                out.add((i, j, o))
    return sorted(out)
