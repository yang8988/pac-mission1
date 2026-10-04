"""Matplotlib 2D/3D visualisation of a pallet state."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .model import PalletState  # noqa: E402


def _cuboid_faces(x, y, z, w, d, h):
    p = [
        (x, y, z), (x + w, y, z), (x + w, y + d, z), (x, y + d, z),
        (x, y, z + h), (x + w, y, z + h), (x + w, y + d, z + h), (x, y + d, z + h),
    ]  # fmt: skip
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return [[p[k] for k in f] for f in idx]


def type_colors(state: PalletState) -> Dict[str, tuple]:
    cmap = plt.get_cmap("tab20")
    ids = sorted(state.inventory)
    return {tid: cmap(k % 20) for k, tid in enumerate(ids)}


def plot_pallet(state: PalletState, path: Optional[str | Path] = None, title: str = "", upto: Optional[int] = None):
    """3D view of the pallet plus a heightmap panel. Returns the figure."""
    pal = state.cfg.pallet
    colors = type_colors(state)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    boxes = state.placed if upto is None else state.placed[:upto]
    for pb in boxes:
        faces = _cuboid_faces(pb.x, pb.y, pb.z, pb.w, pb.d, pb.h)
        ax.add_collection3d(
            Poly3DCollection(faces, facecolors=colors.get(pb.box.type_id, "grey"), edgecolors="k", linewidths=0.4, alpha=0.9)
        )
    deck = _cuboid_faces(0, 0, -40, pal.length, pal.width, 40)
    ax.add_collection3d(Poly3DCollection(deck, facecolors="#c8a165", edgecolors="#8a6d3b", linewidths=0.4, alpha=0.6))
    ax.set_xlim(0, pal.length)
    ax.set_ylim(0, pal.width)
    ax.set_zlim(-40, pal.max_height)
    ax.set_box_aspect((pal.length, pal.width, pal.max_height + 40))
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=25, azim=-60)
    ax.set_title(title or "Pallet")

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(
        state.H.T, origin="lower", extent=(0, pal.length, 0, pal.width), cmap="viridis", vmin=0, vmax=pal.max_height
    )
    rx, ry, _ = state.cfg.robot.base
    ax2.plot([rx], [max(ry, -0.0)], marker="v", color="red", markersize=10, clip_on=False)
    ax2.set_title("Heightmap (robot side ▼)")
    ax2.set_xlabel("x [mm]")
    ax2.set_ylabel("y [mm]")
    fig.colorbar(im, ax=ax2, label="height [mm]")
    fig.tight_layout()
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
