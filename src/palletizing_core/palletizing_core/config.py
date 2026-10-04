"""Configuration dataclasses for the palletizing core (units: mm, kg, N, s)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

GRAVITY = 9.81  # m/s^2, used to convert kg to N


@dataclass
class PalletSpec:
    length: float = 1200.0  # x
    width: float = 1000.0  # y
    max_height: float = 1500.0  # stacking height limit above the pallet deck
    max_mass: float = 1000.0  # kg
    grid: float = 10.0  # heightmap cell size

    @property
    def nx(self) -> int:
        return int(round(self.length / self.grid))

    @property
    def ny(self) -> int:
        return int(round(self.width / self.grid))


@dataclass
class RobotSpec:
    # Positions are expressed in the pallet frame (origin at the pallet corner, z=0 at the deck).
    base: Tuple[float, float, float] = (600.0, -900.0, 0.0)
    pick: Tuple[float, float, float] = (-900.0, 500.0, 800.0)
    reach: float = 2800.0  # max horizontal distance from base to the box center
    max_tool_height: float = 2400.0  # max height the tool can reach above the deck
    payload: float = 60.0  # kg
    gripper_size: Tuple[float, float] = (300.0, 200.0)  # vacuum pad footprint
    approach_clearance: float = 50.0  # vertical clearance kept above the box top
    lin_speed: float = 1500.0  # mm/s, for cycle-time estimates
    rot_time: float = 0.5  # s added for a 90 deg wrist rotation
    fixed_time: float = 2.0  # s for pick + release


@dataclass
class ConstraintParams:
    min_support_ratio: float = 0.75  # H3a
    cog_margin: float = 10.0  # H3b, box COG must be this far inside the support polygon
    height_eps: float = 1.0  # cells within this height of z count as support
    # H5: allowed radius of the pallet COG around the pallet center, shrinking with progress.
    # The start value is large so early (necessarily off-center) placements are not blocked.
    cog_radius_start: float = 0.8  # fraction of min(L, W); >= half diagonal -> inactive at start
    cog_radius_end: float = 0.12  # fraction of min(L, W)
    # When a box type gives no allowable top load: max_load = factor * own weight + area term.
    default_load_factor: float = 3.0
    area_load_capacity: float = 0.003  # N/mm^2 (= 3 kN/m^2) of footprint


@dataclass
class ScoreWeights:
    support: float = 1.0  # f1 +
    contact: float = 1.5  # f2 +
    height_gain: float = 2.0  # f3 -
    roughness: float = 1.0  # f4 -
    cog: float = 0.5  # f5 -
    time: float = 0.3  # f6 -
    low_z: float = 1.0  # f7 -
    far_first: float = 0.5  # f8 +


@dataclass
class Config:
    pallet: PalletSpec = field(default_factory=PalletSpec)
    robot: RobotSpec = field(default_factory=RobotSpec)
    constraints: ConstraintParams = field(default_factory=ConstraintParams)
    weights: ScoreWeights = field(default_factory=ScoreWeights)


def _update_dataclass(obj: Any, values: Dict[str, Any]) -> None:
    names = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in names:
            raise KeyError(f"unknown config key '{key}' for {type(obj).__name__}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _update_dataclass(current, value)
        elif isinstance(current, tuple):
            setattr(obj, key, tuple(value))
        else:
            setattr(obj, key, value)


def config_from_dict(values: Dict[str, Any]) -> Config:
    cfg = Config()
    _update_dataclass(cfg, values)
    return cfg


def load_config(path: str | Path) -> Config:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return config_from_dict(data.get("config", data))
