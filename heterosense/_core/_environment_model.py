"""BehaSim environment model: room geometry and sensor placement."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SensorLayout:
    """Physical layout of sensors in the room."""
    lidar_position: tuple[float, float] = (0.0, 0.0)
    bed_position:   tuple[float, float] = (2.5, 2.5)
    bed_radius:     float               = 0.8

    @classmethod
    def from_config(cls, sensors_cfg: dict, bed_position=None, bed_radius=0.8) -> SensorLayout:
        lp = sensors_cfg.get("lidar_position", [0.0, 0.0])
        bp = bed_position or [2.5, 2.5]
        return cls(
            lidar_position=(float(lp[0]), float(lp[1])),
            bed_position=(float(bp[0]), float(bp[1])),
            bed_radius=float(bed_radius),
        )


class EnvironmentModel:
    """Room geometry: distance and zone calculations.

    Used by calibration tools and visualisation scripts.
    The core simulation loop uses BehaviorModel directly.
    """

    def __init__(self, width: float, height: float, layout: SensorLayout) -> None:
        self.width   = width
        self.height  = height
        self.layout  = layout

    def lidar_distance(self, x: float, y: float) -> float:
        lx, ly = self.layout.lidar_position
        return float(np.hypot(x - lx, y - ly))

    def is_in_room(self, x: float, y: float) -> bool:
        return 0.0 <= x <= self.width and 0.0 <= y <= self.height

    def bed_zone(self, x: float, y: float) -> str:
        bx, by = self.layout.bed_position
        dist = float(np.hypot(x - bx, y - by))
        if dist < self.layout.bed_radius:
            return "ON_BED"
        if dist < 2.0 * self.layout.bed_radius:
            return "BED_EDGE"
        return "OFF_BED"
