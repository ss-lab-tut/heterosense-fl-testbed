"""BehaSim I/O utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Record (v1.0)
# ---------------------------------------------------------------------------

@dataclass
class Record:
    """One row in the unified dataset.

    LiDAR-like and bed-like channels are None when the modality is absent.
    semantic_state / posture_state / bed_zone are None for unlabeled real data.
    """
    client_id:   str
    timestamp:   float   # seconds = step x delta_t

    # LiDAR-like channels
    lidar_presence:              Optional[float]
    lidar_height:                Optional[float]
    lidar_motion:                Optional[float]
    lidar_floor_proximity:       Optional[float]
    lidar_height_drop:           Optional[float]
    lidar_static_floor_presence: Optional[float]

    # Bed-pressure-like channels
    bed_pressure_sum:    Optional[float]
    bed_pressure_edge:   Optional[float]
    bed_pressure_change: Optional[float]
    bed_exit_likelihood: Optional[float]

    # Labels (GT for simulated; optional for real)
    semantic_state: Optional[str]
    posture_state:  Optional[str]
    bed_zone:       Optional[str]
    source:         str   # "simulated" | "real"

    # SharedFeatures (computed from latent for simulated; None for unlabeled real)
    occupancy:             Optional[float] = None
    activity_level:        Optional[float] = None
    transition_likelihood: Optional[float] = None
    inactivity_level:      Optional[float] = None
    in_bed_prob:           Optional[float] = None
    bed_exit_prob:         Optional[float] = None
    upright_prob:          Optional[float] = None
    fall_suspected_prob:   Optional[float] = None
    floor_immobility_prob: Optional[float] = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, row: dict) -> Record:
        def _f(key: str) -> Optional[float]:
            v = row.get(key, "")
            if v is None or v == "":
                return None
            try:
                f = float(v)
                return None if (f != f) else f   # NaN -> None
            except (ValueError, TypeError):
                return None

        def _s(key: str) -> Optional[str]:
            v = row.get(key, "")
            if v is None: return None
            s = str(v).strip()
            return s if s else None

        return cls(
            client_id=str(row.get("client_id", "")),
            timestamp=float(row.get("timestamp", 0)),
            lidar_presence=_f("lidar_presence"),
            lidar_height=_f("lidar_height"),
            lidar_motion=_f("lidar_motion"),
            lidar_floor_proximity=_f("lidar_floor_proximity"),
            lidar_height_drop=_f("lidar_height_drop"),
            lidar_static_floor_presence=_f("lidar_static_floor_presence"),
            bed_pressure_sum=_f("bed_pressure_sum"),
            bed_pressure_edge=_f("bed_pressure_edge"),
            bed_pressure_change=_f("bed_pressure_change"),
            bed_exit_likelihood=_f("bed_exit_likelihood"),
            semantic_state=_s("semantic_state"),
            posture_state=_s("posture_state"),
            bed_zone=_s("bed_zone"),
            source=str(row.get("source", "simulated")),
            occupancy=_f("occupancy"),
            activity_level=_f("activity_level"),
            transition_likelihood=_f("transition_likelihood"),
            inactivity_level=_f("inactivity_level"),
            in_bed_prob=_f("in_bed_prob"),
            bed_exit_prob=_f("bed_exit_prob"),
            upright_prob=_f("upright_prob"),
            fall_suspected_prob=_f("fall_suspected_prob"),
            floor_immobility_prob=_f("floor_immobility_prob"),
        )

    def to_dict(self) -> dict:
        def _fmt(v: Optional[float]) -> str:
            return "" if v is None else format(v, ".10g")

        def _lbl(v: Optional[str]) -> str:
            return "" if v is None else v

        return {
            "client_id":   self.client_id,
            "timestamp":   format(self.timestamp, ".10g"),
            "lidar_presence":              _fmt(self.lidar_presence),
            "lidar_height":                _fmt(self.lidar_height),
            "lidar_motion":                _fmt(self.lidar_motion),
            "lidar_floor_proximity":       _fmt(self.lidar_floor_proximity),
            "lidar_height_drop":           _fmt(self.lidar_height_drop),
            "lidar_static_floor_presence": _fmt(self.lidar_static_floor_presence),
            "bed_pressure_sum":    _fmt(self.bed_pressure_sum),
            "bed_pressure_edge":   _fmt(self.bed_pressure_edge),
            "bed_pressure_change": _fmt(self.bed_pressure_change),
            "bed_exit_likelihood": _fmt(self.bed_exit_likelihood),
            "semantic_state": _lbl(self.semantic_state),
            "posture_state":  _lbl(self.posture_state),
            "bed_zone":       _lbl(self.bed_zone),
            "source": self.source,
            # Shared features
            "occupancy":             _fmt(self.occupancy),
            "activity_level":        _fmt(self.activity_level),
            "transition_likelihood": _fmt(self.transition_likelihood),
            "inactivity_level":      _fmt(self.inactivity_level),
            "in_bed_prob":           _fmt(self.in_bed_prob),
            "bed_exit_prob":         _fmt(self.bed_exit_prob),
            "upright_prob":          _fmt(self.upright_prob),
            "fall_suspected_prob":   _fmt(self.fall_suspected_prob),
            "floor_immobility_prob": _fmt(self.floor_immobility_prob),
        }


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

SENSOR_COLUMNS = (
    "lidar_presence",
    "lidar_height",
    "lidar_motion",
    "lidar_floor_proximity",
    "lidar_height_drop",
    "lidar_static_floor_presence",
    "bed_pressure_sum",
    "bed_pressure_edge",
    "bed_pressure_change",
    "bed_exit_likelihood",
)

LIDAR_COLUMNS = (
    "lidar_presence", "lidar_height", "lidar_motion",
    "lidar_floor_proximity", "lidar_height_drop", "lidar_static_floor_presence",
)

BED_COLUMNS = (
    "bed_pressure_sum", "bed_pressure_edge",
    "bed_pressure_change", "bed_exit_likelihood",
)

SHARED_FEATURE_COLUMNS_IO = (
    "occupancy", "activity_level", "transition_likelihood", "inactivity_level",
    "in_bed_prob", "bed_exit_prob", "upright_prob",
    "fall_suspected_prob", "floor_immobility_prob",
)


# ---------------------------------------------------------------------------
# extract_column (canonical implementation)
# ---------------------------------------------------------------------------

def extract_column(records: list[Record], col: str) -> np.ndarray:
    """Return float64 array of non-None values for *col* across *records*."""
    values = [getattr(r, col, None) for r in records]
    return np.array([v for v in values if v is not None], dtype=np.float64)


def records_for_state(records: list[Record], state: str) -> list[Record]:
    return [r for r in records if r.semantic_state == state]
