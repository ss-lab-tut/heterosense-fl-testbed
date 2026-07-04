"""_pir_model.py — PIR/motion modality for HeteroSense-FL v2.0 (temporal-resolution knob).

Abstracts real PIR physics with three controls (ClientConfig):
  - refractory_s        : dead time after a firing during which the sensor cannot re-fire
  - report_period_s     : event-report quantization period (0 = immediate)
  - bedroom_sensor_count: number of bedroom PIRs (0 disables the modality = v1 behavior)

Design for v2.1 extensibility (coverage + geometry knobs, per EXTENSION_MAP.md §5):
  - PIRSensor carries `position`, `fov_coverage`, `blind_spot_rate` fields (defaults =
    full coverage). v2.1 coverage knob fills these; the firing loop already routes each
    motion sample through `_covers()`, so no redesign is needed.
  - `observe_sequence` reads `room_id` off latent states via getattr(default 0); v2.1
    geometry knob will make per-room PIRs fire only for their own room_id.

Uses a SEPARATE rng (never the shared client rng) so enabling PIR does not perturb the
v1 LiDAR/pressure noise stream (backward compatibility).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PIRSensor:
    sensor_id: str
    refractory_s: float = 0.0
    position: tuple[float, float] | None = None   # v2.1 coverage/geometry hook
    fov_coverage: float = 1.0                       # v2.1: fraction of room covered
    blind_spot_rate: float = 0.0                    # v2.1: spatial dropout
    room_id: int = 0                                # v2.1 geometry: which room this PIR is in

    def _covers(self, rng: np.random.Generator) -> bool:
        """v2.0: always covers (fov_coverage=1, blind_spot=0). v2.1 fills this in."""
        if self.fov_coverage >= 1.0 and self.blind_spot_rate <= 0.0:
            return True
        return (rng.random() < self.fov_coverage) and (rng.random() >= self.blind_spot_rate)


class PIRModel:
    """Turns a per-second bedroom-motion signal into PIR firing events."""

    def __init__(self, config, rng: np.random.Generator | None = None):
        self.count = int(getattr(config, "bedroom_sensor_count", 0))
        self.refractory_s = float(getattr(config, "refractory_s", 0.0))
        self.report_period_s = float(getattr(config, "report_period_s", 0.0))
        # separate rng: enabling PIR must not shift the client's v1 noise draws
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self.sensors = [
            PIRSensor(sensor_id=f"pir{i}", refractory_s=self.refractory_s)
            for i in range(self.count)
        ]

    @property
    def enabled(self) -> bool:
        return self.count >= 1

    def observe_sequence(self, motion, delta_t: float = 1.0) -> dict[str, list[float]]:
        """motion: 1-D bool array (per step). Returns {sensor_id: [event_time_s]}.
        Applies per-sensor refractory then report-period quantization."""
        motion = np.asarray(motion, dtype=bool)
        out: dict[str, list[float]] = {}
        for s in self.sensors:
            last_fire = -np.inf
            events = []
            for step, moving in enumerate(motion):
                if not moving:
                    continue
                t = step * delta_t
                if (t - last_fire) < s.refractory_s:      # refractory: suppress re-fire
                    continue
                if not s._covers(self._rng):              # v2.1 coverage (no-op in v2.0)
                    continue
                last_fire = t
                events.append(t)
            # report-period quantization: collapse events into their period bucket
            if self.report_period_s and self.report_period_s > 0:
                bucketed = sorted({np.floor(e / self.report_period_s) * self.report_period_s
                                   for e in events})
                events = bucketed
            out[s.sensor_id] = events
        return out
