"""HeteroSense extended behavior model with support_state field."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import os

from heterosense._core._behavior_model import (
    LatentState, SemanticState, Posture, BedZone, BehaviorModel
)
from heterosense._core._config_schema import ClientConfig

# ---------------------------------------------------------------------------
# support_state enum（最小：2値）
# ---------------------------------------------------------------------------

class SupportState(str, Enum):
    """Body-support stability state -- NOT bed-contact support.

    This enum represents the stability of how the body weight is distributed,
    which affects pressure-map signal intensity. It is independent of whether
    the person is physically touching the bed.

    FULL    -- body weight evenly and stably supported
              (standing on floor, lying flat on bed surface)
              -> pressure map shows distributed, symmetric signal
    PARTIAL -- body weight unevenly or partially supported
              (sitting on bed edge, mid-transfer posture)
              -> pressure map shows edge-concentrated or asymmetric signal

    OFF_BED always maps to FULL because the person is self-supporting
    on the floor (gravity provides stable downward support).
    """
    FULL    = 'FULL_SUPPORT'
    PARTIAL = 'PARTIAL_SUPPORT'

# ---------------------------------------------------------------------------
# LatentStateV3：v1.0のLatentStateにsupport_stateを付加
# ---------------------------------------------------------------------------

@dataclass
class LatentStateV3:
    """LatentState with support_state field.

    All public fields of v1.0 LatentState are mirrored here so that
    code written against v1.0 LatentState works unchanged with LatentStateV3.

    Added field:
        support_state (SupportState): body-support stability state.
            FULL    -- body weight fully supported (floor or flat on bed).
            PARTIAL -- partial support (bed edge, transitional posture).
            Note: this is NOT bed-contact support; it represents the
            stability of body support, independent of bed_zone.
    """
    base:          LatentState
    support_state: SupportState

    # ------------------------------------------------------------------
    # Full mirror of v1.0 LatentState public interface
    # ------------------------------------------------------------------
    @property
    def t(self):               return self.base.t
    @property
    def state(self):           return self.base.state
    @property
    def x(self):               return self.base.x
    @property
    def y(self):               return self.base.y
    @property
    def velocity(self):        return self.base.velocity
    @property
    def posture(self):         return self.base.posture
    @property
    def abnormal_type(self):   return self.base.abnormal_type
    @property
    def bed_zone(self):        return self.base.bed_zone
    @property
    def abnormal_phase(self):  return self.base.abnormal_phase

    # ------------------------------------------------------------------
    # String aliases -- mirror ModalityBundle naming convention
    # (semantic_state = state.value, posture_state = posture.value)
    # ------------------------------------------------------------------
    @property
    def semantic_state(self):  return self.base.state.value
    @property
    def posture_state(self):   return self.base.posture.value

# ---------------------------------------------------------------------------
# support_state の遷移ルール（最小）
# ---------------------------------------------------------------------------

_SUPPORT_PROBS: dict[BedZone, dict[SupportState, float]] = {
    BedZone.ON_BED:   {SupportState.FULL: 0.85, SupportState.PARTIAL: 0.15},
    BedZone.BED_EDGE: {SupportState.FULL: 0.20, SupportState.PARTIAL: 0.80},
    BedZone.OFF_BED:  {SupportState.FULL: 1.00, SupportState.PARTIAL: 0.00},
}

def _sample_support(bed_zone: BedZone, rng: np.random.Generator) -> SupportState:
    probs = _SUPPORT_PROBS[bed_zone]
    if rng.random() < probs[SupportState.FULL]:
        return SupportState.FULL
    return SupportState.PARTIAL

# ---------------------------------------------------------------------------
# Extended behavior model wrapping BehaviorModel
# ---------------------------------------------------------------------------

class _BehaviorModelExt:
    """v1.0 BehaviorModel + support_state。v1.0側は無修正。"""

    def __init__(self, config: ClientConfig, rng: np.random.Generator) -> None:
        self._base_model = BehaviorModel(config=config, rng=rng)
        self._rng = rng

    def generate(self, n_steps: int) -> list[LatentStateV3]:
        base_states = self._base_model.generate(n_steps)
        _ext_states = []
        for ls in base_states:
            support = _sample_support(ls.bed_zone, self._rng)
            _ext_states.append(LatentStateV3(base=ls, support_state=support))
        return _ext_states

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from heterosense._core._config_manager import ConfigurationManager

    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 500
    cfg.config['random_seed'] = 42
    sim_cfg = cfg.to_sim_config()

    rng = np.random.default_rng(42)
    model = _BehaviorModelExt(sim_cfg.clients[0], rng)
    states = model.generate(500)

    pass  # smoke test removed
    print('=' * 45)

    from collections import Counter
    support_counts = Counter(s.support_state.value for s in states)
    bed_support = Counter(
        (s.bed_zone.value, s.support_state.value)
        for s in states
        if s.bed_zone != BedZone.OFF_BED
    )

    print('Support state distribution:')
    for k, v in support_counts.items():
        print(f'  {k}: {v}')
    print()
    print('BedZone x SupportState:')
    for (bz, ss), cnt in sorted(bed_support.items()):
        print(f'  {bz:10} x {ss:15}: {cnt}')

    # SEATED + BED_EDGE のサンプル確認
    target_a = [s for s in states
                if s.posture == Posture.SEATED
                and s.bed_zone == BedZone.BED_EDGE
                and s.support_state == SupportState.PARTIAL]
    target_b = [s for s in states
                if s.posture == Posture.SEATED
                and s.bed_zone == BedZone.BED_EDGE
                and s.support_state == SupportState.FULL]
    print()
    print(f'Task candidates (n=500):')
    print(f'  SEATED+BED_EDGE+PARTIAL: {len(target_a)}')
    print(f'  SEATED+BED_EDGE+FULL:    {len(target_b)}')
    print()
    pass  # smoke test removed
