"""BehaSim BehaviorModel: latent semantic process via Markov chain."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum

import numpy as np

from heterosense._core._config_schema import ClientConfig


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SemanticState(str, Enum):
    ABSENT     = "ABSENT"
    STATIONARY = "STATIONARY"
    WALKING    = "WALKING"
    TRANSITION = "TRANSITION"
    ABNORMAL   = "ABNORMAL"


class Posture(str, Enum):
    UPRIGHT = "UPRIGHT"
    SEATED  = "SEATED"
    LYING   = "LYING"


class AbnormalType(str, Enum):
    """Subtypes of ABNORMAL events.

    FALL                 : impact (phase 1-2) + early post-fall (phase 3-5)
    PROLONGED_IMMOBILITY : sustained floor immobility (phase >= 6)
    NEAR_FALL            : flagged when ABNORMAL episode is brief (1-2 steps then recovery)
    RECOVERED_FALL       : ABNORMAL episode followed by WALKING within 3 steps
    """
    NONE                 = "NONE"
    FALL                 = "FALL"
    PROLONGED_IMMOBILITY = "PROLONGED_IMMOBILITY"
    NEAR_FALL            = "NEAR_FALL"
    RECOVERED_FALL       = "RECOVERED_FALL"


class BedZone(str, Enum):
    ON_BED   = "ON_BED"
    BED_EDGE = "BED_EDGE"
    OFF_BED  = "OFF_BED"


# ---------------------------------------------------------------------------
# Latent state
# ---------------------------------------------------------------------------

@dataclass
class LatentState:
    t:               int
    state:           SemanticState
    x:               float
    y:               float
    velocity:        float
    posture:         Posture
    abnormal_type:   AbnormalType
    bed_zone:        BedZone = BedZone.OFF_BED
    abnormal_phase:  int     = 0  # 0=not ABNORMAL, 1=impact, 2+=rest


# ---------------------------------------------------------------------------
# Default transition matrix
# ---------------------------------------------------------------------------

DEFAULT_TRANSITION_MATRIX: dict[SemanticState, dict[SemanticState, float]] = {
    SemanticState.ABSENT: {
        SemanticState.ABSENT:  0.70,
        SemanticState.WALKING: 0.30,
    },
    SemanticState.STATIONARY: {
        SemanticState.STATIONARY: 0.60,
        SemanticState.WALKING:    0.25,
        SemanticState.TRANSITION: 0.14,
        SemanticState.ABNORMAL:   0.01,
    },
    SemanticState.WALKING: {
        SemanticState.WALKING:    0.60,
        SemanticState.STATIONARY: 0.25,
        SemanticState.TRANSITION: 0.14,
        SemanticState.ABNORMAL:   0.01,
    },
    SemanticState.TRANSITION: {
        SemanticState.STATIONARY: 0.70,
        SemanticState.WALKING:    0.20,
        SemanticState.ABNORMAL:   0.10,
    },
    SemanticState.ABNORMAL: {
        SemanticState.STATIONARY: 0.80,
        SemanticState.ABNORMAL:   0.20,
    },
}


_POSITION_SIGMA: dict[SemanticState, float] = {
    SemanticState.STATIONARY: 0.05,
    SemanticState.TRANSITION:  0.10,
    SemanticState.ABNORMAL:    0.20,
}
_WALK_STEP_RANGE: float  = 0.50
_ROOM_ENTRY_MARGIN: float = 0.50


# States for which abnormal_rate directly overrides P(->ABNORMAL)
_ABNORMAL_RATE_STATES = (
    SemanticState.STATIONARY,
    SemanticState.WALKING,
    SemanticState.TRANSITION,   # v1.0: included so rate controls ALL direct paths
)

# Default base P(TRANSITION->ABNORMAL) before any rate override
_TRANSITION_ABNORMAL_BASE: float = 0.10


def build_transition_matrix(
    abnormal_rate: float,
) -> dict[SemanticState, dict[SemanticState, float]]:
    """Build transition matrix controlling ALL direct paths to ABNORMAL.

    abnormal_rate sets P(s->ABNORMAL) for every source state that has a
    direct transition to ABNORMAL (STATIONARY, WALKING, TRANSITION).
    When abnormal_rate=0, no direct path to ABNORMAL exists; when
    abnormal_rate=1, every step from those states goes to ABNORMAL.

    Note: ABNORMAL->ABNORMAL (self-loop 0.20) is unchanged -- it controls
    the duration of the abnormal episode, not its entry rate.
    """
    matrix: dict[SemanticState, dict[SemanticState, float]] = {}
    for state, transitions in DEFAULT_TRANSITION_MATRIX.items():
        if state in _ABNORMAL_RATE_STATES:
            base = {s: p for s, p in transitions.items() if s != SemanticState.ABNORMAL}
            base_sum = sum(base.values())
            new_row = {s: (p / base_sum) * (1.0 - abnormal_rate) for s, p in base.items()}
            new_row[SemanticState.ABNORMAL] = abnormal_rate
            matrix[state] = new_row
        else:
            matrix[state] = dict(transitions)
    return matrix


# ---------------------------------------------------------------------------
# BehaviorModel
# ---------------------------------------------------------------------------

# Near-neighbor posture transition weights: LYING↔SEATED↔UPRIGHT
# Each row = [UPRIGHT, SEATED, LYING] weights given current posture
_TRANSITION_POSTURE_WEIGHTS: dict = None  # initialized after Posture is defined

def _init_posture_weights():
    global _TRANSITION_POSTURE_WEIGHTS
    _TRANSITION_POSTURE_WEIGHTS = {
        Posture.UPRIGHT: [0.10, 0.80, 0.10],
        Posture.SEATED:  [0.40, 0.20, 0.40],
        Posture.LYING:   [0.05, 0.80, 0.15],
    }

_init_posture_weights()


class BehaviorModel:
    """Generates a latent state sequence of length n_steps."""

    def __init__(self, config: ClientConfig, rng: np.random.Generator) -> None:
        self.room_width   = config.room_width
        self.room_height  = config.room_height
        self.bed_pos      = config.bed_position
        self.bed_radius   = config.bed_radius
        self.transition_matrix = build_transition_matrix(config.abnormal_rate)
        self.rng = rng

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def _sample_next_state(self, current: SemanticState) -> SemanticState:
        row    = self.transition_matrix[current]
        states = list(row.keys())
        probs  = np.array([row[s] for s in states], dtype=np.float64)
        probs /= probs.sum()
        return states[int(self.rng.choice(len(states), p=probs))]

    def _init_position(self) -> tuple[float, float]:
        mx = min(_ROOM_ENTRY_MARGIN, self.room_width  / 2.0)
        my = min(_ROOM_ENTRY_MARGIN, self.room_height / 2.0)
        return (float(self.rng.uniform(mx, self.room_width  - mx)),
                float(self.rng.uniform(my, self.room_height - my)))

    def _update_position(
        self, prev: SemanticState, nxt: SemanticState, x: float, y: float,
    ) -> tuple[float, float]:
        if nxt == SemanticState.ABSENT:
            return -1.0, -1.0
        if prev == SemanticState.ABSENT:
            return self._init_position()
        if nxt == SemanticState.WALKING:
            dx = float(self.rng.uniform(-_WALK_STEP_RANGE, _WALK_STEP_RANGE))
            dy = float(self.rng.uniform(-_WALK_STEP_RANGE, _WALK_STEP_RANGE))
        elif nxt in _POSITION_SIGMA:
            s  = _POSITION_SIGMA[nxt]
            dx = float(self.rng.normal(0.0, s))
            dy = float(self.rng.normal(0.0, s))
        else:
            dx, dy = 0.0, 0.0
        return (float(np.clip(x + dx, 0.0, self.room_width)),
                float(np.clip(y + dy, 0.0, self.room_height)))

    def _compute_bed_zone(
        self, x: float, y: float, state: SemanticState, posture: Posture
    ) -> BedZone:
        """BedZone is posture-aware: WALKING in bed area is still OFF_BED."""
        if state == SemanticState.ABSENT or x < 0:
            return BedZone.OFF_BED
        if state == SemanticState.WALKING:
            # Walking through bed area is not "in bed"
            return BedZone.OFF_BED
        bx, by = self.bed_pos
        dist = float(np.hypot(x - bx, y - by))
        if dist < self.bed_radius:
            # UPRIGHT on bed = edge (sitting on edge of bed, not lying in it)
            if posture == Posture.UPRIGHT and state == SemanticState.STATIONARY:
                return BedZone.BED_EDGE
            return BedZone.ON_BED
        elif dist < 2.0 * self.bed_radius:
            return BedZone.BED_EDGE
        return BedZone.OFF_BED

    def _update_posture(self, current: Posture, nxt: SemanticState) -> Posture:
        if nxt in (SemanticState.ABSENT, SemanticState.WALKING):
            return Posture.UPRIGHT
        if nxt == SemanticState.ABNORMAL:
            return Posture.LYING
        if nxt == SemanticState.TRANSITION:
            postures = list(Posture)
            weights = np.array(_TRANSITION_POSTURE_WEIGHTS[current], dtype=np.float64)
            weights /= weights.sum()
            return postures[int(self.rng.choice(len(postures), p=weights))]
        return current

    @staticmethod
    def _compute_velocity(
        state: SemanticState, x_prev: float, y_prev: float, x: float, y: float,
    ) -> float:
        if state == SemanticState.ABSENT or x_prev < 0 or y_prev < 0:
            return 0.0
        return float(np.hypot(x - x_prev, y - y_prev))

    @staticmethod
    def _get_abnormal_type(state: SemanticState, abnormal_phase: int) -> AbnormalType:
        """Distinguish fall subtypes by abnormal_phase (provisional; post-hoc reclassification applied in generate()).

        phase=1         : FALL (impact moment)
        phase=2-5       : FALL (early post-fall)
        phase>=6        : PROLONGED_IMMOBILITY (extended floor immobility)
        """
        if state != SemanticState.ABNORMAL:
            return AbnormalType.NONE
        if abnormal_phase <= 5:
            return AbnormalType.FALL
        return AbnormalType.PROLONGED_IMMOBILITY

    @staticmethod
    def _reclassify_abnormal_subtypes(states: list) -> list:
        """Post-hoc reclassification of ABNORMAL subtypes.

        NEAR_FALL     : ABNORMAL episode lasting only 1-2 steps (brief incident, quick recovery)
        RECOVERED_FALL: ABNORMAL episode followed by WALKING within 3 steps
        """
        n = len(states)
        for i, ls in enumerate(states):
            if ls.state != SemanticState.ABNORMAL or ls.abnormal_phase != 1:
                continue
            # Find episode length
            ep_len = 0
            j = i
            while j < n and states[j].state == SemanticState.ABNORMAL:
                ep_len += 1
                j += 1
            # Classify
            if ep_len <= 2:
                ab_type = AbnormalType.NEAR_FALL
            elif j < n and any(
                states[min(j+k, n-1)].state == SemanticState.WALKING
                for k in range(3)
            ):
                ab_type = AbnormalType.RECOVERED_FALL
            else:
                continue  # keep FALL / PROLONGED_IMMOBILITY as-is
            # Apply to all steps in this episode
            for k in range(i, min(i + ep_len, n)):
                ls_k = states[k]
                # Replace the abnormal_type on the dataclass
                states[k] = dataclasses.replace(ls_k, abnormal_type=ab_type)
        return states

    # ------------------------------------------------------------------
    # Main generation loop
    # ------------------------------------------------------------------

    def generate(self, n_steps: int) -> list[LatentState]:
        """Generate n_steps LatentState objects with ABNORMAL 2-phase tracking."""
        result:          list[LatentState] = []
        current_state  = SemanticState.ABSENT
        x, y           = -1.0, -1.0
        posture        = Posture.UPRIGHT
        abnormal_steps = 0   # consecutive ABNORMAL steps
        transition_steps  = 0  # steps spent in current TRANSITION episode
        transition_target = 0  # sampled duration of current TRANSITION episode

        for t in range(n_steps):
            next_state = self._sample_next_state(current_state)
            # Variable-length TRANSITION: sample duration on entry, enforce it
            if current_state != SemanticState.TRANSITION and next_state == SemanticState.TRANSITION:
                # Sample TRANSITION duration [2, 6] steps at entry
                transition_target = int(self.rng.integers(2, 7))
                transition_steps = 0
            if current_state == SemanticState.TRANSITION:
                transition_steps += 1
                if transition_steps < transition_target and next_state != SemanticState.TRANSITION:
                    next_state = SemanticState.TRANSITION  # extend to meet target duration
            elif next_state != SemanticState.TRANSITION:
                transition_steps = 0
                transition_target = 0
            x_prev, y_prev = x, y
            x, y      = self._update_position(current_state, next_state, x, y)
            velocity  = self._compute_velocity(next_state, x_prev, y_prev, x, y)
            posture   = self._update_posture(posture, next_state)
            bed_zone  = self._compute_bed_zone(x, y, next_state, posture)
            # abnormal_phase updated before ab_type so subtype is correct
            if next_state == SemanticState.ABNORMAL:
                abnormal_steps += 1
            else:
                abnormal_steps = 0
            ab_phase  = abnormal_steps if next_state == SemanticState.ABNORMAL else 0
            ab_type   = self._get_abnormal_type(next_state, ab_phase)

            result.append(LatentState(
                t=t, state=next_state, x=x, y=y, velocity=velocity,
                posture=posture, abnormal_type=ab_type,
                bed_zone=bed_zone, abnormal_phase=ab_phase,
            ))
            current_state = next_state

        return self._reclassify_abnormal_subtypes(result)
