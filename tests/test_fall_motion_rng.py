"""Regression tests for fall-motion child RNG isolation."""
import numpy as np
import pytest

from heterosense._core._behavior_model import (
    AbnormalType,
    BedZone,
    LatentState,
    Posture,
    SemanticState,
)
from heterosense._core._observation_model import (
    _apply_post_fall_motion,
    _generate_abnormal_impact,
    _sample_fall_variant,
)


def _fall_state():
    return LatentState(
        t=0,
        state=SemanticState.ABNORMAL,
        x=1.0,
        y=1.0,
        velocity=0.0,
        posture=Posture.LYING,
        abnormal_type=AbnormalType.FALL,
        bed_zone=BedZone.OFF_BED,
        abnormal_phase=2,
    )


def _assert_parent_stream_unchanged(operation):
    parent = np.random.default_rng(2026)
    control = np.random.default_rng(2026)
    operation(parent)
    np.testing.assert_array_equal(parent.random(16), control.random(16))


def test_fall_variant_sampling_uses_child_rng():
    state = _fall_state()
    _assert_parent_stream_unchanged(lambda rng: _sample_fall_variant(state, rng))


@pytest.mark.parametrize(
    "variant",
    ["hard_floor", "side_fall", "braced_fall", "collapse", "occluded_floor"],
)
def test_abnormal_impact_generation_uses_child_rng(variant):
    _assert_parent_stream_unchanged(
        lambda rng: _generate_abnormal_impact(1.0, 1.0, variant, rng, 1.0)
    )


@pytest.mark.parametrize(
    "pattern",
    ["STILL", "ROLL", "LIMB_MOVEMENT", "SIT_UP_ATTEMPT", "CRAWL_SHIFT"],
)
def test_post_fall_motion_uses_child_rng(pattern):
    points = np.column_stack([
        np.linspace(0.0, 1.0, 30),
        np.linspace(1.0, 2.0, 30),
        np.linspace(0.0, 1.5, 30),
    ])
    _assert_parent_stream_unchanged(
        lambda rng: _apply_post_fall_motion(
            points, 1.0, 1.0, pattern, 0.4, 2, rng, 1.0
        )
    )
