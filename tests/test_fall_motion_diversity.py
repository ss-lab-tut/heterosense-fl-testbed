import numpy as np

from heterosense._core._behavior_model import BehaviorModel, SemanticState
from heterosense._core._config_schema import ClientConfig


def _abnormal_states(enabled: bool):
    cfg = ClientConfig(
        client_id="fall",
        abnormal_rate=1.0,
        fall_motion_diversity=enabled,
    )
    return BehaviorModel(cfg, np.random.default_rng(42)).generate(8)


def test_fall_motion_diversity_defaults_to_disabled():
    cfg = ClientConfig(client_id="default")
    assert cfg.fall_motion_diversity is False


def test_fall_motion_diversity_is_opt_in():
    off = [s for s in _abnormal_states(False) if s.state == SemanticState.ABNORMAL]
    on = [s for s in _abnormal_states(True) if s.state == SemanticState.ABNORMAL]

    assert off
    assert on
    assert {getattr(s, "motion_pattern", "NONE") for s in off} == {"NONE"}
    assert any(getattr(s, "motion_pattern", "NONE") != "NONE" for s in on)
