"""Tests for the opt-in fall-motion diversity feature."""
import numpy as np

from heterosense import ClientFactory, ConfigurationManager
from heterosense._core._behavior_model import (
    AbnormalType,
    BedZone,
    LatentState,
    Posture,
    SemanticState,
)
from heterosense._core._config_schema import ClientConfig
from heterosense._core._observation_model import ObservationModel


def _impact_lidar(enabled: bool):
    cfg = ClientConfig(client_id="fall", fall_motion_diversity=enabled)
    obs = ObservationModel(cfg, np.random.default_rng(42))
    state = LatentState(
        t=0,
        state=SemanticState.ABNORMAL,
        x=2.5,
        y=2.5,
        velocity=0.0,
        posture=Posture.LYING,
        abnormal_type=AbnormalType.FALL,
        bed_zone=BedZone.OFF_BED,
        abnormal_phase=1,
    )
    return obs.observe(state, 0.0).lidar


def test_fall_motion_diversity_defaults_to_disabled():
    cfg = ClientConfig(client_id="default")
    assert cfg.fall_motion_diversity is False


def test_fall_motion_diversity_is_opt_in():
    off = _impact_lidar(False)
    on = _impact_lidar(True)

    assert off is not None
    assert on is not None
    assert off.shape != on.shape or not np.array_equal(off, on)


def test_client_factory_override_reaches_typed_config():
    clients = ClientFactory.make(
        1,
        strategy="uniform",
        patterns=["lidar_only"],
        base_overrides={"fall_motion_diversity": True},
    )
    config = ConfigurationManager.from_clients(clients, n_steps=10).to_sim_config()
    assert config.clients[0].fall_motion_diversity is True
