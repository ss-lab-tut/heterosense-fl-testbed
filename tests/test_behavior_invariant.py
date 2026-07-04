"""test_behavior_invariant.py — the latent behavior trajectory must not depend on sensor
configuration. This underwrites the sub-figure's "number vs. kind" contrast as a valid
PAIRED comparison: the same behavior realization is observed under different sensor configs.

BehaviorModel.generate() runs before any observation draws and reads no sensor/PIR fields,
so a given seed yields identical LatentStates regardless of bedroom_sensor_count /
refractory_s / report_period_s / channel_availability.
"""
import hashlib
import numpy as np
from heterosense._core._config_schema import ClientConfig
from heterosense._core._behavior_model import BehaviorModel

SEED = 2031
N = 3000


def _latent_hash(cfg) -> str:
    ls = BehaviorModel(config=cfg, rng=np.random.default_rng(SEED)).generate(N)
    h = hashlib.sha256()
    for s in ls:
        h.update(repr((s.t, s.state.value, round(s.x, 9), round(s.y, 9), round(s.velocity, 9),
                       s.posture.value, s.abnormal_type.value, s.bed_zone.value,
                       s.abnormal_phase, getattr(s, "room_id", 0))).encode())
    return h.hexdigest()


def test_behavior_invariant_to_sensor_config():
    """Same seed, PIR disabled vs enabled -> byte-identical LatentState trajectory."""
    off = ClientConfig(client_id="h")  # PIR disabled (defaults)
    on = ClientConfig(client_id="h", bedroom_sensor_count=2, refractory_s=5.0, report_period_s=30.0)
    assert _latent_hash(off) == _latent_hash(on), (
        "LatentState trajectory changed with PIR config — behavior/observation RNG separation "
        "broken; sub-figure/method comparisons would no longer be paired")


def test_behavior_invariant_to_channel_availability():
    """Behavior is also independent of which sensor channels are present."""
    both = ClientConfig(client_id="h", channel_availability=("lidar", "bed"))
    bed = ClientConfig(client_id="h", channel_availability=("bed",))
    assert _latent_hash(both) == _latent_hash(bed)
