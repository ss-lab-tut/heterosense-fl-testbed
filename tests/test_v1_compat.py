"""test_v1_compat.py — backward-compatibility regression (Phase B pass criterion).

Giving a v1 config to v2 must produce v1-identical output. The golden digest is the
full-dataset SHA-256 computed from the v1.0.0 tag (commit f74a061) with the default
config (seed=42, n_steps=500). If any v2 change alters v1-default behavior, this fails.

Regenerate golden (only as an explicit, CHANGELOG-documented op):
    git worktree add --detach /tmp/v1 v1.0.0
    PYTHONPATH=/tmp/v1 python -c "from tests.test_v1_compat import _digest, _build; print(_digest(_build()))"
"""
import hashlib
from heterosense import ConfigurationManager, DatasetBuilder

# SHA-256 of the default dataset produced by HeteroSense-FL v1.0.0 (tag v1.0.0, f74a061).
GOLDEN_V1_DEFAULT = "30c0e2537a8c539edf7e218efa5a2489b93b1173a3042eb0bbf619fe94c567b8"


def _digest(data) -> str:
    h = hashlib.sha256()
    for cid in sorted(data):
        for b in data[cid]:
            h.update(cid.encode()); h.update(repr(round(b.timestamp, 6)).encode())
            h.update(b.lidar.tobytes() if b.lidar is not None else b"none")
            h.update(b.pressure.tobytes() if b.pressure is not None else b"none")
            h.update(b.semantic_state.encode()); h.update(b.posture_state.encode())
            h.update(b.bed_zone.encode()); h.update(str(b.abnormal_phase).encode())
    return h.hexdigest()


def _build():
    cm = ConfigurationManager(None)
    cm.config["n_steps"] = 500
    cm.config["random_seed"] = 42
    return DatasetBuilder(cm.to_sim_config()).build()


def test_v1_default_config_is_byte_identical():
    """v1 default config through v2 code reproduces v1.0.0 output exactly."""
    assert _digest(_build()) == GOLDEN_V1_DEFAULT, (
        "v2 changed v1-default output — backward compatibility broken")


def test_new_pir_knobs_default_to_disabled():
    """The v2 PIR knobs exist and default to the v1-equivalent (disabled) state."""
    from heterosense import ClientConfig
    c = ClientConfig(client_id="x")
    assert c.bedroom_sensor_count == 0 and c.refractory_s == 0.0 and c.report_period_s == 0.0
