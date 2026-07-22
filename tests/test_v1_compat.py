"""test_v1_compat.py — backward-compatibility regression (Phase B pass criterion).

Giving a v1 config to v2 must reproduce v1.0.0 output. This is checked in two layers so
the guarantee holds *across platforms* (arm64 macOS vs x86_64 Linux/Windows):

1. **Structure / semantics — exact.** The discrete fields (semantic/posture/bed-zone/
   abnormal-phase), timestamps, and per-sample array *shapes* are derived from the
   platform-independent integer RNG stream, so they must match the golden byte-for-byte.
2. **Sensor float values — tolerant.** LiDAR/pressure point clouds pass through
   floating-point math (trig, rotations, ellipsoid scaling) whose last ULP differs between
   CPU architectures' libm. A byte-exact SHA over raw float bytes is therefore NOT portable
   (it only reproduces on the platform that generated it). We instead compare against a
   stored golden with `np.allclose` — tight enough to catch any real behavioral regression
   (those are gross, >1e-3), loose enough to absorb cross-platform ULP + the golden's float32
   quantization.

Golden = the v1.0.0 default output (seed=42, n_steps=500). The current v2 default path is
byte-identical to v1.0.0 on a fixed platform (this is exactly what layers 1+2 assert), so the
golden fixture is generated from the current default build; see tools/make_v1_golden.py.
To reproduce v1 exactly on your own machine: `git checkout v1.0.0`.
"""
import hashlib
from pathlib import Path

import numpy as np

from heterosense import ConfigurationManager, DatasetBuilder

# Golden fixture (float32 sensor arrays) + the exact hash of the platform-independent
# structure. Regenerate both with: python tools/make_v1_golden.py
_GOLDEN = Path(__file__).parent / "golden" / "v1_default.npz"
GOLDEN_STRUCT_HASH = "5b2c2d8d69ecc8d7231cf8eb15e9b2663525293fa8236fe5219bb47cd134143b"

# Tolerance for cross-platform float reproduction. Real regressions are gross (>1e-3);
# cross-platform libm ULP is ~1e-13 relative and the golden's float32 quantization is
# ~1e-7 relative, so 1e-5 / 1e-6 sits comfortably between the two.
_RTOL, _ATOL = 1e-5, 1e-6


def _build():
    cm = ConfigurationManager(None)
    cm.config["n_steps"] = 500
    cm.config["random_seed"] = 42
    return DatasetBuilder(cm.to_sim_config()).build()


def _collect(data):
    """Split a dataset into (lidar_concat, pressure_stack, structure) for comparison.

    Float arrays are concatenated in a fixed order; per-sample presence/shape and all
    discrete fields go into `structure`, which is platform-independent.
    """
    lidar_parts, pressure_parts, structure = [], [], []
    for cid in sorted(data):
        for b in data[cid]:
            structure.append((
                cid,
                repr(round(b.timestamp, 6)),
                b.semantic_state, b.posture_state, b.bed_zone, str(b.abnormal_phase),
                None if b.lidar is None else tuple(b.lidar.shape),
                None if b.pressure is None else tuple(b.pressure.shape),
            ))
            if b.lidar is not None:
                lidar_parts.append(np.ascontiguousarray(b.lidar, dtype=np.float64))
            if b.pressure is not None:
                pressure_parts.append(np.ascontiguousarray(b.pressure, dtype=np.float64))
    lidar = np.concatenate(lidar_parts, axis=0) if lidar_parts else np.zeros((0, 3))
    pressure = np.stack(pressure_parts, axis=0) if pressure_parts else np.zeros((0, 16, 16))
    return lidar, pressure, structure


def _struct_hash(structure) -> str:
    h = hashlib.sha256()
    for row in structure:
        h.update(repr(row).encode())
    return h.hexdigest()


def test_v1_default_config_is_reproduced():
    """v1 default config through v2 code reproduces v1.0.0 output (structure exact, floats tol)."""
    lidar, pressure, structure = _collect(_build())

    # Layer 1: structure / semantics — must be byte-identical (platform-independent).
    assert _struct_hash(structure) == GOLDEN_STRUCT_HASH, (
        "v2 changed v1-default structure/semantics (states, phases, or array shapes) "
        "— backward compatibility broken")

    # Layer 2: sensor float values — tolerant to cross-platform ULP.
    golden = np.load(_GOLDEN)
    g_lidar = golden["lidar"].astype(np.float64)
    g_pressure = golden["pressure"].astype(np.float64)
    assert lidar.shape == g_lidar.shape and pressure.shape == g_pressure.shape, (
        "v2 changed the v1-default point-cloud shapes — backward compatibility broken")
    assert np.allclose(lidar, g_lidar, rtol=_RTOL, atol=_ATOL), (
        "v2 changed v1-default LiDAR output — backward compatibility broken")
    assert np.allclose(pressure, g_pressure, rtol=_RTOL, atol=_ATOL), (
        "v2 changed v1-default pressure output — backward compatibility broken")


def test_new_pir_knobs_default_to_disabled():
    """The v2 PIR knobs exist and default to the v1-equivalent (disabled) state."""
    from heterosense import ClientConfig
    c = ClientConfig(client_id="x")
    assert c.bedroom_sensor_count == 0 and c.refractory_s == 0.0 and c.report_period_s == 0.0
