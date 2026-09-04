"""Portable regression test for the HeteroSense-FL v1 default output.

The historical raw-float SHA-256 is kept below for provenance, but raw floating
point bytes can differ across NumPy and CPU implementations.  This test therefore
checks discrete structure exactly and sensor distributions with a tight tolerance.
On the current development machine, a direct same-environment comparison against
tag v1.0.0 is additionally byte-identical.
"""
import hashlib

import numpy as np

from heterosense import ConfigurationManager, DatasetBuilder


LEGACY_V1_RAW_DIGEST = (
    "30c0e2537a8c539edf7e218efa5a2489b93b1173a3042eb0bbf619fe94c567b8"
)
GOLDEN_STRUCT_HASH = (
    "5b2c2d8d69ecc8d7231cf8eb15e9b2663525293fa8236fe5219bb47cd134143b"
)
GOLDEN_LIDAR_SHAPE = (142303, 3)
GOLDEN_PRESSURE_SHAPE = (1000, 16, 16)
GOLDEN_LIDAR_MEAN = np.array([
    2.4759746506360423,
    3.650271436523987,
    0.7162790237902137,
])
GOLDEN_LIDAR_STD = np.array([
    1.4598903526173852,
    1.3722999928467083,
    0.5533178243043589,
])
GOLDEN_LIDAR_MIN = np.array([
    -3.6788625717163086,
    -0.17091839015483856,
    0.0,
])
GOLDEN_LIDAR_MAX = np.array([
    6.072282314300537,
    7.0962371826171875,
    2.5,
])
GOLDEN_PRESSURE_SUMMARY = np.array([
    0.01702099798930623,
    0.05434850955165205,
    0.9364321231842041,
    0.0012787084560841322,
    0.03431500159204006,
    0.30932157963514734,
])


def _build():
    config = ConfigurationManager(None)
    config.config["n_steps"] = 500
    config.config["random_seed"] = 42
    return DatasetBuilder(config.to_sim_config()).build()


def _collect(data):
    lidar_parts = []
    pressure_parts = []
    structure = []
    for client_id in sorted(data):
        for bundle in data[client_id]:
            structure.append((
                client_id,
                repr(round(bundle.timestamp, 6)),
                bundle.semantic_state,
                bundle.posture_state,
                bundle.bed_zone,
                str(bundle.abnormal_phase),
                None if bundle.lidar is None else tuple(bundle.lidar.shape),
                None if bundle.pressure is None else tuple(bundle.pressure.shape),
            ))
            if bundle.lidar is not None:
                lidar_parts.append(bundle.lidar)
            if bundle.pressure is not None:
                pressure_parts.append(bundle.pressure)
    return (
        np.concatenate(lidar_parts, axis=0),
        np.stack(pressure_parts, axis=0),
        structure,
    )


def _struct_hash(structure) -> str:
    digest = hashlib.sha256()
    for row in structure:
        digest.update(repr(row).encode())
    return digest.hexdigest()


def test_v1_default_structure_is_exact():
    lidar, pressure, structure = _collect(_build())
    assert _struct_hash(structure) == GOLDEN_STRUCT_HASH
    assert lidar.shape == GOLDEN_LIDAR_SHAPE
    assert pressure.shape == GOLDEN_PRESSURE_SHAPE


def test_v1_default_sensor_distribution_is_reproduced():
    lidar, pressure, _ = _collect(_build())
    pressure_summary = np.array([
        pressure.mean(),
        pressure.std(),
        pressure.max(),
        np.quantile(pressure, 0.50),
        np.quantile(pressure, 0.90),
        np.quantile(pressure, 0.99),
    ])

    np.testing.assert_allclose(lidar.mean(axis=0), GOLDEN_LIDAR_MEAN, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(lidar.std(axis=0), GOLDEN_LIDAR_STD, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(lidar.min(axis=0), GOLDEN_LIDAR_MIN, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(lidar.max(axis=0), GOLDEN_LIDAR_MAX, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        pressure_summary,
        GOLDEN_PRESSURE_SUMMARY,
        rtol=1e-6,
        atol=1e-7,
    )
