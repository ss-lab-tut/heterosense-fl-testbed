"""Tests for observable-only condition-estimator features."""
import numpy as np
import pytest

from heterosense import (
    FEATURE_NAMES,
    ModalityBundle,
    ObservationBaseline,
    ObservationStatisticsExtractor,
)


def _bundle(lidar=None, pressure=None):
    return ModalityBundle(
        client_id="test",
        timestamp=0.0,
        lidar=lidar,
        pressure=pressure,
        semantic_state="SHOULD_NOT_BE_USED",
        posture_state="SHOULD_NOT_BE_USED",
        bed_zone="SHOULD_NOT_BE_USED",
        abnormal_phase=999,
    )


def test_extract_known_observable_statistics():
    lidar = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 0.1],
        [2.0, 4.0, 0.3],
        [3.0, 6.0, 0.8],
    ])
    pressure = np.array([[0.0, 0.1], [0.2, 0.3]])
    baseline = ObservationBaseline(
        point_count=8.0, z_mean=0.5, z_std=0.25, pressure_sum=1.0
    )
    row = ObservationStatisticsExtractor(baseline).extract_dict(
        _bundle(lidar, pressure)
    )

    assert tuple(row) == FEATURE_NAMES
    assert row["lidar_available"] == 1.0
    assert row["pressure_available"] == 1.0
    assert row["point_count"] == 4.0
    assert row["point_count_ratio"] == 0.5
    assert row["z_mean"] == pytest.approx(0.3)
    assert row["z_mean_delta"] == pytest.approx(-0.2)
    assert row["floor02_ratio"] == 0.5
    assert row["floor04_ratio"] == 0.75
    assert row["pressure_sum"] == pytest.approx(0.6)
    assert row["pressure_sum_delta"] == pytest.approx(-0.4)


def test_missing_modalities_use_zero_and_availability_flags():
    row = ObservationStatisticsExtractor().extract_dict(_bundle())

    assert row["lidar_available"] == 0.0
    assert row["pressure_available"] == 0.0
    assert row["lidar_baseline_available"] == 0.0
    assert row["pressure_baseline_available"] == 0.0
    assert all(value == 0.0 for value in row.values())


def test_baseline_fit_uses_only_observations():
    a = _bundle(
        lidar=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        pressure=np.ones((2, 2)),
    )
    b = _bundle(
        lidar=np.array([[0.0, 0.0, 1.0]]),
        pressure=np.full((2, 2), 2.0),
    )
    baseline = ObservationBaseline.fit([a, b])

    assert baseline.point_count == 1.5
    assert baseline.z_mean == 1.0
    assert baseline.z_std == 0.5
    assert baseline.pressure_sum == 6.0


def test_ground_truth_fields_cannot_change_features():
    lidar = np.array([[0.0, 0.0, 0.2], [1.0, 1.0, 0.8]])
    first = _bundle(lidar=lidar)
    second = _bundle(lidar=lidar)
    second.semantic_state = "ABNORMAL"
    second.posture_state = "LYING"
    second.bed_zone = "ON_BED"
    second.abnormal_phase = 1

    extractor = ObservationStatisticsExtractor()
    assert extractor.extract_dict(first) == extractor.extract_dict(second)


def test_transform_has_fixed_shape_and_finite_values():
    bundles = [
        _bundle(lidar=np.array([[0.0, 0.0, 0.1]])),
        _bundle(pressure=np.ones((2, 2))),
        _bundle(),
    ]
    matrix = ObservationStatisticsExtractor().transform(bundles)

    assert matrix.shape == (3, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_transform_empty_input():
    matrix = ObservationStatisticsExtractor().transform([])
    assert matrix.shape == (0, len(FEATURE_NAMES))


def test_invalid_sensor_shapes_raise_clear_errors():
    extractor = ObservationStatisticsExtractor()
    with pytest.raises(ValueError, match="lidar must have shape"):
        extractor.extract_dict(_bundle(lidar=np.ones((3, 2))))
    with pytest.raises(ValueError, match="pressure must be"):
        extractor.extract_dict(_bundle(pressure=np.ones(3)))
