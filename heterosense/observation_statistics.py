"""Observable-only statistics for condition estimation.

This module deliberately does not inspect simulator configuration or ground-truth
labels.  Every feature can be computed from a LiDAR point cloud, a pressure map,
or a baseline fitted from sensor observations available during calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from heterosense._core._observation_model import ModalityBundle


FEATURE_NAMES: tuple[str, ...] = (
    "lidar_available",
    "pressure_available",
    "lidar_baseline_available",
    "pressure_baseline_available",
    "point_count",
    "point_count_ratio",
    "z_mean",
    "z_std",
    "z_q10",
    "z_q50",
    "z_q90",
    "z_mean_delta",
    "z_std_delta",
    "floor02_ratio",
    "floor04_ratio",
    "x_std",
    "y_std",
    "xy_spread",
    "pressure_sum",
    "pressure_mean",
    "pressure_std",
    "pressure_max",
    "pressure_active_ratio",
    "pressure_sum_delta",
)


@dataclass(frozen=True)
class ObservationBaseline:
    """Robust calibration baseline derived only from observed sensor values."""

    point_count: Optional[float] = None
    z_mean: Optional[float] = None
    z_std: Optional[float] = None
    pressure_sum: Optional[float] = None

    @classmethod
    def fit(cls, bundles: Iterable[ModalityBundle]) -> "ObservationBaseline":
        """Fit median reference values from caller-selected calibration frames.

        The method intentionally ignores all ground-truth fields on
        :class:`ModalityBundle`.  The caller is responsible for choosing a
        representative, leakage-free calibration period.
        """
        point_counts: list[float] = []
        z_means: list[float] = []
        z_stds: list[float] = []
        pressure_sums: list[float] = []

        for bundle in bundles:
            lidar = _valid_lidar(bundle.lidar)
            if lidar is not None:
                point_counts.append(float(len(lidar)))
                z_means.append(float(np.mean(lidar[:, 2])))
                z_stds.append(float(np.std(lidar[:, 2])))

            pressure = _valid_pressure(bundle.pressure)
            if pressure is not None:
                pressure_sums.append(float(np.sum(pressure)))

        return cls(
            point_count=_median_or_none(point_counts),
            z_mean=_median_or_none(z_means),
            z_std=_median_or_none(z_stds),
            pressure_sum=_median_or_none(pressure_sums),
        )


class ObservationStatisticsExtractor:
    """Extract a fixed-order feature vector from observable sensor data."""

    feature_names = FEATURE_NAMES

    def __init__(
        self,
        baseline: Optional[ObservationBaseline] = None,
        pressure_active_threshold: float = 0.05,
    ) -> None:
        if pressure_active_threshold < 0.0:
            raise ValueError("pressure_active_threshold must be >= 0")
        self.baseline = baseline or ObservationBaseline()
        self.pressure_active_threshold = float(pressure_active_threshold)

    def extract_dict(self, bundle: ModalityBundle) -> dict[str, float]:
        """Return observable statistics for one frame.

        Unavailable modalities are represented by zero-valued statistics and an
        explicit availability flag.  Baseline-derived fields also have explicit
        flags, so zero never silently means that a baseline was available.
        """
        lidar = _valid_lidar(bundle.lidar)
        pressure = _valid_pressure(bundle.pressure)
        lidar_available = lidar is not None
        pressure_available = pressure is not None
        lidar_baseline_available = (
            _positive_finite(self.baseline.point_count)
            and _finite(self.baseline.z_mean)
            and _finite(self.baseline.z_std)
        )
        pressure_baseline_available = _finite(self.baseline.pressure_sum)

        if lidar_available:
            assert lidar is not None
            x = lidar[:, 0]
            y = lidar[:, 1]
            z = lidar[:, 2]
            point_count = float(len(lidar))
            z_mean = float(np.mean(z))
            z_std = float(np.std(z))
            z_q10, z_q50, z_q90 = (float(v) for v in np.quantile(z, (0.1, 0.5, 0.9)))
            x_std = float(np.std(x))
            y_std = float(np.std(y))
            lidar_values = {
                "point_count": point_count,
                "point_count_ratio": (
                    point_count / float(self.baseline.point_count)
                    if lidar_baseline_available else 0.0
                ),
                "z_mean": z_mean,
                "z_std": z_std,
                "z_q10": z_q10,
                "z_q50": z_q50,
                "z_q90": z_q90,
                "z_mean_delta": (
                    z_mean - float(self.baseline.z_mean)
                    if _finite(self.baseline.z_mean) else 0.0
                ),
                "z_std_delta": (
                    z_std - float(self.baseline.z_std)
                    if _finite(self.baseline.z_std) else 0.0
                ),
                "floor02_ratio": float(np.mean(z < 0.2)),
                "floor04_ratio": float(np.mean(z < 0.4)),
                "x_std": x_std,
                "y_std": y_std,
                "xy_spread": float(np.hypot(x_std, y_std)),
            }
        else:
            lidar_values = {name: 0.0 for name in FEATURE_NAMES[4:18]}

        if pressure_available:
            assert pressure is not None
            pressure_sum = float(np.sum(pressure))
            pressure_values = {
                "pressure_sum": pressure_sum,
                "pressure_mean": float(np.mean(pressure)),
                "pressure_std": float(np.std(pressure)),
                "pressure_max": float(np.max(pressure)),
                "pressure_active_ratio": float(
                    np.mean(pressure > self.pressure_active_threshold)
                ),
                "pressure_sum_delta": (
                    pressure_sum - float(self.baseline.pressure_sum)
                    if pressure_baseline_available else 0.0
                ),
            }
        else:
            pressure_values = {name: 0.0 for name in FEATURE_NAMES[18:]}

        values = {
            "lidar_available": float(lidar_available),
            "pressure_available": float(pressure_available),
            "lidar_baseline_available": float(lidar_baseline_available),
            "pressure_baseline_available": float(pressure_baseline_available),
            **lidar_values,
            **pressure_values,
        }
        return {name: float(values[name]) for name in FEATURE_NAMES}

    def transform(self, bundles: Iterable[ModalityBundle]) -> np.ndarray:
        """Return an ``(n_frames, n_features)`` matrix in fixed feature order."""
        rows = [self.extract_dict(bundle) for bundle in bundles]
        if not rows:
            return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        return np.asarray(
            [[row[name] for name in FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )


def _valid_lidar(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"lidar must have shape (N, 3), got {array.shape}")
    if len(array) == 0:
        return None
    if not np.isfinite(array).all():
        raise ValueError("lidar contains non-finite values")
    return array


def _valid_pressure(value: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"pressure must be a non-empty 2-D array, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("pressure contains non-finite values")
    return array


def _median_or_none(values: list[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _finite(value: Optional[float]) -> bool:
    return value is not None and bool(np.isfinite(value))


def _positive_finite(value: Optional[float]) -> bool:
    return _finite(value) and float(value) > 0.0
