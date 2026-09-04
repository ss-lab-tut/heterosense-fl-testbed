"""Small observable-only estimator for deployment conditions.

The estimator intentionally accepts only a numeric feature matrix. Simulator
configuration is never passed to ``fit`` or ``predict``; configuration-derived
condition names may only be used as supervised targets by an experiment.
"""

from __future__ import annotations

import numpy as np


class ObservableConditionEstimator:
    """Standardized nearest-centroid classifier with confidence scores.

    This deterministic model tests whether observation statistics contain
    condition information before a more complex learned model is justified.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.classes_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "ObservableConditionEstimator":
        x, y = _validate_training_data(features, targets)
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("at least two condition classes are required")

        mean = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (x - mean) / scale

        self.classes_ = classes
        self.mean_ = mean
        self.scale_ = scale
        self.centroids_ = np.vstack([
            np.mean(standardized[y == condition], axis=0)
            for condition in classes
        ])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        assert self.classes_ is not None
        return self.classes_[np.argmax(probabilities, axis=1)]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = self._transform(features)
        assert self.centroids_ is not None
        squared_distance = np.mean(
            (x[:, None, :] - self.centroids_[None, :, :]) ** 2,
            axis=2,
        )
        logits = -squared_distance / self.temperature
        logits -= np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits)
        return weights / np.sum(weights, axis=1, keepdims=True)

    def _transform(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.centroids_ is None:
            raise RuntimeError("estimator must be fitted before prediction")
        x = _validate_features(features)
        if x.shape[1] != len(self.mean_):
            raise ValueError(
                f"expected {len(self.mean_)} features, got {x.shape[1]}"
            )
        return (x - self.mean_) / self.scale_


def balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Mean per-class recall."""
    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("targets and predictions must be equal-length 1-D arrays")
    if len(y_true) == 0:
        raise ValueError("targets must not be empty")
    return float(np.mean([
        np.mean(y_pred[y_true == condition] == condition)
        for condition in np.unique(y_true)
    ]))


def confusion_counts(targets: np.ndarray, predictions: np.ndarray) -> dict[str, dict[str, int]]:
    """Return a JSON-serializable confusion matrix keyed by class name."""
    y_true = np.asarray(targets)
    y_pred = np.asarray(predictions)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("targets and predictions must be equal-length 1-D arrays")
    string_true = y_true.astype(str)
    string_pred = y_pred.astype(str)
    classes = sorted(set(string_true) | set(string_pred))
    return {
        actual: {
            predicted: int(np.sum((string_true == actual) & (string_pred == predicted)))
            for predicted in classes
        }
        for actual in classes
    }


def _validate_features(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] == 0:
        raise ValueError("features must be a non-empty 2-D array")
    if not np.isfinite(x).all():
        raise ValueError("features contain non-finite values")
    return x


def _validate_training_data(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _validate_features(features)
    y = np.asarray(targets)
    if y.ndim != 1 or len(y) != len(x):
        raise ValueError("targets must be a 1-D array matching features")
    return x, y.astype(str)
