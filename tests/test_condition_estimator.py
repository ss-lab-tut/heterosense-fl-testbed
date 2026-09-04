import numpy as np
import pytest

from heterosense import (
    ObservableConditionEstimator,
    balanced_accuracy,
    confusion_counts,
)
from heterosense._scripts.condition_identifiability import OCCLUSION_FEATURE_SETS


def test_estimator_separates_observable_clusters():
    features = np.array([
        [0.0, 0.1],
        [0.1, 0.0],
        [10.0, 9.9],
        [9.9, 10.0],
    ])
    targets = np.array(["clear", "clear", "occluded", "occluded"])
    estimator = ObservableConditionEstimator().fit(features, targets)

    predictions = estimator.predict(np.array([[0.05, 0.05], [10.0, 10.0]]))
    assert predictions.tolist() == ["clear", "occluded"]
    probabilities = estimator.predict_proba(np.array([[0.05, 0.05]]))
    assert probabilities.shape == (1, 2)
    assert probabilities.sum(axis=1) == pytest.approx([1.0])


def test_balanced_accuracy_does_not_hide_minority_failure():
    targets = np.array(["common", "common", "common", "rare"])
    predictions = np.array(["common", "common", "common", "common"])
    assert balanced_accuracy(targets, predictions) == pytest.approx(0.5)


def test_confusion_counts_are_explicit():
    matrix = confusion_counts(
        np.array(["clear", "clear", "occluded"]),
        np.array(["clear", "occluded", "occluded"]),
    )
    assert matrix == {
        "clear": {"clear": 1, "occluded": 1},
        "occluded": {"clear": 0, "occluded": 1},
    }


def test_estimator_rejects_invalid_or_unfitted_use():
    estimator = ObservableConditionEstimator()
    with pytest.raises(RuntimeError, match="fitted"):
        estimator.predict(np.ones((1, 2)))
    with pytest.raises(ValueError, match="two condition classes"):
        estimator.fit(np.ones((2, 2)), np.array(["only", "only"]))
    with pytest.raises(ValueError, match="non-finite"):
        estimator.fit(
            np.array([[0.0, np.nan], [1.0, 1.0]]),
            np.array(["a", "b"]),
        )


def test_occlusion_ablation_really_removes_point_count_signal():
    reduced = OCCLUSION_FEATURE_SETS["without_point_count"]
    assert "point_count" not in reduced
    assert "point_count_ratio" not in reduced
    assert set(OCCLUSION_FEATURE_SETS["lidar_shape_only"]).issubset(reduced)
