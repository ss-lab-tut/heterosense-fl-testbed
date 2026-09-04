"""Held-out-geometry pilot for observable structured-occlusion inference."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from heterosense import (
    ConfigurationManager,
    DatasetBuilder,
    FEATURE_NAMES,
    ObservableConditionEstimator,
    ObservationBaseline,
    ObservationStatisticsExtractor,
)


@dataclass(frozen=True)
class Geometry:
    name: str
    occluders: tuple[dict, ...]


GEOMETRIES = (
    Geometry("vertical_near", ({
        "name": "wall", "x_min": 1.2, "x_max": 1.4,
        "y_min": 0.0, "y_max": 5.0, "height": 2.5,
    },)),
    Geometry("vertical_mid", ({
        "name": "wall", "x_min": 2.3, "x_max": 2.5,
        "y_min": 0.0, "y_max": 5.0, "height": 2.5,
    },)),
    Geometry("horizontal_near", ({
        "name": "wall", "x_min": 0.0, "x_max": 5.0,
        "y_min": 1.2, "y_max": 1.4, "height": 2.5,
    },)),
    Geometry("horizontal_mid", ({
        "name": "wall", "x_min": 0.0, "x_max": 5.0,
        "y_min": 2.3, "y_max": 2.5, "height": 2.5,
    },)),
    Geometry("central_furniture", ({
        "name": "furniture", "x_min": 2.0, "x_max": 3.0,
        "y_min": 2.0, "y_max": 3.0, "height": 0.9,
    },)),
)

FEATURE_SETS = {
    "all_observable_features": tuple(FEATURE_NAMES),
    "without_point_count": tuple(
        name for name in FEATURE_NAMES
        if name not in {"point_count", "point_count_ratio"}
    ),
    "lidar_shape_only": (
        "z_mean", "z_std", "z_q10", "z_q50", "z_q90",
        "z_mean_delta", "z_std_delta", "floor02_ratio", "floor04_ratio",
        "x_std", "y_std", "xy_spread",
    ),
}


def _client(occluders: tuple[dict, ...]) -> dict:
    return {
        "client_id": "paired_site",
        "room_width": 5.0,
        "room_height": 5.0,
        "channel_availability": ["lidar"],
        "sensor_noise_level": 1.0,
        "abnormal_rate": 0.01,
        "bed_position": [2.5, 2.5],
        "bed_radius": 0.8,
        "lidar_height_gain": 1.0,
        "lidar_motion_gain": 1.0,
        "lidar_floor_gain": 1.0,
        "lidar_occlusion": 0.0,
        "lidar_occluders": list(occluders),
    }


def _build(seed: int, n_steps: int, occluders: tuple[dict, ...]):
    manager = ConfigurationManager.from_clients(
        [_client(occluders)], n_steps=n_steps, random_seed=seed
    )
    return DatasetBuilder(manager.to_sim_config()).build()["paired_site"]


def build_geometry_dataset(
    seed: int,
    geometry: Geometry,
    n_steps: int,
    calibration_steps: int,
    minimum_loss_ratio: float,
) -> dict[str, np.ndarray]:
    """Build paired clear/occluded observations with an effect-based target."""
    clear = _build(seed, n_steps, ())
    occluded = _build(seed, n_steps, geometry.occluders)
    baseline = ObservationBaseline.fit(clear[:calibration_steps])
    extractor = ObservationStatisticsExtractor(baseline)

    clear_eval = clear[calibration_steps:]
    occluded_eval = occluded[calibration_steps:]
    clear_features = extractor.transform(clear_eval)
    occluded_features = extractor.transform(occluded_eval)

    clear_counts = np.asarray([len(bundle.lidar) for bundle in clear_eval])
    occluded_counts = np.asarray([len(bundle.lidar) for bundle in occluded_eval])
    if np.any(occluded_counts > clear_counts):
        raise RuntimeError("structured occlusion added points unexpectedly")
    loss_ratio = (clear_counts - occluded_counts) / np.maximum(clear_counts, 1)
    affected = loss_ratio >= minimum_loss_ratio

    return {
        "features": np.vstack([clear_features, occluded_features]),
        "targets": np.concatenate([
            np.full(len(clear_features), "clear"),
            np.where(affected, "degraded", "clear"),
        ]),
        "loss_ratio": loss_ratio,
    }


def _binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    positive = targets == "degraded"
    negative = ~positive
    predicted_positive = predictions == "degraded"
    recall = float(np.mean(predicted_positive[positive])) if np.any(positive) else 0.0
    false_positive_rate = (
        float(np.mean(predicted_positive[negative])) if np.any(negative) else 0.0
    )
    return {
        "degraded_recall": recall,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": (recall + 1.0 - false_positive_rate) / 2.0,
    }


def run_experiment(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_steps: int = 800,
    calibration_steps: int = 100,
    minimum_loss_ratio: float = 0.10,
) -> dict:
    """Hold out both one trajectory seed and one occluder geometry per fold."""
    if len(seeds) != len(GEOMETRIES):
        raise ValueError("one unique seed is required per held-out geometry")
    if not 0.0 < minimum_loss_ratio < 1.0:
        raise ValueError("minimum_loss_ratio must be in (0, 1)")

    datasets = {
        (seed, geometry.name): build_geometry_dataset(
            seed, geometry, n_steps, calibration_steps, minimum_loss_ratio
        )
        for seed in seeds
        for geometry in GEOMETRIES
    }

    folds = []
    for held_out_seed, held_out_geometry in zip(seeds, GEOMETRIES):
        train_items = [
            item for (seed, geometry), item in datasets.items()
            if seed != held_out_seed and geometry != held_out_geometry.name
        ]
        test = datasets[(held_out_seed, held_out_geometry.name)]
        x_train = np.vstack([item["features"] for item in train_items])
        y_train = np.concatenate([item["targets"] for item in train_items])

        feature_results = {}
        for feature_set, feature_names in FEATURE_SETS.items():
            columns = [FEATURE_NAMES.index(name) for name in feature_names]
            model = ObservableConditionEstimator().fit(
                x_train[:, columns], y_train
            )
            predictions = model.predict(test["features"][:, columns])
            feature_results[feature_set] = _binary_metrics(
                test["targets"], predictions
            )

        folds.append({
            "held_out_seed": held_out_seed,
            "held_out_geometry": held_out_geometry.name,
            "degraded_prevalence": float(np.mean(test["targets"] == "degraded")),
            "mean_point_loss_when_affected": float(
                np.mean(test["loss_ratio"][test["loss_ratio"] >= minimum_loss_ratio])
            ) if np.any(test["loss_ratio"] >= minimum_loss_ratio) else 0.0,
            "feature_sets": feature_results,
        })

    summary = {}
    for feature_set in FEATURE_SETS:
        summary[feature_set] = {}
        for metric in ("balanced_accuracy", "degraded_recall", "false_positive_rate"):
            values = np.asarray([
                fold["feature_sets"][feature_set][metric] for fold in folds
            ])
            summary[feature_set][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }

    return {
        "status": "provisional_simulation_pilot",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "research_question": (
            "Can observable statistics identify degradation caused by an unseen "
            "structured occluder geometry on an unseen trajectory seed?"
        ),
        "structured_occlusion": (
            "deterministic 2.5-D LiDAR-to-point ray/rectangle intersection"
        ),
        "input_features": list(FEATURE_NAMES),
        "prohibited_inputs_used": False,
        "target_definition": f"paired point-loss ratio >= {minimum_loss_ratio}",
        "split": "held-out trajectory seed and held-out occluder geometry",
        "seeds": list(seeds),
        "geometries": [geometry.name for geometry in GEOMETRIES],
        "n_steps": n_steps,
        "calibration_steps": calibration_steps,
        "folds": folds,
        "summary": summary,
        "limitations": [
            "The target uses paired simulator output and is never an estimator input.",
            "The ray model is 2.5-D because LiDAR mounting height is not configured.",
            "Axis-aligned rectangles do not yet represent arbitrary wall angles.",
            "Results are provisional until compared with real LiDAR point clouds.",
            "The recorded environment uses Python 3.14, outside listed classifiers.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--minimum-loss-ratio", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        n_steps=args.n_steps,
        calibration_steps=args.calibration_steps,
        minimum_loss_ratio=args.minimum_loss_ratio,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[OK] Wrote {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
