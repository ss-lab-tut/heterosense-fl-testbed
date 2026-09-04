"""Controlled pilot: are observation conditions identifiable from statistics?

Simulator parameters define targets only. Estimator inputs are exactly the 24
values produced by ObservationStatisticsExtractor.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from heterosense import (
    ConfigurationManager,
    DatasetBuilder,
    FEATURE_NAMES,
    ObservableConditionEstimator,
    ObservationBaseline,
    ObservationStatisticsExtractor,
    balanced_accuracy,
    confusion_counts,
)


@dataclass(frozen=True)
class ConditionProfile:
    name: str
    availability: str
    occlusion: str | None
    channels: tuple[str, ...]
    lidar_occlusion: float


PROFILES = (
    ConditionProfile("both_clear", "both", "clear", ("lidar", "bed"), 0.0),
    ConditionProfile("both_occluded", "both", "occluded", ("lidar", "bed"), 0.45),
    ConditionProfile("lidar_clear", "lidar_only", "clear", ("lidar",), 0.0),
    ConditionProfile("lidar_occluded", "lidar_only", "occluded", ("lidar",), 0.45),
    ConditionProfile("pressure_only", "pressure_only", None, ("bed",), 0.0),
)

OCCLUSION_FEATURE_SETS = {
    "all_observable_features": tuple(FEATURE_NAMES),
    "without_point_count": tuple(
        name for name in FEATURE_NAMES
        if name not in {"point_count", "point_count_ratio"}
    ),
    "lidar_shape_only": (
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
    ),
}


def build_seed_dataset(seed: int, n_steps: int, calibration_steps: int) -> dict[str, np.ndarray]:
    """Build paired-condition features for one trajectory seed."""
    if not 0 < calibration_steps < n_steps:
        raise ValueError("calibration_steps must be between 0 and n_steps")

    feature_parts = []
    availability_targets = []
    occlusion_targets = []
    lidar_masks = []

    for profile in PROFILES:
        client = {
            "client_id": "paired_site",
            "room_width": 5.0,
            "room_height": 5.0,
            "channel_availability": list(profile.channels),
            "sensor_noise_level": 1.0,
            "abnormal_rate": 0.01,
            "bed_position": [2.5, 2.5],
            "bed_radius": 0.8,
            "lidar_height_gain": 1.0,
            "lidar_motion_gain": 1.0,
            "lidar_floor_gain": 1.0,
            "lidar_occlusion": profile.lidar_occlusion,
            "bed_pressure_gain": 1.0,
            "bed_edge_sensitivity": 0.8,
        }
        manager = ConfigurationManager.from_clients(
            [client], n_steps=n_steps, random_seed=seed
        )
        bundles = DatasetBuilder(manager.to_sim_config()).build()["paired_site"]
        baseline = ObservationBaseline.fit(bundles[:calibration_steps])
        features = ObservationStatisticsExtractor(baseline).transform(
            bundles[calibration_steps:]
        )
        count = len(features)
        feature_parts.append(features)
        availability_targets.extend([profile.availability] * count)
        occlusion_targets.extend([profile.occlusion or "not_applicable"] * count)
        lidar_masks.extend([profile.occlusion is not None] * count)

    return {
        "features": np.vstack(feature_parts),
        "availability": np.asarray(availability_targets),
        "occlusion": np.asarray(occlusion_targets),
        "has_lidar": np.asarray(lidar_masks, dtype=bool),
    }


def run_experiment(
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    n_steps: int = 800,
    calibration_steps: int = 100,
) -> dict:
    """Run leave-one-seed-out condition-identifiability evaluation."""
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("at least two unique seeds are required")
    datasets = {
        seed: build_seed_dataset(seed, n_steps, calibration_steps)
        for seed in seeds
    }
    folds = []
    for held_out_seed in seeds:
        train = [datasets[seed] for seed in seeds if seed != held_out_seed]
        test = datasets[held_out_seed]
        x_train = np.vstack([item["features"] for item in train])
        x_test = test["features"]

        availability_model = ObservableConditionEstimator().fit(
            x_train, np.concatenate([item["availability"] for item in train])
        )
        availability_prediction = availability_model.predict(x_test)

        train_lidar = np.concatenate([item["has_lidar"] for item in train])
        train_features = np.vstack([item["features"] for item in train])
        train_occlusion = np.concatenate([item["occlusion"] for item in train])
        test_lidar = test["has_lidar"]
        occlusion_ablation = {}
        occlusion_confusions = {}
        for feature_set, feature_names in OCCLUSION_FEATURE_SETS.items():
            columns = [FEATURE_NAMES.index(name) for name in feature_names]
            occlusion_model = ObservableConditionEstimator().fit(
                train_features[train_lidar][:, columns],
                train_occlusion[train_lidar],
            )
            occlusion_prediction = occlusion_model.predict(
                x_test[test_lidar][:, columns]
            )
            occlusion_ablation[feature_set] = balanced_accuracy(
                test["occlusion"][test_lidar], occlusion_prediction
            )
            occlusion_confusions[feature_set] = confusion_counts(
                test["occlusion"][test_lidar], occlusion_prediction
            )

        folds.append({
            "held_out_seed": held_out_seed,
            "availability_balanced_accuracy": balanced_accuracy(
                test["availability"], availability_prediction
            ),
            "occlusion_balanced_accuracy": occlusion_ablation["all_observable_features"],
            "occlusion_feature_ablation": occlusion_ablation,
            "availability_confusion": confusion_counts(
                test["availability"], availability_prediction
            ),
            "occlusion_confusion": occlusion_confusions["all_observable_features"],
            "occlusion_ablation_confusions": occlusion_confusions,
        })

    summary = {}
    for metric in ("availability_balanced_accuracy", "occlusion_balanced_accuracy"):
        values = np.asarray([fold[metric] for fold in folds])
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    summary["occlusion_feature_ablation"] = {}
    for feature_set in OCCLUSION_FEATURE_SETS:
        values = np.asarray([
            fold["occlusion_feature_ablation"][feature_set] for fold in folds
        ])
        summary["occlusion_feature_ablation"][feature_set] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
        }
    return {
        "status": "provisional_simulation_pilot",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "research_question": (
            "Are sensor availability and LiDAR occlusion identifiable from "
            "observable statistics on an unseen trajectory seed?"
        ),
        "input_features": list(FEATURE_NAMES),
        "occlusion_feature_sets": {
            name: list(features)
            for name, features in OCCLUSION_FEATURE_SETS.items()
        },
        "prohibited_inputs_used": False,
        "split": "leave-one-seed-out",
        "seeds": list(seeds),
        "n_steps_per_profile": n_steps,
        "calibration_steps_per_profile": calibration_steps,
        "profiles": [asdict(profile) for profile in PROFILES],
        "folds": folds,
        "summary": summary,
        "limitations": [
            "Simulator targets are not evidence of real-world condition identifiability.",
            "This pilot covers modality availability and random point-drop occlusion only.",
            "Installation geometry and ceiling-view effects are not modeled.",
            "The classifier is a deterministic nearest-centroid baseline, not P1 end-to-end.",
            "The recorded pilot environment uses Python 3.14, outside the versions listed in project classifiers.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=800)
    parser.add_argument("--calibration-steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment(
        n_steps=args.n_steps,
        calibration_steps=args.calibration_steps,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[OK] Wrote {args.output}")
    print(rendered)


if __name__ == "__main__":
    main()
