import numpy as np
import pytest

from heterosense import ConfigurationManager, DatasetBuilder, OccluderConfig
from heterosense._core._observation_model import _apply_structured_occlusion


def test_wall_blocks_contiguous_region_behind_it():
    wall = OccluderConfig(
        name="wall",
        x_min=1.0,
        x_max=1.2,
        y_min=-0.5,
        y_max=0.5,
        height=2.0,
    )
    points = np.array([
        [2.0, 0.0, 1.0],   # directly behind wall: blocked
        [2.0, 1.5, 1.0],   # line of sight passes beside wall: visible
        [0.5, 0.0, 1.0],   # in front of wall: visible
        [2.0, 0.0, 2.1],   # above wall: visible in the 2.5-D model
    ])

    visible = _apply_structured_occlusion(points, (0.0, 0.0), (wall,))

    assert visible.tolist() == points[[1, 2, 3]].tolist()


def test_multiple_occluders_form_union_of_shadows():
    vertical = OccluderConfig("vertical", 1.0, 1.2, -0.3, 0.3)
    horizontal = OccluderConfig("horizontal", -0.3, 0.3, 1.0, 1.2)
    points = np.array([
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ])

    visible = _apply_structured_occlusion(
        points, (0.0, 0.0), (vertical, horizontal)
    )

    assert visible.tolist() == [[2.0, 2.0, 1.0]]


def test_empty_configuration_preserves_default_points_exactly():
    points = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    visible = _apply_structured_occlusion(points, (0.0, 0.0), ())
    assert np.array_equal(visible, points)


def test_occluder_config_validation():
    with pytest.raises(ValueError, match="x_min"):
        OccluderConfig("bad", 1.0, 1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="height"):
        OccluderConfig("bad", 0.0, 1.0, 0.0, 1.0, height=0.0)


def test_builder_keeps_behavior_paired_and_only_removes_observed_points():
    base_client = {
        "client_id": "paired",
        "channel_availability": ["lidar"],
        "sensor_noise_level": 1.0,
        "abnormal_rate": 0.01,
        "bed_position": [2.5, 2.5],
        "bed_radius": 0.8,
        "lidar_occlusion": 0.0,
    }
    wall = {
        "name": "wall",
        "x_min": 1.0,
        "x_max": 1.2,
        "y_min": 0.0,
        "y_max": 5.0,
        "height": 2.5,
    }

    def build(occluders):
        client = {**base_client, "lidar_occluders": occluders}
        config = ConfigurationManager.from_clients(
            [client], n_steps=100, random_seed=7
        ).to_sim_config()
        return DatasetBuilder(config).build()["paired"]

    clear = build([])
    occluded = build([wall])
    clear_truth = [
        (item.semantic_state, item.posture_state, item.bed_zone, item.abnormal_phase)
        for item in clear
    ]
    occluded_truth = [
        (item.semantic_state, item.posture_state, item.bed_zone, item.abnormal_phase)
        for item in occluded
    ]

    assert occluded_truth == clear_truth
    assert all(len(after.lidar) <= len(before.lidar) for before, after in zip(clear, occluded))
    assert any(len(after.lidar) < len(before.lidar) for before, after in zip(clear, occluded))
