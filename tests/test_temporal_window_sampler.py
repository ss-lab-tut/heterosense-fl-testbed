"""Tests for TemporalWindowSampler."""
import pytest
import numpy as np
from heterosense import TemporalWindowSampler

KNOWN_STATES = {"ABSENT","STATIONARY","WALKING","TRANSITION","ABNORMAL"}


def test_window_count(bundles0):
    assert sum(1 for _ in TemporalWindowSampler(bundles0, 3)) == 200 - 3 + 1

def test_window_length(bundles0):
    assert all(len(w) == 3 for w in TemporalWindowSampler(bundles0, 3))

def test_center_idx(bundles0):
    assert TemporalWindowSampler(bundles0, 3).center_idx() == 1
    assert TemporalWindowSampler(bundles0, 5).center_idx() == 2
    assert TemporalWindowSampler(bundles0, 1).center_idx() == 0

def test_lidar_z_shape(bundles0):
    sampler = TemporalWindowSampler(bundles0, 3)
    w = next(iter(sampler))
    assert TemporalWindowSampler.lidar_z_series(w).shape == (3,)

def test_pressure_series_shape(bundles0):
    sampler = TemporalWindowSampler(bundles0, 3)
    w = next(iter(sampler))
    assert TemporalWindowSampler.pressure_series(w).shape == (3,)

def test_center_label_is_str(bundles0):
    sampler = TemporalWindowSampler(bundles0, 3)
    w = next(iter(sampler))
    lbl = TemporalWindowSampler.center_label(w, sampler.center_idx())
    assert isinstance(lbl, str)

def test_center_label_known_state(bundles0):
    sampler = TemporalWindowSampler(bundles0, 3)
    w = next(iter(sampler))
    assert TemporalWindowSampler.center_label(w, sampler.center_idx()) in KNOWN_STATES

def test_window1(bundles0):
    assert sum(1 for _ in TemporalWindowSampler(bundles0, 1)) == 200

def test_window5(bundles0):
    assert sum(1 for _ in TemporalWindowSampler(bundles0, 5)) == 196

def test_window0_raises(bundles0):
    with pytest.raises(ValueError):
        TemporalWindowSampler(bundles0, 0)
