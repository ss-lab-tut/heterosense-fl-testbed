"""Tests for DatasetBuilder."""
import pytest
import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder


def test_build_returns_dict(data3):
    data, _ = data3
    assert isinstance(data, dict)

def test_build_n3_keys(data3):
    data, _ = data3
    assert len(data) == 3

def test_bundle_count(data3):
    data, _ = data3
    assert all(len(b) == 200 for b in data.values())

def test_client0_has_lidar(data3):
    data, _ = data3
    assert any(b.lidar is not None for b in data["0"])

def test_client0_has_pressure(data3):
    data, _ = data3
    assert any(b.pressure is not None for b in data["0"])

def test_client1_lidar_only(data3):
    data, _ = data3
    assert any(b.lidar is not None for b in data["1"])
    assert all(b.pressure is None for b in data["1"])

def test_client2_pressure_only(data3):
    data, _ = data3
    assert all(b.lidar is None for b in data["2"])
    assert any(b.pressure is not None for b in data["2"])

def test_lidar_shape(data3):
    data, _ = data3
    assert all(b.lidar.shape[1] == 3
               for b in data["0"] if b.lidar is not None)

def test_pressure_shape(data3):
    data, _ = data3
    assert all(b.pressure.shape == (16, 16)
               for b in data["0"] if b.pressure is not None)

def test_semantic_state_is_str(data3):
    data, _ = data3
    assert all(isinstance(b.semantic_state, str) for b in data["0"])

def test_n10():
    sc = CM.from_clients(ClientFactory.make(10), n_steps=60).to_sim_config()
    assert len(DatasetBuilder(sc).build()) == 10

def test_n50():
    sc = CM.from_clients(ClientFactory.make(50), n_steps=60).to_sim_config()
    assert len(DatasetBuilder(sc).build()) == 50

def test_deterministic_seed():
    def build(seed):
        sc = CM.from_clients(ClientFactory.make(3, seed=seed),
                             n_steps=50).to_sim_config()
        return DatasetBuilder(sc).build()["0"][0].semantic_state
    assert build(7) == build(7)
