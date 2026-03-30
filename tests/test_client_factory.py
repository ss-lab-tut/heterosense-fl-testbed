"""Tests for ClientFactory."""
import pytest
from heterosense import ClientFactory


def test_make_n1():
    assert len(ClientFactory.make(1)) == 1

def test_make_n10():
    assert len(ClientFactory.make(10, "round_robin")) == 10

def test_make_n50():
    assert len(ClientFactory.make(50)) == 50

def test_round_robin_order():
    cs = ClientFactory.make(3, "round_robin")
    assert cs[0]["channel_availability"] == ["lidar", "bed"]
    assert cs[1]["channel_availability"] == ["lidar"]
    assert cs[2]["channel_availability"] == ["bed"]

def test_homogeneous():
    cs = ClientFactory.make(5, "round_robin", patterns=["both"])
    assert all(c["channel_availability"] == ["lidar", "bed"] for c in cs)

def test_client_ids_unique():
    cs = ClientFactory.make(10)
    assert len({c["client_id"] for c in cs}) == 10

def test_client_ids_are_str():
    assert all(isinstance(c["client_id"], str) for c in ClientFactory.make(5))

def test_n0_raises():
    with pytest.raises(ValueError):
        ClientFactory.make(0)
