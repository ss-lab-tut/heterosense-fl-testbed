"""Tests for observation validation."""
import pytest
from heterosense import run_validation


def test_returns_four_results(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    results = run_validation(data, cm)
    assert len(results) == 4

def test_v1_passes(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    assert run_validation(data, cm)[0].passed

def test_v2_passes(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    assert run_validation(data, cm)[1].passed

def test_v3_integrity_passes(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    assert run_validation(data, cm)[2].passed

def test_v4_passes(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    assert run_validation(data, cm)[3].passed

def test_result_fields(data3):
    data, sc = data3
    cm = {c.client_id: list(c.channel_availability) for c in sc.clients}
    for r in run_validation(data, cm):
        assert hasattr(r, "name")
        assert hasattr(r, "passed")
        assert hasattr(r, "reason")
