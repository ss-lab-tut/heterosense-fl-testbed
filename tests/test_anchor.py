"""test_anchor.py — the validation anchor (Phase B pass criterion, §1-2/§3.1).

v2 is only trustworthy if it reproduces the known 61-home fact: a SINGLE bedroom PIR
cannot observe sub-minute bed-to-toilet (B2T) round-trips, so an annotation-free
extractor's recall collapses to a low level (the real 61-home extractor: recall
0.03–0.55). Assertions are on the MECHANISM and range — NOT tuned to a target
(base_gap = longlie_study's quiescence gap G = 5 min; snapshot is real-calibrated).

The snapshot's data hash is pinned; updating it is an explicit CHANGELOG-documented op.
"""
import numpy as np
import pytest
from heterosense._core._b2t import B2TSnapshot, generate_b2t_night
from heterosense._core._pir_model import PIRModel
from heterosense._core._extractor import b2t_recall

# pinned data-only hash of the frozen 61-home B2T snapshot (tools/make_b2t_snapshot.py)
SNAPSHOT_SHA256 = "a3fe5e91d8157378d4228d9b60997620369ba063caa45d5c94711b0ee71a4ef4"
BASE_GAP_S = 300.0   # = longlie_study best-performing quiescence gap G (5 min); not tuned


class _Cfg:
    def __init__(self, count, refr, rep):
        self.bedroom_sensor_count = count; self.refractory_s = refr; self.report_period_s = rep


def _stream(snap, n_nights=40, n_exits=6, night_s=8 * 3600, seed=2031):
    rng = np.random.default_rng(seed)
    motion, truth, off = [], [], 0
    for _ in range(n_nights):
        m, tr = generate_b2t_night(rng, snap, n_exits=n_exits, night_seconds=night_s)
        motion.append(m); truth += [(a + off, b + off) for a, b in tr]; off += night_s
    return np.concatenate(motion), truth


def test_snapshot_hash_pinned():
    snap = B2TSnapshot.load()
    assert snap.distribution_sha256 == SNAPSHOT_SHA256, (
        "B2T snapshot changed; update SNAPSHOT_SHA256 only via a CHANGELOG-documented op")
    # sanity: real-calibrated sub-minute mass present
    assert snap.provenance.get("source_data_doi") == "10.5281/zenodo.15708568"


def test_anchor_single_pir_recall_collapses():
    """Single bedroom PIR: recall lands in the observed 61-home range (0.03–0.55)."""
    snap = B2TSnapshot.load()
    motion, truth = _stream(snap)
    pm = PIRModel(_Cfg(1, 5.0, 0.0), rng=np.random.default_rng(7))
    ev = pm.observe_sequence(motion, delta_t=1.0)
    recall = b2t_recall(ev, truth, report_period_s=0.0, base_gap_s=BASE_GAP_S, sensor_count=1)
    assert 0.03 <= recall <= 0.55, f"single-PIR recall {recall:.3f} outside 61-home range"


def test_anchor_report_period_monotone():
    """Coarser report period never IMPROVES recall (temporal-resolution mechanism)."""
    snap = B2TSnapshot.load()
    motion, truth = _stream(snap)
    prev = 1.0
    for rep in [0, 10, 30, 60]:
        pm = PIRModel(_Cfg(1, 5.0, rep), rng=np.random.default_rng(7))
        ev = pm.observe_sequence(motion, delta_t=1.0)
        r = b2t_recall(ev, truth, report_period_s=rep, base_gap_s=BASE_GAP_S, sensor_count=1)
        assert r <= prev + 1e-9, f"recall increased with coarser report_period at {rep}s"
        prev = r


def test_anchor_second_sensor_recovers():
    """A second absence-confirming sensor substantially recovers recall (sub-figure)."""
    snap = B2TSnapshot.load()
    motion, truth = _stream(snap)
    r1 = b2t_recall(PIRModel(_Cfg(1, 5.0, 0.0), rng=np.random.default_rng(7)).observe_sequence(motion),
                    truth, report_period_s=0.0, base_gap_s=BASE_GAP_S, sensor_count=1)
    r2 = b2t_recall(PIRModel(_Cfg(2, 5.0, 0.0), rng=np.random.default_rng(7)).observe_sequence(motion),
                    truth, report_period_s=0.0, base_gap_s=BASE_GAP_S, sensor_count=2)
    assert r2 > 2 * r1, f"2nd sensor did not recover recall (r1={r1:.3f}, r2={r2:.3f})"
