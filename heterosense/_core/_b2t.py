"""_b2t.py — minimal bed-to-toilet (B2T) behavior for HeteroSense-FL v2.0.

v2.0 (Alt-C, temporal-resolution knob) does NOT model room topology. Instead it
samples B2T *away-durations* directly from a frozen, real-calibrated snapshot
(`heterosense/_data/b2t_snapshot.json`, from the 61-home CASAS study — see
tools/make_b2t_snapshot.py). The geometry knob (v2.1) will later DERIVE these
durations from room distance; this module is the seam it plugs into.

The generator emits, at 1-second resolution, a per-step "motion in bedroom" signal
that a PIR observes, plus ground-truth (exit_s, return_s) pairs for the anchor test.
"""
from __future__ import annotations
import json, os
from dataclasses import dataclass
import numpy as np

_DEFAULT_SNAPSHOT = os.path.join(os.path.dirname(__file__), "..", "_data", "b2t_snapshot.json")


@dataclass
class B2TSnapshot:
    probabilities: np.ndarray   # inverse-CDF grid (0..1)
    values: np.ndarray          # return-time values (minutes) at those probabilities
    n: int
    distribution_sha256: str
    provenance: dict

    @classmethod
    def load(cls, path: str | None = None) -> "B2TSnapshot":
        path = path or os.path.normpath(_DEFAULT_SNAPSHOT)
        with open(path) as f:
            d = json.load(f)
        dist = d["distribution"]
        return cls(
            probabilities=np.asarray(dist["probabilities"], float),
            values=np.asarray(dist["values"], float),
            n=int(dist["n"]),
            distribution_sha256=d["distribution_sha256"],
            provenance=d.get("provenance", {}),
        )

    def sample_minutes(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Inverse-CDF sampling of B2T away-durations (minutes)."""
        u = rng.random(size)
        return np.interp(u, self.probabilities, self.values)


def generate_b2t_night(rng: np.random.Generator, snapshot: B2TSnapshot,
                       n_exits: int, night_seconds: int = 8 * 3600,
                       move_seconds: float = 4.0, flicker_period_s: float = 150.0):
    """Generate one night of B2T episodes at 1-Hz resolution.

    In-bed restlessness ("flicker"): while in bed the resident produces a brief motion
    roughly every `flicker_period_s` seconds. This is why a single bedroom PIR cannot
    resolve sub-minute B2T trips — a short absence is indistinguishable from the normal
    gap between in-bed movements (the anchor mechanism). Set `flicker_period_s<=0` to
    disable. NOTE: this is an UNcalibrated behavioral assumption (sim-to-real table).

    Returns
    -------
    motion : np.ndarray[bool], shape (night_seconds,)
        True where the resident is moving in the bedroom (PIR-observable).
    truth  : list[tuple[int, int]]
        Ground-truth (exit_second, return_second) pairs (the events to be detected).
    """
    motion = np.zeros(night_seconds, dtype=bool)
    truth = []
    durations_s = np.clip(snapshot.sample_minutes(rng, n_exits) * 60.0, 1.0, night_seconds / 2)
    # place exits at random, non-overlapping, ordered
    starts = np.sort(rng.integers(0, max(1, night_seconds - int(durations_s.max()) - 1), n_exits))
    cursor = 0
    mv = max(1, int(round(move_seconds)))
    for start, dur in zip(starts, durations_s):
        exit_s = int(max(start, cursor))
        dur_s = int(round(dur))
        if exit_s + dur_s + mv >= night_seconds:
            break
        # motion burst at exit (leaving bed) and at return; absence in between
        motion[exit_s: exit_s + mv] = True                       # exit movement
        motion[exit_s + dur_s: exit_s + dur_s + mv] = True        # return movement
        truth.append((exit_s, exit_s + dur_s))
        cursor = exit_s + dur_s + mv + 1
    # in-bed restlessness flicker: brief motion ~every flicker_period_s while in bed
    if flicker_period_s and flicker_period_s > 0:
        in_bed = np.ones(night_seconds, dtype=bool)
        for e, r in truth:
            in_bed[e:r] = False                       # away during trips
        t = 0
        while t < night_seconds:
            t += int(max(1, rng.exponential(flicker_period_s)))
            if t < night_seconds and in_bed[t]:
                motion[t:t + 1] = True                # a brief in-bed movement
    return motion, truth
