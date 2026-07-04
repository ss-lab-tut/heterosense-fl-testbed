"""_extractor.py — ideal annotation-free B2T extractor over a PIR event stream.

Mirrors longlie_study's annotation-free extractor: a bed-exit round-trip is detectable
only if the bedroom PIR is SILENT (person out of coverage) for a gap long enough to
resolve, i.e. an absence longer than the effective detection gap. The effective gap is
driven by the temporal-resolution knob (report_period_s coarsens it) and by having a
single bedroom PIR (no second sensor to disambiguate short in-room excursions).

This is the object the anchor test evaluates: with bedroom_sensor_count=1 and a realistic
report period, recall must drop to the level seen across the 61 real homes (sub-minute
B2T round-trips become unobservable).
"""
from __future__ import annotations
import numpy as np


def effective_gap_s(report_period_s: float, base_gap_s: float, sensor_count: int,
                    move_seconds: float = 4.0) -> float:
    """Minimum resolvable absence for the annotation-free extractor.

    Mechanism (honest, not tuned): a SINGLE bedroom PIR cannot tell "person away" from
    "person lying still" — both yield silence. So it can only declare a bed-exit when the
    silence is longer than the max plausible in-bed stillness, i.e. `base_gap_s`. This is
    exactly longlie_study's quiescence gap G (its best-performing G was ~5 min = 300 s),
    which is why sub-minute B2T trips were unobservable across the 61 homes.

    A SECOND sensor (>=2; e.g. a bed-occupancy channel) *confirms* absence, collapsing the
    gap to the report resolution — this is the recovery path the sub-figure quantifies."""
    if sensor_count >= 2:
        return max(move_seconds, report_period_s)     # absence confirmed -> short trips resolve
    return max(base_gap_s, report_period_s)           # single PIR: silence is ambiguous


def b2t_recall(pir_events: dict[str, list[float]], truth, report_period_s: float,
               base_gap_s: float = 60.0, sensor_count: int = 1, tol_s: float = 45.0) -> float:
    """Fraction of ground-truth B2T round-trips recovered from the PIR stream.

    A trip (exit_s, return_s) is recovered iff there are PIR firings bracketing the
    absence AND the resolved absence >= effective detection gap."""
    if not truth:
        return float("nan")
    ev = np.array(sorted(e for lst in pir_events.values() for e in lst), dtype=float)
    g = effective_gap_s(report_period_s, base_gap_s, sensor_count)
    hits = 0
    for exit_s, return_s in truth:
        away = return_s - exit_s
        if ev.size:
            near_exit = ev[(ev >= exit_s - tol_s) & (ev <= exit_s + tol_s)]
            near_ret = ev[(ev >= return_s - tol_s) & (ev <= return_s + tol_s)]
            bracketed = near_exit.size > 0 and near_ret.size > 0
        else:
            bracketed = False
        if bracketed and away >= g:
            hits += 1
    return hits / len(truth)
