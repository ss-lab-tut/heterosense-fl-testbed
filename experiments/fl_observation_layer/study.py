"""study.py — Phase C: FL observation-layer study (flagship).

Learns the observation model p(latent-state | firing-window; c_h) across N heterogeneous
homes, under 4 FL methods x 3 modality series (SS 4.2/4.2.5), then measures the DOWNSTREAM
detection-delay cost by connecting the estimated exit/return times to longlie_study's
injection protocol (4 iron rules, import-only, unmodified).

Constraint (SS 4.2.5): only the PIR modality is newly implemented (heterosense._core._pir_model).
Bed pressure and LiDAR use the v1 observation model AS-IS.

Latent minimal set: IN_BED (0) vs AWAY (1). Exit = IN_BED->AWAY, Return = AWAY->IN_BED.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
LONGLIE = os.environ.get("LONGLIE", "/Users/x-shao/Projects/teraoka/longlie_study")
sys.path.insert(0, os.path.join(LONGLIE, "src"))

from heterosense._core._b2t import B2TSnapshot, generate_b2t_night
from heterosense._core._pir_model import PIRModel
from heterosense._core._config_schema import ClientConfig
from heterosense._core._observation_model import ObservationModel  # v1 pressure/LiDAR AS-IS
from heterosense._core._behavior_model import LatentState, SemanticState, Posture, AbnormalType, BedZone
import injection as ll_injection  # longlie_study, import-only

SNAP = B2TSnapshot.load()
STEP_S = 10.0             # feature/label time step (s)
NIGHT_S = 8 * 3600
WIN = 3                   # feature window (steps) each side
SERIES = ["pir_only", "pir_pressure", "lidar_upper"]
METHODS = ["local", "centralized", "fedavg", "modality_group"]


# --------------------------------------------------------------------------- #
# Home generation: latent occupancy + per-step features + true exit/return
# --------------------------------------------------------------------------- #
class _Cfg:
    def __init__(self, count, refr, rep):
        self.bedroom_sensor_count = count; self.refractory_s = refr; self.report_period_s = rep


def _v1_pressure_feature(in_bed: bool, rng) -> float:
    """Mean bed pressure via the v1 ObservationModel (UNMODIFIED). High iff ON_BED."""
    ls = LatentState(t=0, state=SemanticState.STATIONARY, x=2.5, y=2.5, velocity=0.0,
                     posture=Posture.LYING, abnormal_type=AbnormalType.NONE,
                     bed_zone=BedZone.ON_BED if in_bed else BedZone.OFF_BED)
    cc = ClientConfig(client_id="p", channel_availability=("bed",))
    om = ObservationModel(config=cc, rng=rng)
    p = om._generate_pressure(ls)
    return float(np.mean(p)) if p is not None else 0.0


def _v1_lidar_feature(in_bed: bool, rng) -> float:
    """Mean LiDAR height via the v1 ObservationModel (UNMODIFIED). Separates in/out of bed."""
    ls = LatentState(t=0, state=SemanticState.STATIONARY if in_bed else SemanticState.WALKING,
                     x=2.5, y=2.5, velocity=0.0 if in_bed else 0.5,
                     posture=Posture.LYING if in_bed else Posture.UPRIGHT,
                     abnormal_type=AbnormalType.NONE,
                     bed_zone=BedZone.ON_BED if in_bed else BedZone.OFF_BED)
    cc = ClientConfig(client_id="l", channel_availability=("lidar",))
    om = ObservationModel(config=cc, rng=rng)
    pts = om._generate_lidar(ls)
    return float(np.mean(pts[:, 2])) if pts is not None and len(pts) else 0.0


def generate_home(home_id, series, count, refr, rep, has_pressure, n_nights, seed):
    """Returns dict(X, y, groups, true_events, meta). X columns depend on series."""
    rng = np.random.default_rng(seed)
    occ_all, motion_all, truth, off = [], [], [], 0
    for _ in range(n_nights):
        motion, tr = generate_b2t_night(rng, SNAP, n_exits=6, night_seconds=NIGHT_S)
        occ = np.ones(NIGHT_S, dtype=bool)  # in_bed=True
        for a, b in tr:
            occ[a:b] = False                # away during trip
        occ_all.append(occ); motion_all.append(motion)
        truth += [(a + off, b + off) for a, b in tr]; off += NIGHT_S
    occ = np.concatenate(occ_all); motion = np.concatenate(motion_all)
    total_s = occ.size
    # PIR events -> per-step "time since last fire" feature
    pm = PIRModel(_Cfg(count, refr, rep), rng=np.random.default_rng(seed + 1))
    ev = sorted(t for lst in pm.observe_sequence(motion, delta_t=1.0).values() for t in lst)
    ev = np.array([e for e in ev if np.isfinite(e)], float)
    # sample at STEP_S grid (vectorized)
    steps = np.arange(WIN * int(STEP_S), total_s - WIN * int(STEP_S), int(STEP_S))
    in_bed = occ[steps]                                    # bool, True=in_bed
    y = (~in_bed).astype(int)                              # 1 = away
    # PIR feature: normalized time since last firing (bounded 600s)
    if ev.size:
        idx = np.searchsorted(ev, steps, side="right") - 1
        last = np.where(idx >= 0, ev[np.clip(idx, 0, ev.size - 1)], -1e9)
        tsl = np.where(idx >= 0, np.minimum(steps - last, 600.0), 600.0) / 600.0
    else:
        tsl = np.ones_like(steps, float)
    # precompute v1 sensor feature distributions ONCE (v1 model UNMODIFIED), then draw per step
    frng = np.random.default_rng(seed + 3)
    if series == "pir_pressure":
        if has_pressure:
            pin = np.array([_v1_pressure_feature(True, frng) for _ in range(40)])
            paw = np.array([_v1_pressure_feature(False, frng) for _ in range(40)])
            press = np.where(in_bed, frng.normal(pin.mean(), pin.std() + 1e-6, steps.size),
                             frng.normal(paw.mean(), paw.std() + 1e-6, steps.size))
        else:
            press = np.zeros(steps.size)                   # pressure absent (missing support)
        X = np.column_stack([tsl, press])
    elif series == "lidar_upper":
        lin = np.array([_v1_lidar_feature(True, frng) for _ in range(40)])
        law = np.array([_v1_lidar_feature(False, frng) for _ in range(40)])
        lz = np.where(in_bed, frng.normal(lin.mean(), lin.std() + 1e-6, steps.size),
                      frng.normal(law.mean(), law.std() + 1e-6, steps.size))
        X = lz.reshape(-1, 1)                              # LiDAR replaces PIR feature
    else:
        X = tsl.reshape(-1, 1)
    return dict(home_id=home_id, series=series, X=X, y=y, has_pressure=has_pressure,
                true_events=truth, count=count, refr=refr, rep=rep,
                steps=steps, total_s=total_s, seed=seed)


# --------------------------------------------------------------------------- #
# Observation model: logistic regression (per feature group) + 4 FL methods
# --------------------------------------------------------------------------- #
def _sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _train_logreg(X, y, w0=None, b0=0.0, lr=0.3, epochs=300, l2=1e-3, balanced=True):
    n, d = X.shape
    w = np.zeros(d) if w0 is None else w0.copy()
    b = b0
    if balanced and 0 < y.mean() < 1:
        p1 = y.mean(); sw = np.where(y == 1, 0.5 / p1, 0.5 / (1 - p1))
        sw = sw / sw.mean()
    else:
        sw = np.ones(n)
    for _ in range(epochs):
        p = _sigmoid(X @ w + b)
        gw = X.T @ (sw * (p - y)) / n + l2 * w
        gb = np.mean(sw * (p - y))
        w -= lr * gw; b -= lr * gb
    return w, b


def _predict_away(home, w, b):
    return _sigmoid(home["X"] @ w + b) >= 0.5


def fit_method(method, homes_series):
    """Return {home_id: (w,b)} for a modality series under the given FL method.
    homes_series: list of home dicts (same series, common feature dim)."""
    d = homes_series[0]["X"].shape[1]
    if method == "local":
        return {h["home_id"]: _train_logreg(h["X"], h["y"]) for h in homes_series}
    if method == "centralized":
        X = np.vstack([h["X"] for h in homes_series]); y = np.concatenate([h["y"] for h in homes_series])
        w, b = _train_logreg(X, y)
        return {h["home_id"]: (w, b) for h in homes_series}
    if method == "fedavg":
        locals_ = [_train_logreg(h["X"], h["y"]) for h in homes_series]
        w = np.mean([l[0] for l in locals_], 0); b = float(np.mean([l[1] for l in locals_]))
        return {h["home_id"]: (w, b) for h in homes_series}
    if method == "modality_group":
        # share PIR weight (col 0) across ALL homes; share pressure weight (col 1) only among
        # pressure-having homes (real modality-group heterogeneity); bias stays local (config-specific)
        locals_ = {h["home_id"]: _train_logreg(h["X"], h["y"]) for h in homes_series}
        w_pir = np.mean([locals_[h["home_id"]][0][0] for h in homes_series])
        press_homes = [h for h in homes_series if h.get("has_pressure") and d >= 2]
        w_press = np.mean([locals_[h["home_id"]][0][1] for h in press_homes]) if press_homes else 0.0
        out = {}
        for h in homes_series:
            w = locals_[h["home_id"]][0].copy()
            w[0] = w_pir
            if d >= 2:
                w[1] = w_press if h.get("has_pressure") else 0.0
            out[h["home_id"]] = (w, locals_[h["home_id"]][1])
        return out
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Evaluation: obs-layer (F1, timing error) + downstream (longlie injection)
# --------------------------------------------------------------------------- #
def events_from_away(pred_away, steps):
    """Runs of predicted AWAY -> (exit_s, return_s) events (seconds)."""
    ev = []
    in_run = False; start = 0
    for i, a in enumerate(pred_away):
        if a and not in_run:
            in_run = True; start = steps[i]
        elif not a and in_run:
            in_run = False; ev.append((int(start), int(steps[i])))
    if in_run:
        ev.append((int(start), int(steps[-1])))
    return ev


def match_events(pred_ev, true_ev, tol_s=120.0):
    """Greedy 1:1 match by exit time. Returns per-true (matched?, exit_dt_s)."""
    used = [False] * len(pred_ev)
    per_true = []
    for te, tr in true_ev:
        best, bi = tol_s + 1, -1
        for j, (pe, pr) in enumerate(pred_ev):
            if used[j]:
                continue
            de = abs(pe - te)
            if de < best:
                best, bi = de, j
        if bi >= 0 and best <= tol_s:
            used[bi] = True
            per_true.append((True, abs(pred_ev[bi][0] - te)))
        else:
            per_true.append((False, None))
    recall = sum(1 for m, _ in per_true if m) / max(len(true_ev), 1)
    return recall, per_true


def obs_f1(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


def _episodes_df(events, seed):
    """Build a longlie-compatible midnight-B2T episodes DataFrame from (exit_s,return_s)."""
    base = pd.Timestamp("2020-01-01 02:00:00")
    rows = []
    for k, (e, r) in enumerate(events):
        rows.append(dict(start=base + pd.Timedelta(days=k, seconds=0),
                         return_min=max((r - e) / 60.0, 0.1), context="midnight",
                         censored=False, source="b2t"))
    return pd.DataFrame(rows)


def downstream_delay_min(events, alpha=0.10):
    """Detection delay t*(alpha) via longlie injection (import-only, iron rules intact)."""
    df = _episodes_df(events, 0)
    if len(df) < 8:
        return float("nan")
    tr, ev, _ = ll_injection.calendar_split(df, 0.6)
    cur = ll_injection.delay_fa_curve(tr, ev, [alpha], "personal", None, 21.0,
                                      inject_frac=0.3, seed=2031, B=100)
    return float(cur["delay_median"].iloc[0]) if len(cur) else float("nan")


CAP_MIN = 120.0   # a missed long-lie is discovered next morning (capped); documented assumption


def evaluate(home, w, b):
    """Two-layer eval. Downstream cost holds t* fixed (longlie's design threshold) and
    charges ONLY observation error: detected long-lie -> exit lateness; missed -> morning cap.
    Oracle cost = 0 by construction, so cost_min IS the oracle-ratio delay cost (min)."""
    pred = _predict_away(home, w, b).astype(int)
    f1 = obs_f1(home["y"], pred)
    pred_ev = events_from_away(pred == 1, home["steps"])
    recall, per_true = match_events(pred_ev, home["true_events"])
    dt_exit = [dt for m, dt in per_true if m]
    med_dt_exit_min = float(np.median(dt_exit)) / 60.0 if dt_exit else np.nan
    # oracle detection delay t* = longlie personal quantile of TRUE return times (the design floor)
    t_star = downstream_delay_min(home["true_events"])
    if not np.isfinite(t_star):
        t_star = 5.0
    # per injected long-lie (each true exit treated as a long-lie onset): oracle-ratio cost
    costs = []
    for (matched, dt), (te, tr) in zip(per_true, home["true_events"]):
        costs.append((dt / 60.0) if matched else (CAP_MIN - t_star))  # lateness | miss cap
    cost = float(np.median(costs)) if costs else np.nan
    return dict(f1=f1, timing_err_min=med_dt_exit_min, recall=recall,
                delay_oracle=t_star, delay_obs=t_star + cost, cost_min=cost)
