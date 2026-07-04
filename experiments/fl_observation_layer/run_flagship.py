"""run_flagship.py — Phase C grid: 4 FL methods x 3 modality series x knob cells.
Outputs results.csv (per home) + a coverage grid for the sub-figure. Seeds fixed (2031)."""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import study as S

SEED = 2031
# knob cells (temporal-resolution knob): (bedroom_sensor_count, refractory_s, report_period_s)
KNOB_CELLS = [(1, 5, 0), (1, 5, 30), (1, 5, 60), (1, 30, 30), (2, 5, 0), (2, 5, 60)]


def build_homes(series, n_per_cell, n_nights, missing_support_frac=0.3):
    homes = []
    hid = 0
    rng = np.random.default_rng(SEED)
    for (count, refr, rep) in KNOB_CELLS:
        for _ in range(n_per_cell):
            # PIR+pressure series: some homes LACK pressure (missing-support heterogeneity, (1))
            has_pressure = True
            if series == "pir_pressure":
                has_pressure = rng.random() >= missing_support_frac
            homes.append(S.generate_home(f"{series}_{hid}", series, count, refr, rep,
                                         has_pressure, n_nights, seed=SEED + hid * 7))
            hid += 1
    return homes


def run(n_per_cell=3, n_nights=5):
    rows = []
    for series in S.SERIES:
        methods = S.METHODS if series != "lidar_upper" else ["local"]  # upper-bound: 1 series
        homes = build_homes(series, n_per_cell, n_nights)
        for method in methods:
            fitted = S.fit_method(method, homes)
            for h in homes:
                w, b = fitted[h["home_id"]]
                m = S.evaluate(h, w, b)
                rows.append(dict(series=series, method=method, home=h["home_id"],
                                 count=h["count"], refr=h["refr"], rep=h["rep"],
                                 has_pressure=h["has_pressure"], **m))
        print(f"  done series={series} ({len(homes)} homes x {len(methods)} methods)")
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results.csv")
    df.to_csv(out, index=False)
    print("wrote", out, "rows", len(df))
    return df


def run_coverage_grid(n_nights=5):
    """Sub-figure: recall of <1min B2T over (bedroom_sensor_count x refractory) + PIR1+pressure point."""
    from heterosense._core._pir_model import PIRModel
    from heterosense._core._extractor import b2t_recall
    counts = [1, 2, 3, 4]; refrs = [2, 5, 10, 30, 60]
    rng = np.random.default_rng(SEED)
    motion, truth, off = [], [], 0
    for _ in range(20):
        m, tr = S.generate_b2t_night(rng, S.SNAP, n_exits=6, night_seconds=S.NIGHT_S)
        motion.append(m); truth += [(a + off, b + off) for a, b in tr]; off += S.NIGHT_S
    motion = np.concatenate(motion)
    sub = [(e, r) for (e, r) in truth if (r - e) < 60]   # sub-minute trips only
    grid = []
    for c in counts:
        for rf in refrs:
            pm = PIRModel(S._Cfg(c, rf, 0), rng=np.random.default_rng(7))
            ev = pm.observe_sequence(motion, delta_t=1.0)
            rec = b2t_recall(ev, sub, report_period_s=0.0, base_gap_s=300.0, sensor_count=c)
            grid.append(dict(count=c, refr=rf, recall_sub1min=rec, kind="pir"))
    # PIR x1 + pressure operating point: pressure confirms absence -> sensor_count>=2 branch
    pm = PIRModel(S._Cfg(1, 5, 0), rng=np.random.default_rng(7))
    ev = pm.observe_sequence(motion, delta_t=1.0)
    rec_pp = b2t_recall(ev, sub, report_period_s=0.0, base_gap_s=300.0, sensor_count=2)
    grid.append(dict(count=1, refr=5, recall_sub1min=rec_pp, kind="pir1_pressure"))
    df = pd.DataFrame(grid)
    df.to_csv(os.path.join(HERE, "coverage_grid.csv"), index=False)
    print("wrote coverage_grid.csv; PIR1+pressure sub-min recall =", round(rec_pp, 3))
    return df


def run_missing_support(n_per_cell=3, n_nights=6):
    """(1) IEICE missing-support check: vary the fraction of homes lacking the pressure
    (rare-state support) modality; measure method 3 (FedAvg) vs 4 (modality-group) bias
    on the PRESSURE-having homes (where naive averaging over support-lacking homes hurts)."""
    rows = []
    for frac in [0.0, 0.3, 0.5, 0.7]:
        homes = build_homes("pir_pressure", n_per_cell, n_nights, missing_support_frac=frac)
        press_homes = [h for h in homes if h["has_pressure"]]
        for method in ["fedavg", "modality_group"]:
            fitted = S.fit_method(method, homes)
            costs = [S.evaluate(h, *fitted[h["home_id"]])["cost_min"] for h in press_homes]
            rows.append(dict(missing_frac=frac, method=method, n_pressure_homes=len(press_homes),
                             cost_pressure_homes=float(np.median(costs))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "missing_support.csv"), index=False)
    piv = df.pivot(index="missing_frac", columns="method", values="cost_pressure_homes")
    piv["method4_advantage"] = piv["fedavg"] - piv["modality_group"]
    print("missing-support sweep (cost on pressure homes; method4_advantage>0 = method4 wins):")
    print(piv.round(3))
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_cell", type=int, default=3)
    ap.add_argument("--n_nights", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n_per_cell, a.n_nights = 1, 2
    run(a.n_per_cell, a.n_nights)
    run_coverage_grid(a.n_nights)
    run_missing_support(a.n_per_cell, a.n_nights)
