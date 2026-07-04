"""make_b2t_snapshot.py — freeze the 61-home empirical midnight B2T return-time
distribution into a self-contained, provenance-stamped snapshot for HeteroSense-FL v2.

The snapshot calibrates v2.0's minimal B2T away-duration sampler (anchor: a single
bedroom PIR misses sub-minute B2T trips). It is committed to this repo so the simulator
is reproducible WITHOUT longlie_study present. Regenerate (only as an explicit,
CHANGELOG-documented operation) with:

    python tools/make_b2t_snapshot.py --longlie /path/to/longlie_study

Provenance (longlie_study commit, source Zenodo DOI, date, CC-BY-4.0 attribution) is
embedded. The anchor test pins `distribution_sha256` (data-only hash, stable across
regenerations from the same longlie_study commit).
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, subprocess, datetime as dt

DEFAULT_LONGLIE = "/Users/x-shao/Projects/teraoka/longlie_study"
N_GRID = 1001  # inverse-CDF grid points (probabilities 0..1)
ROUND = 6      # decimals for hash-stable serialization
SOURCE_DOI = "10.5281/zenodo.15708568"  # CASAS labeled data (CC-BY-4.0)


def extract_pooled_b2t(longlie: str):
    import numpy as np, pandas as pd
    sys.path.insert(0, os.path.join(longlie, "src"))
    import loader, episodes_label as el  # from longlie_study, read-only
    roster = pd.read_csv(os.path.join(longlie, "results", "home_roster.csv"))
    homes = roster[roster.residents.astype(str) == "1"]["home"].tolist()
    pooled, per_home = [], {}
    for h in homes:
        p = os.path.join(longlie, "data", "labeled_data", "labeled", f"{h}.csv")
        if not os.path.exists(p):
            continue
        df, _ = loader.load_home(p)
        e = el.episodes_dataframe(el.extract_episodes(df, h))
        v = e[(~e.censored) & e.return_min.notna()
              & (e.context == "midnight") & (e.source == "b2t")]["return_min"].to_numpy()
        if v.size:
            pooled.append(v); per_home[h] = int(v.size)
    x = np.concatenate(pooled)
    return x, per_home


def build_snapshot(x, per_home, longlie_commit, gen_date):
    import numpy as np
    probs = np.linspace(0.0, 1.0, N_GRID)
    grid = np.round(np.quantile(x, probs), ROUND).tolist()  # inverse-CDF values (minutes)
    # data-only hash: quantile grid + n (excludes volatile provenance/date)
    canon = json.dumps({"grid": grid, "n": int(x.size)}, sort_keys=True, separators=(",", ":"))
    dist_hash = hashlib.sha256(canon.encode()).hexdigest()
    snap = {
        "schema": "heterosense-b2t-snapshot/1",
        "provenance": {
            "description": "Pooled midnight bed-to-toilet round-trip return times, "
                           "single-resident labeled homes.",
            "generated_by": "tools/make_b2t_snapshot.py",
            "generated_date": gen_date,
            "longlie_study_commit": longlie_commit,
            "source_data_doi": SOURCE_DOI,
            "source_data_license": "CC-BY-4.0",
            "attribution": "Derived from CASAS Smart Home dataset (Cook, Diane; "
                           "Zenodo " + SOURCE_DOI + ", CC-BY-4.0) via longlie_study.",
            "n_homes": len(per_home),
        },
        "distribution": {
            "kind": "inverse_cdf", "unit": "minutes", "n": int(x.size),
            "probabilities": np.round(np.linspace(0.0, 1.0, N_GRID), ROUND).tolist(),
            "values": grid,
        },
        "summary": {
            "median_min": round(float(np.median(x)), 4),
            "mean_min": round(float(x.mean()), 4),
            "q10_min": round(float(np.quantile(x, 0.1)), 4),
            "q90_min": round(float(np.quantile(x, 0.9)), 4),
            "frac_lt_1min": round(float(np.mean(x < 1.0)), 4),
            "frac_lt_0p5min": round(float(np.mean(x < 0.5)), 4),
        },
        "distribution_sha256": dist_hash,
    }
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longlie", default=DEFAULT_LONGLIE)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "heterosense", "_data", "b2t_snapshot.json"))
    ap.add_argument("--date", default=None, help="override generation date (YYYY-MM-DD)")
    args = ap.parse_args()
    if not os.path.isdir(args.longlie):
        sys.exit(f"longlie_study not found at {args.longlie}; pass --longlie PATH")
    try:
        commit = subprocess.check_output(["git", "-C", args.longlie, "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = "unknown"
    gen_date = args.date or dt.date.today().isoformat()
    x, per_home = extract_pooled_b2t(args.longlie)
    snap = build_snapshot(x, per_home, commit, gen_date)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")
    print(f"  n={snap['distribution']['n']} homes={snap['provenance']['n_homes']} "
          f"median={snap['summary']['median_min']}min frac<1min={snap['summary']['frac_lt_1min']}")
    print(f"  longlie_commit={commit[:7]} source_doi={SOURCE_DOI}")
    print(f"  distribution_sha256={snap['distribution_sha256']}")


if __name__ == "__main__":
    main()
