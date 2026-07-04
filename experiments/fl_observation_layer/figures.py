"""figures.py — Phase C flagship figure + sub-figure + (d) Spearman decision.

Decision rule for (d) is FROZEN before drawing (PI): main x-axis = whichever of
{timing error, F1} has the larger |Spearman| with downstream cost; the other -> appendix.
Both coefficients written to correlations.txt (and REPORT.md)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GRAY, GREEN = "#1f4e9c", "#d9700f", "#8a8a8a", "#2a8a3a"
plt.rcParams.update({"font.size": 8, "figure.dpi": 150, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.3})

MARK = {"local": "o", "centralized": "s", "fedavg": "^", "modality_group": "D"}
SERIES_COL = {"pir_only": BLUE, "pir_pressure": ORANGE, "lidar_upper": GRAY}


def correlations(df):
    d = df[df.series != "lidar_upper"].dropna(subset=["cost_min", "timing_err_min", "f1"])
    s_t, _ = spearmanr(d.timing_err_min, d.cost_min)
    s_f, _ = spearmanr(d.f1, d.cost_min)
    main = "timing_err_min" if abs(s_t) >= abs(s_f) else "f1"
    with open(os.path.join(HERE, "correlations.txt"), "w") as fh:
        fh.write(f"Spearman(timing_err, cost) = {s_t:.3f}\n")
        fh.write(f"Spearman(F1, cost)         = {s_f:.3f}\n")
        fh.write(f"main x-axis (larger |rho|) = {main}\n")
    return s_t, s_f, main


def fig_flagship(df, main):
    lidar = df[df.series == "lidar_upper"]["cost_min"].median()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for series in ["pir_only", "pir_pressure"]:
        for method in ["local", "centralized", "fedavg", "modality_group"]:
            sub = df[(df.series == series) & (df.method == method)]
            if not len(sub):
                continue
            x = sub[main]; y = sub["cost_min"]
            ax.scatter(x.median(), y.median(), s=42, marker=MARK[method],
                       color=SERIES_COL[series], edgecolor="k", linewidth=0.4, zorder=3)
    ax.axhline(lidar, color=GRAY, ls="--", lw=1.1)
    ax.text(0.98, lidar, f"LiDAR upper-bound (obs ceiling): cost≈{lidar:.1f}min\n= quantile limit floor",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6, color=GRAY)
    xlab = "exit timing error (min)" if main == "timing_err_min" else "observation F1"
    ax.set_xlabel(xlab); ax.set_ylabel("oracle-ratio detection-delay cost (min), α=0.1")
    # legends
    from matplotlib.lines import Line2D
    sh = [Line2D([0], [0], marker=MARK[m], color="w", markerfacecolor="gray",
                 markeredgecolor="k", label=m, ms=7) for m in MARK]
    cl = [Line2D([0], [0], marker="o", color="w", markerfacecolor=SERIES_COL[s],
                 label=s, ms=7) for s in ["pir_only", "pir_pressure"]]
    ax.legend(handles=sh + cl, fontsize=6, loc="center right", ncol=1)
    ax.set_title("Observation-error cost vs. observation quality")
    fig.savefig(os.path.join(HERE, "fig_flagship.pdf")); fig.savefig(os.path.join(HERE, "fig_flagship.png"), dpi=200)
    plt.close(fig)


def fig_subfigure(cov):
    counts = sorted(cov["count"].unique()); refrs = sorted(cov["refr"].unique())
    pir = cov[cov.kind == "pir"]
    M = np.array([[pir[(pir["count"] == c) & (pir["refr"] == r)]["recall_sub1min"].iloc[0]
                   for r in refrs] for c in counts])
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
                   extent=[min(refrs), max(refrs), min(counts) - 0.5, max(counts) + 0.5])
    cs = ax.contour(refrs, counts, M, levels=[0.8], colors="w", linewidths=1.5)
    ax.clabel(cs, fmt="0.8 coverage", fontsize=6)
    pp = cov[cov.kind == "pir1_pressure"].iloc[0]
    ax.scatter([pp["refr"]], [pp["count"]], marker="*", s=260, color=ORANGE, edgecolor="k", zorder=5)
    ax.annotate(f"PIR×1 + pressure (KIND):\nrecall={pp['recall_sub1min']:.2f} at count=1\n"
                f"(number alone: 0 at count=1)", (pp["refr"], pp["count"]),
                textcoords="offset points", xytext=(14, 18), fontsize=6.5, color="k",
                bbox=dict(boxstyle="round", fc="white", ec=ORANGE, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1))
    ax.set_xlabel("refractory_s"); ax.set_ylabel("bedroom_sensor_count (PIR)")
    ax.set_title("Sub-minute B2T coverage: number vs. kind")
    fig.colorbar(im, ax=ax, label="recall (<1min B2T)")
    fig.savefig(os.path.join(HERE, "fig_subfigure.pdf")); fig.savefig(os.path.join(HERE, "fig_subfigure.png"), dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(HERE, "results.csv"))
    cov = pd.read_csv(os.path.join(HERE, "coverage_grid.csv"))
    s_t, s_f, main = correlations(df)
    print(f"Spearman timing={s_t:.3f} F1={s_f:.3f} -> main axis = {main}")
    fig_flagship(df, main)
    fig_subfigure(cov)
    print("wrote fig_flagship.{pdf,png}, fig_subfigure.{pdf,png}, correlations.txt")


if __name__ == "__main__":
    main()
