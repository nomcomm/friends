"""
season_consistency.py — rTPJ laughter effect by season (Block B, B3)
====================================================================
Ported from 03_REVISION/scripts/14_season_consistency.py. Shows the rTPJ laughter
vs non-laughter effect broken down by season, to demonstrate robustness across the
six-season longitudinal span (the appropriate inference given N=4 viewers:
consistency across stimuli, not population generalization).

Because laughter is a discrete event, the GLM is the more fitting model, so both
views are shown: the ISC synchrony contrast (faithful B3 port) and the GLM
activation contrast (complementary, stronger).

  inputs  : data/b_laughter/{isc_laugh_by_parcel.npy, isc_nolaugh_by_parcel.npy,
            episodes.csv}          (from laughter_isc.py)
            data/b_laughter/{glm_beta_by_episode.npy, glm_episodes.csv}  (glm_contrast.py)
  outputs : data/b_laughter/season_consistency.csv
            results/analysis_plots/b_laughter/fig_season_consistency.png        (ISC + GLM by season)
            results/analysis_plots/b_laughter/exploratory/fig_season_scatter.png (per-episode scatter)

Usage
  python season_consistency.py            # skip if season_consistency.csv exists
  python season_consistency.py --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from nltools.stats import fisher_r_to_z, fisher_z_to_r

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import B_DIR, B_FIG_DIR, ROI_RTPJ_LAUGHTER

LAUGH_COL, NOLAUGH_COL, GLM_COL = "#27AE60", "#95A5A6", "#8E44AD"


def load_episodes():
    ep = pd.read_csv(B_DIR / "episodes.csv")
    lp = np.load(B_DIR / "isc_laugh_by_parcel.npy")[:, ROI_RTPJ_LAUGHTER]
    nl = np.load(B_DIR / "isc_nolaugh_by_parcel.npy")[:, ROI_RTPJ_LAUGHTER]
    df = pd.DataFrame({"episode": ep["episode"], "season": ep["season"],
                       "isc_laugh": lp, "isc_nolaugh": nl})
    gep = pd.read_csv(B_DIR / "glm_episodes.csv")
    gb = np.load(B_DIR / "glm_beta_by_episode.npy")[:, ROI_RTPJ_LAUGHTER]
    gdf = pd.DataFrame({"episode": gep["episode"], "glm_beta": gb})
    return df.merge(gdf, on="episode", how="left")


def season_table(df):
    rows = []
    for s in sorted(df["season"].unique()):
        sub = df[df["season"] == s]
        lz, nz = fisher_r_to_z(sub["isc_laugh"].values), fisher_r_to_z(sub["isc_nolaugh"].values)
        t_isc, p_isc = stats.ttest_rel(lz, nz)
        t_glm, p_glm = stats.ttest_1samp(sub["glm_beta"].dropna().values, 0)
        rows.append({"season": s, "n": len(sub),
                     "isc_laugh": float(fisher_z_to_r(lz.mean())),
                     "isc_nolaugh": float(fisher_z_to_r(nz.mean())),
                     "isc_delta": float(fisher_z_to_r((lz - nz).mean())),
                     "isc_t": float(t_isc), "isc_p": float(p_isc),
                     "glm_beta": float(sub["glm_beta"].mean()),
                     "glm_t": float(t_glm), "glm_p": float(p_glm),
                     "frac_isc_pos": float((sub["isc_laugh"] > sub["isc_nolaugh"]).mean())})
    return pd.DataFrame(rows)


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"


def fig_main(df, tab):
    seasons = tab["season"].values
    x = np.arange(len(seasons)); w = 0.35
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5)); fig.patch.set_facecolor("white")

    # A: ISC laugh vs non by season
    ml = tab["isc_laugh"].values; mnl = tab["isc_nolaugh"].values
    sl = [fisher_z_to_r(fisher_r_to_z(df[df.season == s].isc_laugh).sem() + 0) for s in seasons]  # approx
    axA.bar(x - w/2, ml, w, color=LAUGH_COL, alpha=.9, label="Laughter")
    axA.bar(x + w/2, mnl, w, color=NOLAUGH_COL, alpha=.9, label="Non-laughter")
    ytop = max(ml.max(), mnl.max()) + 0.02
    for i, row in tab.iterrows():
        axA.text(i, ytop, star(row.isc_p) if row.isc_p < .05 else f"p={row.isc_p:.2f}",
                 ha="center", fontsize=9, color=LAUGH_COL if row.isc_p < .05 else "gray",
                 fontweight="bold" if row.isc_p < .05 else "normal")
    axA.set_xticks(x); axA.set_xticklabels([f"S{s}\n(n={n})" for s, n in zip(seasons, tab["n"])])
    axA.set_ylabel("rTPJ ISC (pairwise mean r)"); axA.set_ylim(0, ytop + 0.02); axA.legend(fontsize=9)
    all_lz = fisher_r_to_z(df.isc_laugh.values); all_nz = fisher_r_to_z(df.isc_nolaugh.values)
    t, p = stats.ttest_rel(all_lz, all_nz)
    axA.set_title(f"A  rTPJ ISC (synchrony) by season   overall Δr={fisher_z_to_r((all_lz-all_nz).mean()):+.4f}, p={p:.3f}",
                  fontsize=10, fontweight="bold", loc="left")

    # B: GLM rTPJ beta by season
    axB.bar(x, tab["glm_beta"].values, 0.6, color=GLM_COL, alpha=.9)
    ytop2 = tab["glm_beta"].max() + 0.03
    for i, row in tab.iterrows():
        axB.text(i, row.glm_beta + 0.005, star(row.glm_p), ha="center", fontsize=9, fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels([f"S{s}" for s in seasons])
    axB.set_ylabel("rTPJ GLM laughter β (vs non-laughter)"); axB.axhline(0, color="k", lw=.6)
    tg, pg = stats.ttest_1samp(df["glm_beta"].dropna().values, 0)
    axB.set_title(f"B  rTPJ GLM (activation) by season   overall β={df['glm_beta'].mean():+.3f}, t={tg:.1f}",
                  fontsize=10, fontweight="bold", loc="left")

    fig.suptitle("rTPJ laughter effect across seasons — ISC synchrony (borderline) vs GLM activation (robust every season)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = B_FIG_DIR / "fig_season_consistency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  Saved {out.name}")


def fig_scatter(df, tab):
    (B_FIG_DIR / "exploratory").mkdir(parents=True, exist_ok=True)
    seasons = tab["season"].values
    lo = min(df.isc_laugh.min(), df.isc_nolaugh.min()) - .02
    hi = max(df.isc_laugh.max(), df.isc_nolaugh.max()) + .02
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for ax, s in zip(axes.flatten(), seasons):
        sub = df[df.season == s]
        ax.scatter(sub.isc_nolaugh, sub.isc_laugh, alpha=.45, s=25, color=LAUGH_COL)
        ax.plot([lo, hi], [lo, hi], "k--", lw=.8, alpha=.5)
        frac = (sub.isc_laugh > sub.isc_nolaugh).mean()
        ax.set_title(f"Season {s} (n={len(sub)})", fontsize=10)
        ax.text(.05, .93, f"{frac*100:.0f}% above diag", transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    fig.supxlabel("rTPJ ISC — non-laughter"); fig.supylabel("rTPJ ISC — laughter")
    fig.suptitle("Per-episode rTPJ ISC by season (above diagonal = higher during laughter)", fontsize=11)
    fig.tight_layout()
    out = B_FIG_DIR / "exploratory" / "fig_season_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved exploratory/{out.name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_csv = B_DIR / "season_consistency.csv"
    if out_csv.exists() and not args.force:
        print(f"Output exists (use --force): {out_csv}"); print(pd.read_csv(out_csv).to_string(index=False)); return

    df = load_episodes()
    tab = season_table(df)
    tab.to_csv(out_csv, index=False)
    print("rTPJ by season (ISC synchrony + GLM activation):")
    print(f"{'S':>2}{'n':>5}{'ISC Δr':>9}{'ISC p':>8}{'%pos':>6}   {'GLM β':>8}{'GLM t':>8}{'GLM p':>9}")
    for _, r in tab.iterrows():
        print(f"{int(r.season):>2}{int(r.n):>5}{r.isc_delta:>+9.4f}{r.isc_p:>8.3f}{r.frac_isc_pos*100:>5.0f}%   "
              f"{r.glm_beta:>+8.3f}{r.glm_t:>8.1f}{r.glm_p:>9.1e}")
    fig_main(df, tab); fig_scatter(df, tab)
    print(f"\nOutputs: {out_csv}  |  Figures: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
