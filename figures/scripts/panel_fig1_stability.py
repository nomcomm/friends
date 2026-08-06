"""
panel_fig1_stability.py — Figure 1, Panel C (clean, title-free).
  C1: two-episode ISC scatter (each point = one parcel), with r.
  C2: histogram of all pairwise spatial ISC correlations, with mean.
From the v3 ISC output. Titles live in the legend.

C1 EPISODE PAIR (changed 2026-08-03): FIRST vs LAST segment of the six-season
run, so the scatter instantiates the "entire longitudinal span" claim rather
than an arbitrary mid-run pair (was s01e02a vs s04e09a, r = 0.67).
  - The run starts at s01e02a, NOT s01e01a: s01e01a/b were annotated as
    classifier training material but never extracted under fMRIPrep, so they
    have no ISC map. s01e02a is the earliest segment that exists.
  - s06e24 is the season-6 finale and ran in FOUR segments; their correlations
    with s01e02a span 0.503-0.679, so the choice among them is not neutral.
    EP_Y is the true final segment (d).
  - The resulting r = 0.64 sits at the 32nd percentile of all 41,905 pairs
    (mean 0.659) — slightly below average, and deliberately not cherry-picked.
"""
from pathlib import Path
import sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent)); import figstyle; figstyle.apply()

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1] / "data" / "a_isc_stability" / "isc_fmriprep"
P = HERE.parent / "panels"

isc = np.load(V3 / "isc_all.npy")[:, :1000]
eps = pd.read_csv(V3 / "episodes.csv")["episode"].tolist()

# C1 — scatter of the FIRST vs the LAST segment of the run (see docstring)
EP_X, EP_Y = "s01e02a", "s06e24d"
i = eps.index(EP_X) if EP_X in eps else 0
j = eps.index(EP_Y) if EP_Y in eps else len(eps) - 1
r = np.corrcoef(isc[i], isc[j])[0, 1]
fig, ax = plt.subplots(figsize=(3.3, 3.2))
ax.scatter(isc[i], isc[j], s=6, color="0.35", alpha=0.45, edgecolors="none")
ax.set_xlabel(f"ISC — {eps[i]}"); ax.set_ylabel(f"ISC — {eps[j]}")
ax.annotate(f"r = {r:.2f}", (0.06, 0.9), xycoords="axes fraction", fontsize=11)
fig.tight_layout(); fig.savefig(P / "fig1_panelC1_stability_scatter.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# C2 — histogram of all pairwise spatial correlations
pairs = np.corrcoef(isc)[np.triu_indices(len(isc), k=1)]
fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.hist(pairs, bins=40, color="#4C72B0", alpha=0.85)
ax.axvline(pairs.mean(), color="k", ls="--", lw=1.2)
ax.set_xlabel("Spatial ISC correlation (r)"); ax.set_ylabel("Episode pairs")
ax.annotate(f"mean r = {pairs.mean():.2f}", (0.05, 0.92), xycoords="axes fraction", fontsize=10)
fig.tight_layout(); fig.savefig(P / "fig1_panelC2_stability_hist.png", dpi=300, bbox_inches="tight"); plt.close(fig)
from scipy import stats as _st
print(f"generated  fig1_panelC1/C2  ({eps[i]} vs {eps[j]}: r={r:.3f}, "
      f"{_st.percentileofscore(pairs, r):.0f}th pct of {len(pairs)} pairs; "
      f"mean pairwise r={pairs.mean():.3f})")
