"""
panel_fig3_laughs_per_min.py — Figure 3, Panel B: laughter rate per episode.

NEW 2026-08-06. Replaces the six-bar "laughs/minute by season" chart that sat in
the right half of the old panel A. That chart had one bar per season, no error
bars, and a different colour per season — colour encoding nothing, since season
was already the x-axis (same fix as Fig 1D and Fig 2C).

WHAT THIS SHOWS. Every episode's laughter rate as a point, with the per-season
distribution behind it. One colour throughout.

WHY THE DISTRIBUTION MATTERS HERE. Within-season spread (sd ~0.48 laughs/min) is
comparable to the whole between-season range (0.90), so the bar chart's six tidy
bars implied a precision the data do not have.

AND THE TREND IS REAL. Laughter rate rises across the run: S1 5.10 -> S6 5.92
(+16%), r(season, rate) = +0.487, p = 1.6e-09 across the 137 episodes. The paper
describes the laugh track as "frequent (~5-6 laughs per minute)", which is true
but flat-sounding; later seasons are measurably denser. Worth knowing alongside
the Figure 2 finding that laughter betas also grow across seasons — more laughter
AND stronger responses later in the run.

  input   : data/0_prep/laughter_annotations/*.csv   (280 segments, per-TR ls)
  output  : figures/panels/fig3_panelB_laughs_per_min.png
"""
from pathlib import Path
import collections
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import figstyle; figstyle.apply()
sys.path.insert(0, str(HERE.parents[1]))
from config import PREP_LAUGHTER_ANN_DIR, TR_SEC

OUT = HERE.parent / "panels" / "fig3_panelB_laughs_per_min.png"

# ---------------------------------------------------------------- CONFIG ----
COLOR   = "#4D4D4D"    # ONE colour — season is the x-axis
YLIM    = (0, 7)       # zero-based (author decision, 2026-08-06). Data span only
                       # 3.99-6.86, so this compresses the violins; the trade is
                       # deliberate — a zero baseline keeps the RATE readable as a
                       # magnitude rather than exaggerating between-season gaps.
DOT_S   = 14
JITTER  = 0.13
# SHARED PANEL GEOMETRY — all Figure 3 panels use this exact canvas and save
# WITHOUT bbox_inches="tight", so every panel comes out at the same aspect
# (1.35, matching a 2x2 quadrant). Cropping to content would make the saved
# aspect depend on labels, and the panels would scale unevenly when composed.
FIGSIZE = (6.6, 4.9)
# -----------------------------------------------------------------------------

seg = {f.stem: pd.read_csv(f)["ls"].values
       for f in sorted(PREP_LAUGHTER_ANN_DIR.glob("*.csv"))}
parts = collections.defaultdict(list)
for k in sorted(seg):
    parts[k[:6]].append(k)

rate, season = {}, {}
for e, v in parts.items():
    x = np.concatenate([seg[k] for k in sorted(v)])
    d = np.diff(np.concatenate([[0], x, [0]]))
    rate[e] = int((d == 1).sum()) / (len(x) * TR_SEC / 60)
    season[e] = int(e[1:3])

r = pd.Series(rate); s = pd.Series(season)
sea_ids = sorted(set(s))

fig, ax = plt.subplots(figsize=FIGSIZE)
rng = np.random.default_rng(0)                 # fixed jitter -> reproducible panel

for k in sea_ids:
    v = r[s == k].values
    for body in ax.violinplot(v, positions=[k], widths=0.75,
                              showextrema=False)["bodies"]:
        body.set_facecolor(COLOR); body.set_edgecolor("none"); body.set_alpha(0.28)
    ax.scatter(k + rng.uniform(-JITTER, JITTER, v.size), v,
               s=DOT_S, color=COLOR, alpha=0.65, edgecolors="none", zorder=3)
    ax.plot([k - 0.24, k + 0.24], [v.mean()] * 2, color="black", lw=1.6, zorder=4)
    ax.annotate(f"{v.mean():.2f}", (k, v.mean()), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=8)

ax.set_xticks(sea_ids); ax.set_xticklabels([f"S{k}" for k in sea_ids])
ax.set_ylim(*YLIM)
ax.set_xlabel("Season"); ax.set_ylabel("Laughs per minute")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300); plt.close(fig)

from scipy import stats
rr, pp = stats.pearsonr(s.values, r.values)
print(f"generated  {OUT.name}  ({len(r)} episodes)")
print(f"   overall {r.mean():.2f} laughs/min (sd {r.std():.2f}, range {r.min():.2f}-{r.max():.2f})")
print("   season means: " + "  ".join(f"S{k}={r[s == k].mean():.2f}" for k in sea_ids))
print(f"   season trend r={rr:+.3f}  p={pp:.1e}")
