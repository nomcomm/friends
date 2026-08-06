"""
panel_fig3_humor_composition.py — Figure 3, Panel D candidate: humor content.

Restyled version of the right half of the old panel B (three bars: Language /
Logic / Identity). Kept deliberately plain — it reports three numbers and should
not pretend to be more.

WHY IT IS NOT SPLIT BY SEASON (checked 2026-08-06). Composition barely moves
across the run: Language 31.5 -> 25.9%, Logic 59.1 -> 64.8%, Identity 9.4 -> 9.3%.
chi2(10) = 520 is p = 2e-105 only because n = 91,144 humor TRs; Cramer's V = 0.053,
i.e. negligible. Decisively, the season-to-season RANGE (9.0 / 7.3 / 2.2 points) is
SMALLER than the per-EPISODE sd (9.8 / 10.5 / 6.3 points) — episodes vary more than
seasons do. A six-season version would be six near-identical stacked bars.

The per-episode spread IS shown here (dots + violin), because that is the variation
that actually exists.

SCHEMA CAVEAT. The three categories cover 99.95% of humor-positive TRs; 50 TRs
(0.055%) came back with an off-schema label (Irony 39, Parody 7, Deceitful_behavior
4) — techniques in the Juckel typology rather than categories. Reported as-is, not
reassigned; see verify_claims.py block C6.

  input   : data/b_laughter/humor_classification/aggregate_humor_by_tr.csv
  output  : figures/panels/fig3_panelD_humor_composition.png
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); import figstyle; figstyle.apply()
sys.path.insert(0, str(HERE.parents[1]))
from config import HUMOR_CLASSIFICATION_DIR

OUT = HERE.parent / "panels" / "fig3_panelD_humor_composition.png"

# ---------------------------------------------------------------- CONFIG ----
CATS    = ["Logic", "Language", "Identity"]        # plotted in rank order
LABELS  = {"Logic": "Logic\n(inference)", "Language": "Language\n(wordplay)",
           "Identity": "Identity\n(character)"}
COLOR   = "#4D4D4D"                                # one colour: category is the x-axis
DOT_S, JITTER = 9, 0.11
# SHARED PANEL GEOMETRY — all Figure 3 panels use this exact canvas and save
# WITHOUT bbox_inches="tight", so every panel comes out at the same aspect
# (1.35, matching a 2x2 quadrant). Cropping to content would make the saved
# aspect depend on labels, and the panels would scale unevenly when composed.
FIGSIZE = (6.6, 4.9)
# -----------------------------------------------------------------------------

d = pd.read_csv(HUMOR_CLASSIFICATION_DIR / "aggregate_humor_by_tr.csv", low_memory=False)
d["is_humor"] = d["is_humor"].astype(str).isin(["1", "True", "true"]).astype(int)
h = d[d.is_humor == 1]

overall = (h.primary_category.value_counts(normalize=True) * 100)
per_ep = pd.crosstab(h.episode, h.primary_category, normalize="index") * 100

fig, ax = plt.subplots(figsize=FIGSIZE)
rng = np.random.default_rng(0)
for i, c in enumerate(CATS):
    v = per_ep[c].values
    for body in ax.violinplot(v, positions=[i], widths=0.7, showextrema=False)["bodies"]:
        body.set_facecolor(COLOR); body.set_edgecolor("none"); body.set_alpha(0.25)
    ax.scatter(i + rng.uniform(-JITTER, JITTER, v.size), v, s=DOT_S, color=COLOR,
               alpha=0.45, edgecolors="none", zorder=3)
    ax.plot([i - 0.24, i + 0.24], [overall[c]] * 2, color="black", lw=1.8, zorder=4)
    ax.annotate(f"{overall[c]:.0f}%", (i, overall[c]), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(CATS))); ax.set_xticklabels([LABELS[c] for c in CATS])
ax.set_ylim(0, 100)
ax.set_ylabel("% of humor TRs")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300); plt.close(fig)

print(f"generated  {OUT.name}   ({len(h):,} humor TRs, {per_ep.shape[0]} episodes)")
for c in CATS:
    print(f"   {c:9s} {overall[c]:5.1f}%   per-episode sd {per_ep[c].std():.1f} pts")
