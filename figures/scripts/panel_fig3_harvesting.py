"""
panel_fig3_harvesting.py — Figure 3, Panel C: humor is near-continuous, laughter is not.

Lineage: the left half of the old panel B (panel_fig3_harvesting.py, still live)
made this point with two bars — 66% vs 21% — parsed out of humor_summary.txt.
The point is worth keeping: the LLM content analysis says Friends is funny almost
all the time, while the laugh track fires on only a fraction of it. Two bars stated
that; this shows it, per episode, with the spread.

DENOMINATOR — MATCHES THE PUBLISHED NUMBERS, WHICH ARE MISLABELLED. The Results,
the supplement and humor_summary.txt all say "~132,000 speech-bearing TRs, 66.2%
humor-positive". Those 132,131 TRs are in fact EVERY TR of the 280 Clf-C-matched
segments, not the speech-bearing ones: only 99,058 (75.0%) carry speech, and the
humor rate among non-speech TRs is exactly 0.0% by construction (no dialogue, no
label). Restricted to speech-bearing TRs the figures are 88.3 / 13.2 / 11.9%.
This panel uses the ALL-TR denominator so it reproduces the published 66.2 / 21.4 /
8.9, and the axis is labelled "% of TRs" rather than repeating the wrong wording.
The 66-vs-21 CONTRAST is unaffected — both shares use the same denominator — so
only the wording needs correcting in the text.

WHAT IS PLOTTED. For every episode-segment, across all its TRs:
  - the share classified humor-positive by the LLM (Juckel typology)
  - the share marked as laughter by the PRIMARY Clf-C classifier
Points are segments; the violin is their distribution; the black bar is the corpus
value quoted in the Results (66.2 / 21.4%).

WHY THE OVERLAP BAR MUST NOT BE READ AS SUPPRESSION — the caveat that ships with
humor_summary.txt: the laugh track fills the pauses AFTER a joke, so a laughter TR
is usually NOT a speech TR, which mechanically lowers same-TR overlap. Confirmed
2026-08-06 by a peri-onset analysis: aligning 16,242 laughter onsets, P(speech)
drops from 0.77 to 0.57 exactly at onset while humor-given-speech stays flat at
~0.89 across the whole +/-30 s window. There is no measurable humor BUILD-UP before
laughter in these labels — humor is simply saturated (~89% of speech TRs), so the
annotation has no headroom to show accumulation. The temporal humor->laughter
relation is carried by the HRF-shifted fMRI analyses, not by this content
comparison.

  input   : data/b_laughter/humor_classification/aggregate_humor_by_tr.csv
            data/0_prep/laughter_annotations/*.csv   (Clf-C, PRIMARY)
  output  : figures/panels/fig3_panelC_harvesting.png
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
from config import HUMOR_CLASSIFICATION_DIR, PREP_LAUGHTER_ANN_DIR

OUT = HERE.parent / "panels" / "fig3_panelC_harvesting.png"

# ---------------------------------------------------------------- CONFIG ----
# The same-TR OVERLAP (8.9%) is deliberately NOT plotted (author decision,
# 2026-08-06). The panel makes one claim — humor is near-continuous, laughter is
# not — and a third bar invited the reading that humor SUPPRESSES laughter, which
# is exactly the misreading the caveat below warns against. The 8.9% stays in the
# Results text, where it can be qualified in words.
# ONE grey, like Fig 3B and 3D: the two categories are already separated by the
# x-axis, so colour would encode nothing. It would also MISLEAD — the previous
# #DD8452 / #55A868 came from the old season palette and sit close to the ROI
# colours used in Fig 1D / 2C (auditory #D35400, rTPJ #27AE60), so a reader moving
# between figures could read the orange bar as "auditory cortex".
COLOR = "#4D4D4D"
BARS  = [("humor",   "Humor\n(LLM content)",   COLOR),
         ("laugh",   "Laughter\n(laugh track)", COLOR)]
DOT_S, JITTER = 8, 0.11
# SHARED PANEL GEOMETRY — see the other Figure 3 panels; identical canvas, saved
# WITHOUT bbox_inches="tight" so all four scale evenly in a 2x2.
FIGSIZE = (6.6, 4.9)
# -----------------------------------------------------------------------------

h = pd.read_csv(HUMOR_CLASSIFICATION_DIR / "aggregate_humor_by_tr.csv", low_memory=False)
h["is_humor"] = h["is_humor"].astype(str).isin(["1", "True", "true"]).astype(int)
h["has_speech"] = h["has_speech"].astype(bool)

rows = []
for ep, g in h.groupby("episode"):
    f = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
    if not f.exists():
        continue                                   # keep the Clf-C-matched set
    ls = pd.read_csv(f)["ls"].values
    g = g.sort_values("tr_idx")
    n = min(len(ls), len(g))
    hu = g["is_humor"].values[:n].astype(bool)
    la = ls[:n].astype(bool)
    if n == 0:
        continue
    rows.append({"episode": ep,
                 "humor": 100 * hu.mean(),
                 "laugh": 100 * la.mean(),
                 "both": 100 * (hu & la).mean(),
                 "n_tr": int(n)})
per_ep = pd.DataFrame(rows)

# corpus totals = pooled over TRs (what the Results quote), not the mean of episodes
tot = {}
pooled = per_ep[["humor", "laugh", "both"]].mul(per_ep.n_tr, axis=0).sum() / per_ep.n_tr.sum()
for k in ("humor", "laugh", "both"):
    tot[k] = pooled[k]

fig, ax = plt.subplots(figsize=FIGSIZE)
rng = np.random.default_rng(0)
for i, (key, label, colour) in enumerate(BARS):
    v = per_ep[key].values
    for body in ax.violinplot(v, positions=[i], widths=0.72, showextrema=False)["bodies"]:
        body.set_facecolor(colour); body.set_edgecolor("none"); body.set_alpha(0.30)
    ax.scatter(i + rng.uniform(-JITTER, JITTER, v.size), v, s=DOT_S, color=colour,
               alpha=0.5, edgecolors="none", zorder=3)
    ax.plot([i - 0.26, i + 0.26], [tot[key]] * 2, color="black", lw=2.0, zorder=4)
    ax.annotate(f"{tot[key]:.1f}%", (i, tot[key]), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(range(len(BARS))); ax.set_xticklabels([b[1] for b in BARS])
ax.set_ylim(0, 100)
ax.set_ylabel("% of TRs")
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300); plt.close(fig)

print(f"generated  {OUT.name}   ({len(per_ep)} segments, {per_ep.n_tr.sum():,} TRs)")
for key, label, _ in BARS:
    print(f"   {key:6s} corpus {tot[key]:5.1f}%   per-episode {per_ep[key].mean():5.1f}% "
          f"(sd {per_ep[key].std():.1f}, range {per_ep[key].min():.1f}-{per_ep[key].max():.1f})")
