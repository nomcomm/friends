"""
panel_fig2_season_glm.py — Figure 2, Panel C.

Lineage: supersedes the six-bar rTPJ-by-season chart, archived verbatim (with its
panel) at figures/_old/panel_fig2_season_glm_v1.py. That panel showed one ROI, aggregated to seasons, with no error bars and a
"***" on every bar — since all six seasons are significant the stars carried no
information, and nothing on it identified the region except the y-label.

WHAT THIS SHOWS. Per-episode laughter beta for the five a-priori regions, every
episode s01 -> s06 in broadcast order. Season boundaries are marked, but no summary
statistic is drawn: per-season means were tried as thick horizontal segments and
removed (author decision, 2026-08-06) — they cluttered the traces. The means are
still printed on run, and the season trend below is documented here so it is not
lost with the markers.

WHY NO rTPJ-ONLY SUMMARY PANEL (author decision, 2026-08-06). An earlier draft had
a per-season rTPJ violin plot stacked on top. It was dropped because the activation
is broadly distributed — 961/1032 regions are FDR-significant and LOC's beta (+0.48)
exceeds rTPJ's (+0.28) — so heading the panel with an rTPJ-only summary overstated
that region's specialness.

THE SEASON TREND IS REAL, NOT FLAT — recorded here because nothing in the panel
now shows it.
Regressing beta on season across the 278 segments:
    rTPJ        +0.20 -> +0.36   r=+0.224  p=1.7e-04   (S1 vs S5-6: +81%, p=0.0008)
    LOC/visual  +0.31 -> +0.55   r=+0.225  p=1.6e-04
    Auditory    -0.36 -> -0.51   r=-0.167  p=5.2e-03
    Dorsal str  +0.17 -> +0.22   r=+0.089  p=.14  (the one flat region)
Three of four regions strengthen monotonically across the run. The Results text
currently reports "beta = 0.20-0.37; all p < 3e-6" as the effect having "reproduced
longitudinally", which is true but reads as STABLE. With no season summary drawn,
that wording is the only place a reader meets the pattern — worth revisiting in the
text.

SIGN CONSISTENCY (printed on run) is a stronger robustness statement than "all six
seasons significant": auditory negative in 96% of episodes, LOC positive in 95%,
rTPJ 93%, dorsal striatum 93%, ventral striatum only 64% — the last visibly hugging
zero, which puts the dorsal-vs-ventral dissociation (supplement S4) in the main text.

UNIT: glm_beta_by_episode.npy holds 278 SEGMENTS, averaged here to 136 EPISODES so
the x-axis matches Figure 1D. Note 136 < Figure 1D's 141: Block B excludes s01e01a/b
and config.EXCLUDED_EPISODES. The printed season means are computed on the
episode-level values actually plotted, so read-out and figure cannot disagree.

  input   : data/b_laughter/glm_beta_by_episode.npy   (278 x 1032)
            data/b_laughter/glm_episodes.csv          (episode, season)
  output  : figures/panels/fig2_panelC_season_glm.png
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
from config import (B_DIR, ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY,
                    DORSAL_STR_IDX, VENTRAL_STR_IDX)

OUT = HERE.parent / "panels" / "fig2_panelC_season_glm.png"

# ---------------------------------------------------------------- CONFIG ----
# Colours match the package's ROI palette (laughter_isc.py ROI_COLORS) and Fig 1D.
ROI_LINES = [
    ("rTPJ",            [ROI_RTPJ_LAUGHTER], "#27AE60"),
    ("LOC / visual",    [ROI_VISUAL_LOC],    "#2980B9"),
    ("Auditory",        [ROI_AUDITORY],      "#D35400"),
    ("Dorsal striatum", DORSAL_STR_IDX,      "#8E44AD"),
    ("Ventral str/NAc", VENTRAL_STR_IDX,     "#C0392B"),
]
SERIES_W, SERIES_A = 0.7, 0.85
DOT_S,   DOT_A     = 6,   0.8
FIGSIZE            = (7.2, 5.0)
# -----------------------------------------------------------------------------

B = np.load(B_DIR / "glm_beta_by_episode.npy")            # (n_seg, 1032)
meta = pd.read_csv(B_DIR / "glm_episodes.csv")

ep_id = pd.Series([e[:6] for e in meta["episode"]])
order = ep_id.drop_duplicates().tolist()                  # broadcast order
ep_season = np.array([int(e[1:3]) for e in order])
sea_ids = sorted(set(meta["season"].values))
x = np.arange(len(order))
bounds = [int(np.where(ep_season == s)[0][0]) for s in sea_ids] + [len(order)]

fig, ax = plt.subplots(figsize=FIGSIZE)

report = []
for label, idx, colour in ROI_LINES:
    seg = np.nanmean(B[:, idx], axis=1)                   # collapse multi-parcel ROIs
    v = pd.Series(seg).groupby(ep_id).mean().reindex(order).values

    ax.plot(x, v, color=colour, lw=SERIES_W, alpha=SERIES_A, zorder=2)
    ax.scatter(x, v, s=DOT_S, color=colour, alpha=DOT_A,
               edgecolors="none", zorder=3, label=label)

    # season means are still computed for the printed read-out, but NOT drawn
    # (author decision, 2026-08-06 — the markers cluttered the traces)
    season_means = [float(np.nanmean(v[bounds[k]:bounds[k + 1]]))
                    for k in range(len(sea_ids))]

    frac = (v > 0).mean() if np.nanmean(v) > 0 else (v < 0).mean()
    report.append((label, float(np.nanmean(v)), 100 * frac, season_means))

for b in bounds[1:-1]:
    ax.axvline(b - 0.5, color="0.85", lw=0.8, zorder=0)
ax.set_xticks([(bounds[k] + bounds[k + 1] - 1) / 2 for k in range(len(sea_ids))])
ax.set_xticklabels([f"S{s}" for s in sea_ids])

ax.axhline(0, color="0.55", lw=0.8, zorder=1)
ax.set_xlim(-1.5, len(order) + 0.5)
ax.set_ylabel("Laughter β")
ax.set_xlabel(f"Episode, in broadcast order (n = {len(order)})")
leg = ax.legend(fontsize=8, loc="upper right", markerscale=2.0,
                handlelength=1.0, handletextpad=0.5, labelspacing=0.3,
                borderpad=0.4, borderaxespad=0.4, framealpha=0.95)
leg.get_frame().set_facecolor("white")
leg.get_frame().set_edgecolor("0.8")
leg.get_frame().set_linewidth(0.6)
leg.set_zorder(6)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig)

print(f"generated  {OUT.name}   ({len(meta)} segments -> {len(order)} episodes)")
for label, mean, pct, sm in report:
    print(f"   {label:16s} mean β {mean:+.3f}  sign-consistent {pct:3.0f}%   "
          f"season means " + " ".join(f"{m:+.2f}" for m in sm))
