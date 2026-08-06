"""
panel_fig1_season_stability.py — Figure 1, Panel D.

Lineage: supersedes the single-subplot version, archived verbatim (with its panel)
at figures/_old/panel_fig1_season_stability_v1.py. That panel becomes the TOP
subplot unchanged; a second subplot is added beneath it.

  TOP    per-season spatial stability of the ISC maps — the existing panel,
         untouched: for each season, the distribution of pairwise spatial
         correlations between that season's episode ISC maps (r = 0.62-0.71).
  BOTTOM per-episode ISC in the three a-priori ROIs (rTPJ / visual / auditory),
         every episode from s01 through s06 in broadcast order, so the reader sees
         the raw episode-level values the summary above is built on. Segments are
         averaged up to whole episodes (a+b, or a-d for the four that ran in four
         parts): 290 segments -> 141 episodes. Plain mean of r, consistent with how
         the package averages ISC elsewhere (compute_isc.py -> isc_all -> .mean(0)).

The two halves answer different questions and are on different scales, hence
separate y-axes: the top is map-to-map SIMILARITY, the bottom is ISC MAGNITUDE.

WHAT THE BOTTOM ADDS. The season means are flat while single episodes range widely
(rTPJ 0.035..0.323 at episode level). Part A claims the spatial PATTERN is
reproducible, not that every episode elicits the same ISC magnitude; showing the
episode series keeps those two claims apart.

ROI ORDERING. LOC (0.435) > auditory (0.230) > rTPJ (0.177) — the expected sensory
ordering. This differs from earlier drafts of this panel, which used the inherited
occipital-pole parcel (RH_Vis_46, ISC 0.115) and so read auditory > rTPJ > visual.
That parcel sat at the 19th percentile of its own network; config.ROI_VISUAL_LOC
documents the 2026-08-05 switch to RH_Vis_49. The visual series here is LOC, NOT
early visual cortex — see the naming caveat in config.py.

  input   : data/a_isc_stability/isc_fmriprep/{isc_all.npy, episodes.csv}
  output  : figures/panels/fig1_panelD_season_stability.png
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
from config import ROI_AUDITORY, ROI_VISUAL_LOC, ROI_RTPJ

V3 = HERE.parents[1] / "data" / "a_isc_stability" / "isc_fmriprep"
OUT = HERE.parent / "panels" / "fig1_panelD_season_stability.png"

# ---------------------------------------------------------------- CONFIG ----
# Bottom subplot: one line per a-priori ROI. Distinct colours are needed here
# (three overlapping series); the top subplot keeps its per-season palette.
ROI_LINES = [
    ("rTPJ",        ROI_RTPJ,     "#27AE60"),
    ("LOC / visual", ROI_VISUAL_LOC,  "#2980B9"),
    ("Auditory",    ROI_AUDITORY, "#D35400"),
]
VIOLIN_COLOR = "#4D4D4D"   # ONE dark grey for the top: season is already the
                           # x-axis, so the old six-colour palette encoded nothing
LINE_W, DOT_S, DOT_A = 0.7, 6, 0.8
FIGSIZE = (7.2, 6.4)
HEIGHT_RATIOS = (1.0, 1.15)
# -----------------------------------------------------------------------------

isc_all = np.load(V3 / "isc_all.npy")
isc = isc_all[:, :1000]
eps = pd.read_csv(V3 / "episodes.csv")["episode"].tolist()
seasons = np.array([int(e[1:3]) for e in eps])
sea_ids = sorted(set(seasons))

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=FIGSIZE,
    gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.42})

# ── TOP: per-season spatial stability (unchanged from v1) ────────────────────
dists, means = [], []
for s in sea_ids:
    mat = isc[seasons == s]
    pairs = np.corrcoef(mat)[np.triu_indices(mat.shape[0], k=1)]
    dists.append(pairs); means.append(pairs.mean())

for body in ax_top.violinplot(dists, positions=sea_ids, widths=0.8,
                              showextrema=False)["bodies"]:
    body.set_facecolor(VIOLIN_COLOR); body.set_edgecolor("none"); body.set_alpha(0.55)
ax_top.scatter(sea_ids, means, color="k", zorder=3, s=18)
for s, m in zip(sea_ids, means):
    ax_top.annotate(f"{m:.2f}", (s, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
ax_top.set_xticks(sea_ids); ax_top.set_xticklabels([f"S{s}" for s in sea_ids])
ax_top.set_ylabel("Spatial ISC correlation (r)"); ax_top.set_xlabel("Season")
# ylim inherited from v1. NOTE this CLIPS the lower KDE tail of two seasons: S4
# reaches 0.126 and S6 reaches 0.086 (single outlier episode pairs). No violin is
# clipped at the top — all six top out at 0.808-0.829. S6 merely LOOKS flat-topped
# because it has the widest spread of the six (0.654 +/- 0.124), so its kernel is
# broad. Widening the floor to show those tails would compress the informative
# 0.6-0.8 band, so the clip is kept deliberately.
ax_top.set_ylim(0.3, 0.95)
ax_top.spines[["top", "right"]].set_visible(False)

# ── BOTTOM: per-episode ISC in the three a-priori ROIs, s01 -> s06 ───────────
x = np.arange(len(eps))
ep_id = pd.Series([e[:6] for e in eps])
order = ep_id.drop_duplicates().tolist()          # 141 episodes, broadcast order
ep_season = np.array([int(e[1:3]) for e in order])
x = np.arange(len(order))

for label, roi, colour in ROI_LINES:
    v = pd.Series(isc_all[:, roi]).groupby(ep_id).mean().reindex(order).values
    ax_bot.plot(x, v, color=colour, lw=LINE_W, alpha=0.85, zorder=2)
    ax_bot.scatter(x, v, s=DOT_S, color=colour, alpha=DOT_A,
                   edgecolors="none", zorder=3, label=label)

# season boundaries + centred season labels
bounds = [int(np.where(ep_season == s)[0][0]) for s in sea_ids] + [len(order)]
for b in bounds[1:-1]:
    ax_bot.axvline(b - 0.5, color="0.85", lw=0.8, zorder=0)
ax_bot.set_xticks([(bounds[k] + bounds[k + 1] - 1) / 2 for k in range(len(sea_ids))])
ax_bot.set_xticklabels([f"S{s}" for s in sea_ids])

ax_bot.axhline(0, color="0.7", lw=0.6, zorder=0)
ax_bot.set_xlim(-1.5, len(order) + 0.5)
ax_bot.set_ylabel("ISC (r)")
ax_bot.set_xlabel(f"Episode, in broadcast order (n = {len(order)})")
leg = ax_bot.legend(fontsize=9, loc="upper right", markerscale=2.2,
                    handlelength=1.0, handletextpad=0.5, labelspacing=0.35,
                    borderpad=0.45, borderaxespad=0.4, framealpha=0.95)
leg.get_frame().set_facecolor("white")     # opaque box: the series run underneath it
leg.get_frame().set_edgecolor("0.8")
leg.get_frame().set_linewidth(0.6)
leg.set_zorder(6)
ax_bot.spines[["top", "right"]].set_visible(False)

fig.savefig(OUT, dpi=300, bbox_inches="tight"); plt.close(fig)

print(f"generated  {OUT.name}")
print("   top    per-season r: " + ", ".join(f"S{s}={m:.2f}" for s, m in zip(sea_ids, means)))
for label, roi, _ in ROI_LINES:
    v = pd.Series(isc_all[:, roi]).groupby(ep_id).mean().reindex(order).values
    print(f"   bottom {label:9s} mean {v.mean():.3f}   episodes {v.min():+.3f}..{v.max():.3f}")
print(f"   bottom {len(eps)} segments averaged to {len(order)} episodes")
