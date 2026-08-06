"""
panel_fig3_raster_all.py — Figure 3, Panel A: laugh track across the whole corpus.

NEW 2026-08-06, and now the OPENING panel of Figure 3 (author decision). The
previous panel A showed the laugh track of ONE episode as a barcode — an
orientation device, not evidence, silent on whether that structure is typical.
This is the corpus view instead: every annotated episode as one row, laughter in
black, time left to right, grouped by season. It revives the dense multi-episode
barcode of the pre-revision figure.

WHAT IT SHOWS. 137 whole episodes (segments a+b, or a-d, concatenated), 280
annotated segments in total. The texture is the point: laughter is dense and
near-continuous in every episode across six seasons, with visible vertical
striping (act breaks / scene structure recur at similar times) and no episode
that looks anomalous.

TIME AXIS IS REAL MINUTES, NOT PERCENT. Episodes run 876-2016 TRs (21.8-50.1 min;
the long ones are the four multi-part episodes). Normalising each to 0-100% would
squash a 50-minute finale onto the same width as a 22-minute episode and make the
striping meaningless. Instead every episode is TRUNCATED to the shortest one
(876 TRs = 21.8 min), so a column is the same clock time in every row. The tails
of the four long episodes are therefore not shown.

Single colour on purpose: season is already encoded by row position and the
labels, so colouring it would be redundant (same reasoning as Fig 1D / 2C).

  input   : data/0_prep/laughter_annotations/*.csv   (280 segments, per-TR ls)
  output  : figures/panels/fig3_panelA_raster_all.png
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

OUT = HERE.parent / "panels" / "fig3_panelA_raster_all.png"

# ---------------------------------------------------------------- CONFIG ----
INK       = "#1A1A1A"   # laughter
PAPER     = "white"
SEASON_LW = 1.1         # separator between seasons
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

episodes = sorted(parts)                                  # broadcast order
ls = {e: np.concatenate([seg[k] for k in sorted(v)]) for e, v in parts.items()}
n_tr = min(len(v) for v in ls.values())                   # common window
M = np.vstack([ls[e][:n_tr] for e in episodes]).astype(float)

season = np.array([int(e[1:3]) for e in episodes])
sea_ids = sorted(set(season))
bounds = [int(np.where(season == s)[0][0]) for s in sea_ids] + [len(episodes)]

fig, ax = plt.subplots(figsize=FIGSIZE)
cmap = matplotlib.colors.ListedColormap([PAPER, INK])
ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest",
          extent=[0, n_tr * TR_SEC / 60, len(episodes), 0])

for b in bounds[1:-1]:
    ax.axhline(b, color="0.45", lw=SEASON_LW)
ax.set_yticks([(bounds[k] + bounds[k + 1]) / 2 for k in range(len(sea_ids))])
ax.set_yticklabels([f"S{s}" for s in sea_ids])
ax.tick_params(axis="y", length=0)

ax.set_xlabel("Time within episode (min)")
ax.set_ylabel("Episode")
ax.set_xlim(0, n_tr * TR_SEC / 60)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300); plt.close(fig)

frac = M.mean()
print(f"generated  {OUT.name}  ({len(episodes)} episodes x {n_tr} TRs = "
      f"{n_tr * TR_SEC / 60:.1f} min common window; {100 * frac:.1f}% of TRs laughter)")
print("   episodes per season: " +
      "  ".join(f"S{s}={int((season == s).sum())}" for s in sea_ids))
