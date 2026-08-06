"""
figure3.py — compose Figure 3 (Part B: the laugh track and the humor it samples).

2x2 layout, equal panels — restructured 2026-08-06 from two stacked wide panels:
  A  laugh track across the whole corpus (137 episodes x 21.8 min)
  B  laughter rate per episode, by season
  C  humor near-continuous vs laughter periodic ("harvesting")
  D  humor composition (Juckel typology)

All four panel scripts share FIGSIZE = (6.6, 4.9) and save WITHOUT
bbox_inches="tight", so every panel is exactly 1980x1470 (aspect 1.347) and the
quadrants scale evenly. Do not reintroduce tight cropping in a panel script — the
saved aspect would then depend on its tick labels and the 2x2 would go ragged.

The figure is deliberately stimulus-side: it characterises what the audience was
shown, while Figures 1-2 carry the brain results. The dose-response analysis was
prototyped for panel D and NOT used — it duplicates supplement S5 and was judged
too complex for the main text (author decision, 2026-08-06); both candidate
scripts are kept at figures/_old/panel_fig3_dose_*_CANDIDATE.py.

Output: output/figure3.png (+ .pdf)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
PANELS = HERE.parent / "panels"
OUT = HERE.parent / "output"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

PLAN = [("A", "fig3_panelA_raster_all.png"),
        ("B", "fig3_panelB_laughs_per_min.png"),
        ("C", "fig3_panelC_harvesting.png"),
        ("D", "fig3_panelD_humor_composition.png")]


def show(ax, path):
    if Path(path).exists():
        ax.imshow(mpimg.imread(path))
    else:
        ax.text(0.5, 0.5, f"[missing {Path(path).name}]", ha="center", va="center")
    ax.axis("off")


def letter(ax, s):
    ax.text(-0.01, 1.01, s, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="right")


fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.06,
              left=0.03, right=0.99, top=0.96, bottom=0.03)

for i, (lab, fname) in enumerate(PLAN):
    ax = fig.add_subplot(gs[i // 2, i % 2])
    show(ax, PANELS / fname)
    letter(ax, lab)

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure3.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("composed  output/figure3.png (+ .pdf)")
