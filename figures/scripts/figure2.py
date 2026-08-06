"""
figure2.py — compose Figure 2 (Part B: laughter-associated activation, GLM).

2x2 layout, equal panels:
  A concept (concept_panels/)  | B whole-brain GLM t-map
  C GLM beta by season         | D per-viewer replication

Panel letters follow READING order, so the two brain montages sit in the same
(right) column, stacked B over D — swapped 2026-08-06. Panel files were renamed to
match their letters at the same time.
Output: output/figure2.png (+ .pdf)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
PANELS = HERE.parent / "panels"
ASSETS = HERE.parent / "concept_panels"
OUT = HERE.parent / "output"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

def show(ax, path):
    ax.imshow(mpimg.imread(path)) if Path(path).exists() else ax.text(0.5, 0.5, f"[missing {Path(path).name}]", ha="center", va="center")
    ax.axis("off")

def letter(ax, s):
    ax.text(-0.01, 1.01, s, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

fig = plt.figure(figsize=(12, 9))
gs = GridSpec(2, 2, figure=fig, hspace=0.10, wspace=0.06, left=0.03, right=0.99, top=0.96, bottom=0.03)

axA = fig.add_subplot(gs[0, 0]); show(axA, ASSETS / "fig2_section1.png"); letter(axA, "A")
axB = fig.add_subplot(gs[0, 1]); show(axB, PANELS / "fig2_panelB_glm_brain.png"); letter(axB, "B")
axC = fig.add_subplot(gs[1, 0]); show(axC, PANELS / "fig2_panelC_season_glm.png"); letter(axC, "C")
axD = fig.add_subplot(gs[1, 1]); show(axD, PANELS / "fig2_panelD_per_viewer.png"); letter(axD, "D")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure2.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("composed  output/figure2.png (+ .pdf)")
