"""
figure1.py — compose Figure 1 (Part A: a stable, shared audience response).

2x2 layout:  A overview schematic (concept_panels/) | B overall ISC (panel)
             C spatial stability: scatter + histogram | D per-season stability
Panels come from panels/ (staged/generated) and concept_panels/. Output: output/figure1.png (+ .pdf)
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

HERE = Path(__file__).resolve().parent
PANELS = HERE.parent / "panels"
ASSETS = HERE.parent / "concept_panels"
OUT = HERE.parent / "output"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

def show(ax, path):
    ax.imshow(mpimg.imread(path)) if Path(path).exists() else ax.text(0.5, 0.5, f"[missing {Path(path).name}]", ha="center", va="center")
    ax.axis("off")

def letter(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right")

fig = plt.figure(figsize=(11, 8))
gs = GridSpec(2, 2, figure=fig, hspace=0.12, wspace=0.08, left=0.04, right=0.98, top=0.96, bottom=0.03)

axA = fig.add_subplot(gs[0, 0]); show(axA, ASSETS / "fig1_section1.png"); letter(axA, "A")
axB = fig.add_subplot(gs[0, 1]); show(axB, PANELS / "fig1_panelB_overall_isc.png"); letter(axB, "B")

gsC = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.05)
axC1 = fig.add_subplot(gsC[0, 0]); show(axC1, PANELS / "fig1_panelC1_stability_scatter.png")
axC2 = fig.add_subplot(gsC[0, 1]); show(axC2, PANELS / "fig1_panelC2_stability_hist.png")
letter(axC1, "C")

axD = fig.add_subplot(gs[1, 1]); show(axD, PANELS / "fig1_panelD_season_stability.png"); letter(axD, "D")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure1.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
print("composed  output/figure1.png (+ .pdf)")
