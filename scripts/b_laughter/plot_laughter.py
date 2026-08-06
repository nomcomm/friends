"""
plot_laughter.py — consolidate Block B figures into main / supplement / exploratory
====================================================================================
Single owner of the Block B figure tiers. Working figures from the individual
analysis scripts land loose in results/analysis_plots/b_laughter/; this script:

  1. GENERATES the composite figure that is otherwise only made ad-hoc, so it is
     reproducible in the package:
       - glm_voxelwise_slices.png voxelwise GLM group t-map (if group_tmap exists)
     (glm_brain.png used to be generated here too; as of 2026-08-06 the Figure 2
      panel-B map is generated in figures/scripts/panel_fig2_glm_brain.py, so that
      paper figures are produced by the figures layer, not the analysis layer.)
  2. SWEEPS any loose *.png at the top level → exploratory/  (keeps top level clean)
  3. CURATES the paper set into:
       main/        Part-B primary = GLM story (tentative narrative: A=ISC, B=GLM)
       supplement/  ISC-laughter + robustness / decision-support
     (exploratory/ holds every working figure.)

Re-run any analysis script, then re-run this to re-tidy. Edit MAIN/SUPPLEMENT
below as the narrative firms up.

Usage
  python plot_laughter.py
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (B_DIR, B_FIG_DIR, MELBOURNE_ATLAS_DIR, N_PARCELS, FDR_ALPHA)

MAIN = B_FIG_DIR / "main"
SUPP = B_FIG_DIR / "supplement"
EXPL = B_FIG_DIR / "exploratory"

# curated tier assignment (tentative narrative: Part B = GLM primary, ISC supplement)
MAIN_FIGS = ["glm_voxelwise_slices.png", "fig_glm_seasons.png",
             "fig_season_consistency.png", "fig_dose_response.png"]
SUPP_FIGS = ["fig_laughter_isc.png", "fig_wholebrain_contrast.png", "fig_striatum.png",
             "fig_humor_harvesting.png", "fig_glm_vs_isc.png",
             "fig_glm_regions.png", "fig_wholebrain_2x2_comparison.png"]


def gen_voxelwise_slices():
    """Voxelwise GLM group t-map slices → main/glm_voxelwise_slices.png (if group_tmap exists)."""
    tmap_path = B_DIR / "glm_voxelwise" / "group_tmap.nii.gz"
    if not tmap_path.exists():
        print("  voxelwise slices: group_tmap.nii.gz missing — skip"); return
    from nilearn import datasets, image, plotting
    tmap = image.load_img(str(tmap_path)); t = tmap.get_fdata(); tv = t[t != 0]
    disp = 12.0; vmax = float(np.percentile(np.abs(tv), 99.5))
    fig = plt.figure(figsize=(15, 7))
    for r, cc, lab in [(1, [40, 48, 56, 64], f"cortical (|t|>{disp:.0f})"), (2, [-6, 0, 6, 12], "striatum / subcortical")]:
        plotting.plot_stat_map(tmap, bg_img=datasets.load_mni152_template(resolution=2), threshold=disp,
                               vmax=vmax, display_mode="z", cut_coords=cc, colorbar=True,
                               axes=fig.add_subplot(2, 1, r), title=f"Voxelwise GLM laughter contrast — {lab}")
    fig.suptitle("Voxelwise GLM laughter contrast (N=278 episodes)  red=activated blue=deactivated",
                 fontsize=13, fontweight="bold")
    MAIN.mkdir(parents=True, exist_ok=True)
    fig.savefig(MAIN / "glm_voxelwise_slices.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print("  generated main/glm_voxelwise_slices.png")


def main():
    for d in (MAIN, SUPP, EXPL):
        d.mkdir(parents=True, exist_ok=True)

    print("Generating composite figures ...")
    gen_voxelwise_slices()

    # sweep loose top-level pngs into exploratory (keep top level = tier folders only)
    swept = 0
    for p in B_FIG_DIR.glob("*.png"):
        shutil.move(str(p), str(EXPL / p.name)); swept += 1
    print(f"Swept {swept} loose figures → exploratory/")

    # curate: copy from exploratory (or main, for generated) into the tiers
    def place(names, dest):
        n = 0
        for name in names:
            src = MAIN / name if (MAIN / name).exists() else EXPL / name
            if not src.exists():
                print(f"    [missing] {name}"); continue
            if src.resolve() != (dest / name).resolve():
                shutil.copy2(src, dest / name)
            n += 1
        return n
    nm = place(MAIN_FIGS, MAIN); ns = place(SUPP_FIGS, SUPP)

    print(f"\nCurated: main/={nm} figures, supplement/={ns} figures, exploratory/={len(list(EXPL.glob('*.png')))}")
    print(f"  main/       (Part B = GLM):     {', '.join(MAIN_FIGS)}")
    print(f"  supplement/ (ISC + robustness): {', '.join(SUPP_FIGS)}")
    print(f"Figures root: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
