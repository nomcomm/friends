"""
panel_fig2_per_viewer.py — Figure 2, Panel D (per-viewer GLM laughter maps).

MOVED HERE 2026-08-06, same rationale as panel_fig2_glm_brain.py. This panel used
to be drawn at the end of the ANALYSIS script scripts/b_laughter/glm_per_viewer.py,
written to results/analysis_plots/b_laughter/fig_glm_per_viewer.png, and then copied
into panels/ by stage_panels.py with its suptitle cropped off (top 6%). Now it is
generated here, directly from the pipeline's saved arrays, so paper figures come
from the figures layer and no crop hack is needed. glm_per_viewer.py keeps its
ANALYSIS outputs (glm_per_viewer.csv, _consistency.csv, _stats.txt) and no longer
draws a figure.

WHAT IT SHOWS. One cortical beta map per viewer, from the SAME first-level fit as
the group map (glm_beta_by_subject_episode.npy, written by glm_contrast.py) — the
group result is arithmetically the viewer-average of these, so the two cannot drift
apart. Each row's title carries that viewer's rTPJ beta. This is the R1.1 robustness
check: the rTPJ activation is present in every individual, not only in the average.

READ WITH CARE: sub-03 (+0.630) is ~4x sub-05 (+0.112) and dominates the group mean
(+0.282). All four are positive and independently significant, but the MAGNITUDE is
carried disproportionately by one viewer — visible here, and worth not overstating.

Canvas is black, matching panel B (2026-08-06). Threshold is 15% of vmax, vmax is
the 99th percentile of |beta| across viewers — both inherited from the original.

  input   : data/b_laughter/glm_beta_by_subject_episode.npy   (n_ep, 4, 1032)
            data/b_laughter/glm_per_viewer_consistency.csv    (inter-viewer r)
  output  : figures/panels/fig2_panelD_per_viewer.png
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from config import B_DIR, N_PARCELS, N_SUBJECTS, SUBJECTS, ROI_RTPJ_LAUGHTER

OUT = HERE.parent / "panels" / "fig2_panelD_per_viewer.png"

# ---------------------------------------------------------------- CONFIG ----
THRESH_FRAC = 0.15     # threshold = this fraction of vmax
CUTS        = 6        # axial cuts per viewer (nilearn picks the positions)
FACECOLOR   = "black"
DPI         = 140
ROW_H       = 2.2      # inches per viewer row
FIGWIDTH    = 13
# -----------------------------------------------------------------------------


def per_viewer_maps():
    """Per-viewer mean cortical beta map — identical construction to glm_per_viewer.py."""
    B = np.load(B_DIR / "glm_beta_by_subject_episode.npy")      # (n_ep, N_SUBJECTS, 1032)
    maps = np.full((N_SUBJECTS, N_PARCELS), np.nan)
    for s in range(N_SUBJECTS):
        present = ~np.isnan(B[:, s, :N_PARCELS]).all(axis=1)
        maps[s] = np.nanmean(B[present, s, :N_PARCELS], axis=0)
    return maps


def main():
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain
    from nilearn import plotting

    maps = per_viewer_maps()

    cons = pd.read_csv(B_DIR / "glm_per_viewer_consistency.csv", index_col=0).values
    iu = np.triu_indices(cons.shape[0], k=1)
    mean_r = float(cons[iu].mean())

    mask = expand_mask(Brain_Data(nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)["maps"]))
    bg = nilearn.datasets.load_mni152_template()
    vmax = float(np.nanpercentile(np.abs(maps), 99))

    fig, axes = plt.subplots(N_SUBJECTS, 1, figsize=(FIGWIDTH, ROW_H * N_SUBJECTS),
                             facecolor=FACECOLOR)
    for ax, s in zip(np.atleast_1d(axes), range(N_SUBJECTS)):
        ax.set_facecolor(FACECOLOR)
        nii = roi_to_brain(pd.Series(maps[s]), mask).to_nifti()
        plotting.plot_stat_map(
            nii, bg_img=bg, threshold=vmax * THRESH_FRAC, vmax=vmax,
            display_mode="z", cut_coords=CUTS, colorbar=True, axes=ax,
            title=f"{SUBJECTS[s]}   rTPJ beta = {maps[s][ROI_RTPJ_LAUGHTER]:+.3f}",
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
    plt.close(fig)
    print(f"generated  {OUT.name}  (vmax {vmax:.3f}, inter-viewer spatial r = {mean_r:.3f})")
    for s in range(N_SUBJECTS):
        print(f"    {SUBJECTS[s]}  rTPJ beta = {maps[s][ROI_RTPJ_LAUGHTER]:+.3f}")


if __name__ == "__main__":
    main()
