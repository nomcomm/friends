"""
panel_fig1_overall_isc.py — Figure 1, Panel B.

Lineage: supersedes the previous single-row version, archived verbatim (with its
panel) at figures/_old/panel_fig1_overall_isc_v1.py. Changed in four respects:
  1. LAYOUT     one row of 6 axial slices  ->  2 x 3 grid
  2. BACKGROUND nilearn's skull-stripped MNI152  ->  the bundled whole-head
                FSL MNI152_T1_2mm (config.MNI_BG_IMG), i.e. the template that
                still shows skull/scalp ("brain with bones").
  3. THRESHOLD  0.05 -> a hard 0.10, as in the earlier version (author decision,
                2026-08-03). This is deliberately conservative: it paints the
                489/1000 parcels the Results call "roughly half", and so hides
                the weaker-but-positive parcels behind the companion claim that
                ISC is "positive in nearly all" (991/1000). The caption states
                the threshold. Display only — no analysis value depends on it.
  4. COLOUR     plot_stat_map -> plot_anat + add_overlay. plot_stat_map forces a
                SYMMETRIC normalisation ([-vmax, +vmax]) whatever symmetric_cbar
                says, so v1 rendered ISC 0.10 at 62% of the colormap and much of
                cortex read as near-maximal; only 10/1000 parcels actually reach
                0.40. add_overlay takes a plain vmin/vmax, so colour now equals
                value and the colourbar is exact. (v1's own colourbar was
                self-consistent but spanned -0.023..0.48, which masked this.)

The DATA path is untouched and identical to v1: mean cortical ISC across
episodes, painted onto the Schaefer-1000 atlas. ISC itself is the median of the
6 pairwise subject correlations (compute_isc.py) — the conservative estimator;
leave-one-out would run ~50% higher.

Cosmetics (title, tick labels, colormap choice, ISC icon) are deliberately left
plain — they are handled separately. Everything adjustable lives in the CONFIG
block below.

  input   : data/a_isc_stability/isc_fmriprep/isc_all.npy   (290 x 1032)
  output  : figures/panels/fig1_panelB_overall_isc.png
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
sys.path.insert(0, str(CODE))
from config import MNI_BG_IMG, A_ISC_FMRIPREP_DIR, N_PARCELS

OUT = HERE.parent / "panels" / "fig1_panelB_overall_isc.png"

# ---------------------------------------------------------------- CONFIG ----
Z_SLICES  = [-12, 0, 12, 24, 36, 48]   # 6 axial cuts, filled row-major into 2 x 3
NROWS, NCOLS = 2, 3
CMAP      = "hot"      # v1 setting; swap freely (magma / inferno are perceptually uniform)
CMAP_LO   = 0.15       # start this far into CMAP, so the weakest painted parcels
                       # are not near-black against a black background (0 = off)
THRESHOLD = 0.10       # hard display threshold (see docstring); v1 used 0.05
VMAX      = 0.40
FACECOLOR = "black"    # matches the earlier figure's dark surround
FIGSIZE   = (7.5, 5.0)
SHOW_ZLABEL = True
# -----------------------------------------------------------------------------


def isc_map_to_nifti():
    """Mean cortical ISC across episodes -> volumetric nifti (identical to v1)."""
    mean_map = np.nanmean(np.load(A_ISC_FMRIPREP_DIR / "isc_all.npy")[:, :N_PARCELS], axis=0)
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain
    mask = expand_mask(Brain_Data(nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)["maps"]))
    return roi_to_brain(pd.Series(mean_map), mask).to_nifti(), mean_map


def _cmap():
    """CMAP, optionally truncated so its darkest end stays visible on black."""
    base = plt.get_cmap(CMAP)
    if CMAP_LO <= 0:
        return base
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        f"{CMAP}_lo{CMAP_LO}", base(np.linspace(CMAP_LO, 1.0, 256)))


def _mask_subthreshold(nii):
    """Sub-threshold parcels (and the 0-valued background) -> NaN, so they are
    simply not painted. Needed because add_overlay has no threshold argument."""
    import nibabel as nib
    d = nii.get_fdata().astype(float)
    d[d < THRESHOLD] = np.nan
    return nib.Nifti1Image(d, nii.affine)


def main():
    from nilearn import plotting

    nii, mean_map = isc_map_to_nifti()

    fig = plt.figure(figsize=FIGSIZE, facecolor=FACECOLOR)
    gs = GridSpec(NROWS, NCOLS, figure=fig, wspace=0.01, hspace=0.01,
                  left=0.01, right=0.87, top=0.97, bottom=0.03)

    cmap = _cmap()
    nii_masked = _mask_subthreshold(nii)

    for k, z in enumerate(Z_SLICES):
        ax = fig.add_subplot(gs[k // NCOLS, k % NCOLS])
        ax.set_facecolor(FACECOLOR)
        # plot_anat + add_overlay, NOT plot_stat_map: plot_stat_map forces a
        # SYMMETRIC normalisation ([-vmax, +vmax]) regardless of
        # symmetric_cbar=False, so ISC 0.10 would render at 62% of the colormap
        # and no colourbar over [0.10, 0.40] could match the image. add_overlay
        # takes a plain vmin/vmax, so colour == value and the bar below is exact.
        d = plotting.plot_anat(
            str(MNI_BG_IMG), display_mode="z", cut_coords=[z],
            annotate=False, black_bg=True, axes=ax,
        )
        d.add_overlay(nii_masked, cmap=cmap, vmin=THRESHOLD, vmax=VMAX)
        if SHOW_ZLABEL:
            # NB: nilearn draws each cut into a sub-axes, so text placed on `ax`
            # in axes-coords gets clipped. Anchor to the cell's figure position.
            pos = ax.get_position()
            fig.text(pos.x0 + 0.008, pos.y0 + 0.008, f"z = {z}",
                     color="white", fontsize=8, ha="left", va="bottom")

    # one shared colourbar for the whole grid
    cax = fig.add_axes([0.89, 0.28, 0.022, 0.44])
    norm = matplotlib.colors.Normalize(vmin=THRESHOLD, vmax=VMAX)
    cb = matplotlib.colorbar.ColorbarBase(
        cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_ticks([THRESHOLD, VMAX])
    cb.set_ticklabels([f"{THRESHOLD:g}", f"{VMAX:g}"])
    cb.ax.tick_params(colors="white", labelsize=8, length=0)
    cb.outline.set_edgecolor("white")

    fig.savefig(OUT, dpi=300, facecolor=FACECOLOR, bbox_inches="tight")
    plt.close(fig)
    print(f"generated  {OUT.name}  ({NROWS}x{NCOLS} grid, whole-head bg, "
          f"mean cortical ISC {np.nanmean(mean_map):.3f})")


if __name__ == "__main__":
    main()
