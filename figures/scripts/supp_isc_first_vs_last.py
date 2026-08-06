"""
supp_isc_first_vs_last.py — supplementary ISC maps for the two episodes in Fig 1C.

Figure 1 panel C1 scatters the whole-brain ISC map of the FIRST segment of the run
against the LAST one (s01e02a vs s06e24d, r = 0.64). That scatter shows the
agreement but not the maps it is computed from. This renders those two maps, one
file each, so they can be shown beside the scatter.

Each point in the C1 scatter is one parcel: its x is that parcel's value in the
first map here, its y is its value in the second.

RENDERING — deliberately imported from panel_fig1_overall_isc (same colormap,
threshold, vmax, slices, background). Identical scale is the whole point: the two
maps must be comparable to each other and to Figure 1B's episode-average map, and
they stay in step automatically if panel B's styling is retouched.

CAVEAT worth stating in the caption: these are SINGLE-segment maps (~470 TRs, 6
subject pairs), so they are far noisier than the 290-segment average in Fig 1B.
That is the point of the pair — the shared spatial pattern is visible in each
individual segment, five years of production apart, not only in the average.

  input   : data/a_isc_stability/isc_fmriprep/{isc_all.npy, episodes.csv}
  output  : figures/output/supplement/supp_isc_map_{episode}.png   (2 files)
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import panel_fig1_overall_isc as pb          # single source of truth for styling
from config import MNI_BG_IMG, A_ISC_FMRIPREP_DIR, N_PARCELS

EPISODES = ["s01e02a", "s06e24d"]            # must match panel_fig1_stability EP_X/EP_Y
OUTDIR = HERE.parent / "output" / "supplement"

# ---------------------------------------------------------------- CONFIG ----
# These render on WHITE for use as an inset (2026-08-05), unlike Fig 1B which is
# on black. The colour SCALE is still imported from panel B, so values remain
# directly comparable; only the surround changes.
WHITE_BG      = True    # white figure surround + black annotation
SKULL_STRIPPED = True   # nilearn's brain-only MNI152; the whole-head template's
                        # scalp reads as a grey halo against white
BG_FLOOR      = 0.02    # anat intensities below this fraction of max render as
                        # the colormap's "under" colour (white) — see main()
# -----------------------------------------------------------------------------


def main():
    from nilearn import plotting
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain

    isc = np.load(A_ISC_FMRIPREP_DIR / "isc_all.npy")[:, :N_PARCELS]
    eps = pd.read_csv(A_ISC_FMRIPREP_DIR / "episodes.csv")["episode"].tolist()

    # fetch/expand the atlas once, reuse for both episodes
    mask = expand_mask(Brain_Data(nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=N_PARCELS, yeo_networks=7, resolution_mm=1)["maps"]))
    cmap = pb._cmap()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    facecolor = "white" if WHITE_BG else pb.FACECOLOR
    fgcolor   = "black" if WHITE_BG else "white"
    bg_img    = nilearn.datasets.load_mni152_template() if SKULL_STRIPPED else str(MNI_BG_IMG)
    bg_cmap   = "gray"

    if WHITE_BG:
        # nilearn normalises the anat image with vmin < 0 (observed: -0.45..1.00),
        # so out-of-brain voxels (value 0) land ~1/3 up the grey ramp and paint
        # MID-GREY tiles over the white axes. NaN-masking the template does NOT
        # fix this — nilearn strips NaNs before display. Instead give the colormap
        # an "under" colour and lift the floor above 0 (done after plot_anat, since
        # nilearn sets the clim itself), so background voxels render white.
        from copy import copy
        bg_cmap = copy(plt.get_cmap("gray"))
        bg_cmap.set_under("white")

    for ep in EPISODES:
        if ep not in eps:
            print(f"  [skip] {ep} not in episodes.csv"); continue
        vec = isc[eps.index(ep)]
        nii = pb._mask_subthreshold(roi_to_brain(pd.Series(vec), mask).to_nifti())

        fig = plt.figure(figsize=pb.FIGSIZE, facecolor=facecolor)
        gs = GridSpec(pb.NROWS, pb.NCOLS, figure=fig, wspace=0.01, hspace=0.01,
                      left=0.01, right=0.87, top=0.97, bottom=0.03)
        for k, z in enumerate(pb.Z_SLICES):
            ax = fig.add_subplot(gs[k // pb.NCOLS, k % pb.NCOLS])
            ax.set_facecolor(facecolor)
            d = plotting.plot_anat(bg_img, display_mode="z", cut_coords=[z],
                                   annotate=False, black_bg=not WHITE_BG, axes=ax,
                                   cmap=bg_cmap)
            if WHITE_BG:
                # must happen BEFORE add_overlay, while the anat is the only image
                for _sl in d.axes.values():
                    for _im in _sl.ax.get_images():
                        _im.set_clim(vmin=BG_FLOOR * _im.norm.vmax, vmax=_im.norm.vmax)
            d.add_overlay(nii, cmap=cmap, vmin=pb.THRESHOLD, vmax=pb.VMAX)
            pos = ax.get_position()
            fig.text(pos.x0 + 0.008, pos.y0 + 0.008, f"z = {z}",
                     color=fgcolor, fontsize=8, ha="left", va="bottom")

        cax = fig.add_axes([0.89, 0.28, 0.022, 0.44])
        norm = matplotlib.colors.Normalize(vmin=pb.THRESHOLD, vmax=pb.VMAX)
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                              orientation="vertical")
        cb.set_ticks([pb.THRESHOLD, pb.VMAX])
        cb.set_ticklabels([f"{pb.THRESHOLD:g}", f"{pb.VMAX:g}"])
        cb.ax.tick_params(colors=fgcolor, labelsize=8, length=0)
        cb.outline.set_edgecolor(fgcolor)

        out = OUTDIR / f"supp_isc_map_{ep}.png"
        fig.savefig(out, dpi=300, facecolor=facecolor, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}   painted {int((vec >= pb.THRESHOLD).sum())}/{N_PARCELS} parcels, "
              f"range {vec.min():+.3f}..{vec.max():.3f}")

    i, j = eps.index(EPISODES[0]), eps.index(EPISODES[1])
    print(f"generated  supplement ISC maps  ({EPISODES[0]} vs {EPISODES[1]}: "
          f"spatial r = {np.corrcoef(isc[i], isc[j])[0, 1]:.3f} — the value in Fig 1C1)")


if __name__ == "__main__":
    main()
