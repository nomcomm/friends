"""
plot_stability.py — Block A figures: overall ISC + spatial stability (fMRIPrep)
================================================================================
Ported from 03_REVISION/scripts/01_fig_isc.py + 01_fig_spatial_stability.py,
rendered from the fMRIPrep overall-ISC maps (compute_isc.py).

  input   : data/a_isc_stability/isc_fmriprep/isc_all.npy   (n_episodes × 1032)
            data/a_isc_stability/isc_fmriprep/episodes.csv
  outputs : results/analysis_plots/a_isc_stability/
              fig_overall_isc.png        (1A) Fisher-z mean ISC brain map
              fig_episode_maps_s1.png    (1C) Season-1 per-episode small multiples
              fig_spatial_scatter.png    (1D) two episodes' ISC maps vs each other
              fig_stability_hist.png     (1E) all pairwise spatial correlations
              stability_stats.txt        headline r ± sd, N, per-season breakdown

Brain maps use the 1000 Schaefer cortical parcels (subcortical ROIs 1000–1031 are
ignored for cortical rendering). Stability uses all included episodes (290); the
4-subject-only value is also reported as a robustness check. Skip-if-exists.

Usage
  python plot_stability.py            # build missing figures
  python plot_stability.py --force    # rebuild all
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import A_ISC_FMRIPREP_DIR, A_FIG_DIR, N_PARCELS, MNI_BG_IMG, VIZ_THRESHOLD_ISC


# ── brain rendering helpers (nltools + nilearn) ─────────────────────────────────
def _roi_mask():
    """Expanded Schaefer-1000 mask for roi_to_brain (built once)."""
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask
    sch = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1)
    return expand_mask(Brain_Data(sch["maps"]))


def _to_nifti(vec_1000, mask):
    from nltools.mask import roi_to_brain
    return roi_to_brain(pd.Series(vec_1000), mask).to_nifti()


def stats_and_stability(cort, episodes):
    """Pairwise spatial-correlation stability, overall + per season."""
    def pair_mean_sd(mat):
        n = mat.shape[0]
        r = np.corrcoef(mat)
        pairs = r[np.triu_indices(n, k=1)]
        return pairs, pairs.mean(), pairs.std(), n
    pairs, mean_r, sd_r, n = pair_mean_sd(cort)

    lines = ["Spatial ISC stability (fMRIPrep, 1000 cortical parcels)",
             "=" * 56,
             f"All episodes: N={n}, {len(pairs)} pairs, mean r = {mean_r:.3f} ± {sd_r:.3f}"]

    # robustness: 4-subject-only
    idx4 = episodes["n_subjects"].values == 4
    if idx4.sum() >= 2:
        _, m4, s4, n4 = pair_mean_sd(cort[idx4])
        lines.append(f"4-subject-only:  N={n4}, mean r = {m4:.3f} ± {s4:.3f}")

    # per-season (A3)
    lines.append("\nPer-season stability:")
    for s in sorted(episodes["episode"].str[:3].unique()):
        m = episodes["episode"].str[:3].values == s
        if m.sum() >= 2:
            _, ms, ss, ns = pair_mean_sd(cort[m])
            lines.append(f"  {s}: N={ns:3d}, mean r = {ms:.3f} ± {ss:.3f}")
    return pairs, mean_r, sd_r, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    A_FIG_DIR.mkdir(parents=True, exist_ok=True)
    episodes = pd.read_csv(A_ISC_FMRIPREP_DIR / "episodes.csv")
    allisc   = np.load(A_ISC_FMRIPREP_DIR / "isc_all.npy")     # (n_ep, 1032)
    cort     = allisc[:, :N_PARCELS]                           # cortical
    eps      = list(episodes["episode"])
    print(f"Loaded {cort.shape[0]} episodes × {cort.shape[1]} cortical parcels")

    # ── stability stats (always cheap; write text) ──
    pairs, mean_r, sd_r, report = stats_and_stability(cort, episodes)
    (A_FIG_DIR / "stability_stats.txt").write_text(report + "\n")
    print(report)

    from nltools.stats import fisher_r_to_z, fisher_z_to_r
    import nilearn
    from nilearn import plotting, datasets

    # ── 1A: overall ISC map ──
    out_a = A_FIG_DIR / "fig_overall_isc.png"
    if args.force or not out_a.exists():
        mask = _roi_mask()
        avg  = fisher_z_to_r(np.mean(fisher_r_to_z(cort), axis=0))
        fig = plt.figure(figsize=(10, 2.5))
        plotting.plot_stat_map(_to_nifti(avg, mask), bg_img=str(MNI_BG_IMG),
                               threshold=VIZ_THRESHOLD_ISC, vmax=0.4, draw_cross=False,
                               display_mode="xz", black_bg=True, annotate=False, figure=fig)
        fig.savefig(out_a, dpi=150, bbox_inches="tight", facecolor="black"); plt.close(fig)
        print(f"Saved: {out_a}")

    # ── 1C: Season-1 small multiples (a-parts) ──
    out_c = A_FIG_DIR / "fig_episode_maps_s1.png"
    if args.force or not out_c.exists():
        mask = _roi_mask()
        s1 = [i for i, e in enumerate(eps) if e.startswith("s01") and e.endswith("a")]
        ncols = 6; nrows = (len(s1) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
        axes = np.atleast_1d(axes).flatten()
        for k, ei in enumerate(s1):
            plotting.plot_stat_map(_to_nifti(cort[ei], mask), bg_img=datasets.load_mni152_template(),
                                   threshold=0.03, vmax=0.4, draw_cross=False, display_mode="z",
                                   cut_coords=[10], annotate=False, axes=axes[k], colorbar=False)
            axes[k].set_title(eps[ei], fontsize=7)
        for ax in axes[len(s1):]:
            ax.axis("off")
        fig.suptitle("Season 1 — per-episode ISC maps (fMRIPrep)", fontsize=12)
        fig.tight_layout(); fig.savefig(out_c, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {out_c}")

    # ── 1D: spatial scatter, two representative episodes ──
    out_d = A_FIG_DIR / "fig_spatial_scatter.png"
    if args.force or not out_d.exists():
        e1, e2 = "s01e02a", "s02e01a"
        i1 = eps.index(e1) if e1 in eps else 0
        i2 = eps.index(e2) if e2 in eps else len(eps) // 2
        r = np.corrcoef(cort[i1], cort[i2])[0, 1]
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(cort[i1], cort[i2], color="gray", s=8, alpha=0.6)
        ax.set_xlim([-0.1, 0.7]); ax.set_ylim([-0.1, 0.7])
        ax.set_xlabel(f"ISC {eps[i1]}"); ax.set_ylabel(f"ISC {eps[i2]}")
        ax.text(0.05, 0.92, f"r = {r:.3f}", transform=ax.transAxes, fontsize=10)
        sns.despine(); fig.tight_layout(); fig.savefig(out_d, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {out_d}")

    # ── 1E: stability histogram ──
    out_e = A_FIG_DIR / "fig_stability_hist.png"
    if args.force or not out_e.exists():
        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(pairs, bins=40, color="steelblue", edgecolor="white")
        ax.axvline(mean_r, color="red", linestyle="--")
        ax.set_xlabel("Spatial ISC correlation (r)"); ax.set_ylabel("Episode pairs")
        ax.set_title(f"Spatial stability: mean r = {mean_r:.3f} ± {sd_r:.3f}  (N={cort.shape[0]})")
        sns.despine(); fig.tight_layout(); fig.savefig(out_e, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {out_e}")

    print(f"\nFigures: {A_FIG_DIR}")


if __name__ == "__main__":
    main()
