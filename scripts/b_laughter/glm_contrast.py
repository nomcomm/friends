"""
glm_contrast.py — whole-brain GLM laughter contrast, PARCEL level (Block B, R1.1)
=================================================================================
Ported/rebuilt from 03_REVISION/scripts/06_glm_contrast.py + 06_glm_aggregate.py,
run on the fMRIPrep parcellated timeseries + Clf-C annotations (same data as the
ISC analysis, so GLM activation and ISC synchrony are directly comparable).

METHOD
  per subject × episode:
    - laughter blocks (Clf-C ls, contiguous runs) → events (onset/duration, sec)
    - nilearn design matrix: SPM-HRF-convolved 'laughter' regressor + cosine drift
      (HRF convolution models the hemodynamic lag — no manual TR shift needed)
    - OLS per parcel (all 1032) → laughter beta   [BARE model: laughter + drift]
  second level:
    - per-episode mean beta across present subjects
    - one-sample t vs 0 across episodes, FDR across 1032 parcels

OUTPUTS  (data/b_laughter/)
  glm_contrast.csv                  roi_idx, label, beta, t, p, p_fdr, sig  (all 1032)
  glm_beta_by_episode.npy           (n_episodes × 1032) per-episode mean betas
  glm_beta_by_subject_episode.npy   (n_episodes × N_SUBJECTS × 1032) per-viewer betas,
                                    NaN where a viewer is absent — the single
                                    authoritative per-viewer GLM fit; consumed by
                                    glm_per_viewer.py (within-subject maps, R1.1)
FIGURE   (results/analysis_plots/b_laughter/)
  fig_glm_contrast.png      cortical t-map, FDR-thresholded, two-sided

Usage
  python glm_contrast.py            # skip if glm_contrast.csv exists
  python glm_contrast.py --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from nilearn.glm.first_level import make_first_level_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, B_FIG_DIR, TR_SEC,
    EXCLUDED_EPISODES, N_PARCELS, N_ROIS_UNIFIED, FDR_ALPHA,
    ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY,
)

READOUT = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}


def laughter_blocks(ls):
    """Contiguous ls==1 runs → list of (onset_TR, length_TR)."""
    out, i, n = [], 0, len(ls)
    while i < n:
        if ls[i] == 1:
            j = i
            while j < n and ls[j] == 1:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def episode_betas():
    """Per-episode laughter beta — subject-averaged AND per-viewer.
    out  : (n_ep, 1032) mean beta across present viewers  — group second level
    subj : (n_ep, N_SUBJECTS, 1032) per-viewer beta, NaN where a viewer is absent
           (viewer axis order = config.SUBJECTS). Feeds glm_per_viewer.py (R1.1).
    The group path (`out`) is unchanged, so glm_contrast.csv stays identical."""
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    rows, out, subj = [], [], []
    for ep in eps:
        ann = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
        if not ann.exists():
            continue
        data = np.load(PREP_TIMESERIES_DIR / f"task-{ep}.npy")
        ls = pd.read_csv(ann)["ls"].values
        n = min(data.shape[1], len(ls)); data, ls = data[:, :n, :], ls[:n]
        bl = laughter_blocks(ls)
        if not bl:
            continue
        ft = np.arange(n) * TR_SEC
        ev = pd.DataFrame({"onset": [o * TR_SEC for o, _ in bl],
                           "duration": [l * TR_SEC for _, l in bl],
                           "trial_type": ["laughter"] * len(bl)})
        dm = make_first_level_design_matrix(ft, ev, hrf_model="spm",
                                            drift_model="cosine", high_pass=0.01)
        X, li = dm.values, list(dm.columns).index("laughter")
        sub = []
        sub_full = np.full((data.shape[0], data.shape[2]), np.nan, dtype=np.float32)
        for s in range(data.shape[0]):
            Y = data[s]
            if np.isnan(Y).any():          # skip absent/partial subject
                continue
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            sub.append(beta[li]); sub_full[s] = beta[li]
        if sub:
            out.append(np.mean(sub, axis=0)); subj.append(sub_full); rows.append(ep)
    print(f"  Episodes with GLM: {len(rows)}")
    return np.array(out), np.array(subj), rows


def season_maps_and_stability(B, eps):
    """Season-averaged beta contrast maps + cross-episode/-season spatial stability.
    Given N=4 viewers, robustness = consistency across stimuli (episodes/seasons),
    mirroring the Block A overall-ISC stability logic."""
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain
    from nilearn import plotting

    seasons = np.array([int(e[1:3]) for e in eps])
    cort = B[:, :N_PARCELS]

    def pairwise_r(mat):
        rr = np.corrcoef(mat)
        return rr[np.triu_indices(mat.shape[0], k=1)]

    ep_pairs = pairwise_r(cort)
    lines = ["GLM laughter-contrast spatial stability (β maps, 1000 cortical parcels)",
             "=" * 60,
             f"Across episodes: N={len(cort)}, mean pairwise spatial r = "
             f"{ep_pairs.mean():.3f} ± {ep_pairs.std():.3f}"]
    sea_ids = sorted(set(seasons))
    sea_maps = np.array([cort[seasons == s].mean(0) for s in sea_ids])
    sea_pairs = pairwise_r(sea_maps)
    lines.append(f"Across the {len(sea_ids)} season-averaged maps: mean pairwise r = "
                 f"{sea_pairs.mean():.3f} ± {sea_pairs.std():.3f}")
    lines.append("\nPer-season N episodes: " + ", ".join(f"s{s:02d}={int((seasons==s).sum())}" for s in sea_ids))
    (B_DIR / "glm_stability_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))

    # small-multiples: season-averaged contrast maps
    mask = expand_mask(Brain_Data(nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=1000, yeo_networks=7, resolution_mm=1)["maps"]))
    bg = nilearn.datasets.load_mni152_template()
    vmax = float(np.percentile(np.abs(sea_maps), 99))
    fig, axes = plt.subplots(len(sea_ids), 1, figsize=(13, 2.2 * len(sea_ids)))
    for ax, s, m in zip(np.atleast_1d(axes), sea_ids, sea_maps):
        nii = roi_to_brain(pd.Series(m), mask).to_nifti()
        plotting.plot_stat_map(nii, bg_img=bg, threshold=vmax * 0.15, vmax=vmax,
                               display_mode="z", cut_coords=6, colorbar=True,
                               axes=ax, title=f"Season {s}  (N={int((seasons==s).sum())} episodes)")
    fig.suptitle(f"GLM laughter contrast (β) by season — cross-season spatial r = {sea_pairs.mean():.3f}",
                 fontsize=13, fontweight="bold")
    fig.savefig(B_FIG_DIR / "fig_glm_seasons.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig_glm_seasons.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True); B_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = B_DIR / "glm_contrast.csv"
    if out_csv.exists() and not args.force:
        print(f"Output exists (use --force): {out_csv}"); return

    B, B_subj, eps = episode_betas()
    np.save(B_DIR / "glm_beta_by_episode.npy", B.astype(np.float32))
    np.save(B_DIR / "glm_beta_by_subject_episode.npy", B_subj.astype(np.float32))

    t, p = stats.ttest_1samp(B, 0, axis=0)
    labels = pd.read_csv(PREP_TIMESERIES_DIR / "roi_labels.csv")["label"].tolist()
    df = pd.DataFrame({"roi_idx": np.arange(N_ROIS_UNIFIED), "label": labels,
                       "beta": B.mean(0), "t": t, "p": p})
    ok = df["p"].notna(); df["p_fdr"] = np.nan
    df.loc[ok, "p_fdr"] = fdrcorrection(df.loc[ok, "p"].values, alpha=FDR_ALPHA)[1]
    df["sig"] = df["p_fdr"].apply(lambda x: "" if pd.isna(x) else
                                  ("***" if x < .001 else "**" if x < .01 else "*" if x < .05 else "ns"))
    df.to_csv(out_csv, index=False)

    n_sig = int((df["p_fdr"] < FDR_ALPHA).sum())
    print(f"\nWhole-brain GLM: {n_sig}/{N_ROIS_UNIFIED} parcels FDR-significant")
    print("\nROI readout (activation contrast, laughter vs baseline):")
    print(f"{'ROI':<16}{'beta':>9}{'t':>8}{'p_fdr':>10}{'sig':>5}")
    for name, idx in READOUT.items():
        r = df.iloc[idx]
        print(f"{name:<16}{r.beta:>9.4f}{r.t:>8.2f}{r.p_fdr:>10.2e}{r.sig:>5}")

    # cortical t-map, FDR-thresholded, two-sided
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain
    from nilearn import plotting
    mask = expand_mask(Brain_Data(nilearn.datasets.fetch_atlas_schaefer_2018(
        n_rois=1000, yeo_networks=7, resolution_mm=1)["maps"]))
    tt = df["t"].values[:N_PARCELS].copy()
    sig = df["p_fdr"].values[:N_PARCELS] < FDR_ALPHA
    tmask = np.where(sig, tt, 0.0)
    npos, nneg = int((tmask > 0).sum()), int((tmask < 0).sum())
    nii = roi_to_brain(pd.Series(tmask), mask).to_nifti()
    fig = plt.figure(figsize=(13, 3))
    plotting.plot_stat_map(nii, bg_img=nilearn.datasets.load_mni152_template(),
                           threshold=0.01, display_mode="z", cut_coords=7, colorbar=True,
                           title=f"GLM laughter contrast (t, FDR<{FDR_ALPHA}): "
                                 f"{npos} activated, {nneg} deactivated", figure=fig)
    fig.savefig(B_FIG_DIR / "fig_glm_contrast.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Saved fig_glm_contrast.png ({npos} activated, {nneg} deactivated cortical)")

    pd.DataFrame({"episode": eps, "season": [int(e[1:3]) for e in eps]}).to_csv(
        B_DIR / "glm_episodes.csv", index=False)
    season_maps_and_stability(B, eps)
    print(f"\nOutputs: {B_DIR}  |  Figures: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
