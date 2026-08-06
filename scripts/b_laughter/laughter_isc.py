"""
laughter_isc.py — laughter vs non-laughter ISC, WHOLE BRAIN (Block B core, B1)
==============================================================================
Ported from 03_REVISION/scripts/13_laughter_isc_unified.py and extended to the
whole brain (all 1032 ROIs), so the regions modulated by laughter are found
data-driven rather than assumed — the same logic by which rTPJ originally stood
out. fMRIPrep pipeline; laughter annotations from the revised classifier Clf-C.

METHOD (faithful to original 02_LaughterISC_HRFShift_FIXEDEFFECT.ipynb)
  per episode:
    1. Clf-C laughter vector (ls); shift ls=1 onsets +HRF_SHIFT_TRS (+3 TRs)
    2. segment: laughter windows (4 TRs) / non-laughter windows (≥4 TRs)
    3. per ROI, pairwise ISC = MEAN of C(4,2) pairwise correlations
       (original metric='mean'; Block A overall-ISC uses median — each faithful)
  aggregate across episodes:
    4. Fisher r→z, paired t-test (laughter vs non) per parcel, FDR across 1032

OUTPUTS  (data/b_laughter/)
  isc_laugh_by_parcel.npy / isc_nolaugh_by_parcel.npy   (n_episodes × 1032)
  episodes.csv
  wholebrain_contrast.csv   roi_idx, label, Δr, t, p, p_fdr, sig  (all 1032)
  isc_summary.csv           5-ROI readout (rTPJ/visual/auditory single parcels;
                            dorsal/ventral striatum = averaged-timeseries ROIs)
FIGURES  (results/analysis_plots/b_laughter/)
  fig_laughter_isc.png              5-ROI bar figure (primary readout)
  fig_wholebrain_contrast.png       cortical t-map, FDR-thresholded, two-sided
  exploratory/fig_seasons.png

Usage
  python laughter_isc.py            # skip if wholebrain_contrast.csv exists
  python laughter_isc.py --force
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from nltools.stats import fisher_r_to_z, fisher_z_to_r

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, B_FIG_DIR,
    ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY, DORSAL_STR_IDX, VENTRAL_STR_IDX,
    HRF_SHIFT_TRS, EVENT_DURATION_TRS, MIN_NONEVENT_DURATION,
    EXCLUDED_EPISODES, FDR_ALPHA, N_PARCELS, N_ROIS_UNIFIED, VIZ_THRESHOLD_ISC,
)

# single-parcel cortical ROIs (readout) + multi-index striatum ROIs (averaged ts)
CORTICAL_ROIS = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}
STRIATUM_ROIS = {"dorsal_str": DORSAL_STR_IDX, "ventral_str": VENTRAL_STR_IDX}
ROI_LABELS = {"rTPJ": "rTPJ", "visual_cortex": "LOC /\nVisual", "auditory_cortex": "Auditory\nCortex",
              "dorsal_str": "Dorsal\nStriatum", "ventral_str": "Ventral\nStriatum"}
ROI_COLORS = {"rTPJ": "#27AE60", "visual_cortex": "#2980B9", "auditory_cortex": "#D35400",
              "dorsal_str": "#8E44AD", "ventral_str": "#C0392B"}


# ── segmentation (exact original while-loop) ────────────────────────────────────
def extract_event_segments(ev, event_duration=EVENT_DURATION_TRS,
                           min_non_event_duration=MIN_NONEVENT_DURATION):
    n = len(ev); laugh, nonl = [], []; i = 0; ne = None
    while i < n:
        if ev[i] == 1:
            if ne is not None and (i - ne) >= min_non_event_duration:
                nonl.append((ne, i))
            ne = None; off = min(i + event_duration, n); laugh.append((i, off)); i = off
        else:
            if ne is None:
                ne = i
            i += 1
    if ne is not None and (n - ne) >= min_non_event_duration:
        nonl.append((ne, n))
    return laugh, nonl


def concat_segments(data, segs):
    if not segs:
        return None
    parts = [data[:, o:off + 1, :] for o, off in segs if off + 1 > o]
    return np.concatenate(parts, axis=1) if parts else None


def pairwise_mean_isc_allparcels(seg):
    """seg (n_sub, T, P) → (P,) mean of pairwise correlations.
    A subject counts only if it has NO NaN in the segment (matches the per-pair
    NaN skip of pairwise_isc_1d, so partial subjects don't poison every parcel)."""
    present = [i for i in range(seg.shape[0]) if not np.isnan(seg[i]).any()]
    if len(present) < 2:
        return np.full(seg.shape[2], np.nan)
    s = seg[present]
    z = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-12)
    pw = [(z[i] * z[j]).mean(axis=0) for i, j in combinations(range(len(present)), 2)]
    return np.mean(pw, axis=0)


def pairwise_isc_1d(ts):
    """Mean of pairwise correlations for a single averaged-ROI timeseries."""
    rs = []
    for i, j in combinations(range(ts.shape[0]), 2):
        a, b = ts[i], ts[j]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def run_all_episodes():
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    laugh_p, nol_p, rows, skipped = [], [], [], 0
    for ep in eps:
        ann = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
        if not ann.exists():
            skipped += 1; continue
        data = np.load(PREP_TIMESERIES_DIR / f"task-{ep}.npy")
        lv = pd.read_csv(ann)["ls"].values
        n = min(data.shape[1], len(lv)); data, lv = data[:, :n, :], lv[:n]
        sh = np.where(lv == 1)[0] + HRF_SHIFT_TRS; ls = np.zeros(n, int); ls[sh[sh < n]] = 1
        l_seg, nl_seg = extract_event_segments(ls)
        ld, nld = concat_segments(data, l_seg), concat_segments(data, nl_seg)
        if ld is None or nld is None or ld.shape[1] < 4 or nld.shape[1] < 4:
            skipped += 1; continue

        laugh_p.append(pairwise_mean_isc_allparcels(ld))     # (1032,)
        nol_p.append(pairwise_mean_isc_allparcels(nld))
        row = {"episode": ep, "season": int(ep[1:3]),
               "n_laugh_trs": ld.shape[1], "n_nolaugh_trs": nld.shape[1]}
        for name, idxs in STRIATUM_ROIS.items():
            row[f"isc_laugh_{name}"]   = pairwise_isc_1d(ld[:, :, idxs].mean(axis=2))
            row[f"isc_nolaugh_{name}"] = pairwise_isc_1d(nld[:, :, idxs].mean(axis=2))
        rows.append(row)
    print(f"  Episodes processed: {len(rows)}  skipped: {skipped}")
    return np.array(laugh_p), np.array(nol_p), pd.DataFrame(rows)


def paired_contrast(laugh_r, nol_r):
    """Column-wise paired t of Fisher-z ISC; returns (delta_r, t, p) arrays."""
    lz, nz = fisher_r_to_z(laugh_r), fisher_r_to_z(nol_r)
    t, p = stats.ttest_rel(lz, nz, axis=0, nan_policy="omit")
    delta = fisher_z_to_r(np.nanmean(lz - nz, axis=0))
    return np.asarray(delta), np.asarray(t), np.asarray(p)


def wholebrain_table(laugh_p, nol_p):
    labels = pd.read_csv(PREP_TIMESERIES_DIR / "roi_labels.csv")["label"].tolist()
    delta, t, p = paired_contrast(laugh_p, nol_p)
    df = pd.DataFrame({"roi_idx": np.arange(N_ROIS_UNIFIED), "label": labels,
                       "delta_r": delta, "t": t, "p": p})
    ok = df["p"].notna()
    df["p_fdr"] = np.nan
    df.loc[ok, "p_fdr"] = fdrcorrection(df.loc[ok, "p"].values, alpha=FDR_ALPHA)[1]
    df["sig"] = df["p_fdr"].apply(lambda x: "" if pd.isna(x) else
                                  ("***" if x < .001 else "**" if x < .01 else "*" if x < .05 else "ns"))
    return df


def roi_summary(wb, ep_df, laugh_p, nol_p):
    """5-ROI readout: cortical from whole-brain rows; striatum from averaged-ts columns."""
    rows = []
    for name, idx in CORTICAL_ROIS.items():
        d, t, p = paired_contrast(laugh_p[:, [idx]], nol_p[:, [idx]])
        rows.append({"roi": name, "label": ROI_LABELS[name].replace("\n", " "), "n": len(laugh_p),
                     "mean_isc_laugh": float(fisher_z_to_r(np.nanmean(fisher_r_to_z(laugh_p[:, idx])))),
                     "mean_isc_nolaugh": float(fisher_z_to_r(np.nanmean(fisher_r_to_z(nol_p[:, idx])))),
                     "mean_delta": float(d[0]), "t_stat": float(t[0]), "p_val": float(p[0])})
    for name in STRIATUM_ROIS:
        pair = ep_df[[f"isc_laugh_{name}", f"isc_nolaugh_{name}"]].dropna()
        lz = fisher_r_to_z(pair[f"isc_laugh_{name}"].values); nz = fisher_r_to_z(pair[f"isc_nolaugh_{name}"].values)
        t, p = stats.ttest_rel(lz, nz)
        rows.append({"roi": name, "label": ROI_LABELS[name].replace("\n", " "), "n": len(pair),
                     "mean_isc_laugh": float(fisher_z_to_r(lz.mean())), "mean_isc_nolaugh": float(fisher_z_to_r(nz.mean())),
                     "mean_delta": float(fisher_z_to_r((lz - nz).mean())), "t_stat": float(t), "p_val": float(p)})
    df = pd.DataFrame(rows)
    df["sig"] = df["p_val"].apply(lambda x: "***" if x < .001 else "**" if x < .01 else "*" if x < .05 else "ns")
    return df


# ── figures ─────────────────────────────────────────────────────────────────────
def fig_roi_bars(df_sum):
    cortical, subcort = ["rTPJ", "visual_cortex", "auditory_cortex"], ["dorsal_str", "ventral_str"]
    fig = plt.figure(figsize=(14, 6)); fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[3, 2], wspace=0.35)

    def _panel(ax, roi_list, title):
        w = 0.35
        for xi, rname in enumerate(roi_list):
            r = df_sum[df_sum.roi == rname].iloc[0]; col = ROI_COLORS[rname]
            ax.bar(xi - w/2, r.mean_isc_laugh, w, color=col, alpha=.88, label="Laughter" if xi == 0 else "")
            ax.bar(xi + w/2, r.mean_isc_nolaugh, w, color=col, alpha=.28, hatch="///",
                   label="Non-laughter" if xi == 0 else "")
            ymax = max(r.mean_isc_laugh, r.mean_isc_nolaugh) + 0.012
            ax.text(xi, ymax, r.sig, ha="center", fontsize=12 if r.sig != "ns" else 9,
                    fontweight="bold" if r.sig != "ns" else "normal", color="black" if r.sig != "ns" else "gray")
        ax.set_xticks(range(len(roi_list))); ax.set_xticklabels([ROI_LABELS[r] for r in roi_list], fontsize=11)
        ax.set_ylabel("Mean pairwise ISC (r)"); ax.axhline(0, color="black", lw=.6)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left"); ax.legend(fontsize=9)

    n_ep = int(df_sum["n"].median())
    _panel(fig.add_subplot(gs[0]), cortical, f"A   ISC — Cortical ROIs (N={n_ep}, fMRIPrep, Clf-C)")
    _panel(fig.add_subplot(gs[1]), subcort,  f"B   ISC — Striatum (N={n_ep}, fMRIPrep, Clf-C)")
    out = B_FIG_DIR / "fig_laughter_isc.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  Saved {out.name}")


def fig_wholebrain(wb):
    """Cortical t-map, FDR-thresholded, two-sided."""
    import nilearn
    from nltools.data import Brain_Data
    from nltools.mask import expand_mask, roi_to_brain
    from nilearn import plotting
    sch = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=1)
    mask = expand_mask(Brain_Data(sch["maps"]))

    t = wb["t"].values[:N_PARCELS].copy()
    sig = (wb["p_fdr"].values[:N_PARCELS] < FDR_ALPHA)
    t_masked = np.where(sig, t, 0.0)
    n_pos, n_neg = int(((t_masked > 0)).sum()), int((t_masked < 0).sum())
    nii = roi_to_brain(pd.Series(t_masked), mask).to_nifti()

    fig = plt.figure(figsize=(13, 3))
    plotting.plot_stat_map(nii, bg_img=nilearn.datasets.load_mni152_template(),
                           threshold=0.01, display_mode="z", cut_coords=7, colorbar=True,
                           title=f"Laughter − non-laughter ISC (t, FDR<{FDR_ALPHA}): "
                                 f"{n_pos} parcels laughter>non, {n_neg} non>laughter",
                           figure=fig)
    out = B_FIG_DIR / "fig_wholebrain_contrast.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {out.name}  ({n_pos} pos, {n_neg} neg FDR-sig cortical parcels)")


def fig_seasons(ep_df, laugh_p, nol_p):
    (B_FIG_DIR / "exploratory").mkdir(parents=True, exist_ok=True)
    order = ["rTPJ", "visual_cortex", "auditory_cortex", "dorsal_str", "ventral_str"]
    fig, axes = plt.subplots(1, len(order), figsize=(18, 4)); fig.patch.set_facecolor("white")
    fig.suptitle("Season-by-season laughter Δz (fMRIPrep, Clf-C)", fontsize=12, fontweight="bold")
    for c, rname in enumerate(order):
        if rname in CORTICAL_ROIS:
            idx = CORTICAL_ROIS[rname]
            dz = fisher_r_to_z(laugh_p[:, idx]) - fisher_r_to_z(nol_p[:, idx])
        else:
            dz = (fisher_r_to_z(ep_df[f"isc_laugh_{rname}"].values.astype(float)) -
                  fisher_r_to_z(ep_df[f"isc_nolaugh_{rname}"].values.astype(float)))
        s = pd.DataFrame({"season": ep_df["season"].values, "dz": dz})
        g = s.groupby("season")["dz"]; m = g.mean().reindex(range(1, 7)); e = g.sem().reindex(range(1, 7))
        axes[c].bar(range(1, 7), m.values, color=ROI_COLORS[rname], alpha=.8, yerr=e.values, capsize=3)
        axes[c].axhline(0, color="black", lw=.7); axes[c].set_xticks(range(1, 7)); axes[c].set_xlabel("Season")
        axes[c].set_title(ROI_LABELS[rname].replace("\n", " "), fontsize=9, fontweight="bold")
        if c == 0:
            axes[c].set_ylabel("Δz (laughter − non)")
    fig.tight_layout(); out = B_FIG_DIR / "exploratory" / "fig_seasons.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  Saved exploratory/{out.name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True); B_FIG_DIR.mkdir(parents=True, exist_ok=True)
    wb_path = B_DIR / "wholebrain_contrast.csv"
    if wb_path.exists() and not args.force:
        print(f"Output exists (use --force): {wb_path}")
        return

    laugh_p, nol_p, ep_df = run_all_episodes()
    np.save(B_DIR / "isc_laugh_by_parcel.npy", laugh_p.astype(np.float32))
    np.save(B_DIR / "isc_nolaugh_by_parcel.npy", nol_p.astype(np.float32))
    ep_df.to_csv(B_DIR / "episodes.csv", index=False)

    wb = wholebrain_table(laugh_p, nol_p); wb.to_csv(wb_path, index=False)
    df_sum = roi_summary(wb, ep_df, laugh_p, nol_p); df_sum.to_csv(B_DIR / "isc_summary.csv", index=False)

    n_sig = (wb["p_fdr"] < FDR_ALPHA).sum()
    pos = wb[(wb.p_fdr < FDR_ALPHA) & (wb.delta_r > 0)].sort_values("t", ascending=False)
    neg = wb[(wb.p_fdr < FDR_ALPHA) & (wb.delta_r < 0)].sort_values("t")
    print(f"\nWhole-brain: {n_sig}/{N_ROIS_UNIFIED} parcels FDR-significant "
          f"({len(pos)} laughter>non, {len(neg)} non>laughter)")
    print("\nTop 8 laughter > non-laughter:")
    print(pos.head(8)[["roi_idx", "label", "delta_r", "t", "p_fdr"]].to_string(index=False))
    print("\nTop 8 non-laughter > laughter:")
    print(neg.head(8)[["roi_idx", "label", "delta_r", "t", "p_fdr"]].to_string(index=False))

    print("\n5-ROI readout:")
    print(df_sum[["roi", "n", "mean_isc_laugh", "mean_isc_nolaugh", "mean_delta", "t_stat", "p_val", "sig"]]
          .to_string(index=False))

    fig_roi_bars(df_sum); fig_wholebrain(wb); fig_seasons(ep_df, laugh_p, nol_p)
    print(f"\nOutputs: {B_DIR}  |  Figures: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
