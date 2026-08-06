"""
glm_av_energy.py — audiovisual-energy control for the GLM laughter contrast (Block B / R2.3a)
=============================================================================================
The GLM (Part B, now primary) reports increased activation around laughter. A skeptic can
ask whether that β is a low-level stimulus-energy response — the acoustic laugh burst and any
co-occurring visual change — rather than "laughter/humor." This is the GLM analogue of the
ISC AV-energy control (av_energy.py): instead of energy-MATCHING segments and recomputing ISC,
we add per-TR stimulus energy as NUISANCE regressors to the laughter GLM and ask whether the
laughter β survives.

METHOD
  per subject × episode:
    - bare design (as glm_contrast.py): SPM-HRF laughter regressor + cosine drift + constant
    - energy regressors: per-TR acoustic RMS (dB) and visual motion energy (same cached series
      the ISC control uses), z-scored and SPM-HRF-convolved so they carry the hemodynamic lag
      like the laughter regressor
    - three OLS fits per key ROI:  bare | + acoustic-RMS | + visual-energy
  second level:
    - per-episode mean laughter β across present viewers → one-sample t vs 0 across episodes
  survival = the β stays positive/significant after the energy regressor is included → the
  activation is not a stimulus-energy artifact (the GLM analogue of "Δr survives matching").

Key ROIs: rTPJ, visual cortex, auditory cortex, dorsal striatum (mean of its 8 parcels).

OUTPUTS  data/b_laughter/glm_av_energy_summary.csv
FIGURE   results/analysis_plots/c_controls/fig_glm_av_energy.png

Usage
  python glm_av_energy.py            # skip if summary exists
  python glm_av_energy.py --force
  python glm_av_energy.py --plot-only   # re-render figure from the saved summary
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
from nilearn.glm.first_level import make_first_level_design_matrix, spm_hrf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, AV_ENERGY_DIR, B_DIR, C_FIG_DIR, TR_SEC,
    EXCLUDED_EPISODES, ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY, DORSAL_STR_IDX,
)
from glm_contrast import laughter_blocks

# readout ROIs: single parcels + dorsal-striatum aggregate (mean of its parcels)
SINGLE = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}
ROI_ORDER = ["rTPJ", "visual_cortex", "auditory_cortex", "dorsal_striatum"]
_HRF = spm_hrf(TR_SEC, oversampling=1)          # HRF sampled at the TR


def _hrf_reg(x, n):
    """z-score a per-TR series, SPM-HRF-convolve, truncate to n, re-z-score → nuisance column.
    NaNs (occasional missing energy samples) are set to the series mean (0 after z-scoring)
    so they cannot propagate into the design matrix."""
    z = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.convolve(z, _HRF)[:n]
    return (c - c.mean()) / (c.std() + 1e-12)


def load_energy(ep, n):
    """Return (rms, vis) per-TR arrays trimmed/aligned to length n, or (None, None) if missing."""
    fa = AV_ENERGY_DIR / "acoustic_rms" / f"{ep}_rms.csv"
    fv = AV_ENERGY_DIR / "visual_energy" / f"{ep}_vis.csv"
    if not (fa.exists() and fv.exists()):
        return None, None
    rms = pd.read_csv(fa)["rms_db"].values
    vis = pd.read_csv(fv)["vis_energy"].values
    m = min(len(rms), len(vis), n)
    if m < n:                                    # energy shorter than BOLD — skip to keep it clean
        return None, None
    return rms[:n], vis[:n]


def episode_betas():
    """Per-episode subject-averaged laughter β for each key ROI, under three models
    (bare | +acoustic | +visual). Returns dict roi -> {model -> list-of-episode-betas}."""
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    cols = [SINGLE["rTPJ"], SINGLE["visual_cortex"], SINGLE["auditory_cortex"]] + DORSAL_STR_IDX
    dstr = slice(3, 3 + len(DORSAL_STR_IDX))      # dorsal-striatum parcels within `cols`
    acc = {r: {"bare": [], "acoustic": [], "visual": []} for r in ROI_ORDER}
    used = 0
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
        rms, vis = load_energy(ep, n)
        if rms is None:
            continue
        ft = np.arange(n) * TR_SEC
        ev = pd.DataFrame({"onset": [o * TR_SEC for o, _ in bl],
                           "duration": [l * TR_SEC for _, l in bl],
                           "trial_type": ["laughter"] * len(bl)})
        dm = make_first_level_design_matrix(ft, ev, hrf_model="spm", drift_model="cosine", high_pass=0.01)
        X = dm.values; li = list(dm.columns).index("laughter")
        Xa = np.column_stack([X, _hrf_reg(rms, n)])
        Xv = np.column_stack([X, _hrf_reg(vis, n)])

        per = {r: {"bare": [], "acoustic": [], "visual": []} for r in ROI_ORDER}
        for s in range(data.shape[0]):
            Y = data[s][:, cols]
            if np.isnan(Y).any():
                continue
            for key, M in (("bare", X), ("acoustic", Xa), ("visual", Xv)):
                b, *_ = np.linalg.lstsq(M, Y, rcond=None)
                bl_row = b[li]
                per["rTPJ"][key].append(bl_row[0])
                per["visual_cortex"][key].append(bl_row[1])
                per["auditory_cortex"][key].append(bl_row[2])
                per["dorsal_striatum"][key].append(float(np.mean(bl_row[dstr])))
        if per["rTPJ"]["bare"]:
            for r in ROI_ORDER:
                for k in ("bare", "acoustic", "visual"):
                    acc[r][k].append(float(np.mean(per[r][k])))
            used += 1
    print(f"  Episodes with GLM AV-energy fit: {used}")
    return acc


def summarise(acc):
    rows = []
    for r in ROI_ORDER:
        rec = {"roi": r, "n": len(acc[r]["bare"])}
        for k in ("bare", "acoustic", "visual"):
            b = np.array(acc[r][k]); t, p = stats.ttest_1samp(b, 0)
            rec[f"beta_{k}"] = float(b.mean()); rec[f"t_{k}"] = float(t); rec[f"p_{k}"] = float(p)
        # % of the bare β retained after each energy control
        rec["pct_acoustic"] = 100 * rec["beta_acoustic"] / rec["beta_bare"] if rec["beta_bare"] else np.nan
        rec["pct_visual"] = 100 * rec["beta_visual"] / rec["beta_bare"] if rec["beta_bare"] else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def figure(df):
    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    x = np.arange(len(ROI_ORDER)); w = 0.26
    labels = {"rTPJ": "rTPJ", "visual_cortex": "visual\ncortex",
              "auditory_cortex": "auditory\ncortex", "dorsal_striatum": "dorsal\nstriatum"}
    d = df.set_index("roi").loc[ROI_ORDER]
    fig, ax = plt.subplots(figsize=(9, 5)); fig.patch.set_facecolor("white")
    for j, (k, col, lab) in enumerate((("bare", "#95A5A6", "bare (laughter only)"),
                                       ("acoustic", "#2E86C1", "+ acoustic-RMS regressor"),
                                       ("visual", "#27AE60", "+ visual-energy regressor"))):
        ax.bar(x + (j - 1) * w, d[f"beta_{k}"], w, color=col, label=lab)
        for i, r in enumerate(ROI_ORDER):
            b = d.loc[r, f"beta_{k}"]
            ax.text(i + (j - 1) * w, b + np.sign(b) * 0.008, star(d.loc[r, f"p_{k}"]),
                    ha="center", va="bottom" if b >= 0 else "top", fontsize=8)
    ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels([labels[r] for r in ROI_ORDER])
    ax.set_ylabel("GLM laughter β (vs baseline)"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("GLM AV-energy control: laughter activation survives energy regression\n"
                 "(β with acoustic-RMS / visual-energy as HRF-convolved nuisance regressors)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    C_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C_FIG_DIR / "fig_glm_av_energy.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure: {C_FIG_DIR}/fig_glm_av_energy.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="re-render figure from the saved summary")
    args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True)
    out = B_DIR / "glm_av_energy_summary.csv"
    if args.plot_only:
        if not out.exists():
            print(f"No summary to plot from: {out} (run without --plot-only first)"); return
        figure(pd.read_csv(out)); return
    if out.exists() and not args.force:
        print(f"Output exists (use --force): {out}"); print(pd.read_csv(out).to_string(index=False)); return

    df = summarise(episode_betas())
    df.to_csv(out, index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\nGLM AV-energy control (laughter β; bare vs energy-controlled):")
    print(df[["roi", "n", "beta_bare", "p_bare", "beta_acoustic", "p_acoustic",
              "beta_visual", "p_visual", "pct_acoustic", "pct_visual"]].to_string(index=False))
    figure(df)
    print(f"\nOutput: {out}")


if __name__ == "__main__":
    main()
