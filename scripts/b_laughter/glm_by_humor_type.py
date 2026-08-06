"""
glm_by_humor_type.py — does laughter-associated ACTIVATION depend on humor TYPE? (R2.3b)
========================================================================================
GLM analogue of isc_by_humor_type.py, on the now-primary measure. Reviewer 2 (following
Samson et al. 2008) asks whether theory-of-mind / social jokes recruit the rTPJ more than
wordplay. We reuse the identical buildup-labeling from isc_by_humor_type.py — each laughter
block is labelled by the DOMINANT humor category (Juckel 2016 verbal typology: Language /
Logic / Identity) in the ~30 s (20-TR) setup window before its onset — but instead of ISC
we estimate the laughter-evoked GLM β per category and compare across categories.

Internal-consistency read: a category-general activation (flat across types) shows the
laughter response is not specific to one joke kind; a graded rTPJ (Identity/Logic > Language)
would additionally support the social-cognition interpretation. EXPLORATORY, uncorrected;
power is very unequal (Logic ≫ Language ≫ Identity), so a null for Identity is under-powered.

METHOD  per subject × episode: one HRF-convolved laughter regressor PER present category
        (+ an 'other' regressor for unlabeled laughter, + cosine drift); OLS → per-category
        β for each key ROI. Second level: per-episode mean β across viewers, then a one-way
        comparison across categories (mirrors the ISC test) + Identity-vs-(Language+Logic).
ROIs    rTPJ, visual cortex, auditory cortex, dorsal striatum (mean of its 8 parcels).

OUTPUTS  data/b_laughter/glm_by_humor_type.csv, _stats.txt
FIGURE   results/analysis_plots/c_controls/fig_glm_by_humor_type.png

Usage
  python glm_by_humor_type.py [--force]
  python glm_by_humor_type.py --plot-only
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
from nilearn.glm.first_level import make_first_level_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, B_DIR, C_FIG_DIR, TR_SEC, EXCLUDED_EPISODES,
    ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY, DORSAL_STR_IDX,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "c_controls"))
from isc_by_humor_type import load_episode, blocks, dominant, BUILDUP_WINDOW_TRS, CATEGORIES

SINGLE = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}
ROI_ORDER = ["rTPJ", "visual_cortex", "auditory_cortex", "dorsal_striatum"]
CAT_COLORS = {"Language": "#3498DB", "Logic": "#9B59B6", "Identity": "#E67E22"}
MIN_CAT_TRS = 4                      # skip a category in an episode with fewer laughter TRs (β too noisy)


def episode_category_betas(bold, ls, ish, cat):
    """Return {category: {roi: β}} for one episode (subject-averaged), for categories present."""
    n = bold.shape[1]
    # label each laughter block by its dominant setup-window humor category
    by_cat = {c: [] for c in CATEGORIES}; other = []
    for onset, offset in blocks(ls):
        b0 = max(0, onset - BUILDUP_WINDOW_TRS)
        dom = dominant(cat[b0:onset], ish[b0:onset])
        (by_cat[dom] if dom in CATEGORIES else other).append((onset, offset))
    present = [c for c in CATEGORIES if sum(e - o for o, e in by_cat[c]) >= MIN_CAT_TRS]
    if not present:
        return {}

    rows = []
    for c in present:
        for o, e in by_cat[c]:
            rows.append({"onset": o * TR_SEC, "duration": (e - o) * TR_SEC, "trial_type": c})
    for o, e in other:
        rows.append({"onset": o * TR_SEC, "duration": (e - o) * TR_SEC, "trial_type": "other"})
    dm = make_first_level_design_matrix(np.arange(n) * TR_SEC, pd.DataFrame(rows),
                                        hrf_model="spm", drift_model="cosine", high_pass=0.01)
    X = dm.values
    ci = {c: list(dm.columns).index(c) for c in present}
    cols = [SINGLE["rTPJ"], SINGLE["visual_cortex"], SINGLE["auditory_cortex"]] + DORSAL_STR_IDX
    dstr = slice(3, 3 + len(DORSAL_STR_IDX))

    per = {c: {r: [] for r in ROI_ORDER} for c in present}
    for s in range(bold.shape[0]):
        Y = bold[s][:, cols]
        if np.isnan(Y).any():
            continue
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        for c in present:
            b = beta[ci[c]]
            per[c]["rTPJ"].append(b[0]); per[c]["visual_cortex"].append(b[1])
            per[c]["auditory_cortex"].append(b[2]); per[c]["dorsal_striatum"].append(float(np.mean(b[dstr])))
    out = {}
    for c in present:
        if per[c]["rTPJ"]:
            out[c] = {r: float(np.mean(per[c][r])) for r in ROI_ORDER}
    return out


def run():
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    rows, cov = [], {c: 0 for c in CATEGORIES}
    for ep in eps:
        loaded = load_episode(ep)
        if loaded is None:
            continue
        bold, ls, ish, cat = loaded
        betas = episode_category_betas(bold, ls, ish, cat)
        for c, roivals in betas.items():
            cov[c] += 1
            rows.append({"episode": ep, "season": int(ep[1:3]), "category": c,
                         **{f"beta_{r}": roivals[r] for r in ROI_ORDER}})
    return pd.DataFrame(rows), cov


def summarise(df, cov):
    lines = ["EXPLORATORY: laughter GLM β by humor type (R2.3b) — Clf-C, fMRIPrep, UNCORRECTED",
             f"Buildup window {BUILDUP_WINDOW_TRS} TRs, min {MIN_CAT_TRS} laughter TRs/category/episode",
             "Coverage (episodes contributing each category): "
             + ", ".join(f"{c}={cov[c]}" for c in CATEGORIES), ""]
    for r in ROI_ORDER:
        col = f"beta_{r}"
        by = {c: df[df.category == c][col].dropna().values for c in CATEGORIES}
        means = {c: float(np.mean(by[c])) if len(by[c]) else np.nan for c in CATEGORIES}
        F, p = stats.f_oneway(*[by[c] for c in CATEGORIES if len(by[c]) > 1])
        lines.append(f"[{r}]  means: " + ", ".join(f"{c}={means[c]:+.4f}(n={len(by[c])})" for c in CATEGORIES))
        lines.append(f"       one-way ANOVA F={F:.2f} p={p:.4f}")
        ident, rest = by["Identity"], np.concatenate([by["Language"], by["Logic"]])
        if len(ident) > 1 and len(rest) > 1:
            ti, pi = stats.ttest_ind(ident, rest)
            lines.append(f"       Identity vs (Language+Logic): t={ti:+.2f} p={pi:.4f}")
    return "\n".join(lines)


def figure(df):
    fig, axes = plt.subplots(1, len(ROI_ORDER), figsize=(15, 4.5)); fig.patch.set_facecolor("white")
    labels = {"rTPJ": "rTPJ", "visual_cortex": "visual\ncortex",
              "auditory_cortex": "auditory\ncortex", "dorsal_striatum": "dorsal\nstriatum"}
    for ax, r in zip(axes, ROI_ORDER):
        col = f"beta_{r}"; x = np.arange(len(CATEGORIES))
        means = [df[df.category == c][col].dropna().mean() for c in CATEGORIES]
        sems = [df[df.category == c][col].dropna().sem() for c in CATEGORIES]
        F, p = stats.f_oneway(*[df[df.category == c][col].dropna().values
                                for c in CATEGORIES if df[df.category == c][col].notna().sum() > 1])
        ax.bar(x, means, yerr=sems, capsize=4, color=[CAT_COLORS[c] for c in CATEGORIES], alpha=.9)
        ax.axhline(0, color="k", lw=.5); ax.set_xticks(x); ax.set_xticklabels(CATEGORIES, rotation=15, fontsize=9)
        ax.set_title(f"{labels[r]}\nANOVA p={p:.3f} (uncorr.)", fontsize=10, fontweight="bold")
        if r == ROI_ORDER[0]:
            ax.set_ylabel("GLM laughter β (vs baseline)")
    fig.suptitle("Laughter GLM activation by humor type (buildup-labeled) — EXPLORATORY, uncorrected\n"
                 "internal-consistency check on the primary measure (GLM), not ISC",
                 fontsize=12, fontweight="bold"); fig.tight_layout()
    C_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C_FIG_DIR / "fig_glm_by_humor_type.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure: {C_FIG_DIR}/fig_glm_by_humor_type.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="re-render figure from the saved CSV")
    args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True)
    out = B_DIR / "glm_by_humor_type.csv"
    if args.plot_only:
        if not out.exists():
            print(f"No CSV to plot from: {out} (run without --plot-only first)"); return
        figure(pd.read_csv(out)); return
    if out.exists() and not args.force:
        print((B_DIR / "glm_by_humor_type_stats.txt").read_text()); return

    df, cov = run()
    df.to_csv(out, index=False)
    report = summarise(df, cov)
    (B_DIR / "glm_by_humor_type_stats.txt").write_text(report + "\n")
    print(report)
    figure(df)
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
