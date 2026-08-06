"""
isc_by_humor_type.py — does rTPJ ISC depend on humor TYPE? (Block C, R2.3b — EXPLORATORY)
=========================================================================================
Reviewer 2 (following Samson et al. 2008): theory-of-mind (social) jokes may recruit rTPJ
more than wordplay. Using the LLM humor categories (Juckel 2016 verbal typology: Language /
Logic / Identity; Identity = most TOM-adjacent), does rTPJ inter-subject synchrony at
laughter differ by the kind of humor? Prediction: Identity > Language/Logic in rTPJ, not in
sensory controls.

EXPLORATORY — power is very unequal across categories (Logic ≫ Language ≫ Identity). All
p-values UNCORRECTED. Prior result (original annotations): a clean NULL (rTPJ ANOVA p≈.50).
Recomputed here with the PRIMARY Clf-C laughter + fMRIPrep timeseries.

Two operationalizations:
  PRIMARY (buildup-labeled): each laughter block is labelled by the DOMINANT humor category
    in the 20-TR (~30 s) window before its onset; rTPJ ISC is computed over the HRF-shifted
    laughter windows, per category. (theory-aligned "harvesting" test)
  SECONDARY (content-labeled): humor-positive TRs grouped by their own category, HRF-shifted,
    ISC per category. (laughter-independent robustness check)

  inputs  : data/0_prep/fmriprep_timeseries/task-{ep}.npy
            data/0_prep/laughter_annotations/{ep}.csv                       (Clf-C ls)
            data/b_laughter/humor_classification/per_episode/{ep}_humor_by_tr.csv
  outputs : data/c_controls/isc_by_humor_type_{primary,secondary}.csv, _stats.txt
  figure  : results/analysis_plots/c_controls/fig_isc_by_humor_type.png

Usage
  python isc_by_humor_type.py [--force]
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
from scipy import stats
from nltools.stats import fisher_r_to_z, fisher_z_to_r

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, HUMOR_CLASSIFICATION_DIR, C_DIR, C_FIG_DIR,
    ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY, HRF_SHIFT_TRS, EXCLUDED_EPISODES,
)

BUILDUP_WINDOW_TRS = 20
MIN_TRS_FOR_ISC = 8
CATEGORIES = ["Language", "Logic", "Identity"]
CAT_COLORS = {"Language": "#3498DB", "Logic": "#9B59B6", "Identity": "#E67E22"}
ROIS = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}
ROI_ORDER = list(ROIS)
HUMOR_PER_EP = HUMOR_CLASSIFICATION_DIR / "per_episode"


def pairwise_isc_1d(ts):
    rs = []
    for i, j in combinations(range(ts.shape[0]), 2):
        a, b = ts[i], ts[j]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def isc_all_rois(concat):
    if concat is None or concat.shape[1] < MIN_TRS_FOR_ISC:
        return {r: np.nan for r in ROI_ORDER}
    return {r: pairwise_isc_1d(concat[:, :, ROIS[r]]) for r in ROI_ORDER}


def load_episode(ep):
    hf = HUMOR_PER_EP / f"{ep}_humor_by_tr.csv"
    bf = PREP_TIMESERIES_DIR / f"task-{ep}.npy"
    af = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
    if not (hf.exists() and bf.exists() and af.exists()):
        return None
    bold = np.load(bf); ls = pd.read_csv(af)["ls"].values.astype(int)
    h = pd.read_csv(hf); is_h = h["is_humor"].fillna(0).astype(int).values
    cat = h["primary_category"].fillna("").values
    n = min(bold.shape[1], len(ls), len(cat))
    return bold[:, :n, :], ls[:n], is_h[:n], cat[:n]


def blocks(ls):
    out, i, n = [], 0, len(ls)
    while i < n:
        if ls[i] == 1:
            o = i
            while i < n and ls[i] == 1:
                i += 1
            out.append((o, i))
        else:
            i += 1
    return out


def dominant(cat_w, ish_w):
    c = {k: 0 for k in CATEGORIES}
    for k in range(len(cat_w)):
        if ish_w[k] == 1 and cat_w[k] in c:
            c[cat_w[k]] += 1
    if sum(c.values()) == 0:
        return None
    top = max(c.values()); win = [k for k, v in c.items() if v == top]
    return win[0] if len(win) == 1 else None


def concat_windows(bold, wins):
    parts = [bold[:, s:e, :] for s, e in wins if e > s]
    return np.concatenate(parts, axis=1) if parts else None


def run():
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    prim, sec = [], []
    cov = {c: {"events": 0, "trs": 0} for c in CATEGORIES}
    for ep in eps:
        loaded = load_episode(ep)
        if loaded is None:
            continue
        bold, ls, ish, cat = loaded; n = bold.shape[1]

        # PRIMARY: buildup-labeled laughter blocks
        wins = {c: [] for c in CATEGORIES}
        for onset, offset in blocks(ls):
            b0 = max(0, onset - BUILDUP_WINDOW_TRS)
            dom = dominant(cat[b0:onset], ish[b0:onset])
            if dom is None:
                continue
            s, e = min(onset + HRF_SHIFT_TRS, n), min(offset + HRF_SHIFT_TRS, n)
            if e > s:
                wins[dom].append((s, e)); cov[dom]["events"] += 1; cov[dom]["trs"] += e - s
        for c in CATEGORIES:
            iscs = isc_all_rois(concat_windows(bold, wins[c]))
            if not np.isnan(iscs["rTPJ"]):
                prim.append({"episode": ep, "season": int(ep[1:3]), "category": c,
                             **{f"isc_{r}": iscs[r] for r in ROI_ORDER}})

        # SECONDARY: content-labeled humor TRs by own category (HRF-shifted)
        for c in CATEGORIES:
            mask = (ish == 1) & (cat == c)
            idx = np.where(mask)[0] + HRF_SHIFT_TRS
            idx = idx[idx < n]
            if len(idx) >= MIN_TRS_FOR_ISC:
                seg = bold[:, idx, :]
                iscs = {r: pairwise_isc_1d(seg[:, :, ROIS[r]]) for r in ROI_ORDER}
                if not np.isnan(iscs["rTPJ"]):
                    sec.append({"episode": ep, "category": c, **{f"isc_{r}": iscs[r] for r in ROI_ORDER}})
    return pd.DataFrame(prim), pd.DataFrame(sec), cov


def summarise(df, label, lines):
    lines.append(f"\n{'='*60}\n{label}\n{'='*60}")
    for r in ROI_ORDER:
        col = f"isc_{r}"
        by = {c: fisher_r_to_z(df[df.category == c][col].dropna().values) for c in CATEGORIES}
        means = {c: float(fisher_z_to_r(by[c].mean())) for c in CATEGORIES}
        F, p = stats.f_oneway(*[by[c] for c in CATEGORIES])
        lines.append(f"[{r}]  means: " + ", ".join(f"{c}={means[c]:.4f}(n={len(by[c])})" for c in CATEGORIES))
        lines.append(f"       one-way ANOVA F={F:.2f} p={p:.4f}")
        ti, pi = stats.ttest_ind(by["Identity"], np.concatenate([by["Language"], by["Logic"]]))
        lines.append(f"       Identity vs (Language+Logic): t={ti:+.2f} p={pi:.4f}")
    return lines


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C_DIR.mkdir(parents=True, exist_ok=True); C_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = C_DIR / "isc_by_humor_type_primary.csv"
    if out.exists() and not args.force:
        print((C_DIR / "isc_by_humor_type_stats.txt").read_text()); return

    prim, sec, cov = run()
    prim.to_csv(out, index=False)
    sec.to_csv(C_DIR / "isc_by_humor_type_secondary.csv", index=False)

    lines = ["EXPLORATORY: rTPJ ISC by humor type (R2.3b) — Clf-C, fMRIPrep, UNCORRECTED",
             f"Buildup window {BUILDUP_WINDOW_TRS} TRs, HRF shift {HRF_SHIFT_TRS}, min {MIN_TRS_FOR_ISC} TRs/ISC",
             "Coverage (PRIMARY, buildup-labeled events):"]
    for c in CATEGORIES:
        lines.append(f"  {c:<9}: {cov[c]['events']:>6} events, {cov[c]['trs']:>6} TRs")
    summarise(prim, "PRIMARY (buildup-labeled laughter)", lines)
    summarise(sec, "SECONDARY (content-labeled humor TRs)", lines)
    report = "\n".join(lines); (C_DIR / "isc_by_humor_type_stats.txt").write_text(report + "\n")
    print(report)

    # figure: primary — ISC by category per ROI
    fig, axes = plt.subplots(1, len(ROI_ORDER), figsize=(14, 4.5), sharey=True); fig.patch.set_facecolor("white")
    for ax, r in zip(axes, ROI_ORDER):
        col = f"isc_{r}"; x = np.arange(len(CATEGORIES))
        means = [float(fisher_z_to_r(fisher_r_to_z(prim[prim.category == c][col].dropna()).mean())) for c in CATEGORIES]
        sems = [prim[prim.category == c][col].dropna().sem() for c in CATEGORIES]
        ax.bar(x, means, yerr=sems, capsize=4, color=[CAT_COLORS[c] for c in CATEGORIES], alpha=.9)
        F, p = stats.f_oneway(*[fisher_r_to_z(prim[prim.category == c][col].dropna().values) for c in CATEGORIES])
        ax.set_xticks(x); ax.set_xticklabels(CATEGORIES, rotation=15, fontsize=9)
        ax.set_title(f"{r}\nANOVA p={p:.2f}", fontsize=10, fontweight="bold")
        if r == ROI_ORDER[0]:
            ax.set_ylabel("ISC at laughter (r)")
    fig.suptitle("rTPJ ISC by humor type (buildup-labeled) — EXPLORATORY, uncorrected (expected: null)",
                 fontsize=12, fontweight="bold"); fig.tight_layout()
    fig.savefig(C_FIG_DIR / "fig_isc_by_humor_type.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"\nOutputs: {C_DIR}  |  Figure: {C_FIG_DIR}/fig_isc_by_humor_type.png")


if __name__ == "__main__":
    main()
