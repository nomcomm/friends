"""
av_energy.py — audiovisual-energy control for the sensory-cortex ISC effect (Block C, R2.3a)
============================================================================================
Reviewer R2.3a: higher auditory/visual ISC during NON-laughter could just reflect reduced
stimulus energy during laugh-track pauses (fewer dialogue/movement events), not humor
processing. Control: does the sensory-cortex laughter-ISC contrast SURVIVE when laughter and
non-laughter TRs are matched on stimulus energy?

Method (ported from 03_REVISION/15_av_energy_control.py; segmentation identical to
laughter_isc.py — HRF shift +3, event windows):
  - Laughter / non-laughter TR sets from the PRIMARY Clf-C annotations.
  - Per-TR stimulus energy (CARRIED OVER, cached — see av_energy/PROVENANCE.txt):
      audio  = acoustic RMS (dB) of the laugh-track      visual = frame-to-frame motion energy
    A BOLD TR at t is driven by stimulus energy at t−HRF_SHIFT.
  - 1:1 nearest-neighbour energy matching (caliper = 0.10·pooled SD, no replacement) →
    energy-balanced laughter/non subsets.
  - ISC (pairwise mean) per ROI on BASELINE (all TRs) vs MATCHED subsets; Δr = laugh − non.
    If Δr survives matching → not a stimulus-energy artifact.
  ROIs: rTPJ (842), visual (545), auditory (598).

  inputs  : data/0_prep/fmriprep_timeseries/, data/0_prep/laughter_annotations/ (Clf-C),
            data/c_controls/av_energy/{acoustic_rms,visual_energy}/  (cached per-TR energy)
  outputs : data/c_controls/av_energy_summary.csv (audio+visual), av_energy_by_episode_*.csv
  figure  : results/analysis_plots/c_controls/fig_av_energy.png

Usage
  python av_energy.py [--force]
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
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, AV_ENERGY_DIR, C_DIR, C_FIG_DIR,
    HRF_SHIFT_TRS, EXCLUDED_EPISODES, ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_laughter"))
from laughter_isc import extract_event_segments

ROIS = {"rTPJ": ROI_RTPJ_LAUGHTER, "visual_cortex": ROI_VISUAL_LOC, "auditory_cortex": ROI_AUDITORY}
ROI_ORDER = list(ROIS)
CALIPER_SD_FRAC = 0.10
MIN_TRS = 8


def condition_tr_sets(ls, n):
    sh = np.zeros(n, int); on = np.where(ls == 1)[0] + HRF_SHIFT_TRS; sh[on[on < n]] = 1
    l_segs, nl_segs = extract_event_segments(sh)
    def idx(segs):
        return np.concatenate([np.arange(o, min(off + 1, n)) for o, off in segs]) if segs else np.array([], int)
    return idx(l_segs), idx(nl_segs)


def pairwise_isc(ts):
    rs = []
    for i, j in combinations(range(ts.shape[0]), 2):
        a, b = ts[i], ts[j]
        if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        rs.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(rs)) if rs else np.nan


def bold_energy(energy, trs):
    src = np.clip(trs - HRF_SHIFT_TRS, 0, None)
    e = energy[src]; ok = np.isfinite(e)
    return trs[ok], e[ok]


def match_energy(l_trs, l_e, nl_trs, nl_e, caliper):
    if len(l_trs) == 0 or len(nl_trs) == 0:
        return np.array([], int), np.array([], int)
    laugh_anchor = len(l_trs) <= len(nl_trs)
    a_trs, a_e, p_trs, p_e = (l_trs, l_e, nl_trs, nl_e) if laugh_anchor else (nl_trs, nl_e, l_trs, l_e)
    used = np.zeros(len(p_trs), bool); a_m, p_m = [], []
    for k in np.argsort(a_e):
        d = np.abs(p_e - a_e[k]); d[used] = np.inf; j = int(np.argmin(d))
        if d[j] <= caliper:
            used[j] = True; a_m.append(a_trs[k]); p_m.append(p_trs[j])
    a_m, p_m = np.sort(np.array(a_m, int)), np.sort(np.array(p_m, int))
    return (a_m, p_m) if laugh_anchor else (p_m, a_m)


def load_energy(ep, modality, n):
    if modality == "audio":
        f = AV_ENERGY_DIR / "acoustic_rms" / f"{ep}_rms.csv"; col = "rms_db"
    else:
        f = AV_ENERGY_DIR / "visual_energy" / f"{ep}_vis.csv"; col = "vis_energy"
    if not f.exists():
        return None
    arr = pd.read_csv(f)[col].values
    return arr[:n] if len(arr) >= n else np.concatenate([arr, np.full(n - len(arr), np.nan)])


def process(ep, modality):
    ann = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"; tsf = PREP_TIMESERIES_DIR / f"task-{ep}.npy"
    if not (ann.exists() and tsf.exists()):
        return None
    data = np.load(tsf); ls = pd.read_csv(ann)["ls"].values
    n = min(data.shape[1], len(ls)); data, ls = data[:, :n, :], ls[:n]
    l_trs, nl_trs = condition_tr_sets(ls, n)
    if len(l_trs) < MIN_TRS or len(nl_trs) < MIN_TRS:
        return None
    energy = load_energy(ep, modality, n)
    if energy is None:
        return None
    l_e_trs, l_e = bold_energy(energy, l_trs); nl_e_trs, nl_e = bold_energy(energy, nl_trs)
    if len(l_e_trs) < MIN_TRS or len(nl_e_trs) < MIN_TRS:
        return None
    sd = np.std(np.concatenate([l_e, nl_e])); cal = CALIPER_SD_FRAC * sd if sd > 0 else np.inf
    l_m, nl_m = match_energy(l_e_trs, l_e, nl_e_trs, nl_e, cal)
    if len(l_m) < MIN_TRS or len(nl_m) < MIN_TRS:
        return None
    rec = {"episode": ep, "n_matched": len(l_m),
           "energy_laugh": float(l_e.mean()), "energy_nolaugh": float(nl_e.mean())}
    for rn, ri in ROIS.items():
        rec[f"base_laugh_{rn}"] = pairwise_isc(data[:, l_e_trs, ri])
        rec[f"base_nolaugh_{rn}"] = pairwise_isc(data[:, nl_e_trs, ri])
        rec[f"match_laugh_{rn}"] = pairwise_isc(data[:, l_m, ri])
        rec[f"match_nolaugh_{rn}"] = pairwise_isc(data[:, nl_m, ri])
    return rec


def summarise(df, modality):
    rows = []
    for rn in ROI_ORDER:
        pb = df[[f"base_laugh_{rn}", f"base_nolaugh_{rn}"]].dropna()
        pm = df[[f"match_laugh_{rn}", f"match_nolaugh_{rn}"]].dropna()
        lb, nb = fisher_r_to_z(pb[f"base_laugh_{rn}"]), fisher_r_to_z(pb[f"base_nolaugh_{rn}"])
        lm, nm = fisher_r_to_z(pm[f"match_laugh_{rn}"]), fisher_r_to_z(pm[f"match_nolaugh_{rn}"])
        tb, pbv = stats.ttest_rel(lb, nb); tm, pmv = stats.ttest_rel(lm, nm)
        db = float(fisher_z_to_r((lb - nb).mean())); dm = float(fisher_z_to_r((lm - nm).mean()))
        rows.append({"modality": modality, "roi": rn, "n": len(pm),
                     "delta_base": db, "t_base": float(tb), "p_base": float(pbv),
                     "delta_match": dm, "t_match": float(tm), "p_match": float(pmv),
                     "attenuation_pct": float((1 - dm / db) * 100) if db != 0 else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    C_DIR.mkdir(parents=True, exist_ok=True); C_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = C_DIR / "av_energy_summary.csv"
    if out.exists() and not args.force:
        print(pd.read_csv(out).to_string(index=False)); return

    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    summaries = []
    for modality in ("audio", "visual"):
        recs = [r for r in (process(ep, modality) for ep in eps) if r is not None]
        df = pd.DataFrame(recs); df.to_csv(C_DIR / f"av_energy_by_episode_{modality}.csv", index=False)
        s = summarise(df, modality); summaries.append(s)
        print(f"\n=== {modality.upper()} energy matching (N={len(df)} episodes) ===")
        print(f"{'ROI':<16}{'Δr base':>9}{'p_base':>8}{'Δr match':>10}{'p_match':>9}{'atten%':>8}")
        for _, r in s.iterrows():
            print(f"{r.roi:<16}{r.delta_base:>+9.4f}{r.p_base:>8.3f}{r.delta_match:>+10.4f}{r.p_match:>9.3f}{r.attenuation_pct:>8.0f}")
    summary = pd.concat(summaries, ignore_index=True); summary.to_csv(out, index=False)

    # figure: baseline vs matched Δr per ROI, audio + visual
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True); fig.patch.set_facecolor("white")
    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
    for ax, modality, s in zip(axes, ("audio", "visual"), summaries):
        x = np.arange(len(ROI_ORDER)); w = 0.38
        s = s.set_index("roi").loc[ROI_ORDER]
        ax.bar(x - w/2, s["delta_base"], w, color="#95A5A6", label="baseline (all TRs)")
        ax.bar(x + w/2, s["delta_match"], w, color="#27AE60", label="energy-matched")
        for xi, rn in enumerate(ROI_ORDER):
            ax.text(x[xi] - w/2, s.loc[rn, "delta_base"], star(s.loc[rn, "p_base"]), ha="center", va="bottom", fontsize=8)
            ax.text(x[xi] + w/2, s.loc[rn, "delta_match"], star(s.loc[rn, "p_match"]), ha="center", va="bottom", fontsize=8)
        ax.axhline(0, color="k", lw=.6); ax.set_xticks(x); ax.set_xticklabels([r.replace("_", "\n") for r in ROI_ORDER], fontsize=9)
        ax.set_title(f"{modality.upper()}-energy matching", fontsize=11, fontweight="bold")
        if modality == "audio":
            ax.set_ylabel("Δr  (ISC laughter − non-laughter)"); ax.legend(fontsize=8)
    fig.suptitle("AV-energy control (Clf-C): does the sensory-cortex laughter-ISC survive energy matching?",
                 fontsize=12, fontweight="bold"); fig.tight_layout()
    fig.savefig(C_FIG_DIR / "fig_av_energy.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"\nOutputs: {C_DIR}  |  Figure: {C_FIG_DIR}/fig_av_energy.png")


if __name__ == "__main__":
    main()
