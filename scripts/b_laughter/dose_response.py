"""
dose_response.py — laughter-intensity dose-response, GLM (primary) + ISC (Block B, R2.2b)
=========================================================================================
Does the laughter response scale with laughter intensity? Because laughter is a
discrete event, the GLM is the more fitting model, so it is the PRIMARY analysis
here; the ISC dose-response is reported alongside.

Intensity (data/b_laughter/laughter_intensity/, Clf-C-based) is median-split per
measure into 0=none (non-laughter) / 1=some (weaker laugh) / 2=full (strongest laugh):
  rms   = audience audio loudness   prob = classifier confidence   humor = comedic buildup

GLM dose (primary): per episode, two HRF-convolved event regressors — `some` and
  `full` — vs the non-laughter baseline; fit per region; second-level one-sample t
  for each tier and a paired t for the dose effect (full vs some). HRF convolution
  models the lag (no manual shift), consistent with glm_contrast.py.
ISC dose (secondary): tiered pairwise-ISC contrast (HRF-shifted segments, as in
  laughter_isc.py): ISC(none/some/full) and contrasts full−none, some−none.

Regions: rTPJ, visual, auditory (single parcels) + dorsal/ventral striatum (averaged).

OUTPUTS  (data/b_laughter/)
  dose_response_glm_by_episode.csv  per-episode tier betas, long format
                                    (episode, season, measure, region, tier, beta)
  dose_response_glm.csv    measure, region, betas & t for some/full, dose test
  dose_response_isc.csv    measure, region, ISC per tier + contrasts
FIGURE   (results/analysis_plots/b_laughter/)
  fig_dose_response.png    GLM β (some vs full) per region, one panel per measure

Usage
  python dose_response.py          # skip if dose_response_glm.csv exists
  python dose_response.py --force
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
from nilearn.glm.first_level import make_first_level_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, LAUGHTER_INTENSITY_DIR, B_DIR, B_FIG_DIR,
    TR_SEC, HRF_SHIFT_TRS, EVENT_DURATION_TRS, MIN_NONEVENT_DURATION, EXCLUDED_EPISODES,
    ROI_RTPJ_LAUGHTER, ROI_VISUAL_LOC, ROI_AUDITORY, DORSAL_STR_IDX, VENTRAL_STR_IDX,
)
from laughter_isc import extract_event_segments, pairwise_isc_1d

MEASURES = ["rms", "prob", "humor"]
REGIONS = [("rTPJ", [ROI_RTPJ_LAUGHTER]), ("visual_cortex", [ROI_VISUAL_LOC]),
           ("auditory_cortex", [ROI_AUDITORY]),
           ("dorsal_str", DORSAL_STR_IDX), ("ventral_str", VENTRAL_STR_IDX)]


def runs(vec, val):
    """Contiguous runs where vec==val → list of (start, length)."""
    out, i, n = [], 0, len(vec)
    while i < n:
        if vec[i] == val:
            j = i
            while j < n and vec[j] == val:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def region_ts(data, idxs):
    return data[:, :, idxs].mean(axis=2) if len(idxs) > 1 else data[:, :, idxs[0]]


def episode_tiers(ep, measure, n):
    csv = LAUGHTER_INTENSITY_DIR / f"{ep}_intensity.csv"
    if not csv.exists():
        return None
    return np.nan_to_num(pd.read_csv(csv)[f"{measure}_tier"].values[:n], nan=0).astype(int)


def glm_dose():
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    # acc[measure][region][tier] = list of per-episode mean beta
    acc = {m: {r: {"some": [], "full": []} for r, _ in REGIONS} for m in MEASURES}
    # per-episode long-format record, so the dose effect can be broken down by
    # season (added 2026-08-06 for Figure 3; does not touch the summary below)
    per_ep = []
    for ep in eps:
        tf = PREP_TIMESERIES_DIR / f"task-{ep}.npy"
        if not (LAUGHTER_INTENSITY_DIR / f"{ep}_intensity.csv").exists():
            continue
        data = np.load(tf); n = data.shape[1]
        ft = np.arange(n) * TR_SEC
        for m in MEASURES:
            tv = episode_tiers(ep, m, n)
            if tv is None:
                continue
            ev_rows = []
            for tier_val, name in [(1, "some"), (2, "full")]:
                for s, l in runs(tv, tier_val):
                    ev_rows.append({"onset": s * TR_SEC, "duration": l * TR_SEC, "trial_type": name})
            if not ev_rows:
                continue
            dm = make_first_level_design_matrix(ft, pd.DataFrame(ev_rows),
                                                hrf_model="spm", drift_model="cosine", high_pass=0.01)
            cols = list(dm.columns); X = dm.values
            for rname, idxs in REGIONS:
                rts = region_ts(data, idxs)                   # (4, n)
                sub_b = {"some": [], "full": []}
                for si in range(4):
                    y = rts[si]
                    if np.isnan(y).any():
                        continue
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                    for name in ("some", "full"):
                        if name in cols:
                            sub_b[name].append(beta[cols.index(name)])
                for name in ("some", "full"):
                    if sub_b[name]:
                        acc[m][rname][name].append(np.mean(sub_b[name]))
                        per_ep.append({"episode": ep, "season": int(ep[1:3]),
                                       "measure": m, "region": rname,
                                       "tier": name, "beta": float(np.mean(sub_b[name]))})
    pd.DataFrame(per_ep).to_csv(B_DIR / "dose_response_glm_by_episode.csv", index=False)
    print(f"  wrote dose_response_glm_by_episode.csv ({len(per_ep)} rows)")
    rows = []
    for m in MEASURES:
        for rname, _ in REGIONS:
            some = np.array(acc[m][rname]["some"]); full = np.array(acc[m][rname]["full"])
            ts_s, ps_s = stats.ttest_1samp(some, 0); ts_f, ps_f = stats.ttest_1samp(full, 0)
            # paired dose test on episodes present in both
            k = min(len(some), len(full))
            td, pd_ = stats.ttest_rel(full[:k], some[:k])
            rows.append({"measure": m, "region": rname,
                         "beta_some": some.mean(), "t_some": ts_s,
                         "beta_full": full.mean(), "t_full": ts_f,
                         "t_full_vs_some": td, "p_full_vs_some": pd_})
    return pd.DataFrame(rows)


def isc_dose():
    eps = sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                 if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)
    # rec[measure][region][tier] = {episode: isc}
    rec = {m: {r: {0: {}, 1: {}, 2: {}} for r, _ in REGIONS} for m in MEASURES}
    for ep in eps:
        tf = PREP_TIMESERIES_DIR / f"task-{ep}.npy"
        if not (LAUGHTER_INTENSITY_DIR / f"{ep}_intensity.csv").exists():
            continue
        data = np.load(tf); n = data.shape[1]
        for m in MEASURES:
            tv = episode_tiers(ep, m, n)
            if tv is None:
                continue
            sh = np.zeros(n, int)                              # HRF-shift tiers (like ISC)
            for i in np.where(tv > 0)[0]:
                if i + HRF_SHIFT_TRS < n:
                    sh[i + HRF_SHIFT_TRS] = tv[i]
            laugh, none = extract_event_segments((sh > 0).astype(int))
            grp = {0: none, 1: [], 2: []}
            for on, off in laugh:
                grp[int(sh[on])].append((on, off))
            for rname, idxs in REGIONS:
                rts = region_ts(data, idxs)                    # (4, n)
                for tier in (0, 1, 2):
                    segs = [rts[:, o:off + 1] for o, off in grp[tier] if off + 1 > o]
                    if not segs:
                        continue
                    seg = np.concatenate(segs, axis=1)
                    if seg.shape[1] >= 4:
                        v = pairwise_isc_1d(seg)
                        if not np.isnan(v):
                            rec[m][rname][tier][ep] = v
    def paired(a, b):
        keys = sorted(set(a) & set(b))
        az = fisher_r_to_z(np.array([a[k] for k in keys])); bz = fisher_r_to_z(np.array([b[k] for k in keys]))
        t, p = stats.ttest_rel(az, bz)
        return float(fisher_z_to_r((az - bz).mean())), float(p)
    rows = []
    for m in MEASURES:
        for rname, _ in REGIONS:
            R = rec[m][rname]
            mn = lambda d: float(fisher_z_to_r(fisher_r_to_z(np.array(list(d.values()))).mean())) if d else np.nan
            d_fn, p_fn = paired(R[2], R[0]); d_sn, p_sn = paired(R[1], R[0])
            rows.append({"measure": m, "region": rname,
                         "isc_none": mn(R[0]), "isc_some": mn(R[1]), "isc_full": mn(R[2]),
                         "d_full_none": d_fn, "p_full_none": p_fn,
                         "d_some_none": d_sn, "p_some_none": p_sn})
    return pd.DataFrame(rows)


def figure(glm):
    rnames = [r for r, _ in REGIONS]
    fig, axes = plt.subplots(1, len(MEASURES), figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor("white")
    x = np.arange(len(rnames)); w = 0.38
    for ax, m in zip(axes, MEASURES):
        sub = glm[glm.measure == m].set_index("region")
        bs = [sub.loc[r, "beta_some"] for r in rnames]; bf = [sub.loc[r, "beta_full"] for r in rnames]
        ax.bar(x - w/2, bs, w, color="#6baed6", label="some (weaker laugh)")
        ax.bar(x + w/2, bf, w, color="#d73027", label="full (strongest laugh)")
        for xi, r in enumerate(x):
            p = sub.loc[rnames[xi], "p_full_vs_some"]
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
            if star:
                ax.text(xi, max(bs[xi], bf[xi]) + 0.01, star, ha="center", fontweight="bold")
        ax.axhline(0, color="k", lw=.6); ax.set_xticks(x)
        ax.set_xticklabels([r.replace("_", "\n") for r in rnames], fontsize=9)
        ax.set_title(f"{m.upper()} intensity", fontsize=11, fontweight="bold")
        if m == MEASURES[0]:
            ax.set_ylabel("GLM laughter β (vs non-laughter)"); ax.legend(fontsize=9)
    fig.suptitle("GLM laughter dose-response: strongest (full) vs weaker (some) laughs  "
                 "(* = full≠some)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = B_FIG_DIR / "fig_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  Saved {out.name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True); B_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = B_DIR / "dose_response_glm.csv"
    if out.exists() and not args.force:
        print(f"Output exists (use --force): {out}"); return

    print("GLM dose-response ...")
    glm = glm_dose(); glm.to_csv(out, index=False)
    print("ISC dose-response ...")
    isc = isc_dose(); isc.to_csv(B_DIR / "dose_response_isc.csv", index=False)

    print("\n=== GLM dose-response (β some vs full, vs non-laughter baseline) ===")
    for m in MEASURES:
        print(f"\n{m.upper()}:")
        s = glm[glm.measure == m]
        print(f"  {'region':<16}{'β_some':>9}{'β_full':>9}{'full>some t':>13}{'p':>9}")
        for _, r in s.iterrows():
            print(f"  {r.region:<16}{r.beta_some:>9.4f}{r.beta_full:>9.4f}{r.t_full_vs_some:>13.2f}{r.p_full_vs_some:>9.4f}")
    figure(glm)
    print(f"\nOutputs: {B_DIR}  |  Figure: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
