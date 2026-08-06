"""
striatum_isc.py — striatal laughter response: ISC vs GLM dissociation (Block B, B5, R2.1a)
==========================================================================================
Clean, unified (fMRIPrep + Clf-C) striatum analysis. Replaces the old split-pipeline
11_striatum_laughter_isc.py. The key result is a DISSOCIATION:
  - GLM (within-subject activation): the dorsal putamen is reliably engaged by laughter
    (moderate β); caudate weaker; NAc (ventral/reward) is significant but the weakest
    (β ~0.04, ~8x below putamen) — NOT null.
  - ISC (between-subject synchrony): NO striatal region shows a laughter synchrony gain.
So the striatum activates (esp. putamen) without synchronizing — ISC ≠ activation, and
the reward hub (NAc) is only weakly engaged (more a dorsal/sensorimotor / laughing-along signal).

Striatal ROIs (Melbourne S2, averaged bilaterally):
  putamen = aPUT+pPUT (rh+lh)   caudate = aCAU+pCAU (rh+lh)   NAc = shell+core (rh+lh)
  dorsal_str = putamen+caudate  ventral_str = NAc (identical parcels — NAc used in the figure)

METHOD  ISC: HRF-shift +3, segment, pairwise-mean ISC on the averaged-ROI timeseries,
             Fisher-z, paired t across episodes (as laughter_isc.py).
        GLM: SPM-HRF-convolved laughter regressor + cosine drift (bare), fit the
             averaged-ROI timeseries, one-sample t across episodes (as glm_contrast.py).

OUTPUTS  data/b_laughter/striatum_summary.csv
FIGURE   results/analysis_plots/b_laughter/fig_striatum.png   (ISC Δr vs GLM β per striatal ROI)

Usage
  python striatum_isc.py [--force]
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
from nltools.stats import fisher_r_to_z, fisher_z_to_r
from nilearn.glm.first_level import make_first_level_design_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, B_FIG_DIR, TR_SEC,
    HRF_SHIFT_TRS, EXCLUDED_EPISODES,
)
from laughter_isc import extract_event_segments, pairwise_isc_1d
from glm_contrast import laughter_blocks

# Melbourne S2 indices (into the 1032-ROI axis): rh 1000-1015, lh 1016-1031
STRIATAL = {
    "putamen":     [1012, 1013, 1028, 1029],              # aPUT, pPUT (rh+lh)
    "caudate":     [1014, 1015, 1030, 1031],              # aCAU, pCAU
    "NAc":         [1008, 1009, 1024, 1025],              # shell, core (ventral)
    "dorsal_str":  [1012, 1013, 1014, 1015, 1028, 1029, 1030, 1031],
    "ventral_str": [1008, 1009, 1024, 1025],
}
ORDER = ["putamen", "caudate", "NAc", "dorsal_str", "ventral_str"]   # compute all (CSV keeps every row)
# ventral_str == NAc (identical parcels), so the figure plots NAc only — no duplicate bar.
PLOT_ORDER = ["putamen", "caudate", "NAc", "dorsal_str"]


def episode_list():
    return sorted(f.stem.replace("task-", "") for f in PREP_TIMESERIES_DIR.glob("task-*.npy")
                  if f.stem.replace("task-", "") not in EXCLUDED_EPISODES)


def compute():
    isc = {r: {"l": [], "nl": []} for r in ORDER}
    glm = {r: [] for r in ORDER}
    for ep in episode_list():
        ann = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
        if not ann.exists():
            continue
        data = np.load(PREP_TIMESERIES_DIR / f"task-{ep}.npy")
        ls = pd.read_csv(ann)["ls"].values
        n = min(data.shape[1], len(ls)); data, ls = data[:, :n, :], ls[:n]

        # --- ISC (HRF-shift + segment) ---
        sh = np.zeros(n, int); on = np.where(ls == 1)[0] + HRF_SHIFT_TRS; sh[on[on < n]] = 1
        l_seg, nl_seg = extract_event_segments(sh)
        # --- GLM design (HRF-convolved, bare) ---
        bl = laughter_blocks(ls)
        dm = None
        if bl:
            ev = pd.DataFrame({"onset": [o * TR_SEC for o, _ in bl],
                               "duration": [l * TR_SEC for _, l in bl],
                               "trial_type": ["laughter"] * len(bl)})
            dm = make_first_level_design_matrix(np.arange(n) * TR_SEC, ev, hrf_model="spm",
                                                drift_model="cosine", high_pass=0.01)
        for r in ORDER:
            roi = data[:, :, STRIATAL[r]].mean(axis=2)          # (4, n) averaged ROI ts
            # ISC
            def cat(segs):
                p = [roi[:, o:off + 1] for o, off in segs if off + 1 > o]
                return np.concatenate(p, axis=1) if p else None
            ld, nld = cat(l_seg), cat(nl_seg)
            if ld is not None and nld is not None and ld.shape[1] >= 4 and nld.shape[1] >= 4:
                vl, vnl = pairwise_isc_1d(ld), pairwise_isc_1d(nld)
                if not (np.isnan(vl) or np.isnan(vnl)):
                    isc[r]["l"].append(vl); isc[r]["nl"].append(vnl)
            # GLM
            if dm is not None:
                li = list(dm.columns).index("laughter"); X = dm.values
                sub = []
                for si in range(4):
                    y = roi[si]
                    if np.isnan(y).any():
                        continue
                    beta, *_ = np.linalg.lstsq(X, y, rcond=None); sub.append(beta[li])
                if sub:
                    glm[r].append(np.mean(sub))

    rows = []
    for r in ORDER:
        lz = fisher_r_to_z(np.array(isc[r]["l"])); nz = fisher_r_to_z(np.array(isc[r]["nl"]))
        t_isc, p_isc = stats.ttest_rel(lz, nz)
        gb = np.array(glm[r]); t_glm, p_glm = stats.ttest_1samp(gb, 0)
        rows.append({"roi": r, "n": len(lz),
                     "isc_laugh": float(fisher_z_to_r(lz.mean())),
                     "isc_nolaugh": float(fisher_z_to_r(nz.mean())),
                     "isc_delta": float(fisher_z_to_r((lz - nz).mean())),
                     "isc_t": float(t_isc), "isc_p": float(p_isc),
                     "glm_beta": float(gb.mean()), "glm_t": float(t_glm), "glm_p": float(p_glm)})
    return pd.DataFrame(rows)


def figure(tab):
    x = np.arange(len(PLOT_ORDER))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5)); fig.patch.set_facecolor("white")
    star = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"

    # A: ISC Δr (synchrony)
    d = tab.set_index("roi").loc[PLOT_ORDER]
    axA.bar(x, d["isc_delta"], 0.6, color="#8E44AD", alpha=.85)
    rng = d["isc_delta"].abs().max() * 1.6
    axA.set_ylim(-rng, rng)
    for i, r in enumerate(PLOT_ORDER):
        dv = d.loc[r, "isc_delta"]
        axA.text(i, dv + np.sign(dv) * 0.06 * rng + 0.02 * rng, star(d.loc[r, "isc_p"]),
                 ha="center", va="bottom", fontsize=9, color="gray")
    axA.axhline(0, color="k", lw=.6); axA.set_xticks(x); axA.set_xticklabels(PLOT_ORDER, rotation=20, fontsize=9)
    axA.set_ylabel("ISC Δr (laughter − non)"); axA.set_title("A  Striatal ISC (synchrony) — all null",
                                                             fontweight="bold", loc="left", fontsize=10)
    # B: GLM beta (activation)
    axB.bar(x, d["glm_beta"], 0.6, color="#C0392B", alpha=.85)
    for i, r in enumerate(PLOT_ORDER):
        axB.text(i, d.loc[r, "glm_beta"] + 0.004, star(d.loc[r, "glm_p"]), ha="center", fontsize=9,
                 fontweight="bold")
    axB.axhline(0, color="k", lw=.6); axB.set_xticks(x); axB.set_xticklabels(PLOT_ORDER, rotation=20, fontsize=9)
    axB.set_ylabel("GLM laughter β (vs non)"); axB.set_title("B  Striatal GLM (activation) — putamen strongest, NAc weakest (all sig.)",
                                                             fontweight="bold", loc="left", fontsize=10)
    fig.suptitle("Striatum: activates (GLM, putamen) but does not synchronize (ISC) — a dissociation",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    (B_FIG_DIR / "exploratory").mkdir(parents=True, exist_ok=True)
    out = B_FIG_DIR / "exploratory" / "fig_striatum.png"   # working figure; plot_laughter.py curates finals
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"  Saved exploratory/{out.name}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render the figure from the saved striatum_summary.csv (no recompute)")
    args = ap.parse_args()
    B_DIR.mkdir(parents=True, exist_ok=True); B_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = B_DIR / "striatum_summary.csv"
    if args.plot_only:
        if not out.exists():
            print(f"No summary to plot from: {out} (run without --plot-only first)"); return
        figure(pd.read_csv(out)); print(f"Re-rendered figure from {out.name}"); return
    if out.exists() and not args.force:
        print(f"Output exists (use --force): {out}"); print(pd.read_csv(out).to_string(index=False)); return

    tab = compute(); tab.to_csv(out, index=False)
    print(f"{'ROI':<13}{'n':>5}{'ISC Δr':>9}{'ISC p':>8}   {'GLM β':>8}{'GLM t':>8}{'GLM p':>10}")
    for _, r in tab.iterrows():
        print(f"{r.roi:<13}{int(r.n):>5}{r.isc_delta:>+9.4f}{r.isc_p:>8.3f}   "
              f"{r.glm_beta:>+8.4f}{r.glm_t:>8.1f}{r.glm_p:>10.1e}")
    figure(tab)
    print(f"\nOutputs: {out}  |  Figure: {B_FIG_DIR}")


if __name__ == "__main__":
    main()
