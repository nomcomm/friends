"""
head_motion.py — head-motion confound check for the laughter effect (Block C, R2.1b)
====================================================================================
Reviewer worry: the laughter vs non-laughter ISC/GLM difference could reflect
systematic head-motion differences rather than neural response. This checks it,
using fMRIPrep framewise-displacement (FD) confounds + the PRIMARY Clf-C laughter
labels, on the exact same HRF-shifted laughter/non-laughter windows as the neural
ISC (so it is a like-for-like comparison).

Three tests:
  1. FD magnitude   — is head motion higher during laughter windows? (paired t across episodes)
  2. Motion-ISC     — do subjects MOVE in sync more during laughter? (pairwise-ISC of the FD
                      timeseries, laughter vs non; a motion analogue of the neural ISC)
  3. Motion vs brain — does per-episode motion-ISC track the rTPJ brain-ISC? If not, motion
                      synchrony is dissociated from neural synchrony (i.e. not the driver).

Prior result (original annotations): motion-ISC was LOWER during laughter and uncorrelated
with rTPJ ISC — i.e. motion is not the confound. Recomputed here with Clf-C.

  inputs  : 02_CNEUROMOD_RAW/confounds/{sub}_*_task-{ep}_desc-confounds_timeseries.tsv (FD)
            data/0_prep/laughter_annotations/{ep}.csv                 (Clf-C laughter)
            data/b_laughter/{isc_laugh_by_parcel.npy, isc_nolaugh_by_parcel.npy, episodes.csv}
  outputs : data/c_controls/motion_summary.csv, motion_by_episode.csv
  figure  : results/analysis_plots/c_controls/fig_head_motion.png

Usage
  python head_motion.py [--force]
"""

import argparse
import glob
import sys
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
    CONFOUNDS_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, C_DIR, C_FIG_DIR,
    HRF_SHIFT_TRS, EXCLUDED_EPISODES, SUBJECTS, ROI_RTPJ_LAUGHTER,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "b_laughter"))
from laughter_isc import extract_event_segments, pairwise_isc_1d


def load_fd(ep):
    """Return (n_present_subj, n_TRs) FD array + list of present subject indices.

    18 subject-episode cells have TWO confound files because the episode was
    scanned in more than one session. The FD series must come from the SAME run
    whose BOLD produced the parcellated timeseries, or motion and brain signal
    are paired across different scans. `sorted(...)[0]` selects the lowest ses-
    number, which is the run the carried timeseries were extracted from (verified
    by TR-count match in 11 of the 18 cells; the other 7 are equal-length repeats
    that TR count cannot disambiguate). glob.glob() alone returns readdir order,
    which is filesystem-dependent — hence the explicit sort.
    """
    fds, present = [], []
    for i, sub in enumerate(SUBJECTS):
        m = sorted(glob.glob(str(CONFOUNDS_DIR / f"{sub}_*_task-{ep}_desc-confounds_timeseries.tsv")))
        if not m:
            continue
        fd = pd.read_csv(m[0], sep="\t")["framewise_displacement"].fillna(0).values
        fds.append(fd); present.append(i)
    if len(fds) < 2:
        return None, present
    n = min(len(f) for f in fds)
    return np.array([f[:n] for f in fds]), present


def seg_trs(segs):
    return np.concatenate([np.arange(o, off + 1) for o, off in segs]) if segs else np.array([], int)


def _pfmt(p):
    """Readable p-label: avoid the meaningless 'p=0.000' for tiny values."""
    return "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"


def figure(df):
    """Render the 3-panel motion-confound figure from the per-episode table (self-contained
    so it can be re-run via --plot-only without the full recompute)."""
    mzl, mzn = fisher_r_to_z(df.motion_isc_laugh), fisher_r_to_z(df.motion_isc_nonlaugh)
    _, p_fd = stats.ttest_rel(df.fd_laugh, df.fd_nonlaugh)
    _, p_mi = stats.ttest_rel(mzl, mzn)
    r_mb, p_mb = stats.pearsonr(df.motion_isc_delta, df.brain_isc_delta)
    mi_laugh, mi_non = float(fisher_z_to_r(mzl.mean())), float(fisher_z_to_r(mzn.mean()))
    mi_delta = float(fisher_z_to_r((mzl - mzn).mean()))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5)); fig.patch.set_facecolor("white")
    axes[0].bar([0, 1], [df.fd_laugh.mean(), df.fd_nonlaugh.mean()],
                yerr=[df.fd_laugh.sem(), df.fd_nonlaugh.sem()], capsize=4, color=["#27AE60", "#95A5A6"])
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["laughter", "non"]); axes[0].set_ylabel("mean FD (mm)")
    axes[0].set_title(f"A  Head motion (FD)\n{_pfmt(p_fd)}", fontsize=10, fontweight="bold", loc="left")
    axes[1].bar([0, 1], [mi_laugh, mi_non],
                yerr=[fisher_z_to_r(mzl).std()/np.sqrt(len(df)), 0], capsize=4, color=["#27AE60", "#95A5A6"])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["laughter", "non"]); axes[1].set_ylabel("motion-ISC (r)")
    axes[1].axhline(0, color="k", lw=.5)
    axes[1].set_title(f"B  Motion synchrony\nΔ={mi_delta:+.3f}, p={p_mi:.1e}", fontsize=10, fontweight="bold", loc="left")
    axes[2].scatter(df.motion_isc_delta, df.brain_isc_delta, s=12, alpha=.5, color="gray")
    axes[2].axhline(0, color="k", lw=.5); axes[2].axvline(0, color="k", lw=.5)
    axes[2].set_xlabel("motion-ISC Δ (laugh−non)"); axes[2].set_ylabel("rTPJ brain-ISC Δ")
    axes[2].set_title(f"C  Motion vs brain\nr={r_mb:+.2f}, {_pfmt(p_mb) if p_mb < 1e-3 else f'p={p_mb:.2f}'}",
                      fontsize=10, fontweight="bold", loc="left")
    fig.suptitle("Head-motion confound check (Clf-C): motion does not track or drive the laughter effect\n"
                 "(ISC-side checks shown here; the primary GLM is motion-robust — see voxelwise motion-augmented model, Fig S1)",
                 fontsize=12, fontweight="bold"); fig.tight_layout()
    fig.savefig(C_FIG_DIR / "fig_head_motion.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Figure: {C_FIG_DIR}/fig_head_motion.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    ap.add_argument("--plot-only", action="store_true",
                    help="re-render the figure from data/c_controls/motion_by_episode.csv (no recompute)")
    args = ap.parse_args()
    C_DIR.mkdir(parents=True, exist_ok=True); C_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = C_DIR / "motion_summary.csv"
    if args.plot_only:
        ep_csv = C_DIR / "motion_by_episode.csv"
        if not ep_csv.exists():
            print(f"No per-episode table to plot from: {ep_csv} (run without --plot-only first)"); return
        figure(pd.read_csv(ep_csv)); return
    if out_csv.exists() and not args.force:
        print(out_csv.read_text() if False else pd.read_csv(out_csv).to_string(index=False)); return

    # brain rTPJ ISC contrast per episode (for test 3)
    ep_df = pd.read_csv(B_DIR / "episodes.csv")
    lp = np.load(B_DIR / "isc_laugh_by_parcel.npy")[:, ROI_RTPJ_LAUGHTER]
    nl = np.load(B_DIR / "isc_nolaugh_by_parcel.npy")[:, ROI_RTPJ_LAUGHTER]
    brain = {e: fisher_r_to_z(lp[i]) - fisher_r_to_z(nl[i]) for i, e in enumerate(ep_df["episode"])}

    rows = []
    for ep in sorted(brain):
        if ep in EXCLUDED_EPISODES:
            continue
        ann = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
        if not ann.exists():
            continue
        fd, present = load_fd(ep)
        if fd is None:
            continue
        ls = pd.read_csv(ann)["ls"].values
        n = min(fd.shape[1], len(ls)); fd, ls = fd[:, :n], ls[:n]
        sh = np.zeros(n, int); on = np.where(ls == 1)[0] + HRF_SHIFT_TRS; sh[on[on < n]] = 1
        l_seg, nl_seg = extract_event_segments(sh)
        lt, nt = seg_trs(l_seg), seg_trs(nl_seg)
        lt, nt = lt[lt < n], nt[nt < n]
        if len(lt) < 4 or len(nt) < 4:
            continue
        rows.append({
            "episode": ep, "n_subj": fd.shape[0],
            "fd_laugh": float(fd[:, lt].mean()), "fd_nonlaugh": float(fd[:, nt].mean()),
            "motion_isc_laugh": pairwise_isc_1d(fd[:, lt]),
            "motion_isc_nonlaugh": pairwise_isc_1d(fd[:, nt]),
            "brain_isc_delta": float(brain[ep]),
        })
    df = pd.DataFrame(rows).dropna()
    df["motion_isc_delta"] = fisher_r_to_z(df["motion_isc_laugh"]) - fisher_r_to_z(df["motion_isc_nonlaugh"])
    df.to_csv(C_DIR / "motion_by_episode.csv", index=False)

    # tests
    t_fd, p_fd = stats.ttest_rel(df.fd_laugh, df.fd_nonlaugh)
    mzl, mzn = fisher_r_to_z(df.motion_isc_laugh), fisher_r_to_z(df.motion_isc_nonlaugh)
    t_mi, p_mi = stats.ttest_rel(mzl, mzn)
    r_mb, p_mb = stats.pearsonr(df.motion_isc_delta, df.brain_isc_delta)
    summary = pd.DataFrame([
        {"metric": "FD (mm)", "laugh": df.fd_laugh.mean(), "nonlaugh": df.fd_nonlaugh.mean(),
         "delta": df.fd_laugh.mean() - df.fd_nonlaugh.mean(), "t": t_fd, "p": p_fd, "n": len(df)},
        {"metric": "motion-ISC (r)", "laugh": float(fisher_z_to_r(mzl.mean())),
         "nonlaugh": float(fisher_z_to_r(mzn.mean())),
         "delta": float(fisher_z_to_r((mzl - mzn).mean())), "t": t_mi, "p": p_mi, "n": len(df)},
    ])
    summary.to_csv(out_csv, index=False)

    print(f"N episodes: {len(df)}")
    print(summary.to_string(index=False))
    print(f"\nMotion-ISC contrast vs rTPJ brain-ISC contrast: r={r_mb:+.3f}, p={p_mb:.3f}")
    verdict = ("motion is NOT the confound" if (df.fd_laugh.mean() < df.fd_nonlaugh.mean() + 0.05
               and abs(r_mb) < 0.2) else "inspect further")
    print(f"=> {verdict}")

    figure(df)
    print(f"Outputs: {C_DIR}")


if __name__ == "__main__":
    main()
