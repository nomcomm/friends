"""
compute_isc.py — whole-brain overall ISC per episode (Block A, PRIMARY = fMRIPrep)
==================================================================================
Ported from 03_REVISION/scripts/01_compute_isc.py, but computed on the fMRIPrep
timeseries (the declared primary pipeline) instead of the H5 pipeline.

For each episode and each ROI, ISC = the MEDIAN of the pairwise (everyone-vs-
everyone) Pearson correlations across subjects. This reproduces the original
manuscript's method: nltools 0.5.0 `isc(..., metric='median')` computes the full
subject×subject correlation matrix and summarizes it by the median of the pairs
(validated identical to nltools to ~1e-9). With 4 subjects that is 6 pairs.

  input   : data/0_prep/fmriprep_timeseries/task-{episode}.npy   (4, TRs, 1032)
            NaN-padded for missing subjects — only present subjects are used.
  outputs : data/a_isc_stability/isc_fmriprep/isc_task-{episode}.npy   (1032,)
            data/a_isc_stability/isc_fmriprep/isc_all.npy   (n_episodes, 1032)
            data/a_isc_stability/isc_fmriprep/episodes.csv  (episode, n_subjects, n_trs)

NOTES
  - ROI axis: 0–999 Schaefer cortical (used for the stability figure),
    1000–1031 Melbourne subcortical (stored for reuse; free to compute).
  - Inclusion: an episode is included if ≥2 subjects are fully present (≥1 pair).
    Truncated/partial subjects (mostly-NaN) are dropped; episodes falling below
    2 usable subjects are skipped.
  - The significance null (phase-randomization) is intentionally NOT computed here
    — Block A's contribution (spatial stability) needs only the observed ISC maps.
    Significance (register A1) is a separate stats step. The H5 pipeline's ISC +
    null are carried over verbatim in ../isc_h5/ (corroboration; see its PROVENANCE).
  - Skip-if-output-exists.

Usage
  python compute_isc.py            # compute all missing episodes
  python compute_isc.py --force    # recompute everything
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PREP_TIMESERIES_DIR, A_ISC_FMRIPREP_DIR, N_ROIS_UNIFIED

MIN_VALID_TR_FRAC = 0.9   # a subject counts as "present" if ≥90% of its TRs are non-NaN


def pairwise_median_isc(data: np.ndarray) -> np.ndarray:
    """
    data : (n_subjects, n_TRs, n_ROIs) — already NaN-free / present subjects only.
    Returns (n_ROIs,) : median over subject-pair Pearson correlations, per ROI.
    """
    z = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-12)
    pairs = list(itertools.combinations(range(data.shape[0]), 2))
    pw = np.stack([(z[i] * z[j]).mean(axis=0) for i, j in pairs])   # (n_pairs, n_ROIs)
    return np.median(pw, axis=0)


def present_subjects(arr: np.ndarray) -> np.ndarray:
    """Return arr restricted to subjects with ≥MIN_VALID_TR_FRAC non-NaN TRs."""
    valid_frac = (~np.isnan(arr).all(axis=2)).mean(axis=1)   # per subject
    keep = valid_frac >= MIN_VALID_TR_FRAC
    return arr[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recompute even if outputs exist")
    args = ap.parse_args()

    A_ISC_FMRIPREP_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PREP_TIMESERIES_DIR.glob("task-*.npy"))
    if not files:
        sys.exit(f"No timeseries in {PREP_TIMESERIES_DIR} — run 0_prep/extract_timeseries.py first.")

    rows, computed, skipped, excluded = [], 0, 0, []
    for f in files:
        ep  = f.stem.replace("task-", "")
        out = A_ISC_FMRIPREP_DIR / f"isc_{f.stem}.npy"

        arr = present_subjects(np.load(f))            # drop absent/truncated subjects
        if arr.shape[0] < 2:
            excluded.append(ep); continue
        # trim any trailing NaN TRs shared handling: use TRs where all kept subjects valid
        good_tr = ~np.isnan(arr).any(axis=(0, 2))
        arr = arr[:, good_tr, :]
        rows.append({"episode": ep, "n_subjects": arr.shape[0], "n_trs": arr.shape[1]})

        if out.exists() and not args.force:
            skipped += 1; continue
        np.save(out, pairwise_median_isc(arr).astype(np.float32))
        computed += 1

    # aggregate stack for the stability analysis (episodes with an isc file, in order)
    idx = pd.DataFrame(rows).sort_values("episode").reset_index(drop=True)
    stack = np.stack([np.load(A_ISC_FMRIPREP_DIR / f"isc_task-{e}.npy") for e in idx["episode"]])
    np.save(A_ISC_FMRIPREP_DIR / "isc_all.npy", stack.astype(np.float32))
    idx.to_csv(A_ISC_FMRIPREP_DIR / "episodes.csv", index=False)

    print(f"Episodes included: {len(idx)}  (computed {computed}, skipped {skipped}) | "
          f"excluded <2 subj: {len(excluded)}")
    if excluded:
        print("  excluded:", ", ".join(excluded))
    print(f"isc_all.npy: {stack.shape}  (episodes × {N_ROIS_UNIFIED} ROIs)")
    print(f"Output: {A_ISC_FMRIPREP_DIR}")


if __name__ == "__main__":
    main()
