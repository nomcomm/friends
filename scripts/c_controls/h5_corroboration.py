"""
h5_corroboration.py — H5-pipeline spatial stability (supplement corroboration)
==============================================================================
Part A's stability result is reported for both pipelines: fMRIPrep (primary,
plot_stability.py) and the inherited H5 parcellation (corroboration). The H5
number was quoted in the supplement with no script emitting it — `isc_h5/` was
carried but nothing read it — so this stage closes that gap.

Method is identical to plot_stability.py so the two pipelines are comparable:
pairwise spatial correlation between every pair of per-episode ISC maps, over the
1000 cortical parcels, on raw ISC r values (no Fisher-z; `np.corrcoef` of the
episode x parcel matrix). Reproduces the published 0.867 +/- 0.030.

The absolute magnitude is higher than fMRIPrep's 0.659 +/- 0.084 by pipeline, not
by finding: H5 applies lighter confound regression, which leaves more shared
stimulus-driven variance in every parcel and so raises the map-to-map correlation.

SCOPE — stability only. The other H5 values in the supplement (the laughter-vs-
non-laughter ISC ROI table, 474/1000 FDR-significant, and the r = 0.961 pipeline
agreement) come from the original submission's notebook, which is NOT in this tree;
they cannot be regenerated here and are inherited as published. Recomputing that
contrast from the T7 H5 timeseries with this package's segmentation gets close but
not equal (449/1000; rTPJ t = 4.03 vs 3.53 published), so it is deliberately not
substituted here — see README "Inherited numbers".

  input   : data/a_isc_stability/isc_h5/isc_task-*.npy   (280 x 1000, carried)
  output  : data/c_controls/h5_stability_stats.txt

Usage
  python h5_corroboration.py            # skip if output exists
  python h5_corroboration.py --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import A_ISC_H5_DIR, C_DIR, N_PARCELS


def pair_mean_sd(mat):
    """Mean/sd of all pairwise spatial correlations between rows of `mat`."""
    r = np.corrcoef(mat)
    pairs = r[np.triu_indices(len(mat), k=1)]
    return float(pairs.mean()), float(pairs.std()), pairs.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    C_DIR.mkdir(parents=True, exist_ok=True)
    out = C_DIR / "h5_stability_stats.txt"
    if out.exists() and not args.force:
        print(f"Output exists (use --force): {out}")
        return

    files = sorted(A_ISC_H5_DIR.glob("isc_task-*.npy"))
    if not files:
        sys.exit(f"no carried H5 ISC maps in {A_ISC_H5_DIR}")
    episodes = [f.stem.replace("isc_task-", "") for f in files]
    maps = np.array([np.load(f) for f in files])[:, :N_PARCELS]

    mean_r, sd_r, n_pairs = pair_mean_sd(maps)
    lines = ["Spatial ISC stability (H5 pipeline, 1000 cortical parcels) — CORROBORATION",
             "=" * 72,
             f"All episodes: N={len(files)}, {n_pairs} pairs, mean r = {mean_r:.3f} +/- {sd_r:.3f}",
             "",
             "Per-season stability:"]
    seasons = np.array([int(e[1:3]) for e in episodes])
    for s in sorted(set(seasons)):
        sel = maps[seasons == s]
        if len(sel) < 2:
            continue
        m, sd, _ = pair_mean_sd(sel)
        lines.append(f"  s{s:02d}: N={len(sel):3d}, mean r = {m:.3f} +/- {sd:.3f}")
    lines += ["",
              "Primary fMRIPrep comparison: 0.659 +/- 0.084 (see plot_stability.py).",
              "Magnitudes differ by pipeline (H5 regresses fewer confounds); the",
              "finding — a stable shared spatial pattern across the six-season run —",
              "holds under both."]

    report = "\n".join(lines)
    out.write_text(report + "\n")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
