"""
qc_coverage.py — data-coverage overview for the fMRIPrep timeseries (0_prep QC)
===============================================================================
Validation artifact for extract_timeseries.py's output. Scans every
data/0_prep/fmriprep_timeseries/task-*.npy and reports, per subject × episode:
  • presence   — fraction of valid (non-all-NaN) TRs
  • length     — number of TRs
  • ROI count  — must be 1032 for every episode

OUTPUT
  data/0_prep/coverage/coverage_overview.png   subject×episode heatmap + length bars
  data/0_prep/coverage/coverage_table.csv      per-episode valid-TR counts + status

Not a data-producing pipeline stage — a QC pass over 0_prep. Cheap to re-run.

Usage
  python qc_coverage.py           # skip if outputs exist
  python qc_coverage.py --force   # always regenerate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import PREP_TIMESERIES_DIR, SUBJECTS, EXCLUDED_EPISODES, N_ROIS_UNIFIED

OUT_DIR  = PREP_TIMESERIES_DIR.parent / "coverage"
FIG_PATH = OUT_DIR / "coverage_overview.png"
CSV_PATH = OUT_DIR / "coverage_table.csv"

PARTIAL_FRAC = 0.9   # < this fraction of valid TRs (but > 0) counts as "partial"


def scan() -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    """Return (episodes, valid[n_subj, n_ep], n_trs[n_ep], n_rois per episode)."""
    files = sorted(PREP_TIMESERIES_DIR.glob("task-*.npy"))
    if not files:
        sys.exit(f"No timeseries found in {PREP_TIMESERIES_DIR} — run extract_timeseries.py first.")
    eps, ntrs, nrois = [], [], []
    valid = np.zeros((len(SUBJECTS), len(files)), dtype=int)
    for j, f in enumerate(files):
        a = np.load(f)                                   # (n_subj, n_trs, n_rois)
        eps.append(f.stem.replace("task-", ""))
        ntrs.append(a.shape[1]); nrois.append(a.shape[2])
        valid[:, j] = (~np.isnan(a).all(axis=2)).sum(axis=1)
    return eps, valid, np.array(ntrs), nrois


def status(v: int, n: int) -> str:
    if v == 0:              return "missing"
    if v < PARTIAL_FRAC * n: return "partial"
    return "full"


def write_table(eps, valid, ntrs, nrois) -> pd.DataFrame:
    tab = pd.DataFrame({"episode": eps, "season": [e[:3] for e in eps],
                        "n_trs": ntrs, "n_rois": nrois,
                        "excluded": [e in EXCLUDED_EPISODES for e in eps]})
    for i, s in enumerate(SUBJECTS):
        tab[f"{s}_valid_trs"] = valid[i]
        tab[f"{s}_status"]    = [status(valid[i, j], ntrs[j]) for j in range(len(eps))]
    tab.to_csv(CSV_PATH, index=False)
    return tab


def plot(eps, valid, ntrs, nrois) -> None:
    frac    = valid / ntrs[None, :]
    seasons = [e[:3] for e in eps]

    fig = plt.figure(figsize=(20, 7))
    gs  = fig.add_gridspec(3, 1, height_ratios=[4, 1.4, 0.5], hspace=0.35)

    axA = fig.add_subplot(gs[0])
    im  = axA.imshow(frac, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                     interpolation="nearest")
    axA.set_yticks(range(len(SUBJECTS))); axA.set_yticklabels(SUBJECTS)
    axA.set_xticks([]); axA.set_ylabel("subject")
    axA.set_title(f"fMRIPrep timeseries coverage — {len(eps)} episodes × {len(SUBJECTS)} "
                  f"subjects × {nrois[0]} ROIs   (color = fraction of valid TRs)",
                  fontsize=13, pad=10)
    for j, e in enumerate(eps):
        if e in EXCLUDED_EPISODES:
            axA.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(SUBJECTS), fill=False,
                          edgecolor="black", lw=1.2, hatch="///", zorder=5))
    fig.colorbar(im, ax=axA, fraction=0.02, pad=0.01).set_label("valid-TR fraction")
    axA.legend(handles=[Patch(fc=plt.cm.RdYlGn(1.0), label="full"),
                        Patch(fc=plt.cm.RdYlGn(0.5), label="partial"),
                        Patch(fc=plt.cm.RdYlGn(0.0), label="missing"),
                        Patch(fc="none", ec="black", hatch="///", label="in EXCLUDED_EPISODES")],
               loc="upper left", bbox_to_anchor=(1.035, 1.0), fontsize=8, frameon=False)

    axB = fig.add_subplot(gs[1])
    scol = {s: plt.cm.tab10(i / 10) for i, s in enumerate(sorted(set(seasons)))}
    axB.bar(range(len(eps)), ntrs, color=[scol[s] for s in seasons], width=1.0)
    axB.axhline(np.median(ntrs), color="k", ls="--", lw=0.8,
                label=f"median {int(np.median(ntrs))} TR")
    axB.set_ylabel("n TRs"); axB.set_xlim(-0.5, len(eps)-0.5); axB.set_xticks([])
    axB.legend(loc="upper right", fontsize=8)
    axB.set_title(f"episode length (TRs)   range {ntrs.min()}–{ntrs.max()}, "
                  f"median {int(np.median(ntrs))}", fontsize=10)

    axC = fig.add_subplot(gs[2]); axC.set_xlim(-0.5, len(eps)-0.5); axC.axis("off")
    prev = None
    for j, s in enumerate(seasons):
        if s != prev:
            axC.axvline(j-0.5, color="grey", lw=0.6)
            axC.text(j, 0.5, s, fontsize=9, va="center", ha="left")
            prev = s

    fig.savefig(FIG_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="regenerate even if outputs exist")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if FIG_PATH.exists() and CSV_PATH.exists() and not args.force:
        print(f"Coverage outputs already exist (use --force to regenerate):\n  {FIG_PATH}\n  {CSV_PATH}")
        return

    eps, valid, ntrs, nrois = scan()
    write_table(eps, valid, ntrs, nrois)
    plot(eps, valid, ntrs, nrois)

    n_issue = sum(status(valid[i, j], ntrs[j]) != "full"
                  for i in range(len(SUBJECTS)) for j in range(len(eps)))
    print(f"Episodes: {len(eps)}  | n_trs {ntrs.min()}–{ntrs.max()} "
          f"(median {int(np.median(ntrs))}) | all 1032 ROIs: {set(nrois) == {N_ROIS_UNIFIED}}")
    print(f"Non-full subject-cells: {n_issue} of {len(SUBJECTS)*len(eps)}")
    print(f"Wrote:\n  {FIG_PATH}\n  {CSV_PATH}")


if __name__ == "__main__":
    main()
