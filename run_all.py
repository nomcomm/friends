#!/usr/bin/env python3
"""
Orchestrator for the Friends fMRI revision pipeline (v3).

The STAGES list below IS the pipeline. Read it top-to-bottom to understand the
entire analysis in one glance. Order is defined here, not encoded in filenames.

Usage:
    python run_all.py                 # run every stage in order
    python run_all.py --list          # print the pipeline and exit
    python run_all.py --stage a_isc_stability/compute_isc.py   # run one stage
    python run_all.py --from b_laughter/laughter_isc.py        # run from a stage onward
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"

# ---------------------------------------------------------------------------
# THE PIPELINE. Each entry is a script path relative to scripts/.
# Grouped by block; blocks map 1:1 to ../MANUSCRIPT/results_register.md.
# ---------------------------------------------------------------------------

STAGES = [
    # --- 0_prep: foundations everything depends on ---
    "0_prep/extract_timeseries.py",
    "0_prep/qc_coverage.py",
    "0_prep/train_laughter_classifier.py",
    "0_prep/predict_laughter.py",

    # --- Block A: stable shared ISC at scale (primary contribution) ---
    "a_isc_stability/compute_isc.py",
    "a_isc_stability/plot_stability.py",

    # --- Block B: laughter modulation of ISC (main-text narrative) ---
    "b_laughter/laughter_isc.py",
    "b_laughter/glm_contrast.py",
    "b_laughter/glm_per_viewer.py",
    "b_laughter/humor_classification.py",
    "b_laughter/striatum_isc.py",
    "b_laughter/dose_response.py",
    "b_laughter/season_consistency.py",
    "b_laughter/plot_laughter.py",

    # --- Block C: control analyses (supplement) ---
    "c_controls/head_motion.py",
    "c_controls/av_energy.py",
    "b_laughter/glm_av_energy.py",   # GLM twin of the AV-energy control (reuses the av_energy cache)
    "c_controls/isc_by_humor_type.py",
    "b_laughter/glm_by_humor_type.py",  # GLM twin (primary measure) of the humor-type test; reuses isc_by_humor_type labeling
    "c_controls/classifier_validation.py",
    "c_controls/h5_corroboration.py",   # H5-pipeline spatial stability (supplement corroboration)

    # --- read-out: every number the manuscript quotes, from the outputs above ---
    "verify_claims.py",
]


def run_stage(rel_path: str) -> None:
    script = SCRIPTS / rel_path
    if not script.exists():
        print(f"  [SKIP] {rel_path} — not yet ported")
        return
    print(f"\n{'='*70}\n  RUN  {rel_path}\n{'='*70}")
    subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print pipeline and exit")
    ap.add_argument("--stage", help="run a single stage")
    ap.add_argument("--from", dest="from_stage", help="run from this stage onward")
    args = ap.parse_args()

    if args.list:
        for s in STAGES:
            marker = " " if (SCRIPTS / s).exists() else "·"  # · = not yet ported
            print(f"  [{marker}] {s}")
        return

    if args.stage:
        run_stage(args.stage)
        return

    stages = STAGES
    if args.from_stage:
        if args.from_stage not in STAGES:
            sys.exit(f"unknown stage: {args.from_stage}")
        stages = STAGES[STAGES.index(args.from_stage):]

    for s in stages:
        run_stage(s)


if __name__ == "__main__":
    main()
