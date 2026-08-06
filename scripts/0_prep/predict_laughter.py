"""
predict_laughter.py — annotate every episode with the PRIMARY classifier (Clf-C)
================================================================================
Ported from 03_REVISION/scripts/09a_predict_all_episodes.py (Clf-C prediction),
with the per-episode CSV output of 02_predict_laughter.py. Runs the carried-over
Clf-C model over each episode's TR grid and writes a laughter annotation per TR.

  input   : data/0_prep/laughter_classifier.pkl        (Clf-C, 86-feat RF)
            episode audio from config.STIMULI_ROOT       (native 48 kHz, T7 drive)
            original TR-onset grid from config.LAUGHTER_ANN_DIR (read-only)
  output  : data/0_prep/laughter_annotations/{episode}.csv
            columns: onsets, minutes_seconds, offsets, ls, prob

NOTES
  - Native 48 kHz audio, shared 86-dim extractor (laughter_features.py) — this is
    the path that reproduces the authoritative annotations bit-identically. The
    old 02_predict_laughter.py 22 kHz-resample + 32-feat path is intentionally
    NOT carried over (that was the SR bug; see CONSOLIDATION §8).
  - Episodes in config.EXCLUDED_EPISODES are skipped (same as the ISC analysis).
  - The 280 annotations are CARRIED OVER (see laughter_annotations/PROVENANCE.txt);
    a normal run is a skip-if-exists no-op. --verify re-derives and compares.

Usage
  python predict_laughter.py                 # annotate missing episodes (skip existing)
  python predict_laughter.py --episode s02e05a
  python predict_laughter.py --verify        # re-derive vs carried-over (no writes)
"""

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    STIMULI_ROOT, LAUGHTER_ANN_DIR, EXCLUDED_EPISODES, TR_SEC,
    PREP_LAUGHTER_MODEL, PREP_LAUGHTER_ANN_DIR, LAUGHTER_PROB_THRESHOLD,
)
from laughter_features import extract_features_np


def episode_mp3(ep: str) -> Path:
    season = int(re.search(r"s(\d+)", ep).group(1))
    return STIMULI_ROOT / f"s{season}" / f"friends_{ep}.mp3"


def predict_episode(clf, ep: str) -> pd.DataFrame | None:
    """Annotate one episode on the original TR-onset grid. None if inputs absent."""
    import librosa
    mp3 = episode_mp3(ep)
    grid = LAUGHTER_ANN_DIR / f"friends_{ep}.csv"      # original onsets grid
    if not mp3.exists() or not grid.exists():
        return None

    orig = pd.read_csv(grid, index_col=0)
    y, sr = librosa.load(str(mp3), sr=None, mono=True)
    tr_samples, n_tot = int(TR_SEC * sr), len(y)

    X = []
    for _, row in orig.iterrows():
        start = int(row["onsets"] / 1000 * sr)
        end   = min(start + tr_samples, n_tot)
        X.append(extract_features_np(y[start:end], sr))

    probs = clf.predict_proba(np.array(X))[:, 1]
    ls    = (probs >= LAUGHTER_PROB_THRESHOLD).astype(int)
    return pd.DataFrame({
        "onsets":          orig["onsets"].values,
        "minutes_seconds": orig["minutes_seconds"].values,
        "offsets":         orig["offsets"].values,
        "ls":              ls,
        "prob":            np.round(probs, 4),
    })


def episodes_to_do() -> list[str]:
    """All episodes with an original onset grid, minus the excluded set."""
    eps = sorted(p.name.replace("friends_", "").replace(".csv", "")
                 for p in LAUGHTER_ANN_DIR.glob("friends_s*.csv"))
    return [e for e in eps if e not in EXCLUDED_EPISODES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default=None, help="annotate a single episode")
    ap.add_argument("--verify", action="store_true",
                    help="re-derive and compare against carried-over annotations (no writes)")
    args = ap.parse_args()

    clf = joblib.load(PREP_LAUGHTER_MODEL)

    if args.verify:
        eps = [args.episode] if args.episode else episodes_to_do()[:3]
        print(f"Verifying {len(eps)} episode(s) against carried-over annotations:")
        for ep in eps:
            saved_path = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
            df = predict_episode(clf, ep)
            if df is None or not saved_path.exists():
                print(f"  {ep}: inputs/annotation missing — skipped"); continue
            saved = pd.read_csv(saved_path)
            m = min(len(df), len(saved))
            agree = (df["ls"].values[:m] == saved["ls"].values[:m]).mean() * 100
            print(f"  {ep}: ls {agree:.2f}% identical  "
                  f"(re-derived laugh {df['ls'].mean()*100:.1f}%, saved {saved['ls'].mean()*100:.1f}%)")
        return

    PREP_LAUGHTER_ANN_DIR.mkdir(parents=True, exist_ok=True)
    eps = [args.episode] if args.episode else episodes_to_do()
    done = skipped = failed = 0
    for ep in eps:
        out = PREP_LAUGHTER_ANN_DIR / f"{ep}.csv"
        if out.exists():
            skipped += 1; continue
        df = predict_episode(clf, ep)
        if df is None:
            print(f"  {ep}: audio or onset grid missing (T7 mounted?) — skipped"); failed += 1; continue
        df.to_csv(out, index=False)
        done += 1
        print(f"  {ep}: {df['ls'].sum()}/{len(df)} laughter TRs ({df['ls'].mean()*100:.1f}%)")

    print(f"\nDone. wrote {done}, skipped {skipped} (already present), failed {failed}.")
    print(f"Annotations: {PREP_LAUGHTER_ANN_DIR}")


if __name__ == "__main__":
    main()
