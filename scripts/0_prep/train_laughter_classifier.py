"""
train_laughter_classifier.py — train the PRIMARY laughter classifier (Clf-C)
============================================================================
Reconstructed trainer for the revision's improved laughter classifier (Clf-C,
reviewer R2.1c). An 86-feature Random Forest trained on HUMAN annotations from
TWO episodes (s01e01a + s04e09a). Its predictions feed the laughter-ISC analysis.

  input   : data/0_prep/laughter_training/{s01e01a,s04e09a}_manual.csv  (human labels)
            episode audio from config.STIMULI_ROOT (native 48 kHz, T7 drive)
  output  : data/0_prep/laughter_classifier.pkl   (RandomForestClassifier)

PROVENANCE — read data/0_prep/laughter_training/PROVENANCE.txt.
The original training script was not saved; this is a reconstruction: exact
hyperparameters (config.LAUGHTER_RF_PARAMS, recovered from the pickled model) + the
86-dim feature extractor preserved verbatim in laughter_features.py.

IMPORTANT — the reconstruction DOES NOT reproduce the published model. Measured
2026-08-02: 94.44% label agreement with the carried clf_C_both.pkl on the training
TRs, despite identical hyperparameters, random_state and features (library-version
drift). The carried model is therefore AUTHORITATIVE — it produced every published
annotation and every number in the paper — and this script will not overwrite it.
The original 32-feature model is the c_controls comparison.

Usage
  python train_laughter_classifier.py            # report the existing model, do nothing
  python train_laughter_classifier.py --verify   # retrain to a temp file and compare
  python train_laughter_classifier.py --force    # retrain to a SIDE file
                                                 # (laughter_classifier_reconstructed.pkl);
                                                 # never overwrites the authoritative model
"""

import argparse
import re
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    STIMULI_ROOT, PREP_LAUGHTER_MODEL, PREP_LAUGHTER_TRAIN_DIR,
    LAUGHTER_TRAIN_EPISODES, LAUGHTER_RF_PARAMS, LAUGHTER_N_FEATURES,
)
from laughter_features import extract_features_np, N_FEATURES


def episode_mp3(ep: str) -> Path:
    season = int(re.search(r"s(\d+)", ep).group(1))
    return STIMULI_ROOT / f"s{season}" / f"friends_{ep}.mp3"


def build_dataset(episodes: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Extract 86-dim features for every human-labelled TR across all episodes."""
    import librosa
    X, y = [], []
    for ep in episodes:
        manual = PREP_LAUGHTER_TRAIN_DIR / f"{ep}_manual.csv"
        mp3    = episode_mp3(ep)
        if not manual.exists():
            sys.exit(f"Missing human labels: {manual}")
        if not mp3.exists():
            sys.exit(f"Missing audio (T7 drive not mounted?): {mp3}")
        df = pd.read_csv(manual)
        audio, sr = librosa.load(str(mp3), sr=None, mono=True)
        n_tot = len(audio)
        print(f"  {ep}: {len(df)} labelled TRs  ({df['ls'].mean()*100:.1f}% laughter)  sr={sr}")
        for _, row in df.iterrows():
            start = int(row["onset_ms"]  / 1000 * sr)
            end   = min(int(row["offset_ms"] / 1000 * sr), n_tot)
            X.append(extract_features_np(audio[start:end], sr))
            y.append(int(row["ls"]))
    return np.asarray(X), np.asarray(y)


def train(episodes: list[str]) -> "RandomForestClassifier":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import make_scorer, cohen_kappa_score

    print(f"Building training set from {len(episodes)} episodes: {', '.join(episodes)}")
    X, y = build_dataset(episodes)
    assert X.shape[1] == LAUGHTER_N_FEATURES == N_FEATURES, \
        f"feature count {X.shape[1]} != expected {LAUGHTER_N_FEATURES}"
    print(f"  Total: {len(X)} samples, {X.shape[1]} features, "
          f"{y.sum()} laughter ({y.mean()*100:.1f}%)")

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf  = RandomForestClassifier(**LAUGHTER_RF_PARAMS)
    acc = cross_val_score(rf, X, y, cv=cv, scoring="balanced_accuracy")
    kap = cross_val_score(rf, X, y, cv=cv, scoring=make_scorer(cohen_kappa_score))
    print(f"  5-fold CV  balanced_acc: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"  5-fold CV  kappa:        {kap.mean():.3f} ± {kap.std():.3f}")

    rf.fit(X, y)                     # fit on all human-labelled data
    return rf


def describe(model) -> str:
    p = model.get_params()
    return (f"RF: n_estimators={p['n_estimators']}, class_weight={p['class_weight']}, "
            f"random_state={p['random_state']}, n_features_in_={model.n_features_in_}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force",  action="store_true", help="retrain even if the model exists")
    ap.add_argument("--verify", action="store_true",
                    help="retrain to a temp file and compare against the existing model")
    args = ap.parse_args()

    if args.verify:
        if not PREP_LAUGHTER_MODEL.exists():
            sys.exit("No existing model to verify against.")
        existing = joblib.load(PREP_LAUGHTER_MODEL)
        print(f"Existing (authoritative) model: {describe(existing)}")
        remade = train(LAUGHTER_TRAIN_EPISODES)
        print(f"Reconstructed model:            {describe(remade)}")
        # compare on the training features (deterministic given same feature matrix)
        X, ytrue = build_dataset(LAUGHTER_TRAIN_EPISODES)
        agree = (existing.predict(X) == remade.predict(X)).mean()
        print(f"\nLabel agreement (reconstructed vs authoritative) on training TRs: "
              f"{agree*100:.2f}%")
        print("  (Authoritative clf remains data/0_prep/laughter_classifier.pkl; "
              "reconstruction NOT written.)")
        return

    if PREP_LAUGHTER_MODEL.exists():
        model = joblib.load(PREP_LAUGHTER_MODEL)
        if not args.force:
            print(f"Model already present (carried over; see laughter_training/PROVENANCE.txt):"
                  f"\n  {PREP_LAUGHTER_MODEL}\n  {describe(model)}")
            return
        # --force used to overwrite in place. It must not: the carried clf_C_both.pkl
        # is AUTHORITATIVE (it produced every published annotation), and this trainer
        # is a reconstruction that does NOT reproduce it — measured 2026-08-02 at
        # 94.44% label agreement on the training TRs, despite identical
        # hyperparameters, random_state and features (library-version drift). Silently
        # replacing it would invalidate every downstream result with no warning.
        side = PREP_LAUGHTER_MODEL.with_name(PREP_LAUGHTER_MODEL.stem + "_reconstructed.pkl")
        print("REFUSING to overwrite the authoritative classifier.\n"
              f"  authoritative : {PREP_LAUGHTER_MODEL}\n                  {describe(model)}\n"
              "  This trainer is a reconstruction and does not reproduce it "
              "(~94% label agreement).\n"
              f"  Retraining to a side file instead: {side.name}")
        remade = train(LAUGHTER_TRAIN_EPISODES)
        joblib.dump(remade, side)
        X, _ = build_dataset(LAUGHTER_TRAIN_EPISODES)
        agree = (model.predict(X) == remade.predict(X)).mean()
        print(f"\nWrote {side}\n  {describe(remade)}\n"
              f"  agreement with authoritative on training TRs: {agree*100:.2f}%\n"
              "  The authoritative model is unchanged. To genuinely re-derive the\n"
              "  classifier, delete it deliberately and re-run — but note that every\n"
              "  carried annotation and published number came from the authoritative one.")
        sys.exit(1)

    print("No classifier present — training from scratch.\n"
          "  WARNING: this is a RECONSTRUCTED trainer. The published Clf-C was trained\n"
          "  with a different library build and is not reproduced exactly (~94% label\n"
          "  agreement). Downstream results will drift from the published ones.")
    PREP_LAUGHTER_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model = train(LAUGHTER_TRAIN_EPISODES)
    joblib.dump(model, PREP_LAUGHTER_MODEL)
    print(f"\nSaved: {PREP_LAUGHTER_MODEL}\n  {describe(model)}")


if __name__ == "__main__":
    main()
