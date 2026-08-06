"""
laughter_features.py — shared 86-dim audio feature extractor (0_prep)
=====================================================================
Single source of truth for the laughter-classifier feature vector, imported by
BOTH train_laughter_classifier.py and predict_laughter.py so training and
inference are guaranteed identical. Verbatim from the revision's Clf-C pipeline
(03_REVISION/scripts/09a_predict_all_episodes.py :: extract_features_np).

Feature layout (86 dims):
  MFCC(13)  mean+std                = 26
  ΔMFCC(13) mean+std                = 26
  Δ²MFCC(13) mean+std               = 26
  spectral centroid  mean, std      = 2
  spectral bandwidth mean, std      = 2
  zero-crossing rate mean, std      = 2
  RMS energy         mean, std      = 2
                                    ---- 86

Audio MUST be passed at native sample rate (48 kHz for the Friends MP3s) — never
resample before extraction (the 22 kHz-vs-48 kHz bug; see CONSOLIDATION §8).
"""

import numpy as np
import librosa

N_FEATURES = 86


def extract_features_np(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract the 86-dim feature vector from a 1-D audio segment (native sr)."""
    if len(y) < 64:                      # too short → zeros (matches original)
        return np.zeros(N_FEATURES, dtype=np.float32)
    hop   = max(1, len(y) // 20)
    n_fft = min(512, len(y))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop, n_fft=n_fft)
    dm   = librosa.feature.delta(mfcc)
    dm2  = librosa.feature.delta(mfcc, order=2)
    sc   = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
    sb   = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop, n_fft=n_fft)
    zcr  = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)
    rms  = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)

    ms = lambda x: np.hstack([np.mean(x, axis=1), np.std(x, axis=1)])
    return np.hstack([
        ms(mfcc), ms(dm), ms(dm2),
        np.mean(sc), np.std(sc), np.mean(sb), np.std(sb),
        np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms),
    ]).astype(np.float32)
