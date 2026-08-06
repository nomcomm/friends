"""
glm_contrast_voxelwise.py — whole-brain VOXELWISE GLM laughter contrast (Block B)
=================================================================================
Voxelwise counterpart to glm_contrast.py. First-level GLM per subject × episode
on the fMRIPrep preproc BOLD (MNI152NLin2009cAsym 2mm, read from T7), then
second-level one-sample t across episodes. Clf-C annotations.

Two models:
  bare    (default) : SPM-HRF laughter regressor + cosine drift
  motion  (--motion): bare + 6 motion params + framewise displacement (robustness)

Resumable: per subject×episode laughter-beta saved (masked .npy) and skipped on
re-run. Heavy I/O (~1.1 TB read from T7) — run in the background. bare and motion
write to separate beta dirs and separate group maps, so they don't clobber.

OUTPUTS  (data/b_laughter/glm_voxelwise/)
  mask.nii.gz
  betas/{sub}_{ep}.npy            bare        | betas_motion/{sub}_{ep}.npy   motion
  group_tmap.nii.gz               bare        | group_tmap_motion.nii.gz      motion
  progress.log
FIGURE   (results/analysis_plots/b_laughter/)
  fig_glm_contrast_voxelwise.png  bare        | ..._motion.png                motion

Usage
  python glm_contrast_voxelwise.py                  # bare (resumable) then second-level
  python glm_contrast_voxelwise.py --motion         # motion model
  python glm_contrast_voxelwise.py --aggregate-only  # (+ --motion) second-level only
"""

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.maskers import NiftiMasker
from nilearn.masking import compute_epi_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    BIDS_ROOT, CONFOUNDS_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, B_FIG_DIR, TR_SEC,
    EXCLUDED_EPISODES, SUBJECTS,
)

OUT = B_DIR / "glm_voxelwise"
MASK_PATH = OUT / "mask.nii.gz"
LOG = OUT / "progress.log"
MOT = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "framewise_displacement"]


def beta_dir(model):
    return OUT / ("betas" if model == "bare" else "betas_motion")


def group_paths(model):
    tag = "" if model == "bare" else "_motion"
    return OUT / f"group_tmap{tag}.nii.gz", B_FIG_DIR / f"fig_glm_contrast_voxelwise{tag}.png"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def find_bold(sub, ep):
    pat = str(BIDS_ROOT / sub / "*/func" /
              f"{sub}_*_task-{ep}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    m = glob.glob(pat)
    return Path(m[0]) if m and Path(m[0]).stat().st_size > 100_000_000 else None


def confounds(sub, ep, n):
    m = glob.glob(str(CONFOUNDS_DIR / f"{sub}_*_task-{ep}_desc-confounds_timeseries.tsv"))
    if not m:
        return None
    c = pd.read_csv(m[0], sep="\t")[MOT].fillna(0).values
    return c[:n] if len(c) >= n else None


def laughter_blocks(ls):
    out, i, n = [], 0, len(ls)
    while i < n:
        if ls[i] == 1:
            j = i
            while j < n and ls[j] == 1:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


def design_for(ep, n):
    ls = pd.read_csv(PREP_LAUGHTER_ANN_DIR / f"{ep}.csv")["ls"].values[:n]
    bl = laughter_blocks(ls)
    if not bl:
        return None
    ft = np.arange(n) * TR_SEC
    ev = pd.DataFrame({"onset": [o * TR_SEC for o, _ in bl],
                       "duration": [l * TR_SEC for _, l in bl],
                       "trial_type": ["laughter"] * len(bl)})
    return make_first_level_design_matrix(ft, ev, hrf_model="spm",
                                          drift_model="cosine", high_pass=0.01)


def episodes():
    return sorted({m.group(1)
                   for sub in SUBJECTS
                   for f in (BIDS_ROOT / sub).rglob("*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
                   if (m := re.search(r"task-(s\d+e\d+[a-z]*)", f.name))
                   and m.group(1) not in EXCLUDED_EPISODES
                   and (PREP_LAUGHTER_ANN_DIR / f"{m.group(1)}.csv").exists()})


def build_mask(ref_bold):
    if MASK_PATH.exists():
        return
    log("Building brain mask from reference BOLD ...")
    compute_epi_mask(str(ref_bold)).to_filename(str(MASK_PATH))


def first_level(model):
    bdir = beta_dir(model); bdir.mkdir(parents=True, exist_ok=True)
    eps = episodes()
    log(f"First-level [{model}]: {len(eps)} episodes × up to {len(SUBJECTS)} subjects")
    ref = next((find_bold(s, e) for e in eps for s in SUBJECTS if find_bold(s, e)), None)
    if ref is None:
        log("No BOLD found on T7 — aborting."); sys.exit(1)
    build_mask(ref)
    masker = NiftiMasker(mask_img=str(MASK_PATH), standardize="zscore_sample", detrend=True).fit()

    done = 0
    for ep in eps:
        dm = None
        for sub in SUBJECTS:
            out = bdir / f"{sub}_{ep}.npy"
            if out.exists():
                done += 1; continue
            bold = find_bold(sub, ep)
            if bold is None:
                continue
            try:
                Y = masker.transform(str(bold)); n = Y.shape[0]
                if dm is None or dm.shape[0] != n:
                    dm = design_for(ep, n)
                    if dm is None:
                        log(f"  {ep}: no laughter blocks — skip"); break
                li = list(dm.columns).index("laughter")
                X = dm.values
                if model == "motion":
                    cf = confounds(sub, ep, n)
                    if cf is None:
                        log(f"  {sub} {ep}: no confounds — skip"); continue
                    X = np.column_stack([X, cf])
                beta, *_ = np.linalg.lstsq(X, Y[:X.shape[0]], rcond=None)
                np.save(out, beta[li].astype(np.float32)); done += 1
                log(f"  {sub} {ep}: beta saved ({Y.shape[1]} vox, {n} TRs)")
            except Exception as e:
                log(f"  {sub} {ep}: ERROR {e}")
    log(f"First-level [{model}] complete. betas on disk: {done}")


def second_level(model):
    tmap_path, fig_path = group_paths(model)
    masker = NiftiMasker(mask_img=str(MASK_PATH)).fit()
    by_ep = {}
    # exclude macOS AppleDouble "._" sidecar files (created on exFAT/USB drives);
    # pathlib .glob matches dotfiles and np.load chokes on them.
    for f in sorted(p for p in beta_dir(model).glob("*.npy") if not p.name.startswith("._")):
        _, ep = f.stem.split("_", 1)
        by_ep.setdefault(ep, []).append(np.load(f))
    ep_betas = np.array([np.mean(v, axis=0) for v in by_ep.values()])
    log(f"Second-level [{model}]: {ep_betas.shape[0]} episodes × {ep_betas.shape[1]} voxels")
    t, _ = stats.ttest_1samp(ep_betas, 0, axis=0)
    tmap = masker.inverse_transform(np.nan_to_num(t)); tmap.to_filename(str(tmap_path))
    from nilearn import plotting
    fig = plt.figure(figsize=(14, 4))
    plotting.plot_glass_brain(tmap, threshold=3.0, colorbar=True, plot_abs=False,
                              title=f"Voxelwise GLM laughter contrast [{model}] (group t, N={ep_betas.shape[0]})",
                              figure=fig)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    log(f"Saved {tmap_path.name} + {fig_path.name} (|t|>3: {int((np.abs(t) > 3).sum())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", action="store_true", help="motion-augmented model")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    model = "motion" if args.motion else "bare"
    OUT.mkdir(parents=True, exist_ok=True)
    log(f"=== voxelwise GLM [{model}] start ===")
    if not args.aggregate_only:
        first_level(model)
    second_level(model)
    log(f"=== done [{model}] ===")


if __name__ == "__main__":
    main()
