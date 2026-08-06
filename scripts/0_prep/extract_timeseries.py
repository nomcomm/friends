"""
extract_timeseries.py — unified fMRIPrep timeseries extraction (0_prep foundation)
==================================================================================
Ported from 03_REVISION/scripts/12_extract_fmriprep_timeseries.py (the *correct*
extraction — the earlier 10_… script used wrong control-ROI indices; see WORKFLOW.md).

Extracts every cortical + subcortical ROI in one pass, for all episodes:

  data/0_prep/fmriprep_timeseries/task-{episode}.npy
    shape  : (4, n_TRs, 1032)  float32
    axis 0 : subjects  [sub-01, sub-02, sub-03, sub-05]   (NaN-padded if missing)
    axis 1 : TRs       (variable per episode)
    axis 2 : ROIs      0–999    Schaefer 1000-parcel 7-network atlas
                       1000–1031 Melbourne S2 subcortical (32 bilateral regions)

  data/0_prep/fmriprep_timeseries/roi_labels.csv
    columns: roi_idx, roi_id, label, source, network, hemi   (all 1032 ROIs)

KEY ROI INDICES (0-based, axis 2) — see config.py:
  842 rTPJ (parcel 843, RH_Cont_Par_1) — the ONE rTPJ definition, used for every
      analysis in the paper (Part A overall ISC, Part B GLM + laughter ISC), so the
      region is identical across measures. A second candidate (920 / parcel 921,
      RH_Default_Par_9) was defined during development but never used by any script;
      it was removed 2026-08-02 because having two "rTPJ" constants had produced a
      mismatched ROI value in the Part A text. Do not reintroduce it.
  545 visual cortex             598 auditory cortex
  Subcortical groupings: config.DORSAL_STR_IDX, config.VENTRAL_STR_IDX

PROCESSING
  Fetch BOLD from CONP HTTP mirror one file at a time (fetch-extract-drop),
  parcel means via np.add.reduceat, detrend + zscore_sample per run,
  drop BOLD after extraction. Resume-safe: skips episodes whose .npy exists.

NOTE ON REUSE
  The 290 interim .npy files + roi_labels.csv are CARRIED OVER verbatim from
  03_REVISION (see fmriprep_timeseries/_provenance/PROVENANCE.txt). A normal run
  is therefore a cheap skip-if-exists no-op that just verifies they are present.
  Regeneration from scratch needs ~1.45 TB transient BOLD and the T7 drive.

Usage
  python extract_timeseries.py            # 2 parallel workers (default)
  python extract_timeseries.py --workers 4
  python extract_timeseries.py --workers 1   # sequential / debug
"""

import sys
import re
import json
import glob
import argparse
import subprocess
import tempfile
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image as nlimage
from nilearn.signal import clean

# ── config (single source of paths & parameters) ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (
    SUBJECTS, BIDS_ROOT, BOLD_CACHE, CONP_BASE, MELBOURNE_ATLAS_DIR,
    PREP_TIMESERIES_DIR, N_PARCELS, N_MELBOURNE, N_ROIS_UNIFIED,
)

N_SCHAEFER = N_PARCELS          # 1000
N_ROIS     = N_ROIS_UNIFIED     # 1032
OUT_DIR    = PREP_TIMESERIES_DIR


# ── ROI label table ────────────────────────────────────────────────────────────
def build_roi_labels() -> pd.DataFrame:
    """Build and return the 1032-row ROI label lookup table."""
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2, verbose=0)
    sch_labels = [l.decode() if isinstance(l, bytes) else l for l in sch.labels]

    rows = []
    for idx, lbl in enumerate(sch_labels):
        parts = lbl.split("_")
        hemi  = parts[1] if len(parts) > 1 else ""
        net   = parts[2] if len(parts) > 2 else ""
        rows.append({"roi_idx": idx, "roi_id": idx + 1,
                     "label": lbl, "source": "Schaefer2018_1000P_7N",
                     "network": net, "hemi": hemi})

    melb_labels = (MELBOURNE_ATLAS_DIR / "Tian_Subcortex_S2_3T_label.txt"
                   ).read_text().strip().splitlines()
    for i, lbl in enumerate(melb_labels):
        hemi = "rh" if lbl.endswith("-rh") else "lh"
        rows.append({"roi_idx": N_SCHAEFER + i, "roi_id": i + 1,
                     "label": lbl, "source": "Melbourne_S2_3T",
                     "network": "subcortical", "hemi": hemi})

    return pd.DataFrame(rows)


# ── label arrays (built once from the first BOLD, thread-safe) ──────────────────
_label_arrays: dict | None = None
_label_lock   = threading.Lock()


def get_label_arrays(ref_bold_path: Path) -> dict:
    """
    Build sorted label arrays for fast reduceat-based extraction, once, from the
    first BOLD file. Returns precomputed sort indices + boundaries for both atlases,
    enabling O(N_voxels) extraction with np.add.reduceat (no Python loops).
    """
    global _label_arrays
    if _label_arrays is not None:
        return _label_arrays

    with _label_lock:
        if _label_arrays is not None:
            return _label_arrays

        ref_img = nib.load(str(ref_bold_path))
        ref3d   = nib.Nifti1Image(ref_img.get_fdata(dtype=np.float32)[..., 0],
                                  ref_img.affine, ref_img.header)

        # Schaefer: resample to BOLD space
        sch     = datasets.fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2, verbose=0)
        sch_r   = nlimage.resample_to_img(sch.maps, ref3d, interpolation="nearest")
        sch_lbl = sch_r.get_fdata(dtype=np.float32).flatten().astype(np.int32)

        # Melbourne: resample to BOLD space (already MNI152NLin2009cAsym 2mm)
        melb_r   = nlimage.resample_to_img(
            str(MELBOURNE_ATLAS_DIR / "Tian_Subcortex_S2_3T_2009cAsym.nii.gz"),
            ref3d, interpolation="nearest")
        melb_lbl = melb_r.get_fdata(dtype=np.float32).flatten().astype(np.int32)

        def _build_reduceat_info(lbl_flat, n_labels):
            brain_idx  = np.where(lbl_flat > 0)[0]
            brain_lbl  = lbl_flat[brain_idx]
            sort_order = np.argsort(brain_lbl, kind="stable")
            sorted_idx = brain_idx[sort_order]
            sorted_lbl = brain_lbl[sort_order]
            boundaries = np.searchsorted(sorted_lbl, np.arange(1, n_labels + 1))
            counts = np.diff(np.append(boundaries,
                                       np.searchsorted(sorted_lbl, n_labels + 1)))
            counts = np.maximum(counts, 1)   # avoid division by zero
            return sorted_idx, boundaries, counts

        sch_idx, sch_bounds, sch_counts = _build_reduceat_info(sch_lbl,  N_SCHAEFER)
        mel_idx, mel_bounds, mel_counts = _build_reduceat_info(melb_lbl, N_MELBOURNE)

        print(f"  Reduceat arrays ready: Schaefer {int((sch_lbl>0).sum())} vox, "
              f"Melbourne {int((melb_lbl>0).sum())} vox")

        _label_arrays = {
            "sch_idx": sch_idx, "sch_bounds": sch_bounds, "sch_counts": sch_counts,
            "mel_idx": mel_idx, "mel_bounds": mel_bounds, "mel_counts": mel_counts,
        }
    return _label_arrays


# ── download helpers ─────────────────────────────────────────────────────────────
def get_conp_url(annex_ptr: Path) -> str:
    key = annex_ptr.read_text().strip().lstrip("/annex/objects/")
    res = subprocess.run(
        ["git", "-C", str(BIDS_ROOT), "annex", "examinekey", "--json", key],
        capture_output=True, text=True, check=True)
    hdir = json.loads(res.stdout)["hashdirmixed"]
    return f"{CONP_BASE}/{hdir}{key}/{key}"


def download_file(url: str, dest: Path) -> bool:
    """Download url → dest. Retries 3× on 5xx errors (30 / 90 / 300 s delays)."""
    import time
    from urllib.error import HTTPError
    delays = [30, 90, 300]
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return True
        except HTTPError as e:
            if dest.exists():
                dest.unlink()
            if e.code in (502, 503, 504) and attempt < 3:
                wait = delays[attempt]
                print(f"    Server {e.code} — retry {attempt+1}/3 in {wait}s", flush=True)
                time.sleep(wait)
            else:
                print(f"    Download failed: {e}", flush=True)
                return False
        except Exception as e:
            print(f"    Download failed: {e}", flush=True)
            if dest.exists():
                dest.unlink()
            return False
    return False


# ── extraction ─────────────────────────────────────────────────────────────────
def extract_subject(bold_path: Path) -> np.ndarray | None:
    """
    Fast extraction via np.add.reduceat — sort voxels by parcel label (precomputed),
    sum within each parcel in one C-level pass, divide by parcel size. Then
    detrend + z-score per run. O(N_voxels × N_TRs).
    """
    try:
        la        = get_label_arrays(bold_path)
        bold_d    = nib.load(str(bold_path)).get_fdata(dtype=np.float32)  # (X,Y,Z,T)
        n_trs     = bold_d.shape[3]
        bold_flat = bold_d.reshape(-1, n_trs)                              # (N_vox, T)

        ts = np.zeros((n_trs, N_ROIS), dtype=np.float32)

        s_bold = bold_flat[la["sch_idx"], :]
        sums   = np.add.reduceat(s_bold, la["sch_bounds"])     # (1000, T)
        ts[:, :N_SCHAEFER] = (sums / la["sch_counts"][:, None]).T

        m_bold = bold_flat[la["mel_idx"], :]
        msums  = np.add.reduceat(m_bold, la["mel_bounds"])     # (32, T)
        ts[:, N_SCHAEFER:] = (msums / la["mel_counts"][:, None]).T

        ts = clean(ts, detrend=True, standardize="zscore_sample")
        return ts.astype(np.float32)
    except Exception as e:
        print(f"    Extraction error: {e}")
        return None


# ── episode inventory ──────────────────────────────────────────────────────────
def find_bold_ptr(subject: str, episode: str) -> Path | None:
    pattern = str(BIDS_ROOT / subject / "*/func" /
                  f"{subject}_*_task-{episode}_space-MNI152NLin2009cAsym"
                  f"_desc-preproc_bold.nii.gz")
    # sorted(): some episodes were scanned in >1 session, so several BOLD runs
    # match. glob.glob() alone returns readdir order (filesystem-dependent), which
    # would make the extracted timeseries non-deterministic. Lowest ses- number
    # wins, matching the run c_controls/head_motion.py reads confounds from.
    matches = sorted(glob.glob(pattern))
    return Path(matches[0]) if matches else None


def all_episodes() -> list[str]:
    eps = set()
    for sub in SUBJECTS:
        for f in (BIDS_ROOT / sub).rglob(
                "*MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"):
            m = re.search(r"task-(s\d+e\d+[a-z]*)", f.name)
            if m:
                eps.add(m.group(1))
    return sorted(eps)


# ── per-subject worker (called in parallel within one episode) ─────────────────
def process_subject(sub: str, episode: str, tmp_dir: Path) -> tuple[int, np.ndarray | None]:
    """
    Resolve + extract one subject's BOLD. Source priority:
      1. BIDS structure (already >500 MB — resolved by a prior download)
      2. bold_cache on T7 (>1 MB fallback)
      3. HTTP download → move to BIDS path (permanently populates BIDS on T7)
    Returns (sub_index, timeseries or None).
    """
    sub_idx = SUBJECTS.index(sub)
    ptr     = find_bold_ptr(sub, episode)

    if ptr is not None and ptr.stat().st_size > 500_000_000:
        print(f"  [{episode}] {sub}: using BIDS file", flush=True)
        return sub_idx, _extract_safe(ptr, episode, sub)

    cached = BOLD_CACHE / sub / f"task-{episode}.nii.gz"
    if cached.exists() and cached.stat().st_size >= 1_000_000:
        print(f"  [{episode}] {sub}: using bold_cache", flush=True)
        return sub_idx, _extract_safe(cached, episode, sub)

    if ptr is None:
        print(f"  [{episode}] {sub}: no BIDS pointer found", flush=True)
        return sub_idx, None
    try:
        url = get_conp_url(ptr)
    except Exception as e:
        print(f"  [{episode}] {sub}: key error: {e}", flush=True)
        return sub_idx, None

    tmp_bold = tmp_dir / f"{sub}_{episode}.nii.gz"
    print(f"  [{episode}] {sub}: downloading...", flush=True)
    if not download_file(url, tmp_bold):
        return sub_idx, None

    try:
        shutil.move(str(tmp_bold), str(ptr))
        bold_path = ptr
    except Exception:
        (BOLD_CACHE / sub).mkdir(parents=True, exist_ok=True)
        shutil.move(str(tmp_bold), str(cached))
        bold_path = cached

    return sub_idx, _extract_safe(bold_path, episode, sub)


def _extract_safe(bold_path: Path, episode: str, sub: str) -> np.ndarray | None:
    try:
        get_label_arrays(bold_path)
        return extract_subject(bold_path)
    except Exception as e:
        print(f"  [{episode}] {sub}: extraction error: {e}", flush=True)
        return None


# ── per-episode worker ─────────────────────────────────────────────────────────
def process_episode(episode: str, tmp_dir: Path) -> tuple[str, str]:
    """Download all 4 subjects in parallel, extract, stack into (4, TRs, 1032),
    NaN-pad missing subjects, save. Resume-safe: skips if .npy already exists."""
    out_path = OUT_DIR / f"task-{episode}.npy"
    if out_path.exists():
        return episode, "skipped"

    subject_ts = [None] * len(SUBJECTS)
    with ThreadPoolExecutor(max_workers=4) as inner:
        futures = {inner.submit(process_subject, sub, episode, tmp_dir): sub
                   for sub in SUBJECTS}
        for fut in as_completed(futures):
            try:
                idx, ts = fut.result()
                subject_ts[idx] = ts
            except Exception as e:
                print(f"  [{episode}] subject error: {e}")

    if not any(ts is not None for ts in subject_ts):
        return episode, "failed"

    n_trs = max(ts.shape[0] for ts in subject_ts if ts is not None)
    out   = np.full((len(SUBJECTS), n_trs, N_ROIS), np.nan, dtype=np.float32)
    for i, ts in enumerate(subject_ts):
        if ts is not None:
            t = min(ts.shape[0], n_trs)
            out[i, :t, :] = ts[:t, :]

    np.save(out_path, out)
    n_ok = sum(ts is not None for ts in subject_ts)
    return episode, f"done ({n_ok}/{len(SUBJECTS)} subjects, {n_trs} TRs)"


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = OUT_DIR / "roi_labels.csv"
    if not labels_path.exists():
        print("Building ROI label table...")
        build_roi_labels().to_csv(labels_path, index=False)
        print(f"  Saved {labels_path.name}")

    # If the BIDS root is not mounted, we can only work with what is already on disk.
    have_bids = BIDS_ROOT.exists()
    episodes  = all_episodes() if have_bids else \
        sorted(p.stem.replace("task-", "") for p in OUT_DIR.glob("task-*.npy"))
    todo = [ep for ep in episodes if not (OUT_DIR / f"task-{ep}.npy").exists()]

    n_on_disk = len(list(OUT_DIR.glob("task-*.npy")))
    print(f"\nEpisodes on disk: {n_on_disk}  |  known: {len(episodes)}  |  "
          f"todo: {len(todo)}  |  BIDS mounted: {have_bids}  |  workers: {args.workers}")
    print(f"Output: {OUT_DIR}\n")

    if not todo:
        print(f"Nothing to do — {n_on_disk} episodes already extracted "
              f"(carried over; see _provenance/PROVENANCE.txt).")
        return

    if not have_bids:
        sys.exit("BIDS root not mounted (T7 drive); cannot fetch missing episodes:\n  "
                 + ", ".join(todo))

    tmp_dir = Path(tempfile.mkdtemp(prefix="friends_fmriprep_"))
    counts  = {"done": 0, "skipped": len(episodes) - len(todo), "failed": 0}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_episode, ep, tmp_dir): ep for ep in todo}
            for fut in as_completed(futures):
                ep = futures[fut]
                try:
                    _, status = fut.result()
                except Exception as e:
                    status = f"exception: {e}"
                    counts["failed"] += 1
                else:
                    if status.startswith("done"):
                        counts["done"] += 1
                    elif status == "failed":
                        counts["failed"] += 1
                total = counts["done"] + counts["skipped"]
                print(f"[{total}/{len(episodes)}] {ep}: {status}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"Done: {counts['done']}  Skipped: {counts['skipped']}  Failed: {counts['failed']}")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
