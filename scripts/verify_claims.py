"""
verify_claims.py — print every number the manuscript quotes, in one place.
=========================================================================
Reads only the pipeline's own outputs under data/ and prints the headline values
cited in ../MANUSCRIPT/{revision_draft,revision_supplement,results_register}.md,
grouped by register block. Nothing is recomputed here: this is a read-out, so a
number that appears in the paper but not below has no reproducible source.

WHY THIS EXISTS. Every value in the paper should be traceable to a script output.
Three values were previously quoted in the manuscript without any script emitting
them (the Part A per-ROI ISC values), and one of them was wrong — the right-TPJ
figure had been written into the visual-cortex slot — with nothing able to catch
it. Run this after `run_all.py` and read it side by side with the manuscript.

  usage:  python scripts/verify_claims.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    A_ISC_FMRIPREP_DIR, B_DIR, C_DIR, PREP_TIMESERIES_DIR, PREP_LAUGHTER_ANN_DIR,
    ANALYSIS_PLOTS_DIR, HUMOR_CLASSIFICATION_DIR, N_PARCELS, TR_SEC, FDR_ALPHA,
    ROI_AUDITORY, ROI_VISUAL_LOC, ROI_RTPJ_LAUGHTER,
)

# "LOC / visual" — the visual ROI is parcel 549 (RH_Vis_49), lateral occipital
# cortex around LOC/hMT+, NOT early visual cortex. See config.ROI_VISUAL_LOC.
ROIS = [("auditory cortex", ROI_AUDITORY), ("LOC / visual", ROI_VISUAL_LOC), ("rTPJ", ROI_RTPJ_LAUGHTER)]


def h(title):
    print(f"\n{title}\n" + "-" * len(title))


def main() -> None:
    print("=" * 72)
    print("  CLAIM CHECK — every manuscript number, from pipeline outputs only")
    print("=" * 72)

    # ---- episode inventory (the three counts that must never be conflated) ----
    h("COUNTS")
    segs = sorted(p.stem.replace("task-", "") for p in PREP_TIMESERIES_DIR.glob("task-*.npy"))
    ann = sorted(p.stem for p in PREP_LAUGHTER_ANN_DIR.glob("*.csv"))
    epA = pd.read_csv(A_ISC_FMRIPREP_DIR / "episodes.csv")
    epB = pd.read_csv(B_DIR / "glm_episodes.csv")
    nA4 = int((epA.n_subjects == 4).sum()); nA3 = int((epA.n_subjects == 3).sum())
    sub = epA[epA.episode.isin(set(epB.episode))]
    print(f"  fMRIPrep timeseries segments : {len(segs)}   ({len({e[:6] for e in segs})} distinct episodes)")
    print(f"  laughter annotations         : {len(ann)}")
    print(f"  Block A (stability)          : {len(epA)}   ({nA4} four-viewer, {nA3} three-viewer)")
    print(f"  Block B (GLM / laughter ISC) : {len(epB)}   "
          f"({int((sub.n_subjects==4).sum())} four-viewer, {int((sub.n_subjects==3).sum())} three-viewer)")

    # ---- A1 / A2 / A3 ----
    h("A1  overall ISC (Part A, descriptive)")
    m = np.load(A_ISC_FMRIPREP_DIR / "isc_all.npy")[:, :N_PARCELS].mean(0)
    labels = pd.read_csv(PREP_TIMESERIES_DIR / "roi_labels.csv")["label"].tolist()[:N_PARCELS]
    net = pd.Series([l.split("_")[2] for l in labels])
    print(f"  mean cortical ISC   r = {m.mean():.3f}      range {m.min():+.3f} .. {m.max():.3f}")
    print(f"  parcels > 0.10      {int((m>.10).sum())}/{N_PARCELS}   > 0.05  {int((m>.05).sum())}   > 0  {int((m>0).sum())}")
    print("  network means:      " + "  ".join(
        f"{k} {v:.2f}" for k, v in net.groupby(net).apply(lambda g: m[g.index].mean()).sort_values(ascending=False).items()))
    # The draft quotes this as "the most strongly synchronized individual parcels
    # (r = 0.43-0.48) lay in lateral temporal, auditory/somatomotor, and visual
    # cortex" — that anatomy is the TOP FIVE (the visual parcel is the 5th). The
    # labels are printed so the range and the anatomical description stay checkable
    # together; quoting the top four instead would drop visual cortex and give 0.44.
    top = np.argsort(m)[::-1][:5]
    print(f"  strongest parcels   r = {m[top[-1]]:.2f} .. {m[top[0]]:.2f}   (top 5)")
    for i in top:
        print(f"      {m[i]:.4f}  {labels[i]}")
    for name, idx in ROIS:
        print(f"    {name:<16} r = {m[idx]:.3f}   ({labels[idx]})")

    h("A2 / A3  spatial stability")
    print("  " + (ANALYSIS_PLOTS_DIR / "a_isc_stability" / "stability_stats.txt").read_text().strip().replace("\n", "\n  "))

    # ---- B1 GLM ----
    h("B1  GLM laughter contrast (PRIMARY Part B)")
    g = pd.read_csv(B_DIR / "glm_contrast.csv")
    for name, idx in ROIS:
        r = g.iloc[idx]
        print(f"  {name:<16} beta = {r.beta:+.4f}   t = {r.t:7.2f}   p_fdr = {r.p_fdr:.2e}")
    print(f"  positive betas {int((g.beta>0).sum())}/{len(g)}   negative {int((g.beta<0).sum())}   "
          f"FDR-sig {int((g.p_fdr<FDR_ALPHA).sum())}")
    print("  " + (B_DIR / "glm_stability_stats.txt").read_text().strip().replace("\n", "\n  "))

    h("B1  per-viewer GLM (R1.1)")
    pv = pd.read_csv(B_DIR / "glm_per_viewer.csv")
    for _, r in pv[pv.roi == "rTPJ"].iterrows():
        print(f"  {r.viewer}  rTPJ beta = {r.beta:+.3f}  t = {r.t:5.2f}  p = {r.p:.1e}  (n={int(r.n_episodes)})")
    cons = pd.read_csv(B_DIR / "glm_per_viewer_consistency.csv").set_index("Unnamed: 0").values
    iu = np.triu_indices(cons.shape[0], k=1)
    print(f"  inter-viewer whole-brain spatial r = {cons[iu].mean():.3f} +/- {cons[iu].std():.3f}")

    # ---- B2 ISC contrast ----
    h("B2  laughter vs non-laughter ISC (supplement)")
    print("  " + pd.read_csv(B_DIR / "isc_summary.csv").to_string(index=False).replace("\n", "\n  "))
    w = pd.read_csv(B_DIR / "wholebrain_contrast.csv"); ws = w[w.p_fdr < FDR_ALPHA]
    print(f"  whole-brain FDR-sig {len(ws)}/{len(w)}   higher-during-laughter {int((ws.delta_r>0).sum())}   "
          f"lower {int((ws.delta_r<0).sum())}")

    # ---- B3 seasons ----
    h("B3  season consistency")
    print("  " + pd.read_csv(B_DIR / "season_consistency.csv").to_string(index=False).replace("\n", "\n  "))

    # ---- B4 humor ----
    h("B4  humor content vs laughter (harvesting)")
    print("  " + (B_DIR / "humor_summary.txt").read_text().strip().replace("\n", "\n  "))

    h("B4  laughter event statistics")
    ev = tr = 0; lens = []
    for f in sorted(PREP_LAUGHTER_ANN_DIR.glob("*.csv")):
        ls = pd.read_csv(f)["ls"].values
        tr += int(ls.sum())
        d = np.diff(np.concatenate(([0], ls, [0])))
        on, off = np.where(d == 1)[0], np.where(d == -1)[0]
        ev += len(on); lens.extend((off - on).tolist())
        mins = None
    total_min = sum(len(pd.read_csv(f)) for f in sorted(PREP_LAUGHTER_ANN_DIR.glob("*.csv"))) * TR_SEC / 60
    print(f"  laugh events {ev:,}   laughter TRs {tr:,}   mean event {np.mean(lens):.2f} TRs   "
          f"rate {ev/total_min:.1f}/min")

    # ---- B5 striatum, B6 dose ----
    h("B5  striatum (GLM vs ISC dissociation)")
    print("  " + pd.read_csv(B_DIR / "striatum_summary.csv").to_string(index=False).replace("\n", "\n  "))

    h("B6  dose-response")
    print("  GLM:\n  " + pd.read_csv(B_DIR / "dose_response_glm.csv").to_string(index=False).replace("\n", "\n  "))
    print("  ISC:\n  " + pd.read_csv(B_DIR / "dose_response_isc.csv").to_string(index=False).replace("\n", "\n  "))

    # ---- C controls ----
    h("C1  head motion")
    print("  " + pd.read_csv(C_DIR / "motion_summary.csv").to_string(index=False).replace("\n", "\n  "))
    md = pd.read_csv(C_DIR / "motion_by_episode.csv")
    from scipy import stats as _s
    r_mb, p_mb = _s.pearsonr(md.motion_isc_delta, md.brain_isc_delta)
    print(f"  motion-ISC vs rTPJ brain-ISC across episodes: r = {r_mb:+.3f}, p = {p_mb:.2f}")

    h("C2  AV-energy control")
    print("  GLM:\n  " + pd.read_csv(B_DIR / "glm_av_energy_summary.csv").to_string(index=False).replace("\n", "\n  "))
    print("  ISC:\n  " + pd.read_csv(C_DIR / "av_energy_summary.csv").to_string(index=False).replace("\n", "\n  "))

    h("C3  humor type (exploratory)")
    print("  " + (B_DIR / "glm_by_humor_type_stats.txt").read_text().strip().replace("\n", "\n  "))

    h("C4  classifier validation")
    print("  " + (C_DIR / "classifier_validation_stats.txt").read_text().strip().replace("\n", "\n  "))

    h("C5  H5-pipeline corroboration (Part A stability)")
    print("  " + (C_DIR / "h5_stability_stats.txt").read_text().strip().replace("\n", "\n  "))

    # ---- voxelwise motion robustness (carried nifti) ----
    h("B1  voxelwise motion robustness (from carried group t-maps)")
    try:
        import nibabel as nib
        a = nib.load(B_DIR / "glm_voxelwise" / "group_tmap.nii.gz").get_fdata()
        b = nib.load(B_DIR / "glm_voxelwise" / "group_tmap_motion.nii.gz").get_fdata()
        k = (a != 0) & (b != 0) & np.isfinite(a) & np.isfinite(b)
        print(f"  bare vs motion-augmented voxelwise r = {np.corrcoef(a[k], b[k])[0,1]:.4f}  "
              f"({int(k.sum()):,} in-mask voxels)")
    except Exception as e:
        print(f"  [skipped: {e}]")

    # ---- the honest remainder: published numbers with no generator here -------
    # Everything above is regenerated from carried inputs. These are not, and are
    # printed so the gap is visible rather than merely absent. Their source code
    # (the original submission's notebooks) is not in this tree, so they are
    # inherited as published; recomputing them here would not reproduce them.
    h("INHERITED — published numbers this package does NOT regenerate")
    for claim, where in [
        ("H5 laughter-ISC ROI table (rTPJ .133/.115 t=3.53; visual .151/.187; auditory .310/.325)",
         "supplement 'H5 pipeline corroboration'"),
        ("474/1000 parcels FDR-significant under H5", "same table"),
        ("pipeline agreement r = 0.961 (H5 vs fMRIPrep contrast maps); Methods 'r ~ 0.96'",
         "supplement Fig S3 + Methods"),
        ("legacy-classifier comparison: 18,814 events, 1.87 TRs, 35,140 laughter TRs, 26.4%",
         "supplement 'Clf-C is more conservative'"),
        ("84-97% of classifier-human disagreements within +/-2 TRs", "supplement boundary analysis"),
        ("Gillick et al. detector kappa = 0.359; LLM kappa = 0.091-0.505", "supplement 'Alternatives evaluated'"),
        ("Figure S3 (staged legacy panel; its source map is not carried)", "supplement Fig S3"),
    ]:
        print(f"  - {claim}\n      ({where})")
    print("  H5 spatial stability USED to be on this list; it is now computed above (C5).")

    # Verified 2026-08-02 to recompute exactly from the carried aggregate, so these
    # are checked here rather than inherited.
    h("C6  humor schema adherence (supplement 'Schema adherence')")
    hj = pd.read_csv(HUMOR_CLASSIFICATION_DIR / "aggregate_humor_by_tr.csv", low_memory=False)
    hp = hj[hj.is_humor.astype(str).isin(["1", "True", "true"])]
    cats = ["Language", "Logic", "Identity"]
    vc = hp.primary_category.value_counts(dropna=False)
    in_schema = int(vc.reindex(cats).fillna(0).sum())
    off = vc.drop([c for c in cats if c in vc.index])
    print(f"  humor-positive TRs {len(hp):,}   in-schema {in_schema:,} "
          f"({100*in_schema/len(hp):.2f}%)   off-schema {int(off.sum())}")
    print(f"  off-schema labels: {dict(off)}")
    print(f"  distinct free-text technique strings: {hj.techniques.nunique()}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
