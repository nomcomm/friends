# Friends fMRI — Reproducible Analysis & Figure Package (CODE)

Self-contained package that **reproduces every result and figure** in the revised
manuscript. It runs locally with no external drive needed for the analysis or figures;
the raw source data is archived on the external T7 drive and only matters for optional
from-raw re-derivation (see below).

Sibling folder `../MANUSCRIPT/` holds the paper (`revision_draft.md`,
`revision_supplement.md`, `response_to_reviewers.md`, `results_register.md`).

## Principle

The folder tree tells the story. Execution order comes from the block prefix
(`0_ → a_ → b_ → c_`), and every script is named for **what it does**. `data/` mirrors
`scripts/` exactly: a script in `scripts/<block>/` writes only to `data/<block>/`.

## Structure

```
config.py            all paths & parameters, in one place
run_all.py           the orchestrator — defines & runs the pipeline in order
scripts/
  0_prep/            foundations (timeseries, classifier, annotations)
  a_isc_stability/   BLOCK A — stable shared ISC at scale (Part A)
  b_laughter/        BLOCK B — laughter modulation: GLM (primary) + ISC + content
  c_controls/        BLOCK C — control analyses (supplement)
data/                mirrors scripts/ — carried inputs + derived outputs
assets/              bundled reference atlases (MNI template, Melbourne subcortex)
results/
  tables/
  analysis_plots/    interim analysis plots the pipeline emits (quick-look + source panels)
figures/             THE paper figures — assembles panels into figure1–3 + figS1–8
  scripts/ panels/ output/ figure_plan.md
```

Note the two distinct roles: `results/analysis_plots/` are pipeline byproducts; `figures/`
is where the **paper** figures are made (it stages a curated subset of the analysis plots
and composes the final figures). The blocks map 1:1 to `../MANUSCRIPT/results_register.md`.

## How to run

```bash
python run_all.py                 # reproduce all results (regenerates data/ + analysis_plots/)
python run_all.py --list          # print the pipeline stages
bash figures/scripts/build_all.sh # assemble the paper figures into figures/output/
```

The cheap analysis stages are deterministic: a `--force` re-run reproduces every number
**bit-for-bit** (verified 2026-07-31 in an isolated clean-room copy; max|Δ| = 0 across all
key outputs).

## Conventions

- Each script is config-driven (`config.py`) and self-contained; no hardcoded paths.
- Scripts **skip if their output exists** (safe to re-run); most also accept `--force`,
  and the plotting/summary stages accept `--plot-only` (re-render from saved outputs).
- Primary pipeline = **fMRIPrep**; the original H5 pipeline is corroborating evidence only.
- Primary laughter classifier = **Clf-C** (86-feature RF, human labels s01e01a + s04e09a; κ ≈ 0.6).

## What's carried vs. what regenerates

The expensive raw stages were run once; their outputs are **carried** in `data/` (each
subfolder has a `PROVENANCE.txt`). A normal run therefore reproduces the analysis+figures
from those carried intermediates:

| stage | cost | needs |
|---|---|---|
| `a_*`, `b_*` (parcel GLM/ISC/striatum/dose/season/av-energy/humor-type), `c_*` controls | **seconds–minutes** | only carried inputs — cheap, deterministic |
| `0_prep/extract_timeseries` | ~1.5 TB, hours | T7 + CONP mirror (timeseries carried → normal run skips) |
| `0_prep/predict_laughter` | minutes | T7 audio (model + annotations carried → skips) |
| `humor_classification` | LLM API | Gemini key (per-TR labels carried → skips) |
| `glm_contrast_voxelwise.py` | hours, ~1.5 TB read | T7 BOLD; **not in `run_all`** — heavy voxelwise step. Its *result* (`data/b_laughter/glm_voxelwise/group_tmap*.nii.gz`) is carried locally and feeds Figure S1; only the per-voxel `betas`/`betas_motion` (symlinked → T7) are external. |

## Raw sources (archived on T7)

To keep the local project lean, the raw source folders are archived on the external drive
at `/Volumes/T7 Shield/friends/paper_OLD/` (`01_ORIGINAL_GITHUB`, `02_CNEUROMOD_RAW`, the
old `03_REVISION*` folders, etc.). `config.py` points `ORIGINAL_DIR` / `CNEUROMOD_DIR` /
`CONFOUNDS_DIR` there. **The analysis and figures never touch them** — they run from the
bundled `assets/` and the carried `data/`. T7 is needed only for a from-raw re-derivation
(e.g., `head_motion.py --force` reads the confounds; `head_motion.py --plot-only`
reproduces the control from the carried result with no T7).

## Inherited numbers (not regenerated here)

`scripts/verify_claims.py` runs last and prints every manuscript number the package
produces. It closes with an explicit **INHERITED** list: published values whose
source code is the original submission's notebooks, which are not in this tree.
They are reported as published and are *not* recomputed, because recomputing them
with this package's methods gets close but not equal — substituting those numbers
would silently contradict the printed supplement. The list covers the H5
laughter-ISC ROI table (and its 474/1000, and the r = 0.961 pipeline agreement),
the legacy-classifier event/TR comparison, the boundary and alternative-detector
κ values, the humor schema-adherence figures, and Figure S3.

H5 *spatial stability* (0.867 ± 0.030) was on that list until
`c_controls/h5_corroboration.py` was added; it reproduces exactly from the carried
`isc_h5/` maps and is now printed as block **C5**.

## Notes / latent items

- Rebuilding `data/` fully from raw is **not** one command (the carry-over was done once,
  per-folder `PROVENANCE.txt`); the reproducible unit is analysis + figures from carried data.
- **AppleDouble hazard:** the T7 drive is exFAT and accrues `._` files; any script that
  `pathlib.glob("*.npy")` over a T7 path must skip names starting with `._`
  (already handled in `glm_contrast_voxelwise.py`).
- `config.EXCLUDED_EPISODES` vs. per-episode coverage: a mismatch was flagged in
  `qc_coverage.py` and left as-is (does not affect the reported analyses).
