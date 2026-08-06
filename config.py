"""
Central configuration for the Friends fMRI revision analysis pipeline (v3, clean).

All paths and analysis parameters live here. Scripts import from this module and
never hardcode paths. Outputs follow the block structure:
    data/0_prep/  data/a_isc_stability/  data/b_laughter/  data/c_controls/

Primary pipeline is fMRIPrep. The old H5 pipeline is corroborating evidence only.
Some data are on external folders
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

REVISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT    = REVISION_DIR.parent

# --- Bundled reference assets (small; live inside the code package) ---------
# The analysis+figure layer only needs the MNI template and the Melbourne subcortex
# atlas from the old raw folders; those are copied into CODE/assets/ so reproduction
# is self-contained and does not depend on the (retired) raw source folders.
ASSETS_DIR       = REVISION_DIR / "assets"

# --- Read-only raw source folders (archived onto the external T7 drive to keep the local
#     project lean; used only by the 0_prep stages, which re-derive the carried
#     timeseries/annotations and need the external BOLD/stimuli on T7 anyway). Requires T7
#     mounted; the analysis+figure layer never touches these (it runs from bundled assets +
#     carried data). ---
T7_ARCHIVE       = Path("/Volumes/T7 Shield/friends/paper_OLD")
ORIGINAL_DIR     = T7_ARCHIVE / "01_ORIGINAL_GITHUB"
RAW_FMRI_DIR     = ORIGINAL_DIR / "data" / "00_raw_fmri"   # H5 pipeline parcellated npy (corroboration)
LAUGHTER_ANN_DIR = ORIGINAL_DIR / "data" / "02_laughter" / "laughter_annotations"
# NOTE: the ORIGINAL 32-feature Gemini-trained baseline model lives (read-only) at
#   ORIGINAL_DIR/"data"/"02_laughter"/"laughter_classifier_ model.pkl"  (embedded space in
# the published filename). No v3 script loads it directly — classifier_validation.py reads
# its precomputed predictions from PREP_LAUGHTER_TRAIN_DIR/*_all_classifiers.csv — so the
# brittle space-path is intentionally not exposed as a constant here.
MNI_BG_IMG       = ASSETS_DIR / "MNI152_T1_2mm.nii.gz"                          # bundled (used by plotting)
SCHAEFER_ATLAS   = ORIGINAL_DIR / "data" / "Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"  # unused at runtime (nilearn fetch_atlas is used); in OLD/

# Stimulus media (external T7 drive): STIMULI_ROOT / s{N} / friends_{episode}.{mp3,mkv}
STIMULI_ROOT     = Path("/Volumes/T7 Shield/friends/stimuli")

# CNeuroMod raw (fMRIPrep source + confounds) — retired to OLD/. CONFOUNDS_DIR below is the
# only sizeable raw input the analysis layer can use (head_motion --force); the carried
# motion_by_episode.csv means head_motion --plot-only reproduces the result without it.
CNEUROMOD_DIR    = T7_ARCHIVE / "02_CNEUROMOD_RAW"
CONFOUNDS_DIR    = CNEUROMOD_DIR / "confounds"
BOLD_PREPROC_DIR = CNEUROMOD_DIR / "bold_preprocessed"
TRANSCRIPTS_DIR  = CNEUROMOD_DIR / "data" / "03_transcripts"

# --- fMRIPrep extraction sources (used only by 0_prep/extract_timeseries.py) --
# BOLD is fetched from the CONP HTTP mirror one file at a time (fetch-extract-drop).
# The full run requires ~1.45 TB of transient BOLD; the interim timeseries this
# produces are carried over from 03_REVISION (see PROVENANCE.txt in the output dir),
# so a normal run of extract_timeseries.py is a cheap skip-if-exists no-op.
BIDS_ROOT           = Path("/Volumes/T7 Shield/friends/conp-dataset/projects/"
                           "cneuromod_processed/fmriprep/friends")
BOLD_CACHE          = Path("/Volumes/T7 Shield/friends/bold_cache")
CONP_BASE           = ("https://sftp.conp.ca/users/cneuromod/ria-conp/"
                       "cbf/d6482-5670-4f39-ba1b-9623f1f460b8/annex/objects")
MELBOURNE_ATLAS_DIR = ASSETS_DIR / "melbourne"                                  # bundled (subcortical atlas)

# Template source (old tangled revision folder — archived on T7; provenance only, not used at runtime)
OLD_REVISION_DIR = T7_ARCHIVE / "03_REVISION"

# --- Outputs (this package) -------------------------------------------------
DATA_DIR    = REVISION_DIR / "data"
PREP_DIR    = DATA_DIR / "0_prep"
PREP_TIMESERIES_DIR = PREP_DIR / "fmriprep_timeseries"   # (4, n_TRs, 1032) per episode
PREP_LAUGHTER_MODEL = PREP_DIR / "laughter_classifier.pkl"   # PRIMARY = Clf-C (86-feat RF)
PREP_LAUGHTER_TRAIN_DIR = PREP_DIR / "laughter_training"     # human labels + Clf-A/B (comparison)
PREP_LAUGHTER_ANN_DIR   = PREP_DIR / "laughter_annotations"  # per-episode Clf-C predictions
LAUGHTER_INTENSITY_DIR  = DATA_DIR / "b_laughter" / "laughter_intensity"  # per-TR intensity tiers (rms/prob/humor)
HUMOR_CLASSIFICATION_DIR = DATA_DIR / "b_laughter" / "humor_classification"  # LLM humor content (Gemini)
AV_ENERGY_DIR = DATA_DIR / "c_controls" / "av_energy"  # per-TR acoustic RMS + visual motion energy
A_DIR       = DATA_DIR / "a_isc_stability"
A_ISC_FMRIPREP_DIR = A_DIR / "isc_fmriprep"   # PRIMARY overall-ISC maps (pairwise-median)
A_ISC_H5_DIR       = A_DIR / "isc_h5"         # H5-pipeline ISC carried over (supplement/corroboration)
B_DIR       = DATA_DIR / "b_laughter"
C_DIR       = DATA_DIR / "c_controls"

# Interim analysis plots emitted by the pipeline (quick-look + source panels for the paper).
# NOT the paper figures — those are assembled under CODE/figures/ (the single "figures" folder).
ANALYSIS_PLOTS_DIR = REVISION_DIR / "results" / "analysis_plots"
TABLES_DIR  = REVISION_DIR / "results" / "tables"
A_FIG_DIR   = ANALYSIS_PLOTS_DIR / "a_isc_stability"   # Block A analysis plots (overall ISC + stability)
B_FIG_DIR   = ANALYSIS_PLOTS_DIR / "b_laughter"        # Block B analysis plots (laughter modulation)
C_FIG_DIR   = ANALYSIS_PLOTS_DIR / "c_controls"        # Block C analysis plots (control analyses)

# ---------------------------------------------------------------------------
# fMRI data parameters
# ---------------------------------------------------------------------------

N_PARCELS   = 1000     # Schaefer 2018, 1000 parcels, 7 networks
N_MELBOURNE = 32       # Melbourne S2 subcortical (bilateral)
N_ROIS_UNIFIED = N_PARCELS + N_MELBOURNE   # 1032 ROIs in unified fMRIPrep timeseries
TR_SEC      = 1.49     # repetition time in seconds
SUBJECTS    = ["sub-01", "sub-02", "sub-03", "sub-05"]   # note: no sub-04 in dataset
N_SUBJECTS  = len(SUBJECTS)

# Episode-segments withheld from the Block B (laughter) analyses.
#
# PROVENANCE — this list is INHERITED from the original H5/Algonauts release, where
# these segments lacked scans for one or more subjects. It is the reason the laughter
# annotations cover 280 segments: predict_laughter.py skips this list, and the
# annotated set is exactly the original submission's episode set (280 = the 278 of
# these that also have fMRIPrep timeseries, plus s01e01a/b, which were annotated as
# classifier training material but were not extracted under fMRIPrep).
#
# IMPORTANT — the per-line reasons below describe the H5 release, NOT the fMRIPrep one.
# Under fMRIPrep re-extraction (which includes runs the H5 release did not), 8 of these
# 12 segments in fact have usable data from all four viewers: s04e13a, s05e20a/b,
# s06e03b, s06e24a–d. They are retained here anyway, deliberately, so that the
# reprocessed analyses cover the same material as the original submission and the two
# pipelines stay directly comparable. Conversely, this list is NOT a map of fMRIPrep
# data completeness: 14 segments outside it have only three usable viewers and do enter
# Block B, where estimates are averaged over the viewers actually present.
# Verified 2026-08-02 against the carried timeseries.
EXCLUDED_EPISODES = [
    "s04e01a", "s04e01b",   # H5: sub-05 missing
    "s04e13a", "s04e13b",   # H5: sub-05 missing
    "s05e20a", "s05e20b",   # H5: sub-02 missing
    "s06e03a", "s06e03b",   # H5: one subject missing
    "s06e24a", "s06e24b", "s06e24c", "s06e24d",  # H5: incomplete
]

# ---------------------------------------------------------------------------
# Laughter classifier / segmentation parameters
# ---------------------------------------------------------------------------

SEGMENT_DURATION_SEC  = 1.49   # audio snippet length for classifier
AUDIO_NATIVE_SR       = 48000  # native sample rate — DO NOT resample before features
HRF_SHIFT_TRS         = 3      # HRF lag (in TRs) applied to laughter onsets
EVENT_DURATION_TRS    = 4      # TRs to include after each laughter onset
MIN_NONEVENT_DURATION = 4      # minimum non-laughter segment length (TRs)

# --- PRIMARY laughter classifier: Clf-C (retrained on human labels) ----------
# The revision retrained an 86-feature Random Forest on hand-annotated TRs from
# TWO episodes (s01e01a + s04e09a) — this is the PRIMARY classifier whose
# predictions feed the laughter-ISC analysis. The original 32-feature Gemini-
# trained model is kept as a comparison baseline in c_controls (see PORTING_PLAN.md).
LAUGHTER_TRAIN_EPISODES = ["s01e01a", "s04e09a"]   # episodes with human labels
LAUGHTER_N_FEATURES     = 86                       # MFCC(13)+Δ+Δ² mean/std + sc/sb/zcr/rms mean/std
LAUGHTER_RF_PARAMS      = dict(n_estimators=300, class_weight="balanced",
                               random_state=42, n_jobs=-1)
LAUGHTER_PROB_THRESHOLD = 0.5                      # prob ≥ threshold ⇒ laughter TR

# ---------------------------------------------------------------------------
# ROI indices — Schaefer 1000-parcel (0-based; parcel ID = index + 1)
# Verified against Neurosynth TPJ map + original code. See WORKFLOW.md.
# ---------------------------------------------------------------------------

ROI_AUDITORY  = 598    # parcel 599: RH_SomMot_18 (Heschl's gyrus), MNI (59,-22,10)

# --- Visual ROIs: TWO regions, deliberately given DISTINCT names ------------
#
# NAMING CAVEAT — READ BEFORE USING EITHER. There is no bare `ROI_VISUAL_LOC`, and
# there must never be one. These are two different parts of visual cortex and the
# manuscript must never call both "visual cortex": Part A / the ROI analyses use
# ROI_VISUAL_LOC, while any early-visual statement uses ROI_VISUAL_V1. A reader
# who sees one label over two regions will read the ISC value and the GLM value as
# the same place. This package has already been bitten by exactly that failure —
# two rTPJ constants once coexisted under one name and put a mismatched ROI value
# into the Part A text (see ROI_RTPJ below), which is why the bare name is omitted
# here: any site that has not been updated fails loudly instead of silently
# picking a region.
#
# ROI_VISUAL_LOC is the one wired into the analyses (ROIS_KEY, GLM, laughter-ISC,
# dose-response, AV-energy, humor-type). ROI_VISUAL_V1 is for reference/text only:
# its laughter activation is already in glm_contrast.csv (all 1032 parcels), so
# quoting it costs no recomputation — but the per-ROI tables (dose_response_*,
# glm_av_energy_summary, glm_per_viewer, *_by_humor_type) store ONLY the selected
# ROIs, so putting V1 in those WOULD require a re-run.
#
# Empirically the two do not dissociate: all 162 Vis-network parcels have a
# positive laughter beta, with only a mild medial->lateral gradient (medial
# |x|<20 beta +0.281, lateral |x|>=40 +0.370; laterality-vs-beta r = 0.32).
# V1 +0.438 (t 27.9) and LOC +0.484 (t 24.7) are close. Report a gradient, not a
# dissociation.
ROI_VISUAL_V1  = 537   # parcel 538: RH_Vis_38, MNI (9,-91,1), 1.2mm from canonical
                       # V1. EARLY visual cortex. ISC 0.303, GLM beta +0.438.
                       # Reference only — not used by any analysis stage.

# ROI_VISUAL_LOC — CHANGED 2026-08-05 (author decision), 545 -> 548.
#
# The inherited parcel 546 (RH_Vis_46) sits at the extreme occipital pole,
# MNI (11,-98,5), 109 voxels. It is a poor stand-in for visual cortex: its overall
# ISC is 0.115, which is the 19th percentile of the visual network and ranks
# 131st of the 162 Vis parcels, while the Vis network mean is 0.203 — the highest
# of the seven networks. Because the auditory (91st pct of SomMot) and rTPJ (95th
# pct of Cont) parcels sit high within their networks, the three-ROI comparison
# read "auditory > rTPJ > visual", inverting the expected sensory ordering purely
# as an artefact of which parcel was picked.
#
# Parcel 549 (RH_Vis_49) is both the HIGHEST-ISC visual parcel (0.435) and a
# lateral one, MNI (49,-66,9), 162 voxels — lateral occipital cortex, around
# LOC/hMT+. Report it as "LOC / visual", NOT as "early visual cortex": it is
# motion- and object-sensitive mid-level visual cortex, not V1. (V1 proper is
# parcel 538 / index 537 at MNI (9,-91,1), ISC 0.303, if an early-visual ROI is
# ever wanted instead.)
#
# All downstream effects keep their sign and strengthen: GLM beta +0.191 -> +0.484,
# laughter-ISC delta_r -0.011 -> -0.034. NOTE the visual GLM beta now EXCEEDS rTPJ
# (+0.28), so Part B should acknowledge that the sensory response is the larger one.
# The internal key stays "visual_cortex" so CSV columns are unchanged.
ROI_VISUAL_LOC    = 548    # parcel 549: RH_Vis_49 (LOC / lateral visual)

# ONE rTPJ definition, used for every analysis in the paper (Part A overall ISC,
# Part B GLM + laughter ISC), so the region is identical across measures.
# (A second candidate, parcel 921 RH_Default_Par_9, 74.8% TPJ overlap, was defined
# during development but never used by any script; it was removed 2026-08-02 because
# having two "rTPJ" constants had produced a mismatched ROI value in the Part A text.)
ROI_RTPJ_LAUGHTER = 842  # parcel 843: RH_Cont_Par_1 (83% TPJ overlap)
ROI_RTPJ          = ROI_RTPJ_LAUGHTER

ROIS_KEY  = {"auditory_cortex": ROI_AUDITORY,
             "visual_cortex":   ROI_VISUAL_LOC,
             "rTPJ":            ROI_RTPJ}

# ---------------------------------------------------------------------------
# Unified fMRIPrep timeseries — 1032 ROIs per episode (PRIMARY pipeline)
#   0–999    : Schaefer 1000 parcels (same ordering as H5)
#   1000–1031: Melbourne S2 subcortical (32 bilateral regions)
# ---------------------------------------------------------------------------

MELB_LABEL_ORDER = [
    "aHIP-rh","pHIP-rh","lAMY-rh","mAMY-rh",
    "THA-DP-rh","THA-VP-rh","THA-VA-rh","THA-DA-rh",
    "NAc-shell-rh","NAc-core-rh","pGP-rh","aGP-rh",
    "aPUT-rh","pPUT-rh","aCAU-rh","pCAU-rh",
    "aHIP-lh","pHIP-lh","lAMY-lh","mAMY-lh",
    "THA-DP-lh","THA-VP-lh","THA-VA-lh","THA-DA-lh",
    "NAc-shell-lh","NAc-core-lh","pGP-lh","aGP-lh",
    "aPUT-lh","pPUT-lh","aCAU-lh","pCAU-lh",
]
DORSAL_STR_IDX  = [1012, 1013, 1014, 1015,   # rh: aPUT, pPUT, aCAU, pCAU
                   1028, 1029, 1030, 1031]    # lh: aPUT, pPUT, aCAU, pCAU
VENTRAL_STR_IDX = [1008, 1009,                # rh: NAc-shell, NAc-core
                   1024, 1025]                # lh: NAc-shell, NAc-core

# ---------------------------------------------------------------------------
# Statistical thresholds
# ---------------------------------------------------------------------------

FDR_ALPHA         = 0.05
N_PERMUTATIONS    = 1000
VIZ_THRESHOLD_ISC = 0.10
