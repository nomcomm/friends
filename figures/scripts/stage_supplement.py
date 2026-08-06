"""
stage_supplement.py — assemble the supplement figures into output/supplement/.

Supplement figures are the analysis-package panels used as-is (functional on-panel
titles are acceptable in a supplement). Supplement text sections are UNNUMBERED
(expressive titles only); figures carry the numbers and are contiguous S1–S8, in
order of appearance in revision_supplement.md:
  S1 GLM robustness (voxelwise) → S2 original ISC result → S3 ISC robustness (2x2)
  → [classifier validation: text-only, no figure] → S4 striatum → S5 dose-response
  → S6 motion → S7 AV-energy → S8 humor-type.
Text-only sections (no figure): classifier validation, per-viewer GLM (= main
Fig 2C), and humor typology (= a table).
"""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
V3 = HERE.parents[1] / "results" / "analysis_plots"
SUPP = HERE.parent / "output" / "supplement"
SUPP.mkdir(parents=True, exist_ok=True)

# ordered: (v3 source, figS# target) — figures contiguous S1–S8
COPY = [
    ("b_laughter/main/glm_voxelwise_slices.png",              "figS1_voxelwise_glm.png"),
    ("b_laughter/supplement/fig_laughter_isc.png",            "figS2_laughter_isc.png"),
    # figS3 (ISC robustness 2x2) gets an ISC title banner — see below.
    # classifier validation is text-only — no figure (see revision_supplement.md).
    ("b_laughter/supplement/fig_striatum.png",                "figS4_striatum.png"),
    ("b_laughter/main/fig_dose_response.png",                 "figS5_dose_response.png"),
    ("c_controls/fig_head_motion.png",                        "figS6_head_motion.png"),
    # figS7 (AV-energy) is a composite — see below (GLM control leads, ISC matching follows).
    ("c_controls/fig_glm_by_humor_type.png",                  "figS8_glm_by_humor_type.png"),
]
for src, dst in COPY:
    shutil.copy(V3 / src, SUPP / dst)
    print(f"staged  supplement/{dst}")


def _titlebar(src, dst, text, band=70, fs=34, bg=(255, 255, 255), fg=(0, 0, 0)):
    """Prepend a bold centered title band to a staged figure.
    Used for figS3, whose original generator is no longer in the tree, to make the
    ISC-vs-GLM identity explicit on the panel (per the standing ISC/GLM convention)."""
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font = ImageFont.truetype(
        str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"), fs)
    im = Image.open(src).convert("RGB")
    out = Image.new("RGB", (im.width, im.height + band), bg)
    out.paste(im, (0, band))
    d = ImageDraw.Draw(out)
    box = d.textbbox((0, 0), text, font=font)
    d.text(((im.width - (box[2] - box[0])) // 2, (band - (box[3] - box[1])) // 2 - box[1]),
           text, fill=fg, font=font)
    out.save(dst)


def _vstack(top, bottom, dst, pad=28, bg=(255, 255, 255)):
    """Vertically stack two panels (equal width, white gutter) → one figure."""
    from PIL import Image
    a = Image.open(top).convert("RGB"); b = Image.open(bottom).convert("RGB")
    w = max(a.width, b.width)
    fit = lambda im: im if im.width == w else im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
    a, b = fit(a), fit(b)
    out = Image.new("RGB", (w, a.height + pad + b.height), bg)
    out.paste(a, (0, 0)); out.paste(b, (0, a.height + pad))
    out.save(dst)


# figS3 — ISC-contrast robustness: add an explicit "ISC" title banner.
# CAVEAT: this panel's generator is NOT in the tree — it compares the 2x2 of pipeline
# (fMRIPrep/H5) x classifier (original/Clf-C), and the H5 laughter-contrast maps and the
# original-classifier annotations were not carried over, so it cannot be regenerated here.
# It is staged from a pre-existing panel if one is present. Missing it must NOT abort the
# rest of the supplement build (figS7 below is fully reproducible).
_s3_src = V3 / "b_laughter" / "supplement" / "fig_wholebrain_2x2_comparison.png"
if _s3_src.exists():
    _titlebar(_s3_src, SUPP / "figS3_isc_robustness.png",
              "Laughter − non-laughter ISC contrast (t, FDR < .05) — pipeline × classifier")
    print("staged  supplement/figS3_isc_robustness.png (ISC title banner added)")
else:
    print(f"SKIP    supplement/figS3_isc_robustness.png — source panel absent ({_s3_src.name}); "
          "not regenerable from this package (H5 maps + original-classifier annotations not carried)")

# figS7 — AV-energy control: GLM (primary, top) + ISC matching (bottom)
_vstack(V3 / "c_controls" / "fig_glm_av_energy.png",
        V3 / "c_controls" / "fig_av_energy.png",
        SUPP / "figS7_av_energy.png")
print("staged  supplement/figS7_av_energy.png (composite: GLM control + ISC matching)")
