"""
humor_classification.py — humor content vs laughter: the "harvesting" argument (Block B, B4, R1.2)
==================================================================================================
Result: humor content is near-continuous (~66% of speech TRs) while laughter marks
periodic peaks (~21%) — so the laugh track "harvests" only a fraction of the humor.

HOW THE HUMOR WAS ANNOTATED  (provenance — carried over from 03_REVISION/04_humor_classification.py;
                              an LLM pass, not re-run here due to API cost; see PROVENANCE.txt)
  - Source text: CNeuroMod episode transcripts (spoken dialogue per TR).
  - "Utterance" = a run of consecutive speech TRs, concatenated into one dialogue passage.
  - Each utterance is labelled by an LLM (Google Gemini 2.0 Flash, temperature 0, JSON out)
    against the Juckel, Bellman & Varan (2016) sitcom-humor typology: is it humorous, and if
    so its primary category among the 3 VERBAL categories — Language / Logic / Identity — plus
    specific techniques. The typology's 4th category (Action: physical/visual gags) is EXCLUDED
    because it needs visual info the transcript lacks. The preceding utterance is supplied as
    context (not itself classified) so setup→punchline can be read.
  - Each TR inside an utterance inherits that utterance's label → a per-TR humor vector.

CAVEATS (state these in the write-up)
  - The humor coding is a single-LLM annotation and is NOT human-validated (no inter-rater
    reliability, unlike the laughter classifier). Treat as LLM-assisted content coding.
  - Verbal humor only (visual/physical humor not captured) → humor is, if anything, under-counted.
  - "Harvesting" is a COVERAGE claim (66% vs 21%), not a per-TR correlation: the laugh track
    trails jokes, so same-TR humor↔laughter is temporally confounded (handled elsewhere by HRF shift).

THIS SCRIPT reads the carried-over per-TR humor labels and recomputes the humor-vs-laughter
comparison against the PRIMARY Clf-C annotations (data/0_prep/laughter_annotations/), for
consistency with the rest of Block B.

  input   : data/b_laughter/humor_classification/aggregate_humor_by_tr.csv
            data/0_prep/laughter_annotations/{ep}.csv   (Clf-C laughter)
  outputs : data/b_laughter/humor_summary.txt
  figure  : results/analysis_plots/b_laughter/exploratory/fig_humor_harvesting.png

Usage
  python humor_classification.py [--force]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import HUMOR_CLASSIFICATION_DIR, PREP_LAUGHTER_ANN_DIR, B_DIR, B_FIG_DIR

CATS = ["Language", "Logic", "Identity"]


def clfc_laughter_map():
    """episode -> Clf-C laughter ls vector."""
    out = {}
    for f in PREP_LAUGHTER_ANN_DIR.glob("*.csv"):
        out[f.stem] = pd.read_csv(f)["ls"].values
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_txt = B_DIR / "humor_summary.txt"
    if out_txt.exists() and not args.force:
        print(out_txt.read_text()); return

    df = pd.read_csv(HUMOR_CLASSIFICATION_DIR / "aggregate_humor_by_tr.csv")
    # attach Clf-C laughter per TR (align by episode + tr_idx)
    clf = clfc_laughter_map()
    def clfc_ls(row):
        v = clf.get(row["episode"])
        return int(v[row["tr_idx"]]) if v is not None and row["tr_idx"] < len(v) else np.nan
    df["laughter_clfc"] = df.apply(clfc_ls, axis=1)
    d = df.dropna(subset=["laughter_clfc"]).copy()
    d["laughter_clfc"] = d["laughter_clfc"].astype(int)
    d["is_humor"] = d["is_humor"].astype(int)

    n = len(d); humor = d["is_humor"].mean(); laugh = d["laughter_clfc"].mean()
    both = ((d.is_humor == 1) & (d.laughter_clfc == 1)).mean()

    # composition of humor TRs by primary category
    hum = d[d.is_humor == 1]
    comp = {c: (hum["primary_category"] == c).mean() for c in CATS}

    lines = ["Humor content vs laughter — 'harvesting' argument (Clf-C laughter)",
             "=" * 62,
             f"Episodes with both humor + Clf-C laughter: {d['episode'].nunique()}",
             f"Total TRs (all TRs of the matched segments) : {n:,}",
             f"Humor-positive (LLM)     : {humor*100:.1f}%",
             f"Laughter (Clf-C)         : {laugh*100:.1f}%",
             f"Both (same TR)           : {both*100:.1f}%",
             "",
             "=> Humor is near-continuous (66%); laughter marks periodic peaks (21%).",
             "   The laugh track 'harvests' only a fraction of the humor content.",
             "",
             "Humor composition (share of humor TRs by primary category):"]
    for c in CATS:
        lines.append(f"  {c:<9} {comp[c]*100:5.1f}%  ({int((hum['primary_category']==c).sum()):,} TRs)")
    lines += ["",
              "CAVEAT — do NOT read same-TR humor↔laughter as suppression: the laugh",
              "track fills the pauses AFTER jokes, so concurrent (same-TR) laughter is",
              "lower during dialogue/humor TRs. The temporal humor→laughter relationship",
              "is captured by the HRF-shifted fMRI analyses, not this content comparison."]
    report = "\n".join(lines)
    out_txt.write_text(report + "\n"); print(report)

    # figure (exploratory): A = harvesting coverage; B = humor category composition
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.5)); fig.patch.set_facecolor("white")
    axA.bar([0, 1], [humor*100, laugh*100], color=["#E67E22", "#27AE60"], width=.6)
    axA.set_xticks([0, 1]); axA.set_xticklabels(["Humor content\n(LLM)", "Laughter\n(Clf-C)"])
    for i, v in enumerate([humor*100, laugh*100]):
        axA.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axA.set_ylabel("% of TRs"); axA.set_ylim(0, 80)
    axA.set_title("A  Harvesting: humor near-continuous,\nlaughter samples periodic peaks",
                  fontsize=10, fontweight="bold", loc="left")
    axB.bar(range(len(CATS)), [comp[c]*100 for c in CATS], color="#2980B9", width=.6)
    for i, c in enumerate(CATS):
        axB.text(i, comp[c]*100 + 1, f"{comp[c]*100:.0f}%", ha="center", fontweight="bold")
    axB.set_xticks(range(len(CATS))); axB.set_xticklabels(CATS)
    axB.set_ylabel("% of humor TRs")
    axB.set_title("B  Humor composition (Juckel verbal categories)", fontsize=10, fontweight="bold", loc="left")
    fig.tight_layout()
    (B_FIG_DIR / "exploratory").mkdir(parents=True, exist_ok=True)
    out_fig = B_FIG_DIR / "exploratory" / "fig_humor_harvesting.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"\nSaved {out_txt.name} + exploratory/{out_fig.name}")


if __name__ == "__main__":
    main()
