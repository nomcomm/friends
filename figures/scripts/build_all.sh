#!/usr/bin/env bash
# Rebuild every figure from scratch, in order. Run from anywhere:
#     bash CODE/figures/scripts/build_all.sh
set -e
cd "$(dirname "$0")/.."                      # → figures/

echo "cleaning panels/ and output/ ..."
rm -f panels/*.png output/*.png output/*.pdf output/supplement/*.png

python3 scripts/panel_fig2_glm_brain.py         # Fig 2 panel B (whole-brain GLM contrast)
python3 scripts/panel_fig2_per_viewer.py        # Fig 2 panel D (per-viewer GLM maps)
python3 scripts/panel_fig1_overall_isc.py       # Fig 1 panel B (sequential-cmap ISC montage)
python3 scripts/panel_fig1_stability.py         # Fig 1 panel C (scatter + histogram)
python3 scripts/panel_fig1_season_stability.py  # Fig 1 panel D
python3 scripts/panel_fig2_season_glm.py        # Fig 2 panel C (per-episode β, 5 ROIs)
python3 scripts/panel_fig3_raster_all.py        # Fig 3 panel A (corpus laugh-track raster)
python3 scripts/panel_fig3_laughs_per_min.py    # Fig 3 panel B (laughter rate per episode)
python3 scripts/panel_fig3_harvesting.py        # Fig 3 panel C (humor vs laughter)
python3 scripts/panel_fig3_humor_composition.py # Fig 3 panel D (humor composition)
python3 scripts/figure1.py                      # compose output/figure1.png
python3 scripts/figure2.py                      # compose output/figure2.png
python3 scripts/figure3.py                      # compose output/figure3.png
python3 scripts/stage_supplement.py             # supplement figures → output/supplement/
python3 scripts/supp_isc_first_vs_last.py       # the two episode ISC maps behind Fig 1C1

echo ""
echo "done — panels/ and output/ rebuilt:"
ls -1 panels/ output/*.png output/supplement/
