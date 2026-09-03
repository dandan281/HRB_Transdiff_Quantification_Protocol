# Note to the Conversion_Efficiency lane — a plate-level CE command now exists

**From:** the T04 tracer session, 2026-09-03, at the operator's request
("do we have the same thing for the conversion efficiency pipeline? if not
can you build one?"). **Nothing in your scripts was modified.**

## What was built

`Conversion_Efficiency/quantify_ce_plate.py` — one command, one env:

    conda run -n fijiconv python Conversion_Efficiency/quantify_ce_plate.py \
        --plate "<folder of .nd2>" [--layout well,treatment.csv] [--wells ...]

Outputs under `model_labs/tracer_lab/_runs/ce_plates/<PLATE>/`: `wells.csv`
(nuclei, desmin+ %, CE25 %, CE50 %, territory coverage, per well),
`summary.json` (every chosen cut with its plateau range and the full sweep
tables), `sweeps.png` (the three knob curves), `channels.png`, per-well
overlays, `per_nucleus.npz`.

## What it wraps — your method, generalised, not a new readout

- **Nuclei**: the Plate-44 Fiji recipe from `Plate44_Tdiffs/extract_per_nucleus.py`
  (Gaussian → rolling-ball → absolute DAPI cut → fill → open → watershed →
  area gate), run in-process via scyjava exactly as you do.
- **Desmin per nucleus**: mean rolling-ball-subtracted desmin in the
  perinuclear shell (your 6 px @1.72 µm = 10.3 µm).
- **CE**: nucleus ≥25 % / ≥50 % inside the FILLED absolute-threshold
  territory at k·σ_plate — the form `Plate32_Fusion_GT/eval_v2_candidate.py`
  validated against the ground truth.
- **All cuts by the plateau rule, per plate, from pooled data**: T_DAPI
  (count vs T), the desmin cut (fraction-positive vs cut), and k (CE25 vs k).
- Treatments from `--layout` or the file names; η² + label-shuffle
  permutation p before any ranking.

Two generalisations you may want to adopt:

1. **Recipe constants are in micrometres**, converted per plate from the
   nd2 pixel size. Your pixel constants (8–700 px nuclei, 6 px shell,
   25/150 px balls) reproduce exactly at 1.7246 µm/px; applied unchanged to
   a 0.65 µm/px Q-plate they would gate away real nuclei (a 24 µm² nucleus
   is 57 px there).
2. **The fraction-positive curve is flat by construction at both ends**
   (f→1 as cut→0). Unrestricted, the plateau rule picked cut 22 → 95 %
   positive on Plate 44 — your "degenerate on a continuum" finding,
   reproduced. The search is confined to cuts calling 5–80 % positive
   (exclusion of the trivial flats, not a calibration to any expectation).
   With that, Plate 44 (B02 B03 C02 C03) gives cut 121 → 15.0 % desmin+
   (controls 13.9–15.4 %) against your band analysis' 80 → 18.9 %
   (controls 16.1 %): the same plateau read two ways.

## Verification

- PLATE_26 (3 wells): channels inferred DAPI=ch2 / desmin=ch1 (the
  Q-plate map); T* 1293, cut* 197 → 21.3 % desmin+; B06/ctrl fold 1.58 vs
  1.53 in `New_Quantif_P26/percell_desmin.json` (different nuclei method).
- Plate 44 subset: channels DAPI=ch0 / desmin=ch1; T* 834 (yours: 800
  bands / 600 full-frame); k* 8.
- `model_labs/tests/test_quantify_ce_plate.py`: 8 contract tests
  (µm→px scaling reproduces your pixel constants; plateau rule on known
  curves incl. the trivial-flat case; robust σ; parsing).

## What it does NOT do (yours to decide)

It does not touch the tile-forest artefact screen, the ROI-band machinery,
or the B11 failed-stain gauge; it reports whole-well values. If you want the
cleaned (vote-masked) CE as a column, the per-nucleus table it writes has
x, y, area, dapi, peri and the overlap fraction at k*, so `Tile_Forest` can
consume it without re-segmentation.
