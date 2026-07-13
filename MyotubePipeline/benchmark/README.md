# benchmark/ — Phase-0 accuracy harness

Scores pipeline predictions against the **Q_Plates** hand-traced ground truth. Model-agnostic: it
judges the current Ridge pipeline today and will judge the fragment-linker later, on the same ruler.

Standalone (base anaconda: numpy / scipy / scikit-image / roifile). No napari, no torch.

## Run
```
cd MyotubePipeline
C:/Users/liqig/anaconda3/python.exe -m benchmark --status        # what has GT / predictions
C:/Users/liqig/anaconda3/python.exe -m benchmark --all           # score every scoreable well
C:/Users/liqig/anaconda3/python.exe -m benchmark --all --sweep   # + F1 vs dilate radius
C:/Users/liqig/anaconda3/python.exe -m benchmark --well P23_C08_BR223_IGF1R
```
Outputs → `benchmark/out/`: per-well `*.json`, `detection_summary.csv`, `endpoint_summary.csv`,
`endpoint_per_well.csv`.

## What it measures
**Tier 1 — detection (geometry).** Each GT ROI and predicted trace is rasterized as a dilated
centerline (radius `DILATE_RADIUS_PX`=5); a bbox-prefiltered overlap graph gives greedy 1:1 matches →
precision / recall / F1, and the error classes:
- `too_short` — matched pair with pred length < 0.8× GT length (memory error #1).
- `false_split` (strict) / `fragmented` (≥2 preds each covering ≥20% of one GT) — one fibre broken up.
- `over_merge` — one prediction spanning ≥2 GT fibres (memory errors #2/#3).
- `boundary_flip_rate`, `boundary_weighted_mae_um` — length errors weighted near the 300 µm line.

**Tier 2 — scientific endpoint.** Per-well fibre count and **% below/above 300 µm**, GT vs predicted,
straight from the two Results CSVs — the published readout. This tier is authoritative for length.

## Baseline of the current Ridge pipeline (2026-07-06, 4 scoreable wells, r=5px)
| well | GT/pred | P | R | F1 | too_short | frag | over_merge | med_len_ratio | %<300 GT→pred |
|------|--------:|--:|--:|---:|----------:|-----:|-----------:|--------------:|--------------:|
| P32_B02_Ctrl (dev)        | 356/426 | 0.57 | 0.68 | 0.62 | 62% | 15% | 0 | 0.70 | 81→95 (+14) |
| P32_C03_ACT104_EGFR (dev) | 446/518 | 0.45 | 0.53 | 0.49 | 65% | 10% | 4 | 0.72 | 74→94 (+20) |
| P32_C02_ACT104_FGFR (dev) | 589/671 | 0.45 | 0.51 | 0.48 | 68% | 8%  | 8 | 0.65 | 79→95 (+15) |
| P23_C08_BR223_IGF1R (test)| 133/371 | 0.22 | 0.60 | 0.32 | 66% | 14% | 0 | 0.72 | 23→87 (+64) |

**Headline:** predicted fibres are ~half the GT median length and 94–95% fall below 300 µm vs 74–81%
for GT → the scientific readout is biased high by **+14 to +20 points**. Dominant errors: systematic
**under-tracing** (matched fibres ~70% length) plus a population of short excess predictions that sit
**on/beside real fibres** (only ~3–9% are free-floating noise) → the fragment-**linker** is the right lever.
`over_merge` ≈ 0, i.e. the signal-gated merge is conservative, not aggressive.

## PLATE_26 — held-out plate (2026-07-08, `--auto --no-learn` runs, r=5px)
Ran the current pipeline on all 3 PLATE_26 wells (all complete GT; none used in linker training).

| well | GT→pred | over-count | F1 | P | R | med_len_ratio | %<300 GT→pred (Δ) |
|------|--------:|:----------:|---:|--:|--:|--------------:|------------------:|
| P26_B02_Ctrl        |  72→215 | 3.0× | 0.38 | 0.26 | 0.76 | 0.83 | 63.9→87.4 (**+23.6**) |
| P26_C08_BR223_IGF1R | 166→365 | 2.2× | 0.37 | 0.27 | 0.58 | 0.76 | 48.5→87.1 (**+38.6**) |
| P26_B06_ACT104_TrkA | 106→252 | 2.4× | 0.39 | 0.28 | 0.66 | 0.83 | 42.5→81.7 (**+39.3**) |

**The held-out plate is markedly WORSE than the PLATE_32 dev wells** (%<300 bias +24…+39 vs +14…+20)
→ the pipeline generalizes poorly; external validity is low. **Root cause = stage-3 dim-recovery
over-generation.** Per well, bright→dim→final:
- B02: bright 34 (68%<300) → +dim 182 (91%<300, med 95µm) → final 215 (87%). GT 72 (64%).
- C08: bright 45 (71%) → +dim 329 (91%, med 100µm) → final 365 (87%). GT 165 (48%).
- B06: bright 38 (76%) → +dim 216 (83%, med 118µm) → final 252 (82%). GT 106 (42%).

The dim pass adds 180–330 short traces (median ~100µm, ~90% below 300) — 2–4× the entire GT count —
flooding the result. It adds a roughly *fixed* dim burden, so it is proportionally devastating on
these **sparse, long-fibre wells** (GT only 42–64% below 300) and was masked on the dense PLATE_32
dev wells. Bright-only under-counts (34–45 vs 72–166 GT); neither extreme is right. So for the %<300
readout the **dim-recovery calibration is a lever at least as large as the linker** — recover real
dim fibres without admitting short noise. Under-tracing (len_ratio 0.76–0.83) and fragmentation
(10–15%) are still present but secondary here.

## Caveats baked into interpretation
- **32_C08 GT is suspect.** Q_Plates `C08 _N` has only 133 ROIs / 22.8% below 300 µm (vs the
  BR223_IGF1R group mean 60.6%); it looks like a partial trace of the longer fibres. Its P=0.22 is
  dominated by GT mismatch, not pipeline quality. Trust the 3 PLATE_32 wells (full GT).
- **GT ROI geometry vs GT Results.csv differ ~10–17%** in measured length (vertex density / `_N`
  re-trace vintage). Tier-1 `len_ratio` (uses ROI geometry) is therefore ~12% pessimistic; **Tier-2
  uses Fiji Results.csv and is the authoritative length metric.** Verified: predicted length = the
  pipeline's own Fiji length to 0.000 µm.
- **Absolute F1 rises with `DILATE_RADIUS_PX`** (r=3/5/8 → ~0.29/0.49/0.60 on the PLATE_32 wells).
  Hold the radius fixed when comparing models; the relative delta is what counts.
- Only **4 of 22** usable wells have predictions. PLATE_26/28 (8 wells, ~1365 GT fibres) have **no**
  pipeline runs — running them is the way to get a real held-out test set. `--status` tracks this;
  wells auto-become scoreable once their `stage4_qc/final_traces.txt` + `stage5_measure/final_results.csv` exist.
- Excluded: `PLATE_23/C05` (corrupt ROI zip → single .roi), `PLATE_23/B02` (no ROI zip). Both retain
  a Results.csv so they can still contribute to Tier-2 if wired in later.
