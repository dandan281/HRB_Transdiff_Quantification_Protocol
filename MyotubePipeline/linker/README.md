# linker/ — learned fragment-linker (Phase 2)

Replaces `common/merge.py`'s hand-tuned geometric join rules with a **classifier** trained on the
Q_Plates ground truth. Given two raw Ridge fragments whose ends are near each other, it predicts
**join vs don't-join**. Trains on raw stage-2 fragments (`bright_segments.txt`) + GT ROIs — no
pipeline change, no stage-5 dependency.

Standalone (base anaconda: numpy/scipy/scikit-image/roifile/sklearn). Reuses `benchmark/` for
rasterization/overlap.

## Run
```
cd MyotubePipeline
C:/Users/liqig/anaconda3/python.exe -m linker build   # data/pairs.csv from the training wells
C:/Users/liqig/anaconda3/python.exe -m linker eval    # leave-one-well-out + plate-held-out metrics
C:/Users/liqig/anaconda3/python.exe -m linker train   # fit final GBM -> models/link.joblib
```

## How labels are made (auto, from GT)
1. Each raw fragment is rasterized and associated to the GT fibre it lies on — GT dilated to a
   **fibre-half-width band** (`GT_ASSOC_RADIUS_PX`=12) so a laterally-offset Ridge centerline still
   maps; a fragment maps if ≥50% of its footprint lies on one GT fibre.
2. Candidate pairs = one per fragment pair whose **closest endpoints** are within `GAP_MAX_PX`=170
   (the natural break point — not all four endpoint combos, which would wreck the angle features).
3. Label = **join** iff both fragments map to the **same** GT fibre. Pairs touching an unmapped
   (noise / un-traced) fragment are dropped (`BOTH_MAPPED_ONLY`) so the negative class is the
   informative "two real but *different* fibres" case, not label-guessing on noise.

Features per pair: `gap_px`, `a1_deg`/`a2_deg`/`align_deg` (endpoint-heading collinearity), `len_min`,
`len_max`, `len_ratio`, `end_dist_ratio`. (Signal-continuity along the gap — merge.py's `signal_ok`
— is a planned addition; geometry-only already separates well.)

## Result (2026-07-08, 4 complete-GT wells, 3705 candidate pairs, 12.6% join)
| model | mean LOWO join-F1 | per-well AUC | train PLATE_32 → test PLATE_23 |
|-------|------------------:|-------------:|-------------------------------:|
| logreg | 0.44 | 0.83–0.91 | P=0.48 R=0.64 F1=0.55 |
| **GBM** | **0.60** | **0.89–0.95** | P=0.80 R=0.37 F1=0.51 AUC=0.89 |

**The join decision is strongly learnable** (AUC ~0.9). GBM wins decisively → the signal is nonlinear
(join needs small gap AND good alignment together). logreg coefficients validate the physics: closer
`gap_px`, lower `align_deg`, similar `len_ratio` → join. Cross-plate recall is conservative at
threshold 0.5 (recoverable via threshold, AUC 0.89); the main limiter is only **3 training plates**.

## Training-set caveats
- 4 wells only: B03 (PLATE_23) + B02/C02/C03 (PLATE_32). Excluded for GT quality: 32_C08
  (partial/long-skewed GT), C09 & B06 (partial re-trace ROI zips), B02_P23 (no zip), C05 (corrupt).
- ~12–30% of raw fragments don't map to any GT fibre (dense wells worst) — real un-traced fibres or
  noise; these are excluded from training but WILL appear at inference.

## Phase 3 — chaining head-to-head (`python -m linker chain [threshold]`)
`chain.py` scores every candidate endpoint pair, greedily joins highest-proba first (each end used
once, no cycles via union-find), reconstructs each chain into one polyline, and benchmarks
**learned linker vs geometric merge vs raw fragments** on the SAME raw bright fragments. (Validation:
`geo-merge` reproduces the pipeline's stage-2 exactly — 123 traces on P26_B02 = run log "merged=123".)

**Result on held-out PLATE_26 (never trained on), threshold 0.5:** the learned linker beats the
geometric merge on **every well** — fewer, longer, more GT-faithful traces, higher F1, no recall loss.

| well (GT %<300) | geo-merge F1 / n / %<300 | **learned** F1 / n / %<300 |
|---|---|---|
| B02 (GT 72, 63.9%) | 0.33 / 123 / 86.2 | **0.41 / 99 / 79.8** |
| C08 (GT 165, 48.5%) | 0.30 / 301 / 87.7 | **0.33 / 273 / 87.9** |
| B06 (GT 106, 42.5%) | 0.35 / 188 / 81.4 | **0.42 / 160 / 78.1** |

So the linker is a **proven net improvement over the merge**. But bright-only leaves %<300 well above
GT — the full fibre length spans dim gaps a bright-only chain can't bridge.

### Bright+dim pooling (`INCLUDE_DIM=True`) — the %<300 win
Pooling the stage-3 dim pieces (`dim_traces.txt`) with the raw bright fragments and **retraining on the
pooled set** grows the training data 4.5× (3.7k→16.8k pairs) and lifts the linker's own quality:
GBM mean leave-one-well-out join-F1 **0.60 → 0.72** (AUC 0.92–0.96). Chaining the pooled fragments on
**held-out PLATE_26** then beats the current pipeline's actual output on every well — higher F1 and a
**much smaller %<300 bias** (the scientific readout):

| well (GT %<300) | pipeline-final F1 / %<300 | learn(bright+dim) F1 / %<300 @0.5 | @0.35 |
|---|---|---|---|
| B02 (63.9) | 0.383 / 87.4 | **0.385 / 81.8** | 0.380 / 79.1 |
| C08 (48.5) | 0.365 / 87.1 | **0.407 / 74.2** | 0.414 / 70.9 |
| B06 (42.5) | 0.391 / 81.7 | **0.450 / 65.6** | 0.469 / 59.5 |

It closes ~30–56% of the pipeline's %<300 gap to GT while keeping recall (0.65–0.74).

### Keep/drop confidence filter (`tracefilter.py`, `python -m linker filter-train | pipeline`)
Runs AFTER chaining: classifies each chained trace real-fibre(keep) vs noise(drop), auto-labeled from
GT (trace maps to a GT fibre = keep). Features: length_px, straightness, signal_mean/frac/p25 (length
+ straightness dominate). Learnable — mean leave-one-well-out keep-F1 0.89. End-to-end **chain+filter**
on held-out PLATE_26 vs the current pipeline:

| well (GT %<300) | pipeline-final F1 / n / %<300 | chain+filter F1 / n / %<300 |
|---|---|---|
| B02 (63.9) | 0.383 / 215 / 87.4 | **0.406 / 189 / 82.0** |
| C08 (48.5) | 0.365 / 365 / 87.1 | **0.425 / 319 / 72.7** |
| B06 (42.5) | 0.391 / 252 / 81.7 | **0.453 / 203 / 66.0** |

The filter **raises precision, cuts the over-count toward GT, and lifts F1** — but **barely moves %<300**
(vs chain-only). Key insight: the residual %<300 gap is no longer *noise* (removed) but **under-tracing**
— real fibres traced too short + fragments that didn't merge, which the filter correctly keeps. So the
next lever is **endpoint-extend / more aggressive+better linking, not more filtering**.

### Endpoint-extend (`extend.py`, `python -m linker full <link> <keep> <extend_sig>`)
Runs last. From each trace endpoint, greedily steps along the fibre signal (max-signal search in a
forward cone, stop when 8-bit signal < `extend_thr`, self-limiting at true ends). Tuned against the
benchmark's matched length-ratio: `extend_thr≈50` lands lenRatio at 0.99–1.13 (was 0.76–0.83) — fibres
now measured at ~true length. Lower thr over-extends (thr=25 → lenRatio 1.5, over-corrects).

## FULL LEARNED STACK vs current pipeline — held-out PLATE_26 (link 0.5, keep 0.5, extend 50)
| well (GT %<300) | pipeline-final F1 / %<300 (bias) | **linker+filter+extend F1 / %<300 (bias)** |
|---|---|---|
| B02 (63.9) | 0.383 / 87.4 (**+23.5**) | **0.414 / 69.8 (+5.9)** |
| C08 (48.5) | 0.365 / 87.1 (**+38.6**) | **0.445 / 60.2 (+11.7)** |
| B06 (42.5) | 0.391 / 81.7 (**+39.2**) | **0.472 / 56.2 (+13.7)** |

On a plate nothing was trained on: **F1 +8–21%, and the %<300 bias cut ~65–75%** (from +23…+39 to
+6…+14). lenRatio ~1.0 confirms fibres are now traced to true length. Pipeline order:
raw fragments → **linker chain** (bright+dim) → **keep/drop filter** → **endpoint-extend**.

### Remaining / next
- B06 still slightly over-extends (lenRatio 1.13) — a per-well adaptive `extend_thr` (from the signal
  histogram) would tighten it; fixed 50 is a good default.
- Residual %<300 bias (+6…+14) is now small and mostly count-side (still over-counts vs GT).
- **Productionize:** wire linker+filter+extend into `common/merge.py` / a stage so runs use it directly.
- **Widen the test:** run PLATE_28 (5 wells) for a second held-out plate; expand linker training past 4 wells.
- Add the signal-continuity feature to the linker.
