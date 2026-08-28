# T04 tracer lane — T03 bootstrap run, 2026-08-27

The trace→mask bridge is built and the sealed PLATE_23 numbers exist. All
local, $0 cluster spend. Numbers go to Codex; nothing here is a ruling.

## 1. Predeclarations (fixed before any PLATE_23 inference)

- **Checkpoint**: `_runs/net_cv/B02/best.pt` — trained on the nine non-B02
  wells (the lane's historical single-split training set, retains all five
  threshold-tune wells). No ensembling: ensemble-mean fields would shift the
  centre statistics the frozen thresholds were calibrated on, and that shift
  cannot be validated on any PLATE_32 well without leakage.
- **Walk + preps**: the frozen CV configuration verbatim (`cv_report.py`) —
  walk seed 0.4 / support 0.3 / claim 3.5 px / rescue 1; nms prep crossing
  0.4 valid 0.2 (shipped arm); raw prep (coverage arm) reported alongside.
- **Bridge**: each traced object's paths stamped at the corpus ribbon
  convention (`width_px 8.0`, band = EDT(1 px spine) ≤ 4.0), OR-ed per
  object, exported overlap-safe through `export_prediction` →
  `benchmark_instances` — the same path as every other candidate.
- **Gates**: 0 / 25 / 50 µm computed in one pass, all reported; primary
  declared as nms / 50 µm (the lane's frozen counting convention).
- **Metric order**: `length_mdape` vs 0.3169, `false_split_count` vs 52/375,
  pooled `recall` vs 0.928.

## 2. Leakage check

The dense-corpus wells B02 and C05 carry nd2 names that collide with
bootstrap wells (`23_B02_ctrl`, `29_C05`). Pixel correlation between the
corpus and bootstrap images: 0.0006 and 0.0061 — **different fields, no
overlap with training**. The independent-test premise stands.

## 3. The numbers (6 wells, 375 certified GT instances)

Script: `model_labs/tracer_lab/eval_tracer_on_bootstrap.py`; full per-well
table in `_runs/eval_bootstrap_v1/eval_summary.json`.

| config | n_pred | tp | recall | fsplit | omerge | med mdape |
|---|---|---|---|---|---|---|
| **nms ≥50 µm (PRIMARY)** | 2019 | 209 | **0.557** | **6** | 4 | **0.0864** |
| nms ≥25 µm | 2345 | 220 | 0.587 | 6 | 4 | 0.0903 |
| nms all | 2899 | 220 | 0.587 | 11 | 4 | 0.0903 |
| raw ≥50 µm | 1142 | 209 | 0.557 | 1 | 4 | 0.0451 |
| raw ≥25 µm | 1183 | 216 | 0.576 | 1 | 4 | 0.0465 |
| raw all | 1199 | 216 | 0.576 | 1 | 4 | 0.0465 |

Against the classical floor, primary row:

- `length_mdape` **0.0864** vs 0.3169 — **3.7× better**, and at the human
  self-consistency ceiling (0.096, one window).
- `false_split_count` **6** vs 52 — **8.7× better**.
- pooled `recall` **0.557** vs 0.928 — **worse**.

## 4. Attribution of the recall gap — width hypothesis REFUTED

A match requires IoU ≥ 0.5, so a fixed-width ribbon could cap recall on wide
fibres. Measured (read-only) on the sealed GT: median certified width is
**8.2 px** (p25/p75 = 7.2/10.0), and the fraction of GT matchable at
IoU 0.5 under a 9 px painted band is **0.952**
(`_runs/eval_bootstrap_v1/width_cap_diagnostic.json`). The 8 px ribbon is
nearly ideal for this corpus; **the recall shortfall is real partial
coverage** — the missing-middles failure (§11 of the 2026-08-23 report), now
visible cross-plate: a fibre traced over ~half its length fails IoU 0.5 even
at perfect width.

## 5. Caveats that must travel with the numbers to Codex

1. **The GT is proposal-conditioned** (evidence class
   `single_operator_proposal_conditioned`): the certified instances
   originated as classical-pipeline proposals the operator triaged. The
   recall floor 0.928 was set by the candidate whose own proposals defined
   the GT; an independent candidate cannot inherit that advantage. The
   length and split floors are less exposed (computed on matched pairs).
2. **`length_mdape` is matched-only**: 0.0864 says "what it matched, it
   measured at human repeatability" — it does not speak for the 44% of
   certified fibres it failed to match.
3. The raw arm's mask-level numbers (mdape 0.0451, fsplit 1) look better
   than its PLATE_32 polyline metrics (mdape ~22) because IoU 0.5 matching
   silently discards the halo-roaming duplicate walks that polyline scoring
   charged it for. Same candidate, different filter — do not read the raw
   arm as fixed. n_pred raw ≥50 µm is 1142 against 375 certified (sparse GT;
   precision uninterpretable, as predeclared).
4. One checkpoint (fold B02), one run, no tuning on PLATE_23; the smoke well
   (23_B02_ctrl) was run first alone to validate mechanics — same frozen
   config, its numbers are unchanged in the full run.

## 6. Where this leaves the lane

The two failure modes the sealed benchmark was designed to catch — length
error and false splits — are beaten by large margins; coverage is the open
front, and it is the same missing-middles gap the lane already diagnosed.
The history-conditioned stepping head (the one untried mechanism) remains
gated on the user asking for it.

## 7. Artifacts

| path | what |
|---|---|
| `model_labs/tracer_lab/eval_tracer_on_bootstrap.py` | the bridge + sealed run (committed) |
| `_runs/eval_bootstrap_v1/eval_summary.json` | per-well + pooled numbers (local, untracked) |
| `_runs/eval_bootstrap_v1/width_cap_diagnostic.json` | GT width distribution + ribbon ceiling |
| `_runs/eval_bootstrap_v1/predictions/` | exported InstanceSets per config |
