# Classical ridge/graph laboratory — T02 candidate 1 (the reproducible floor)

The T02 contract (`coordination/requests/claude/2026-07-21-t02-start.md`) requires a
**deterministic classical ridge/graph candidate as the floor** before any learned
candidate. This lab is that floor. It has no weights: same image + same parameters →
same instances, on CPU, with no framework that could touch
`Conversion_Efficiency/cpenv`.

## Pipeline

| Stage | What | Cost | Depends on parameters? |
|---|---|---|---|
| A | Canonical `precision_myotube.segmentation.semantic_territory` (tophat → CLAHE → Sato tubeness → hysteresis → intensity gate, threshold-plateau selection) | ~3 min / 3636² field | no — cached per well |
| B | Skeletonise → skan branch graph → pair anti-parallel branch ends at junctions → union-find into whole fibres | seconds | `spur_um`, `straight_dot` |
| C | Assign each territory pixel to its nearest **retained fibre** pixel, then gate by traced length and area | seconds | `min_length_um`, `min_area_px` |

## Relationship to the canonical code

Stage A is imported unchanged. Stage B is a **parameterised re-implementation** of
`precision_myotube.fiber_gate.trace_fibers`, which hard-codes `SPUR_UM = 10.0` and
`STRAIGHT_DOT = -0.5` as module constants. The canonical file is Codex-owned and is
deliberately left untouched. `test_defaults_reproduce_canonical_tracer_grouping`
pins the equivalence: at the canonical constants both tracers produce an identical
fibre-length multiset, so the candidate cannot silently drift away from the
validated recipe.

Exposing those two constants is the whole point of the candidate. Measured on
`23_B02_ctrl`, the tracer at canonical settings puts **~4 predicted fragments on
every reviewed myotube** and recovers only **0.59×** its area — the classical floor
independently reproduces the dominant human-correction error class (`too_short`
under-tracing, 35 of the 40 real correction pairs). Junction pairing is therefore
where this floor actually loses, and it deserves to be fitted rather than assumed.

## One deliberate fix vs. the canonical helper

`length_gated_territory` computes nearest-skeleton indices over the **whole**
skeleton, so any territory pixel whose closest skeleton pixel lies on a pruned spur
is orphaned — measured at **32% of territory area** on `23_B02_ctrl`. Stage C
measures distance to retained fibre pixels only. Effect on that well:

| | assigned territory | median best-overlap IoU | GT matched at IoU ≥ 0.5 |
|---|---|---|---|
| nearest-any-skeleton | 0.679 | 0.490 | 16 / 35 |
| nearest-retained-fibre | **1.000** | **0.588** | **22 / 35** |

## Fold protocol

`run_folds.py` runs six whole-well leave-one-well-out folds. For each fold the
parameter grid is scored on the **five training wells only**, the best point by
`precision_weighted_score` ((2P + R)/3, the canonical selection metric) is chosen,
and that point is applied unchanged to the held-out well. Ties break to the lowest
parameter index — a fixed documented order, never toward the held-out result. Two
assertions guard the invariant (`held-out ∉ training set`; exported `image_id`
matches the held-out field), and predictions are re-checked as `reviewed=False`
after export.

Because stage A is cached and stages B/C are seconds, the grid is scored once per
(well, parameter) and every fold reuses that table — no fold ever recomputes, and
no fold can see its own well's score during selection.

## Evaluation ground truth

Scoring uses each well's reviewed `complete` masks **minus the two binding
`training_exclude.json` exclusions** (`19_B06/myotube_0377`, `22_B03/myotube_0321`).
This matters: the source `*.qc.instances.json` still carries those two as
reviewed/complete, so a naive scorer would evaluate against masks the plan forbids.
The filtered sets are written to `eval_gt/` and hashed — 375 masks total, matching
`bootstrap_manifest.json` — so T03 can reproduce exactly what was scored.

## Results (v1 run, six folds)

Pooled held-out means: precision 0.067, recall 0.915, F1 0.124, matched IoU 0.917,
length MdAPE 0.000, false-split rate 0.147, over-merge rate 0.000 (375 GT, 5,279
predictions, 348 matched). Every fold selected the same parameters — `spur_um=10.0,
straight_dot=0.0, min_length_um=25.0` — which is a reassuring sign that the fit is
stable across wells rather than chasing one well's noise.

**Do not report those pooled numbers as segmentation performance.** A length MdAPE
of exactly 0.000 is the tell. `circularity_audit.py` explains it:

| subset | n | recall | median IoU | |
|---|---|---|---|---|
| unedited GT | 335 | 0.979 | 1.000 | **circular** |
| edited GT (real correction pairs) | 40 | **0.500** | **0.648** | **meaningful** |

Only 40 of the 375 reviewed masks were ever edited, so **89.3% of the ground truth
is a verbatim accepted proposal**, and this candidate re-derives its instances from
the same deterministic recipe that produced those proposals — **47.4% of matched
pairs are pixel-identical (IoU = 1.000)**. On the unedited majority the pipeline is
scoring against its own output.

**The honest floor is recall 0.500 at median IoU 0.648**, measured on the 40 pairs
where ground truth and proposal genuinely differ. That is the number a learned
candidate has to beat.

This matters for T03: a learned candidate does **not** share the proposal generator
and so gets none of this structural advantage. Comparing it to this floor on the
pooled metric is biased toward the floor. Compare on the edited subset, or report
both splits.

Using the 40 correction pairs this way is evaluation only — they were not used for
training or tuning (`run_manifest.json` records `correction_pairs_used: false`).

## Reading the numbers honestly

**Precision here is dominated by a sparse-GT effect, not by hallucinated objects.**
The reviewed `complete` set is a small subset of the fibre-like structure in each
field (across the six wells the operator marked 377 complete against 839 ambiguous
and 553 rejected). A predicted instance that matches no GT mask is usually a real
structure the operator did not certify as a fully-measurable myotube — not a
detection error in the ordinary sense. **Recall, matched IoU, and length MdAPE are
the meaningful floor metrics**; precision is reported because the contract requires
it, not because it means what it usually means.

All results are exploratory, single-operator, proposal-conditioned, retrospective
development evidence. They are not consensus, not inter-rater agreement, and not
prospective validation. Predicted instance counts are never authoritative
independent-myotube counts.

The classical floor emits **mutually exclusive** masks, so it structurally cannot
represent a crossing as two overlapping instances. That is a real limitation of the
floor and a thing a learned candidate should beat, not a defect to paper over.

## Run

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python model_labs/classical/run_folds.py --out model_labs/classical/_runs/v1
python -m pytest model_labs/tests/test_classical_ridge_graph.py -q --basetemp tmp/pytest_classical
```

Outputs under `--out`:

- `predictions/<model>/<version>-fold-<well>/<image_id>.instances.json` — sealed,
  unreviewed canonical `InstanceSet` handed to Codex for T03;
- `eval_gt/<well>.eval_gt.instances.json` — the hashed scoring set;
- `run_manifest.json` — command, input-manifest hash, environment record and hash,
  per-fold selected parameters, held-out metrics, timing, and failures;
- `grid_scores.json` — the full (well × parameter) score table behind every fold's
  selection, so the fit is auditable.

`_territory_cache/` and `_work/` are regenerable scratch.
