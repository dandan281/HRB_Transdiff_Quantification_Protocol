# Junction round 2 — labels plateaued, root cause found and fixed

2026-07-28. Continues `claude_junction_active_round2_build_2026-07-28.md`.
Operator labeled all 150 round-2 junctions. Headline: **more labels did not
help, and the reason was a 3-pixel direction window. Fixing it took LOWO AUC
0.693 -> 0.892 and junction accuracy 37% -> 58%, with no new labels.**

## 1. Round-2 labels

150 decided: **84 through-pair, 52 branch point, 14 unsure**. Zero overlap
with round 1 (verified). Combined corpus: **395 junctions, 341 labeled
(non-unsure), 1,023 pair rows, 229 positive.**

Export: `PrecisionMyotube/annotation_work/junctions_round2/junctions_active_r2.junctions.json`.

## 2. The plateau (the honest bad news first)

Retraining on rounds 1+2 combined made the model **worse, not better**:

| training set | LOWO AUC | junction accuracy |
|---|---:|---:|
| round 1 only (245 junctions) | 0.701 | 41.0% |
| round 2 only (150 junctions) | 0.664 | -- |
| **rounds 1+2 (395 junctions)** | **0.693** | **37.2%** |

Round 2 is intrinsically harder (uncertainty sampling worked -- it served the
model's coin-flips, and 38% of them turned out to be branch points vs 29% in
round 1), so the accuracy drop is partly a harder evaluation set. But the AUC
not moving at all with +61% data is the real signal.

**Learning curve** (train on N junctions from 5 wells, evaluate the full
held-out well, 12 seeds, LOWO):

| N junctions | LOWO AUC | sd |
|---:|---:|---:|
| 40 | 0.6766 | 0.0115 |
| 80 | 0.6886 | 0.0088 |
| 120 | 0.6898 | 0.0060 |
| 160 | 0.6894 | 0.0053 |
| 200 | 0.6896 | 0.0036 |
| 262 (max) | 0.6924 | 0.0010 |

**Flat from ~120 junctions onward**, gaining **+0.005 AUC per additional 100
junctions**. At that rate reaching AUC 0.85 would need ~3,000 more labels --
and that is a linear extrapolation of a curve that is visibly asymptoting, so
in practice it would never get there. **More labeling was the wrong lever.**

## 3. Root cause: a 3-pixel direction window

`tangent_cos` was computed from the direction vectors `build_branch_graph`
attaches to each branch end, which use `TracerParams.direction_step = 3` --
a **3-pixel** window, chosen for the tracer's own spur/junction bookkeeping,
never for discriminating myotubes. Over 3 px a skeleton direction is
pixelation noise.

Measured on all 341 labeled junctions, LOWO by well, changing nothing else:

| direction window | LOWO AUC (that feature alone) |
|---|---:|
| **3 px (the original)** | **0.606** |
| 8 px | 0.833 |
| **15 px (chosen)** | **0.858** |
| 30 px | 0.852 |
| 60 px | 0.844 |
| whole-branch PCA | 0.805 |

In the full feature set, swapping 3 px -> 15 px and changing nothing else:

| | LOWO AUC | junction acc | over-merge | under-merge | wrong-pair |
|---|---:|---:|---:|---:|---:|
| classical floor (fixed `STRAIGHT_DOT`) | -- | 23.8% | 112 | 0 | 148 |
| shipped v2 (3 px tangent) | 0.693 | 37.2% | 96 | 29 | 89 |
| **fixed (15 px tangent)** | **0.892** | **58.4%** | 98 | 14 | **30** |
| + multi-scale + branch PCA axis | 0.893 | 58.4% | 97 | 15 | 30 |

Wrong-pair errors fall **89 -> 30 (-66%)**. Multi-scale windows and a
whole-branch PCA axis on top gained nothing (0.893 vs 0.892), so the single
15 px window is kept -- fewer features, same result.

15 px ~= 10 um at 0.6493 um/px, about one myotube width: long enough to
average out pixelation, short enough not to straighten away a genuinely
curving fibre.

**This repeats the fragment linker's own history**, which found its local
endpoint tangent (`min_cos`, a 12-px patch) too noisy and was rescued by a
whole-object axis feature. Same lesson twice: *estimate direction over a
fibre-scale window, not a pixel-scale one.* Worth treating as a project-level
rule, not a one-off.

## 4. What changed in code

- `junction_features.py`: new `DIRECTION_WINDOW_PX = 15` and
  `end_direction()`. `compute_pair_features` now computes `tangent_cos`
  **itself from the branch paths** instead of accepting the tracer's value --
  so the classifier's window can never be silently changed by a tracer
  parameter, and train/inference cannot drift onto different windows. Full
  measurement table recorded in the module docstring.
- `junction_model.py` / `junction_active.py`: updated to the new signature.
- `junction_model.py`: `load_decision_rows()` -- every export-reading entry
  point (`recompute_training_pairs`, `classical_floor_decisions`,
  `ground_truth_decisions`) now takes one export path or many, so multi-round
  training is expressed the same way everywhere. `train-junction-model
  --export` is now `nargs="+"`.
- Tests: 5 new in `test_junction_features.py`, including a **regression guard
  pinning `DIRECTION_WINDOW_PX` well above the tracer's `direction_step`** --
  if that default is ever reset to the tracer's, the classifier silently loses
  ~0.2 AUC with nothing failing. Full suite **261 passed** (was 256).

Model: `junctions_round2/model/junction_classifier_v3.joblib`.
Summary: `junctions_round2/junction_model_v3_summary.json`.

## 5. Where it stands

Junction accuracy **58.4% vs the classical floor's 23.8%** on the floor's own
hardest junctions -- 2.5x the baseline. Remaining error is dominated by
**over-merge** (98 of 112 true branch points still get joined): the model
rarely says "branch point". A decision-threshold sweep only moves accuracy
0.584 -> 0.592 (at 0.55), so this is not a threshold problem -- the model
lacks a feature that positively indicates "this is a real branch point"
rather than merely a weak through-pair.

## 6. Attacking the over-merge error (same session, after the window fix)

Two changes, both measured before implementing:

**(a) `node_intensity_ratio`** -- mean Desmin in a small square at the junction
node over the mean of the pair's two branch-end intensities. Best of six
candidate features tried (also tested: tangent margin vs rivals, junction-best
tangent, third-branch length/width ratios, width-sum conservation -- all
+0.00 to +0.002). Worth **+0.010 AUC (0.892 -> 0.902)**.

Caveat found while testing, and recorded in the module docstring: the
branch-end sampling window *starts at* the node, so it overlaps the node
square, and the ratio is a local contrast rather than a clean "stain continues
through / does not" signal. It earns its place empirically; the tests assert
only the invariants that hold (neutral 1.0 on a uniform field, responsive to
node-region stain), **not** a direction the geometry does not guarantee. A
variant sampling each branch 8-20 px *away* from the node measured weaker
alone (0.896) but complementary (**0.907 with both**) -- banked, not adopted,
to avoid growing the feature set for +0.005 late in the session.

**(b) A two-stage decision rule.** The single-stage rule scores the three pairs
independently and can only call a branch point when all three fail at once --
which a pointwise model trained on pair labels is not optimised to do. Stage 1
is now a dedicated **branch-point gate**: a binary model over whole-junction
features (`JUNCTION_FEATURE_KEYS` -- the three tangents sorted, branch
length/width spread, node intensity; all built from *sorted* statistics so they
are invariant to the arbitrary A/B/C labelling). Only if the gate says
"something continues" does stage 2 pick which pair.

The gate threshold is selected **inside each LOWO fold, on that fold's training
wells only**. Selecting it on the pooled set scored 0.648 -- a threshold tuned
on its own test data; the honest per-fold figure is **0.645**, and the chosen
thresholds were stable across folds (0.65-0.75), which is why it is trustworthy
rather than a fluke. A test pins this contract.

### Progression, all on the same 341 labeled junctions

| | LOWO AUC | junction accuracy | over-merge | under-merge | wrong-pair |
|---|---:|---:|---:|---:|---:|
| classical floor (fixed `STRAIGHT_DOT`) | -- | 23.8% | 112 | 0 | 148 |
| shipped v2 (3 px tangent) | 0.693 | 37.2% | 96 | 29 | 89 |
| v3: 15 px direction window | 0.892 | 58.4% | 98 | 14 | 30 |
| v4a: + `node_intensity_ratio` | 0.902 | 60.1% | 93 | 15 | 28 |
| **v4b: + two-stage branch-point gate** | 0.902 | **64.5%** | **68** | 23 | 30 |

**Over-merge 112 -> 68** against the classical floor, on the floor's own
hardest junctions; overall **2.7x the baseline accuracy**. Model:
`junctions_round2/model/junction_classifier_v4.joblib`, summary
`junction_model_v4_summary.json`. Full suite **273 passed** (was 261).

## 7. Learning curve RE-RUN at the v4 feature set -- labeling stays closed

The §2 curve was measured on the broken 3-px features, so it could not settle
whether a *stronger* model would have a different data appetite. Re-measured
on v4 (all three stages: pair model, branch-point gate, end-to-end two-stage
decision; 10 seeds; train on N junctions from 5 wells, evaluate the full
held-out well, gate threshold picked on training wells only):

| N junctions | pair AUC | gate AUC | two-stage accuracy |
|---:|---:|---:|---:|
| 40 | 0.8892 | 0.6922 | 0.5874 |
| 60 | 0.8940 | 0.7235 | 0.6192 |
| 80 | 0.8973 | 0.7320 | 0.6237 |
| 120 | 0.8997 | 0.7455 | 0.6341 |
| 160 | 0.9007 | 0.7511 | 0.6334 |
| 200 | 0.9011 | 0.7561 | 0.6379 |
| 240 | 0.9015 | 0.7576 | 0.6425 |
| 262 (max) | 0.9018 | 0.7575 | 0.6344 |

Slope over the last 62 junctions, **per 100 additional junctions**:
pair AUC **+0.0012**, gate AUC **+0.0023**, two-stage accuracy **-0.0056**.

**Verdict: still flat, and flatter than before.** The better feature set did
*not* create an appetite for more data -- it raised the ceiling and then hit
it sooner. Two things stand out:

- The **pair model is essentially saturated at N=40** (0.889 of its eventual
  0.902). Nearly all of the 395 labels collected were surplus to what the
  pairwise stage needed.
- The **gate** is the only stage that used the extra data, and it saturates
  around **N=120-160**. We have 341. Everything past ~160 bought nothing.

So the honest reading of the two rounds: **round 1 alone (245) was already
past saturation; round 2's 150 labels bought ~0.** The value round 2 delivered
was diagnostic, not statistical -- it is what exposed the 3-px window bug, and
that was worth far more than the labels themselves.

**Labeling should not resume.** The ceiling (~0.90 pair AUC, ~64% junction
accuracy) is set by the features, and no amount of additional junctions moves
it.

## 8. All three remaining levers tested -- all three failed

Each was measured before implementing. **None beat v4 end-to-end.** Reported
because negative results here are what justify stopping rather than grinding.

**(1) Image-derived branch-point features for the gate.** Three ideas, all
targeting "does anything actually continue through this node":

| gate feature set | gate AUC |
|---|---:|
| **v4 geometric only (current)** | **0.7587** |
| + through-chord minimum stain (the linker's `bridge_over_bg`, moved to a junction) | 0.7556 |
| + all four chord statistics | 0.7512 |
| + node fatness (distance-to-bg at node vs branch widths) | 0.7562 |
| + node blob area (territory in a disk vs one fibre's worth) | 0.7554 |
| + everything new | 0.7515 |

Every variant **hurt**. Alone they are weak (chord 0.582, node fatness 0.553,
blob 0.684). The physical intuition -- that a genuine merge point looks fatter,
or that a real pass-through keeps stain along the chord -- does not survive
contact with the data at this scale.

**(2) Non-linear models.** The fragment linker's precedent said GBM should beat
logistic regression (F1 0.60 vs 0.44 there). It does not here:

| pair model | gate model | pair AUC | gate AUC | two-stage accuracy |
|---|---|---:|---:|---:|
| **logreg** | **logreg** | **0.9019** | 0.7587 | **0.6450** |
| gbm | logreg | 0.8855 | 0.7587 | 0.6300 |
| logreg | gbm | 0.9019 | 0.7543 | 0.6220 |
| rf | rf | 0.8896 | **0.7748** | 0.6360 |
| gbm | gbm | 0.8855 | 0.7543 | 0.6160 |

Random forest gives the best *gate AUC* (0.7748) and still a **worse**
end-to-end accuracy (0.636). With 341 junctions the non-linear models overfit.

**(3) The banked offset node feature.** Raises pair AUC as advertised but does
not move the objective:

| | pair AUC | gate AUC | two-stage accuracy |
|---|---:|---:|---:|
| v4 shipped | 0.9019 | 0.7587 | **0.6450** |
| + `node_offset_ratio` (pair) | **0.9067** | 0.7587 | 0.6420 |
| + offset on the gate | 0.9019 | 0.7547 | 0.6280 |

A useful reminder in its own right: **pair AUC is not the objective.** +0.005
AUC bought −0.003 accuracy. Not adopted.

**Conclusion: v4 is the ceiling for this approach.** Data is saturated (§7),
the feature space is explored, and the model class is already the right one.

## 9. What v4 is actually good for: a high-precision assistant

64.5% overall is not deployable unattended -- but that was also true of the
fragment linker, which became useful by auto-applying its confident minority
and routing the rest to review. Same question asked here, ranking junctions by
confidence (gate distance from its threshold, and how clearly the top pair
beats the second):

| coverage | n | accuracy on that subset | errors |
|---:|---:|---:|---:|
| 10% | 34 | **0.882** | 4 |
| 20% | 68 | 0.838 | 11 |
| 30% | 102 | 0.824 | 18 |
| 50% | 170 | 0.759 | 41 |
| 100% | 341 | 0.645 | 121 |

The ranking is meaningful and monotone, so **"auto-apply the top ~30% at ~82%
accuracy, route the remaining 70% to review"** is a real operating point --
the same shape of deliverable the linker settled on. It is *not* good enough
for unattended application at any coverage (never reaches ~95%), and that
should be stated plainly wherever this model is used.

## 10. Honest status

The junction classifier is **2.7x the classical floor** (64.5% vs 23.8%) on
the floor's own hardest junctions, usable as a high-precision assistant at
partial coverage, and **at the ceiling of what this feature set, this data,
and this model class can do**. Three independent attempts to push further all
failed and are recorded above so they are not repeated.

If the project wants materially better junction splitting, the evidence points
away from incremental work on this classifier and toward a different
information source -- which is the case the parked Omnipose candidate was
originally meant to serve, and which the direction-pipeline plan explicitly
flagged as the residual (parallel bundles / genuinely ambiguous nodes where
direction cannot decide). That is an operator/integrator call, not something
to settle by more tuning here.

If labeling does resume later, the 220 round-2 junctions dropped as
least-uncertain are already scored and banked in
`junctions_round2.manifest.json`, and the broader pool is still available via
`--reasons all`.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence -- not consensus, not inter-rater
agreement, not prospective validation.
