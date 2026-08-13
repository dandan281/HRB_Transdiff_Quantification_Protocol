# Junction classifier round 1 — trained and evaluated

2026-07-28. Continues `claude_junction_ambiguity_measurement_2026-07-23.md`
(step 1: measured the pool, 245 junctions) and
`claude_junction_labeling_round1_build_2026-07-23.md` (step 2+3: built the
labeling tool + features). This is step 4: train the classifier, compare to
the classical floor's fixed pairing.

## Labels

Operator (`reviewer_01`) decided all 245 round-1 junctions:
**145 through-pair, 59 genuine branch point, 40 unsure** (dropped). Export:
`PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.junctions.json`.

Each decided junction yields 3 pair-level rows → **735 rows, 615 usable
(145 positive / 470 negative)**, 120 dropped as unsure. 615/145 is comfortably
past the linker's `MIN_POSITIVES=8` floor and far more starting data than the
linker's own round 1 (91 pairs / 27 positives) had.

Code: `annotation_tools/annotation_tools/qc_review/junction_model.py`
(`recompute_training_pairs`, `select_feature_set`, `fit_junction_classifier`,
`leave_one_well_out_junction_decisions`, `classical_floor_decisions`,
`decision_accuracy`) + `train-junction-model` CLI subcommand. 11 new tests, all
passing; full suite **248 passed** (was 237).

```
python -m annotation_tools.qc_review.cli train-junction-model \
  --export PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.junctions.json \
  --out PrecisionMyotube/annotation_work/junctions_round1/junction_model_v1_summary.json \
  --model-out PrecisionMyotube/annotation_work/junctions_round1/model/junction_classifier_v1.joblib
```

## Feature-set selection (LOWO AUC, pair level)

| feature set | AUC |
|---|---:|
| tangent_only (the classical floor's only signal) | 0.580 |
| tangent_turn | 0.584 |
| tangent_width | 0.604 |
| tangent_intensity | 0.662 |
| tangent_width_intensity | 0.666 |
| **all (chosen)** | **0.701** |

`tangent_cos` alone is barely better than a coin flip (0.58) on this pool --
unsurprising, since the pool is specifically the junctions where that one
signal was already ambiguous. Every additional feature helps monotonically;
`all` (tangent_cos + turn_angle_deg + width_ratio + intensity_ratio +
length_min_um) wins outright, so no feature was dropped for overfitting at
this sample size (unlike the linker's round 1, which needed to cut down to 1-2
features with only 27 positives).

## Junction-level decision accuracy: classifier vs. classical floor

Both scored against the same 205 decided (non-unsure) junctions the operator
labeled. "Decision" = which pair continues through, or none (branch point).

| | accuracy | correct | over-merge (wrongly joins) | under-merge (wrongly splits) | wrong pair chosen |
|---|---:|---:|---:|---:|---:|
| **classical floor** (fixed `STRAIGHT_DOT=-0.5`) | **20.5%** (42/205) | 42 | 60 | 0 | 103 |
| **junction classifier** (LOWO out-of-fold) | **41.0%** (84/205) | 84 | 47 | 20 | 54 |

**The classifier roughly doubles the classical floor's accuracy on its own
hardest cases.** Two things stand out:

1. **The classical floor is at-or-below chance on this pool, by construction.**
   These 205 junctions were selected *because* the floor's decision was
   ambiguous, so this is an adversarial evaluation set, not a random sample --
   20.5% against a ~25% random-guess floor (3 pairs + branch point) is not "the
   floor is bad in general," it's "the floor is uniformly bad on exactly the
   cases we handed it that are hard." That was the premise for building this
   classifier at all, and the measurement bears it out sharply.
2. **The floor never predicts "branch point" on this pool** (0 under-merge
   errors) -- every one of the 59 true branch points became an over-merge
   error (wrongly joins two things that are not the same myotube). Combined
   with 103 wrong-pair errors on the 145 true through-pairs, the floor is
   *confidently wrong* far more often than it is uncertain: only 42/205 (20%)
   correct, and it always commits to *some* answer. The classifier trades some
   of that overconfidence for correctly declining (20 under-merge errors) and
   for correctly recognizing branch points (47 vs 60 over-merge), while more
   than halving wrong-pair errors (54 vs 103).

## Where this sits relative to the linker

AUC 0.701 / junction accuracy 41% is a real, measured improvement over the
floor, but by the linker's own precedent (round 1 AUC ~0.80, "not deployable"
until round 3 reached 0.902 / precision 0.68), this is **a first-round result,
not a finished model** -- 41% junction accuracy is not something to wire into
production. The natural next step, mirroring the linker's exact playbook, is
an **active-learning round 2**: score new (or the excluded broader-pool, 615
vs 245) junctions with this model, serve the ones nearest 0.5 probability to
the operator, and retrain. `junction_active.py` does not exist yet -- not
built this session, since it depends on the operator's judgment about whether
to proceed now or bank this as the round-1 result.

## Not yet done

- **Active-learning round 2** (`junction_active.py`, modeled on
  `link_active.py`) -- not built, awaiting operator direction.
- **Wiring into the classical floor's `trace_fibers_parameterised`** -- this
  classifier only scores candidate pairs; nothing yet replaces
  `pair_junction_ends`'s greedy `STRAIGHT_DOT` rule with the learned model in
  the actual tracer. That is a separate integration step once the model is
  good enough to trust.
- Model saved at
  `PrecisionMyotube/annotation_work/junctions_round1/model/junction_classifier_v1.joblib`;
  full summary at
  `PrecisionMyotube/annotation_work/junctions_round1/junction_model_v1_summary.json`.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence -- not consensus, not inter-rater
agreement, not prospective validation.
