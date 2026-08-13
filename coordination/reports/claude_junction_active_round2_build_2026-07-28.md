# Junction active-learning round 2 — built, ready for the operator

2026-07-28. Continues `claude_junction_classifier_round1_results_2026-07-28.md`
(round 1: 245 labels → LOWO AUC 0.701, junction accuracy 41% vs the classical
floor's 20.5%). This session built the active-learning round-2 tool and
generated the page. **Awaiting operator labeling.**

## What's built

- `annotation_tools/annotation_tools/qc_review/junction_active.py` (new) --
  the round-2 builder, mirroring `link_active.py`'s proven structure:
  1. **train** the classifier on every prior round's labels,
  2. **widen the pool** from round 1's narrower ambiguity criteria
     (`near_threshold_winner | width_or_intensity_conflict`) to the full
     near-threshold-broad pool (`reasons=None`), a strict superset,
  3. **rank by uncertainty** (`1 - 2*|proba - 0.5|`) on each junction's
     best-guess pair, most-uncertain first,
  4. **never re-serve** any `(well, node)` from any prior export -- including
     branch-point and unsure outcomes, so the operator never re-decides
     something they already looked at.
  The model's guess is never sent to the page (same anti-anchoring principle as
  the linker and blind-repeat tools; enforced by an existing round-1 test).
- `build-junction-active-round` CLI subcommand.
- `find_junction_cases` now also returns `node_ends`, so the scorer reuses the
  branch graph the candidate finder already walked instead of rebuilding it.

**Bug found and fixed while testing:** `leave_one_well_out_auc` and
`leave_one_well_out_junction_decisions` crashed with a numpy zero-size
reduction when the label set contained only one well (holding it out leaves an
empty training set). Now guarded, and `select_feature_set` raises a message
naming the actual cause ("leave-one-well-out needs at least 2 wells") instead
of surfacing the numpy error. Not hypothetical — a single-well debugging run
would have hit it.

Tests: `annotation_tools/tests/test_junction_active.py` (8 new). Full suite
**256 passed** (was 248).

## Round-2 page generated

```
python -m annotation_tools.qc_review.cli build-junction-active-round \
  --prior-export PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.junctions.json \
  --out PrecisionMyotube/annotation_work/junctions_round2/junctions_round2.html \
  --reviewer reviewer_01 --batch-id junctions_active_r2 --max-junctions 150
```

Widening the pool surfaced **370 new junctions** (never offered in round 1)
across the six wells; the 150 most-uncertain were served, **220 dropped as
least-uncertain and reported** (not silently truncated).

| well | pool candidates | new (not in round 1) |
|---|---:|---:|
| 19_B06_act104_trka | 143 | 85 |
| 22_B03_act104_egfrc | 161 | 110 |
| 23_B02_ctrl | 20 | 11 |
| 29_C05_br223_egfrc | 99 | 55 |
| 32_C08_br223_igf1r | 143 | 79 |
| 33_C09_br223_trka | 49 | 30 |
| **total** | **615** | **370** |

Verified on the generated page: 150 cases, ordered by uncertainty, **zero
overlap with round 1**. The most-uncertain served junctions sit at model
probability 0.5014 / 0.4969 / 0.4953 — essentially coin-flips for the current
model, which is exactly what uncertainty sampling should surface. Even the
*least*-uncertain served junction is at 0.69, so the whole batch is
genuinely informative rather than padding.

`PrecisionMyotube/annotation_work/junctions_round2/junctions_round2.html`
(5.4 MB, self-contained). Manifest with per-junction probabilities at
`junctions_round2.manifest.json`.

## Operator instructions

Identical to round 1 — same tool, same keys. `1`/`2`/`3` = which pair
continues through, `N` = genuine branch point, `U` = unsure, `L` = toggle
outlines, `Enter` = confirm and advance, `J` = jump to next undecided.
Export → download `junctions_active_r2.junctions.json`.

Per the 2026-07-27 operator note (`myotube-no-orthogonal-branching` memory):
a near-orthogonal candidate is **`N` (branch point)**, decided, not `U` —
myotubes never branch at ~90°, so that is resolvable domain knowledge rather
than genuine ambiguity.

## Next (after labeling)

Retrain on round 1 + round 2 combined:

```
python -m annotation_tools.qc_review.cli train-junction-model \
  --export <round1+round2 combined or each in turn> ...
```

`train-junction-model` currently accepts a single `--export`; combining two
rounds needs either a small multi-export flag or a merged file. That is a
~5-line change, deliberately not made until round-2 labels actually exist so
it can be tested against real data rather than guessed at.

Watch for: whether AUC moves past 0.701 and junction accuracy past 41%. The
linker's precedent (0.804 → 0.839 → 0.895 → 0.902 over three rounds, with the
biggest jump coming from a *feature* insight rather than more labels) suggests
round 2 should give a solid lift, and that if it plateaus early the answer is
a better feature — not more rounds.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence.
