# Junction-splitting labeling round 1 — built (step 2+3 of the plan)

2026-07-23. CPU-only. Continues `claude_junction_ambiguity_measurement_2026-07-23.md`
(step 1: measured the pool). This session built the operator-labeling tool
(step 2) and the reusable feature module (step 3) it will feed. **Step 4
(train the classifier) is not started** -- it needs the operator's actual
labels, which don't exist yet.

## What's built

New, in the Claude lane, nothing committed:

- `model_labs/classical/ridge_graph.py` -- refactored (no behaviour change,
  confirmed by the existing 16 tests) to expose `build_branch_graph`,
  `junction_candidates`, `pair_junction_ends` so downstream tools reuse the
  exact branch-graph/pairing logic instead of re-deriving it.
- `model_labs/classical/junction_ambiguity.py` -- refactored to expose
  `evaluate_junction`, the single source of truth for "is this junction
  ambiguous and why," now shared between the measurement report and the
  candidate finder below (confirmed identical counts before/after: 615/893/167).
- `annotation_tools/annotation_tools/qc_review/junction_pairs.py` (new) --
  candidate finder. Scoped to **degree-3 junctions only** (degree>=4 never
  fired in the measurement, so it's out of scope rather than mismodelled).
  `ROUND1_REASONS = (near_threshold_winner, width_or_intensity_conflict)` is
  the recommended 245-junction pool; `reasons=None` gives the broader 615.
- `annotation_tools/annotation_tools/qc_review/junction_features.py` (new) --
  `tangent_cos`, `turn_angle_deg`, `width_ratio`, `intensity_ratio`,
  `length_min_um` per candidate pair, in the style of `link_features.py`.
  Built now (ahead of having labels) so it's ready the moment labels exist;
  not yet wired to a trained model.
- `annotation_tools/annotation_tools/qc_review/junction_page.py` (new) --
  the labeling page. Reuses the linker page's proven shell (theme,
  decide-and-advance, arrow nav, brightness/contrast, `?` shortcuts,
  localStorage draft, copy/download export dialog) but the **decision model
  is deliberately different**: a junction is one location where exactly
  three branch-ends meet, and a real fibre passes through at most one
  partner there -- so this is a **single choice** among the three possible
  pairs (keys `1`/`2`/`3`) plus `N` (genuine branch point) and `U` (unsure),
  not the linker's multi-select. Whichever pair isn't chosen exports as a
  labelled negative automatically. The classical floor's current dot-product
  pairing is never sent to the page (confirmed by test) -- same
  anchoring-avoidance principle as hiding the linker's model score.
- `build-junction-page` subcommand wired into
  `annotation_tools/annotation_tools/qc_review/cli.py`.

Tests: `annotation_tools/tests/test_junction_pairs.py` (7),
`test_junction_features.py` (9), `test_junction_page.py` (17) -- 33 new,
all passing. Full suite: **237 passed** (was 204).

## Round 1 page generated

```
python -m annotation_tools.qc_review.cli build-junction-page \
  --out PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.html \
  --reviewer reviewer_01 --batch-id junctions_round1
```

**245 junctions**, matching the recommended pool exactly:

| well | junctions |
|---|---:|
| 19_B06_act104_trka | 58 |
| 22_B03_act104_egfrc | 51 |
| 23_B02_ctrl | 9 |
| 29_C05_br223_egfrc | 44 |
| 32_C08_br223_igf1r | 64 |
| 33_C09_br223_trka | 19 |

`PrecisionMyotube/annotation_work/junctions_round1/junctions_round1.html`
(11.7 MB, self-contained -- open directly in a browser, no server). A sample
crop was decoded and visually verified: three distinctly coloured branch
paths (cyan/amber/violet) correctly traced on the raw Desmin/DAPI composite
at a real junction.

## Operator instructions

Open the html file. For each junction: pick `1`/`2`/`3` for whichever pair of
branches is the same myotube passing straight through, `N` if it's a genuine
branch point (none continue), or `U` if unsure. `L` toggles the branch
outlines off to see the bare stain. `Enter` confirms and advances, `J` jumps
to the next undecided junction. When done, "Export all" -> download
`junctions_round1.junctions.json` (or copy/paste) and hand it back.

## Not yet done

Step 4 (train a small sklearn classifier, LOWO by well, baseline = the fixed
`STRAIGHT_DOT` pairing) is blocked on the operator's labels from this round.
`junction_model.py` (the `link_model.py` analogue) is not built yet --
building it against zero real data risks bugs going unnoticed, so it waits
for the labeled export.
