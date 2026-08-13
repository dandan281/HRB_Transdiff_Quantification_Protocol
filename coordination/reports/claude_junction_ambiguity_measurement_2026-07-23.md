# Junction-ambiguity measurement (step 1 of the junction-classifier plan)

2026-07-23. CPU-only. Six bootstrap wells, cached stage-A territory, canonical
`TracerParams()` defaults (`spur_um=10.0`, `straight_dot=-0.5`, `direction_step=3`
-- reproduces `precision_myotube.fiber_gate.trace_fibers`'s grouping exactly).

Code: `model_labs/classical/junction_ambiguity.py`. Refactored
`model_labs/classical/ridge_graph.py` to expose `build_branch_graph`,
`junction_candidates`, `pair_junction_ends` (pure extraction of existing logic,
zero behaviour change -- the 16 `test_classical_ridge_graph.py` tests and the
full 204-test suite still pass). These are reusable infrastructure for the
junction classifier's feature-building step, not single-use.

## Headline finding: the raw junction count is dominated by skeleton noise, not crossings

`build_branch_graph` only prunes short **dead-end** branches (the canonical spur
rule); it never prunes short junction-to-junction branches. A raw degree>=3 node
count therefore includes enormous numbers of sub-pixel skeletonisation whiskers
off an irregular territory boundary. Measured on `23_B02_ctrl`: 30,631 branches,
median length **3.9 um (~6 px)**, 95th percentile only 12.0 um; of 2,957 raw
degree>=3 nodes, only 27 (0.9%) have every incident branch >= 10 um.

So the measurement gates on **fibre-scale candidate junctions**: every incident
branch must clear `MIN_BRANCH_UM = 10.0` (matching the codebase's own `spur_um`
convention for "real fibre," not noise). This is a real methodological finding,
not just filtering -- the naive count would have been ~10x too large and mostly
meaningless.

## Ambiguity criteria (a candidate junction is "ambiguous" if any fire)

- `near_threshold` -- any candidate pair at the junction (winner or loser) has
  |dot - straight_dot| <= 0.15. Deliberately evaluated over every candidate, not
  just the winner: the classifier's training unit is a candidate *pair*, and a
  junction whose losing candidates are also decisively far from the boundary
  offers a labeler nothing the winner didn't already establish.
- `near_threshold_winner` (reported separately, stricter) -- the *adopted* pairing
  (or the best available candidate, if none qualified) is itself near the
  boundary -- i.e. a slightly different fold-fitted `straight_dot` would flip
  this specific decision.
- `degree_ge4` -- 4+ branch-ends meet here (X-crossing or busier): more than one
  simultaneous pairing is geometrically possible.
- `width_or_intensity_conflict` -- the adopted pair's local branch width or mean
  Desmin intensity ratio (min/max over a 5 px window at each end) is < 0.5,
  despite being paired by direction alone.

## Results (six bootstrap wells)

| well | raw degree>=3 | fibre-scale candidates (>=10um) | ambiguous | near_threshold | degree_ge4 | width/intensity conflict |
|---|---:|---:|---:|---:|---:|---:|
| 19_B06_act104_trka | 975 | 209 | 143 | 136 | 0 | 28 |
| 22_B03_act104_egfrc | 936 | 223 | 161 | 153 | 0 | 17 |
| 23_B02_ctrl | 2957 | 27 | 20 | 18 | 0 | 6 |
| 29_C05_br223_egfrc | 1222 | 147 | 99 | 95 | 0 | 15 |
| 32_C08_br223_igf1r | 896 | 207 | 143 | 137 | 0 | 29 |
| 33_C09_br223_trka | 1598 | 80 | 49 | 44 | 0 | 10 |
| **total** | **8584** | **893** | **615** | **583** | **0** | **105** |

**Headline count: 615 ambiguous junctions out of 893 fibre-scale candidates**
(the operator-labeling pool, any-candidate-near-boundary definition). Of those,
**167** are cases where the actually-adopted pairing itself sits near the
boundary (the stricter sub-count) -- closer in scale to what one active-learning
round can absorb, matching the linker's precedent (~65 fragments / 251 pairs
across 3 rounds).

Two things worth flagging before designing the labeling round:
1. `near_threshold` (91-95% of ambiguous flags) dominates completely --
   `width_or_intensity_conflict` (12%) is a modest secondary signal and
   `degree_ge4` **never fires** on fibre-scale candidates across all six wells.
   True X-crossings either don't present as single degree-4 skan nodes in this
   data, or are genuinely rare; this criterion currently contributes nothing and
   may not be worth keeping in the feature set.
2. The gap between 615 (any-candidate) and 167 (winner-only) is the real design
   choice for round 1: label the broader pool for maximum contrastive training
   signal, or start with the tighter, more tractable 167 and expand later like
   the linker did.

Full per-junction detail (node id, centroid, reasons, pair diagnostics) is in
`model_labs/classical/_runs/junction_ambiguity_v1.json`.

## Not yet done (per instruction: report the count first)

Steps 2-4 of the plan (labeling-round UI, junction feature extraction, sklearn
classifier) are **not started**. Awaiting operator direction on: which pool size
to label first (615 vs 167), and whether `degree_ge4` should be dropped given it
never fired.
