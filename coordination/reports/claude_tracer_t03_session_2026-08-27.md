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

## 7a. Fragmentation troubleshoot (same day, after the sealed run)

The operator reported continued over-fragmentation. A cross-plate census
(`frag_census.py`) and a per-pixel attribution (`frag_attribution.py`) on
two bootstrap wells plus a never-seen PLATE_32 reference (C05, fold-C05):

| well | unwalked spine | of which: field hole | unseedable crest | seedable crest, walk missed |
|---|---|---|---|---|
| 23_B02 (P23) | 45% | 78% | 12% | 10% |
| 19_B06 (P23) | 21% | 46% | 11% | 42% |
| C05 (P32 ref) | 51% | 60% | 3% | 37% |

Findings, in order of weight:

1. **True many-piece fragmentation is the minor failure**: 7–14% of
   certified fibres are covered by ≥2 substantial pieces (19% on dense
   P32). The dominant failure is partial/zero coverage — which is also what
   fails IoU 0.5 matching.
2. **No cross-plate domain shift**: the in-plate reference behaves the same
   or worse (walk cov 0.47 vs 0.46/0.71 on P23); normalized spine
   brightness matches (0.054 vs 0.050). The fold generalizes.
3. **The NMS crest is confetti**: 996–4101 connected components per well,
   only ~47% containing a seed-level pixel. Raw hole sizes are SMALL —
   walk-gap p50 4–6 px, p75 8–20 px, p90 25–64 px. The §11 "90 px" figure
   is the distance between the pieces that SURVIVE pruning, not the hole
   size; most holes are strideable.
4. **Field holes dominate on dim wells** (78% on 23_B02, image spine p50
   0.05) — the known dim-stretch blindness; network-side fixes are
   exhausted (nine refutations).
5. **A second, untried mechanism is real**: 10–42% of missing coverage sits
   on existing crest the walk failed to cover (died at small holes; short
   components pruned or never seeded). In-walk gap tolerance (§5d item 1 of
   the 2026-08-23 report) was never among the nine rejections — the
   rejected bridging was post-hoc, end-to-end, 120 px probes. The failed
   variant's mdape cost came from permanently appended probe paths; an
   in-walk variant can roll back unlanded probes so failure adds no length.

Bounded expectation if all seedable-crest coverage were recovered: 19_B06
walk cov 0.71 → ~0.83; 23_B02 0.46 → ~0.50 (stays field-limited).

Artifacts: `_runs/eval_bootstrap_v1/frag_diag.json`,
`unwalked_attribution.json`, `frag_overlay_23_B02_ctrl_1200_1200.png`.

## 7b. The junction weld: the crossing-cut fix, tuned, frozen, claimed

The operator's complaint made concrete: one intact myotube crossing another
gets traced as 2–3 objects. Break-point attribution
(`frag_attribution.py` → `_runs/eval_bootstrap_v1/break_attribution.json`):
~1 in 5 fibres is cut; breaks sit at predicted crossings 2–3× above base
rate (53% vs 17% on C05); ~1/3 of breaks ABUT (pieces touch, only identity
failed); breaks are also ~half as bright as average spine.

Two mechanisms tested (sweep: `weld_rescue_sweep.py`, tune wells only,
selection rule predeclared in the docstring):

1. **Rescue widening — INERT.** `rescue_window_steps` 2/4/8 and reach 4
   moved pooled splits by ≤4 of 568 and bought nothing. The frozen 1/2.0
   stands. (Refutes the "rescue barely gets to fire" hypothesis.)
2. **Junction weld — ACTIVE and dose-responsive**
   (`weld_objects` in `oracle_trace.py`): post-walk identity merge of
   co-linear pieces meeting at a predicted crossing; connector-angle guard
   blocks parallel-neighbour merges (the 2026-07 linker failure); crossing
   gate blocks open-field fragment joining; merges bookkeeping only — no
   fabricated arc. Validated on 5 synthetic contract cases first
   (3 new tests in `test_oracle_trace.py`, suite 18/18).

Tune sweep (C02 C03 C05 C11 D02, never-seen folds, pooled):
baseline 568 splits / 292 merges; weld dist 14 deg 12.5 → 521 / 307
(net 32, guards pass). Dose-response continues past the grid edge
(dist 26: 439 splits, net 92) but the predeclared mdape guard
(≤ base + 0.01) binds at dist 14 — the steeper points cost per-fibre
length (mdape 0.306 → 0.342 at dist 26) and are a user decision, not a
rule change. **Frozen: weld_dist 14 px, weld_deg 12.5°, gate 12 px,
rescue unchanged.**

CLAIM on test wells (B02 D04 D08 D09 D11, one shot,
`_runs/weld_rescue_claim.json`):

| | splits | merges | identity_x | mdape |
|---|---|---|---|---|
| baseline | 776 | 437 | 0.379 | 0.359 |
| + weld (frozen) | **707** | 449 | **0.417** | 0.374 |

−69 splits for +12 merges (5.8 repairs per added merge); improvement holds
on every well and drop-one-well (net 41–50). Cost: mdape +0.015, driven by
D04 (+0.063); D09 improved, D08 flat. Visual:
`_runs/weld_before_after_D04_2400_1400.png`.

Open decision for the operator: adopt dist 14 (frozen claim above), or
re-prioritize splits over per-fibre length and re-freeze at a steeper
tune-well operating point (up to −23% splits at dist 26 for mdape +0.036
on tune). Not yet wired into `cv_report.py`/`quantify_plate.py`; any T03
re-run with the weld is candidate #2 and goes through Codex.

## 7c. Never-seen plates: PLATE_26 + PLATE_28 vs the operator's ROIs

The purest data the project owns: 8 annotated wells (P26: B02 B06 C08;
P28: B02 B04 B08 E08 E10) that took no part in training (PLATE_32), tuning
(PLATE_32 tune wells), or the sealed benchmark (PLATE_23). The recurring
``23_B02_ctrl.nd2`` filename is a per-plate acquisition index — all four
same-named nd2s hash-distinct. Pixel size 0.6493 µm (0.1% off the training
plate; no rescale). Config predeclared: fold-B02 checkpoint, frozen nms
walk, frozen weld (dist 14). Script: `eval_unseen_plates.py`; results:
`_runs/eval_unseen_plates.json`; overlays `_runs/unseen_overlay_*.png`.

Pooled over 8 wells (weld arm; baseline in the JSON):

| metric | value | context |
|---|---|---|
| trace recall (mean) | **0.87** | human self-consistency 0.71; P32 CV 0.645 |
| count ratio | 1.59× | human repeat 1.72×; P32 CV 1.17× |
| total length ratio | **1.78×** | P32 CV 0.95× |
| length rank ρ (n=8) | +0.81 | P32 CV +0.90 (n=10) |
| count rank ρ (n=8) | +0.59 | P32 CV +0.95 |
| matched per-fibre mdape | 0.516 | P32 CV 0.323; human 0.096 |
| splits / merges | 296 / 86 | weld: −18 splits, +6 merges, idx +0.03 vs base |

Interpretation, unresolved pending the operator's read of the overlays:
the tracer finds nearly everything annotated (0.87) plus ~80% more length.
If these plates' ROI annotation is selective (the PLATE_32 re-trace showed
fresh eyes find +72% fibres on even the dense corpus), the 1.78× is partly
real unannotated fibre and the matched mdape is inflated wherever the
operator traced part of a fibre the tracer traced in full. If the
annotation is complete, the tracer over-traces on these plates. The
overlays (P28_B04: many tracer-only fibres; P26_B02: close agreement)
suggest annotation completeness varies by well. The operator adjudicates.

## 7d. Length-measurement convention: the "calibration" discrepancy resolved

The ~12% gap between the operator's Fiji CSV lengths and raw arc-length
re-measure of the same ROIs is NOT calibration. Diagnosis
(scratchpad `fit_calibration.py` / `staircase_test.py`, all 8 P26/P28
wells): the ROIs are `freeline` (freehand, ~4 px point spacing); Fiji
smooths freehand coordinates before measuring; raw point-to-point arc
counts the drawing jitter and inflates by 10–15%, more for wigglier
fibres — which is why the apparent "calibration" varied per fibre and per
well (0.548–0.594). Under a 5-point moving-average smoothing the fitted
scale collapses to **0.657 ± 0.003 across all eight wells** — one
constant, ≈ the nd2 metadata 0.6493 (residual ~1.3% = my smoother vs
ImageJ's exact freehand algorithm). **The operator's CSVs were correct all
along.**

Consequences:

1. Tracer paths are already smooth (raw/smoothed = 1.007) — the inflation
   is one-sided, on raw-arc measures of freehand HUMAN traces only.
2. B02 (P26) matched-fibre ratio under the converged convention:
   tracer/operator **1.74** (was 1.61 under the inflated human measure);
   totals 44.4 vs 22.3 mm. The "tracer claims more length" finding
   strengthens slightly.
3. **Lane-wide implication, unresolved**: every `human_mm` and `gt_len`
   computed by raw arc on freeline ROIs — including the PLATE_32 corpus
   and the CV report's 0.95× total-length ratio — carries ~10–15% human-
   side inflation. If PLATE_32 traces jitter similarly, the true CV ratio
   is nearer 1.05–1.10× and per-fibre mdape values shift. Adopting the
   smoothed convention lane-wide is a metric redefinition: it needs a
   deliberate decision (operator/Codex), not a quiet edit, and would
   require re-stating prior tables under both conventions once.

## 7e. Convention restate + the length-class distribution
(`length_distribution_report.py`, `_runs/length_distribution_report.json`,
figure `_runs/length_proportions.png`)

Restated totals (smoothed convention both sides; PLATE_32 tracer = current
version walk+weld): the PLATE_32 total-length ratio moves from ~0.95× (raw
human, no weld) to **~1.02× mean** (per well 0.84–1.35). The unseen-plate
ratios vs the operator's own CSVs are 1.67–2.75 (mean ~2.0). Old numbers
kept beside new in the JSON — nothing silently replaced.

The operator's stated metric of interest — the PROPORTION of myotubes per
length class, not totals (short/long over-tracing cancels in totals):

- **PLATE_32 (dense GT): the tracer skews short.** 50–150 µm share 51% vs
  the operator's 29%; 150–500 µm under-represented (42% vs 66%); median
  148 vs 210 µm; KS D = 0.227. This is fragmentation expressed as a
  distribution: surplus short pieces at the expense of mid-length fibres,
  plus 2% vs 0.2% in >800 µm (mega-merges).
- **PLATE_26/28 (never-seen): the shapes nearly agree.** 50–150 µm 43% vs
  43%; medians 175 vs 169 µm; KS D = 0.099. Residual differences: tracer
  slightly light in 150–300 (28% vs 38%) and heavier in the ≥500 µm tail
  (12% vs 5%).

The asymmetry is informative: against dense exhaustive annotation the
fragmentation bias is visible; on the sparser-annotated plates the tracer's
length mix matches the operator's closely, with the long tail (over-merges/
extensions) as the main distortion.

## 7f. Decompose-and-retrace (operator's sketch, 2026-08-28): built, six
iterations on the tune well, promising and not yet net-positive

`decompose_retrace.py`: first pass (frozen walk+weld) → transverse-conflict
graph → graph coloring into K sparse groups → per-group masked sub-image
(corridors + endpoint extensions) → re-predict + re-walk each → sum, each
sub-image answering only for its own members. Iteration ledger on C05
(tune well; baseline splits 105 / merges 59 / recall 0.65 / idx 0.40 /
mdape 0.32):

| version | change | splits | merges | recall | idx | mdape |
|---|---|---|---|---|---|---|
| v1 | path-contact conflicts only | 128 | 69 | 0.71 | 0.40 | 0.351 |
| v2 | + responsibility filter | 127 | 69 | 0.71 | 0.40 | 0.347 |
| v3 | + extended-spine conflicts, balanced groups, ext 90 | 227 | 131 | 0.92 | 0.42 | 1.067 |
| v4 | mask ext 90→40 | 173 | 116 | 0.89 | 0.50 | 0.890 |
| v5 | mask ext 40→20 | 146 | 96 | 0.86 | 0.51 | 0.757 |
| v6 | + parallel-run conflicts | 146 | 95 | 0.86 | 0.51 | 0.755 |

What was learned, each by its own measurement:

1. Path-contact conflicts miss the conflicts that matter — cut fibres stop
   short of the crossing that cut them (45 edges → 1011 with extended
   spines).
2. The mechanism DOES what it promises where pass 1 saw the fibres:
   recall 0.65→0.86, identity 0.40→0.51 (above baseline).
3. Long mask extensions rebuild the fragments-joined class (mdape 1.07 at
   90 px); the detect/mask extension split (90/20) recovers most of it.
4. Union-vs-sum mdape (0.60 vs 0.76) says the residual inflation is merge
   geometry, not duplicate arcs.
5. **The structural boundary**: the conflict graph can only separate what
   pass 1 traced. The ~35% of fibre material pass 1 missed floats
   unattributed inside every sub-image and is absorbed into whichever
   corridor covers it — that is where the residual splits/merges/mdape
   come from, and no grouping rule fixes it because the group assignment
   of unseen material is undefined.

Status of the full-replacement mode: not deployed; results in
`_runs/decompose_v1/`.

## 7g. Identity-repair mode (v7): the deployable form — tuned, frozen,
claimed

Option 2 implemented (`--mode repair`): the sparse re-traces are WITNESSES
only — a re-traced object touching ≥ 2 first-pass members for
≥ `WITNESS_PX` along each testifies they are one myotube; identities are
unioned, geometry stays pass-1, so absorbed unseen material cannot enter
any measurement. One knob swept on tune wells: WITNESS_PX 15 vs 40 — 40
dominates (net repairs 66 vs 60, merges +36 vs +47, mdape +0.002 vs
+0.011). **Frozen: witness 40 px, mask ext 20, detect ext 90, contact 6,
conflict 30°, bg pct 30.**

Tune pooled (baseline = walk+weld): splits 521→419, merges 307→343,
idx 0.440→0.485, mdape 0.313→0.315, recall unchanged.

CLAIM, test wells (one shot):

| | splits | merges | idx | mdape | recall |
|---|---|---|---|---|---|
| walk+weld | 707 | 449 | 0.417 | 0.374 | 0.641 |
| + identity repair | **593** | 493 | **0.474** | 0.396 | 0.641 |

−114 splits (−16%) for +44 merges (2.6:1); splits improved on EVERY test
well. Cumulative vs the pre-weld frozen tracer: splits 776→593 (−24%),
identity 0.379→0.474, merges 437→493, mdape 0.359→0.396. Cost: ~5-6
network inferences + walks per well (~80 s/well GPU).

## 7. Artifacts

| path | what |
|---|---|
| `model_labs/tracer_lab/eval_tracer_on_bootstrap.py` | the bridge + sealed run (committed) |
| `_runs/eval_bootstrap_v1/eval_summary.json` | per-well + pooled numbers (local, untracked) |
| `_runs/eval_bootstrap_v1/width_cap_diagnostic.json` | GT width distribution + ribbon ceiling |
| `_runs/eval_bootstrap_v1/predictions/` | exported InstanceSets per config |
