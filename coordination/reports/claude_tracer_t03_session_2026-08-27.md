# T04 tracer lane — sealed T03 run and the work it opened
### 2026-08-27 → 2026-09-02 · all local, $0 cluster spend

Numbers here are measurements, not rulings; Codex owns the T03 ruling
(§7h). Section anchors §1–§7k are stable — memory notes and Codex's ruling
document cite them — so later work was appended under new letters rather
than renumbering.

## 0. State of the lane (read this first)

**Product status.** The tracer measures a plate at parity with the operator
and ranks wells reliably: PLATE_32 total-length ratio **1.00×**, length
rank **ρ +0.90**, count rank **ρ +0.97**, all surviving drop-one-well
(§7k). Every well is scored by a network that never saw it.

**The deployed stack** is walk → junction weld → decompose-retrace identity
repair. Against the pre-weld tracer on never-tuned test wells: **fibres cut
into pieces −24%** (776 → 593) and **identity through crossings 0.379 →
0.474**, at a cost of +56 merges and +0.037 per-fibre length error
(§7b, §7g).

**The sealed benchmark** beat two of three predeclared floors by wide
margins (length error 0.086 vs 0.317; false splits 6 vs 52) and failed the
third (recall 0.557 vs 0.928). Codex ruled candidate #2 not promoted;
standing T03 candidate: **none** (§3, §7h).

**Three questions answered by measurement this week:**

1. *Is the tracer over-tracing on new plates?* **No** — the operator's own
   two blind passes over one window overlap on only ~⅓ of their length, and
   98.5% of tracer length sits on real fibre signal. The apparent 2.5×
   excess was annotation selectivity; the honest excess is ~1.3× (§7j).
2. *Was there a pixel-calibration error?* **No** — the operator's Fiji CSVs
   were always right; raw arc-length re-measurement of freehand traces
   inflates 10–15% with drawing jitter (§7d).
3. *Is Omnipose a viable alternative?* **No** — it reports more objects
   than the operator but half the length, with 84% of them in the shortest
   class (median 86 µm vs 210 µm), and its well ranking collapses under
   drop-one-well (§7k).

**Open front:** coverage. The T03 recall gap and the dim-stretch field
holes are the same problem, and it is not an identity problem (§4, §7a).

### Contents

| § | subject |
|---|---|
| 1–6 | the sealed T03 run: predeclarations, leakage, numbers, recall attribution, caveats |
| 7a | fragmentation diagnosis (census + per-pixel attribution) |
| 7b | the junction weld — tuned, frozen, claimed |
| 7c | never-seen plates PLATE_26 / PLATE_28 |
| 7d | the freeline length convention (the "calibration" non-bug) |
| 7e | convention restate + the length-class distribution |
| 7f | decompose-and-retrace, full-replacement mode |
| 7g | identity-repair mode — the deployed form |
| 7h | Codex's ruling on candidate #2 |
| 7i | the pass-2 loop |
| 7j | the B04 blind re-trace adjudication |
| 7k | the Omnipose benchmark row |
| 7l | length-class quantification hardened + tested |
| 7m | `quantify_new_plate.py` — the product entry point |
| 8 | artifacts |
| 9 | open items and how to reproduce |

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
scale collapses to **0.6568 ± 0.0011 across all eight wells** — one
constant, ≈ the nd2 metadata 0.6493 (residual ~1.2% = our smoother vs
ImageJ's exact freehand algorithm). **The operator's CSVs were correct all
along.**

(First reported as 0.657 ± 0.003; re-measured 2026-09-02 with the
corrected symmetric-window smoother in `length_classes.py` — see §7l. The
headline is unchanged and the between-well spread tightened.)

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

## 7f. Decompose-and-retrace (operator's sketch, 2026-08-28)

Six iterations on the tune well: promising, not net-positive as a
full replacement.

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

## 7g. Identity-repair mode (v7) — the deployed form: tuned, frozen, claimed

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

## 7h. Codex ruling on candidate #2 (2026-09-01)

Codex authorized and executed the one-shot candidate-2 run
(`--candidate 2`: walk + frozen weld + frozen identity repair) and ruled
(`coordination/reports/codex_t03_candidate2_ruling_2026-09-01.md`):

| metric | cand 2 | cand 1 | floor |
|---|---|---|---|
| length_mdape | 0.0864 | 0.0864 | 0.3169 |
| false_split_count | 4 | 6 | 52 |
| pooled recall | 0.5493 | 0.5573 | 0.928 |

Ruling: passes length/split floors, fails recall; every cand-1-vs-2
whole-well bootstrap interval includes zero; with the 2-of-2 multiplicity
accounted, **candidate 2 is not promoted; standing T03 candidate: none;
neither authorized for automatic measurement.** Codex disclosed the
concurrent (unrelated) `decompose_retrace.py` loop-mode edits and
confirmed the repair function and results were unaffected.

Interpretation, for the record and not as re-litigation: the weld+repair
stack targets crossing cuts, and the bootstrap's certified fibres barely
cross (predicted-crossing contact on their spines 0.4–1%, vs 11.7% on the
dense PLATE_32 well where the stack was validated). T03 was structurally
insensitive to this mechanism; its measured value lives on dense plates
(PLATE_32 test wells: splits −24%, identity +0.095). The T03 recall gap
(0.55 vs 0.93) remains the lane's real open front and is a coverage
problem, not an identity problem.

## 7i. The pass-2 loop (2026-09-01) — built and measured, not the default

`--mode loop` in `decompose_retrace.py`: repair → residual harvest (blank
all claimed corridors +6 px — the rim-sliver guard; un-dilated, B02 ground
>40 min in never-claiming sub-min walks — trace what remains) → fold new
objects in → second identity repair across old and new. All constants
inherited frozen; nothing swept.

Tune pooled (base = walk+weld): splits 521→462, merges 307→355, recall
0.650→0.684, idx 0.440→0.501, mdape 0.313→0.344.

Test-well claim (one shot): splits 707→703 (FLAT — worse on the densest
wells D04/D11), merges 449→514, **recall 0.641→0.690, idx 0.417→0.485**,
mdape 0.374→0.428.

Verdict: the loop's coverage and identity gains generalize (+0.05 recall,
+0.07 identity); its split reduction does NOT, and it costs +65 merges and
+0.054 mdape. **Repair-only stays the deployed operating point; the loop
is an available coverage-oriented mode**, to be preferred only where
recall matters more than merge/length fidelity (e.g., feeding the
length-class distribution on sparse-annotated plates). Runtime ~2.5-9
min/well.

Also this date: `length_classes.py` — the operator's metric (share of
myotubes per 50-150/150-300/300-500/500-800/>800 µm class) is now a
standard per-well output of `cv_report.py`, `quantify_plate.py`, and
`eval_unseen_plates.py`, alongside the freeline smoothing convention
helpers.

## 7j. B04 blind re-trace (2026-09-01): the 1.78× question ADJUDICATED

The operator blind-retraced the predeclared center 1200² window of
PLATE_28 B04 (`coordination/retrace_check_p28b04/`, 52 fibres). Frame
verified (transform test: identity wins, on-bright 0.815 vs ~0.30 for
every flip/transpose). In-window, smoothed convention, ≥50 µm:

| source | n | mm | on-bright (bg rate 0.302) |
|---|---|---|---|
| original ROIs | 30 | 5.1 | 0.939 |
| fresh blind re-trace | 52 | 6.4 | 0.815 |
| union of both passes | — | 11.5 | — |
| tracer (walk+weld) | 57 | 12.8 | **0.985** |

Findings:

1. **The annotation is strongly selective**: the two passes overlap on only
   ~1/3 of their length — each pass samples a DIFFERENT subset — and their
   union is 1.9× the original alone. Single-pass totals on this plate
   under-count by roughly half.
2. **The tracer's surplus is largely real fibre**: 98.5% of tracer length
   sits on bright signal (vs 30% chance) — cleaner than either human pass
   — and 75% of it lies on the union of the two passes. Tracer/union
   ratio: **1.27** (was 2.5× vs the original alone).
3. Residual ~25% of tracer length is on-signal but outside both passes —
   most plausibly fibre neither pass traced (the union would likely keep
   growing with a third pass), but not certified.
4. **The human ceiling is plate-dependent**: point-level agreement between
   the operator's own passes here (~0.28–0.36 at 6 px) is far below the
   D04 window's trace-level 0.71 — the sparse-annotation practice on
   these plates is a sampling, not a census. Comparisons of the tracer
   against single-pass ROIs on P26/P28 must be read accordingly.

Ruling for the standing question: on the never-seen plates, the tracer's
extra length is predominantly REAL — annotation selectivity, not
over-tracing. The honest tracer-vs-truth excess is ~1.3× against a
two-pass union, with the remainder unadjudicated.

## 7k. Omnipose benchmark row (2026-09-02) — checkpoint rescued, plate-32 comparison done

The fine-tuned Omnipose checkpoint was copied off the purge-scheduled
`/gpfs/scrubbed` to `/gpfs/home/danlovuw/rescued_checkpoints/` and then to
`model_labs/omnipose/checkpoints/` — sha256 `5250ee87…` verified at all
three locations. (One fix needed for local inference:
`rescale=False` is read as 0.0 and resizes the field to nothing —
`rescale=None` + the `eval_on_bootstrap` keyword set, so the numbers are
commensurable with the T03 Omnipose path.) 10 wells, ~16 min total,
`model_labs/omnipose_lab/_runs/plate32_omnipose.json`.

All three measured against the same operator column (smoothed convention):

| metric | tracer (walk+weld) | Omnipose |
|---|---|---|
| total length ratio (plate) | **1.004** | 0.513 |
| length Pearson r | **0.806** | 0.512 |
| length Spearman ρ | **+0.903** | +0.297 |
| count ratio (plate) | 1.116 | 1.181 |
| count Spearman ρ | **+0.967** | +0.888 |
| drop-one-well length ρ | +0.867…+0.967 | **+0.033…+0.567** |

Length mix (pooled shares ≥50 µm; figure
`_runs/three_way_length_classes.png`):

| | 50-150 | 150-300 | 300-500 | 500-800 | >800 | median |
|---|---|---|---|---|---|---|
| operator | 29.4% | 44.5% | 21.2% | 4.7% | 0.2% | 210 µm |
| tracer | 50.7% | 29.1% | 13.1% | 5.0% | 2.1% | 148 µm |
| Omnipose | **84.4%** | 14.1% | 1.5% | 0.0% | 0.0% | **86 µm** |

Reading: Omnipose finds MORE objects than the operator (6137 vs 5196) but
half the length (632 vs 1231 mm) and essentially nothing above 300 µm — it
fragments myotubes into short pieces, the one-label-per-pixel failure this
lane was created to escape, now quantified on the operator's own metric.
Its per-well length ranking does not survive drop-one-well (ρ falls to
+0.03 without B02), so it cannot rank wells. The tracer is the better
candidate on every axis. Omnipose stays benchmark-only (user decision); no
further development.

## 7l. Length-class quantification hardened (2026-09-02)

The operator's metric (share of myotubes per length band) was wired into
the plate scripts on 2026-09-01 but only exercised end-to-end by
`quantify_plate_omnipose.py`. Completing it surfaced two defects:

1. **The human column used a different convention from the report.**
   `cv_report.py` and `quantify_plate.py` measured operator traces by raw
   arc while every figure in this report uses smoothed (§7d), so the
   standard output would have disagreed with the published numbers by
   10-15% on the human side. Both now emit BOTH, explicitly keyed
   (`human_length_classes` raw, `human_length_classes_smoothed`, plus
   `human_mm_smoothed`). Adopting one convention lane-wide remains the
   open decision in §9 — the script does not make it quietly.
2. **`smooth_polyline` bent the ends of every trace.** Edge-padding plus a
   fixed window replicates the first point, so on a straight line the
   second point moved 1 -> 1.2 (w=5), putting a small kink at both ends.
   Replaced with a symmetric window that shrinks near the boundaries,
   which reproduces straight lines exactly and leaves endpoints untouched
   by construction. Caught by a new contract test, not by inspection.

Re-measured after the fix: the §7d fitted scale is **0.6568 ± 0.0011**
across the eight P26/P28 wells (first reported 0.657 ± 0.003) — the
finding is unchanged at reported precision and the between-well spread
tightened. No other reported number depends on the smoother at the
precision quoted.

`model_labs/tests/test_length_classes.py` (8 tests) now pins: bin edges
and the right-open boundary rule, shares summing to 1 *within their 4-dp
rounding* (the first version of this assertion passed only because its
inputs rounded exactly — worst real deviation is 1e-4), the <50 µm gate,
empty-well safety (no NaN), endpoint/straight-line invariance of the
smoother, short-polyline pass-through, µm scaling, and the §7d
raw-inflates-freehand finding itself. Suite: **26/26 green**.

**End-to-end verification** (all four consumers executed after the edits,
not merely compiled): `eval_unseen_plates.py` (8 wells),
`cv_report.py` (10 wells), `quantify_plate_omnipose.py` (10 wells) and
`quantify_plate.py` share the same helper; every well record now carries
`*_length_classes`, and the CV table also carries both human conventions.
Cross-check: the plate human total is 1325.2 mm raw vs **1231.3 mm
smoothed (raw inflates 7.6%)**, the smoothed figure matching §7e's restate
exactly. The convention is not cosmetic for this metric — on B02 the
operator's own 50–150 µm share moves 32.8% → 36.2% between conventions,
which is why both are emitted rather than one chosen quietly.

## 7m. The product entry point: `quantify_new_plate.py` (2026-09-02)

Every earlier runner had its dataset baked in (PLATE_32 corpus paths, an
explicit P26/P28 well list, a fixed pixel size and channel index). For real
experimental use there was no "point at a plate and go" command. Now there
is: a folder of `.nd2` files in, per-well counts / totals / length-class
mix out (`wells.csv`, `summary.json`, `length_classes.png`), operator
comparison only if ROI zips exist. Two stages because the nd2 reader and
torch live in different envs (`--extract` in `base`, tracing in
`pm-omnipose`).

Acquisition handling from metadata, not constants:

- **channel** via the project's `resolve_roles` (ch1 prior, morphology
  otherwise; `--fiber-ch` override), with the CHOICE and its basis
  recorded per well. Found while testing: a **saturated channel scores
  zero on every morphology feature** and silently derails the role
  assignment — PLATE_44 B07's DAPI (ch0, p97.5 = 4095) was assigned to
  ch2, and the fibre channel was decided by the ch1 prior alone (right
  answer, fragile reason; confirmed by eye). The runner now flags
  saturated channels and records `fiber_choice` = prior / morphology /
  override.
- **pixel size** read from the nd2; a plate off the training scale
  (0.650 µm) by > 2% is resampled to it before inference and lengths
  reported in µm. Mechanically correct and **unvalidated** — flagged
  loudly per well and in the summary; PLATE_44 (1.7246 µm, ×2.65) is the
  live example. Per the standing rule, such a plate needs its own sweep
  before its numbers are trusted.

Visuals: per well, `<well>_overlay.png` — the whole field at half
resolution with every traced object in its own colour (a colour change
along one fibre is a cut, an uncoloured fibre is a miss), and the
operator's ROIs as a left panel in the same style where they exist. This
is the picture that decides whether a well's numbers are believed;
`--no-overlay` skips it. Plate-level: `length_classes.png` (operator vs
tracer when ROIs exist, tracer alone otherwise). Caught while wiring it:
a loop variable named `path` shadowed the output-path argument, so the
first version handed `imwrite` a polyline — fixed, both branches
re-verified on real wells.

Verification: PLATE_26 end to end (3 wells, repair mode; human recall
0.85/0.88/0.88 identical to `eval_unseen_plates`); strict reproduction on
B02 in weld mode — smoothed total **44.4 mm = the report's 44.4**, raw arc
44.7 vs 44.9 / 125 vs 126 objects (the runner measures 1 px-resampled
paths, as the length-distribution report did). PLATE_44 B07 exercised the
rescaled path end to end. 7 contract tests
(`test_quantify_new_plate.py`): well-token discovery, collision refusal,
ROI matching, the rescale trigger.

## 8. Artifacts

Code (committed; `model_labs/tracer_lab/` unless noted):

| path | what |
|---|---|
| `oracle_trace.py` | the walk; `weld_objects` = the junction weld (§7b) |
| `decompose_retrace.py` | decomposition: `--mode full` / `repair` (deployed) / `loop`; `apply_repair` is the reusable entry point (§7f, §7g, §7i) |
| `eval_tracer_on_bootstrap.py` | trace→mask bridge + sealed T03 run; `--candidate 1\|2` (§1–3, §7h) |
| `weld_rescue_sweep.py` | weld/rescue tune sweep + test claim (§7b) |
| `frag_census.py`, `frag_attribution.py` | fragmentation diagnostics (§7a) |
| `eval_unseen_plates.py` | PLATE_26/28 vs operator ROIs (§7c) |
| `length_distribution_report.py` | convention restate + length-class shares (§7e) |
| `length_classes.py` | the length-class + freeline-smoothing convention, shared (§7d, §7i) |
| `cv_report.py`, `quantify_plate.py` | plate tables; both now emit `*_length_classes` |
| `quantify_new_plate.py` | ANY plate: nd2 folder -> per-well counts, totals, length mix; `--extract` then trace (§7m) |
| `model_labs/omnipose_lab/quantify_plate_omnipose.py` | the Omnipose benchmark row (§7k) |
| `model_labs/tests/test_oracle_trace.py` | 3 weld contract tests (§7b) |
| `model_labs/tests/test_length_classes.py` | 8 length-class + smoothing contract tests (§7l); suite 26/26 green |

Results (local, `_runs/` is gitignored):

| path | what |
|---|---|
| `_runs/eval_bootstrap_v1/` | candidate #1 sealed run: `eval_summary.json`, `width_cap_diagnostic.json`, `frag_diag.json`, `unwalked_attribution.json`, `break_attribution.json`, exported InstanceSets |
| `_runs/eval_bootstrap_candidate2/eval_summary.json` | candidate #2 sealed run (Codex-executed) |
| `_runs/weld_rescue_sweep.json`, `_runs/weld_rescue_claim.json` | weld tune + claim |
| `_runs/decompose_v1/results.json` | decomposition modes, all wells |
| `_runs/length_distribution_report.json` | both conventions, both plate sets |
| `_runs/three_way_plate32.json` | human vs tracer vs Omnipose |
| `model_labs/omnipose_lab/_runs/plate32_omnipose.json` | Omnipose per-well |
| `_runs/*.png` | overlays and figures (weld before/after, unseen plates, length proportions, three-way, B04 re-trace) |

Data and coordination:

| path | what |
|---|---|
| `model_labs/omnipose/checkpoints/v1-fold-B02-paint_out_epoch_299` | rescued Omnipose checkpoint, sha256 `5250ee87…` (also `/gpfs/home/danlovuw/rescued_checkpoints/`) |
| `coordination/retrace_check/` | D04 human ceiling (2026-08-25) |
| `coordination/retrace_check_p28b04/` | B04 blind re-trace + adjudication (§7j) |
| `coordination/CODEX_PROMPT_T03_CANDIDATE2_2026-09-01.md` | the candidate-2 submission prompt |
| `coordination/reports/codex_t03_candidate2_ruling_2026-09-01.md` | Codex's ruling (§7h) |

## 9. Open items and reproduction

**Decisions resting with the operator**

1. **Steeper weld point** — dist 26 removes ~23% more cuts on tune wells for
   mdape +0.036; frozen at dist 14 by the predeclared guard (§7b).
2. **Smoothed convention lane-wide** — adopting it is a metric redefinition
   and would require restating prior tables once, under both conventions
   (§7d, §7e). The restate itself is already computed.
3. **Candidate #3 to Codex?** Recall is T03's only failing metric and the
   pass-2 loop is the only mechanism that moves it (+0.05); expected landing
   ~0.60 against a 0.928 floor, so the submission may not be worth its
   multiplicity cost (§7h, §7i).

**Technical front (needs an explicit go)**

4. **The history-conditioned stepping head** — the one untried mechanism
   from the original plan, and the only one aimed at walking where the
   centre field goes dark, which is 46–78% of missing coverage (§7a). Gates
   must be pre-declared on tune wells before any training.
5. **A third blind pass** on the B04 window would certify the ~25% of
   tracer length that is on-signal but outside both existing passes (§7j).

**Reproduction** (GPU env `pm-omnipose`; CPU work in `pm-annotate`):

    # a NEW plate, start to finish (the product path)
    conda run -n base python model_labs/tracer_lab/quantify_new_plate.py --plate <folder of .nd2> --extract
    conda run -n pm-omnipose python model_labs/tracer_lab/quantify_new_plate.py --plate <folder of .nd2>

    # deployed stack, any PLATE_32 well
    python model_labs/tracer_lab/decompose_retrace.py --wells C05 --mode repair

    # plate tables (now including length-class shares)
    python model_labs/tracer_lab/cv_report.py
    python model_labs/tracer_lab/length_distribution_report.py

    # never-seen plates (two stages: extract needs an env with `nd2`)
    conda run -n base python model_labs/tracer_lab/eval_unseen_plates.py --extract
    python model_labs/tracer_lab/eval_unseen_plates.py

    # Omnipose benchmark row
    python model_labs/omnipose_lab/quantify_plate_omnipose.py \
        --checkpoint model_labs/omnipose/checkpoints/v1-fold-B02-paint_out_epoch_299

    # tests
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
        model_labs/tests/test_oracle_trace.py model_labs/tests/test_tracer_targets.py -q

**Traps paid for in this arc** (beyond those in the 2026-08-23 report):

- `rescale=False` in a `cellpose_omni` eval call is read as 0.0 and resizes
  the field to nothing — use `rescale=None` (§7k).
- Blanking a corridor without dilating it leaves bright halo rims that
  spawn endless never-claiming walks (B02: >40 min, zero output) (§7i).
- Endpoint extensions used for *conflict detection* must be long (90 px) and
  those used for *masking* short (20 px); one constant for both rebuilds the
  fragments-joined error class (§7f).
- `conda run` cannot take multi-line `python -c`; write a file instead.
