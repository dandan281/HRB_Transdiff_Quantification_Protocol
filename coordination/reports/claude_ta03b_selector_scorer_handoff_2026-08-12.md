# TA03b handoff — stratified selector and 2-D-versus-3-D scorer

Date: 2026-08-12
Lane: Claude (model laboratories)
Authority: `coordination/requests/claude/2026-07-23-tier-a-selector-and-scorer.md`;
`codex_tier_a_validation_ratification_2026-07-23.md` as amended
2026-08-12 by `codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md` §4.

**Tests: 483 passed** (429 before, 54 added). No acquisition, no relabelling, no threshold
tuning, no GPU job, no commit, and nothing written under `Conversion_Efficiency/`.

---

## 1. What was built

Two modules and one test file.

**`model_labs/tier_a_audit/selector.py`** — enumerates the sampling frame from the accepted
audit's ring intensities and draws a deterministic stratified sample.

**`model_labs/tier_a_audit/scorer.py`** — built against the **amended** contract: whole-field
or registered-mosaic 3-D reacquisition, DAPI-based registration, prespecified one-to-one
post-hoc matching. Not the superseded per-nucleus stage-targeting design.

**`model_labs/tests/test_tier_a_selector_scorer.py`** — 54 tests covering the nine groups the
request enumerates, on synthetic data only. Nothing in the suite reads the plate; a test that
needed the real masks could not exercise the fail-closed paths at all.

## 2. Decisions that were choices, and why

**Ring intensity is imported from `audit.py`, never reimplemented.** That function is
transcribed verbatim from the production package and is why the audit reproduces exactly. A
second copy would be a second method, and the first time they drifted the selector would be
sampling from a population the audit never measured.

**`relocalization_feasible` stays `False`.** Per the ruling, raw ND2 field-centre XYZ does not
flip it: the pixel-to-stage affine is absent and frame/event metadata conflict. The selector
exports the flag with a recorded reason rather than emitting centroids that look actionable.

**Desmin cannot touch the match.** `match_nuclei` raises if any candidate record carries a
Desmin field, and `score` raises unless the transform record declares a nucleus-channel basis.
This is enforced rather than documented because it is the one route by which the harness could
manufacture agreement — letting the quantity under test select its own reference.

**A ratio of exactly 1.0 is a positive call.** Strata are half-open upward, so 1.0 lands in
`1.0_1.25` rather than `0.8_1.0`. Parametrised at every one of the six edges.

**Nucleus-level binomial intervals are refused, not offered.** `field_cluster_bootstrap` raises
if given fewer than two fields rather than silently degrading to an object-level interval.

**Attrition is retained and counted, never replaced.** Eight prespecified reasons. A selected
nucleus that vanished silently would bias the weighted estimate by exactly the amount that made
it hard to match.

**Per-cell RNG seeding.** Each (well, field, stratum) cell draws from a generator seeded on
`sha256(seed|well|field|stratum)`, not from one shared stream, so changing one stratum's target
or adding a well cannot reshuffle the others. Pinned by test.

**Missing evidence fails a gate, never passes it.** With no negative-control material the
fourth gate reports `not_evaluable` and `passed=False`.

## 3. Commands

```bash
# full suite (basetemp only needed where the default temp dir is not writable)
python -m pytest model_labs/tests annotation_tools/tests PrecisionMyotube -q

# this task only
python -m pytest model_labs/tests/test_tier_a_selector_scorer.py -q
```

Run in `pm-annotate`. Neither module needs a GPU; neither imports Omnipose.

## 4. Worked example

Synthetic, 4 fields x 40 nuclei, boundary stratum oversampled at p=0.1 against p=0.4,
2 attrition rows, 1000 bootstrap draws:

```
Tier-A 2-D-versus-3-D validation - weighted, field-clustered
==============================================================
matched 160 of 162 selected (match rate 0.988)

  sensitivity      0.9742   95% CI [0.9484, 1.0000]
  specificity      0.9857   95% CI [0.9571, 1.0000]
  ppv              0.9855   95% CI [0.9584, 1.0000]
  npv              0.9745   95% CI [0.9509, 1.0000]
  fp_inflation     0.0145   95% CI [0.0000, 0.0416]

adverse-bound gates (ratified 2026-07-23, not tunable):
  [PASS] specificity            lower bound vs 0.95
  [PASS] sensitivity            lower bound vs 0.9
  [PASS] fp_inflation           upper bound vs 0.1
  [FAIL] negative_control_fpr   upper bound vs 0.05  (no negative-control material supplied)

co-primary gates all passed: True
all gates passed:            False
```

The last two lines are the behaviour to note: all three co-primary gates pass and the overall
verdict is still `False`, because the negative control has not been supplied. That is the
intended reading, not a bug — an absolute conversion percentage requires **all four**.

## 5. Limitations

- **Neither module has been run on real data**, because there is none. The selector can run
  against the plate today; the scorer cannot, since no 3-D reference exists. Both are
  exercised only on synthetic fixtures.
- **The scorer's inputs are a contract, not an implementation.** Registration, transform
  estimation and candidate generation are upstream of it and unbuilt — it consumes a
  `transform_record` and a candidate list and verifies their properties. Whoever builds the
  registration step must produce those.
- **The mask-overlap criterion assumes a projected 3-D mask** is available per reference
  nucleus. If the acquisition yields only centroids, `MatchRule.min_mask_overlap` cannot be
  evaluated and the rule needs re-specifying before use.
- **`population_mixture` is required and unvalidated against reality.** It must be
  preregistered. The harness checks it sums to 1.0 and names known strata; it cannot check
  that it describes the target population.
- **Field-cluster bootstrap with few fields is wide and honest about it.** With one field per
  well in current data, a real run would have six clusters; percentile intervals at that
  cluster count are coarse. ~~The simulation subcommand the request permits is not built.~~
  **Built 2026-08-12** as `model_labs/tier_a_audit/planning.py` (+13 tests), and it found
  that six fields is *not* enough: at ICC 0.10 only 86% of replicates clear the ratified
  10-point half-width, p90 = 0.117. Twelve fields clears it every time. See
  `claude_ta03c_acquisition_brief_2026-08-12.md`, which is the operator-facing version and
  the thing that needs to reach the microscope before 2026-08-19.
- Single plate, single operator, proposal-conditioned. Unchanged.

## 6. Changed paths

| path | status |
|---|---|
| `model_labs/tier_a_audit/selector.py` | new |
| `model_labs/tier_a_audit/scorer.py` | new |
| `model_labs/tests/test_tier_a_selector_scorer.py` | new, 54 tests |
| `coordination/reports/claude_ambiguous_pool_characterisation_2026-08-11.md` | corrected: B06 contributes **119**, not 120, authoritative masks — the operator certified 120 and one is removed by the binding exclusion. The 0.500 certification rate is unaffected. Ruling outcome recorded. |
| `coordination/reports/claude_session_state_2026-08-06.md` | Stage 1 results, trap 5, fine-tuning verification, corrected workboard note |
| `coordination/requests/codex/2026-08-11-…md` | the request this closes out |

Nothing committed; HEAD remains `0322ebf`.

## 7. What I would do next, in order

1. ~~The simulation subcommand.~~ **Done 2026-08-12** — `planning.py`, 13 tests, and the
   operator brief in `claude_ta03c_acquisition_brief_2026-08-12.md`. Headline: acquire **≥2
   fields per well**, not 1, and note that grading more nuclei per field barely helps because
   effective sample size per field asymptotes at `1/ICC`.
2. **Reason codes on `ambiguous`** in the QC review tool, before any further review happens.
   Recommendation 2 of the ambiguous-pool report and still open. This is the cheapest
   unclaimed win in the project: one dropdown makes the largest untapped data pool (839
   objects) diagnosable instead of inference-only.
3. **Stage 2** — currently **cannot complete**. Preemption on `ckpt-g2` arrives faster than a
   fold trains (longest window 37 min against a ~120 min fold, with no mid-fold checkpoint),
   so the array requeues forever and writes no sidecar. Open decision: build checkpoint/resume
   or pay for uninterrupted GPU. Detail in `claude_session_state_2026-08-06.md` §4 and in
   memory as `klone-stage2-failure-modes`.

The real-data run in §4 above also exercised the selector end to end for the first time: its
sampling frame totals **51,869 nuclei, exactly matching the accepted audit's `pooled_cells`**,
which is the check that it samples the population the audit measured rather than a near-miss.
Per-stratum yields are in the acquisition brief; even the thinnest well holds 214 and 150
nuclei in the two boundary strata, so supply is not the constraint — fields are.
