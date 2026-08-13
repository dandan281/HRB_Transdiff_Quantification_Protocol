# Claude lane — session state, developer plan & resume guide (rewritten 2026-07-28)

> **2026-07-31: two handoff docs supersede this file as the entry point.**
> Next Claude → `claude_handoff_next_session_2026-07-31.md` (has a ready-to-paste
> opening prompt). Codex/integrator → `claude_handoff_codex_2026-07-31.md`.
> This file remains the long-form history.

**Read this first if you are the next Claude.** Supersedes the 2026-07-23
version (which described the pre-junction-classifier state). Formal evidence
for each item is in the dated `coordination/reports/claude_*` files cited
inline. **Nothing this session is committed.** No plan/workboard edit was made.

Governance (unchanged): Claude lane = `annotation_tools/**`, `model_labs/**`,
`competitors/**`, `docs/**`, `coordination/reports/claude_*`. **Do NOT edit**
`PrecisionMyotube/DEVELOPMENT_PLAN.md` or `coordination/WORKBOARD.md`
(integrator-owned). `Conversion_Efficiency/**` and `cpenv` are read-only.
Single operator id is `reviewer_01`.

---

## 0. TL;DR — where we are now

- **Junction classifier: BUILT, MEASURED, SHELVED.** Went 20.5% → 64.5%
  junction-decision accuracy (2.7x the classical floor). Wired into the tracer
  it delivers **recall −0.012, F1 −0.003 — nothing** — because it only reaches
  **1.8% of pairing decisions**. Not deleted; mis-scoped, and ready if those
  junctions ever matter.
- **Fragment linker: benefit measured, COST NOT MEASURED.** At threshold **0.9**,
  pooled over 6 wells: **false splits 52 → 41 objects (−21% rel)**, **recall flat
  (+1 object of 375)**. Wired into production by Codex 2026-07-29 at 0.90.
  **Two corrections to my own earlier claims, both the same mistake — using a
  favourable number without checking its denominator:**
  1. "recall +0.015 / F1 +0.050" was mean-of-wells and outlier-driven
     (`claude_linker_per_well_correction_2026-07-29.md`);
  2. **"3 over-merges" is a CEILING, not a rate** — only **3 of 216** accepted
     merges have ≥2 reviewed reference masks and are therefore examinable at all;
     all 3 were flagged (`claude_over_merge_review_results_2026-07-30.md`).
  **Never write "3 over-merges for 11 fewer false splits"** — false splits are
  measured against the full 375-object GT, over-merges against 3 examinable
  objects. Honest headline: *11 fewer false splits measured; over-merge cost
  unmeasured; recall flat.*
- **2026-07-31 — THE TRADE CAN NO LONGER BE ASSUMED FAVOURABLE.** A blind repeat
  came back at **kappa 1.00, 0 flips of 15**: the operator is self-consistent, so
  round 1's "reviewer was guessing" reading is dead. **6 of 12 ordinary merges the
  flag did NOT raise were called two different myotubes** → ~108 over-merges
  implied in those two wells (95% CI 46–170) against **10** false splits recovered
  there. And **link confidence runs backwards: AUC 0.107, all 4 merges at
  P = 1.0000 called wrong** (Mann-Whitney p = 0.012) — so raising the threshold
  makes it worse, not better. Mechanism named by the operator: **fibres
  overlapping in z**, invisible to the linker's 2-D features.
  See `claude_over_merge_round2_results_2026-07-31.md`.
- **Labeling is saturated everywhere.** Three independent learning curves
  (junction pair model, junction gate, linker) are all flat. Do not commission
  another labeling round without re-measuring first.
- **Omnipose: still parked, case now weaker.** Its rationale was cluster/junction
  splitting; that residual is now measured as worth ~0 at instance level.
  Machine still crashing (verified 2026-07-28, see §6).
- **Tests: 276 pass** in `pm-annotate`. Baseline command in §7.

---

## 1. What this session did

1. Measured the classical floor's junction ambiguity: **8,584 raw degree-3
   nodes, but only 893 fibre-scale** (89.6% are sub-pixel skeletonisation
   whiskers); **615 ambiguous**, of which 245 formed the round-1 pool.
   (`claude_junction_ambiguity_measurement_2026-07-23.md`)
2. Built the junction labeling tool + features + page; operator labeled
   **245 junctions** (round 1) then **150** (active round 2) = 395.
   (`claude_junction_labeling_round1_build_2026-07-23.md`,
   `claude_junction_active_round2_build_2026-07-28.md`)
3. Trained the classifier; round 1 → LOWO AUC 0.701 / 41% junction accuracy.
   (`claude_junction_classifier_round1_results_2026-07-28.md`)
4. **Round 2 added 61% more labels and the model got no better.** A learning
   curve showed the plateau was not a data shortage. Root cause: `tangent_cos`
   was computed over a **3-pixel** window (the tracer's `direction_step`).
   Fixing it to 15 px moved **AUC 0.693 → 0.892, accuracy 37% → 58%** with no
   new labels. Then `node_intensity_ratio` (+0.010 AUC) and a **two-stage
   branch-point gate** (accuracy → **64.5%**).
   (`claude_junction_round2_results_and_feature_fix_2026-07-28.md`)
5. Tested and **rejected** three further levers: image-derived branch-point
   features (all hurt), GBM/RF (all worse than logistic regression end-to-end),
   the banked offset feature (raised pair AUC, lowered accuracy). Same report §8.
6. **Integration measurement — the important one.** Wired the classifier into
   `trace_fibers_parameterised` and scored instance-level metrics:
   **no improvement**, because it reaches 1.8% of decisions.
   (`claude_junction_integration_measurement_2026-07-28.md`)
7. Ran the same harness on the **fragment linker**: positive at threshold 0.9.
   Also ran its learning curve (flat) and measured train-vs-deploy distribution
   shift (AUC 0.639 — moderate).
   (`claude_linker_integration_measurement_2026-07-28.md`)

---

## 2. Code built this session (all Claude lane, uncommitted)

**`annotation_tools/annotation_tools/qc_review/`**
- `junction_pairs.py` — candidate finder (degree-3, fibre-scale gate, pool selection)
- `junction_features.py` — pair + junction features. **`DIRECTION_WINDOW_PX = 15`
  is load-bearing**; a test pins it above the tracer's `direction_step`
- `junction_page.py` — labeling UI (single-choice among 3 pairs + branch point + unsure)
- `junction_model.py` — pair classifier, branch-point gate, two-stage decisions,
  `load_decision_rows` (multi-export), LOWO helpers
- `junction_active.py` — active-learning round builder (uncertainty sampling)
- `cli.py` — 3 new subcommands: `build-junction-page`,
  `train-junction-model` (`--export` is `nargs="+"`),
  `build-junction-active-round`

**`model_labs/classical/`**
- `ridge_graph.py` — extracted `build_branch_graph` / `junction_candidates` /
  `pair_junction_ends`; added **`junction_decider` hook** to
  `trace_fibers_parameterised`. **Default `None` is bit-identical to the sealed
  floor — test-pinned.**
- `junction_ambiguity.py` — ambiguity measurement + `evaluate_junction`
- `learned_junctions.py` — decider wrapping the trained models
- `run_learned_junction_folds.py` — instance-level A/B harness (junctions)
- `run_linker_folds.py` — instance-level A/B harness (linker)

**Over-merge hand review** (2026-07-29) — see
`claude_over_merge_review_packet_2026-07-29.md`
- `model_labs/classical/extract_over_merges.py` — pulls the 3 flagged predictions
  (2 in 19_B06, 1 in 22_B03) plus a size-matched control pool; reproduces the
  benchmark's own coverage rule and **aborts unless it recovers the published
  per-well counts**. `LOCKED_THRESHOLD = 0.90`.
- `annotation_tools/annotation_tools/qc_review/over_merge_page.py` — 5-panel
  blinded review page (Desmin / fragments / proposed link / linked mask /
  reference masks); decisions `same_myotube` | `different_myotubes` |
  `ambiguous_2d`; per-case `decided_at`. `assert_no_separating_field` fails the
  build if any displayed field separates cases from controls **by value or by
  presence** — it caught two real leaks. **v2 (2026-07-30)**: a
  `different_myotubes` call requires a written reason before the case counts;
  dwell/panel-view telemetry and `reference_panel_seen_before_decision` ship in
  the export; `INSTRUMENT_VERSION` stamped so rounds cannot be mixed silently.
- `model_labs/classical/over_merge_flaggability.py` — the analysis showing only
  **3 of 216** accepted merges are examinable by the over-merge rule, i.e.
  `over_merge_count` is a ceiling.
- `cli.py` — `build-over-merge-page` (matches controls on reference density;
  `--repeat-of` pins an exact object set for a blind repeat; `--seed` selects
  controls while `--order-seed` shuffles presentation, kept separate on purpose)
  and `score-over-merge-review` (applies the pre-registered rules; `--first-pass-*`
  adds intra-rater agreement + Cohen's kappa, paired by object identity).
- Packets: `over_merge_r1/` (reviewed, inconclusive — page + key + export +
  score) and `over_merge_r2/` (**awaiting the operator**). Each key must stay
  closed until that round's export exists.

**`model_labs/tier_a_audit/audit.py`** (2026-07-29 fix)
- `build_manifest` keyed every hashed path with `p.relative_to(ROOT)`, which
  raises for any caller-chosen `out_dir` outside the repo — so
  `run_audit(<out-of-tree dir>)` crashed and 2 tests failed under an out-of-tree
  pytest `--basetemp`. Extracted `manifest_key()` with an absolute-path fallback;
  in-repo inputs keep their repo-relative keys, so **existing manifests are
  byte-identical**. 2 tests pin both halves.

**Tests** (+123 since session start): `test_junction_pairs.py` (7),
`test_junction_features.py` (22), `test_junction_page.py` (17),
`test_junction_model.py` (15), `test_junction_active.py` (8), plus 3 added to
`test_classical_ridge_graph.py`.

**Data/artifacts**: `PrecisionMyotube/annotation_work/junctions_round1/`,
`junctions_round2/` (pages, operator exports, models v1–v4),
`model_labs/classical/_runs/{junction_ambiguity_v1,learned_junctions_v1,linker_instance_v1}.json`

---

## 3. THE RECOMMENDATION — wire the linker in at threshold 0.9

> **AMENDED 2026-07-29 — read `claude_linker_per_well_correction_2026-07-29.md`
> before quoting anything in this section.** The table below is *mean-of-wells*
> and overstates the recall gain ~5x. Object-weighted (pooled) over the same six
> wells the recall gain is **+0.0027 = one matched object out of 375**, and it
> reverses sign if the single outlier well `23_B02_ctrl` is dropped. **Precision
> and F1 are invalid under sparse GT; recall is valid but too low-resolution and
> too well-dependent to support a benefit claim** — a distinction an earlier draft
> got wrong. What survives is the
> false-split reduction: **52 → 41 objects pooled (−21% rel), improving in 3
> wells, unchanged in 3, worsening in none**, against **3 new over-merges**.
> Codex's production run already used the pooled framing; the error was mine.
>
> Honest one-liner: *at 0.9 the linker trades 3 over-merges for 11 fewer false
> splits, with recall flat.* The choice of 0.9 over 0.5/0.7 is **strengthened**
> (pooled recall there is −0.093 / −0.043).

Means over 6 LOWO folds, classical floor vs floor+linker (superseded framing —
see the amendment above):

| metric | classical | linked@0.9 | pooled Δ (authoritative) |
|---|---:|---:|---|
| recall | 0.9149 | 0.9298 | +0.0027 (348→349 of 375 TP) |
| false_split_rate | 0.1476 | 0.1294 | 0.1387→0.1093 (52→41 objs) |
| over_merge_rate | 0.0000 | 0.0009 | 0→3 objects |
| n_pred | 879.8 | 634.5 | 5279→3807 |

**Production wiring was Codex's lane** (§8) and is **DONE** as of 2026-07-29
16:11: `PrecisionMyotube/runs/t02/classical_linker_v1/`,
`linked_candidate.py DEFAULT_THRESHOLD = 0.90`, `base_floor_mutated: false`.

Caveats that must travel with the number: (a) the linker was trained on
annotation-package proposal masks and applied to classical-floor instances
(measured shift AUC 0.639 — moderate); (b) over-merge goes from exactly 0 to
0.0009, a new error class; (c) precision/F1 are uninformative in both arms
because the eval GT is a sparse reviewed subset (62.5 masks vs ~880
predictions).

---

## 4. Things measured and CLOSED — do not redo

- **More junction labels** — 3 flat learning curves. Junction pair model
  saturates at N=40, gate at N=120-160, linker at ~100-120 pairs.
- **Image-derived branch-point features** — chord-minimum stain, node fatness,
  node blob area: all *hurt* the gate.
- **Non-linear models** — GBM/RF worse than logistic regression end-to-end.
- **Junction splitting as a lever on the science** — 1.8% coverage, ~0 effect.

---

## 5. Open / next, in priority order

1. ~~**Codex: wire the linker at 0.9** (§3)~~ — **DONE 2026-07-29 16:11**
   (`runs/t02/classical_linker_v1/`). Remaining on this thread: T03 write-up must
   **not** quote recall or F1 as a linker *benefit* (§3 amendment) — F1 because
   sparse GT makes it invalid, recall because +1/375 is too small and
   well-dependent, not because recall is invalid. Ask Codex to record that
   distinction in the manifest's `limitations`.
   **The 3 over-merged objects were extracted and hand-reviewed — result
   INCONCLUSIVE.** Operator called 1 of 3 flagged a real over-merge (the
   P=1.0000 chain, 22_B03 label 530), but called **6 of 12 controls**
   `different_myotubes` too, so the verdicts show no association with the flag
   (flagged 0.33 vs controls 0.50) and fail the pre-registered calibration rule.
   **DONE 2026-07-31 — round 2 scored, see the §0 bullet.** Next: a
   **control-only round of ~60 merges sampled UNIFORMLY** across all six wells
   (no density matching, no flagged cases, reasons required) to turn "~50%, wide
   CI, biased sample" into a defensible population rate; and **a second rater**,
   since two passes by one person is intra-rater reliability, not consensus.
   Historical detail below.

   **Round 2 was BUILT and is now reviewed**:
   `PrecisionMyotube/annotation_work/over_merge_r2/` — a blind repeat of the same
   15 objects, reordered, new uids, plus a fixed instrument (reason required for
   `different_myotubes`; dwell/panel telemetry shipped; reference-panel-viewed
   recorded). See `claude_over_merge_repeat_packet_2026-07-30.md` for the
   pre-registered reading rules and the scoring command (it computes Cohen's
   kappa against pass 1). **If kappa ≈ 0, both passes are void** and the next
   step is a control-only round of ~60 merges, not a third pass.
   Threshold stays locked at 0.90.
2. **Re-measure the linker after any retrain on classical-floor instances** —
   the domain shift is the main uncertainty in §3. `run_linker_folds.py` is reusable.
3. **If labeling ever resumes**, sample from the *deployment* candidate pool
   (classical-floor instances), not the banked fragment set — that is where the
   10-18% out-of-distribution mass lives. This is the only "more labels" option
   the evidence leaves open.
4. **Junction classifier stays shelved** unless fragmentation is fixed and
   instance counts stop being dominated by broken fibres.
5. Unchanged from before: Tier-A release held; Tier-A validation parked
   (no Desmin-negative control); ~38 merged linker objects still
   `status="ambiguous"` and need a completeness-review pass.

---

## 6. Laptop crash triage — STILL LIVE (verified 2026-07-28)

Read from the Windows System log this session:
- **2026-07-25 14:25** — unexpected shutdown (id 41 + 6008), **no bugcheck
  record** → hard hang or power loss. This is *after* the last resume doc.
- 2026-07-22 — the acute cluster (0x20001 ×2, 0x133), as documented
- 0x13A KERNEL_MODE_HEAP_CORRUPTION on 07-19, 07-06, 07-02, 06-26 — still ~weekly
- **0x116 VIDEO_TDR_ERROR on 06-22** — *new to this analysis*: a GPU/display
  driver timeout bugcheck, direct extra evidence for the NVIDIA-driver
  hypothesis already named as prime suspect

Minidump dir is not readable without elevation. Fix order unchanged: admin
WinDbg `!analyze -v` on `C:\Windows\Minidump\*.dmp` to *name* the driver →
clean-install NVIDIA driver → MemTest86 → BIOS/EC. **Keep GPU work off this
machine.**

---

## 7. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
# clear the stale basetemp first -- a leftover tmp/pytest_resume makes the
# tmp_path fixture raise "is not a normalized and relative path" (not a code bug)
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
# 2026-07-29: 330 passed (276 at the 07-28 session close; Codex's linker wiring
# added 6, the audit fix 2, the over-merge review packet 46). Re-derive the
# count rather than trusting a number in a doc -- it moves under you.
```

Reproduce the two headline measurements (CPU-only, ~4 min and ~10 min):
```
python model_labs/classical/run_learned_junction_folds.py --out model_labs/classical/_runs/learned_junctions_v1.json
python model_labs/classical/run_linker_folds.py --out model_labs/classical/_runs/linker_instance_v1.json
```

---

## 8. Not mine to advance

- **Codex/integrator:** linker→production wiring (now has evidence, §3); T03
  scoring of the sealed floor; canonical-plan reconciliation (ring 33% vs
  traced-fibre 6.62%); whether DEVELOPMENT_PLAN should record the junction
  classifier as shelved.
- **Operator/wet-lab:** a Desmin-negative control well (unblocks Tier-A
  validation); the completeness-review pass for the ~38 merged linker objects.
- **Omnipose** — parked; resume trigger is stable compute **and** a decision to
  still want a learned pixel candidate. The junction evidence weakens, not
  strengthens, that case.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence — not consensus, not inter-rater agreement,
not prospective validation.
