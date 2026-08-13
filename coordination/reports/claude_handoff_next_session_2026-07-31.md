# Handoff → next Claude session, 2026-07-31

Paste the block in §6 as your opening prompt. Everything above is the reasoning behind
it.

---

## 1. Where things stand in one paragraph

The fragment linker was wired into production at threshold 0.90 by Codex on 2026-07-29
on the strength of a false-split reduction I measured. Since then **two corrections
and one new finding have moved the evidence against it**, and the current position is:
the benefit is real and measured (**52 → 41 false-split objects pooled, −21% rel**),
the cost is real and **unmeasured**, and a blinded two-pass human review
(**kappa 1.00**) suggests the cost may be ~10× the benefit. The single measurement
that would settle it is a control-only review round, which is buildable now and needs
the operator to run. Threshold 0.90 stays locked and must not be tuned on any of this.

## 2. Read in this order

1. `claude_over_merge_round2_results_2026-07-31.md` — the current position, kappa 1.00,
   the confidence inversion, the z-overlap mechanism.
2. `claude_over_merge_review_results_2026-07-30.md` — why `over_merge_count = 3` is a
   ceiling (3 of 216 merges examinable).
3. `claude_linker_per_well_correction_2026-07-29.md` — the pooled-vs-mean correction,
   and the answers to Codex's five checks (a)–(e).
4. `claude_handoff_codex_2026-07-31.md` — what the other lane has been asked to do.
5. `claude_resume_state.md` §0 for the compressed state.

## 3. Done, so you do not redo it

- **Junction classifier**: built, 2.7× the classical floor, **shelved** — reaches 1.8%
  of pairing decisions, ~0 instance-level effect. Do not resume.
- **Labeling saturation**: four flat learning curves (junction pair model, gate,
  linker, and the re-run at the v4 feature set). More labels *of the same kind* buy
  nothing. Do not commission a labeling round without a new distribution.
- **Linker instance-level A/B**: done, corrected twice. Pooled, not mean-of-wells.
- **Over-merge extraction + flaggability**: `model_labs/classical/extract_over_merges.py`,
  `over_merge_flaggability.py`. The 3 flagged objects are named and reproducible.
- **Two blinded review rounds**: `over_merge_r1/` and `over_merge_r2/` — page, key,
  export and score in each. Round 2 is a true blind repeat (`--repeat-of`).
- **Instrument v2**: reason required on `different_myotubes`; dwell/panel telemetry;
  `reference_panel_seen_before_decision`. It earned its keep immediately — the
  0-of-15 reference-panel finding was not recoverable from round 1's timestamps.
- **Tier-A audit bug**: `manifest_key()` fallback; `run_audit()` no longer crashes on
  an out-of-tree `out_dir`.
- Tests **276 → 373**.

## 4. Waiting, in priority order

**1. Control-only round, ~60 merges sampled uniformly (top priority).**
The production decision hangs on it. Current estimate is 6/12 = 50% (95% CI 21–79%)
from a **density-matched, non-uniform** sample — good enough to raise the alarm, not
good enough to act on. Needs:
- a uniform sampler in `extract_over_merges.py` (today `--controls` samples only
  components whose fragment count matches a flagged case, in the two wells that
  contain flagged cases — you want all six wells, no matching, no flagged cases);
- `build-over-merge-page` already handles a controls-only packet, but check the
  blinding guard: with no flagged cases `assert_no_separating_field` short-circuits,
  which is correct but means the packet is unblinded-by-construction — decide whether
  that matters when every object is the same class (I think it does not, but say so
  explicitly in the report);
- reasons required is already enforced.
You can build it all; **only the operator can run it.**

**2. Probe whether z-overlap is detectable from 2-D image features.**
This is the modelling question the mechanism raises, and it is CPU-only and needs no
operator. You have 15 labelled merges (7 called different, 8 same) — thin, but enough
for a first look, and the round in §4.1 would take it to ~75. Candidate signals: local
fibre density around the bridge, a second ridge crossing the bridge at an angle,
width discontinuity across the join, stain *intensity* step (an overlap sums two
fibres, so the overlap region should be brighter than either). **The last one is the
most promising and the cheapest to test.** If a feature separates them, the linker can
be gated rather than abandoned; if nothing does, that is the strongest argument yet
for z-information and should be recorded as such.

**3. A second rater.** Everything so far is one operator. Two passes by one person is
intra-rater reliability. Needs a human; not yours to schedule, but flag it.

**4. Older, still open:**
- ~38 merged linker objects still `status="ambiguous"` — need a completeness-review pass.
- Tier-A release held; Tier-A validation parked (no Desmin-negative control well).
- Canonical-plan reconciliation (ring 33% vs traced-fibre 6.62%) — integrator's.

## 5. Rules that keep biting

- **Check the denominator before quoting any favourable number.** I got this wrong
  twice: mean-of-wells recall (a +1-object effect looked like +0.015), and
  `over_merge_count = 3` (a saturated detector looked like a small cost). Before
  quoting a *count* of detected errors, ask **how many objects the detector could even
  examine**.
- **Blinding is about rendering, not labels.** Three leaks got past a "no forbidden
  key" check; two were rendering differences. `assert_no_separating_field` now guards
  by value *and* by presence.
- **Pre-registered rules are revisable, but say so loudly and justify it in a way that
  is outcome-independent.** I revised one on 2026-07-30; the justification (a direct
  measurement of reviewer noise supersedes a proxy for it) would have applied
  identically had kappa come out 0, and there is a test pinning the losing branch.
- The measurement that matters is usually the cheapest one, and I keep running it last.

## 6. Opening prompt for the next session

```
Resuming PrecisionMyotube (HRB_Transdiff). Previous session closed 2026-07-31.

READ FIRST: coordination/reports/claude_handoff_next_session_2026-07-31.md
Then: claude_over_merge_round2_results_2026-07-31.md (the current position)
      claude_handoff_codex_2026-07-31.md (what the other lane owes)

Confirm the baseline (expect 373; re-derive rather than trust it — both lanes add tests):
  $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
  Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
  & "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
    PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume

STATE: the fragment linker is live at threshold 0.90 (Codex, 2026-07-29). Its benefit
is measured (false splits 52 -> 41 objects pooled, -21% rel). Its COST is not: the
over-merge detector can examine only 3 of 216 accepted merges, and a blinded two-pass
human review (kappa 1.00, 0 flips of 15) called 6 of 12 UNFLAGGED merges "two different
myotubes" -> ~108 implied over-merges in two wells vs 10 false splits recovered there.
Link confidence runs BACKWARDS (AUC 0.107; all 4 merges at P=1.0000 called wrong).
Mechanism, from the operator: fibres overlapping in z, invisible to 2-D features.

TASK, in priority order:
1. Build the control-only review round: ~60 accepted merges sampled UNIFORMLY across
   all six wells, no flagged cases, no density matching, reasons required. This is the
   number the production decision needs. Build it; the operator runs it.
2. While that waits: probe whether z-overlap is detectable from 2-D features on the 15
   merges already labelled (7 different / 8 same). Start with stain intensity in the
   overlap region — two superimposed fibres should sum brighter than either. If a
   feature separates them the linker can be gated instead of abandoned.
3. Do NOT move threshold 0.90. Do NOT tune anything on the review rounds. Confidence
   is anti-correlated with correctness, so raising it selects more errors.

CONSTRAINTS: CPU ONLY (chronic bugchecks incl. 0x116 VIDEO_TDR / NVIDIA driver). No
GPU or Omnipose. Do not edit PrecisionMyotube/DEVELOPMENT_PLAN.md or
coordination/WORKBOARD.md. Conversion_Efficiency/** and cpenv are read-only. Nothing is
committed (47 entries, HEAD 0322ebf). Claude lane = annotation_tools/, model_labs/,
competitors/, docs/, coordination/reports/claude_*.
```
