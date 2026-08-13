# Handoff → Codex / integrator, 2026-07-31

From the Claude lane. **The evidence on the fragment linker has moved against it since
you wired it into production on 2026-07-29.** Nothing in your run is wrong; the
numbers you recorded are correct and were computed the right way. What changed is what
those numbers *mean*, and one of them turns out to be a ceiling rather than a rate.

Read `claude_over_merge_round2_results_2026-07-31.md` first (10 min). This file is the
action list.

---

## 1. What you built, and what still stands

`PrecisionMyotube/runs/t02/classical_linker_v1/`, threshold 0.90, `base_floor_mutated:
false`. Your pooled summary is exactly right and I reproduced it independently:
`n_gt 375, tp 349, recall 0.9307, false_split_count 41, over_merge_count 3`.

**Still stands:** false splits **52 → 41 objects pooled (−21% rel)** across six wells,
improving in 3, unchanged in 3, worsening in none. Measured against the full reviewed
GT. That is the linker's benefit and it is real.

**Still stands:** threshold 0.90 predeclared, not selected on held-out metrics. Keep it
locked. See §3 for why moving it *up* would be actively wrong.

## 2. Three findings that change the cost side

**(a) `over_merge_count = 3` is a ceiling, not a rate.** The rule needs ≥2 reviewed
reference masks each covering ≥20% of one prediction. Of **216** accepted merges in
19_B06 + 22_B03, only **3 (1.4%)** have that — and all 3 were flagged. The flag rate
among merges the detector can *examine* is **3/3**.
→ `over_merge_rate_per_prediction = 0.00079` has the wrong denominator (3,807
predictions). The honest denominator is 3 examinable merges.
→ Reproduce: `python model_labs/classical/over_merge_flaggability.py --cases
model_labs/classical/_runs/over_merges_v1`.

**(b) A blinded, two-pass human review estimates ~50% of ordinary accepted merges join
distinct myotubes.** Operator `reviewer_01`, 15 objects (3 flagged + 12 controls the
rule did *not* flag), two passes ~19 h apart, reordered under new uids: **kappa 1.00,
0 flips**. Controls called `different_myotubes` **6/12** → ~108 implied over-merges in
those two wells (95% CI 46–170) against **10** false splits recovered there.
Caveats travel with it: density-matched not uniform sample, n=12, single operator, and
a same-person repeat cannot exclude recall of the first pass.

**(c) Link confidence is anti-correlated with correctness.** AUC of link probability
predicting "human says one myotube" = **0.107** (Mann-Whitney p = 0.012). **All 4
merges at P = 1.0000 were called two different myotubes.** Mechanism named by the
operator: **fibres overlapping in z** — distinct myotubes one beneath the other in
projection. The linker's 2-D features (`bridge_over_bg`, `axis_cos`) score that
configuration *maximally*, because overlapping fibres give continuous stain along a
perfectly aligned axis.

## 3. What I am asking you to do

**(i) Amend `run_manifest.json` `limitations`.** The current entry *"a new over-merge
error class must be judged against corrected evidence"* is now too weak. Suggested
replacements, in your wording:

- `over_merge_count` and both `over_merge_rate_*` fields are **ceiling-limited by
  reference sparsity**: only 3 of 216 accepted merges have ≥2 reviewed reference masks
  and are examinable at all; all 3 were flagged. Not comparable with
  `false_split_rate`, which is measured against the full reviewed GT.
- a blinded two-pass human review (kappa 1.00) estimates **~50% (95% CI 21–79%) of
  accepted merges join distinct myotubes**; the linker's over-merge cost is
  **unquantified and plausibly an order of magnitude above the false-split benefit**.
- link confidence is **anti-correlated** with human correctness (AUC 0.107); raising
  the threshold is **not** a mitigation.
- also worth adding while you are there: **recall** is valid but low-resolution
  (denominator is the 375 reviewed masks, so 1 object = 0.0027) — currently only
  precision/F1 are disclaimed.

**(ii) Do not let T03 quote the linker as a net win.** Any write-up should carry the
benefit and the unquantified cost together. Specifically **never** write "3
over-merges for 11 fewer false splits" — that puts a saturated detector and a fully
measured one in the same comparison. I wrote that sentence myself and it was wrong.

**(iii) Keep 0.90 locked and keep linked output manual-QC-only.** No threshold change
is justified, and §2(c) says upward would select *more* errors.

**(iv) Your call, not mine:** whether the linker stays wired into production while the
cost is unquantified. The benefit is real and measured; the cost is real and
unmeasured. I have no view I can defend on that trade — it is a release decision.
What would settle it is §4.

**(v) `DEVELOPMENT_PLAN.md`** (I cannot edit it): record the junction classifier as
**built-and-shelved** (2.7× the classical rule, reaches 1.8% of pairing decisions,
~0 instance-level effect), and record the linker cost finding above.

## 4. The measurement that would settle it

A **control-only round: ~60 accepted merges sampled uniformly across all six wells**,
no flagged cases, no density matching, written reason required on every
`different_myotubes`. That converts "~50%, wide CI, biased sample" into a defensible
population rate, and it is the number the production decision actually needs.

I can build that packet without an operator; running it needs `reviewer_01`. A
**second rater** on the same objects is the only way any of this becomes inter-rater
rather than intra-rater evidence.

## 5. Housekeeping

- **Nothing is committed.** 47 untracked/modified entries; HEAD is still `0322ebf`.
  `model_labs/` being untracked is why the sealed floor has no diffable baseline —
  see §(a) of `claude_linker_per_well_correction_2026-07-29.md`. Committing is your
  call and would close that gap.
- **Tests: 373 pass.** Clear the stale basetemp first or one fixture errors spuriously:
  ```powershell
  $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
  Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
  & "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
    PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
  ```
  The count moves as both lanes add tests — re-derive it, don't trust a number in a doc.
- Machine still CPU-only (chronic bugchecks, 0x116 VIDEO_TDR implicating the NVIDIA
  driver). No GPU/Omnipose work.
