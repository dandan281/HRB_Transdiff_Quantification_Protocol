# Round 2: kappa = 1.00. The reviewer was right, my rule was wrong, and the linker has a real problem

2026-07-31. Scores the blind repeat built in
`claude_over_merge_repeat_packet_2026-07-30.md`. Same 15 objects, reordered, new uids,
~19 h after round 1.

**Headline: the reviewer reproduced all 15 verdicts exactly (kappa 1.00, 0 flips).
Round 1's "the reviewer was not discriminating" reading is refuted. What remains is
the other reading — the over-merge flag is blind to almost all over-merges — and it is
now supported by a self-consistent reviewer, seven written reasons, and a named
mechanism. The linker's confidence runs BACKWARDS: every object it merged at
P = 1.0000 was called two different myotubes (AUC 0.107, Mann-Whitney p = 0.012).**

This changes my recommendation. It does not change the threshold, which stays locked.

---

## 1. Intra-rater agreement

| | value |
|---|---|
| objects paired (by identity, not position) | 15 / 15 |
| agreement | **1.00** |
| expected by chance at observed marginals | 0.502 |
| **Cohen's kappa** | **1.00** |
| flips | **0** |
| P(15/15 by chance) | 3.3 × 10⁻⁵ |

Seven `different_myotubes` calls, seven written reasons, **0 missing** — the note
requirement worked. Zero `ambiguous_2d` in either round.

**Caveat I cannot remove:** same reviewer, same 15 distinctive images, ~19 h apart.
Perfect agreement cannot fully separate *stable judgement* from *remembering my own
earlier answer*. What argues against pure recall is the reasons: they are specific,
image-referenced, and mechanistic (below), which verdict-recall alone would not
produce. A stronger design needs a second rater, not a second pass.

## 2. I am revising a pre-registered rule, and here is why

Round 1 pre-registered: *"a high control `different_myotubes` rate means the reviewer
is not discriminating, so the case verdicts carry little weight."* That rule fires
again here — controls are 6/12.

**The rule was mis-specified.** It used the control rate as a *proxy* for reviewer
noise, because at the time nothing measured reviewer noise directly. A blind repeat
measures it directly. When a direct measurement is available it supersedes the proxy —
and that reasoning is outcome-independent: it would have applied identically had kappa
come out 0, in which case the same rule would have voided the round. The scorer now
implements exactly that fork, and a test pins the losing branch so the revision cannot
become a way to rescue any result:

- high control rate **+ kappa ≥ 0.8** → the flag is under-detecting;
- high control rate **+ kappa < 0.8** → round void;
- high control rate **+ no repeat** → unresolved, *go run a repeat*.

Recorded in the report payload as `rule_revision_2026_07_30`.

## 3. What the reviewer actually said

Seven reasons, and they converge on one mechanism — **fibres overlapping in z**:

> *"This is a typical three-dimensionality shit. We can see obviously that the yellow
> part (right hand side) is one myotube that's nearer to the microscope; however the
> purple and blue are the ones underneath it and you wrongly identify them together."*
> — 19_B06 label 83, merged at P = 0.966

> *"This is a dimensionality mistake. You wrongly classify two [myotubes] that overlap
> but are one beneath the other as one while it should be two."* — 22_B03 label 1078,
> merged at **P = 1.0000**

> *"Those are two distinctive ones that overlap with each other."* — 19_B06 label 821,
> merged at **P = 1.0000**

This is a **mechanism, not a complaint**, and it explains the confidence inversion.
The linker's features are 2-D: bridge stain along the gap, axis alignment. Two fibres
crossing or lying one above the other in projection produce *continuous stain* along a
*perfectly aligned axis* — precisely the configuration that maximises the score. The
model is not failing at hard cases; it is confidently wrong at a case type its feature
space cannot represent.

**Vocabulary note:** the reviewer never used `ambiguous_2d`, including on the cases
they described as 3-D. They are asserting the occlusion is *readable* — "obviously",
"nearer to the microscope" — so `different_myotubes` is their considered call, not a
fallback. My vocabulary assumed z-overlap would land in `ambiguous_2d`; it does not.
Future rounds should say so explicitly rather than leaving the reviewer to choose.

## 4. Confidence runs backwards

| | called `same_myotube` (n=8) | called `different_myotubes` (n=7) |
|---|---:|---:|
| median link probability | 0.9303 | **1.0000** |
| range | 0.904 – 0.983 | 0.928 – 1.000 |

- **AUC of link probability predicting "human says same" = 0.107.** (0.5 = no
  information; below 0.5 = more confident means more likely wrong.)
- **Mann-Whitney U = 6.0, p = 0.012** — significant despite n = 15.
- **All 4 objects merged at P = 1.0000 were called `different_myotubes`.** 4/4.

The operational consequence is blunt: **raising the threshold does not help.** The
usual remedy for a precision problem — merge only above 0.95, or only at 1.0 — would
select *more* of the errors, not fewer. This is now reproducible via
`score-over-merge-review` (`confidence_calibration` block).

(The association with 3-fragment chains — 4/5 called different vs 3/10 of 2-fragment —
is in the same direction but **not significant**, Fisher p = 0.119. Not claimed.)

## 5. What this does to the linker's cost/benefit

Controls are ordinary accepted merges the flag did **not** raise. **6 of 12** were
called two different myotubes by a self-consistent reviewer.

| | estimate |
|---|---|
| control over-merge rate | 6/12 = **0.50**, 95% CI (Clopper-Pearson) **[0.21, 0.79]** |
| accepted merges in these two wells | 216 |
| implied over-merges, same two wells | **~108** (95% CI 46 – 170) |
| false splits actually recovered, same two wells | **10** objects (18→9 and 6→5) |

**Both sides of that table have limits and I will not repeat my denominator mistake.**
The false-split figure is counted against the *reviewed* GT only (179 objects in these
two wells), so it is itself ceiling-limited and understates the true benefit. The
over-merge figure comes from direct human inspection of merges, so it is *not*
GT-limited — but it is extrapolated from 12 density-matched objects, not a uniform
random sample, and dense neighbourhoods plausibly over-merge more than sparse ones.

So the two numbers are not commensurable. What is defensible is the **order of
magnitude and the direction**: the cost is plausibly ~10× the benefit in the same
wells, and the lower CI bound alone (46) already exceeds it by 4×. That is enough to
say the trade **can no longer be assumed favourable**, which is a change from every
prior report of mine.

## 6. Recommendation

1. **Tell Codex the evidence has moved.** The linker is wired into production at 0.90
   (`runs/t02/classical_linker_v1/`). Its `limitations` should now record: over-merge
   count is ceiling-limited (3 of 216 merges examinable); a blind, self-consistent
   human review estimates ~50% of accepted merges join distinct myotubes; and link
   confidence is *anti*-correlated with correctness (AUC 0.107), so thresholding up is
   not a mitigation. **Linked output stays manual-QC-only** — it already is, and this
   strengthens that considerably.
2. **Do not move the threshold.** 0.90 stays locked. Nothing here was used to tune it,
   and §4 says tuning it upward would make things worse, not better.
3. **Run the control-only round (prior report §5.2), now the top priority** — ~60
   accepted merges sampled **uniformly** from all six wells, no flagged cases, no
   density matching, reasons required. That converts "~50%, wide CI, biased sample"
   into a defensible population rate. It is the number the trade actually needs.
4. **A second rater on the same objects** would address the memory caveat in §1 and is
   the only way this becomes inter-rater rather than intra-rater evidence.
5. **The z-overlap mechanism is the real finding for the modelling lane.** It is not
   fixable with 2-D features. It is the first evidence in this project that argues
   *for* z-information — worth noting because the junction work argued against extra
   modelling capacity, and this points the other way for a different reason.

## 7. What has not changed

- **False splits 52 → 41 objects pooled (−21% rel)** across six wells: still measured,
  still real, still improving in 3 wells and worsening in none.
- **Recall flat** (348 → 349 of 375), descriptive only.
- Threshold **0.90 locked**.
- Still **single-operator**. Two passes by one person is intra-rater reliability, not
  consensus and not inter-rater agreement. Everything above is development evidence.

## 8. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
# 373 passed
```

```
python -m annotation_tools.qc_review.cli score-over-merge-review \
  --decisions PrecisionMyotube/annotation_work/over_merge_r2/over_merge_r2.over_merge_review.json \
  --key       PrecisionMyotube/annotation_work/over_merge_r2/over_merge_r2.key.json \
  --first-pass-key       PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.key.json \
  --first-pass-decisions PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.over_merge_review.json
```

Report: `PrecisionMyotube/annotation_work/over_merge_r2/over_merge_r2.score.json`.
New tests (+5): the revised rule's winning *and* losing branches, the no-repeat
branch, and the confidence-calibration direction.

## 9. One instrument finding worth keeping

**The reference-masks panel was never opened — 0 of 15**, in either the flagged cases
or the controls, and panels 3 (proposed link) and 4 (linked mask) were never opened
either. Every verdict came from the fragments overlay plus raw Desmin.

That is not the defect it first looks like. The question is *"is this one myotube or
two"*, and the honest evidence for it is the stain, not the reference masks — which
are the thing under dispute. It means these verdicts are **independent of the eval
GT**, which is what makes them usable as a check on a GT-derived metric at all. But it
also means the reviewer never assessed "did the reference set split a fibre", so the
two `same_myotube` verdicts on flagged cases are *not* evidence that those flags were
reference-set artefacts. They are unexplained.

Telemetry earned its place on the first run: none of §9 was recoverable from round 1's
timestamps.
