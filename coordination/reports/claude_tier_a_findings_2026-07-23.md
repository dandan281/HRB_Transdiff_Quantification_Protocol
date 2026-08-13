# Tier-A claim-quality findings — for the integrator

**From:** Claude model laboratories
**Date:** 2026-07-23 session
**Routes:** `claude_resume_state.md` §9 items 6 and 7
**Action requested:** a plan/workboard ruling. Neither item is mine to fix — both
touch `Conversion_Efficiency/` (validated baseline) and Tier-A release claims,
which are outside the Claude lane.

Both findings come from the MyoFuse assessment (`competitors/MyoFuse/ASSESSMENT.md`
§6). Neither affects T02/T03. Both affect **Tier A**, which
`DEVELOPMENT_PLAN.md` §12 identifies as the product closest to release — which is
why they are worth raising now rather than after T03.

Licence note: MyoFuse's **code is MIT but the article is CC BY-NC-ND**. Everything
below is a paraphrase and an independent calculation on our own numbers. No figure
or text is adapted, and nothing of theirs has been copied into the project
(`competitors/MyoFuse/NOTICE.md`; adoption status: none).

---

## Finding 1 — our conversion-efficiency rule is the method they showed to be biased

**What they showed.** Classifying a nucleus as fused by its *overlap with a myotube
mask* counts myoblasts lying above or below the fibre in projection. They report
this bias persists even at a **90%** overlap threshold.

**Where we stand.** `DEVELOPMENT_PLAN.md` §8 records the validated baseline as
"670 [nuclei] meet the primary 50% territory-overlap threshold, yielding 6.6245%
conversion efficiency". Our rule is the mask method, at a threshold **more
permissive** than the one they already showed biased.

**Why it is likely worse for us than for them.** Their fusion index is 30–50%, so
most nuclei are genuinely fused and contamination is a small relative error. Our
conversion is ~6.6%, so **~93% of nuclei are non-converted and therefore available
to be counted spuriously** when they happen to lie over a fibre. The same absolute
contamination is a much larger relative error at our base rate.

**Consequence for claims.** 6.6245% should be described as an **upper bound** until
the contamination is measured, not as a point estimate. The project convention of
reporting both the 25% and 50% overlap thresholds does not address this — both are
mask-overlap rules, so both carry the same bias; the pair brackets threshold
sensitivity, not projection error.

**Independent supporting evidence, on our own data.** We already know their
*mechanism* does not transfer: `competitors/MyoFuse/evidence/` shows that on our
Desmin the signal is inverted (median inside/ring 1.60; only 4.9% of counted-converted
nuclei show the dark hole their classifier keys on), because Desmin forms a
perinuclear cage. So we cannot simply adopt their classifier to fix this. The
critique lands; the remedy does not transfer.

**Cheapest way to size the problem** (needs operator time, ~not scheduled): have the
operator label a stratified sample of nuclei as in/out on Desmin and measure how
inflated the 50% rule actually is. Until then the direction of the bias is known
(inflation) and the magnitude is not.

---

## Finding 2 — one field per well is probably too few for the CI we imply

They report ~66 tiles for a ±5% fusion-index confidence interval. Their optics are
10×/0.3 with a 6.45 µm-pixel ORCA-R2 → 0.645 µm/px; ours is 0.6493 µm/px.
Effectively identical, so the arithmetic transfers directly.

| | tile / field | area |
|---|---|---|
| MyoFuse tile | 1168 × 1005 px @ 0.645 µm | 0.49 mm² |
| our field | 3636 × 3636 px @ 0.6493 µm | 5.57 mm² |

One of our fields ≈ **11.4** of their tiles, so ±5% needs ≈ **6 of our fields per
well**. We currently analyse **one**.

**Consequence for claims.** Per-well conversion-efficiency numbers carry a wider
interval than a single field supports. This is a sampling-adequacy issue, so it is
governed by `STATISTICAL_ANALYSIS_PLAN.md` rather than by anything in the model
lane. Note it interacts with the plan's existing rule that nuclei nested in a well
are observational subsamples, not biological replicates: adding fields improves the
*technical* precision of a well's estimate and does nothing for biological n.

**Caveat on transferring their number.** ±5% is calibrated to *their* variance at
30–50% FI. At our ~6.6% base rate the binomial variance per nucleus is lower but
the field-to-field heterogeneity of a transdifferentiation culture is unmeasured,
so 6 fields is an order-of-magnitude guide, not a derived requirement. A direct
calculation from our own field-to-field variance would be better — and is not
possible from one field per well, which is itself the point.

---

## What I am asking for

1. A ruling on whether 6.6245% may continue to be reported as a point estimate, or
   must be labelled an upper bound pending measurement.
2. A decision on whether Tier-A release requires additional fields per well, and if
   so whether existing plates can supply them retrospectively or new acquisition is
   needed.
3. Whether to schedule the stratified nucleus-labelling sample that would size
   finding 1.

Not requested: any change to T02/T03, which these findings do not touch.

All of this is exploratory, single-operator, retrospective development evidence.
