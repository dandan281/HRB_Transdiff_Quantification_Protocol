# The 839 ambiguous objects — what they are, how much is recoverable, and a labelling shift

Date: 2026-08-11
Lane: Claude (annotation tooling / model laboratories)
Status: measurement only. No code changed, no labels touched, nothing committed.

Prompted by the question "do we need to generate more data?". The inventory answer was
that 839 of 1,800 reviewed objects are marked `ambiguous` and excluded from supervision —
2.2x the size of the training set, already paid for in operator time. This asks what that
pool actually contains.

The headline is two findings, and the second one was not what I went looking for.

1. **At most ~290–390 of the 839 are plausibly recoverable**, not the whole pool. Half of
   it is marginal on basic size or intensity, and the median ambiguous object is half the
   length and as dim as a *rejected* one.
2. **The first well reviewed was certified at twice the rate of the next four.** It
   supplies 32% of the training set. This is a labelling-standard question that touches
   T02/T03 directly and is independent of anything about the ambiguous pool.

---

## 1. There is no recorded rationale, anywhere

Every `ambiguous` call carries an empty `note` in `*.decisions.json` and empty `notes` in
`*.qc.instances.json`. Non-empty count across all six wells: **zero**. The review logs
(`*.qc.instances.review_log.jsonl`) carry only `id`, `action`, `status`, `reviewer` — no
reason, and no timestamp either.

So "why was this ambiguous?" is not answerable from the labels. It can only be inferred
from the nine geometric features recorded per decision, or by re-examining the images.
That is a tooling gap worth closing before the next review round, not a data problem:
a reason code on `ambiguous` would make this entire analysis unnecessary next time.

Status maps 1:1 onto action, with no cross-terms — `ambiguous`↔`ambiguous`,
`complete`↔`accept`, `border_truncated`↔`accept`, and rejected objects are simply absent
from the instances export (1,800 decisions → 1,247 instances).

## 2. The ambiguous pool is systematically marginal

Medians across all six wells, with the Mann-Whitney AUC for separating ambiguous from
complete (0.5 = no separation; below 0.5 means ambiguous is smaller on that feature):

| feature | complete | ambiguous | rejected | AUC amb-vs-comp |
|---|--:|--:|--:|--:|
| length_um | 141.92 | **71.57** | 99.99 | 0.285 |
| area_um2 | 832.20 | **383.20** | 868.10 | 0.270 |
| fiber_mean | 2455.40 | **1639.50** | **1644.90** | 0.272 |
| width_um | 11.61 | 8.80 | 21.23 | 0.330 |
| aspect | 11.82 | 8.79 | 4.57 | 0.365 |
| solidity | 0.59 | 0.70 | 0.48 | 0.636 |
| extent | 0.16 | 0.23 | 0.21 | 0.639 |

The ambiguous population is **half the length, less than half the area, blobbier** (higher
solidity and extent, lower aspect) than the certified one.

**The number that matters most is `fiber_mean`.** Ambiguous objects sit at 1,639 and
rejected objects at 1,645 — statistically indistinguishable — while certified objects sit
at 2,455. On Desmin intensity the ambiguous pool resembles the *rejected* pool, not the
certified one. That is not the profile of "real myotubes the operator could not finish
tracing." Much of it looks like material the operator could not confirm was a myotube at
all.

## 3. How much is recoverable — an upper bound of ~290–390

Taking the certified population's 10th–90th percentile band as "indistinguishable from a
certified myotube on the recorded features":

| constraint | ambiguous inside band | share |
|---|--:|--:|
| length only | 567 | 0.676 |
| + fiber_mean | 390 | 0.465 |
| + aspect | 365 | 0.435 |
| + width | **289** | **0.344** |

And from the other direction: 300 (35.8%) are dimmer than the certified 10th percentile,
260 (31.0%) are shorter than it, and **426 (50.8%) fail on one or the other**. Half the
pool is marginal on size or signal alone.

**Read this as a ceiling, not an estimate.** Every one of these objects was looked at and
*not* certified. The operator had a reason; it simply was not recorded. The most likely
unrecorded reasons — a fibre crossing or overlapping another, an endpoint that cannot be
resolved in projection — are invisible in all nine features, and they are the same
z-overlap problem that killed the fragment linker and that no 2-D method can fix. So the
true recoverable count is at most ~290–390 and plausibly well below it.

Even so, ~290 objects is 77% of the current 375-instance training set, obtainable from
operator time rather than microscope time. That is the honest answer to "do we need more
data": **there is a substantial recoverable pool, but it needs re-review, not acquisition,
and it is worth targeting the ~290 rather than re-reviewing all 839.**

## 4. The labelling shift — the more consequential finding

Review order is not recorded in the logs, but the six `decisions.json` files were written
in sequence on 2026-07-21 between 13:06 and 18:27 PT, which gives an ordering. In that
order:

| # | well | proposals | rejected | candidates | complete | ambiguous | **certified / candidate** |
|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | 19_B06_act104_trka | 300 | 60 | 240 | 120 | 111 | **0.500** |
| 2 | 22_B03_act104_egfrc | 300 | 63 | 237 | 61 | 176 | 0.257 |
| 3 | 29_C05_br223_egfrc | 300 | 59 | 241 | 59 | 177 | 0.245 |
| 4 | 32_C08_br223_igf1r | 300 | 75 | 225 | 54 | 164 | 0.240 |
| 5 | 33_C09_br223_trka | 300 | 65 | 235 | 48 | 182 | 0.204 |
| 6 | 23_B02_ctrl | 300 | 231 | 69 | 35 | 29 | 0.507 |

**The first well certified 50% of its candidates. The next four certified 20–26%**, tightly
clustered with a mild continued decline. The candidate pools are nearly identical in size
(240, 237, 241, 225, 235), so the detector proposed comparable material in every treated
well and what changed is how much of it got certified.

Three things make drift a better explanation than biology:

- **Candidate counts are flat across all five treated wells.** A genuinely better-converting
  well should produce *more* candidate myotubes, not the same number certified at twice the
  rate.
- **Well 1's certified objects are unremarkable.** Median certified length 149.5 µm against
  131–146 elsewhere; median certified `fiber_mean` 2,490 against a 1,590–3,789 spread. Its
  certified objects do not look like a better-converting well's.
- **Well 1's *ambiguous* objects are among the shortest** (median 58.1 µm, versus 95.8 in
  well 2 and 92.8 in well 4). Consistent with the longer mid-range objects having been
  certified there and left ambiguous elsewhere.

Well 6 is not evidence either way. It is the control, it rejected 231 of 300, and its 69
candidates are the few clear myotubes a control well contains — a high certification rate
on a small easy pool is expected, and it is confounded with being reviewed last.

### Why this matters beyond the ambiguous pool

`19_B06_act104_trka` contributes **119 of 375 authoritative training masks — 31.7%**. If it
was certified under a looser standard, then roughly half of those would not have been
certified under the standard applied to wells 2–5, which is on the order of 16% of the
training set holding a `complete` label the later standard would not have given it.

> **Corrected 2026-08-12.** This originally read "120 of 375", conflating two different
> counts. The operator certified **120** objects in this well — that is the numerator of
> the 0.500 certification rate above and is unchanged — but one is removed by the binding
> exclusion in `training_exclude.json`, so **119** reach training. The certification-rate
> table is unaffected; only the training-contribution figure was wrong.

**Resolved 2026-08-12 by the integrator ruling** (`codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md`).
G-SO2 now discloses the certification shift; T03 retains the predeclared all-six-well
pooled primary and adds a **mandatory drop-one-well sensitivity analysis**; no relabelling
and no post-hoc reweighting. The measured impact is small: classical primary
**52/375 = 0.1387**, omitting B06 **34/256 = 0.1328**. So the shift is real and the concern
was worth raising, but it does not threaten the classical floor's false-split rate. The
three consequences below stand as the reasoning that prompted that ruling.

Three consequences, none fatal, all needing disclosure:

- **Leave-one-well-out is not homogeneous.** The fold holding out well 1 is scored against a
  more permissive ground truth than the other five. Its `false_split_count` is not strictly
  commensurable with theirs.
- **`false_split_count` pooled across six wells** — the predeclared T02 primary metric —
  pools two labelling standards. Since it is object-weighted, well 1's larger certified set
  carries proportionally more weight.
- **G-SO2** requires a leakage audit and explicit single-operator limitations. This belongs
  in that disclosure whether or not anything is done about it.

### What this is not

This is a **hypothesis supported by consistent evidence, not an established fact.** The
ordering rests on file mtimes, which are a proxy and could mislead if a file was rewritten.
Well 1 also carries a treatment combination (act104 + trka) that appears in no other well,
so biology cannot be formally excluded — only made less likely by the flat candidate counts.
Nothing here should be treated as a defect in the operator's work; a standard settling over
the first well of a 1,800-object single-operator review is ordinary, and it is exactly what
a blinded repeat round exists to detect.

## 5. What I would do about it, in order

1. **Do not re-review all 839.** Target the ~289 that sit inside the certified band on all
   four recorded features. That is the only subset where re-review has a good prior, and it
   is a tenth of the effort for most of the available yield.
2. **Record a reason code on `ambiguous`** in the QC review tool before any further review
   happens. Cheap, in my lane, and it makes the next round analysable instead of requiring
   this kind of inference. Suggested minimum: gap / crossing-or-overlap / faint / endpoint
   unclear / not-a-myotube.
3. **Test the drift directly rather than inferring it.** Re-present a sample of well 1's
   certified objects, blinded and interleaved with wells 2–5, and measure whether they are
   still certified. Thirty cases would settle it. The round-2 blinded machinery already
   exists.
4. **Disclose the certification-rate table to Codex for G-SO2** regardless of what is done
   about it. A 2x swing in certification rate across a single-operator corpus is exactly the
   kind of limitation that gate is for.
5. **Do not change any label.** Nothing above justifies unilaterally reclassifying an
   operator's calls, and doing so during a live T02 run would invalidate `dataset_sha256`
   for folds already trained.

## 6. Reproducing this

Three scripts, all read-only, in the session scratchpad: `ambig_recon.py` (rationale and
status/action inventory), `ambig_features.py` (feature distributions and AUCs),
`ambig_recover.py` (band membership and per-well rates). They read only
`PrecisionMyotube/annotation_work/*/{*.decisions.json,*.qc.instances.json}` and write
nothing. The `32_C08_br223_igf1r` well lives in a directory named `32_C08_smoke`, per the
remap already recorded in `bootstrap_manifest.json`.
