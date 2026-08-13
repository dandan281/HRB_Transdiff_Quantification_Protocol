# The control-only round: the linker's over-merge cost is now measured, and it is not survivable

2026-08-04. Scores `over_merge_c1`, the round the release ruling named as the
measurement that would settle automatic use. Sixty accepted merges, ten per well,
sampled with equal probability inside each of the six wells, no flagged cases and no
density matching.

**Headline: the population over-merge rate is 0.649 (95% CI 0.450 – 0.832), implying
~350 wrong merges of the 540 the linker accepts across six wells, against 11 false
splits recovered. The cost is roughly 32x the benefit at the point estimate and 22x at
the lower confidence bound. "Unquantified" is no longer the problem; the quantity is.**

This ends the open question in `codex_linker_release_ruling_2026-07-31.md` §4. It does
not change the threshold, which stays locked, and nothing here was used to tune it.

---

## 1. The primary result

Predeclared before the reviewer saw the page, carried in the packet key under
`control_only_round`, and read back by the scorer rather than restated by it.

| | value |
|---|---|
| merges reviewed | 60 |
| `different_myotubes` | **36** |
| `same_myotube` | 19 |
| `ambiguous_2d` (unresolved) | 4 |
| left undecided | 1 |
| **population over-merge rate** | **0.649** |
| 95% CI, stratified bootstrap over wells | **0.450 – 0.832** |
| implied over-merges across six wells | **~350 of 540** |
| false splits recovered, same six wells | **11** |

The estimator is the well-size-weighted mean of the per-well rates, because the design
draws ten merges from wells holding between 70 and 109 accepted merges. The unweighted
pool is 0.655, so the weighting is not doing the work — it is there to be correct, not
to move the number.

| well | S | D | A | — | rate | accepted merges |
|---|--:|--:|--:|--:|--:|--:|
| 19_B06_act104_trka | 6 | 4 | 0 | 0 | 0.400 | 107 |
| 22_B03_act104_egfrc | 2 | 7 | 0 | 1 | 0.778 | 109 |
| 23_B02_ctrl | 5 | 3 | 2 | 0 | 0.375 | 71 |
| 29_C05_br223_egfrc | 1 | 8 | 1 | 0 | 0.889 | 70 |
| 32_C08_br223_igf1r | 2 | 8 | 0 | 0 | 0.800 | 89 |
| 33_C09_br223_trka | 3 | 6 | 1 | 0 | 0.667 | 94 |

**Every well is above 0.37 and four of six are above 0.65.** The interval is wide
because six wells is a small outer sample, not because the wells disagree about the
direction.

## 2. The exclusions are not load-bearing

Five cases sit outside the rate: four `ambiguous_2d` and one the reviewer never
decided. Assign all five the most favourable way and the rate is **0.600**; assign them
the least favourable way and it is **0.677**. The conclusion is identical at both ends,
so nobody has to take the exclusion rule on trust.

**One disclosure.** The predeclared spec covered `ambiguous_2d` but not a case left with
no decision at all. `over_merge_c1_020` is excluded on the same reasoning, which is a
judgement made *after* seeing the export. The bounds above are exactly why that is safe
to state rather than hide, and the scorer records
`undecided_handling_was_predeclared: false` in the artifact.

## 3. Round 2 was, if anything, too kind

Round 2 estimated 0.50 from 12 density-matched controls in two wells and was criticised
in its own report as a biased sample. Those same two wells now return 11 of 19 resolved
= 0.58 on a uniform draw, and the six-well population rate is higher still.

The direction of the earlier bias is now clear and it ran the wrong way: matching
controls to the flagged cases on fragment count restricted round 2 to 2-3 fragment
objects. The uniform sample includes the tail the earlier rounds structurally could not
show — chains of 7, 11, 14, 18, 23 and 38 fragments — and that tail is where the linker
fails hardest.

## 4. Three exploratory findings. None were predeclared; all are hypotheses, not results

**(a) Fragment count separates almost perfectly.**

| fragments | n | called different |
|---|--:|--:|
| 2 | 32 | 0.56 |
| 3 | 8 | 0.50 |
| 4–6 | 5 | 0.80 |
| **7+** | **10** | **1.00** |

Every one of the ten merges joining seven or more fragments was called two different
myotubes; 14 of 15 at four or more. The reviewer proposed the rule unprompted and twice,
in their own words: *"if the branch is more than two, you can directly categorize them
into a different myotube"* and *"MORE THAN 3 OR 4 = TRASH"*.

**This must not be turned into a cap on these data.** The bucket boundary was chosen
after seeing the verdicts, from ten objects. Fitting a filter here and then reporting its
benefit on the same fifty-five objects is precisely the circularity this project has
already corrected once. It is a strong hypothesis for a new round, on new merges.

**(b) Confidence still runs backwards, and now it replicates.** Round 2 found AUC 0.107
at n=15 and the obvious objection was sample size. At n=55 the AUC of link probability
predicting "human says same myotube" is **0.323** (Mann-Whitney p = 0.027). **Twenty of
the twenty-five merges the linker scored at P = 1.0000 were called two different
myotubes.** Raising the threshold remains the wrong direction, now on an independent
sample four times larger.

**(c) The named mechanisms, tallied over the 36 `different_myotubes` notes.**

| mechanism | notes |
|---|--:|
| z-overlap ("dimension problem") | 15 |
| parallel adjacent fibres | 10 |
| too many fragments / "too many tubes" | 6 |
| near-orthogonal | 2 |

Z-overlap replicates as the leading mechanism. **Parallel fibres are new** and were not
visible in round 2 — distinct myotubes lying side by side and end to end get joined
because a bridge of stain runs along a well-aligned axis, which is exactly what the
feature set rewards.

**The two orthogonal calls are a specific bug lead.** The candidate window uses
`cos_min = 0.70`, roughly 45 degrees, so a near-orthogonal pair should never have been a
candidate at all. Either the endpoint-local axis diverges sharply from the fibre's global
orientation, or the union of a chain is being judged on one member's axis. Project rule:
myotubes do not branch at ~90 degrees. Worth a direct look at those two objects.

## 5. What this does to the release decision

The 2026-07-31 ruling kept the linker as a reproducible manual-QC candidate on the
grounds that the benefit was measured and the cost was not. **The cost is now measured
and it is about 32x the benefit.** The premise of that ruling no longer holds.

My recommendation, for the integrator to accept or reject:

1. **Automatic use stays not released.** Unchanged, and now for a measured reason rather
   than an unquantified one.
2. **Withdraw the manual-QC-proposal use as well.** At a 65% error rate the linker is not
   a useful proposal generator: a reviewer would reject roughly two of every three
   objects it proposes, and each rejection costs more attention than tracing from
   fragments would have. This is a stronger position than the ruling took and it is the
   part I would most want argued with.
3. **Keep the code, the run, and this round.** They are reproducible evidence and the
   only measured account of how this failure mode behaves. Deleting them would destroy
   the baseline that any future linker has to beat.
4. **The forward constraint stands and hardens:** linked output must not seed new
   reviewed masks. At 65% it would corrupt the reference set faster than review could
   catch it.
5. **`P >= 0.90` stays locked.** Nothing here was used to tune it. §4(b) says upward is
   worse, and §4(a) is not a licence to fit a new filter on this sample.

## 6. Limitations that travel with this number

- **Single operator.** This is one person's judgement, twice-checked in earlier rounds at
  kappa 1.00, but it is not consensus, not inter-rater agreement, and not independent
  ground truth. A second rater remains the only way that changes.
- **`different_myotubes` is being used more broadly than its name.** Several notes
  describe a partially wrong merge rather than two clean myotubes — *"the upper part of
  the yellow is correct, the lower part is not"*, *"you included one more fragment"*. For
  a safety rate that is arguably the right target, since a partially wrong merge is still
  a wrong object, but the verdict should be read as "this merge is wrong" rather than
  strictly "these are two distinct myotubes".
- **Six wells, one plate, retrospective, proposal-conditioned.** The interval covers
  sampling variation within this corpus and nothing beyond it.
- **The reviewer knew the round concerned linker merges.** The packet was blinded and had
  no group to find, but it was not blind to its own purpose.
- **Reference panel never opened before a decision — 0 of 59**, matching round 2. The
  verdicts are independent of the eval GT, which is what lets them check a GT-derived
  metric. It also means they say nothing about whether the reference set itself splits
  fibres.

## 7. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m annotation_tools.qc_review.cli `
  score-control-only-review `
  --decisions PrecisionMyotube/annotation_work/over_merge_c1/over_merge_c1.over_merge_review.json `
  --key       PrecisionMyotube/annotation_work/over_merge_c1/over_merge_c1.key.json `
  --out       PrecisionMyotube/annotation_work/over_merge_c1/over_merge_c1.score.json
```

Packet build (extraction ~8 min, page build ~4 min):

```powershell
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" model_labs/classical/extract_over_merges.py `
  --out model_labs/classical/_runs/over_merges_uniform_v1 `
  --wells 19_B06_act104_trka 22_B03_act104_egfrc 23_B02_ctrl 29_C05_br223_egfrc 32_C08_br223_igf1r 33_C09_br223_trka `
  --controls 10 --uniform-controls --seed 20260731
```

The extraction reproduced the published per-well over-merge counts (2 and 1) before
emitting anything, and independently recovered **216** accepted merges across the two
reviewed wells — the same denominator the flaggability audit reported.

Tests: **391 passed** (382 before this round; +9 for the control-only scorer).

| Artifact | SHA-256 |
|---|---|
| `over_merge_c1.over_merge_review.json` | `3874ae959f8014e252369650d6b81a891e15195b76004ddfcda8d8e0f7672784` |
| `over_merge_c1.key.json` | `9a6bf6f9adb16db911c8c36edbebdc79e00b5c4d7c5da64e60d8ec891f9f9436` |
| `over_merge_c1.score.json` | `e49b3c416d549c269007cb6b8b3182819de2b5ccb499a0569d629889094cf64f` |
| `over_merges_uniform_v1/cases.json` | `4a66cebee21e028848fed7e3dc2237f1ddfcb9ba34cd45ac1fca80bb6ca6a37a` |
| `qc_review/control_only_score.py` | `6ad4ee10e3b3cf35a5327b8da3239789018d74f8102a9688ce2788e7087f8621` |

No threshold was tuned, no prediction or decision ledger was touched, no GPU, Omnipose,
Tier-A or `Conversion_Efficiency/**` work occurred, and nothing was committed.
