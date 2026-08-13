# Over-merge review: scored — and the "3 over-merges" figure does not mean what I said

2026-07-30. Results for the packet built in
`claude_over_merge_review_packet_2026-07-29.md`. Operator `reviewer_01` reviewed all
15 objects blinded and exported at 04:49:56Z.

**Two findings, and the second is the important one.**

1. The review is **inconclusive on the three flagged cases** — it fails the
   calibration check that was pre-registered before it ran.
2. Independently of the review, `over_merge_count = 3` is a **ceiling imposed by the
   sparse reference set, not a rate**. Of **216** accepted merges in these two wells,
   **213 (98.6%) cannot be flagged as over-merges at all**, and **all 3 that could be
   were**. So the trade I recommended — "3 over-merges for 11 fewer false splits" —
   **compares a saturated detector against a fully measured one. It should not be
   quoted in that form.**

---

## 1. What the reviewer decided

| | same_myotube | different_myotubes | ambiguous_2d | n |
|---|---:|---:|---:|---:|
| **flagged (real over-merge candidates)** | 2 | **1** | 0 | 3 |
| **controls (merges the rule did NOT flag)** | 6 | **6** | 0 | 12 |

The single flagged case called `different_myotubes` is **`22_B03_act104_egfrc` label
530** — the three-fragment chain merged at **P = 1.0000 twice**. That is the case the
packet report singled out in advance: *"if a human calls this one
different_myotubes, that is a more serious finding than a marginal 0.90-something
merge going wrong — it says confidence is not tracking correctness at the top of the
range."* It was called exactly that. But see §2 before weighting it.

Scored with `qc-review score-over-merge-review`; report at
`PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.score.json`.

## 2. The pre-registered calibration check fails

The packet report fixed this rule in advance: *"If the reviewer calls a large share of
the 12 controls `different_myotubes`, they are not discriminating and the verdicts on
the 3 real cases carry little weight."*

**Controls were called `different_myotubes` 6 times out of 12 (50%).**

And the sharper, cutoff-free version of the same test:

> flagged called `different` = **0.33** · controls called `different` = **0.50**

The flagged cases were called over-merges **less** often than the controls. The
verdicts carry **no association with the benchmark's flag** — if anything the sign is
inverted. So:

- **"2 of 3 flags were reference-set artefacts" is NOT a supported conclusion.**
- **The one confirmed over-merge is weakly supported**, because a reviewer whose
  control verdicts are at chance provides little evidence either way on any single
  case. It is a lead, not a result.
- Per the pre-registered rules, the flagged cases remain **UNRESOLVED**.

I set the numeric "large share" cutoff (25%) only now, when scoring — that part is
post-hoc and I am flagging it. The conclusion does not rest on it: the
flagged-rate ≤ control-rate comparison needs no cutoff.

### Session facts, reported because they bear on the weight

Median time-to-decision **6.0 s**, min 3.0 s, max 61.0 s; **11 of 15 decided in
under 10 s**. The first case took 61 s and the last seven averaged ~5 s.
**Zero** `ambiguous_2d` calls and **zero** notes, in a task the packet explicitly
framed as often unresolvable in a single 2-D plane.

Two readings, and I cannot separate them from this export:

- **(a) the pass was too fast to be discriminating** — 3–5 s is not long for five
  panels plus a brightness adjustment; or
- **(b) the reviewer was discriminating, and over-merges really are common among
  ordinary accepted merges** — which §3 shows is entirely possible.

Some of the blame for (a) may be the instrument's: the page does not require a note
on a `different_myotubes` call, does not track whether the reference panel was ever
opened, and makes a decision reachable in one keypress from the default view.

## 3. The structural finding: `over_merge_count = 3` is a ceiling

This needs **no human judgement** and does not depend on §1 or §2 at all.

The benchmark flags an over-merge when **≥2 reviewed reference masks each cover ≥20%
of one prediction**. The eval GT is a sparse reviewed subset. So a merge overlapping
zero or one reference **cannot be flagged however wrong it is**. Measured directly
(`model_labs/classical/over_merge_flaggability.py`):

| well | accepted merges | 0 refs ≥20% | 1 ref ≥20% | **≥2 → eligible** | flagged |
|---|---:|---:|---:|---:|---:|
| 19_B06_act104_trka | 107 | 83 | 22 | **2 (1.9%)** | 2 |
| 22_B03_act104_egfrc | 109 | 102 | 6 | **1 (0.9%)** | 1 |
| **both** | **216** | **185** | **28** | **3 (1.4%)** | **3** |

**98.6% of accepted merges are invisible to the over-merge rule. Of the 3 the rule
could examine, it flagged 3 — a 100% flag rate among eligible merges.**

Consequences, stated plainly:

- **`over_merge_count = 3` is a lower bound whose ceiling is also 3.** It is not a
  measurement of how often the linker over-merges.
- **`over_merge_rate = 0.0008` is meaningless.** Its denominator is 3,807
  predictions; the honest denominator is **3 eligible merges**, giving 3/3.
- **The trade is not "3 over-merges for 11 fewer false splits."** False splits are
  measured against the full 375-object reviewed GT and improved 52 → 41. Over-merges
  are measured against 3 examinable objects. The two numbers are not commensurable,
  and putting them in one sentence — which I did — implies a cost/benefit comparison
  the data cannot support.
- The correct statement of the cost side is: **unmeasured**, not small.

### This is the same mistake I made before, twice now

The mean-of-wells recall error (`claude_linker_per_well_correction_2026-07-29.md`)
and this one are the same failure: **accepting a favourable number without checking
what its denominator actually was.** There it was wells weighted equally regardless
of object count; here it is a rate over predictions when the detector could only see
three of them. Recorded as a durable lesson, not an anecdote.

## 4. Where this leaves the linker

Unchanged and still true:

- **False splits 52 → 41 objects pooled (−21% relative)**, improving in 3 wells,
  unchanged in 3, worsening in none. Measured against the full reviewed GT.
- **Recall flat** (348 → 349 of 375) — descriptive, not a benefit claim.
- Threshold **0.90 remains locked.** Nothing in this review was used to tune it, and
  nothing here licenses moving it.

Changed:

- The over-merge cost is **unquantified**, with one weakly-supported confirmed case
  and a detector that saturates at 3. The previous framing overstated how well that
  cost was pinned.
- This **reinforces manual-QC-only** use of the linked output, which is where the
  pre-registered rules already pointed if any case confirmed. It does not by itself
  unwind the linker: the false-split gain is real and separately measured.
- Codex's `run_manifest.json` already carries *"a new over-merge error class must be
  judged against corrected evidence"* in `limitations`. That line should now say
  explicitly that `over_merge_count` / `over_merge_rate` are **ceiling-limited by
  reference sparsity (3 of 216 merges examinable)** and cannot be compared with
  `false_split_rate`.

### The honest headline now

> **11 fewer false splits, measured. Over-merge cost unmeasured — the detector could
> examine only 3 of 216 merges and flagged all 3. Recall flat (348 → 349 of 375).**

## 5. What would actually settle it

In priority order. None of this is started.

1. **A blind repeat of this packet** — same 15 objects, reshuffled, new uids, second
   pass by the same reviewer — to measure intra-rater reliability and separate
   §2(a) from §2(b). The repo already has this pattern (`blind_repeat/`,
   `qc-review blind-compare`). Cheapest decisive step, no new extraction.
2. **A control-only round with a real sample size** — say 60 accepted merges sampled
   across all six wells, no flagged cases, requiring a note on every
   `different_myotubes`. That estimates the true over-merge rate among accepted
   merges, which is the number the trade actually needs and which no current
   artifact contains.
3. **Instrument changes before either**: require a note on `different_myotubes`,
   record which panels were opened and for how long, and consider gating the
   decision until the reference panel has been viewed. A 3-second verdict should be
   visible in the export, not inferred from timestamps afterwards.
4. **Independently sampled, fully annotated tiles** remain the only real fix for the
   sparse-GT problem underneath all of this — the same conclusion reached when
   GT-centred neighbourhood scoring was rejected.

## 6. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
# 353 passed
```

Reproduce both results (CPU-only, seconds — reads cached artifacts):
```
python -m annotation_tools.qc_review.cli score-over-merge-review \
  --decisions PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.over_merge_review.json \
  --key PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.key.json
python model_labs/classical/over_merge_flaggability.py \
  --cases model_labs/classical/_runs/over_merges_v1 \
  --out model_labs/classical/_runs/over_merge_flaggability_v1.json
```

New code: `cli.py::cmd_score_over_merge_review` (implements the pre-registered rules
rather than a post-hoc summary; refuses a threshold or batch mismatch),
`model_labs/classical/over_merge_flaggability.py`. New tests:
`annotation_tools/tests/test_over_merge_score.py` (11),
`model_labs/tests/test_over_merge_flaggability.py` (8).

All results remain exploratory, single-operator, proposal-conditioned, retrospective
development evidence — not consensus, not inter-rater agreement, not prospective
validation. **Single-operator is doing real work in this report: with one reviewer at
chance on the controls, no verdict here is consensus about anything.**
