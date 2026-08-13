# Over-merge round 2: instrument fixed, blind repeat built

2026-07-30. Acts on `claude_over_merge_review_results_2026-07-30.md` §5 items 3 and
1, in that order — the report put **instrument changes before any re-review**, because
round 1's central problem was that a 3-second verdict had to be *inferred* from
`decided_at` gaps after the fact instead of being visible in the export.

**Nothing is decided here. The packet is built and awaiting the operator.**

---

## 1. Why a repeat rather than more cases

Round 1 left two readings that its own data could not separate:

- **(a)** the reviewer was not discriminating (median 6 s, no notes, no ambiguous calls);
- **(b)** the reviewer *was* discriminating and over-merges are simply common among
  ordinary accepted merges — which is entirely possible, since only **3 of 216**
  merges are examinable by the over-merge rule at all.

A blind repeat of the *same 15 objects* separates them without any new extraction.
If the reviewer agrees with themselves, (b) survives and the control verdicts start to
look like a real signal about over-merge prevalence. If agreement is at chance, (a) is
established, round 1 is void, and the flagged cases stay unresolved.

## 2. Instrument changes (`over_merge_page/v2-telemetry`)

**(i) A `different_myotubes` call now requires a reason.** Round 1 exported seven of
them with empty notes, so there was nothing to audit. Enforced as a soft block: the
decision registers immediately, but the case is **not complete** until a reason exists,
the header shows an "awaiting a reason" count, the status line says *NEEDS A REASON
before it counts*, the note field turns red, and the field is **focused at the moment
of the call** rather than asked for at export time. Export refuses to proceed without a
confirm, and records `note_required` / `note_missing` per case either way.

**(ii) Dwell telemetry is recorded and shipped.** Per case: `ms_on_case`,
`panel_dwell_ms` and `panel_views` broken down by which view was actually on screen
(overlay hidden is attributed to `desmin`, since that is what the reviewer is looking
at), and at the moment of the decision `ms_on_case_at_decision`,
`panels_seen_at_decision`, and `reference_panel_seen_before_decision`. Away-time is
excluded — the timer flushes on window blur and restarts on focus.

**(iii) The reference panel is tracked, and its absence is shown.** While the
reference masks have not been opened for the current object, the status line carries
*"reference masks not yet viewed — 5"*. It is not a hard gate: forcing a click would
just add a reflex click. But a verdict reached without ever looking at the masks whose
disagreement raised the flag is now visible in the data, per case.

`INSTRUMENT_VERSION` is stamped into the page and the export so a scored report can
never silently mix a round that carried telemetry with one that did not. The scorer
degrades honestly on round 1's export: `instrument: v1-no-telemetry`, `telemetry:
{available: false}` with the reason, and note discipline reported as
`enforced: false, 7 of 7 missing`.

## 3. The repeat packet

`PrecisionMyotube/annotation_work/over_merge_r2/over_merge_review.html` + its own key.

Built with a new `--repeat-of` mode that pins the object set from round 1's key:

| property | round 1 | round 2 |
|---|---|---|
| objects | 3 flagged + 12 controls | **identical set** (verified) |
| uids | `over_merge_r1_001…015` | `over_merge_r2_001…015` |
| flagged at positions | 002, 004, 012 | **002, 012, 014** |
| objects at the same position | — | 3 of 15 |
| order seed | 20260729 | **20260730** |

**Why `--repeat-of` and not just a new seed.** Control *selection* consumes the same
rng as the presentation order, so changing one seed would have silently changed which
controls were chosen and the two passes would not have been comparable. The seeds are
now separate (`--seed` selects controls, `--order-seed` shuffles presentation) *and*
the repeat pins the exact object set from the prior key rather than trusting seed
reproducibility. The builder refuses a repeat that reuses the first pass's
`--batch-id`, since colliding uids would unblind it.

Pairing is recorded as `first_pass_uid` in the round-2 key and agreement is computed
**by object identity, never by presentation position** — pinned by a test, because
position-pairing would compare different objects in a deliberately reordered packet.

## 4. Scoring the repeat when it comes back

```
python -m annotation_tools.qc_review.cli score-over-merge-review \
  --decisions .../over_merge_r2.over_merge_review.json \
  --key       .../over_merge_r2.key.json \
  --first-pass-key       .../over_merge_r1.key.json \
  --first-pass-decisions .../over_merge_r1.over_merge_review.json
```

Adds an `intra_rater` block: agreement, expected-by-chance from the observed
marginals, **Cohen's kappa**, and every flip listed with its well and merged label.

### Reading rules, again fixed before the data exists

- **Kappa at or near 0** → round 1 described the reviewer, not the merges. Both passes
  become unusable as evidence about the three flagged cases, and the over-merge cost
  stays **unmeasured**. The next step would be §5.2 of the prior report (a control-only
  round with real sample size), not a third pass.
- **High agreement with controls still called `different` at ~50%** → reading (b).
  That would be a substantive finding: the over-merge rule is blind to most merges and
  the true rate is materially higher than 3. It would argue for restating the linker's
  cost side, not for moving the threshold.
- **High agreement and controls mostly `same_myotube`** → the round-1 control result
  was a fluke of the first pass; the flagged-case verdicts become usable.
- Reasons written for `different_myotubes` calls are now part of the evidence. A
  round where they are absent or contentless should be treated as round 1 was.
- **Threshold 0.90 remains locked**, and neither pass may be used to tune it.

## 5. Unchanged

- **False splits 52 → 41 objects pooled (−21% rel)** — measured against the full
  375-object reviewed GT, improving in 3 wells, worsening in none.
- **Recall flat** (348 → 349 of 375), descriptive only.
- **Over-merge cost unmeasured** — the detector can examine 3 of 216 merges and
  flagged all 3. Never quote "3 over-merges for 11 fewer false splits".

## 6. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
# 368 passed (353 before this session)
```

Rebuild the repeat (CPU-only, ~1 min — reads cached arrays, no re-extraction):
```
python -m annotation_tools.qc_review.cli build-over-merge-page \
  --cases model_labs/classical/_runs/over_merges_v1 \
  --repeat-of PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.key.json \
  --out PrecisionMyotube/annotation_work/over_merge_r2/over_merge_review.html \
  --key PrecisionMyotube/annotation_work/over_merge_r2/over_merge_r2.key.json \
  --reviewer reviewer_01 --batch-id over_merge_r2 --order-seed 20260730
```

New tests: `annotation_tools/tests/test_over_merge_repeat.py` (15 — note enforcement,
telemetry shipped and away-time excluded, honest degradation on a v1 export, kappa at
chance, identity-not-position pairing, and the built packet's same-objects/different-
order property).

All results remain exploratory, single-operator, proposal-conditioned, retrospective
development evidence. A blind repeat measures **intra-rater** reliability only — it is
one person agreeing with themselves, which is not consensus and not inter-rater
agreement.
