# Verification pass over the linker amendment — and one default that inverted the burden of proof

Date boundary: 2026-07-30 America/Los_Angeles; evidence artifacts dated 2026-07-31 UTC
Author: Claude Code session, integrator scope
Execution constraint: CPU only
Status: amendment **verified**; one code default **fixed**; one divergence **left open for a decision**

This is a verification pass over the amendment recorded in
`codex_linker_release_ruling_2026-07-31.md`, which answers
`claude_handoff_codex_2026-07-31.md`. It is not a second opinion on the science; the
ruling's reading of the evidence is correct and this session agrees with it.

## 0. Two sessions edited the same files

The amendment and this verification ran concurrently on the same working tree. This
session read the pre-amendment state, began amending it, and detected the collision when
an edit failed against a file that had changed underneath it.

One edit was made and then reverted: the `limitations` literal in
`PrecisionMyotube/precision_myotube/linked_candidate.py`. The revert is byte-exact and was
verified by hashing the file back to `4de4f383…a351f`, the value the run manifest records
in `source_hashes.linked_candidate` **as executed**. Leaving that edit in place would have
silently staled the run's own source provenance, which is presumably why the amendment
deliberately left that file alone. Nothing else was written until the concurrent session
went quiet.

Consequence worth recording: a re-run of `linked-candidate-run` still emits the weak
2026-07-29 limitation `"a new over-merge error class must be judged against corrected
evidence"`. The amended text lives in the manifest, not in the generator. That is a
defensible trade — it keeps the executed-source hash honest — but it is a live footgun and
should be decided explicitly rather than inherited.

## 1. The amendment verifies

| Check | Result |
|---|---|
| Flaggability, reproduced independently from the cases directory | 216 accepted merges, 213 unflaggable, **3 eligible, 3 flagged** |
| `run_manifest.json` `source_hashes` vs on-disk sources | 4 / 4 match |
| Posthoc evidence hashes (report, flag artifact, score artifact) | 3 / 3 match |
| Metrics after amendment | `n_gt 375, tp 349, recall 0.9307, false_split_count 41, over_merge_count 3` — unchanged |
| Threshold and floor | `0.9`; `base_floor_mutated: false` — unchanged |
| Test suite | 375 passed before this session's change; **376 passed** after |

The four asks in the handoff are covered: the manifest limitations are amended and bound
to hashed evidence, the T03 assessment carries the safety limits and a fifth gate reason,
`DEVELOPMENT_PLAN.md` 2.5 §2 forbids the three-flags-versus-eleven-false-splits
comparison, and §9 records the junction classifier as built and shelved.

## 2. Fixed: an unaudited over-merge rate was being stamped valid

`PrecisionMyotube/precision_myotube/t03.py` computed:

```python
"over_merge_rate_interpretable": not bool(posthoc_safety),
```

So a run with **no** flaggability audit was reported as having an interpretable over-merge
rate. Absence of an audit raised the claim.

That is the same inference the round-2 review refuted, re-encoded as a default. The
`>=2 reviewed reference masks, each covering >=20% of one prediction` rule is a property of
the sparse reviewed-complete GT — 375 masks against a field that was never censused — not
a property of the linker. It ceilings **every** run scored against that reference set. The
sealed classical floor is not exempt and is arguably worse off: its predictions are more
fragmented (5,279 versus 3,807), so fewer of them can accumulate two references above the
coverage threshold, and its `over_merge_count: 0` is a tighter ceiling, not a cleaner
result.

Fixed by defaulting the flag to `False` for all runs and recording audit status separately,
so the two states stay distinguishable without one implying the other:

```python
"over_merge_rate_interpretable": False,
"over_merge_rate_audit": "posthoc flaggability audit attached" | "no posthoc flaggability audit attached",
```

The unaudited note now reads *"this rate is unestablished rather than valid"* and states
why the ceiling is a property of the reference set.

`test_unaudited_over_merge_rate_is_not_claimed_interpretable` pins the branch that was
wrong, following the same discipline §2 of the round-2 report used: assert on the losing
branch so the correction cannot quietly revert.

This was latent, not published. The on-disk `runs/t03/classical_v1/assessment.json` is
still assessment version 1.0 and predates the field, so no artifact ever carried the bad
stamp. It would have appeared the first time anyone regenerated the floor assessment.

`runs/t03/classical_linker_v1/assessment.json` was regenerated so source and artifact
agree. The diff is one added field plus the expected `assessor_source_sha256` change; no
metric moved and `over_merge_rate_interpretable` was already `False` there.

## 3. Open: the official classical floor assessment no longer reproduces

`DEVELOPMENT_PLAN.md` §2 names `PrecisionMyotube/runs/t03/classical_v1/assessment.json`
as "The official artifact". Regenerating it from current source produces a 30-line diff.

**Every numeric metric is byte-identical.** This was checked explicitly, not assumed. What
differs is interpretation and provenance:

- `assessment_version` 1.0 → 1.1;
- added `f1_interpretable`, `recall_interpretable_for_reviewed_subset`, `recall_note`,
  `recall_resolution_per_reviewed_object`, `over_merge_note`, and (after §2) the corrected
  `over_merge_rate_interpretable` and `over_merge_rate_audit`;
- a reworded gate reason: *"only one completed candidate is available; no candidate
  comparison is possible"* → *"no genuinely independent second candidate is available; a
  linked variant of the classical floor does not satisfy the two-candidate comparison"*;
- `assessor_source_sha256` differs, as expected;
- `statistical_analysis_plan_sha256` differs, which means `STATISTICAL_ANALYSIS_PLAN.md`
  changed at some point after that assessment was cut. That predates this work and is not
  explained by it.

This is a disclosure decision, not a correctness problem, and it is deliberately left
undone: silently re-cutting a sealed artifact that a plan and an accepted audit both cite
is exactly the kind of move that should be taken on purpose. Either regenerate it and
record the new hash in the plan, or state in §2 that the official artifact is version 1.0
and does not reproduce from current source. Doing neither leaves the plan and the code
disagreeing.

Two smaller items, both cosmetic and neither fixed here: the linker assessment now
concatenates run-level and assessment-level limitations, so "single operator…" and the
non-nested-LOWO caveat each appear twice in different wording; and
`coordination/WORKBOARD.md` still reflects the 2026-07-23 state, with no linker row.

## 4. Release call, and one constraint to add

Agreed: keep the linker wired at locked `P >= 0.90`, manual-QC-only, not released for
automatic use. The benefit is measured, the gate that catches the cost is a human looking
at every merge — the gate it already sits behind — the settling measurement is one packet
away, and deleting the run would destroy reproducible evidence.

**Add to the six binding rules: linked output must not be used as a proposal source for
new reviewed masks until the control-only round returns a population rate.**

The reason is specific rather than precautionary. The linker's error is precisely the
error that would contaminate the reference set it is scored against. And a pre-merged
proposal is not neutral: it pre-commits the hardest decision, leaving the reviewer to
actively split an object rather than to choose whether to join two. §9 of the round-2
report is what makes this concrete — all 15 verdicts came from the fragments overlay plus
raw Desmin, with the reference panel never opened once. Those verdicts were possible
*because* the reviewer was shown fragments. A linked proposal removes that evidence before
the human sees it.

Checked before asserting: no annotation packet builder reads the linked run. The only
Python references to `classical_linker_v1` are the CLI, the generator, and their tests. So
this is a forward constraint, not a rollback of anything in flight.

## 5. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
# 376 passed
```

```powershell
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" `
  model_labs/classical/over_merge_flaggability.py `
  --cases model_labs/classical/_runs/over_merges_v1
# 216 accepted merges; 213 (98.6%) cannot be flagged at all; 3 eligible; 3 flagged

& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m precision_myotube t03-assess `
  --run PrecisionMyotube/runs/t02/classical_linker_v1 `
  --out PrecisionMyotube/runs/t03/classical_linker_v1/assessment.json `
  --bootstrap-resamples 10000 --seed 20260723
```

| Artifact | SHA-256 |
|---|---|
| `precision_myotube/t03.py` | `d99ffb71a6c71c7840f3054edc8004c813b8df28a740a3de6f6658c24029da01` |
| `tests/test_t03.py` | `94984224578c0f039e89aa7b76ea78c05a6b313d217d0605f86a61f115f7b732` |
| `runs/t03/classical_linker_v1/assessment.json` | `6b3c76fb9b4ab5693fdded36e2c7277920f7bd2dc45b65e2223e95b15b7eb1ad` |
| `runs/t02/classical_linker_v1/run_manifest.json` (unchanged here) | `8b3699119edb045287b4c9d21dc83fadfbcde3690c5c1def34b13205b2b5c2df` |
| `runs/t03/classical_v1/assessment.json` (untouched, version 1.0) | `b7ee25428872ed6a1ab6482d8187e226f86179588953c9bce2d2f9ee44da0f8b` |
| `precision_myotube/linked_candidate.py` (reverted, matches manifest) | `4de4f383ddf79a91aca70fdeb70121bb01a67ad19d2bd7aff5850ce3393a351f` |

No threshold was tuned, no prediction or decision ledger was touched, no GPU, Omnipose,
Tier-A, or `Conversion_Efficiency/**` work occurred, no workboard edit was made, and
nothing was committed.
