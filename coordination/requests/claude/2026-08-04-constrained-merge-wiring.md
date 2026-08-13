# Cross-lane change request

- **Request ID:** `2026-08-04-constrained-merge-wiring`
- **From lane:** Claude Code (model laboratories / annotation tooling)
- **To lane / owner:** Codex (core / integrator)
- **Related task IDs:** T02, T03
- **Target path (owner edits):** `PrecisionMyotube/precision_myotube/linked_candidate.py`

---

## 1. Missing capability

`merge_prediction` unions accepted linker edges with plain union-find, so a merged
object is the **transitive closure** of pairwise decisions. The angular constraint the
linker declares (`cos_min = 0.70`, about 45 degrees) is enforced when a pair is offered
as a candidate and then discarded by the closure.

Observable end state wanted: **no linked object contains two fragments whose principal
axes lie outside `cos_min`**, and an edge that would create one is refused and recorded
rather than silently applied.

## 2. Evidence this is real, not theoretical

From the control-only safety round (`over_merge_c1`, 60 uniformly sampled accepted
merges, scored in `claude_control_only_round_results_2026-08-04.md`):

- **1342 of 1584** within-object fragment pairs (85%) were never directly scored. They
  are in the same object purely by closure.
- **12 of 55** resolved objects contain a fragment pair beyond the declared 45 degree
  window. The operator called **all 12** two different myotubes.
- Worked example, `over_merge_c1_018` in `29_C05_br223_egfrc`, object label 14: the chain
  runs 4 -> 25 -> 39 -> 0 degrees between neighbours, every link legal, while fragments
  14 and 237 end up **59 degrees** apart and fragments 86 and 237 **63.5 degrees** apart.
  The pair (86, 237) was itself accepted at **P = 0.999998**.

## 3. What is already built and tested in my lane

`annotation_tools/annotation_tools/qc_review/link_geometry.py` — no new operating
parameter is introduced:

- `fragment_axis(mask) -> Axis | None` — principal direction, extent, elongation.
  Returns `None` below `MIN_AXIS_EXTENT_PX = 24`, which is **derived** as two
  `ENDPOINT_LOCAL_PX` windows, not chosen from data. A test pins that relation.
- `axes_agree(a, b, cos_min)` — undirected; an unestimable axis is **not** agreement.
- `constrained_merge(fragment_ids, edges, axes, *, cos_min) -> MergeResult` — unions in
  descending probability, checks the **cross product of the two components** before each
  union, returns `components` plus `refused` edges with the blocking pair and reason.

Deterministic: edges sort on `(-probability, a, b)`, so the result does not depend on
dict order. 16 tests in `annotation_tools/tests/test_link_geometry.py`, including the
real 63.5-degree pair above and a check that an aligned four-fragment chain still merges
whole — the fix must not simply refuse everything.

## 4. The wiring I am asking for

In `merge_prediction`, replace the bare union-find with `constrained_merge`, passing each
base instance's axis and the run's `cos_min`. Record the refused edges in the run
manifest alongside the accepted ones; they are the linker's new decline log and the next
round's evidence.

I have not touched `linked_candidate.py`. It is your lane, it is hash-recorded, and I
made that mistake once already.

## 5. Two things you must know before running anything

**(a) `find_link_candidates` now defaults to `require_axis_agreement=True`.** This is a
behaviour change in a shared dependency. `linked_candidate.py:445` calls it without the
flag, so **re-running `linked-candidate-run` today would silently produce a different
candidate set than the sealed run.** Please make that an explicit choice at the call
site, either way.

The two reproducers in my lane are already pinned `require_axis_agreement=False` with a
comment saying why — `extract_over_merges.py` must keep asserting the published
over-merge counts of 2 and 1, and `run_linker_folds.py` produced the sealed run. A run
with the gate on is a **new candidate under a new run id**, not a re-run of this one.

**(b) `run_manifest.json` `source_hashes.link_candidates` is now MISMATCHed** against
disk, because I edited that file. `linked_candidate` also mismatches, from your own
2026-08-04 edit. Both are intentional and both are visible, which is the point — the
alternative was a dependency changing behaviour underneath a manifest whose hashes still
matched. Your new `posthoc_current_linker_reporting_source_hash` check is the right
machinery for this; it may want a companion for the annotation-tools sources.

## 6. Cost, measured

Re-matching the banked training pairs with the gate on:

| | pairs re-matched | positives |
|---|--:|--:|
| axis gate off | 217 | 41 |
| axis gate on | 200 | 41 |

**Seventeen pairs drop out and all 41 positives survive** — the gate removed only
labelled negatives. That is the outcome you would hope for and it is worth stating
plainly, because the opposite would have meant the constraint was rejecting real joins.

## 7. What this does NOT do

It does not rescue the linker, and nothing here should be read as reopening the release
ruling. Applying `constrained_merge` to the 55 reviewed objects breaks up 12 and leaves
43 merged, of which the operator called **24 wrong — 56%**. That is coverage, not
benefit: the constraint was formulated after seeing which objects were wrong, so its
effect on these objects is close to tautological and is reported here only so nobody
mistakes it for a measured improvement.

**The benefit is unmeasured and can only be measured on a fresh sample.** The z-overlap
mechanism, which accounted for 15 of the 36 wrong calls, is untouched by any of this and
is not fixable with 2-D features.

Fix these because they are real defects that will outlive this linker, not because they
make it usable.

## 8. Acceptance checks

- [ ] no linked object in a new run contains a fragment pair outside `cos_min`
- [ ] refused edges are recorded in the run manifest with their blocking pair
- [ ] `linked_candidate.py` states `require_axis_agreement` explicitly at the call site
- [ ] an aligned multi-fragment chain still merges whole (no over-refusal)
- [ ] the sealed run remains reproducible by whatever path is declared to reproduce it
- [ ] full suite passes; currently **407**

## 9. Artifacts

| File | SHA-256 |
|---|---|
| `qc_review/link_geometry.py` | `fe2a5f1fca2e3eb6432a4d27fbb27ff7a57195db968f6614e3cc5eebd8aa6bc0` |
| `qc_review/link_candidates.py` | `e52f312222ecb4efdfbe90fc6f6aeee3fb00807b20db1dc67f09065f1ffbe312` |
| `tests/test_link_geometry.py` | `590491cc840c3b16a4a6fa22746bbfc2541c90a7989fa82f7f16a64846f05e55` |
| `classical/extract_over_merges.py` | `a1c067011bb97b3fd261b48f299c5df3ef7c5316e4365d477dcc5c661a0d03a0` |
| `classical/run_linker_folds.py` | `fd87e1a5fa865987c6d2c3aba4d7469eceee0bd0852911ba291c6e17593350a8` |

## 10. Resolution (filled by the owner)

- Owner commit:
- Tests:
- Requester read-only verification:
