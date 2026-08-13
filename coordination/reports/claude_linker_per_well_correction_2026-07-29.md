# Linker@0.9 — per-well audit of my own headline, and a correction

2026-07-29. Amends `claude_linker_integration_measurement_2026-07-28.md` §"Instance-level
result". No new computation: this is a re-read of the *same* artifact
(`model_labs/classical/_runs/linker_instance_v1.json`) at per-well and pooled
granularity instead of mean-of-wells.

Codex wired threshold 0.90 into production during this session
(`PrecisionMyotube/runs/t02/classical_linker_v1/`, `linked_candidate.py`
`DEFAULT_THRESHOLD = 0.90`, run completed 16:11 today). **Two things to say up
front, in fairness:**

1. **Codex did not inherit my error.** Their `run_manifest.json` summary is
   *pooled* — `n_gt 375, tp 349, recall 0.93067, false_split_count 41,
   over_merge_count 3` — which matches the pooled figures I derive below exactly,
   independently computed. They also record `base_floor_mutated: false` and
   `threshold_origin: "predeclared ... not selected on held-out instance
   metrics"`. The production wiring is sound.
2. **The error is in my own report**, which used mean-of-wells and therefore
   overstated the recall gain by ~5x. That framing should not be quoted from
   `claude_linker_integration_measurement_2026-07-28.md` §"Instance-level result".

The one substantive gap remaining in Codex's `limitations` list: it flags
precision/F1 as uninterpretable under sparse GT, but says nothing about the
**recall** claim, which fails for a different and narrower reason (§1). That is
the change this report asks for.

### Why recall is not in the same category as precision and F1

Corrected 2026-07-29 after review — an earlier draft of this report said recall
was "uninformative, like precision and F1." **That was wrong, and the distinction
matters.** Sparse GT breaks precision structurally: its denominator is *all*
predictions, and most predictions have no reference to match because the reviewed
set covers only part of each field, so a low precision measures the annotation
budget rather than the model. F1 inherits that. **Recall's denominator is the GT
set itself** — the 375 reviewed, `complete` masks — so `tp/n_gt` is a valid
*descriptive* statistic for exactly that reviewed, proposal-conditioned subset.

What is wrong with the recall *claim* is therefore not the metric but the
inference from it: the effect is **+1 object in 375**, and it **reverses sign if
one of six wells is dropped**. Too small and too well-dependent to support "the
linker improves recall" — while remaining a perfectly good description of what
happened on this subset. Quote it as a descriptive number if useful; do not
promote it to a benefit.

---

## 1. The correction

I reported **recall +0.0149** at threshold 0.9. That is a *mean over six wells*.
Object-weighted (pooled) over the same six wells:

| threshold | pooled recall | pooled Δ | TP of 375 GT | mean-of-wells Δ |
|---|---:|---:|---|---:|
| classical floor | 0.9280 | – | 348 | – |
| linked@0.5 | 0.8347 | −0.0933 | 313 | −0.0712 |
| linked@0.7 | 0.8853 | −0.0427 | 332 | −0.0152 |
| **linked@0.9** | **0.9307** | **+0.0027** | **349** | +0.0149 |

**The recall gain at 0.9 is one additional matched object out of 375** (TP
348 → 349). That is noise, not a +1.5-point improvement.

The mean-of-wells figure is inflated because it weights all six wells equally
while their GT counts range **35–119**, and the well carrying the entire gain
(`23_B02_ctrl`, +0.1143) has the **smallest** GT set in the corpus — 35 objects,
9% of the GT, weighted at 1/6 of the headline. In object terms the whole result is
**+4 TP in 23_B02, +2 in 33_C09, −2 in 19_B06, −3 in 22_B03 = +1 net** — and
because the two gaining wells are the two smallest, equal weighting turns that +1
into +0.0149.

### Per well, threshold 0.9 vs the floor

| well | n_GT | TP floor → @0.9 | Δrecall | false-split objs floor → @0.9 | over-merged objs | n_pred |
|---|---:|---|---:|---|---:|---|
| 19_B06_act104_trka | 119 | 112 → 110 | **−0.0168** | 18 → 9 | **2** | 842 → 547 |
| 22_B03_act104_egfrc | 60 | 58 → 55 | **−0.0500** | 6 → 5 | **1** | 853 → 500 |
| 23_B02_ctrl | 35 | 27 → 31 | **+0.1143** | 11 → 11 | 0 | 1030 → 844 |
| 29_C05_br223_egfrc | 59 | 56 → 56 | 0.0000 | 4 → 3 | 0 | 789 → 573 |
| 32_C08_br223_igf1r | 54 | 51 → 51 | 0.0000 | 8 → 8 | 0 | 873 → 650 |
| 33_C09_br223_trka | 48 | 44 → 46 | +0.0417 | 5 → 5 | 0 | 892 → 693 |
| **pooled** | **375** | **348 → 349** | **+0.0027** | **52 → 41** | **3** | 5279 → 3807 |

Recall **improves in 2 of 6 wells, is unchanged in 2, and falls in 2.**

Leave-one-well-out on the mean recall delta itself:

| drop this well | mean Δrecall over the other five |
|---|---:|
| (all six) | +0.0149 |
| 19_B06 | +0.0212 |
| 22_B03 | +0.0278 |
| **23_B02_ctrl** | **−0.0050** |
| 29_C05 | +0.0178 |
| 32_C08 | +0.0178 |
| 33_C09 | +0.0095 |

**Remove `23_B02_ctrl` and the recall claim reverses sign.** The entire headline
rested on the single most-fragmented well — the same well my previous report
already flagged as "the informative outlier" for threshold 0.5, without noticing
it was also carrying the 0.9 result. I checked the outlier's effect at one
threshold and not at the one I recommended.

---

## 2. What does survive — and it is the thing that was always the target

False splits, pooled over all six wells:

| | objects false-split (of 375 GT) | rate |
|---|---:|---:|
| classical floor | 52 | 0.1387 |
| linked@0.9 | **41** | **0.1093** |

**11 fewer false-split objects, −21% relative.** Consistent, not outlier-driven:
improves in 3 wells, unchanged in 3, **worsens in none.** Fragmentation is the
error class the project has repeatedly named as dominant, and `false_split_rate`
is the one metric here measured against GT that is not distorted by GT sparsity.

Over-merge, the new error class: **3 objects across all six wells** (2 in 19_B06,
1 in 22_B03), pooled rate 0.0080. Both affected wells are exactly the two where
recall fell — the same merges cause both, as expected.

### So the honest one-line justification for 0.9

> At 0.9 the linker trades **3 new over-merges for 11 fewer false splits**, with
> **recall flat (+1 matched object of 375)** and instance count down 28%.

Not "recall +0.015, F1 +0.050". Both of those should be dropped from any T03
write-up — but for different reasons: **F1 because sparse GT makes it invalid**,
**recall because +1 object in 375 is too small and too well-dependent to be a
benefit claim**, not because the number is meaningless.

The choice of 0.9 **over 0.5 and 0.7 is strengthened**, not weakened: pooled recall
at those thresholds is −0.093 and −0.043, far worse than the mean-of-wells figures
suggested. 0.9 is clearly the only viable point of the three.

---

## 3. Answers to the five checks addressed to Codex

**(a) Is `test_default_decider_is_the_classical_rule` sufficient?** **No, on its own —
but the gap is closed by artifact evidence.** The test compares
`junction_decider=None` against an explicit decider that re-calls
`pair_junction_ends`, on a synthetic cross field. Both arms run through the
*refactored* code, so the test cannot detect a refactor that shifted the floor —
which is precisely the risk, since `build_branch_graph` / `junction_candidates` /
`pair_junction_ends` were extracted from the sealed tracer and `model_labs/` is
untracked (no committed baseline to diff against).

I checked it end-to-end instead. The `classical` arm of **both** integration
harnesses reproduces the sealed v1 run (`_runs/v1/run_manifest.json`) **per fold,
on all 13 recorded metrics and on the fold-selected tracer/filter params**, to the
sealed manifest's rounding (5e-4); integer `n_pred`/`tp` match exactly, and
`n_pred` equals the sealed `n_instances_predicted` in every well:

| well | sealed n_pred | harness classical n_pred |
|---|---:|---:|
| 19_B06_act104_trka | 842 | 842 |
| 22_B03_act104_egfrc | 853 | 853 |
| 23_B02_ctrl | 1030 | 1030 |
| 29_C05_br223_egfrc | 789 | 789 |
| 32_C08_br223_igf1r | 873 | 873 |
| 33_C09_br223_trka | 892 | 892 |

So the floor did not move. The residual weakness is *forward-looking*: nothing
cheap currently guards it, because a real guard needs the ~4-minute fold run. If
that matters, the durable fix is to commit `model_labs/` so the tracer has a
diffable baseline — not another synthetic-field test.

**(b) LOWO honesty — no leakage found in either harness.**
- `run_learned_junction_folds.fit_fold_models`: `train_pairs` and
  `train_junctions` both filter `!= held_out`; the gate-threshold sweep scores
  only junction keys built from `train_pairs`, so the held-out well cannot enter
  threshold selection. The threshold is chosen on training-well accuracy *of
  training-fitted models* — in-sample for the models, but held-out-clean, and the
  prior report already discloses the pooled-vs-per-fold difference (0.648 → 0.645).
- `run_linker_folds`: `fold_train = [p for p in train_pairs if p.well != well]`,
  refit per fold. Thresholds are a **fixed reported grid** (0.5/0.7/0.9), never
  fitted — so there is no threshold selection to leak.
- Both take per-fold `TracerParams`/`FilterParams` from the sealed run, where they
  were selected on training wells only.

One caveat that is *not* leakage but is adjacent: picking 0.9 because it looked
best across a reported 3-point grid is a human selection made while looking at
held-out means. With n=6 wells and §1's outlier dependence, that is worth stating
wherever 0.9 is justified. Codex's "predeclared" note is the right mitigation.

Codex's own manifest independently corroborates (a) with
`base_floor_mutated: false`, and already satisfies (b)/(d) in its
`limitations` list except for the recall point above.

**(c) Is over-merge 0.0000 → 0.0009 acceptable?** Now answerable with counts
rather than rates: **3 objects across six wells**, against **11 recovered false
splits**. My read: yes, worth it — but the trade is "3 over-merges for 11 fewer
splits", and it is *not* free of recall cost in the two wells where it fires
(−0.017, −0.050). It remains a judgement call, and it is the operator's, since
over-merge is one of their three named error classes. The 3 objects are
identifiable and small enough to review by hand; that is the cheapest way to
settle it, and I'd recommend it before T03 quotes the number.

**(d) Eval-GT sparsity.** Confirmed and now sharper: 375 GT objects total against
~5,279 floor predictions. **Precision and F1 are invalid** in every arm — their
denominator includes thousands of predictions that no reviewed mask could ever
match. **Recall is valid but low-resolution**: its denominator *is* the reviewed
set, so `tp/n_gt` describes that subset honestly; with 375 objects one object is
0.0027, so a +0.0027 delta is a single object and cannot carry a claim (§1). The
28% drop in instance count remains unvalidatable against a sparse GT. Net: a T03
claim should rest on `false_split_count` and `over_merge_count` — raw counts,
where the resolution is visible — with recall quoted descriptively if at all.

**Closed direction, recorded so it is not re-proposed:** I had floated
GT-centred neighbourhood scoring to rescue precision. It would not — restricting
evaluation to the neighbourhood of reviewed masks estimates *conditional local
matching performance*, not field-level precision, because the denominator is
still chosen by where the annotation happened to be. The only real fix is
**independently sampled, fully annotated tiles**. Not started; not this session's
work.

**(e) DEVELOPMENT_PLAN.md.** Still my recommendation that it record the junction
classifier as built-and-shelved with the 1.8%-coverage reason; I cannot edit it.

---

## 4. Baseline / housekeeping (this session)

- **The test count moved under me mid-session.** The documented baseline of 276
  was correct at session start; `PrecisionMyotube/tests/test_t03.py` and
  `test_linked_candidate.py` were rewritten at 15:57 today (Codex's linker
  wiring), taking the tree to **282**. With my 2 new tests: **284 passed**, under
  both an in-repo and an out-of-tree `--basetemp`. Anyone confirming a baseline
  against this tree should re-derive the count rather than trust 276.
- **The documented verification command emitted a spurious error once**, at
  `tmp_path` setup for `test_build_active_round_serves_only_unlabeled_junctions`
  (`ValueError: ... is not a normalized and relative path`). It came from the
  *stale* `tmp/pytest_resume` left by a previous session; deleting that directory
  first makes it pass. Not a code defect. Worth adding `Remove-Item -Recurse
  -Force tmp/pytest_resume` ahead of the command in §7 of the resume doc.
- **Fixed a real latent bug in my lane** (`model_labs/tier_a_audit/audit.py`):
  `build_manifest` keyed every hashed path with `p.relative_to(ROOT)`, which
  raises for any caller-chosen `out_dir` outside the repo — so
  `run_audit(<out-of-tree dir>)` crashed, and 2 tests failed whenever pytest's
  `--basetemp` sat outside the checkout. The key logic is now a testable
  `manifest_key()` with an absolute-path fallback; declared inputs keep their
  repo-relative, forward-slashed keys, so **existing manifests are byte-identical**.
  Two tests pin both halves.

Reproduce §1 from the committed artifact — no re-run needed, it is a re-read of
`model_labs/classical/_runs/linker_instance_v1.json`.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence — not consensus, not inter-rater agreement,
not prospective validation.
