# T02 model lane — progress report, 2026-07-22

**From:** Claude model laboratories
**Task:** T02 (`coordination/requests/claude/2026-07-21-t02-start.md`)
**Status:** candidate 1 (classical floor) **complete and sealed**; candidate 2
(Omnipose) environment **solved and GPU-verified**, harness not yet written.

Nothing in `PrecisionMyotube/` or `coordination/WORKBOARD.md` was edited. No
commit, push, or cleanup was performed. `Conversion_Efficiency/cpenv` was
re-checked after all environment work and is unchanged (torch 2.11.0+cu128,
cellpose 4.2.1.1, CUDA available).

---

## 1. Headline finding — the classical floor's pooled score is largely circular

The six-fold run scores **recall 0.915 / matched IoU 0.917 / length MdAPE 0.000**.
That length error of exactly zero is not a success; it is a symptom.

`model_labs/classical/circularity_audit.py` quantifies it:

- only **40 of the 375** reviewed `complete` masks were ever edited, so **89.3% of
  the ground truth is a verbatim accepted proposal**;
- the classical candidate re-derives instances from the same deterministic recipe
  (`semantic_territory`) that generated those proposals;
- **47.4% of matched pairs (165/348) are pixel-identical**, IoU = 1.000; median
  matched IoU is exactly 1.000 in three of six wells.

Split honestly:

| subset | n | recall | median IoU | |
|---|---|---|---|---|
| unedited GT | 335 | 0.979 | 1.000 | **circular** |
| edited GT (real correction pairs) | 40 | **0.500** | **0.648** | **meaningful** |

**The defensible classical floor is recall 0.500 at median IoU 0.648.**

### Request to the T03 scorer

Please do **not** rank the classical floor against a learned candidate on the
pooled metric. Omnipose will not share the proposal generator and so receives none
of this structural advantage; the pooled comparison is biased toward the floor.
Report the edited/unedited split, or score on the edited subset.

This is consistent with DEVELOPMENT_PLAN.md section 9, which already labels the
classical proof-of-loop figures "illustrative and circular"; this run turns that
warning into a measured number.

The 40 correction pairs were used for **evaluation only** — not for training and
not for tuning (`run_manifest.json`: `correction_pairs_used: false`,
`synthetic_pairs_used: false`).

---

## 2. Second finding — the bootstrap's ignore mask is not a review-coverage mask

`bootstrap_v1/<well>/ignore.tif` marks **only pixels where two reviewed instances
overlap**. It says nothing about which regions were reviewed. Any candidate that
trains on `labels.tif` + `ignore.tif` therefore treats every non-`complete` pixel
as **background**, including the 839 `ambiguous` and 31 `border_truncated`
proposals.

Measured across the six wells:

| class | area (% of all six fields) |
|---|---|
| `complete` (trainable) | 1.415 |
| `ambiguous` (unreviewed) | **1.613** |
| `border_truncated` | 0.136 |

**More real-myotube pixels would be trained as background than as foreground.** A
model trained that way is explicitly taught to suppress genuine myotubes, and
nothing in the loss curve would reveal it.

The plan's rule — "`border_truncated`, `ambiguous`, `occluded`, rejected ... are
not complete training targets" — governs *targets*. `model_labs/_shared/
training_masks.py` supplies the missing half: they must also not be *background*.

Policy implemented (7 tests):

| decision | training role |
|---|---|
| `complete` (reviewed) | foreground target |
| `ambiguous` | **ignore** — operator declined to assert identity |
| `border_truncated` | **ignore** — real fibre, not measurable; "background" would be false |
| instance overlap | **ignore** — a flat raster cannot hold two identities |
| binding `training_exclude.json` ids | **ignore** — see below |
| rejected / unproposed | background — an informative negative, kept |

Note the binding exclusions specifically: `19_B06/myotube_0377` and
`22_B03/myotube_0321` still read `complete`/`reviewed` in the source
`*.qc.instances.json`. Without explicit handling they fall through to *background*,
asserting "empty field" over fibres the operator re-reviewed as ambiguous. They are
now ignored.

**This is a note for the T01 owner, not a request to change the freeze.** The
bootstrap is hash-frozen and correct as specified; the gap is that its `ignore.tif`
is easily mistaken for a review-coverage mask. Candidates should build the fuller
mask at training time.

---

## 3. Candidate 1 — deterministic classical ridge/graph floor

`model_labs/classical/` (`ridge_graph.py`, `run_folds.py`, `circularity_audit.py`,
`README.md`).

Pipeline: canonical `semantic_territory` (imported unchanged, cached ~3 min/field)
→ skan branch graph with anti-parallel junction pairing → nearest-retained-fibre
territory assignment → traced-length and area gates. CPU-only, no weights, no
framework that could touch `cpenv`.

**Canonical fidelity.** Stage B is a parameterised re-implementation of
`precision_myotube.fiber_gate.trace_fibers`, which hard-codes `SPUR_UM = 10.0` and
`STRAIGHT_DOT = -0.5`. The canonical file was **not modified**. A test pins the
equivalence: at those constants both tracers yield an identical fibre-length
multiset, so the candidate cannot silently drift from the validated recipe.

**Why parameterise those two constants.** At canonical settings the tracer puts
~4 predicted fragments on every reviewed myotube and recovers 0.59x its area — the
floor independently reproduces the dominant human-correction error class
(`too_short` under-tracing, 35 of 40 correction pairs). Relaxing `straight_dot`
from −0.5 to 0.0 lifts single-well recall 0.457 → 0.629 and median IoU 0.444 →
0.625.

**One deliberate fix.** `length_gated_territory` measures nearest-skeleton distance
over the *whole* skeleton, so territory nearest a pruned spur is orphaned — 32% of
territory area on `23_B02_ctrl`. Stage C measures distance to retained fibre pixels
only: assignment 0.679 → 1.000, median best-overlap IoU 0.490 → 0.588, GT matched
16/35 → 22/35.

**Fold protocol.** Six whole-well leave-one-well-out folds. The 32-point grid is
scored once per (well, parameter); each fold selects by mean
`precision_weighted_score` over its **five training wells only**, ties breaking to
the lowest parameter index. Two assertions guard it (held-out ∉ training set;
exported `image_id` == held-out field), and predictions are re-checked
`reviewed=False` after export. All six folds independently selected
`spur_um=10.0, straight_dot=0.0, min_length_um=25.0`.

The grid was fixed before the run and **not pruned afterwards**, even though
`spur_um=4.0` was visibly worse on an early single-well probe — pruning it would
have been selecting the grid after inspecting a well that later serves as a
held-out fold.

**Evaluation GT.** Scoring uses reviewed `complete` masks **minus the two binding
exclusions** — 375 total, matching `bootstrap_manifest.json`. This matters because
the source files still carry those two as reviewed/complete. The filtered sets are
written and hashed under `eval_gt/` so T03 can reproduce exactly what was scored.

**Precision caveat.** Pooled precision is 0.067, but that is dominated by a
sparse-GT effect rather than hallucination: the reviewed `complete` set is a small
subset of the fibre-like structure in each field (377 complete vs 839 ambiguous and
553 rejected). An unmatched prediction is usually real structure the operator did
not certify as fully measurable.

**Structural limitation.** The floor emits mutually exclusive masks, so it cannot
represent a crossing as two overlapping instances. Over-merge rate is 0.000 partly
for that reason. This is a thing a learned candidate should beat.

---

## 4. Candidate 2 — Omnipose environment solved

`pm-omnipose` created (isolated; `cpenv` untouched and re-verified afterwards).

| | |
|---|---|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| omnipose / cellpose_omni | 1.1.4 / 1.1.4 |
| numpy | 2.2.6 |
| device | RTX 5070 Ti Laptop, capability 12.0 (sm_120) |
| environment hash | `1ff21d7cd24ae291f29c002a740f78a6168427a71c0366ffc302bcb32da9cb6d` |

Two traps worth recording, both now encoded in `environment.yml` and
`verify_env.py`:

1. **Install order.** `pip install omnipose` pulls a CPU-only torch and silently
   *replaced* the working CUDA build (2.11.0+cu128 → 2.13.0+cpu, exit code 0).
   Omnipose must be installed **first**, then CUDA torch forced back on top.
2. **`is_available()` is not proof on this GPU.** Blackwell is sm_120; a torch
   built only through sm_90 reports available and then fails at the first kernel
   launch. `verify_env.py` therefore runs a real matmul and asserts the device
   capability appears in `torch.cuda.get_arch_list()`.

Also note: the cu128 wheel index tops out at torch 2.11.0, so the torch 2.13.0 that
omnipose's resolver wants has no CUDA build. Accepted inert conflict: torch
upgrades `fsspec` past what `aicsimageio`/`s3fs` declare; this lab reads images
with `tifffile` and never uses those readers.

**Open design decision (not yet made):** `CellposeModel.train(...)` exposes no
per-pixel loss mask, so the ignore mask from §2 cannot be handed to the trainer
directly. Options are tile exclusion, painting ambiguous regions out to
background-like texture, or subclassing the train step. This is a scientifically
consequential choice and is flagged for the owner before implementation.

---

## 5. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python -m pytest PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_t02
python model_labs/classical/run_folds.py --out model_labs/classical/_runs/v1
python model_labs/classical/circularity_audit.py --run model_labs/classical/_runs/v1
python model_labs/omnipose/verify_env.py          # from pm-omnipose
```

**80 tests pass** (was 57 at handoff: 31 canonical + 26 lab). 23 added — 16 for the
classical candidate (canonical-tracer equivalence, assignment coverage, gates, fold
honesty, memory contract, export/scoring contract) and 7 for the ignore-mask policy.
The canonical 31 and the pre-existing lab tests are unchanged and still pass.

One shared-core change: `_shared/predict_export.py` gained a `status` parameter,
defaulting to the existing `"ambiguous"` (an existing test pins that default). T02
passes `status="complete"` because `benchmark_instances` scores only predictions
whose status is `"complete"` — an export left at the default would have been
silently scored as **zero detections**.

## 6. Sealed artifacts for T03

Under `model_labs/classical/_runs/v1/`:

- `predictions/classical_ridge_graph/v1-fold-<well>/<image_id>.instances.json` —
  unreviewed canonical `InstanceSet`, one per fold, each with a
  `prediction_manifest.json` carrying full `ModelProvenance` (architecture,
  environment hash, input-manifest hash, seed, thresholds, selected-on wells);
- `eval_gt/<well>.eval_gt.instances.json` — hashed scoring sets (375 masks);
- `run_manifest.json` — command, hashes, seed, per-fold parameters and metrics,
  timing, failures (none), limitations;
- `grid_scores.json` — the full (well × parameter) table behind every selection;
- `circularity_audit.json` — §1.

All results are exploratory, single-operator, proposal-conditioned, retrospective
development evidence. Not consensus, not inter-rater agreement, not prospective
validation. Predicted instance counts are not authoritative independent-myotube
counts.
