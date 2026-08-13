# Junction classifier: instance-level integration measurement — NEGATIVE

2026-07-28. Continues `claude_junction_round2_results_and_feature_fix_2026-07-28.md`.
Everything before this scored the junction classifier on a **proxy** — junction
decisions in isolation (64.5% vs the classical rule's 23.8%, 2.7x). This wires
it into the actual tracer and scores the readout the project cares about.

**Result: no instance-level improvement. Recall −0.012, F1 −0.003, matched IoU
+0.003. The classifier is 2.7x better at its own task and that does not reach
the science.**

## Protocol

- `ridge_graph.trace_fibers_parameterised` gained an optional
  `junction_decider` hook. `None` (default) keeps the deterministic classical
  rule **bit-identical** — pinned by a new test, because the sealed floor must
  not shift under prior results. The learned rule is injected through that hook
  rather than by forking the tracer, so both arms differ in the junction rule
  and *nothing else*: same cached stage-A territory, same fold-selected
  `TracerParams`/`FilterParams` from the sealed v1 run, same sealed eval GT.
- **Leave-one-well-out, honestly.** For each held-out well the pair model, the
  branch-point gate, *and* the gate threshold are fitted on the other five
  wells' junction labels only.
- `straight_dot` differs between the sealed floor (0.0, fold-selected) and the
  labeling rounds (−0.5, canonical), but it is consumed only by the pairing
  step and never by `build_branch_graph` — so the branch graph, and every
  feature the classifier was trained on, is identical either way. Checked
  before running, not assumed.

Code: `model_labs/classical/learned_junctions.py`,
`model_labs/classical/run_learned_junction_folds.py`. Output:
`model_labs/classical/_runs/learned_junctions_v1.json`.

## Instance-level result (means over 6 LOWO folds)

| metric | classical | learned | delta |
|---|---:|---:|---:|
| n_pred | 879.8 | 886.5 | +6.7 |
| recall | 0.9149 | 0.9031 | **−0.0118** |
| precision | 0.0677 | 0.0661 | −0.0016 |
| f1 | 0.1244 | 0.1218 | −0.0026 |
| mean_matched_iou | 0.9174 | 0.9205 | **+0.0031** |
| false_split_rate | 0.1476 | 0.1473 | −0.0003 |
| over_merge_rate | 0.0000 | 0.0000 | 0.0000 |
| length_mdape | 0.0002 | 0.0002 | 0.0000 |

Matched masks are a hair better shaped (+0.003 IoU); recall is slightly worse.
Net: nothing. (Precision here is dominated by the sparse-GT effect the sealed
run already documents — the reviewed `complete` set is a small subset of the
fibre-like structure per field — so precision/F1 are not informative in either
arm. Recall, IoU and the error-class rates are the meaningful columns.)

## Why: the rule almost never fires

| well | all graph nodes | degree≠3 | degree-3 but sub-fibre-scale | **learned fired** | % of degree-3 |
|---|---:|---:|---:|---:|---:|
| 19_B06_act104_trka | 5,244 | 4,269 | 766 | 209 | 21.4% |
| 22_B03_act104_egfrc | 5,198 | 4,262 | 713 | 223 | 23.8% |
| 23_B02_ctrl | 15,760 | 12,808 | 2,925 | 27 | 0.9% |
| 29_C05_br223_egfrc | 7,594 | 6,373 | 1,074 | 147 | 12.0% |
| 32_C08_br223_igf1r | 5,228 | 4,332 | 689 | 207 | 23.1% |
| 33_C09_br223_trka | 10,570 | 8,973 | 1,517 | 80 | 5.0% |
| **total** | **49,594** | **41,017** | **7,684** | **893** | **10.4%** |

The learned rule reaches **893 of 49,594 pairing decisions = 1.8%**. Of the
8,577 degree-3 junctions, **89.6% are sub-fibre-scale** skeletonisation
whiskers that fall back to the classical rule — the exact population excluded
from the labeling pool back in step 1 as noise, for good reasons that still
hold.

So the classifier's instance-level effect was **structurally bounded to
near-zero before any of the accuracy work began**, and no amount of improving
it would have changed that. The measurement that mattered was the cheapest one
and it was run last.

## What this means

1. **Junction splitting is not the bottleneck.** The pipeline's instance-level
   quality is set elsewhere — consistent with the original parking rationale
   that the dominant error is `fragment_too_short` (Desmin signal-gap
   fragmentation), which is a *linking* problem, not a junction problem.
2. **It weakens the case for Omnipose further, not strengthens it.** The
   argument for a learned pixel model as "the residual solver" was about
   junction/cluster splitting. That residual has now been measured and it is
   worth ~0 at instance level. Spending 20+ GPU-hours (on a machine still
   crashing as of 2026-07-25) to attack it is not supported by evidence.
3. **The junction classifier is not worthless — it is mis-scoped.** It is
   genuinely 2.7x the classical rule on fibre-scale degree-3 junctions and is
   a usable high-precision assistant there (~82% at 30% coverage). If those
   junctions ever matter more — e.g. after fragmentation is fixed and instance
   counts are no longer dominated by broken fibres — it is built, tested and
   ready. It should be **kept and shelved, not deleted.**

## Recommended next

Attack fragmentation, not junctions. The fragment linker already exists
(AUC 0.902, high-precision assistant) and has **never been measured at
instance level either** — exactly the same gap this report just closed for the
junction classifier, and now runnable through the same `junction_decider`-style
integration harness. That is the cheapest high-value measurement remaining,
it is CPU-only, and it directly targets the error the project has repeatedly
identified as dominant.

Also worth recording (verified 2026-07-28 from the Windows System log): the
machine is **still crashing** — unexpected shutdown 2026-07-25, the 2026-07-22
bugcheck cluster (0x20001 ×2, 0x133), chronic 0x13A KERNEL_MODE_HEAP_CORRUPTION
through 07-19/07-06/07-02/06-26, plus a **0x116 VIDEO_TDR_ERROR** on 06-22.
That last one is a GPU/display-driver timeout bugcheck and is direct additional
evidence for the NVIDIA-driver hypothesis the resume doc already named as prime
suspect. GPU work should stay off this machine.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence.
