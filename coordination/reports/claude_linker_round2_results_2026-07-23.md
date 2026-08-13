# Fragment linker — round-2 results (operator export processed)

**From:** Claude annotation lane
**Date:** 2026-07-23
**Inputs:** operator export `links_active_b02.links.json` (52 fragments, all decided)
**Banked at:** `PrecisionMyotube/annotation_work/links_active/banked/`

## What the operator returned
- **14 new positive links, 116 new negatives** (usable), 5 excluded (silent/no-selection), **0 conflicts**.
- **13 merged objects** absorbing confirmed chains — exported `status="ambiguous"`
  (not certified complete; a completeness pass is still required before they can
  support length/width, same rule as round 1).
- Combined with round 1: **216 usable pairs, 41 positive / 175 negative** (positive
  class grew 27 → 41, +52%).

## Did the operator's axis heuristic help? Yes — measured on the SAME test set
LOWO out-of-fold, combined 216 pairs, old features vs the two new PCA features:

| feature set | AUC | best-F1 precision | recall @ P≥0.90 |
|---|---|---|---|
| `bridge_territory` (old) | 0.853 | 0.557 | 0.05 |
| **`bridge_axis` (new)** | **0.895** | **0.643** | **0.15** |

The heuristic adds **+0.042 AUC** and **triples** the fraction of true joins that
are auto-mergeable at high precision (0.05 → 0.15). In the final fit on all 216
pairs, `axis_cos` is the **largest coefficient** (2.02, vs `bridge_over_bg` 1.31,
`displacement_along_axis` 1.05) — the "are the fibres parallel and offset
end-to-end?" rule is now the model's primary signal, above stain-bridging.

## Did precision clear the 0.59 "deployable" bar? Partly — be precise about it
- At the **best-F1 operating point, precision is 0.643** — above the round-1 floor
  of 0.59 and the old-feature 0.557. So as a balanced classifier it improved.
- For **confident auto-merge** (where a wrong join fuses two real myotubes, which
  the plan calls worse than leaving a split), you want P≥0.90. There, **recall is
  only 0.15** — the model can safely auto-join ~15% of true joins and no more.

**Honest bottom line:** the linker is now a usable **high-precision assistant** —
auto-join the small confident set (P≥0.90), route everything else to the operator
— but it is **not** yet good enough to replace human review of all joins. Round 2
clearly moved it in the right direction; it did not finish the job.

## Caveats (do not over-read these numbers)
- Only **41 positives** across 5 wells; LOWO precision is noisy, AUC is the stabler
  measure. The precision points can swing with a few pairs.
- The combined test set is **deliberately harder** than round 1: active learning
  surfaced near-boundary pairs, so absolute precision looks lower than a
  narrow-window-only test would. That is the honest distribution, not a regression.
- Single operator, proposal-conditioned, retrospective. Not consensus, not
  prospective. The persisted model is LOWO evidence, **not wired into any
  production pipeline**.

## Artifacts
- `banked/link_pairs.jsonl` (round-2 pairs), `banked/combined_pairs.jsonl` (216).
- `banked/*.merged.instances.json` — 13 merged objects, `ambiguous`.
- `banked/model/linker.joblib` + `linker_summary.json` — final `bridge_axis` model,
  hashes recorded.
- `links_manifest.json` — apply_links banking manifest.

## For the integrator
- New `link_round2`-class merged objects (13) need a completeness-review pass
  before they can train measurements.
- A third round would help: at ~15 more positives per round the confident-merge
  recall should keep climbing. But see the project's biggest-risk note — if the
  Tier-B single-myotube program is acquisition-limited, linker precision is not the
  binding constraint.
