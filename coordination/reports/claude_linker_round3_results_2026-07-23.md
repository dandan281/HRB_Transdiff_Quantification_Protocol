# Fragment linker — round-3 results (plateau reached)

**From:** Claude annotation lane
**Date:** 2026-07-23
**Input:** operator export `links_active_b03.links.json` (29 fragments, all decided)
**Banked at:** `PrecisionMyotube/annotation_work/links_active_r3/banked/`

## What round 3 returned
- **6 positives, 29 negatives** (usable), 6 merged objects, **0 conflicts**, 2 `unsure`.
- Combined all three rounds: **251 usable pairs, 47 positive / 204 negative**.
- Round 3 was a **focused** round by design (40 uncertain-ranked pairs, not a padded
  150) because the informative pool had collapsed — see the round-2 results note.

## Round 3 gave a marginal gain — the model has converged on this fragment set
`bridge_axis`, LOWO out-of-fold:

| training set | AUC | best-F1 precision | recall @ P≥0.90 |
|---|---|---|---|
| r1+r2 (226 pairs) | 0.899 | 0.655 | 0.15 |
| **r1+r2+r3 (251 pairs)** | **0.902** | **0.683** | 0.15 |

- AUC **+0.003**, best-F1 precision **+0.028**, confident-merge recall **unchanged**.
- Contrast with round 2, which moved AUC 0.853 → 0.895 and precision 0.557 → 0.643
  on the same test set. **Round 3's return is an order of magnitude smaller** — the
  diminishing-returns tail, exactly as predicted before running it.
- `axis_cos` remains the top coefficient (2.05), then `bridge_over_bg` (1.44),
  `displacement_along_axis` (1.06).

## Round 3 was not wasted, though
Widening `cos_min` to 0.6 recovered **6 real joins the earlier rounds' geometry
excluded**, via two mechanisms:
- **long-gap** joins (4 pairs at gap 90–118 µm), beyond the round-2 gap-80 window;
- **bent short-gap** joins (2 pairs: gap 2.7 µm/cos 0.61 and 16.3 µm/cos 0.638) that
  were near the fragment all along but excluded by the stricter cos≥0.7–0.8 of
  earlier rounds. The candidate generator was too strict on collinearity for bent
  fibres; the axis features let a looser window be used safely.

## Deployability (unchanged conclusion)
Best-F1 precision **0.683** is comfortably above the old 0.59 "not deployable"
floor, but confident auto-merge (P≥0.90) is still only **15% recall**. The linker
is a solid **high-precision assistant** — auto-join the confident minority, route
the rest to review — not a blind auto-merger. Three rounds did not change that
ceiling; the last one barely moved it.

## Recommendation: stop rounds on these 65 fragments
Round 3 confirmed the plateau. Further rounds on the **same** `fragment_too_short`
set will return progressively less. To lift the linker further, the lever is **more
fragments** (re-triage additional `ambiguous` proposals into `fragment_too_short`,
then link those), not more passes over the current ones. And the standing
biggest-risk caveat holds: if Tier-B single-myotube measurement is
acquisition-limited (Desmin signal gaps, no external ground truth), linker
precision is not the binding constraint.

## Artifacts
- `banked/link_pairs.jsonl` (round 3), `banked/combined_pairs_r123.jsonl` (266 rows).
- `banked/*.merged.instances.json` — 6 merged objects, `status="ambiguous"`.
- `banked/model/linker.joblib` + `linker_summary.json` — final `bridge_axis` model
  on 251 pairs, hashes recorded.

Single operator, proposal-conditioned, LOWO development evidence. Not consensus,
not prospective, not wired into any production pipeline.
