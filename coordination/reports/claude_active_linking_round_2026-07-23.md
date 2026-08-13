# Active-learning linking round (round 2) — ready for the operator

**From:** Claude annotation lane
**Date:** 2026-07-23
**Page:** `PrecisionMyotube/annotation_work/links_active/links_active_b02.html`
**Manifest:** `…/links_active_b02.manifest.json`
**Operator time:** ~15 min (52 fragment cards, most-informative first; stop anytime)

## Why this round exists
Round 1 labelled every collinear pair in a narrow window (gap ≤ 40 µm, cos ≥ 0.8)
— 91 pairs, all decided. The linker on them reaches AUC ~0.80 / precision **0.59**,
not deployable: a wrong join fuses two real myotubes, worse than leaving one split.
Lifting precision needs labels where the model is currently **uncertain**, and the
narrow window has none left (measured: 0 new pairs at gap 40 / cos 0.8).

## What this round does (plan §8.D)
1. **Widens the window** to gap ≤ 80 µm, cos ≥ 0.7 → **140 new candidate pairs**
   the narrow round never offered, including partners for previously-unresolved
   fragments.
2. **Ranks by uncertainty.** Each new pair is scored by the linker trained on the
   91 labels; pairs with predicted probability nearest 0.5 are served first. The
   prediction is **never shown** — it only orders the queue, so the operator's
   judgement stays independent (same principle as the blind-repeat tool).

## The linker used for ranking
- Feature set chosen by leave-one-well-out AUC over 5 candidate sets: **`bridge_territory`** (`bridge_over_bg` + `territory_frac`), **AUC 0.804**.
- This reproduces round 1's finding: `bridge_over_bg` alone AUC 0.801 (dominant coef 1.13), and adding geometry *lowers* it (all-features 0.788) — the operator's own "is stain crossing the gap?" cue carries the signal, not collinearity.
- Trained on 91 pairs (28 positive / 63 negative).

## The round, in numbers
| | |
|---|---|
| new candidate pairs (wide window) | 140 |
| served | **135** (uncertainty 0.18–0.95, median 0.33) |
| dropped as least-uncertain (budget 160) | 0 |
| dropped as 6th+ candidate on a fragment (cap 5) | 5 |
| fragment cards | 52 |
| per well (fragments → new pairs) | 19_B06 5→11 · 22_B03 23→42 · 29_C05 13→37 · 32_C08 19→35 · 33_C09 5→15 |
| ctrl well 23_B02 | 0 (no `fragment_too_short` there) |

Most-uncertain served pairs sit at proba 0.47–0.55 (e.g. `22_B03 myotube_0092 →
myotube_0135`, gap 46.8 µm) — exactly the cases a label resolves best.

## How to run it
1. Open the HTML in a browser. Same instrument as before: one fragment at a time,
   `L` hides outlines to see bare Desmin, letter keys toggle a link, `N` no-join,
   `U` unsure, Enter advances. Cards are ordered most-informative first.
2. Export (downloads `links_active_b02.links.json`) — same `fragment_links.v1`
   schema as round 1, so `apply_links` consumes it unchanged.
3. Hand the export back; I fold it into the training set and re-measure precision.

## Honest limitations
- The linker is trained on 91 narrow-window pairs and **extrapolates** to wider
  gaps, so its uncertainty ordering is itself provisional — good enough to
  prioritise, not a guarantee the hardest cases are exactly these.
- Candidates only exist within the widened window; a true join beyond 80 µm still
  cannot be expressed. 2 of 65 fragments have no partner even now.
- Round-1 partners are excluded (no redundant re-labelling), so a fragment's
  already-known link is not re-shown for context.
- Single operator, proposal-conditioned. Not consensus, not inter-rater agreement.

## Addendum — operator direction heuristic added as features (later 2026-07-23)
The operator taught a rule while annotating: two fragments are the same broken
fibre only if their long axes are parallel **and** the offset between them runs
*along* that axis (end-to-end), not across it (side-by-side parallel neighbour).
Encoded as two whole-object PCA features in `link_features.py`:
- `axis_cos` — |cos| between the two fragments' principal axes;
- `displacement_along_axis` — |cos| between the centroid offset and the fibre axis.

These are the robust version of `min_cos`, whose endpoint tangents come from a
12-px local patch and are noisy on a frayed end. On the 91 labelled pairs they
separate joins from non-joins more sharply than `min_cos` (Δmean +0.14 / +0.10 vs
+0.03), and **LOWO AUC rose 0.804 → 0.839** (`bridge_axis` = bridge_over_bg +
axis_cos + displacement_along_axis). This is the model that will fold in the
operator's export.

The **live page was not regenerated** — the operator was mid-session (27/52) and
card order is the only thing that would change; their decisions are preserved by
uid. The improved model is applied when the export is processed, not to this
round's ordering.

## Build (this session)
- `annotation_tools/qc_review/link_features.py` — intensity/territory + axis features.
- `annotation_tools/qc_review/link_model.py` — recompute features for the 91 pairs
  (endpoints + PCA geometry re-derived), LOWO feature selection, fit, uncertainty.
- `annotation_tools/qc_review/link_active.py` + `build-active-links` CLI.
- 15 new tests; **188 pass total** (was 173).
