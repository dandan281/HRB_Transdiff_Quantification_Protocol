# Stage 4 — Composite QC + Interactive Review (the heart)

> Read `../conventions.md` first. Write **only** into `runs/<stem>/stage4_qc/`.
> Read **only**: `runs/<stem>/stage1_threshold/`, `stage2_bright/`, `stage3_dim/`.

## Why this stage matters most
Detection is never perfect. The two failure modes that ruined the old pipeline both get caught
here, and where the evidence is ambiguous the **human decides** — that is what makes the numbers
trustworthy. This stage makes no silent biological calls beyond clearly auto-confident ones.

## Steps (run by orchestrator)
1. `composite.ijm` (Fiji `-batch`) → `<stem>_composite.png` using the **same primary display max
   from Stage 1** (R=overlap, G=fiber, B=DAPI). Crops are cut from this.
2. `flag.py` combines bright + dim + bright-mask-excluded → `combined_traces.txt`, then proposes:
   - **SPLIT** (under-segmentation): trace crossing an internal **dark** gap (relative to the
     fibre's own brightness) → cut at the gap centre(s); or a sharp **kink** (review-only);
   - **MERGE** (over-segmentation): two traces collinear + endpoints close + **continuous signal**
     between → join into one;
   - **OCCLUDED**: a dim fibre Stage 3 hid behind the bright mask → human can **restore** it
     (default: drop).
   Each case gets a zoomed crop (`crops/*.png`) and a confidence. Only ultra-confident merges and
   dark-gap splits are marked `auto`. → `flags.json`.
3. `build_review_html.py` → `review.html` (serverless): per case, keep / split (toggle each point) /
   merge / separate / reject + a note; "Accept all proposals" and "Reject all edits" buttons;
   produces a downloadable `decisions.json`.
4. **GATE**: orchestrator pauses. Human opens `review.html`, curates, saves `decisions.json` here,
   then re-runs with `--resume`. (If `flags.json` has zero review cases, no pause.)
5. `reconcile.py` applies decisions (or auto defaults) → `final_traces.txt` + `reconcile_summary.json`.
6. `../common/trace_render_measure.ijm` renders `final_traces.txt` → `final_rois.zip`,
   `final_results.csv`, `final_overlay_*.png`.

## Outputs (this folder only)
`<stem>_composite*.png`, `combined_traces.txt`, `crops/*.png`, `flags.json`, `review.html`,
`decisions.json` (saved by the human), `final_traces.txt`, `reconcile_summary.json`,
`final_rois.zip`, `final_results.csv`, `final_overlay_*.png`.

## Must NOT
- Re-detect fibers (Stages 2–3 own detection). Invent split/merge decisions the human rejected.
- Read or write any other stage's folder.
