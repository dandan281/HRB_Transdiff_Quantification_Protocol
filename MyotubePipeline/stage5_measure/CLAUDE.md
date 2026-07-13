# Stage 5 — Measure & Store (contract)

> Read `../conventions.md` first. Write **only** into `runs/<stem>/stage5_measure/`.
> Read **only**: `runs/<stem>/stage4_qc/final_traces.txt` (+ `stage2_bright/bright_traces.txt`,
> `stage3_dim/dim_traces.txt` if those figures are requested) and `stage1_threshold/` channels.

## Responsibility
Produce the final deliverables. For **each figure** separately, measure every overlaid trace
(ImageJ Measure — length + per-channel mean) and store the ROI set and the measurements as **two
separate files**. Also store the overlay indices on their own.

## Steps (run by orchestrator)
For each figure in `final` (always), and optionally `bright`, `dim`:
1. `../common/trace_render_measure.ijm` (Fiji `-batch`) on that figure's `*_traces.txt`
   → `<fig>_rois.zip`, `<fig>_results.csv`, `<fig>_overlay_{clean,labeled,preview}.png`.
2. `store.py` validates the pair and writes `<fig>_index.csv` (id → mid_x, mid_y, length_um).

## Guarantees enforced by store.py
- `<fig>_rois.zip` and `<fig>_results.csv` are **separate files** (never merged).
- ROI count == CSV row count; `length_um ≈ length_px × 0.6493`.
- `id` aligns across the labeled overlay, the ROI names, and the CSV.

## Outputs (this folder only)
`final_rois.zip` + `final_results.csv` + `final_index.csv` + `final_overlay_*.png`
(and the same triplet for `bright_` / `dim_` if requested), `store_summary.json`.

## Must NOT
- Re-decide splits/merges (Stage 4 owns that). Read or write any other stage's folder.
