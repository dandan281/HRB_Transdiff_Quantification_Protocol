# Stage 3 — Dim / Short Tracer (contract)

> Read `../conventions.md` first. Write **only** into `runs/<stem>/stage3_dim/`.
> Read **only**: `runs/<stem>/stage1_threshold/` and `runs/<stem>/stage2_bright/`
> (`all_traces.txt`, `bright_traces.txt`).

## Responsibility
Recover the fibers Stage 2 deliberately ignored — the **short and the dim** — without
double-counting Stage 2's bright fibers. Output lands in this stage's OWN folder (never Stage 2's).

## Steps (run by orchestrator)
1. (optional) `../common/detect.ijm` (Fiji `-batch`, **fixed/dim-boost** scaling, lower display
   max) → `dim_segments.txt`: a second detection that brightens faint fibers. Bounded by a
   timeout; if it stalls the stage proceeds without it.
2. `select_dim.py`:
   - dim candidates = `all_traces.txt` minus `bright_traces.txt`, plus genuinely-new dim-boost
     traces (overlap < 40% of existing geometry);
   - recover additional long/bright centerline candidates by skeletonizing Desmin-positive signal
     that is not already near an existing trace; this catches broad or low-contrast fibers Ridge
     Detection missed;
   - build a bright mask with a **small** dilation (default 8 px) so nearby dim fibers are not
     hidden; a candidate >60% on the bright mask is a bright fragment → written to
     `excluded_by_brightmask.txt` (a visible review layer, not silently dropped);
   - drop only true noise (below min length / min brightness); keep the rest → `dim_traces.txt`.
3. `../common/trace_render_measure.ijm` (Fiji `-batch`) on `dim_traces.txt` →
   `dim_rois.zip`, `dim_results.csv`, `dim_overlay_*.png`.

## Outputs (this folder only)
`dim_segments.txt` (if run), `dim_traces.txt`, `excluded_by_brightmask.txt`, `dim_table.csv`,
`dim_rois.zip`, `dim_results.csv`, `dim_overlay_*.png`, `*_log.txt`.

## Must NOT
- Re-trace or relabel Stage 2's bright fibers; modify Stage 2's folder; run the composite or
  final reconciliation (Stage 4).
