# Stage 2 — Bright / Long Tracer (contract)

> Read `../conventions.md` first. Write **only** into `runs/<stem>/stage2_bright/`.
> Read **only**: `runs/<stem>/stage1_threshold/` (ch TIFFs, `metadata.json`, `signal.png`).

## Responsibility
Find the **obvious** myotubes — the long, bright fibers — and nothing else. Short/dim fibers are
Stage 3's job. This stage also runs the single shared Ridge detection and exposes the full merged
trace set (`all_traces.txt`) for Stage 3.

## Steps (run by orchestrator)
1. `../common/detect.ijm` (Fiji `-batch`, standard regime) reads stage1 `ch{primary}_raw16.tif`
   → `bright_segments.txt` (raw ridge centrelines).
2. `select_bright.py` merges segments (signal-gated, via `../common/merge.py`) →
   `all_traces.txt`, then keeps traces that are **long AND bright** (length ≥ l_long, signal
   brightness ≥ floor; adaptive defaults, overridable) → `bright_traces.txt` (spatially ordered).
3. `../common/trace_render_measure.ijm` (Fiji `-batch`) on `bright_traces.txt` →
   `bright_rois.zip`, `bright_results.csv`, `bright_overlay_{clean,labeled,preview}.png`.

## Outputs (this folder only)
`bright_segments.txt`, `all_traces.txt`, `bright_traces.txt`, `bright_table.csv`,
`bright_rois.zip`, `bright_results.csv`, `bright_overlay_*.png`, `*_log.txt`.

## Must NOT
- Touch dim/short fibers, run the composite, or make split/merge decisions (Stages 3–4).
- Drop edge fibers (kept; Stage 4 decides). Read/write any other stage's folder.
