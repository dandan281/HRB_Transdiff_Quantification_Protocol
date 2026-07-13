# Pipeline conventions (shared contract for all stages)

This file is the single source of truth for the data formats that cross stage
boundaries. Every stage script and every `CLAUDE.md` defers to this file. If a
format changes, change it here first.

## Coordinate system & units
- Images are `3636 x 3636` px, 16-bit, 3 channels. Pixel size `UM = 2360.74 / 3636 = 0.649267 um/px`.
- All coordinates are pixel coordinates in the full-resolution image, origin top-left,
  `x` = column, `y` = row.

## Channels (resolved per image, never hardcoded for scaling)
- Roles: `primary` (fiber / trace target), `overlap` (cross-ref marker), `dapi` (nuclei).
- For `32_C08_br223_igf1r`: primary=ch1, overlap=ch0, dapi=ch2.
- Roles are decided by `stage1_threshold/threshold.py` and written to `metadata.json`.
- Channel TIFFs are named `ch{N}_raw16.tif` where N is the **0-based** channel index.

## metadata.json (written by Stage 1, read-only for all later stages)
The single source of truth for channel roles AND display scaling.
```json
{
  "stem": "32_C08_br223_igf1r",
  "src_nd2": "C:/.../32_C08_br223_igf1r.nd2",
  "pixel_um": 0.649267,
  "width": 3636, "height": 3636,
  "channels": {"primary": 1, "overlap": 0, "dapi": 2},
  "channel_scores": {"0": {"nuclei": 302, "fiber": 330170, "p975": 4260.0}, "...": {}},
  "display": {"primary_min": 0, "primary_max": 4231, "method": "p975+bgsub"},
  "bc_candidates": [{"max": 3149, "metric": 0.41}, {"max": 4231, "metric": 0.55}],
  "created_by": "stage1_threshold"
}
```

## traces.txt  (the canonical polyline format passed between Python stages)
- One trace per line. A line is a flat CSV of polyline vertices:
  `x0,y0,x1,y1,x2,y2,...` (floats, 2 decimals).
- A valid trace has >= 2 vertices (>= 4 numbers).
- This is the SAME format as the old `segments.txt` / `merged.txt`, kept for reuse.
- Stage-specific filenames: `bright_traces.txt`, `dim_traces.txt`, `final_traces.txt`.
- Detection raw output is `*_segments.txt`; merged/selected output is `*_traces.txt`.

## signal.png  (8-bit fiber-signal map, used by merge + flag logic)
- Background-subtracted primary channel scaled `0..display.primary_max` -> 8-bit.
- Written by Stage 1 (`adjust_primary.ijm`) at the chosen display max. Used by `merge.py`
  (continuity gate), the Stage 2/3 brightness proxy, and `stage4_qc/flag.py` (dark-gap detection).
- `FIBER_T = 15` (8-bit, in `common/signalmap.py`) is the canonical "fiber present" floor; Stage 3
  `--min-bright` defaults to the same value so the stages agree on "is there fiber here".
- `stage3_dim/excluded_by_brightmask.txt` (dim fibres hidden by the bright mask) is **read by
  Stage 4** (`flag.py`), which emits an `occluded` review case per trace (default: drop).

## ROI sets (.zip) and results (.csv) — produced only by Fiji macros
- ROI `.zip` = ImageJ RoiManager save; ROI names are zero-padded spatial ids (`001`, `002`, ...).
- `results.csv` columns (Stage 5 canonical):
  `id,mid_x,mid_y,length_px,length_um,primary_mean,overlap_mean,dapi_mean`
- **Per figure, the ROI `.zip` and the `results.csv` are SEPARATE files** (never merged),
  and `id` aligns with the number drawn on the overlay PNG.

## Spatial numbering (deterministic, matches the old pipeline)
- Sort kept traces by `mid_y * 4000 + mid_x` ascending (top->bottom, then left->right).
- Ids are assigned 1..N after sorting. The same order is used for labels, ROI names, and CSV.

## Run directory layout (one per image)
```
runs/<stem>/
  stage1_threshold/  ch0_raw16.tif ch1_raw16.tif ch2_raw16.tif
                     ch{primary}_adjusted8.tif  metadata.json  bc_contactsheet.png  signal.png
  stage2_bright/     bright_segments.txt bright_traces.txt bright_rois.zip bright_overlay*.png bright_table.csv
  stage3_dim/        dim_segments.txt dim_traces.txt dim_rois.zip dim_overlay*.png dim_table.csv
                     excluded_by_brightmask.txt
  stage4_qc/         <stem>_composite.png  flags.json  review.html  decisions.json
                     final_traces.txt  final_rois.zip  final_overlay*.png
  stage5_measure/    final_rois.zip final_results.csv  bright_rois.zip bright_results.csv
                     dim_rois.zip dim_results.csv  (one ROI.zip + one CSV per figure)
  run.log            append-only log of every stage invocation (params, counts, file hashes)
```

## Isolation rule (enforced by orchestrator.py, documented in each CLAUDE.md)
- A stage may **read** only: its own folder + the explicitly named upstream outputs.
- A stage may **write** only inside its own `runs/<stem>/stageN_*/` folder.
- The orchestrator passes every path explicitly via `--in/--out`; no stage discovers
  files by globbing the run tree. This is the anti-hallucination guarantee.

## Fiji invocation
- Always `fiji-windows-x64.exe -batch <macro.ijm> "<k=v;k=v;...>"`.
- **Never `--headless`** — Ridge Detection needs AWT and will stall/blank headless.
- Macros parse args with the shared `arg(k)` helper (`key=value;` pairs).
- Fiji path is read from `MyotubePipeline/common/config.json` (`fiji_exe`).
</content>
