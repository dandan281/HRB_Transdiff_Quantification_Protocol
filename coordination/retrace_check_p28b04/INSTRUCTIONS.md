# Re-trace check, PLATE_28 B04 — settles the 1.78× question

The tracer reports ~2.4× your annotated length on this well. This blind
re-trace decides between the two explanations: your original ROI pass was
selective (then the extra tracer length is real fibre), or the tracer
over-traces (then it's a fault). Same protocol as the D04 check.

1. Open `B04_retrace_rows1218-2418_cols1218-2418.tif` in Fiji.
2. Trace **every** myotube ≥ 50 µm (~77 px here) — exhaustively, not just
   the clear ones: freehand/segmented line, ONE line per fibre, full
   visible extent.
3. Do NOT look at your original B04 ROIs first — blind repeatability is
   the point.
4. Add each trace to the ROI Manager (press T), then
   ROI Manager > More > Save... → `B04_retrace_ROIs.zip` in this folder.
5. Tell Claude when the zip is there.

How it will be read (predeclared): your fresh pass, your original ROIs
(window-clipped), and the tracer are all scored against each other.
If fresh-you also finds ~1.8–2.4× the original's length, the annotation
was selective and the tracer stands; if fresh-you lands near 1.0×, the
tracer over-traces on this plate. Window chosen as the exact center of
the field, declared before anyone looked at what it contains.
