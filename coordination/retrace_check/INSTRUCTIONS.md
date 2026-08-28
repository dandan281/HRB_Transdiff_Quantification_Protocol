# Re-trace check (defines what 'perfect' means)

1. Open `D04_retrace_rows900-2100_cols900-2100.tif` in Fiji.
2. Trace every myotube >= 50 um (~77 px here), exactly as you normally do:
   freehand/segmented line, ONE line per fibre, full visible extent.
3. Do NOT look at your original D04 ROIs first - the point is blind repeatability.
4. Add each trace to the ROI Manager (press T), then
   ROI Manager > More > Save... -> save as `D04_retrace_ROIs.zip` in this folder.
5. Tell Claude when the zip is there. Expected effort: ~45-60 min for ~65 fibres.

Nothing else is needed - coordinates are translated back automatically.
