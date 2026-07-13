# napari-myotube

A napari plugin for the myotube pipeline. It replaces the serverless `stage4_qc/review.html`
with a real interactive GUI, and adds the bridge to **Cellpose** (the planned replacement for
Ridge-Detection tracing).

## Why
Ridge Detection emits short segments, so the whole split/merge/learning apparatus exists to
stitch them back into whole fibres — and over-segmentation still dominates (final ≈ 371 objects
vs 246 hand-traced in 32_C08). A model that traces whole fibres directly removes that failure at
the source. You already have ground truth (ImageJ ROI sets for Plates 23/26/28/32), which is
exactly what a Cellpose fine-tune needs.

## What it does
| Widget | Purpose |
|---|---|
| **Load run** | open a `runs/<stem>/` — composite image + traces as an editable Shapes layer |
| **Save traces** | write the edited layer back to a pipeline `traces.txt` |
| **Import ROIs** | load an ImageJ ROI `.zip` (e.g. your ground truth) as editable traces |
| **Export ROIs** | write traces to an ImageJ ROI `.zip` (round-trips to Fiji) |
| **Export training pair** | image + curated traces → Cellpose `<stem>.tif` + `<stem>_masks.tif` |
| **Run Cellpose** *(experimental)* | run a model, load its traces for review |

The pure-Python I/O (`_traces_io.py`, `_roi_io.py`) imports no napari and is unit-tested in the
base env. napari uses `(y, x)`; on-disk files stay in pipeline `(x, y)` order.

## Install (use a dedicated env — napari + torch are heavy)
```
conda create -n myotube python=3.11 -y
conda activate myotube
pip install -e MyotubePipeline/napari_plugin          # plugin + napari + roifile + magicgui
pip install "cellpose"                                  # optional, for tracing/training
napari                                                  # Plugins ▸ Myotube Tracer
```
`roifile` also installs into your base anaconda env for the CLIs below.

## Cellpose workflow
1. **Organize labels** — one subfolder per well with `image.tif` (fiber channel, full-res, same
   preprocessing you'll use at inference) + `rois.zip` (your *corrected* ground truth).
2. **Build the training set**
   ```
   python napari_myotube/build_training_set.py --src <organized> --out training --width 15
   ```
   → `training/<well>.tif` + `training/<well>_masks.tif`.
3. **Fine-tune**
   ```
   python napari_myotube/train_cellpose.py --dir training --epochs 300 --pretrained cyto3
   ```
   Hold out ≥ 1 plate for validation.
4. **Trace + review** — "Run Cellpose" with your model, curate, "Save traces", then measure via
   the existing Fiji `trace_render_measure.ijm` path.

## Caveats to test on your data
- **Overlap:** Cellpose labels are one-int-per-pixel, so crossing fibres can't both own a pixel.
- **Width:** masks come from dilating centrelines by a fixed `--width`; tune it, or move to an
  intensity-aware mask later.
- **Label quality:** train on corrected ROIs, not over-segmented automated output.
