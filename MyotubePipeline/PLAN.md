# Plan: Deterministic Staged Myotube Pipeline with Human-in-the-Loop QC (HRB_Transdiff)

## Context

The previous Claude/Codex pipeline (Ridge Detection + greedy collinear merge, two-tier outputs)
produced two scientifically fatal errors:

1. **Over-segmentation** — one intact myotube split into 2, 3, 4+ separate labeled traces.
2. **Under-segmentation** — several distinct fibers lying together (often with a dark gap) merged
   into one "intact" trace.

It also drifted per-plate and gave inconsistent counts. We are rebuilding as a **deterministic,
staged pipeline with a human-in-the-loop QC gate** — *not* a system of independent LLM "agents".
Each stage is a reproducible Fiji macro / Python script with explicit input/output paths; the
**orchestrator owns all control flow and logging**, so no stage silently reads another's internals
and the LLM never makes invisible biology decisions. The decisive trustworthiness comes from
**Stage 4 (QC review)** — detection will never be perfect, so the review interface is the heart of
the system, not an afterthought.

**Locked decisions (from user) + revisions:**
- Trace the **fiber channel**. For 32_C08: ch1 primary, ch0 overlap, ch2 DAPI. **Do not hardcode**
  for scaling — every image runs a channel-role check and records the resolved roles in metadata.
- Stages = **deterministic modules**, isolated folders + `CLAUDE.md` contracts, but the real
  guarantees live in `orchestrator.py` (explicit paths, no hidden cross-reading, reproducible logs).
- Ambiguous one-vs-many cases go to a **rich HTML review** (not just "type a number").
- **Calibrate/validate on multiple Plate23 images** — B02, B06, C09, and 32_C08 — to avoid
  overfitting one visual texture. Ground truth is **guidance, not gospel**: the target is *coherent
  biological agreement*, not exact reproduction of every human curve/count.

**Environment (verified):** Fiji 2.17 at `C:\Users\liqig\Desktop\Fiji\fiji-windows-x64.exe`
(Bio-Formats 8.5 reads `.nd2`); Python 3.13 (anaconda3) with numpy/scipy/scikit-image/tifffile/
Pillow. Pixel size 0.6493 µm/px. **Fiji is invoked with `-batch` (NOT `--headless`)** because
Ridge Detection needs AWT — matching the prior environment notes.

---

## Architecture

New root: `HRB_Transdiff/MyotubePipeline/`

```
MyotubePipeline/
  orchestrator.py            # OWNS control flow, path wiring, logging, the review gate
  common/                    # roi io, spatial numbering, length/measure, channel-role check
  stage1_threshold/  CLAUDE.md + threshold.ijm / threshold.py
  stage2_bright/     CLAUDE.md + detect_bright.ijm / .py
  stage3_dim/        CLAUDE.md + detect_dim.ijm / .py
  stage4_qc/         CLAUDE.md + composite.ijm + flag.py + build_review_html.py + reconcile.py
  stage5_measure/    CLAUDE.md + measure.ijm
  calibration/               # B02, B06, C09, 32_C08 reference set + tuned params.json
  runs/<image_stem>/         # ALL outputs here, one isolated subfolder per stage:
      stage1_threshold/ ch0_raw16.tif ch1_raw16.tif ch2_raw16.tif
                        ch1_adjusted8.tif  metadata.json (channel roles + B&C)  bc_contactsheet.png
      stage2_bright/    bright_rois.zip  bright_overlay.png  bright_table.csv
      stage3_dim/       dim_rois.zip  dim_overlay.png  dim_table.csv  excluded_by_brightmask.zip
      stage4_qc/        composite.png  review.html  flags.json  decisions.json
                        final_rois.zip  final_overlay.png
      stage5_measure/   <fig>_rois.zip  +  <fig>_results.csv  (separate file per figure)
```

**Guarantees enforced by `orchestrator.py` (the anti-hallucination layer):** every stage is called
with explicit `--in`/`--out` paths; a stage may read only its declared inputs (its own folder +
the named upstream outputs) and write only its own `runs/<stem>/stageN_*/` folder. Each run writes
a `run.log` (params, file hashes, counts) for reproducibility. `metadata.json` from Stage 1 is the
single source of truth for **channel roles + display scaling**, reused by Stages 2/3 (render) and 4
(composite). `CLAUDE.md` files document the same contracts but are not the enforcement mechanism.

**Reuse existing code:** `02_extract_channels.ijm` (ND2→TIFF), `analyze_channels.py` (channel ID →
feeds the recorded role check), `10_export_segments.ijm` (Ridge Detection), `merge_segments.py` /
`merge_b06.py` (collinear merge + signal-continuity gate), `audit_b06.py` (dark-gap / empty-bridge
detection — core of the under-segmentation flag), `gen_composite.ijm`, `gen_render.ijm`,
`auto_thresholds.py` / `calib_brightness_rule.py`.

---

## Stage specs

### Stage 1 — Threshold Builder
- **In:** `.nd2`. **Out:** `stage1_threshold/`.
- Extract ch0/ch1/ch2 (16-bit TIFF). **Run channel-role check** (`analyze_channels.py`: nuclei vs
  fiber content) and **record resolved roles in `metadata.json`** — never assume ch1 for new images.
- On the fiber channel: rolling-ball background subtraction + choose display min/max for maximum
  fiber visibility; render a **B&C contact sheet** of candidate maxima and pick by a fiber-visibility
  metric. Write `ch1_adjusted8.tif` (the shared "duplicate"), `metadata.json` (roles + B&C +
  pixel_um), and optionally surface the contact sheet for a one-click user confirm.

### Stage 2 — Bright / Long Tracer
- **In:** `ch1_adjusted8.tif` + `metadata.json`. **Out:** `stage2_bright/`.
- Ridge Detection tuned for **strong** fibers; keep only **long AND bright** traces. Spatial
  numbering (top→bottom, left→right). Write `bright_rois.zip`, overlay, table.

### Stage 3 — Dim / Short Tracer
- **In:** `ch1_adjusted8.tif` + `metadata.json` + `stage2_bright/bright_rois.zip`. **Out:** `stage3_dim/`.
- Lower thresholds to catch **dim/short** fibers. Mask out Stage-2 fibers with a **small dilation
  radius** (avoid hiding nearby dim myotubes) and **preserve `excluded_by_brightmask.zip`** as a
  visible review layer so the QC step can see what the mask suppressed. Write `dim_rois.zip`,
  overlay, table.

### Stage 4 — Composite QC + Interactive Review  ← the heart of the system
- **In:** `metadata.json`, ch0/ch1/ch2 TIFFs, `bright_rois.zip`, `dim_rois.zip`,
  `excluded_by_brightmask.zip`. **Out:** `stage4_qc/`.
- Build a **3-channel composite** (R=overlap, G=fiber, B=DAPI) at the **same fiber threshold** from
  Stage 1 (reuse `gen_composite.ijm`).
- `flag.py` runs two detectors and proposes concrete edits (`flags.json`):
  - **Over-segmentation:** collinear, end-to-end-adjacent traces with continuous fiber signal in the
    gap → propose **merge candidates** with the join point.
  - **Under-segmentation:** a trace crossing a dark gap / fiber break (reuse `audit_b06.py`) →
    propose **candidate split points** (the gap midpoints).
  - Also surface bright-mask-excluded fibers near dim traces for a "missed?" check.
- **Confidence gate:** clear cases auto-resolved (logged); uncertain cases go to `review.html`.
- **`review.html` UI (rich, not just a number):** per case, a zoomed crop (fiber + composite side by
  side) with current labels and **proposed split/merge markers**, plus actions:
  **Keep as one · Split at proposed gaps (toggle each point) · Merge selected candidates ·
  Reject trace (noise) · Override with a free-text note**. Submitting writes `decisions.json`.
- `reconcile.py` applies decisions deterministically → `final_rois.zip`, `final_overlay.png`.

### Stage 5 — Measure & Store
- **In:** `stage4_qc/final_rois.zip` (+ optional bright-only / dim-only figures). **Out:** `stage5_measure/`.
- Replicate ImageJ **Measure ("M")** via `roiManager("Measure")` (Length + Mean); ids match overlay
  numbers. **Two separate files per figure:** `<fig>_rois.zip` and `<fig>_results.csv`
  (id, mid_x, mid_y, length_px, length_um, fiber_mean, dapi_mean, overlap_mean). Never mixed.

---

## Orchestration & the loop/gate

`orchestrator.py` runs Stages 1→2→3 and Stage 4 **up to writing `review.html`**, then **pauses**
(prints the path). After the user submits, `--resume <stem>` runs `reconcile.py` then Stage 5. If
`flags.json` has zero uncertain cases it auto-continues with no pause. Every step appends to
`run.log`.

---

## Calibration & Validation (multi-image, ground-truth as guidance)

1. Build and tune on **four Plate23 references**: `32_C08`, plus `B02`, `B06`, `C09`
   (`Plate23/PLATE_23/P23_B02|B06|C09_withHighlight.png` exist as overlays). Store tuned params in
   `calibration/params.json`; require the params to work acceptably across **all four**, not just one.
2. Success = **coherent biological agreement** (no intact fiber split; no distinct fibers merged;
   counts in a sane range), *not* pixel-exact reproduction of the human overlays.
3. Confirm Stage 5 emits one ROI `.zip` + one results `.csv` per figure, indices aligned to overlays.

## Verification (after implementation)
- `python MyotubePipeline/orchestrator.py --nd2 "Plate23/PLATE_23/32_C08_br223_igf1r.nd2"` runs
  through the review gate; open the printed `runs/.../stage4_qc/review.html`, exercise
  split/merge/reject/note actions.
- `python MyotubePipeline/orchestrator.py --resume 32_C08_br223_igf1r` → `final_*` + `stage5_measure/*`.
- Fiji invocation pattern: `fiji-windows-x64.exe -batch <stageN>.ijm "<args>"` (**`-batch`, never
  `--headless`** — Ridge Detection needs AWT).
- Re-run on B02/B06/C09 and eyeball against their `withHighlight.png` for the two failure modes.
- Sanity: `length_um ≈ length_px * 0.6493`; ROI count == CSV row count per figure.

## Open items to settle during build
- Exact bright/long floors, dilation radius, and gap thresholds — tuned across the 4 calibration
  images (seed from prior values: display max ~4231, minLen ~170 px, fiber floor ~3900).
- Whether bright-only and dim-only sets are measured as separate figures or only the reconciled final.
