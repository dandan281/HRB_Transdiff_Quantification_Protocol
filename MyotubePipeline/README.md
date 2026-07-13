# MyotubePipeline — deterministic staged myotube quantification with human-in-the-loop QC

Replaces the old Ridge-Detection + greedy-merge pipeline, which kept **splitting one myotube into
several** and **merging separate myotubes across dark gaps**. Here detection is deterministic and
isolated per stage, one threshold is the single source of truth, and every ambiguous one-vs-many
call is escalated to a **human review page** (Stage 4 — the heart of the system).

See `PLAN.md` (approved plan) and `conventions.md` (the cross-stage data contract). Each
`stageN_*/CLAUDE.md` states that stage's read/write boundary.

## The five stages
1. **stage1_threshold** — extract channels, resolve channel roles per image, choose the display
   brightness/contrast on the fiber channel. Writes `metadata.json` (the single source of truth)
   + `signal.png` + `ch{primary}_adjusted8.tif`.
2. **stage2_bright** — detect (shared Ridge pass) + keep the **long, bright** fibers.
3. **stage3_dim** — keep the **short / dim** fibers Stage 2 left, without double-counting; a small
   bright-mask dilation + an `excluded_by_brightmask` review layer.
4. **stage4_qc** — composite at the Stage-1 threshold; flag **splits** (dark-gap) and **merges**
   (collinear + continuous signal); `review.html` for the human; `reconcile.py` applies decisions.
5. **stage5_measure** — per figure, ROI `.zip` + `results.csv` as **separate files** + index.

## Learning loop (adapts to your reviews)
The system draws all the traces; you review only the **ambiguous** cases. Each review is logged as
`(features → your decision)` and a small **scikit-learn** model per case type (split/merge/occluded)
learns your pattern, then **pre-sets each case's default** on the next run — so over time you confirm
rather than correct. Classical ML only (interpretable logistic regression), **no pytorch**. It only
sets review *defaults* (never auto-applies an edit), and stays inert until ≥12 of your decisions of a
type accumulate. See `learning/CLAUDE.md`. Disable with `--no-learn`.

## Run it
```
# fresh run -> stops at the review gate
python orchestrator.py --nd2 "C:/Users/liqig/Documents/HRB_Transdiff/Plate23/PLATE_23/32_C08_br223_igf1r.nd2"

# open runs/32_C08_br223_igf1r/stage4_qc/review.html, curate, save decisions.json there, then:
python orchestrator.py --resume 32_C08_br223_igf1r

# or run straight through with auto/default decisions (no human pause):
python orchestrator.py --nd2 "<...>.nd2" --auto
```
Outputs land under `runs/<stem>/`; deliverables in `runs/<stem>/stage5_measure/`.

## Requirements
- Fiji at the path in `common/config.json` (invoked with **`-batch`**, never `--headless` —
  Ridge Detection needs AWT).
- Python (anaconda3) with numpy, scipy, scikit-image, tifffile, Pillow.

## Notes
- Channel roles are detected per image and recorded; `--force-primary N` / `--force-max V` override.
- Ground truth (the hand overlays) is **guidance, not gospel**: the goal is coherent biological
  agreement (no split intact fibers, no merged distinct fibers), not pixel-exact reproduction.
- Calibration set: `32_C08`, `P23_B02`, `P23_B06`, `P23_C09` (see `common/config.json`).
