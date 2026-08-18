# Dense corpus + first Omnipose training run — session report

Date: 2026-08-17
Lane: Claude (model laboratories / annotation tooling)
Status: **training run 239017 in flight on UW Tillicum; no held-out metric yet**

---

## 1. What changed today, in one line

The T02 blocker moved from "no second candidate" to "a candidate is training on a
corpus 13x larger than the one the project has been arguing about", because the
operator's existing Fiji tracing turned out to be far denser than `bootstrap_v1`
and had never been imported.

## 2. Sequence

| step | outcome |
|---|---|
| Zero-shot Omnipose (plan step 1) | **failed**, and cleanly |
| Found 23 Fiji ROI sets already in the repo | PLATE_32 complete: 10 wells, ROIs + Results |
| Built ROI reader + plate importer | 5,233 instances imported from PLATE_32 |
| Built training corpus | `plate32_dense_v1`, 5,004 instances |
| Window tiling + Omnipose links | 144 tiles, 6,771 whole fibres, 13,619 links |
| Smoke run (klone, 5 epochs) | proved `train_links` is accepted AND used |
| Full run (Tillicum H200, 300 epochs) | running, ~10 s/epoch, ~50 min |

### 2.1 Zero-shot result (negative, and worth keeping)

12 cells: 3 checkpoints x 2 wells x 2 polarities, on 1024 px crops.

- Every configuration returned objects of **aspect ratio 1.8–2.7**; the reviewed
  GT is **38**. It finds ovals, not fibres.
- `bact_fluor_omni`, `bact_phase_affinity` and `worm_omni` all behave the same,
  so this is not a polarity or checkpoint-choice artifact.
- Mechanism: every shipped model reports `diam_mean=30.0` and we ran with no
  rescaling, so each is primed for compact ~30 px objects.
- **The scale axis was never run** (`--diameters`), so the negative is not yet
  airtight. It is the one cheap thing still owed on this question.

Consequence for the fine-tune: pretrained weights alone are useless here, so any
success after fine-tuning is attributable to the fine-tuning. That closes the
attribution gap the design otherwise had — there is no initialisation ablation.

### 2.2 The corpus

`PrecisionMyotube/annotation_work/plate32_dense_v1/` — 10 wells, 5,004 instances.

- Source: the operator's own Fiji ROI sets, freehand **centrelines** (Results.csv
  shows Area/Length implying 1 px width), rasterised to ribbons and snapped to
  local signal.
- Width validates against an independent reference: imported P32 B02 gives width
  median **5.81 µm** / aspect **38.1**, against **5.81 µm** / **38.4** for the
  human-certified `bootstrap_v1`. The snap reproduces the human convention
  rather than inventing one.
- Scale: **13.3×** the 375-instance corpus. Target area per field rose from
  1.1–2.6% to 6–14%.

---

## 3. Bugs found (project code)

1. **~8% of every field was being taught as background.** Labelled area is
   1.409% of a field; the ridge detector calls ~10% fibre-like. `ambiguous`
   (1.7%) was correctly ignored; the rest was not. Omnipose's background target
   is "distance 0, flow 0", so those pixels actively taught suppression of real
   myotubes. Fixed by ignoring unlabelled fibre-like territory, with a 6 px halo
   kept as background so the edge signal survives.
2. **`train_links=[None] * len(images)`** in `train_fold.py`, commented "each
   reviewed instance is one connected object". True of the sparse corpus, false
   here: **59.5%** of dense fibres are broken by a crossing. Passing None would
   have taught a false split at every crossing — the exact error
   `false_split_count` measures. Now passes 13,619 links.
3. **Per-instance tiling destroys dense data.** Measured **2.65 fibres painted
   out for every 1 kept**. It was designed for 35 fibres/field where clipping was
   rare. Replaced with 1280 px overlapping windows: ratio falls to **0.72**, and
   a tile now looks like the crowded field seen at inference.
4. **`TILE_PX = 1024` too small.** Sized to the old corpus's 965 px maximum; the
   dense corpus reaches 1534 px and 16 instances would have hit the
   "ceiling too small" assertion mid-build. Raised to 1792, with a pre-flight
   check that reports the value needed.
5. **`--nuclei-ch 2` default is wrong on PLATE_44** (ch0/ch2 swapped). Would have
   segmented the receptor channel as nuclei while still producing a plausible
   Desmin mask — nothing in the output would look wrong.
6. **Rasteriser was O(field) per trace** — two full-field distance transforms
   each, so the plate import never finished. Cropped to each trace's bbox. The
   same bug was then found in the colour overlay.
7. **Snap threshold degenerated.** A fixed percentile over a search band that is
   deliberately mostly background lands in the background mode, returning masks
   up to 2.2× too wide with no visible symptom. Caught by a unit test, replaced
   with Otsu plus three explicit refusals.
8. Handoff doc's own verify command does not work: `pytest model_labs/tests
   annotation_tools/tests PrecisionMyotube -q` yields 342 passed / 154 errors.
   Only the DEVELOPMENT_PLAN §13 two-invocation form reproduces 496.

## 4. Mistakes I made

Recorded because three cost real time and one cost money.

1. **Said preemption was survivable. It was not.** I saw per-epoch checkpoints
   being written and concluded a preemption costs ~12 s. Nothing *reads* them —
   `train_fold.py` has no resume path — so `--requeue` restarts at epoch 0. Four
   dead attempts, ~45 min. I had read and summarised this exact failure from the
   project's own notes earlier in the same session.
2. **Over-estimated epoch cost 6×** (200 s vs measured ~12 s) by scaling from the
   old 256-tile fold without accounting for `nimg_per_epoch` capping samples at
   the tile count. Then under-estimated the preemption risk. Both in one run.
3. **Said Tillicum was free.** It bills, and on hours *requested*: a 4 h wall time
   on a ~1 h job estimated $3.60. Cut to 1.5 h = $1.35.
4. **Used `conda activate` in the batch script.** Submitting from an activated
   shell exports `CONDA_PREFIX` into the job; re-activating on top left `python`
   resolving elsewhere, and since `conda activate` returns 0 the failure surfaced
   much later as `ModuleNotFoundError`. Now calls the interpreter by absolute
   path and verifies the import on line one.
5. **Reported ratios from a stale ROI file.** The repo's
   `B03_ACT104_EGFR_ROIs.zip` (25 Jun, 224 ROIs) is far behind the operator's
   live session (~580). Should have checked mtimes first.

## 5. Infrastructure friction (for the next person)

- `nd2` lives in `cpenv`, not `pm-annotate`; plate imports need the former.
- PowerShell parses `$T:$P` as a drive-qualified variable — use literal paths.
- `.slurm` files carry CRLF out of git and `sbatch` refuses them: `sed -i 's/\r$//'`.
- A silent `pip` failure left `cellpose_omni` uninstalled while torch succeeded;
  root cause was **disk quota**, not packaging. 29 GB of it was an
  `apptainer_cache` in an unrelated project directory.
- `/gpfs/scrubbed` is purge-scheduled. The checkpoint must be copied off.

## 6. Next steps

**Immediately (tomorrow morning)**

1. `sacct -j 239017` — confirm COMPLETED; read the final loss.
2. **Copy the checkpoint off `/gpfs/scrubbed`.** The only artifact that cannot be
   regenerated in minutes.
3. Confirm `13619 links` appears in the fold line of the run log.

**The actual result, not yet written**

4. Inference on **PLATE_23** (independent test set: different plate, session and
   annotation method, so leakage is structurally impossible) reporting:
   - **`length_mdape`** — floor **0.3169**. This is the question.
   - `false_split_count` — floor **52/375**.
   - recall — floor 0.928.

   Expect precision to look poor against P23's sparse GT; that is the documented
   sparse-GT effect, not a regression.

**Open, not blocking**

5. Zero-shot scale sweep (`--diameters 9 15 30`) to close that negative properly.
6. Learning-rate decay. Constant 0.1 shows as oscillation: loss reached ~3.9 by
   epoch 4 and moved ~7% over the next 154 epochs, with epoch-to-epoch noise
   larger than the trend. A lower floor is probably available cheaply.
7. Resume-from-checkpoint, so preemptible compute becomes usable.
8. Plan amendment: §10 predeclares leave-one-well-out on `bootstrap_v1`. Training
   on P32 and testing on P23 is a better design but IS an amendment, and belongs
   to the integrator to ratify.
9. Width convention: hand-draw ~20 boundaries and compare against the snap, so
   width carries a measured error bar rather than a 1%-agreement anecdote.
10. The relabelling web tool (`annotation_tools.relabel serve`) was built and is
    untested against a browser. Fiji superseded the immediate need.

## 7. Limits that none of this fixes

- **Z-overlap.** A flat raster holds one identity per pixel; at a crossing the
  output must assign those pixels to one fibre. Links keep each fibre one object
  with its full length, which is what matters for length — but the pixel-level
  overlap is arbitrated, not shared, and no 2-D model changes that.
- **Circularity.** P32 is one operator, one session, one convention. Benchmarking
  against annotations from the same hand measures reproduction, not correctness.
  The 40 held-out correction pairs remain the only non-circular reference.
