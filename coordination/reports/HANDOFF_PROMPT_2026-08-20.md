# Handoff prompt — paste this into a fresh Claude Code session

Copy everything below the line.

---

You are picking up the T02 Omnipose track of the PrecisionMyotube project in
`c:\Users\liqig\Documents\HRB_Transdiff`, on branch `cleanup-2026-08`.

## Where things stand

A 300-epoch Omnipose fine-tune on the operator's dense PLATE_32 corpus completed
on UW Tillicum and **produces zero masks on every field, including its own
training data**. A full session of diagnosis on 2026-08-20 eliminated every
data-side explanation and localised the failure to the loss function. Read
`coordination/reports/claude_omnipose_training_failure_diagnosis_2026-08-20.md`
first — it has the measurements, the mistakes, and the exact next command.

Established by measurement, do not re-litigate:

- **Images and labels are paired.** AUC of intensity as a classifier of
  "labelled" = 0.800, null controls 0.49–0.54, cross-correlation offset (0,0).
- **The training target is correct.** The step delivers the documented 7-channel
  stack: masks, thresholded mask, boundary, smooth distance (background at −5),
  weights, and two flow channels at 5× scale. The 13,619 links do not corrupt it.
- **The initialisation is good and fine-tuning destroys it.** Untrained
  `bact_phase_affinity` predicts 8.02% foreground against 7.27% true, with a
  well-formed distance field. Weights load; the architecture and eval path work.
- **Learning rate is not the cause.** 0.1 and 0.01 plateau identically.
- **The logged "Epoch Loss" is `raw_loss`**, the unnormalised mean of the nine
  loss terms. It is meaningful. (An earlier claim that it was rescaled and
  therefore uninformative was wrong and is corrected in the report.)

The remaining hypothesis, with a named mechanism: `omnipose.core.loss` ends with

```python
losses = [scale_to_tenths(l, max_gain=1e12) for l in losses]
```

`scale_to_tenths` returns the mantissa of its argument, so every term is divided
by its own magnitude before backprop. A term that improves is amplified back to
O(1); relative weights never change; the objective never settles. This is a
hypothesis, not a proven cause.

## Your first action

Run the experiment that tests it. It is written and committed (`e057fbd`), not
yet run. On Tillicum (`danlovuw@tillicum.hyak.uw.edu`, repo at
`/gpfs/scrubbed/danlovuw/precision_myotube/repo`, interpreter at
`/gpfs/scrubbed/danlovuw/precision_myotube/envs/pm-omnipose/bin/python`):

```bash
$PY -c "import cellpose_omni, omnipose; print(cellpose_omni.version, omnipose.__version__)"

srun --account=hrbomics --partition=gpu-h200 --gpus=h200:1 -c 4 --mem=32G --time=20:00 \
  bash -c "$PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.01 2>&1 | grep -v 'Train epoch\|models/trace'; \
           echo '=== NO RESCALE, lr 0.01 ==='; \
           $PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.01 --no-rescale 2>&1 | grep -v 'Train epoch\|models/trace'; \
           echo '=== NO RESCALE, lr 0.0001 ==='; \
           $PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.0001 --no-rescale 2>&1 | grep -v 'Train epoch\|models/trace'"
```

The operator runs cluster commands and pastes output back; you do not have SSH.
Give commands in copy-paste blocks and **label clearly whether each block runs on
Windows or on Tillicum** — mixing them has already caused three wasted runs.

Then:

- **Raw loss falls with `--no-rescale`** → cause confirmed. Prefer pinning a
  stable `omnipose`/`cellpose_omni` release over monkey-patching; a patch inside
  the training path is a provenance liability for a T03 claim. Either way the
  change belongs in the binding-settings table next to `init_model`/`nclasses`,
  and the integrator (Codex lane) should be told, because it changes what the
  candidate *is*.
- **Raw loss flat in all three** → one specific term is responsible and the
  per-term table names it. Look at `AffinityLoss` first: it consumes flows,
  distance and the link structure together.

## Then, in order

1. **Retrain** on `plate32_dense_v1`, held out B02, with whatever the fix turns
   out to be. Use `model_labs/omnipose/tillicum_dense_train.slurm`. ~50 min,
   ~$1.35 on `gpu-h200`.
2. **Add per-epoch loss logging to `train_fold.py`.** It records none. This is
   the third time on this project that a missing artifact hid a failure — see the
   `klone-stage2-failure-modes` memory. Log `raw_loss` per epoch into the run
   record.
3. **Score it**: `model_labs/omnipose_lab/eval_on_bootstrap.py` against sealed
   `bootstrap_v1` (PLATE_23). The numbers that matter, in order: `length_mdape`
   vs the classical floor **0.3169**; `false_split_count` vs **52/375** (the
   predeclared T03 primary); pooled recall vs **0.928**. Precision and F1 are not
   interpretable against this sparse proposal-conditioned GT — the script says so
   and it is correct.
4. **Owed, cheap, still unread**: the `AFTER training` lines from
   `overfit_one_tile.py` (does fine-tuning degrade the good init?), and the
   zero-shot scale sweep `--diameters 9 15 30`, which is the one thing that would
   make the 2026-08-17 zero-shot negative airtight. That negative now sits oddly
   beside §2.3 of the report and deserves re-reading.
5. **Resume-from-checkpoint** in `train_fold.py`, so preemptible compute becomes
   usable. `save_every=10` plus resume-from-latest, slicing the LR schedule.
6. **Plan amendment**: training on P32 and testing on P23 is not the predeclared
   leave-one-well-out on `bootstrap_v1`. The integrator has to ratify the swap
   before any T03 claim rests on it.

## Rules that bind this lane

- **Sealed artifacts are read-only.** Nothing under `PrecisionMyotube/runs/`,
  `annotation_work/bootstrap_v1`, `model_labs/classical/_runs/`, or the
  `Conversion_Efficiency` result folders gets modified.
- **Never score your own candidate into a ruling.** Codex owns T03. Report
  numbers; do not declare a winner.
- **Predeclared metrics do not get swapped after seeing results**, and no
  threshold search on the test set.
- **Report pooled, object-weighted, with a drop-one-well check** — never a mean
  of per-well rates. A past "+0.015 recall" was really one object out of 375.
- **The operator's tracing is ground truth.** Do not re-measure it to check it.
- Cleanup work follows `cleanup_2026-08/ACTION_LOG.md`: append-only log, quarantine
  instead of delete, test gate before and after each item.

## Traps this project has already hit

- `conda activate` inside a batch script submitted from an activated shell breaks
  the interpreter resolution. Call the env's `python` by absolute path.
- `.slurm` files written on Windows carry CRLF; `sed -i 's/\r$//'` after every
  transfer.
- klone `ckpt-g2` is preemptible with windows as short as 34 s and there is no
  resume path, so a fold never finishes. Tillicum `gpu-h200` is not preemptible
  but bills on hours **requested** — ask for what you need, not a safe margin.
- A diagnostic that does not go through the production code path is evidence
  about the diagnostic. `labels_to_flows` called directly returns 5 channels;
  training's own path delivers 7.
- Copy only what is needed over `scp`; whole package directories include large
  TIFFs and trip SSH rate-limiting.

## How to work

Cheapest decisive test first, where "decisive" means it splits the hypothesis
space — not that it tests the most likely branch. Read the source of a function
before reasoning about the numbers it returns; `inspect.getsource` is what
resolved this session after two wrong inferences from the same log line. State
what is measured separately from what is inferred.

---
