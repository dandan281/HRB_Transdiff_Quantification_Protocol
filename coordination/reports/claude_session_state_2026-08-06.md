# Session state, 2026-08-06 → 08-12 — linker withdrawn, Omnipose fine-tuning, Stage 2 blocked

Lane: Claude (annotation tooling / model laboratories)
Tests: **496 passed** (428 at the 08-06 start; 429 after Codex's T03 regeneration, +54 for
the TA03b selector/scorer, +13 for the sample-size planner). Nothing committed; HEAD is
still `0322ebf`.

**Current status in one line:** Omnipose Stage 2 cannot complete on `ckpt-g2` — preemption
arrives faster than a fold trains — and the open decision is checkpoint/resume versus
~$25-90 of uninterrupted GPU. Tier-A tooling is built and waiting on z-stacks due
2026-08-19. A step-back review is being handed to another agent; see
`HANDOFF_REVIEW_PROMPT_2026-08-12.md`, which is the best single entry point to this project.

---

## 1. The headline: the fragment linker is measured and it is bad

The control-only round (`over_merge_c1`) answered the question the release ruling said
would settle automatic use. Sixty accepted merges, ten per well, sampled with equal
probability inside each of the six wells, no flagged cases, no density matching.

| | |
|---|---|
| population over-merge rate | **0.649** (95% CI 0.450 – 0.832) |
| implied wrong merges | **~350 of 540** accepted across six wells |
| false splits recovered, same wells | **11** |
| ratio | ~32× cost over benefit; ~22× at the lower bound |

Every well ≥ 0.37. Sensitivity to the five excluded cases: 0.600 to 0.677, so the
exclusion rule is not load-bearing. Confidence still runs backwards, now replicated at
n=55: AUC 0.323, p = 0.027, and 20 of 25 merges scored at P = 1.0000 called wrong.

Round 2's 0.50 was biased **low** — its controls were matched to the flagged cases on
fragment count, which restricted it to 2–3 fragment objects and structurally hid the
tail. Codex has since set the disposition to
`automatic_and_manual_QC_proposal_use_withdrawn`.

Full detail: `claude_control_only_round_results_2026-08-04.md`.

## 2. Three linker bugs, found and fixed

New module `annotation_tools/annotation_tools/qc_review/link_geometry.py`, 16 tests.

- **The gate never compared fragment axes** — only whether each endpoint pointed at the
  other, over a 12 px neighbourhood. 12 of 55 reviewed objects contained a pair beyond
  the declared 45° window and the operator called all 12 wrong.
- **Union-find discarded the constraint.** 1342 of 1584 within-object pairs (85%) were
  never scored; one object ran 4°→25°→39°→0° between neighbours with its ends 59° apart.
  `constrained_merge` now checks the cross product of both components before each union.
- **Blobs have no direction.** `fragment_axis` returns `None` below `MIN_AXIS_EXTENT_PX`
  (24 px, derived as two `ENDPOINT_LOCAL_PX` windows), and an unestimable axis counts as
  disagreement rather than a pass.

No new operating parameter: `cos_min = 0.70` is the window the linker already declared.
Measured cost — banked training pairs re-match 217 → 200 with **all 41 positives
surviving**, so it removed only negatives. It does **not** rescue the linker: 43 objects
still merge whole and 24 of those are wrong.

Codex wired `constrained_merge` into `merge_prediction` and created
`classical_linker_constrained_v2`.

## 3. Gap augmentation — the Omnipose blocker removed

Omnipose was parked partly because gap-bridging cases are labelled `ambiguous` and so
excluded from its training targets. Measurement says that was too pessimistic:

| longest internal near-background stretch inside one certified mask | instances |
|---|--:|
| ≥ 5 µm | 157 gaps |
| ≥ 10 µm | 56 |
| ≥ 20 µm | 18 |
| longest | **79.2 µm** — the linker's candidate window is 80 |

173 of 374 masks (46%) contain at least one internal gap. `model_labs/omnipose_lab/
gap_augment.py` widens that: take a certified mask, attenuate a band of its Desmin to
match the empirical gap distribution, **leave the mask untouched**. Wired into
`build_fold` as an opt-in arm that *appends* gapped copies rather than replacing.

Two bugs caught during the build, neither by unit tests:

- **Gaps came out 2× too long.** Isotropic dilation grew the band along the fibre by its
  half-width at each end; 7 µm requested re-measured at 16 µm. Fixed by nearest-centreline
  assignment. Ratio is now 1.00, and a test parametrised over fibre thickness pins it.
- **The augmentation silently did nothing.** `tile.labels` is remapped to contiguous local
  ids while `centred_on` stays a well-level id, so the mask was empty for 21 of 24 tiles —
  2 augmented instead of 30, no error anywhere. `local_label_id` plus an assertion.

## 4. klone (Hyak) — Stage 0 complete, Stage 1 queued

| | |
|---|---|
| account / partition | `-A iscrm -p ckpt-g2`, `--gpus=l40s:1 -c 16 --mem=64G` |
| **GPU access** | **checkpoint only** — `iscrm` owns no GPUs (`hyakalloc`: GPUS 0 on all three owned partitions). Every GPU job is preemptible, requeued every 8–9 h |
| data | `/gscratch/iscrm/danlovuw/precision_myotube/bootstrap_v1`, verified `ALL_39_FILES_MATCH` |
| code | `.../precision_myotube/repo` |
| env | `.../precision_myotube/envs/pm-omnipose` — all pins exact, real matmul + conv2d on L40S |
| probe jobs, attempt 1 | 38247995 (GAPS=0), 38248037 (GAPS=1) — **both FAILED, exit 1:0, 2 s elapsed**, at `module load conda`; see trap 5. Never reached our code |
| probe jobs, attempt 2 | **38253704 (GAPS=0)**, **38253705 (GAPS=1)** — submitted after the trap-5 fix landed in both slurm scripts |

### Five things that cost hours and should not be repeated

1. **`sm_89` is absent from torch 2.11.0+cu128's arch list and that is fine.** CUDA is
   binary compatible within a major capability from a lower or equal minor upward, so
   sm_86 cubins run on the L40S. `verify_env.py` demanded an exact match and raised a
   false failure; it now accepts a compatible arch. It still correctly rejects
   `gpu-p100` (sm_60).
2. **Windows-authored shell scripts carry CRLF and `sbatch` refuses them.** `bash`
   tolerates them badly, which masked it. `sed -i 's/\r$//'`. Python files are fine.
3. **Lmod's init is not `set -u` clean** (`LD_LIBRARY_PATH`, then `LD_PRELOAD`, then
   conda's own). Enable `set -u` only after the environment is up.
4. **Install torch BEFORE omnipose.** Omnipose-first pulls torch 2.13.0 and the whole
   CUDA-13 stack — ~2.5 GB written to GPFS and then discarded. Torch-first satisfies
   `torch>=1.10` and skips it entirely.

5. **Do not `sbatch` from a shell with the conda environment active.** `--export=ALL`
   is Slurm's default and copies the submitting shell's environment into the job, so
   `CONDA_PREFIX` travels with it. Lmod then refuses outright — *"Conda is already
   active (CONDA_PREFIX is set). Module conda/Miniforge3-25.9.1-0 not loaded"* — and
   `set -e` converts that to exit 1 about two seconds in, before a single line of our
   code runs. It killed both probe jobs. The scripts now `unset` the inherited
   `CONDA_*` variables before `module load`, so submitting from an active environment
   is safe; the operator no longer has to remember. Note the failure leaves **no**
   `runs/` directory at all, which is a useful signature: `train_fold.py` creates
   `checkpoints/<tag>/` before training starts, so a missing `checkpoints/` means the
   job died upstream of training rather than during it.

Also: **build environments on an owned partition** (`-p compute`, infinite time limit),
never on checkpoint. The first attempt was preempted at package 43 of 144.

## 5. Open decisions

**For the project owner:**

- ~~**From-scratch versus fine-tuning.**~~ **RESOLVED 2026-08-10: fine-tune from
  `bact_phase_affinity`.** Decided before Stage 2 and before any held-out metric existed.
  `bact_phase_omni`, the model originally proposed, turned out to silently rewrite
  `nchan` to 2 and the head to the 4-channel boundary variant; `bact_phase_affinity` is
  the only shipped bacterial model that is a true drop-in. Now predeclared as §5(f) of
  the training plan, guarded by an architecture assertion in `train_fold.py`, recorded in
  each fold manifest as `init_model` / `init_model_sha256`, and part of the sidecar
  resume key. Full reasoning in `claude_omnipose_initialisation_decision_2026-08-06.md`
  and §3 of the training plan.

**For Codex** (`coordination/requests/claude/`):

- `2026-07-31-linker-amendment-followups.md` item **(a)**: `runs/t03/classical_v1/
  assessment.json` no longer reproduces from current source — metrics byte-identical,
  version 1.0→1.1, reworded gate reason, plus a `statistical_analysis_plan_sha256` drift
  that predates this work. Regenerate and record the new hash, or state in the plan that
  it does not reproduce.
- `2026-08-04-constrained-merge-wiring.md`: mostly resolved; the remaining item is that
  `find_link_candidates` now defaults to `require_axis_agreement=True`, so
  `linked_candidate.py` should state the flag explicitly at its call site.
- ~~`coordination/WORKBOARD.md` still reflects 2026-07-23 and has no linker row.~~
  **Wrong, corrected 2026-08-11:** it was reconciled 2026-08-04 and does carry the linker
  rows (T02, T03, T03-LS). It does not yet reflect gap augmentation, klone, the probe, or
  the initialisation decision — but only the integrator edits that board.
- **New, 2026-08-11:** a certification-rate shift across wells, and a bounded estimate of
  how much of the 839-object ambiguous pool is recoverable. See
  `claude_ambiguous_pool_characterisation_2026-08-11.md`. The item needing Codex is the
  G-SO2 disclosure: certified/candidate runs 0.500 in the first well reviewed against
  0.204–0.257 in the next four, on near-identical candidate counts.

## 6. What to read when the probe finishes

Three numbers, predeclared, in priority order:

```bash
squeue -u $USER
grep -E "n_synthetic_gap_tiles|train_seconds|peak_gpu_gb" \
  /gscratch/iscrm/danlovuw/precision_myotube/runs/probe/checkpoints/*/train_manifest.json
```

1. **`n_synthetic_gap_tiles`** on 38248037. Zero means the augmentation is not reaching
   the data and everything downstream is meaningless. A two-well fold gave 30 synthetic
   tiles from 61 originals; expect that proportion.
2. **Seconds per epoch** = `train_seconds ÷ 100`. Workstation baseline ~22 s with the
   dataloader off. Linux has `fork` so the worker dataloader is available for the first
   time — whether it helps is **unmeasured**, and this settles it.
3. **Peak GPU GB.** Batch 8 / tyx 384 killed a 12 GB card; 48 GB should be comfortable.

Gate: proceed to Stage 2 only if augmentation is reaching tiles and the epoch time
projects to a run that fits the budget.

### Stage 1 results — **GATE PASSED**, 2026-08-06

Fold `23_B02_ctrl`, 100 epochs, L40S. Jobs 38253704 (GAPS=0) / 38253705 (GAPS=1).

| | gaps OFF | gaps ON |
|---|--:|--:|
| training tiles | 256 | 256 + **133** synthetic |
| `n_synthetic_gap_tiles` | 0 | **133** |
| `train_seconds` | 2072.3 | 3140.0 |
| **s/epoch** | **20.7** | **31.4** |
| `peak_gpu_gb` | 27.86 | 27.86 |
| `data_prep_seconds` | 8.9 | — |

1. **Augmentation reaches the data.** 133 synthetic from 256 originals = 0.52, against
   the 30-from-61 (0.49) reference and a `gap_probability` of 0.5. Per-well 38/26/29/22/18
   sums exactly to 133 across the five training wells. The silent no-op has not recurred.
2. **20.7 s/epoch vs the workstation's ~22 s is not a wash.** The workstation ran batch 4
   / tyx 224; this is batch 8 / tyx 384, so 2.9x the pixels per epoch in 0.94x the time —
   about **3.1x throughput**. The Linux worker dataloader was the unmeasured variable and
   it paid. Epoch time also scales linearly with tile count (1.515x time for 1.520x
   tiles), so the augmented arm costs exactly what its extra tiles imply.
3. **27.86 GB of 48 GB**, identical across arms as expected — peak allocation is per
   batch, not per dataset. Batch 8 / tyx 384 is settled.

**Stage 2 projection: ~24-26 GPU-hours**, above the plan's 20. At 300 epochs the arms are
1.73 h and 2.62 h per fold, so 10.4 h + 15.7 h = 26.1 h if every fold matched this one.
This fold holds out the *smallest* well and therefore trains on the *largest* set, making
it the slowest of the six; scaling by per-fold instance counts gives ~24 h.

**Scheduling consequence:** each arm exceeds the 8 h `--time` limit on its own, so both
array tasks **will** hit the wall and requeue at least once by construction, before
preemption is considered. Expected behaviour, not a fault — that is what the per-fold
sidecars exist for.

### Fine-tuning path verified end to end, 2026-08-10 (job 38383260)

Two epochs on `23_B02_ctrl` into a throwaway `runs/init_check`, after the initialisation
decision landed. `klone_probe.slurm` gained an `OUT_DIR` override for exactly this: the
checkpoint tag is built from well + policy + arm flags and contains neither epochs nor
initialisation, so without it this run would have silently overwritten the Stage 1
checkpoint and manifest the numbers above came from.

Confirmed on a real L40S, not by CPU-side inspection:

- `init_model` = `bact_phase_affinity` in the manifest, at top level and inside `config`
- `init_model_sha256` = `2abf815c…55ead1`, identical to the staged weights on `/gscratch`
  and to the local copy whose 311 tensors were shape-checked
- log line `init: fine-tuning from bact_phase_affinity (2abf815cb0e4)` — the run really
  did transfer rather than silently fall back to random initialisation
- the run completed, so the post-construction architecture assertion passed against the
  GPU build
- `n_train_tiles` = 256, unchanged from Stage 1, so initialisation did not disturb the
  data path

Weights are staged at `$PROJECT/cellpose_models` and both slurm scripts export
`CELLPOSE_LOCAL_MODELS_PATH`, so no training job touches the network.

**Stage 2 is clear to submit:** `sbatch model_labs/omnipose/klone_folds.slurm`.

### Stage 2 attempt 1 failed: 16 GPU-hours, zero folds (job 38384273, 2026-08-11)

Both array tasks ran the full 8 h and hit `TIMEOUT` with **zero sidecars written**. Two
independent causes, both mine:

1. **The full run never enabled the worker dataloader.** `klone_probe.slurm` passes
   `--dataloader --num-workers 14`; `klone_folds.slurm` passed neither, and `run_folds.py`
   had no such options at all, so both fell through to `DEFAULTS` at `dataloader=False,
   num_workers=0` — the in-process path where CPU flow recompute is the critical path, and
   which the workstation measured at ~57 s/epoch against the probe's 20.7. One fold then
   exceeds 8 h, so no fold ever finished and no sidecar was ever written. **The Stage 1
   timing described a configuration Stage 2 did not use, and nobody checked the two scripts
   agreed.** Both now pass the flags; `run_folds.py` gained `--dataloader`/`--num-workers`.
2. **`--requeue` does not cover `TIMEOUT`.** It covers preemption and node failure. A job
   that hits its `--time` limit ends permanently. The earlier claim in this document that
   each arm "will requeue by construction" was wrong: at ~10.4 h and ~15.7 h per arm against
   an 8 h limit, both arms die at the wall and need **manual** resubmission. Sidecars make
   that a cheap resume, but it is not automatic. Set `--time` to the partition maximum, and
   expect one resubmission per arm regardless.

Generalisable lesson, and the second time this shape of bug has appeared here: a measurement
is only about the configuration it was measured in. The probe was trusted to size a run it
did not share flags with, exactly as the gap augmentation was once trusted to be reaching
tiles it was not reaching.

### Stage 2 attempt 2 — job **38424681**, submitted 2026-08-12

Both fixes verified present on the klone copy before submission: `--time=2-00:00:00` and
`--dataloader --num-workers $((CPUS-2))`. `sinfo -p ckpt-g2` reports **TIMELIMIT infinite**,
so the previous 8 h ceiling was self-imposed; with a generous limit the only interruption
left is preemption, which `--requeue` does handle.

Also visible in `sinfo`: `ckpt-g2` carries **`gpu:h200:8`** as well as `l40s:8` and `l40:8`,
and torch 2.11.0+cu128 covers sm_90 natively. H200 is a live option for a future run, worth
perhaps 2-3x, but it is all-or-nothing — splitting folds across GPU models would put
hardware inside the leave-one-well-out comparison — and H200 contention on a preemptible
partition may cost more in queue wait than it saves in compute.

Optimisation levers identified but **not** applied, pending a clean baseline from this run:

1. **Epochs.** 300 was chosen when the plan was from-scratch training. Fine-tuning from
   pretrained weights converges faster, so 300 may be ~2x more than needed. The *training*
   loss curve settles this and reading it involves no held-out metric, so it is free of
   selection concerns.
2. **cuDNN autotuning.** `deterministic=True, benchmark=False` (`train_fold.py:123-124`)
   forgoes 1.2-2x on a fixed crop size. Strict bit-reproducibility is already partial here
   given preemption across nodes, which is why the manifest records GPU and env hash.
3. **Mixed precision.** ~1.5-2x, blocked upstream (`cellpose_omni` calls `autocast()` with
   no `device_type`; torch 2.11 requires it). Needs a shim and changes numerics; last resort.
4. **Inference time is still unmeasured** and is paid 12 times. Full-field mask
   reconstruction in Omnipose can be slow. This could be a large hidden fraction.

### Stage 2 attempt 2 also failed — the preemption livelock. **CANCELLED 2026-08-12**

The dataloader fix worked: 0.64 s/batch, 24.3 s/epoch, the fast path. It made no difference,
because of a third failure mode neither earlier attempt exposed.

`sacct` over job 38424681 gives the uninterrupted windows actually granted:

| arm | windows before preemption |
|---|---|
| 0 (gaps off) | 00:00:34, 00:26:59, 00:01:40 |
| 1 (gaps on) | 00:02:46, 00:37:27 |

**The longest window was 37 minutes. A fold is ~120 minutes** (300 epochs x 24 s).
`save_every=max(50, n_epochs)` = 300 writes a checkpoint only at the end, and
`train_one_fold` has no resume path, so every preemption restarts that fold at epoch 0.
The array requeues indefinitely and never writes a sidecar. Arm 0 reached `requeue count: 2`
with nothing produced.

**Cutting epochs does not fix this.** No fold length survives a 34-second window. The 8-9 h
preemption cadence this document assumed earlier does not hold on `ckpt-g2` under load.

**The fix, and it is the only path that terminates on this partition:** `save_every=10`
(~4 min of work at risk instead of 2 h) plus resume-from-latest-checkpoint.
`cellpose_omni` accepts `learning_rate` as an array of length `n_epochs`
(`core.py:1094-1112`), so a resume at epoch N can reconstruct the original 300-epoch
schedule and pass `LR[N:]`, giving the exact trajectory an uninterrupted run would have
followed. Only SGD momentum is lost — ~10 steps of memory, well under half an epoch per
resume — and the fold manifest should record resume count and epoch boundaries so the
interruption pattern is disclosed evidence rather than invisible.

At roughly a 50% duty cycle a 2 h fold becomes ~4 h wall, so twelve folds is ~2 days. Slow,
but it terminates, which the present configuration provably does not.

**Open question that would change this:** whether any non-preemptible GPU allocation is
reachable. `iscrm` owns none. If one is, running there uninterrupted is scientifically
cleaner than resume machinery and should be preferred.

### The pattern across all three failures

Each was invisible in the artifact meant to catch it. The probe looked healthy but did not
share flags with the run it sized. `--requeue` looked like insurance but does not cover
`TIMEOUT`. Per-fold sidecars looked like resume but only checkpoint at fold granularity,
which is useless when preemption is finer-grained than a fold. Recorded to memory as
`klone-stage2-failure-modes`.

## 7. Untouched

Conversion_Efficiency, the fusion index, and the plate results are unaffected — they use
the ridge floor and Cellpose nuclei, and the linker was never wired into them. Tier-A
remains blocked on z-axis evidence and the unidentified 561 nm channel, neither of which
this work advances.
