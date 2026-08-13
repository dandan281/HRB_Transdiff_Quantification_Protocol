# Omnipose training plan — what we run on klone, and what each result would mean

Date boundary: 2026-08-04 America/Los_Angeles; klone timestamps 2026-08-05 UTC
Lane: Claude (model laboratories)
Status: **predeclared before any training run.** §4 and §5 are fixed and may not be
revised after seeing a held-out metric. §2, §3 and §8 are descriptive and may be
corrected for accuracy at any time.

---

## 1. Why this run exists

Two separate reasons, and they should not be confused.

**T02 needs a second candidate.** The contract requires two. The classical floor is
one; the fragment linker was never a valid second — Codex ruled that *"a linked variant
of the classical floor does not satisfy the two-candidate comparison"* — and the
control-only round has since withdrawn it from use entirely. Omnipose is the only
remaining path to a completed T02 and therefore to any T03 selection.

**The methodological objection has been removed.** Omnipose was parked on 2026-07-23
partly because the dominant failure, signal-gap fragmentation, is labelled `ambiguous`
and so excluded from its training targets. Measurement on 2026-08-04 showed that was
too pessimistic — 173 of 374 certified masks already contain an internal gap — and
`gap_augment` now widens that supervision from certified masks without inventing any.
The parking reason is answered; the hardware reason is answered by klone.

## 2. What we are asking Omnipose to do

**The job: take one Desmin field and return one mask per myotube.**

That is instance segmentation, and the reason it is hard here is not the staining but
the shape. Myotubes are long, thin, curved, they touch, they cross, and they overlap in
projection. A classical ridge filter traces them and then fragments them wherever the
Desmin signal dips — 5,279 predictions for 375 real objects, 52 of them false splits.

Omnipose is the right family to try because it was built for exactly this shape class.
Cellpose predicts flows toward a cell centre, which works for round objects and fails
for a fibre 400 µm long with no meaningful centre. Omnipose replaces that with a
distance field and flows along the medial axis, and its published wins are on bacteria
— elongated, touching, variable length. Myotubes are a plausible transfer of that.

**Concretely, the input is one 3636×3636 16-bit Desmin channel** (`nchan=1`; DAPI is
present in the bootstrap but not fed to the model) and the output is an overlap-aware
`InstanceSet` exported through the canonical adapter, unreviewed.

**What would count as working**: fewer false splits than the classical floor's 52,
at comparable recall. Not better precision — precision is not interpretable against a
sparse reviewed GT, and never will be with this reference set.

## 3. How it is trained

| | setting | source |
|---|---|---|
| architecture | `CellposeModel(omni=True, dim=2, nchan=1, nclasses=2)` | `train_fold.py` |
| **initialisation** | **`pretrained_model=False` — from scratch** | `train_fold.py:141` |
| optimiser | SGD, lr 0.1, weight decay 1e-5 | `DEFAULTS` |
| epochs | 300 | `DEFAULTS` |
| batch / crop | 8 / 384² | `DEFAULTS` |
| normalisation | percentile 1.0–99.5 at **whole-field** scope, `normalize=False` downstream | `data.normalize_field` |
| training units | one tile per reviewed instance, bbox + 96 px margin, floored 256 / capped 1024 | `instance_tiles` |
| ignored pixels | painted out of the image and replaced with real local background | `ignore_policy`, option 2 |
| inference | full field, `bsize=224`, `tile_overlap=0.1`, stitched in **flow space** | `infer_fold` |
| thresholds | `mask_threshold=0.0`, `flow_threshold=0.0` (off), `min_size=15` | `DEFAULT_THRESHOLDS` |

Three of these deserve their reasons stated, because each was a decision:

**Tiles are sized per instance, not fixed.** A fixed 1024 tile left the median myotube
at 3.5% of its own crop and Omnipose's sampler died with *"Sparse or over-dense image
detected"*. Sizing each tile to its instance fixed it.

**Ignored regions are painted out, not masked in the loss.** `cellpose_omni` accepts no
per-pixel loss mask, and `omnipose.core.loss` applies its weight to only 5 of its 9
terms — so zeroing the weight would leave ~44% of the loss still supervised on pixels
the operator never certified. Masking properly means forking the loss, which would make
T03 a statement about our fork rather than about Omnipose. Painting them out with real
background makes the `background` label *true* for the modified image.

**Inference stitches the flow field, then reconstructs masks once over the whole
field.** Reconstructing per tile and merging afterwards would split any object crossing
a seam — reintroducing the fragmentation this candidate exists to reduce.

### RESOLVED 2026-08-10 — initialisation is transfer, from `bact_phase_affinity`

The project owner settled this before Stage 2 and before any held-out metric existed:
the probe measured tile counts, epoch time and memory only. Amending the plan at this
point is therefore not selection on held-out results, which is the whole reason it had
to be decided now rather than later.

`train_fold.py` now defaults to `init_model="bact_phase_affinity"`. Three findings from
checking the weights against the architecture, all verified rather than assumed:

- **`diam_mean=0.0` was never a compatibility problem.** The bacterial branch's
  `diam_mean` assignment is commented out at `models.py:449`, so our value survives.
- **`bact_phase_omni` — the model this section originally named — is unusable, and fails
  silently.** It is in both `C2_MODEL_NAMES` and `BD_MODEL_NAMES`, so `__init__` rewrites
  `nchan` 1→2 and the head 3→4 channels before building the net, loads the checkpoint
  without complaint, and then expects 2-channel input we do not have plus the boundary
  head this harness rejects. 12 tensors mismatch. `train_fold.py` now asserts the
  resolved `(nchan, nclasses)` after construction so this fails loudly.
- **`bact_phase_affinity` is a genuine drop-in**: 311/311 tensors, zero shape mismatches,
  `nchan=1` and the 3-channel head preserved, weights confirmed loading.

**Accepted caveat, stated for the record.** That model was trained with `affinity_field`,
so its scalar channel is a summed connectivity graph rather than a distance field
(`omnipose/core.py:1094`). Our targets supply a distance field, so that one channel is
semantically mismatched — but the mismatch is confined to the 33 parameters of the scalar
output slice out of 6,610,100, and the encoder plus both flow channels transfer normally.
This is disclosed rather than discovered later; if Omnipose underperforms, it is one of
the hypotheses to test, not a surprise.

### The original discrepancy, as recorded before it was resolved

`DEVELOPMENT_PLAN.md` §10 specifies candidate 2 as *"real Omnipose **transfer**/fine-tuning
on the bootstrap masks"*. The harness passes `pretrained_model=False`, which trains
**from scratch**. Those are different things.

It matters because 375 instances is a small dataset to train a segmentation network from
random initialisation, and Omnipose ships pretrained weights for bacteria — the closest
published match to this shape class. Fine-tuning from `bact_phase_omni` is plausibly the
stronger candidate and is what the contract asks for.

I have **not** changed it, for two reasons: `nchan=1` and `diam_mean=0.0` may be
incompatible with the pretrained weights' expectations, and swapping initialisation
after the probe would be a design change made mid-run. Resolve it before Stage 2, in one
of two ways — either establish that from-scratch is intended and amend the plan's
wording, or add fine-tuning and let the probe compare the two. This is a question for
the integrator, not something to settle silently.

## 4. The design

| axis | setting | why |
|---|---|---|
| folds | 6, leave-one-well-out, whole well | binding project rule |
| ignore policy | **fixed at `paint_out`** | already decided by measurement; re-opening it here would double the run for a settled question |
| gap augmentation | **off / on** — the ablation | the one thing this run is designed to answer |
| epochs | 300 | harness default |
| batch / tyx | 8 / 384 | the intended defaults, which OOM'd a 12 GB card; the L40S has 48 GB |

12 fold-trainings. The `ambiguous_as_background` policy arm is **not** run now — it is a
question about the ignore policy, already answered, and running it would cost another 12
trainings to re-litigate a decision nobody is disputing.

## 5. Predeclared, and binding

**(a) The T02 candidate is the gaps-OFF arm.** Plain Omnipose on the bootstrap masks is
what the contract asks for. Gaps-ON is a declared ablation of that candidate.

This is fixed now, in advance, for one reason: if the candidate were chosen after seeing
which arm scored better on held-out wells, that would be selection on held-out metrics —
which §10 of the development plan forbids outright, and which is the discipline the
linker's threshold followed when 0.90 was predeclared. If gaps-ON wins, that is a
finding about augmentation to be reported and then tested properly, **not** a licence to
swap the candidate.

**(b) The primary comparison is `false_split_count`, pooled across six wells, paired by
well.** It is the metric the fragmentation failure actually shows up in, it is measured
against the full reviewed GT rather than the sparse subset, and it is the same statistic
the classical floor scores 52 on — so the numbers are commensurable.

**(c) Precision and F1 are not interpretable** and will not be quoted as detector
performance. The reviewed-complete GT is a sparse subset of field structure, not a
census. This is already binding across the project.

**(d) Recall is low-resolution.** Its denominator is 375 masks, so one object moves it by
0.0027. A change of fewer than 5 objects (0.013) will be reported as flat.

**(f) The candidate is fine-tuned from `bact_phase_affinity`, not trained from scratch.**
Added 2026-08-10, before Stage 2 and before any held-out metric existed — see the
resolution note in §3. Initialisation is part of what the candidate *is*, so it is
predeclared here alongside the gap arm rather than left to the harness default. There is
no initialisation ablation: running both would double Stage 2 to ~50 GPU-hours and would
invite exactly the post-hoc selection §5(a) exists to forbid. `run_folds.py` records
`_init_model` in each sidecar and refuses to resume across a change of it, for the same
reason it refuses to resume across the gap arms.

**(e) The threshold-free comparison is the only one.** No post-hoc search over
`mask_threshold` or `flow_threshold` on held-out results. `DEFAULT_THRESHOLDS` stands.

## 6. Stages and gates

### Stage 0 — environment — **COMPLETE 2026-08-06**
Build on an **owned, non-preemptible** partition (`-p compute`), not on checkpoint. The
first attempt built it interactively on `ckpt-g2` and Slurm preempted the node at 43 of
144 packages, losing the half-built environment. `pip` needs no GPU; only the final
arch check does, and that takes seconds.

The gate is a **real kernel launch**, not `torch.cuda.is_available()`: that returns True
on a GPU whose architecture the build does not cover, and fails only at the first matmul.

Verified on klone: `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `omnipose 1.1.4`,
`cellpose_omni 1.1.4`, and the four pinned scientific packages all exact. Six wells and
375 trainable masks reachable. Every project module imports. Real matmul and conv2d run
on the L40S.

### Stage 1 — probe: one fold, one arm, 100 epochs
`klone_probe.slurm`, well `23_B02_ctrl` (smallest, so the honest answer arrives soonest).
Run it twice: `GAPS=0` and `GAPS=1`.

**Three numbers to read, in order of importance:**

1. **`n_synthetic_gap_tiles`** on the `GAPS=1` run. If it is 0 or near it, the
   augmentation is not reaching the data and every downstream result is meaningless.
   This is not paranoia: a silent no-op here has already happened once, when a
   well-level id was compared against a tile-local one and 2 tiles were augmented
   instead of 30, with no error anywhere.
2. **seconds per epoch.** The workstation baseline is ~22 s at batch 4 / tyx 224 with the
   dataloader off. Linux has `fork`, so the worker dataloader is available here for the
   first time. Whether that helps is **unmeasured** — the resource note says CPU flow
   recompute dominates, but at 256 tiles / batch 4 the GPU side alone accounts for ~19 s
   of the 22. This probe settles it.
3. **peak GPU memory.** Batch 8 / tyx 384 exhausted a 12 GB card and torch reported it as
   `CUDA error: unknown error` rather than a clean OOM. On 48 GB it should be
   comfortable; confirm rather than assume.

**Gate:** proceed only if the augmentation is reaching tiles and the epoch time projects
to a full run that fits the budget. If seconds/epoch has not improved, the 20-hour
estimate stands — that is a cost decision, not a blocker.

### Stage 2 — full run
`klone_folds.slurm`, a 2-task array over the gap axis, 6 folds each, resumable.

Preemption is not an edge case: `iscrm` owns no GPUs (`hyakalloc` reports GPUS: 0 on all
three of its partitions), so every GPU job runs on checkpoint and will be requeued
roughly every 8–9 hours without notice. Per-fold sidecars plus `--requeue` make that cost
one fold, not one arm.

### Stage 3 — seal and hand over
Sealed predictions, hash-bound, unreviewed, exported through the canonical adapter. Then
Codex scores them under T03. **I do not score my own candidate** — that is the lane
split, and it is what makes the comparison worth anything.

## 7. What happens after the run

In order, and none of these are automatic:

1. **T03 scores both candidates on the same benchmark** — classical floor versus
   Omnipose, six folds, micro and macro, whole-well bootstrap intervals. Codex owns this.
2. **A selection becomes possible for the first time.** T03 has been blocked on having
   only one candidate since 2026-07-23. Note that "possible" is not "certain": the
   assessment may still decline to select either, and that is a legitimate outcome.
3. **G-SO2** — the training-evidence gate. Hash-frozen data, all folds present, leakage
   audit, exclusions visible, G-SO1 status disclosed, nothing described as consensus or
   independent truth.
4. **Then the workflow freezes** — code, model, environment, thresholds, review protocol,
   data hashes — and only after that does a prospective plate get acquired. Tier C.

**What this run does not advance**: Tier-A conversion efficiency, which is blocked on
z-axis evidence and the unidentified 561 nm channel, not on segmentation. Those are
independent tracks and finishing T02 does not move them.

## 8. What would make this run a failure, stated now

- **Augmentation reaches too few tiles.** Watch `n_synthetic_gap_tiles`. On a two-well
  fold at probability 0.5 it produced 30 synthetic tiles from 61 originals; anything far
  below that ratio needs explaining before the full run.
- **Omnipose is no better on fragmentation.** Entirely possible — it was the original
  reason for parking, and training from scratch on 375 instances makes it more likely.
  That is a real result and completes T02 regardless; a second candidate that loses is
  still a second candidate.
- **Both arms identical.** Would suggest the ablation is not actually differing. The
  resume check now separates the arms by sidecar name and by an `_augment_gaps` field,
  because without that the baseline's sidecars would have been silently resumed into the
  augmented arm and the ablation would have compared a run against itself.

## 9. What this run cannot answer

- **Z-overlap.** Omnipose is 2-D. Fifteen of the 36 wrong linker merges were "dimension
  problem", and no 2-D architecture sees which of two overlapping fibres is on top. That
  needs z-stacks, which are also blocking Tier-A's `TA03c`.
- **Whether augmentation helps in general.** One corpus, one plate, six wells, one
  operator. A positive result here is development evidence.
- **Anything prospective.** Plate 23 is retrospective and proposal-conditioned. This is
  internal model evaluation and nothing else.

## 10. Data and code on klone

| what | where |
|---|---|
| training data | `/gscratch/iscrm/danlovuw/precision_myotube/bootstrap_v1` — 39 files, verified `ALL_39_FILES_MATCH` |
| code | `/gscratch/iscrm/danlovuw/precision_myotube/repo` |
| environment | `/gscratch/iscrm/danlovuw/precision_myotube/envs/pm-omnipose` |
| runs | `/gscratch/iscrm/danlovuw/precision_myotube/runs/` |

`$HOME` is 10 GB and ~80% full, so nothing conda-related may live there; the package and
pip caches are redirected to `/gscratch` as well.

The minimal re-stage set, if this ever has to be rebuilt on a fresh allocation, is about
240 MB: `bootstrap_v1`, the six per-well `*.qc.instances.json` (7 MB — `load_well` reads
them and a scoped copy that omitted them would fail at the first fold),
`training_exclude.json`, and ~5 MB of code.

### Four things that cost time and are worth not repeating

**`sm_89` is not in torch 2.11.0+cu128's arch list, and that is fine.** The build ships
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`; the L40S is sm_89. `verify_env.py`
demanded an exact match and raised a false failure. CUDA guarantees binary compatibility
within a major compute capability from a lower or equal minor upward, so sm_86 cubins run
on sm_89 — confirmed by real matmul and conv2d. The verifier now accepts a compatible
arch and lets the kernel launch decide. It still correctly rejects `gpu-p100` (sm_60),
which genuinely has nothing usable in this build.

**Shell scripts written on Windows carry CRLF and `sbatch` refuses them outright.**
`bash` mostly tolerates them, which is why the setup script ran (badly) while `sbatch`
would not start at all. Strip with `sed -i 's/$//'`. Python files are unaffected.

**Lmod's init is not `set -u` clean** — it dereferences `LD_LIBRARY_PATH`, then
`LD_PRELOAD`, and conda's hook has its own. Naming them one at a time is whack-a-mole.
Enable `set -u` only after the environment is up.

**Install order matters and the original order was wrong.** Installing `omnipose` first
pulls torch 2.13.0 and the entire CUDA-13 stack — about 2.5 GB written to GPFS and then
discarded by the forced cu128 reinstall. Installing `torch==2.11.0+cu128` **first**
satisfies omnipose's `torch>=1.10` and skips that round trip entirely.

If the environment fights us again, the fallback is an NGC PyTorch container via
apptainer (`ngc/1.31.0` is available). One `.sif` is a single large sequential write
rather than a hundred thousand small ones, which is what GPFS is bad at. The cost is
losing the version pin; the environment record would then be the container digest.

## 11. Verification at each stage

```bash
# stage 0
python -c "import torch; print(torch.cuda.get_arch_list())"

# stage 1
grep -E "n_synthetic_gap_tiles|train_seconds|peak_gpu_gb" \
  /gscratch/iscrm/danlovuw/precision_myotube/runs/probe/checkpoints/*/train_manifest.json

# stage 2
squeue -u $USER
ls /gscratch/iscrm/danlovuw/precision_myotube/runs/omnipose_v1/fold_results/
```

Local suite before anything is sealed: **428 tests**, re-derived rather than trusted.
