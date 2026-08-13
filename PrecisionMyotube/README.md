# PrecisionMyotube

Canonical, precision-first analysis of 2-D ND2 myotube fields. It produces:

1. per-myotube geodesic length;
2. local width distributions;
3. total valid nuclei;
4. field-level conversion efficiency;
5. nuclei per reviewed independent myotube.

The pipeline deliberately separates **semantic Desmin territory** from **independent myotube
instances**. Territory supports conversion efficiency even when a contact is ambiguous. Only an
expert-reviewed `complete` instance can enter length, width, or multinucleation statistics.

See the active [single-operator development plan](DEVELOPMENT_PLAN.md),
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md), and central
[execution workboard](../coordination/WORKBOARD.md). The latest restart summary is the
[2026-07-22 session handoff](../coordination/SESSION_HANDOFF_2026-07-22.md). The older
parallel-execution PDF is historical;
its two-annotator gate has been superseded because only one human operator is available.
All group summaries and release claims must follow the binding
[statistical analysis plan](STATISTICAL_ANALYSIS_PLAN.md), which distinguishes objects, technical
replicates, and independent biological units.

## Install

The existing `Conversion_Efficiency/cpenv` already contains most runtime dependencies:

```powershell
Conversion_Efficiency\cpenv\Scripts\python.exe -m pip install -e PrecisionMyotube
```

Cellpose is optional if validated nucleus masks are supplied. µSAM/napari and Omnipose should live
in their own environments to avoid changing the validated Cellpose CUDA environment.

## One-field workflow

Prepare and resolve the channels. For the existing Q plates, explicitly confirming `fiber=1` and
`DAPI=2` is recommended even though automatic role scoring is recorded:

```powershell
python -m precision_myotube prepare --nd2 Q_PLATES/Q_Plates/PLATE_23/32_C08_br223_igf1r.nd2 `
  --out PrecisionMyotube/runs/32_C08 --fiber-ch 1 --dapi-ch 2
python -m precision_myotube territory --run PrecisionMyotube/runs/32_C08
python -m precision_myotube nuclei --run PrecisionMyotube/runs/32_C08
python -m precision_myotube proposals --run PrecisionMyotube/runs/32_C08
python -m precision_myotube annotation-package --run PrecisionMyotube/runs/32_C08 `
  --instances PrecisionMyotube/runs/32_C08/instance_proposals.json `
  --out PrecisionMyotube/annotation_work/32_C08
```

Correct the masks in napari/µSAM according to [ANNOTATION_PROTOCOL.md](ANNOTATION_PROTOCOL.md).
Then import a curated mutually exclusive label TIFF and its per-instance status/review table:

```powershell
python -m precision_myotube import-labels --labels corrected_labels.tif `
  --image-id 32_C08_br223_igf1r --properties instance_properties.csv `
  --out PrecisionMyotube/annotations/32_C08.instances.json
python -m precision_myotube analyze --run PrecisionMyotube/runs/32_C08 `
  --instances PrecisionMyotube/annotations/32_C08.instances.json
```

`analyze` writes `myotubes.csv`, `nuclei.csv`, `field_summary.csv`, `analysis_summary.json`,
`qc_flags.json`, `qc_overlay.png`, and `review.html`. A mask touching the field border is demoted to
`border_truncated` even if accidentally annotated as complete.

`--reviewed-complete` remains available only for a label image whose every object has already been
expert-reviewed as complete. For normal fields, use `--properties` so truncated, occluded, and
ambiguous objects retain their individual statuses.

For existing validated nucleus masks, skip Cellpose and provide them to the one-command workflow:

```powershell
python -m precision_myotube run --nd2 <image.nd2> --out <run_dir> `
  --fiber-ch 1 --dapi-ch 2 --nuclei-masks <validated_masks.npy>
```

Without `--instances`, `run` intentionally stops at the review gate.

## Restartable batch execution

Use the versioned manifest interface in `batch_manifest.example.json` for plate-scale work. Each
field records independent hashes and checkpoints for prepare, territory, nuclei, proposals,
analysis, and report stages. A missing or modified artifact invalidates that stage and all later
stages when the batch is resumed.

```powershell
python -m precision_myotube batch --manifest batch_manifest.json --summary-dir batch_results
python -m precision_myotube batch --manifest batch_manifest.json --summary-dir batch_results --resume
```

The command writes `batch_summary.csv` and `batch_summary.json`. Fields without reviewed instance
input stop as `review_required`; a failed field is reported without corrupting completed fields.

Candidate framework outputs can be normalized through `adapt-prediction`; see
[PREDICTION_ADAPTERS.md](PREDICTION_ADAPTERS.md). Verify that a run still refers to its original
source, instance, nucleus, and territory bytes with:

```powershell
python -m precision_myotube verify-run --run <run_dir>
```

## Legacy dual-annotation commands

Build review-target strata from canonical proposal runs, then select a deterministic,
development-only pilot:

```powershell
python -m precision_myotube pilot-candidates --manifest pilot_runs.json `
  --out pilot_candidates.json
python -m precision_myotube pilot-select --candidates pilot_candidates.json `
  --target 100 --minimum-hard 25 --out pilot_manifest.json
python -m precision_myotube gate-g1 --evidence g1_evidence.json --out g1_result.json
```

These commands are retained for reproducibility of the original plan, but their 100-task
dual-annotation gate is superseded by the single-operator plan. Do not ask the user to complete it.
The active path reconciles the existing six-well reviews and runs a blinded 30-case test-retest
audit. Plate 26 is retrospective rather than locked because its labels and prior runs are already
present in this repository.

## Annotation and model bake-off

- `dataset-audit` retains the original 1,000 complete / 250 hard-case checks as legacy targets. The
  active low-labor plan uses a hash-frozen bootstrap set, field-level cross-validation, bounded
  active learning, and a future prospective plate instead.
- `export-training` creates lossless image/label pairs from reviewed complete instances only.
- `model-commands` records version-visible Cellpose-SAM and Omnipose training commands plus the
  µSAM entry point.
- `benchmark-manifest` scores every model using the same overlap-aware instance metric and enforces
  plate- and density-stratified collapse checks. `benchmark` remains useful for a single field.
- `select-model` disqualifies precision below 0.85 or over-merging above 5%, then ranks eligible
  candidates with precision weighted twice recall.
- `release-check` applies the stricter prospective scientific gates. Failure automatically limits
  reporting to field metrics and manual-QC-only instance measurements.

Example:

```powershell
python -m precision_myotube dataset-audit --manifest dataset_manifest.json --out audit.json
python -m precision_myotube model-commands --train training --test locked_test --out commands.json
python -m precision_myotube benchmark --ground-truth gt.instances.json `
  --prediction omnipose.instances.json --model omnipose --out omnipose.metrics.json
python -m precision_myotube release-check --metrics prospective_metrics.json --out release.json
```

## Measurement definitions

- **Length:** longest geodesic path on the full-area instance skeleton.
- **Width:** twice the distance to the boundary sampled along the centerline; endpoint caps and
  branch neighborhoods are excluded when possible. Median, IQR, P10/P90, CV, and area/length are
  retained.
- **Total nuclei:** Cellpose nucleus instances inside the physical 50–500 µm² band by default.
- **Conversion efficiency:** proportion of valid nuclei with at least 50% overlap with semantic
  Desmin territory. Results at 40% and 60% are included as sensitivity checks.
- **Multinucleation:** uniquely assigned valid nuclei per reviewed complete myotube. If the best and
  second-best overlaps differ by less than 0.25, assignment is ambiguous and excluded.

## Statistical summaries

Do not treat individual nuclei or myotubes as independent biological replicates. Declare the
technical and biological units in a statistics manifest, collapse technical observations first,
and report biological-unit effects with confidence intervals:

```powershell
python -m precision_myotube statistics-summary `
  --manifest statistics_manifest.json --out statistics_result.json
```

See `statistics_manifest.example.json`. With the current single Plate-23 development set,
treatment-effect output is descriptive only; the six wells support internal model evaluation but
not general biological inference.

## Limits and acquisition

The existing ND2 files are single 2-D planes. No algorithm can recover an absent boundary at an
unresolved Desmin-positive contact. Future validation should add a membrane-outline pilot (WGA first,
then a sarcolemmal antibody if needed) and Nyquist-sampled z-stacks while preserving DAPI, Desmin/MyHC,
and the receptor channel. A separate 3-D model requires separate 3-D annotations and validation.
