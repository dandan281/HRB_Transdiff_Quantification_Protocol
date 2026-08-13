# Implementation status

## Implemented

- Native 16-bit, 2-D ND2 extraction with source SHA-256, pixel size, channel-role provenance, and
  write-once run metadata.
- A read-only, hash-bound Tier-A conversion audit that exactly reproduces the frozen six-well
  10 µm perinuclear-ring estimator with one plate-wide pooled log-Otsu threshold. Field conversion
  efficiency is independent of instance reconstruction.
- Cellpose-SAM nucleus segmentation/threshold sweep, external validated-mask import, physical-area
  QC, full-resolution labels, and 40/50/60% territory-overlap sensitivity.
- Overlap-safe COCO-style instance records with `complete`, `border_truncated`, `occluded`, and
  `ambiguous` states. Connected components are review prompts and can never become authoritative
  without an explicit expert-reviewed import.
- Per-instance geodesic length, total skeleton length, branch count, local width distribution,
  area/length cross-check, and conservative nucleus assignment.
- Canonical CSV/JSON outputs, full-field overlay, review queue, append-only QC provenance, and a
  napari/micro-sam annotation round trip that preserves per-object statuses.
- Dataset audit for annotation volume, hard cases, historical dual review, and required
  plates/splits. Plate 26 is now explicitly classified as retrospective rather than locked.
- Common candidate exports, training-command manifests, per-field and plate/density-stratified
  benchmarking, precision-first selection, and prospective release gates.
- Restartable batch manifests with hashed per-stage checkpoints, isolated field failures,
  review-required outcomes, and CSV/JSON plate summaries.
- Framework-neutral label, polygon, and RLE prediction adapters that preserve overlaps, force
  predictions to remain unreviewed, and record checkpoint/environment/threshold provenance.
- Matched-instance benchmark evidence with automatic coverage, split/merge errors, length/width
  MdAPE, required-stratum fail-closed checks, and deterministic precision-weighted selection.
- Source and latest-analysis artifact verification through the `verify-run` integrity command.
- Reproducible environment fingerprinting with an exact package freeze, stable environment hash,
  Python/platform and Torch/CUDA/GPU metadata, and validation-artifact binding. The validated
  Cellpose environment reproduces the C08 total of 10,114 valid nuclei.
- A binding statistical-analysis layer that collapses raw observations to technical and then
  biological units, with deterministic whole-unit bootstrap intervals, paired effects, Wilson
  descriptive count intervals, Benjamini-Hochberg adjustment, and explicit descriptive-only output
  when independent replication is insufficient. Benchmark manifests now report micro and macro
  field metrics plus whole-field confidence intervals; prospective release uses adverse confidence
  bounds and requires declared experimental units and biological replication.
- Legacy H01 pilot tooling remains reproducible: it derives review strata, excludes Plate 26 from
  that historical manifest, audits the approximately-100/25-hard-case targets, and evaluates the
  retired G1 gate without inventing missing evidence. It is not the active workflow.
- Hashed six-field pilot annotation-package handoff, 100-task reviewer templates, canonical review
  validation, and independent-review disagreement comparison are ready. Claude's headless core,
  overlap round-trip, and serverless package-level QC page work. The available interfaces do not
  yet consume the frozen 100-task manifest or emit the required reviewer-linked, four-status pilot
  contract, so CL01 remains incomplete for G1. Native napari also fails OpenGL initialization on
  this workstation, but the serverless path means that failure alone is no longer the blocker.
- Claude's current QC-review workflow has now been used by the sole human operator on all six
  Plate-23 wells: 1,800 decisions produced 377 complete masks, 31 border-truncated masks, 839
  ambiguous retained cases, 553 rejects, six reviewer logs, and 40 correction pairs. These are
  provisional single-operator, proposal-conditioned labels rather than consensus ground truth.

## Current plan position

The authoritative [single-operator development plan and continuation record](DEVELOPMENT_PLAN.md)
supersedes the old two-annotator
H01/H02/G1 path. R02 and R03 now pass integrator validation: the six-well snapshot reconciles and
the triage CSV is an exact 961-row match to the current accept/reject decisions. The official
round-2 G-SO1 metric checks also pass. The blind export provenance defect is fixed; its T01-era
annotation/model-lab checkpoint had 26 passing tests. The two complete-to-ambiguous disagreements
are now bound to an
exclusion manifest. At the user's direction, the remaining repeatability evidence gap no longer
blocks development. T01 bootstrap v1 is materialized with 375 real masks, 40 real correction pairs,
and 2,290 eligible synthetic pairs; 12 synthetic derivatives of excluded masks were removed.
Its manifest SHA-256 is
`44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94`.
The classical T02 candidate is sealed and its official T03 single-candidate assessment is complete.
The fragment linker is implemented but rejected: a predeclared uniform six-well review measured
population over-merge rate 0.6487 (95% CI 0.4497-0.8318), approximately 350 wrong merges among
540 accepted. Automatic use and manual-QC proposal use are withdrawn. Its sealed v1 remains only
as reproducible rejected development evidence; a constrained-merge closure fix is wired under a
new candidate identity but does not rescue the linker. Omnipose's real harness is implemented and
ported to klone: Stage 1 confirms active gap augmentation (133/256 tiles), 20.7/31.4 s per epoch
baseline/augmented, and 27.86 GB peak; Stage 2 is launched as a two-task gap-off/gap-on array.
Initialization is predeclared `bact_phase_affinity` with architecture and resume guards. No sealed
held-out result has returned, so the two-candidate T02/T03 contract remains incomplete. G-SO2 must
disclose the first-well certification shift (0.500 versus 0.204-0.257 in the next four treated
wells). T03 retains its predeclared primary and adds whole-well drop-one sensitivity; the classical
false-split rate is 52/375 primary and 34/256 omitting B06.
The Tier-A audit also passes independent integration review. The ring estimator is now canonical
for internal conversion analysis; the older 6.6245% traced-fiber method is superseded. This method
freeze establishes reproducibility, not biological correctness or scientific release.
The orthogonal Tier-A validation design is statistically ratified with amendments. Its binding
contract requires selection-probability weighting, field/biological-unit clustering, disjoint
threshold calibration and validation, adverse-confidence-bound acceptance gates, and
pilot-derived—not guessed—sample sizes. The selector is implemented but still needs tests. Raw ND2
event tables contain field-centre XYZ, but no certified pixel-to-stage affine exists. The scorer is
authorized under the amended whole-field/mosaic reacquisition plus DAPI-registration matching
design; imaging acquisition is not.
No human action is currently required. See [the official evidence status](HUMAN/g_so1_result.json)
and [the dated session handoff](../coordination/SESSION_HANDOFF_2026-07-22.md).
The canonical suite now has 44 passing tests. Version 1.1 of the binding
[statistical analysis plan](STATISTICAL_ANALYSIS_PLAN.md) includes the orthogonal conversion
validation contract. The current annotation/model-lab suite
has 160 passing tests at the official T03 checkpoint; base Python is missing its
`skan` dependency.

## Verified on a real field

The canonical internal Tier-A estimator uses a 10 µm perinuclear ring, one plate-wide pooled
log-uniform Otsu threshold (`440.76596787901417`), and the canonical `[50,500] µm²` nucleus-area
filter. It reproduces all six wells exactly. For `PLATE_23/32_C08_br223_igf1r.nd2`, 3,341 of 10,114
canonical nuclei are positive (`33.03%`); the control is `15.27%`, and the six same-plate folds are
`1.0–2.58×`. These are descriptive internal results, not treatment effects.

The two C08 mask paths contain byte-identical arrays. Their apparent count difference comes from
filters on the same 10,588 labels: 10,562 meet the MyoFuse `>=30 px` floor and 10,114 meet the
canonical physical-area filter; MyoFuse ring validity removes two more to produce 10,560. The older
traced-fiber `670 / 10,114 = 6.6245%` result remains only as superseded provenance. The 500 semantic
components remain ambiguous review proposals, with zero authoritative independent myotubes.

## Intentionally not claimed complete

- Automatic independent-myotube detection is not scientifically released: the required expert
  full-area annotations do not yet exist.
- Inter-rater agreement and consensus ground truth cannot be claimed because only one operator is
  available. The replacement test-retest workflow is in [HUMAN/README.md](HUMAN/README.md).
- The rebuilt triage CSV contains 961 unique current rows and no stale C08 rows. Its 0.889 training
  accuracy is in-sample triage evidence only, not segmentation performance.
- The historical round-2 export lacks reviewer/time provenance. This limits any repeatability claim
  but no longer pauses model engineering; the targeted recovery is optional.
- Plate 26 is not a sealed test plate because its labels and prior runs are already present in the
  repository. Final validation requires a newly acquired prospective plate.
- The current six wells are all from Plate 23. They quantify internal model-evaluation variability
  but cannot establish biological treatment effects. Independent differentiation batches/plates,
  with the experimental unit and sample-size rationale declared prospectively, are still required.
- Tier-A conversion is reproducible and internally canonical but not scientifically released.
  A 2-D Desmin review cannot resolve missing-z ambiguity. Orthogonal evidence such as confocal
  z-stacks, a validated additional marker, and/or a Desmin-negative control well is still required,
  along with a field-sampling design and prospective biological replication. Desmin territory was
  not canonized by the audit.
- Classical v1 has six independently validated sealed folds. Its proposal-conditioned six-well
  diagnostic is not independent segmentation evidence: 350/375 GT masks have no recorded
  correction and 165/348 matches are pixel-identical. On the 25 accepted corrected masks, spanning
  only two wells, recall is 0.800, mean matched IoU 0.667, length MdAPE 0.317, and width MdAPE 0.078.
  No interval or candidate-selection claim is permitted from two corrected wells. Omnipose remains
  untrained despite a working harness, so T03 comparison and G-SO2 are not complete.
- Length, width, and multinucleation remain manual/QC-only until the prospective-plate release
  check passes.
- A future membrane-marker/z-stack acquisition cannot be implemented in software; it requires the
  planned imaging experiment and separate 3-D annotations.

This is deliberate fail-closed behavior. At present, total and valid nuclei may be reported only as
descriptive single-plate measurements with frozen mask sources and hashes. Conversion efficiency,
Desmin territory, and single-myotube measurements remain held from scientific release until their
respective gates pass.
