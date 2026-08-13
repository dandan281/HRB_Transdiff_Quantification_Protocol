# Precision Myotube development plan and continuation record

Version: 2.7  
Effective date: 2026-08-12  
Last reconciled by: Codex integrator  
Biological decision owner: sole human operator  
Technical lanes: Codex canonical core/integration/benchmarking; Claude annotation tooling/model laboratories

Statistical authority: `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md`

## 1. Purpose and authority

This is the authoritative continuation document for the active PrecisionMyotube project. A new
session should read this file first, then `coordination/WORKBOARD.md`, then the task-specific
artifacts linked below. If an older PDF, report, request, or lane note conflicts with this plan and
the workboard, this plan and the current workboard win.

The original Parallel Execution Manual assumed two annotators plus an adjudicator. That labor is
not available. The project owner replaced it with a single-operator workflow. The historical
100-task dual-annotation pilot, H01/H02, and its G1 gate are not active requirements. They remain
only for reproducibility and must not be presented to the user as required work.

On 2026-07-21 the project owner explicitly directed implementation to continue without waiting for
the optional repeatability recheck. G-SO1 is an evidence and claim-quality check, not a software
development lock. Engineering may proceed immediately with explicitly provisional,
single-operator, proposal-conditioned data. An incomplete G-SO1 limits scientific claims; it does
not block T01, T02, T03, or other development.

No commit, push, branch rewrite, destructive cleanup, or worktree creation has been authorized.
R01 remains deferred and nonblocking. Preserve all existing untracked and modified files.

## 2. Immediate state and next action

The classical ridge/graph floor is sealed and its official single-candidate T03 assessment is
complete. The high-confidence fragment linker has now been **rejected** for both automatic use and
manual-QC proposal use. It is retained only as a reproducible rejected development baseline at its
locked `P >= 0.90` operating point.

The decisive predeclared control-only round uniformly sampled 10 accepted merges within each of
all six wells. It measured a well-size-weighted population over-merge rate of **0.6487** (stratified
bootstrap 95% CI **0.4497-0.8318**), implying approximately **350 wrong merges among 540 accepted**.
Sensitivity to the four `ambiguous_2d` cases and one undecided case is 0.600-0.6769. Every well was
at least 0.375. Confidence again ran backwards (`AUC=0.323`, Mann-Whitney `p=0.027`; 20/25
`P=1.0000` merges were judged wrong). The 11 fewer sparse-reference false splits do not offset this
measured cost. Never write "3 over-merges for 11 fewer false splits": the three remains only a
sparse-reference flaggability ceiling because 213/216 accepted merges in the earlier two-well
audit could not be examined by that rule.

The threshold stays locked only for reproducibility. Linked output must not be used for unattended
analysis, authoritative counts, manual-QC proposals, or proposals for new reviewed masks. The
approximately 60-case safety action is complete; no additional linker review or labeling is
authorized.

The legacy transitive-closure defect is fixed for future infrastructure by wiring
`constrained_merge` into a new constrained-v2 candidate identity and recording refused edges. The
sealed-v1 path explicitly keeps the pre-gate candidate finder and legacy merge policy; a gate-on or
constrained run is a new candidate under a new run ID. The constraint does not rescue the linker:
43 reviewed objects still merged whole and 24 were wrong (56%). No full constrained run is needed
for the closed branch.

The full T03 comparison remains incomplete because:

1. Omnipose Stage 2 has been launched remotely as a two-task gap-off/gap-on array, but no sealed
   six-fold candidate or held-out result has returned, so only one completed candidate exists;
2. the meaningful accepted-correction subset contains 25 masks from only two held-out wells, so
   whole-well confidence intervals are withheld;
3. the sealed run did not declare locked density strata; and
4. all current evidence is retrospective, single-operator, proposal-conditioned Plate-23 evidence.

The official artifact is
`PrecisionMyotube/runs/t03/classical_v1/assessment.json`; the readable decision report is
`coordination/reports/codex_t03_classical_assessment_2026-07-23.md`. The read-only Tier-A
reconciliation/provenance audit is also complete and independently accepted in
`coordination/reports/codex_tier_a_audit_acceptance_2026-07-23.md`.

The linked-candidate artifacts are
`PrecisionMyotube/runs/t02/classical_linker_v1/run_manifest.json` and
`PrecisionMyotube/runs/t03/classical_linker_v1/assessment.json`. Their current release ruling is
in `coordination/reports/codex_linker_release_ruling_2026-07-31.md`; the supporting human-review
evidence is `coordination/reports/claude_control_only_round_results_2026-08-04.md`.

On 2026-08-12 PT, Codex deliberately regenerated the official classical-floor assessment from the
current T03 source so the named artifact reproduces and carries the mandatory labelling-shift
sensitivity. Its SHA-256 is
`cc42b25bd0266119cc26a5780b8384d59b172fdfff8ef181dd57d2aaf8636bcf` (assessment
version 1.2); the superseded version-1.1 hash was
`9feda342938d824d579eafc9ac4cbd346d12aad5026ff9d840274ee8c1f77de7`, and version 1.0 was
`b7ee25428872ed6a1ab6482d8187e226f86179588953c9bce2d2f9ee44da0f8b`. Every numeric
metric is unchanged. The statistical-plan hash change from
`570d6c3edebe2e53c87f0daebbba9c42d1ce47a8e51cc9d15bbd8d604ee5e50f` to
`1ef189fa024d13dbb38b3d3f841d32c7e5fb5768c72a3d49898c48f397e7deca` is explicitly
accepted. Exact line-level provenance for the older file state is unavailable because the active
tree is untracked, but the current version 1.2 is the documented binding authority in
`coordination/reports/codex_tier_a_validation_ratification_2026-07-23.md`, and the recut changed no
numeric metric. Both current T03 assessments now bind that same current statistical authority;
the rejected linker assessment v1.2 SHA-256 is
`ceaf1ee4390dc4069bc9149d8a0feb07a698a66e2a748cdcc3e0af27162297f7`.

G-SO2 now has a binding labelling-standard disclosure. The first reviewed well was certified
complete at 0.500 versus 0.204-0.257 in the next four treated wells. Review order is inferred from
filesystem mtimes rather than logged, and the first well's unique treatment means biology cannot
be excluded. It contributed 120 complete calls before exclusions and 119/375 authoritative masks
after the binding exclusion. T03 keeps its predeclared all-six-well primary unchanged and adds a
mandatory drop-one-whole-well sensitivity beside it for every candidate. For the classical floor,
false-split rate is 52/375 = 0.1387 primary and 34/256 = 0.1328 with B06 omitted. No relabelling or
post-hoc reweighting is authorized. See
`coordination/reports/codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md`.

The project-canonical internal conversion estimator is now the declared 10 µm perinuclear-ring
method with one plate-wide pooled log-Otsu threshold (`440.76596787901417`) and canonical nucleus
area filter `[50,500] µm²`. It exactly reproduces all six wells; C08 is
`3,341 / 10,114 = 33.03%`. This is an internal method freeze, not a scientific release. Conversion
efficiency and Desmin territory remain held pending orthogonal evidence and prospective biological
validation.

The Tier-A orthogonal-validation design is statistically ratified with binding amendments in
`coordination/reports/codex_tier_a_validation_ratification_2026-07-23.md` and version 1.1 of the
statistical analysis plan. The deterministic selector is implemented but not yet test-covered.
Original ND2 event tables retain field-centre XYZ coordinates, calibrated pixel size, and camera
rotation; derived caches do not. Because the full pixel-to-stage affine is absent and frame/event
metadata conflict, direct per-nucleus targeting is not certified. Claude may build the scorer for
whole-field/mosaic reacquisition, DAPI-based registration, and prespecified one-to-one nucleus
matching. Acquisition remains an operator decision. The 561-nm source
channel has excitation/filter metadata but no recorded biological identity and cannot be treated as
an orthogonal marker until the laboratory record identifies it.

T03 can advance to a selection only after an independent second candidate is completed or the
project owner explicitly revises the plan to a single-candidate feasibility closeout.

## 3. Repository map and ownership

| Path | Role and authority |
|---|---|
| `PrecisionMyotube/` | Canonical schemas, analysis, manifests, adapters, gates, integrity checks, benchmarks, status, and this plan. Codex lane. |
| `annotation_tools/annotation_tools/qc_review/` | Human QC browser, blind-repeat workflow, correction export, and triage tooling. Claude lane. |
| `model_labs/` | Experimental model environments, training, inference, and canonical prediction export. Claude lane; not production models yet. |
| `Conversion_Efficiency/` | Validated field-level nuclei, Desmin territory, and conversion-efficiency baseline. Its validated Cellpose environment must remain isolated. |
| `MyotubePipeline/` | Legacy centerline workflow. Its free-line decisions are weak topology/length references, not full-area instance truth. |
| `Q_PLATES/` | Source ND2 images and legacy ImageJ ROI references. The required source plates are here. |
| `coordination/WORKBOARD.md` | Live cross-lane task status. Only the integrator should reconcile it. |
| `coordination/reports/` | Dated evidence and lane history. A stale lane report never overrides this plan or the workboard. |
| `coordination/requests/` | Unresolved cross-lane handoffs and requests. |
| `tmp/`, `output/`, generated HTML/TIFF/NPY/NPZ | Reproducible/generated artifacts unless a manifest explicitly makes them evidence. |

`archive/` and the older parallel-execution documents are historical. Do not silently restore
their labor assumptions, numeric targets, or gates.

The source data the project needs are already under `Q_PLATES/Q_Plates`, including Plates 23, 26,
28, and 32. Do not ask the user to relocate or re-upload them. Their roles differ: Plate 23 supplies
the active six-well bootstrap, Plates 28/32 are possible retrospective weak-reference sets, and
Plate 26 is retrospective and previously exposed.

## 4. Governing scientific rules

- Semantic Desmin territory and independent myotube instances are separate products.
- Field conversion efficiency does not require independent-myotube reconstruction.
- Only reviewed `complete` full-area instances may train or support length, width, or
  multinucleation measurements.
- `border_truncated`, `ambiguous`, `occluded`, rejected, disconnected, provenance-invalid, and
  binding-exclusion records are not complete training targets.
- In the low-labor QC interface, a would-be `occluded` object is conservatively recorded as
  `ambiguous` unless the canonical core is used to preserve `occluded` explicitly.
- Flat label TIFFs cannot faithfully represent overlapping instances. Canonical overlap-aware
  `InstanceSet` JSON is authoritative; raster labels are training conveniences with mappings and
  ignore masks.
- Model predictions are always `reviewed=false`; a prediction never grants itself human authority.
- Splits are by whole well/field. Never distribute crops from one well across training and held-out
  folds.
- Synthetic data may support pretraining or augmentation but never replace real held-out
  evaluation.
- Single-operator evidence must never be described as consensus, inter-rater agreement,
  independent ground truth, or prospective validation.
- Plate 26 is retrospective, not sealed. A genuinely prospective claim requires a newly acquired
  plate kept outside the workspace until code, model, thresholds, and workflow are frozen.
- Nuclei and myotubes nested inside a well are observational subsamples, not independent biological
  replicates. Every group analysis must declare raw-object, technical-unit, and biological-unit
  counts separately and follow `STATISTICAL_ANALYSIS_PLAN.md`.
- Current Plate-23 wells support descriptive summaries and internal model evaluation only. They do
  not support treatment-effect inference across biological experiments.

### Approved annotation vocabulary and interface record

The human operator approved `PrecisionMyotube/ANNOTATION_PROTOCOL.md`. Its decision terms are
binding:

- `complete`: the entire in-field object has one defensible identity, both ends and boundaries are
  sufficiently visible, and its full shape can be measured without inventing missing pixels;
- `border_truncated`: the object reaches the outer image boundary and appears to continue beyond
  the captured field; canonical analysis demotes any border-touching mask to this state;
- `occluded`: the identity/continuation remains clear, but a material internal part of the body or
  boundary is hidden, so full measurement could be biased; if multiple continuations are plausible,
  use `ambiguous` instead;
- `ambiguous`: the 2-D evidence does not support one defensible identity, connectivity, endpoint,
  or boundary; uncertainty is preserved rather than guessed away.

The practical order is border exit -> `border_truncated`; uncertain identity/connectivity ->
`ambiguous`; clear identity with hidden material geometry -> `occluded`; otherwise, only when fully
measurable -> `complete`.

The old `HUMAN/pilot_manifest.json`, reviewer templates, checklist, and G1 result are historical
100-task artifacts. A duplicate Codex-created `HUMAN/pilot_site` was intentionally removed after
the user identified overlap with Claude's annotation lane; that directory should remain absent.
Human-facing QC belongs to Claude's `annotation_tools/annotation_tools/qc_review/` workflow. The
completed 1,800-decision review and blind-repeat artifacts supersede the old pilot site.

## 5. Audited annotation evidence

The active six-well Plate-23 QC stream contains 1,800 decisions:

| Outcome | Count | Permitted use |
|---|---:|---|
| `complete` before disagreement exclusions | 377 | Candidate pool only. |
| Binding complete-to-ambiguous exclusions | 2 | Excluded from real training and every synthetic derivative. |
| Trainable `complete` after exclusions | 375 | Provisional bootstrap training targets. |
| `border_truncated` | 31 | QC/censored-case analysis; never complete targets. |
| `ambiguous` retained proposals | 839 | Uncertainty analysis and active-learning queue only. |
| Rejected proposals | 553 | Triage negatives only. |
| Human correction pairs | 40 | Separate real-error refinement/evaluation set: 35 `too_short`, 2 `spillover`, 3 `reshape`. |

The six decision exports and six 300-row reviewer logs validate. All logs identify `reviewer_01`.
The frozen six-well snapshot SHA-256 is
`5171286b5bcb153ad45cfe5db7ae532c4f6158a4f6fcaebbc25ad2649dc36994`.

The advisory logistic triage dataset has exactly 961 current accept/reject rows: 408 accepts and
553 rejects. It is an exact multiset match to the six current decision files, with no stale rows
and no duplicates. Its 0.889 training accuracy is in-sample triage evidence, not segmentation
performance.

| Triage artifact | SHA-256 |
|---|---|
| Current accept/reject CSV | `a1781a4cdae5c881a0179ce57575c5a2107dc7314c1a717ff85173dae50c2fe4` |
| Logistic triage model | `a0129ad36f9ec941974b28c129359373398a74c2910923dd080a064624dd716d` |
| Triage model summary | `89bb52c80d003de1910cbc95537aee6c8e6cfcb3440d711300ce82a575154736` |

Legacy ImageJ archives provide 7,156 readable free-line centerlines: Plate 23: 559, Plate 26: 344,
Plate 28: 1,019, and Plate 32: 5,234. Plate-23 C05 has one corrupt ROI zip. These line annotations
may support retrospective centerline/length checks but cannot validate full-area masks.

## 6. G-SO1: what happened and what it means

The sole operator completed a blinded 30-case repeat. It contained 10 complete, 5 border, 10
ambiguous, and 5 reject cases; included all six wells; used seed `20260723`; and had zero overlap
with round 1.

Numeric checks passed:

- disposition agreement: 27/30 = 90%;
- unsafe border-to-complete errors: 0;
- complete/complete comparable pairs: 8;
- pair IoUs: 1, 1, 1, 1, 1, 1, 1, and 0.7865466101694916;
- median complete-pair IoU: 1.0.

The historical round-2 export did not contain reviewer ID or review timestamps, so its washout and
identity cannot be proven. Therefore `PrecisionMyotube/HUMAN/g_so1_result.json` correctly remains
`passed=false`, `metric_outcome=pass`, `evidence_outcome=fail`, and
`development_blocking=false`.

The three disagreements were:

| Case | Well / instance | First -> repeat | Binding action |
|---|---|---|---|
| `case_02` | `19_B06_act104_trka / myotube_0377` | complete -> ambiguous | Exclude real mask and all derivatives. |
| `case_24` | `22_B03_act104_egfrc / myotube_0321` | complete -> ambiguous | Exclude real mask and all derivatives. |
| `case_25` | `22_B03_act104_egfrc / myotube_0113` | ambiguous -> reject | Already non-trainable. |

The exclusion manifest is
`PrecisionMyotube/annotation_work/training_exclude.json`, SHA-256
`b15492c167c8555dd8d306db5285792eea5ca6447cdc935268aa160d7ff847fb`.

Claude fixed the blind exporter so new exports require a reviewer and carry session-start,
export, and per-decision UTC timestamps. A provenance-clean 10-case recovery set was built with
seed `20260724`: 4 complete, 2 border, 3 ambiguous, 1 reject, four wells, zero prior-round overlap,
valid source lineage, no real source IDs/well names in the HTML, and no learned action shown.

| Optional recovery artifact | SHA-256 |
|---|---|
| Recheck key | `8784d3ec54f76f59df67ce5a8c911b9df43187d1eb2a26d15922e1abc939ce7f` |
| Recheck HTML | `0377ae15b3a2036f0ceaac7f46c7965eca4091e8161b1f92a8391be76a06251f` |

The original seven-day washout would allow a session after
`2026-07-29T02:23:02.904447Z` (`2026-07-28 19:23:02 PDT`; recommended date July 29). The project
owner subsequently deferred this recheck. Keep it only if a formal single-operator repeatability
claim is later desired. It is not current human work and must not pause model development.

## 7. T01 bootstrap v1: completed freeze

`model_labs/freeze_bootstrap.py` now performs the real exporter instead of aborting outside dry-run.
`model_labs/tests/test_freeze_bootstrap.py` adds two exporter tests. The output is materialized at
`PrecisionMyotube/annotation_work/bootstrap_v1`.

The output contains 39 files totaling 238,898,612 bytes (about 227.8 MiB): six well directories,
each with `image_fiber.tif`, `image_dapi.tif`, `labels.tif`, `ignore.tif`,
`instance_mapping.jsonl`, and `training_manifest.json`, plus the root
`bootstrap_manifest.json`, `corrections.jsonl`, and `synthetic.jsonl`.

| Bootstrap artifact | SHA-256 / count |
|---|---|
| `bootstrap_manifest.json` | `44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94` |
| `corrections.jsonl` | `6d6bd461b38c90c6c61879a3193633d13623c5b107611cdf3e43e38d1ae10e98`; 40 eligible |
| `synthetic.jsonl` | `3d8b7ad632dd94feac8bac78ed5f9f44eb76a8e75e0c1ae5a66dee7e20410a99`; 2,290 eligible |
| Real complete masks | 375 |
| Excluded real masks | 2 |
| Excluded synthetic derivatives | 12, six for each excluded real mask |

| Well | Complete kept | Excluded ID | Overlap pixels in real masks |
|---|---:|---|---:|
| `32_C08_br223_igf1r` | 54 | none | 0 |
| `19_B06_act104_trka` | 119 | `myotube_0377` | 962 |
| `22_B03_act104_egfrc` | 60 | `myotube_0321` | 0 |
| `29_C05_br223_egfrc` | 59 | none | 0 |
| `33_C09_br223_trka` | 48 | none | 0 |
| `23_B02_ctrl` | 35 | none | 0 |

All per-file hashes in the bootstrap manifest were independently rechecked. Label maxima and
mapping counts reproduce 375 instances, and every referenced correction/synthetic NPZ hash was
verified. The 12 removed synthetic records were derivatives of the two binding exclusions; the
source collection contained 2,302 before filtering.

Bootstrap usage is binding:

- real complete masks: exploratory development targets;
- synthetic pairs: pretraining/augmentation only;
- 40 real correction pairs: separate real-error refinement/evaluation, with any tuning use
  disclosed;
- split policy: six-fold whole-well leave-one-well-out;
- limitations: single operator, proposal-conditioned, not consensus, not prospective.

The bootstrap directory is generated/ignored and currently uncommitted. Do not delete it merely
because `git status` does not list it. If it is missing, regenerate with:

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python model_labs/freeze_bootstrap.py `
  --exclude PrecisionMyotube/annotation_work/training_exclude.json
```

The exporter refuses to overwrite an existing output. Do not remove or replace the frozen output
without first preserving its manifest and explicitly documenting the reason.

## 8. Completed engineering ledger

### Canonical core / Wave 0

- C01-C04 are complete: restartable batch manifests, prediction adapters, common benchmarking,
  integrity/provenance checks, environment fingerprints, deterministic selection, and release
  gates.
- C05 statistical guardrails are complete at the canonical framework level: hierarchical
  technical-to-biological aggregation, whole-unit bootstrap intervals, paired effects, descriptive
  Wilson intervals, Benjamini-Hochberg adjustment, macro/micro T03 summaries, and adverse-confidence-
  bound release checks. Prospective data and power justification remain future experimental work.
- R02 is complete: six-well snapshot reconciled, hashed, and validated.
- R03 is complete: triage data rebuilt from current decisions only; 961 current rows.
- R04 is complete: explicit conservative QC statuses, blind-repeat export, reviewer and timestamp
  provenance, border demotion, correction export, and overlap-safe behavior.
- CL02 is complete: masks, overlaps, IDs, statuses, and logs survive the annotation round trip.
- CL05 is complete as a provisional bootstrap: all six Plate-23 queues were reviewed.
- T01 is complete: bootstrap v1 is exported and independently validated.

### Canonical internal Tier-A estimator

The accepted frozen estimator uses a 10 µm perinuclear ring, one pooled log-uniform Otsu threshold
over all six Plate-23 wells (`440.76596787901417`), and canonical nuclei with physical area in
`[50,500] µm²`. There is no per-well threshold or tuning. The audit reproduces every declared count;
C08 has 10,114 canonical nuclei, of which 3,341 are positive (`33.03%`), versus `15.27%` in the
control well. Same-plate fold changes of `1.0–2.58×` are descriptive and are not treatment effects.

The C08 nucleus-count discrepancy is resolved. The canonical and MyoFuse-local mask files have
identical SHA-256 hashes and byte-identical arrays: 10,588 labels become 10,562 at the MyoFuse
`>=30 px` floor and 10,114 under the canonical area filter. The MyoFuse ring measurement drops two
additional objects, yielding its reported 10,560.

The earlier crossing-aware 50 µm traced-fiber result (`670 / 10,114 = 6.6245%`) is preserved only
as superseded method provenance; it is not the current conversion estimate. The absolute-threshold
`k` sweeps are diagnostics, not operating points. This audit canonizes the per-cell ring estimator
only; it does not canonize Desmin territory or establish biological correctness.

### Verification status at session close

- Canonical PrecisionMyotube suite: 44 passing tests at the T03 checkpoint.
- Annotation/model-lab suite: 160 passing tests at the official T03 checkpoint.
- The default base Python lacks `skan`; use `pm-annotate` for the current classical/model-lab suite.
- The earlier T01 checkpoint had 26 tests and the provenance-fix checkpoint had 24.
- T01 artifact hash, file hashes, counts, mappings, and referenced pair hashes were independently
  checked after export.

Use the commands in section 13 to reproduce these checks.

## 9. Work not yet implemented

### R01 - repository baseline/worktrees

Deferred and nonblocking. The repository remains on `main` at
`0322ebf534fa5c279f109c1145ce2da39fa69fe4` (`Add Q_Plates hand-label ground truth`), with earlier
commit `c5ee2bf`. There is one worktree. `.gitignore` is modified and most active code, plans,
evidence, and generated directories are untracked. This is deliberate preservation, not a clean
release state. Do not commit implicitly.

### T02 - real candidates

Partially complete. Candidate 1, `classical_ridge_graph/v1`, has six sealed whole-well folds, a
fold-honest 32-point grid, hash-bound predictions, and a complete run manifest under
`model_labs/classical/_runs/v1`. Candidate 2's real Omnipose harness, ignore policy, resumable
six-fold orchestration, sealing, and ablation plumbing are implemented and smoke-validated, but the
local GPU run was parked on 2026-07-23 because of workstation GPU/driver instability. The work was
ported to klone. Stage 1 measured 20.7 s/epoch baseline and 31.4 s/epoch with gap augmentation,
27.86 GB peak GPU memory, and 133/256 augmented tiles, so augmentation is demonstrably active.
Stage 2 is launched as a two-task array over gap-off/gap-on, six folds per task. Its predeclared
candidate initializes from `bact_phase_affinity`, chosen before Stage 2 and before any held-out
metric. `bact_phase_omni` is prohibited because it silently changes one-channel input to two and
selects the four-channel boundary head; an architecture assertion now fails loudly. Manifests bind
initialization name/hash and resume refuses cross-initialization sidecars. No Stage-2 held-out
result is yet integrated, so the two-candidate T02 contract remains incomplete.

Two additional classical-floor extensions have now been measured:

- The learned junction classifier is **built and shelved**. It raises junction-decision accuracy
  from the classical rule's 23.8% to 64.5% (2.7x), but reaches only 893 of 49,594 tracer pairing
  decisions (1.8%) and has approximately zero instance-level effect (recall delta -0.0118; F1
  delta -0.003). Three learning curves are flat. Preserve the implementation, but do not resume
  labeling or promote it unless upstream fragmentation changes its operational reach.
- The fragment linker is **rejected and its branch is closed**. The predeclared uniform six-well
  review measured population over-merge rate 0.6487 (95% CI 0.4497-0.8318), approximately 350
  wrong merges among 540 accepted, versus 11 fewer sparse-reference false splits. Automatic use,
  authoritative counts, manual-QC proposals, and reviewed-mask proposal sourcing are prohibited.
  Confidence remains anti-calibrated (`AUC=0.323`, `p=0.027`), so threshold escalation is not a
  mitigation. The locked v1 artifacts remain only as rejected reproducible evidence.
- A component-axis-constrained merge policy is wired under the new
  `classical_linker_constrained_v2` identity and records refused edges. It fixes a real transitive-
  closure defect but is not a rescue: 24/43 objects that still merged whole were wrong. No full
  constrained run is authorized for this closed branch.

The linked run preserves `4de4f383...a351f` as the source hash executed when its predictions were
generated. Its manifest separately binds current hashes for `linked_candidate.py`,
`link_candidates.py`, `link_model.py`, and `link_geometry.py`; T03 verifies those hashes while
preserving each historical executed-source hash. Sealed-v1 reproduction explicitly selects
`require_axis_agreement=False` and `legacy_transitive_closure`. The new constrained candidate
selects the axis gate and constrained closure under a new run ID.

The classical candidate's `InstanceSet` files are unreviewed and hash-bound to complete provenance
in adjacent prediction manifests. Their embedded `InstanceSet.provenance` objects are empty; T03
records this as a mitigated compliance warning rather than a hash-integrity failure.

### T03 - common evaluation

The official assessment of the sealed classical floor is complete, but T03 candidate selection is
not. Codex independently reproduced all six fold metrics, verified the bootstrap/prediction/GT
hashes, re-derived fold-honest parameter selection, confirmed both binding exclusions, and checked
that every prediction remains unreviewed.

The six-well proposal-conditioned diagnostic is 348/375 matched against 5,279 predictions:
micro precision 0.0659, recall 0.9280, F1 0.1231, mean matched IoU 0.9146, false-split rate 0.1387,
over-merge rate 0, length MdAPE 0, and width MdAPE 0. Whole-well 95% intervals are present in the
machine-readable assessment. These near-ceiling measurement errors are circular: 350/375 sealed GT
masks have no recorded human correction and 165/348 matches are pixel-identical.

The primary meaningful floor uses only the 25 accepted correction masks that remain
reviewed-complete. They span B06 and B03 only: recall 20/25 = 0.8000, mean matched IoU 0.6667,
median matched IoU 0.6480, false-split rate 0.1200, length MdAPE 0.3169, and width MdAPE 0.0779.
Precision against this deliberately sparse subset is not detector precision. Because only two
whole wells contain accepted corrections, inferential intervals are withheld.

The linked candidate reproduces 349/375 reviewed-subset matches and 41 reference-detectable false
splits at the locked operating point. Recall is valid only as a descriptive selected-subset metric:
one object changes pooled recall by `1/375 = 0.0027`, so the 348-to-349 difference is flat,
low-resolution evidence rather than a recall benefit. Precision and F1 remain uninterpretable as
detector metrics. The three stored over-merge flags are not an error rate and must never be framed
as the measured cost of the linker; reference sparsity made 213/216 accepted merges in the two
audited wells ineligible for that flag. The separate uniform six-well raw-image review measured
the population over-merge rate as 0.6487 (95% CI 0.4497-0.8318), approximately 350/540 accepted
merges. This rejects both automatic use and manual-QC proposal use; only reproducibility survives.

Claude's earlier `circularity_audit.json` used all 40 correction records as its denominator even
though 15 ended `ambiguous`; this produced an impossible per-well recall above 1.0. The official
T03 assessment supersedes that denominator and the earlier claimed edited-subset recall of 0.500.
No candidate is selected and G-SO2 remains pending. The linker disposition is rejected development
baseline only; its uniform safety sample is complete and the branch must not be reopened by more
linker labeling or threshold tuning.

### AL01 - bounded active learning

Conditional, not scheduled. If T02/T03 shows a predefined need, show at most 20 uncertain or
structurally novel cases per round for at most two rounds. Maximum possible added Wave-2 labor is
40 decisions. Do not ask for it before model evidence demonstrates value.

### G-SO2 and later waves

Pending T02/T03. Retrospective comparison, workflow freeze, prospective validation, and scientific
release remain future work. A new prospective plate does not yet exist in the project.

### Annotation import passthrough

`coordination/requests/claude/2026-07-16-instances-import-passthrough.md` remains unresolved. It
requests an overlap-safe canonical `InstanceSet` CLI import passthrough or an explicit warning.
There is no current analysis blocker because `analyze --instances` already works, but the interface
request should be resolved before polishing the production annotation workflow.

## 10. T02 implementation contract

Claude's model lane owns T02. It may begin immediately.

Required candidate order:

1. a deterministic classical ridge/graph baseline as the reproducible floor;
2. real Omnipose transfer/fine-tuning on the bootstrap masks;
3. micro-sam only if it adds measurable value after the first two;
4. a learned junction classifier only after real split examples justify it.

Item 4 has since been implemented, measured, and shelved for its 1.8% operational reach and
approximately zero instance-level effect. The fragment linker was also implemented as a bounded
extension of item 1 and rejected after its population safety review; it is not a manual-QC tool,
not a substitute for the missing independent candidate, and not a production method.

For each of the first two candidates:

- document an isolated environment; never install into `Conversion_Efficiency/cpenv`;
- record exact command, package/environment hash, checkpoint hash, input-manifest hash, seed,
  channels, thresholds, fold, timing, and failures;
- run six folds, holding out one complete well at a time;
- prevent crop, correction, and synthetic derivative leakage;
- do not alter candidate selection after inspecting a held-out fold;
- export predictions through the canonical adapter as overlap-aware, unreviewed `InstanceSet`
  JSON; a label TIFF may be emitted only as a non-authoritative convenience;
- keep the 40 real correction pairs separate and disclose any tuning use;
- label all results exploratory, single-operator, proposal-conditioned, retrospective development
  evidence.

T02 is complete only when both candidate commands run from documented environments, all six folds
exist for each candidate, predictions validate and remain unreviewed, leakage checks pass, and a
compact run manifest records hashes/seeds/timing/failures.

## 11. T03 and G-SO2 contract

Codex scores sealed T02 predictions with the same precision-first benchmark. For every candidate
and fold, report:

- instance precision and recall;
- matched-mask IoU;
- split and merge errors;
- length and width error;
- fraction excluded as ambiguous;
- per-well and density-stratified behavior;
- automatic coverage and any failures.

T03 must report micro estimates, macro mean/median across all six wells, and 95% intervals produced
by resampling entire held-out wells. Individual myotubes must never be resampled as if they were
independent folds. These intervals quantify internal model-evaluation variability only because all
six wells are from Plate 23.

The development benchmark is proposal-conditioned internal evidence. It is not prospective
performance. G-SO2 passes when dataset and prediction hashes are frozen, all six folds ran, field
leakage is absent, exclusions and failed folds are visible, G-SO1 status is disclosed, and no
result is described as consensus or independent truth. Incomplete G-SO1 narrows the claim but does
not invalidate the engineering comparison.

## 12. Release tiers and future goals

### Tier A - mature field-level outputs

Total and valid nuclei are the most mature outputs and may be reported as descriptive single-plate
measurements once mask sources and hashes are frozen. The per-cell conversion estimator is
reproducible and internally canonical, but conversion efficiency and Desmin territory are not yet
scientifically released. Their release requires orthogonal evidence capable of resolving the
missing z-axis ambiguity, such as confocal z-stacks, a validated additional marker, and/or a
Desmin-negative control well, plus declared field sampling and prospective biological replication.
Same-plate fold changes remain descriptive.

The ratified validation gate uses a two-stage field/nucleus sample, recorded inclusion
probabilities, population weighting for threshold-stratified oversampling, and field/biological-unit
clustered intervals. Calibration and validation material must be disjoint. Absolute conversion
requires all adverse two-sided 95% bounds to pass: specificity lower bound `>=0.95`, sensitivity
lower bound `>=0.90`, projection false-positive-inflation upper bound `<=0.10`, and
negative-control false-positive-rate upper bound `<=0.05`. Failure does not automatically authorize
fold-change release.

### Tier B - exploratory single-myotube outputs

Length, width, instance count, and nuclei per myotube may be developed with the current labels but
must remain exploratory and single-operator until their own prospective gate passes.

### Tier C - prospective single-operator validation

After T02/T03 candidate selection, freeze code, model, environment, thresholds, review protocol,
and data hashes. Acquire a new plate afterward and keep it outside the repository until predictions
are sealed. The sole operator then reviews a stratified 30-case validation sample, with 10 repeated
after washout to monitor drift. Report prospective performance and measurement error with the
single-operator limitation prominent. This still does not establish inter-rater reproducibility.

Before acquisition, declare the biological experimental unit, primary outcome/comparison, exclusion
rules, multiplicity family, scientifically meaningful effect, and an a priori precision or power
calculation. A minimum of three biological replicates is a software guardrail, not an assertion that
three is adequately powered. Authoritative release requires every metric's adverse 95% confidence
bound, not only its point estimate, to pass.

Future ordered goals are:

1. preserve the fragment linker only as rejected reproducible evidence; do not use it for
   automatic output, authoritative counts, or manual-QC/review proposals;
2. test/freeze the implemented Tier-A selector and build the amended whole-field-registration
   2-D-versus-3-D scorer, while the operator resolves 561 identity and sample availability;
3. acquire the orthogonal Tier-A evidence and variance pilot, then determine the final
   cluster-aware sample size before confirmatory validation;
4. monitor the remote Omnipose Stage-2 array; after all twelve folds are sealed, independently
   verify hashes/leakage and run official T03 without selecting on intermediate held-out results;
5. complete T03 candidate comparison and select or reject candidates using predefined thresholds;
6. use AL01 only if it is quantitatively justified;
7. pass G-SO2 and freeze the workflow;
8. perform disclosed retrospective weak-reference comparisons on Plates 28/32 and, if useful,
   Plate 26;
9. acquire and evaluate a genuinely new prospective plate;
10. release only the measurement tiers that pass their own gates.

Failure narrows the scientific claim; it never lowers the threshold.

## 13. Reproduction and verification commands

Run from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python -m pytest PrecisionMyotube/tests -q --basetemp tmp/pytest_precision_plan
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_labs_plan
python model_labs/freeze_bootstrap.py --dry-run `
  --exclude PrecisionMyotube/annotation_work/training_exclude.json
Get-FileHash PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json -Algorithm SHA256
```

Test counts move as both lanes add regression coverage; derive the current total from the command
rather than treating an older count as a gate. The dry run should reproduce 375 real complete
masks and the same exclusions without changing the materialized freeze.

## 14. Known operational cautions

- Native napari previously failed OpenGL initialization on this workstation. The serverless/static
  QC path is the practical annotation fallback.
- A prior attempt to open the static recheck page through the in-app browser found no browser
  backend; structural validation of the HTML was completed instead. This does not affect T02.
- Keep explicit `--basetemp` paths under `tmp/` when running pytest to avoid workstation temp-path
  issues.
- The base Python environment lacks `skan`; run annotation/model-lab tests in `pm-annotate`.
- Do not train on `border_truncated`, `ambiguous`, rejected, occluded, or excluded masks.
- Do not restore the 20 stale C08 triage rows; the authoritative current count is 961, not 981.
- Do not call Plate 26 locked, blinded, sealed, or prospective.
- Do not treat the 7,156 legacy free-line ROIs as full-area masks.
- Do not treat deterministic fallback smoke tests or stub checkpoints as trained models.
- Do not use pooled nuclei/myotube counts as treatment-replicate `n`; legacy pooled plate summaries
  are descriptive totals only.
- Do not modify the validated Cellpose/nucleus environment while solving model environments.
- Generated bootstrap artifacts are currently ignored/untracked; preserve them by path and hash.
- Claude's `coordination/reports/claude_report.md` is lane history and contains older gating
  language. Do not rewrite Claude's report; use this plan and the current workboard for status.

## 15. Human labor and stop conditions

| Stage | Required new decisions |
|---|---:|
| Current T02/T03 development | 0 |
| Linker control-only safety gate | 0 new; 60 completed and branch rejected |
| Optional G-SO1 recovery | 10, only if a formal repeatability claim is wanted |
| Conditional Wave-2 active learning | 0-40, only if metrics justify it |
| Future prospective validation | 40, only after workflow freeze |

The sole operator has no action required for the rejected linker branch.
Agents should not ask the user to complete the retired 100-task pilot or the optional recovery
before continuing engineering. Do not request more linker review unless a genuinely new
architecture is separately authorized and pre-registered.

The session may be considered safely handed off when the workboard agrees with this file, the
current CPU-only combined suite passes, the bootstrap hash matches, and the next session continues
the active T02/T03 or validation dependency rather than reconstructing or reopening completed
annotation work.
