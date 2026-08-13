# Precision Myotube version 2 workboard

Last reconciled: 2026-08-12 PT  
Plan: `PrecisionMyotube/DEVELOPMENT_PLAN.md`
Session handoff: `SESSION_HANDOFF_2026-07-22.md`

The version-2 plan supersedes the original manual's dual-annotator H01/H02/G1 path because the
project has one human operator. The old PDF and 100-task pilot artifacts are retained as history,
not as active blockers. Only the integrator edits this board.

## Current evidence

- The current combined CPU suite must be run in `pm-annotate`; test counts move as both lanes add
  regression coverage. Do not use the 2026-07-23 checkpoint totals as the current gate.
- Plate 23: 1,800 single-operator decisions across six wells.
- Current outputs: 377 complete, 31 border-truncated, 839 ambiguous, 553 rejected.
- Correction set: 40 pairs.
- Triage CSV: 961 current accept/reject rows; Codex confirmed an exact multiset match with the six
  current decision files and no duplicates.
- Official round-2 G-SO1 metric checks pass (90% agreement, zero unsafe border transitions,
  median IoU 1.0 over eight pairs). The provenance fix and 10-case recovery design are validated;
  the two complete-to-ambiguous cases are applied through `training_exclude.json` (377 to 375).
  Per the user's directive, this validation gap no longer blocks development.
- T01 bootstrap v1 is materialized: 375 real masks, 40 real correction pairs, and 2,290 eligible
  synthetic pairs. Twelve synthetic derivatives of excluded masks were also removed.
- Plate 26 is retrospective, not sealed; its labels and prior runs are already visible.
- The Tier-A read-only audit passes independent integration review: the 10 µm ring estimator with
  one pooled log-Otsu threshold (`440.76596787901417`) exactly reproduces all six wells. C08 is
  `3,341 / 10,114 = 33.03%`. This is canonical internal evidence, not scientific release.
- Labelling-standard disclosure: the first reviewed well's complete/candidate fraction is 0.500
  versus 0.204-0.257 in the next four treated wells. Review order is an mtime proxy and treatment
  confounds B06. B06 supplies 119/375 authoritative masks after exclusions. T03 retains the
  predeclared primary and adds a mandatory whole-well drop-one sensitivity.
- Omnipose is ported to klone. Stage 1 measured 20.7 s/epoch baseline, 31.4 s/epoch augmented,
  27.86 GB peak, and 133/256 augmented tiles. Stage 2 is launched as a two-task gap-off/gap-on
  array. These launch/probe facts are Claude-lane reports; no held-out result is integrated.

## Wave 0 - reconciliation

| Task | Owner | Status | Completion evidence |
|---|---|---|---|
| R01 - commit baseline and isolate worktrees | Project manager | Deferred, nonblocking | Intentional commit still requires separate authorization; implementation continues in the shared tree. |
| R02 - freeze six-well annotation snapshot | Claude produces; Codex validates | Complete | Six valid exports and six 300-row `reviewer_01` logs; hashes match; 377/31/839/553 = 1,800. See `reports/g_so1_validation_2026-07-21.md`. |
| R03 - rebuild triage data | Claude | Complete | Exactly 961 unique current rows; CSV SHA-256 `a1781a4c…c2fe4`, model SHA-256 `a0129ad3…716d`. |
| R04 - close single-operator UI gaps | Claude | Complete | Blind export now requires reviewer identity and records session/export/per-decision UTC timestamps; current annotation/model-lab suite has 49 passing tests. |
| C01-C04 - canonical engineering | Codex | Complete | Batch, adapters, benchmark, integrity/provenance, environment fingerprint, 31 tests. |
| C05 - statistical guardrails | Codex | Complete framework | Hierarchical technical/biological units, whole-unit bootstrap CIs, macro/micro benchmarking, multiplicity helper, and confidence-bound release gates. Current one-plate data remain descriptive for biology. Canonical total: 41 tests. |
| CL02 - overlap round trip | Claude | Complete | Overlap-safe masks, IDs, statuses, and logs survive round trip. |
| CL05 - six-well QC loop | Claude plus human | Complete as provisional bootstrap | All six Plate-23 queues reviewed; outputs remain single-operator and proposal-conditioned. |

## Wave 1 - single-operator readiness

| Task | Owner | Status | Completion evidence |
|---|---|---|---|
| SO01 - select 30 repeat cases | Claude produces; Codex validates | Complete | Round 2 has 30 unique blind cases, correct 10/5/10/5 strata, all six wells, and zero round-1 overlap. |
| H-SO01 - blinded repeat review | Human | Optional/deferred | No current human action. The 10-case recovery is retained only for a future repeatability claim. |
| G-SO1 - consistency evidence | Codex | Incomplete, nonblocking | Round-2 metrics pass but historical provenance is incomplete. This limits claims, not development. See `../PrecisionMyotube/HUMAN/g_so1_result.json`. |

## Wave 2 - low-labor model bootstrap

| Task | Owner | Status | Dependency |
|---|---|---|---|
| T01 - version bootstrap dataset v1 | Codex/Claude | Complete | `annotation_work/bootstrap_v1`: 375 real masks, 40 corrections, 2,290 synthetic pairs; manifest SHA-256 `44e38114…de94`. |
| T02 - train classical and Omnipose candidates | Claude model lane | Stage 2 running remotely; no sealed second candidate yet | Omnipose initializes from predeclared `bact_phase_affinity`; architecture/resume guards bind initialization. Klone Stage 2 is a two-task gap-off/gap-on array, six folds each. Classical v1 is sealed; junction model shelved; linker rejected. |
| T03 - six-fold leave-one-well-out benchmark | Codex scorer | Floor current; linked branch rejected; comparison pending | Assessments carry the labelling-shift disclosure and mandatory drop-one sensitivity without replacing the primary. Classical false splits: 52/375 primary, 34/256 omitting B06. Score Omnipose only after sealed handoff. |
| T03-LS - linker population safety | Human plus Claude/Codex | Complete; branch closed | 60 uniform accepted merges across six wells, predeclared weighted estimator, locked P>=0.90, sensitivity 0.600-0.6769. No more linker review is authorized. |
| AL01 - at most two 20-case active-learning rounds | Human plus Claude | Conditional | Only if a predefined metric improves. |
| G-SO2 - training evidence gate | Codex | Pending; disclosure specified | Disclose 0.500 first-well certification versus 0.204-0.257 next four, mtime order proxy, treatment confounding, 119/375 B06 authority, and paired drop-one T03 sensitivity. Also requires hash-frozen data, all folds, and leakage audit. |

T03 statistics are governed by `../PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md`: resample whole
held-out wells, report macro and micro metrics, and do not interpret six same-plate wells as
independent biological replication.

## Tier A - field-level release track

| Task | Status | Evidence or dependency |
|---|---|---|
| TA01 - reproduce method and bind provenance | Complete | Exact six-well reproduction, full hashes, pooled-threshold test, and read-only verification. |
| TA02 - reconcile the canonical estimator | Complete | The 10 µm ring/pooled-Otsu estimator is canonical internally. The older 6.6245% traced-fiber result is superseded provenance. |
| TA03a - orthogonal protocol/statistical contract | Ratified | Three-axis design accepted with population weighting, clustered intervals, disjoint calibration/validation, and adverse-bound gates. See `reports/codex_tier_a_validation_ratification_2026-07-23.md`. |
| TA03b - selector and scoring harness | Selector built, tests pending; scorer unblocked under amended design | Direct nucleus targeting is not certified. Build scorer for whole-field/mosaic z-stack reacquisition, DAPI registration, one-to-one nucleus matching, attrition, and existing inclusion weights. |
| TA03c - orthogonal biological validation | Pending; release-blocking | Raw ND2 has field-centre event XYZ but no certified pixel-to-stage affine. Operator must identify 561, confirm retained-sample/field reacquisition, and acquire z-axis/marker/negative-control evidence. |
| TA04 - sampling and prospective replication | Pending; release-blocking | Declare field sampling and experimental units, then acquire independent prospective biological replication. |

## Waves 3 and 4

| Task | Status | Rule |
|---|---|---|
| Retrospective comparison | Pending | Plates 28/32 line ROIs are weak length references; Plate 26 is disclosed retrospective data. |
| Workflow freeze | Pending | Freeze code/model/environment/data hashes before prospective data enter workspace. |
| Prospective validation | Pending | Requires a newly acquired plate held outside the workspace until predictions are sealed. |
| Scientific release | Pending | Total/valid nuclei may be descriptive; conversion efficiency, Desmin territory, and single-myotube claims remain held until their respective gates pass. |

## Immediate execution order

1. Preserve the locked linker only as rejected reproducible evidence. Do not use linked output
   automatically, for authoritative counts, or as a manual-QC/review proposal source.
2. Claude tests/freezes the Tier-A selector and builds the field-registration/post-hoc-matching
   scorer under the amended contract; do not acquire data or tune the threshold.
3. The operator identifies the 561-nm channel and decides whether matched z-stacks, a validated
   second marker, a Desmin-negative control, and the field-variance pilot are feasible.
4. Preserve the accepted Tier-A audit and cite the 10 µm ring/pooled-Otsu estimator as the canonical
   internal method; do not cite 6.6245% as the current conversion estimate.
5. Preserve both version-1.2 T03 assessments; cite the uniform six-well population review for
   linker safety and never present the three sparse-reference flags as its cost.
6. Monitor the remote Omnipose Stage-2 array without using intermediate held-out metrics for
   decisions. After all folds are sealed, hand them to Codex for independent T03.
7. Keep all future splits at the whole-well level and keep correction pairs out of training/tuning.
8. Leave the 10-case repeatability recheck optional and outside the development critical path.

The user does not need to complete the old 100-task dual-annotation pilot.

## Session close note

The current assessments, labelling-shift sensitivity, linker rejection, Tier-A audit, amended
relocalization contract, and remote Omnipose Stage-2 launch are recorded in version 2.7 of the plan
and dated reports. Continue remote T02 monitoring or TA03b scorer implementation. Do not
reopen completed annotation/bootstrap work, revert to the superseded 6.6245% estimate, or treat
either classical-floor variant as a completed model selection.
