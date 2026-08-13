# Repository audit - 2026-07-21

## Conclusion

The active project is no longer accurately described by the original dual-annotator manual.
Claude's CL05 work has produced a complete single-operator pass over six Plate-23 review queues.
The feasible path is to reconcile that evidence, measure test-retest consistency on 30 blinded
cases, and use bounded active learning rather than broad manual annotation.

## Active and legacy systems

| System | Finding |
|---|---|
| `PrecisionMyotube` | Active canonical analysis and validation layer; 31 tests pass. |
| `annotation_tools/qc_review` | Active single-operator UI; provenance/border fixes and correction export are implemented. |
| `model_labs` | Export and proof-of-loop infrastructure; real segmentation training has not started. |
| `Conversion_Efficiency` | Active validated semantic/nucleus reference for field-level results. |
| `MyotubePipeline` | Older centerline pipeline. Contains useful human topology preferences but not full-area truth. |

## Verified current counts

- Six Plate-23 decision files, 300 decisions each: 1,800 total.
- 408 accepts: 377 complete plus 31 border-truncated after safe demotion.
- 839 ambiguous decisions/instances.
- 553 rejects.
- Six canonical QC instance exports and six reviewer logs use reviewer `pilot`.
- 40 correction pairs: 35 `too_short`, 2 `spillover`, 3 `reshape`.
- Current triage table: 981 rows, 426 accepts, training accuracy 0.89.
- Current decision files contain only 961 accept/reject rows; 20 old C08 rows remain stale in the
  accumulated training CSV.
- Latest illustrative C08 loop: 54 complete masks; baseline precision/recall 0.108/1.000;
  QC-filtered 0.203/0.815. These are circular development metrics, not scientific results.
- Canonical test suite: 31 passed.
- Annotation/model-lab suite: 21 passed with 16 third-party Pydantic deprecation warnings.

## Data leakage and reference-label audit

- Q_PLATES contains 7,156 readable ImageJ free-line ROIs: Plate 23 = 559, Plate 26 = 344,
  Plate 28 = 1,019, Plate 32 = 5,234.
- Plate-23 C05 ROI archive is corrupt and unreadable.
- These are free-line/centerline references, not full-area instance masks.
- Plate 26 labels are committed and the legacy pipeline has already run on Plate 26. It is not a
  genuinely sealed test set.
- A future prospective plate held outside the workspace is required for a blind final test.

## Remaining technical issues incorporated into plan v2

1. Commit and isolate the currently untracked active stack.
2. Reapply six wells with stable reviewer ID and hashed decision/package provenance.
3. Rebuild triage training data from current decisions to remove 20 stale rows.
4. Add blind-repeat mode and a 30-case single-operator repeatability manifest.
5. Treat occluded cases conservatively as ambiguous unless explicitly labeled in the full tool.
6. Never describe current labels as consensus, independent ground truth, or prospective evidence.

The replacement plan is `PrecisionMyotube/DEVELOPMENT_PLAN.md`.
