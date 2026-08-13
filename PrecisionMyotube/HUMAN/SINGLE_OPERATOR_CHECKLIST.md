# Active single-operator checklist

## Agents prepare

- [x] Six current decision files are hash-frozen.
- [x] Six instance exports and review logs use stable reviewer ID `reviewer_01`.
- [ ] Provenance records single-operator mode, tool version, package hash, and decision hash.
- [x] Triage CSV is rebuilt to exactly 961 current accept/reject rows.
- [x] Thirty repeat tasks are selected across six wells and required outcome strata.
- [x] Blind-repeat page hides prior answers and model suggestions.
- [x] Corrected blind export records reviewer plus session/export/per-decision UTC timestamps.
- [x] Targeted 10-case recovery design and two-case exclusion list pass Codex validation.
- [x] T01 development dataset is built with 375 real masks, 40 corrections, and 2,290 eligible
  synthetic pairs.

## Human completes

- [x] Review all 30 round-2 blind tasks.
- [x] Save/export the repeat dispositions.
- [ ] Optional: complete the targeted recheck later if repeatability evidence is wanted.

## Codex evaluates

- [ ] Every repeat task has complete lineage and reviewer metadata.
- [x] Disposition agreement is at least 85 percent: 90%.
- [x] No border case is unsafely marked complete.
- [x] Median repeat-complete mask IoU is at least 0.80 with at least eight pairs: 1.0 over eight.
- [x] Every disagreement is ambiguous or excluded from development training.
- [x] Gate report states that inter-rater agreement was not measured.
- [x] Project owner authorized exploratory model development without waiting for `G-SO1`.
