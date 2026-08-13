# Official T03 assessment — classical ridge/graph v1

**Integrator:** Codex  
**Date:** 2026-07-23  
**Candidate:** `classical_ridge_graph/v1`  
**Machine-readable result:** `PrecisionMyotube/runs/t03/classical_v1/assessment.json`
**Assessment SHA-256:** `b7ee25428872ed6a1ab6482d8187e226f86179588953c9bce2d2f9ee44da0f8b`

## Verdict

The sealed classical run passes artifact-integrity review and is retained as the reproducible
classical floor. It is **not selected** for automatic instance measurement. T03 candidate
comparison is incomplete, G-SO2 does not pass, and the current disposition remains
**manual-QC-only**.

This is an internal, retrospective, single-operator, proposal-conditioned evaluation on six wells
from Plate 23. It is not prospective performance and supports no biological treatment inference.

## Integrity result

Codex independently verified:

- the frozen bootstrap hash;
- six unique whole-well held-out folds;
- every fold's five-well training complement;
- the 32-point grid selection and lowest-index tie rule without using the held-out well;
- every prediction and evaluation-GT hash;
- prediction schema, image identity, unreviewed state, and scored status;
- the two binding exclusions;
- all independently recomputed fold metrics; and
- hashes and full-field reconstruction of all accepted correction masks.

Complete provenance is present in each hash-bound prediction sidecar. The corresponding
`InstanceSet.provenance` object is empty in all six predictions. This is a mitigated compliance
warning because each sidecar binds to the exact `InstanceSet` SHA-256; future exports should embed
the same provenance in the authoritative JSON.

## Six-well proposal-conditioned diagnostic

| Metric | Micro estimate | Whole-well 95% interval |
|---|---:|---:|
| Precision | 0.0659 | 0.0425–0.0959 |
| Recall / automatic coverage | 0.9280 | 0.8787–0.9519 |
| F1 | 0.1231 | 0.0811–0.1740 |
| Mean matched IoU | 0.9146 | 0.8908–0.9418 |
| False-split rate | 0.1387 | 0.0971–0.1906 |
| Over-merge rate | 0.0000 | 0.0000–0.0000 |
| Length MdAPE | 0.0000 | 0.0000–0.0000 |
| Width MdAPE | 0.0000 | 0.0000–0.0000 |

Counts are 375 GT masks, 5,279 predictions, and 348 matches. Intervals use 10,000 deterministic
resamples of entire held-out wells, seed `20260723`. The low precision is not ordinary detector
precision because reviewed-complete GT is sparse; unmatched predictions include uncertified
structures rather than a fully adjudicated background.

These pooled measurement numbers are also circular. Of 375 sealed GT masks, 350 have no recorded
human correction, and the candidate shares the proposal family that generated them. Of 348 matches,
165 (47.4%) are pixel-identical.

## Accepted-correction subset — primary meaningful floor

Only correction records with `action=accept` that remain reviewed-complete are eligible. That is
**25 masks**, not all 40 correction records:

- B06: 22 GT, 18 matches, recall 0.8182;
- B03: 3 GT, 2 matches, recall 0.6667;
- the other four wells contain no accepted corrected-complete GT.

Pooled corrected-subset results:

| Metric | Estimate |
|---|---:|
| Recall | 20/25 = 0.8000 |
| Mean matched IoU | 0.6667 |
| Median matched IoU | 0.6480 |
| False-split rate | 0.1200 |
| Over-merge rate | 0.0000 |
| Length MdAPE | 0.3169 |
| Width MdAPE | 0.0779 |

The subset spans only two held-out wells, so the binding statistical plan withholds confidence
intervals. Precision against a deliberately sparse 25-mask subset is not detector precision and
must not be used for selection.

## Correction to prior evidence

`model_labs/classical/_runs/v1/circularity_audit.json` treated all 40 correction records as though
they were reviewed-complete GT. Fifteen ended `ambiguous`; including them in the denominator caused
an impossible per-well unedited recall of 1.133 and produced the previously cited edited-subset
recall of 0.500. This official assessment supersedes that denominator. The valid accepted-correction
recall is 0.800, but its two-well coverage remains insufficient for an interval or release claim.

## Why T03 and G-SO2 remain incomplete

1. Only the classical candidate has completed real six-fold inference; Omnipose training is parked.
2. Non-circular accepted-correction evidence spans only two wells.
3. The sealed run has no locked density-stratum labels.
4. The candidate cannot represent overlapping crossing instances.
5. No prospective plate or independent biological replication exists.

The next decision is either to complete a genuinely independent second candidate or explicitly
revise the plan to a single-candidate feasibility closeout. Neither choice changes the requirement
for prospective validation before scientific release.
