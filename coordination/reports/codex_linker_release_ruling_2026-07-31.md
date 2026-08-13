# Codex release ruling — fragment linker after the population safety review

Original ruling: 2026-07-31  
Amended: 2026-08-04 America/Los_Angeles  
Integrator: Codex  
Execution constraint: CPU only  
Ruling: **linker rejected; automatic use and manual-QC proposal use withdrawn**

## Decision

The sealed run in `PrecisionMyotube/runs/t02/classical_linker_v1/` remains a reproducible
application of the fold-refit linker at the locked `P >= 0.90` operating point. It is retained
only as rejected development evidence. It must not provide authoritative instance counts,
unattended measurements, manual-QC proposals, or proposals for new reviewed masks.

The 2026-07-31 ruling retained manual-QC proposal use because benefit had been measured while cost
had not. The predeclared uniform control-only round has now measured that cost, so the premise of
the earlier conditional retention is false. The linker branch is closed. Its threshold stays
locked for reproducibility; no result in the safety review was used to tune it.

## Decisive evidence

The control-only review sampled 10 accepted merges uniformly within each of all six wells, with no
flagged-case enrichment or density matching. The well-size-weighted population estimator and
interval were declared in the packet key before review.

- 60 cases were reviewed: 36 `different_myotubes`, 19 `same_myotube`, four `ambiguous_2d`, and one
  undecided.
- The population over-merge rate is **0.6487**, with stratified-bootstrap 95% CI
  **0.4497–0.8318**.
- That corresponds to approximately **350 wrong merges among 540 accepted** across the six wells.
- The earlier sparse-reference scoring found 11 fewer false splits. These denominators measure
  different evidence scopes, but even the safety interval's lower bound implies a cost roughly
  22 times that count; the point estimate is roughly 32 times it.
- Every well's reviewed rate is at least 0.375. Assigning all five excluded cases safe or wrong
  changes the population estimate only to 0.600 or 0.6769, so exclusion handling is not
  load-bearing.
- Confidence again predicts in the wrong direction: AUC 0.323, Mann–Whitney p=0.027 at n=55; 20
  of 25 merges scored at `P=1.0000` were judged wrong. Raising the threshold is not a supported
  mitigation.

The earlier `over_merge_count=3` remains correctly computed but is only a sparse-reference
flaggability ceiling: only three of 216 accepted merges in the two previously reviewed wells had
at least two reviewed reference masks and were examinable. Neither `3/3807` nor `3/375` estimates
accepted-merge error. The uniform raw-image review is the population safety evidence.

Reviewed-subset recall is also low-resolution: with 375 masks, one match moves pooled recall by
`1/375 = 0.0027`; the linker changes 348 to 349 matches. Precision and F1 are not detector metrics
because the reference set is not a complete field census.

## Constrained-merge defect fix

The legacy transitive closure could accept pairwise-plausible edges whose completed component no
longer had a coherent major axis. `constrained_merge` is now wired into the canonical runner under
a new candidate identity and records refused edges. It uses the linker's declared `cos_min`; the
minimum axis extent is derived from the existing endpoint-local window, so it introduces no new
operating parameter.

This is an infrastructure correction, not a linker rescue. The measured check still left 43
objects merging whole and 24 of those were judged wrong (56%). A constrained result is therefore
a new, unvalidated candidate under a new run ID, not a reinterpretation or rerun of sealed v1.
No six-well constrained run was launched because the parent linker branch is rejected.

The shared candidate finder now defaults to a global-axis gate. The canonical call sites make the
choice explicit: sealed-v1 reproduction passes `False`; the new constrained-v2 candidate passes
`True`. Training-pair reconstruction receives the same explicit choice. This prevents a future
v1 rerun from silently changing its candidate set.

## Binding rules

1. Keep `P >= 0.90` locked; never tune it on either review.
2. Retain sealed v1 only as a reproducible rejected development baseline.
3. Do not use linked output automatically, for authoritative counts, for manual-QC proposals, or
   as a proposal source for new reviewed masks.
4. Never report “3 over-merges for 11 fewer false splits.” Three is a flaggability ceiling; the
   population review measured 0.6487 (95% CI 0.4497–0.8318).
5. Do not use link-confidence escalation as a safety mitigation.
6. Any different candidate gate, merge policy, architecture, or retraining is a new candidate with
   a new run ID and requires independent validation; it does not reopen v1.
7. Preserve pre-link fragments and raw Desmin evidence in any historical inspection packet.

## T02/T03 consequence

T02 has no viable second candidate. The learned junction classifier remains built and shelved:
64.5% junction accuracy versus 23.8% for the classical rule, but only 1.8% operational reach and
approximately zero instance-level benefit. The fragment linker is now rejected. Omnipose is the
only remaining independent path and remains parked because of the unstable NVIDIA driver.

T03 therefore remains a complete official assessment of the classical floor plus a rejected
linked development variant, not a completed two-candidate selection. No biological treatment
claim is supported by the single-plate evidence.

## Verification authority

- Control-only score: `PrecisionMyotube/annotation_work/over_merge_c1/over_merge_c1.score.json`
  (`e49b3c416d549c269007cb6b8b3182819de2b5ccb499a0569d629889094cf64f`).
- Control-only decisions: `3874ae959f8014e252369650d6b81a891e15195b76004ddfcda8d8e0f7672784`.
- Predeclared key: `9a6bf6f9adb16db911c8c36edbebdc79e00b5c4d7c5da64e60d8ec891f9f9436`.
- Uniform cases: `4a66cebee21e028848fed7e3dc2237f1ddfcb9ba34cd45ac1fca80bb6ca6a37a`.
- Independent Codex reproduction was byte-identical to the canonical score.

No GPU, Omnipose, Tier-A acquisition, or `Conversion_Efficiency/**` action was taken. Nothing was
committed.

## 2026-08-04 implementation verification

- Full CPU suite: **409 passed**, with 16 existing Pydantic deprecation warnings.
- Official floor assessment v1.1: integrity passed; metrics unchanged; SHA-256
  `9feda342938d824d579eafc9ac4cbd346d12aad5026ff9d840274ee8c1f77de7`.
- Rejected linker assessment v1.1: all integrity checks passed; SHA-256
  `990d90f45842b25d590b0dabd955e85272f835f7c9485e60b24aa66669e3d9b5`.
- Sealed-v1 linker manifest: SHA-256
  `45daf9dda45ee7a0f51450eb9bacb5817736513b4baab52b90ac480bff8d5196`.
- T03 assessor source: `eedb398de34c9c0bfb4b46a48fc55cdc522b5b02c601daba5fcb2a9c6f22b173`.
- Canonical linker runner source:
  `5eb7d4026917b649dbb8984a354fe7f45cf6335d8f39e8c051d993e32e55006a`.
