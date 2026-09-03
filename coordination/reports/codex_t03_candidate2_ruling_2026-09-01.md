# Official T03 ruling — tracer Candidate 2 of 2, 2026-09-01

## Decision

The one authorized Candidate 2 run completed successfully. Candidate 2 is
**not selected**. Candidate 1 remains the historical comparator, but **neither
candidate is the project's standing selected T03 candidate** because both miss
the recall floor by a large margin. No automatic instance-measurement or
scientific-release authority follows from this run.

Candidate 2 does what it was designed to do: it consolidates identities and
reduces the reference-detectable false-split count from 6 to 4. That two-object
improvement is accompanied by three fewer matched GT objects (recall 0.5573 to
0.5493), two additional reference-detectable over-merge flags (not a population
rate), 233 fewer predictions, no change in the primary median length MdAPE, and
substantially greater execution complexity. None of the paired whole-well
intervals excludes zero. Because this is submission 2 of 2 on the same sealed
set, promoting it merely because one point metric improved would also incur an
unaccounted selection/multiplicity cost.

## Authorized execution and integrity

The pre-result authorization and frozen hashes are recorded in
`coordination/reports/codex_t03_candidate2_authorization_2026-09-01.md`.
The command was executed exactly once:

```powershell
conda run -n pm-omnipose python model_labs/tracer_lab/eval_tracer_on_bootstrap.py --candidate 2
```

- Completed with exit code 0; no failed well or configuration.
- Output: `model_labs/tracer_lab/_runs/eval_bootstrap_candidate2/`.
- Summary SHA-256:
  `af3ec92c7d3bb01c5800ed50a936cea58529e53ab3ff776f464d7816f54e9b77`.
- All 36 prediction manifests carry version
  `cv_foldB02_weld_repair_v2` and checkpoint SHA-256
  `5725f8c1f61e85e74148cda3307bfda91eeb8b8f43d9f5d7ed577cd58f8c0fba`.
- The sealed bootstrap manifest remained byte-identical at
  `44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94`.
- Candidate configuration was frozen on PLATE_32 tune wells and claimed once
  on separate PLATE_32 wells. No prior Candidate 2 PLATE_23 output existed.

### Concurrent-source disclosure

During the run, another session rewrote `decompose_retrace.py` at 12:39:22,
after the 12:37:19 authorization hash was recorded. The concurrent diff added
`residual_trace`, `loop_pipeline`, and a separate `run_well(mode="loop")` path.
It did not change the authorized `apply_repair` function, the frozen constants,
or the evaluator, and none of the new functions is referenced by Candidate 2.
Therefore the evaluated candidate is unchanged regardless of whether Python's
first module import preceded or followed that filesystem update. The authorized
hash, rather than the later whole-file hash, remains the Candidate 2 source
record.

## Primary results and floors

Primary configuration: `nms_min50`. Intervals are exact percentile intervals
over all `6^6 = 46,656` whole-well bootstrap resamples. For false splits the
interval is expressed as the equivalent count per 375 GT objects so unequal
well sizes are preserved. These are internal Plate-23 model-evaluation
intervals, not biological-replicate inference.

| metric, in predeclared order | Candidate 2 | adverse 95% bound/interval | classical floor | floor ruling | Candidate 1 | C2−C1 |
|---|---:|---:|---:|---|---:|---:|
| median-well length MdAPE | 0.0864 | 0.0651–0.1038 | 0.3169 max | pass | 0.0864 | 0.0000 |
| false-split count | 4/375 | 2.09–5.84 equiv./375 | 52/375 max | pass | 6/375 | −2 |
| pooled recall | 206/375 = 0.5493 | 0.4777–0.6302 | 0.928 min | **fail** | 209/375 = 0.5573 | −0.0080 |

Candidate 2 beats the floor on matched-only length error and false splits, but
fails recall. Even the recall interval's upper endpoint is only 0.6302. The
standing width-cap diagnostic shows that this is a real coverage failure rather
than an 8-pixel ribbon artifact.

## Candidate 2 per-well primary results

| well | GT | predictions | TP | recall | false splits | length MdAPE | ΔTP vs C1 | Δsplits vs C1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 19_B06_act104_trka | 119 | 338 | 59 | 0.4958 | 1 | 0.0886 | −2 | −1 |
| 22_B03_act104_egfrc | 60 | 352 | 40 | 0.6667 | 1 | 0.0666 | 0 | 0 |
| 23_B02_ctrl | 35 | 132 | 12 | 0.3429 | 0 | 0.1110 | 0 | 0 |
| 29_C05_br223_egfrc | 59 | 286 | 30 | 0.5085 | 1 | 0.0843 | 0 | 0 |
| 32_C08_br223_igf1r | 54 | 373 | 36 | 0.6667 | 1 | 0.0635 | −1 | 0 |
| 33_C09_br223_trka | 48 | 305 | 29 | 0.6042 | 0 | 0.0966 | 0 | −1 |
| **pooled / median-well** | **375** | **1,786** | **206** | **0.5493** | **4** | **0.0864** | **−3** | **−2** |

Macro recall is 0.5474 mean / 0.5563 median. Per-well length MdAPE is
0.0851 mean / 0.0864 median. Per-well false-split rate is 0.0101 mean /
0.0125 median.

## Paired Candidate 2 minus Candidate 1 uncertainty

The same whole-well resample indices were used for both candidates.

| metric | point delta | paired whole-well 95% interval | interpretation |
|---|---:|---:|---|
| median-well length MdAPE | 0.0000 | −0.00028 to 0.00000 | effectively tied |
| false splits, equivalent count/375 | −2.00 | −4.12 to 0.00 | directionally fewer; not separated from tie |
| pooled recall | −0.0080 | −0.0142 to 0.0000 | directionally worse; not separated from tie |

No paired interval excludes zero. No formal multiplicity-adjusted hypothesis
test was predeclared, so none is invented after observing the result.

## Mandatory drop-one-whole-well sensitivity

The all-six-well result remains primary. Rates below use the remaining GT
denominator. `Δ` is Candidate 2 minus Candidate 1 under the same omission.

| omitted well | C1 splits/rate | C2 splits/rate | Δsplits | C1 recall | C2 recall | Δrecall |
|---|---:|---:|---:|---:|---:|---:|
| 19_B06 | 4/256 = 0.0156 | 3/256 = 0.0117 | −1 | 0.5781 | 0.5742 | −0.0039 |
| 22_B03 | 5/315 = 0.0159 | 3/315 = 0.0095 | −2 | 0.5365 | 0.5270 | −0.0095 |
| 23_B02 | 6/340 = 0.0176 | 4/340 = 0.0118 | −2 | 0.5794 | 0.5706 | −0.0088 |
| 29_C05 | 5/316 = 0.0158 | 3/316 = 0.0095 | −2 | 0.5665 | 0.5570 | −0.0095 |
| 32_C08 | 5/321 = 0.0156 | 3/321 = 0.0093 | −2 | 0.5358 | 0.5296 | −0.0062 |
| 33_C09 | 5/327 = 0.0153 | 4/327 = 0.0122 | −1 | 0.5505 | 0.5413 | −0.0092 |

The split direction and recall direction survive every omission; the effect
size remains small. Median-well length MdAPE is unchanged between candidates
under every omission. Omitting the suspected permissive first-reviewed B06
does not reverse the interpretation.

## Binding limitations

- This is Candidate 2 of 2 evaluated on the same sealed set. Its independent
  PLATE_32 freeze protects the individual run from tuning leakage, but selection
  between two observed submissions still carries multiplicity.
- Ground truth is single-operator and proposal-conditioned on the classical
  pipeline. The classical 0.928 recall floor has structural advantage.
- The 0.0864 length result is matched-only and says nothing about the roughly
  45% of certified fibres Candidate 2 did not match.
- Precision and F1 are not detector-performance estimates against this sparse
  certified subset and are not used in the ruling.
- The six reference-detectable over-merge flags are not a population
  over-merge rate; sparse reference coverage prevents that interpretation.
- All wells come from one plate. These results are retrospective engineering
  evidence, not treatment-effect or prospective-performance evidence.

## Standing status

- **Candidate 2:** evaluated, final, not selected.
- **Candidate 1:** retained as the historical T03 comparator, not promoted to
  automatic measurement.
- **Standing selected T03 candidate:** **none**.
- **Next model requirement:** improve full-fibre coverage/recall without
  post-hoc tuning on PLATE_23. Further identity-only repair is not the limiting
  step exposed by this benchmark.
