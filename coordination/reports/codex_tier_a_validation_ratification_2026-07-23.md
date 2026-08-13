# Codex ratification — Tier-A orthogonal validation protocol

Integrator/statistical owner: Codex  
Date: 2026-07-23  
Proposal reviewed: `coordination/reports/claude_tier_a_validation_protocol_2026-07-23.md`  
Binding authority: `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md` version 1.1

## Ruling

The biological design is **conditionally ratified with statistical amendments**. Its three claims
and orthogonal modules are correct:

- V1 requires z-resolved evidence; another 2-D Desmin review is not a valid reference.
- V2 requires independently calibrated and validated threshold performance.
- V3 requires a measured field-variance pilot and independent biological replication.

Claude is authorized to build the deterministic stratified selector and 2-D-versus-3-D scoring
harness under the contract below. This does not authorize acquisition, alter the frozen production
method, release Tier A, or answer the operator's biological questions.

## 2026-08-12 relocalization amendment

Read-only inspection found field-centre XYZ coordinates, calibrated XY pixel size, and camera
rotation in all six original ND2 files, but no `pixelToStageTransformationMatrix`. The frame-level
XYZ accessor is constant across files and conflicts with the distinct, plate-plausible event-table
coordinates. Thus raw navigation metadata exists, but direct pixel-centroid-to-stage targeting is
not certified. The selector must remain `relocalization_feasible=false` unless a retained-sample
fiducial/calibration test validates the complete transform.

The acquisition/matching implementation is amended to whole-field or registered-mosaic 3-D
reacquisition followed by DAPI-based registration and prespecified one-to-one nucleus matching.
Selected-nucleus inclusion probabilities remain binding. Unmatched, duplicate, split, and
ungradable targets remain attrition and are never silently replaced. Desmin cannot be used to fit
the registration. Partial-field reacquisition requires an additional declared field/area inclusion
probability. Full details and the metadata table are in
`coordination/reports/codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md`.

## Required statistical contract

### Sampling and estimation

1. Use a two-stage design: prespecified spatially distributed fields, then nuclei sampled within
   field and ring/threshold stratum.
2. Retain well, field, nucleus ID, stratum, inclusion probability, RNG seed, source hash, and
   relocalization coordinates/feasibility for every selected record.
3. Boundary oversampling is encouraged, but primary population metrics must use inverse-probability
   or post-stratification weights. Unweighted results are stratum diagnostics.
4. Treat nuclei within a field as clustered. Use a field-cluster bootstrap or a prespecified
   hierarchical binary model. Never use nuclei as independent inferential `n`.
5. Preserve all missing, failed, and ungradable targets in the manifest with locked reasons. Do not
   silently replace a difficult target after its 2-D call or 3-D truth is known.
6. Declare the target population and condition mixture before 3-D labels are revealed. PPV and
   projection inflation are prevalence-dependent and must be recalculated for any different target
   population.

### Separation of calibration and validation

The cases used to choose a threshold cannot also certify it. Use disjoint biological material or a
prospectively locked nested split:

- calibration material may fit a threshold using a declared false-positive constraint;
- the threshold is then frozen;
- independent validation material supplies confirmatory sensitivity, specificity, predictive
  value, and inflation estimates.

The statement "440.8 is within the ROC optimal band" is rejected as written because the band and
loss are undefined. Threshold 440.8 passes only if its independently evaluated adverse bounds pass.
If a replacement threshold is calibrated, it becomes a new candidate and must be independently
validated before release.

### Acceptance criteria

Descriptive absolute conversion percentage is accepted only if **all** weighted, cluster-aware
adverse two-sided 95% confidence bounds pass:

| Metric | Gate |
|---|---:|
| Specificity | lower 95% bound ≥ 0.95 |
| Sensitivity | lower 95% bound ≥ 0.90 |
| Projection false-positive inflation (`1 - PPV`) | upper 95% bound ≤ 0.10 |
| Negative-control false-positive rate | upper 95% bound ≤ 0.05 |

The first three are conjunctive co-primary gates. Report them overall, by ring-score stratum, and by
condition; a pooled result may not conceal a boundary or condition failure.

### Sample-size method

- Begin the nuisance/variance pilot with at least eight prespecified fields in each of a control and
  a high-converter well, including a locked centre/edge or equivalent spatial scheme.
- Use the pilot to estimate between-field variance, intrafield correlation, stratum yields, and
  class prevalence.
- Select final fields and nuclei by reproducible cluster-aware Monte Carlo simulation. At the
  prespecified anticipated performance, the design must have at least 80% probability that all
  adverse-bound gates pass and must target no more than 10 percentage points 95% interval
  half-width within each boundary stratum.
- For the field-sampling endpoint, choose fields per well to target at most 5 percentage points
  absolute half-width for a 95% well-level conversion interval; 3 points is a higher-precision
  option. Use a checked beta-binomial/hierarchical model or cluster-aware simulation and round up.
- Eight fields in two wells are a technical pilot, not a universal sample-size result.
- Treatment-effect power must use between-independent-batch variance and a prespecified meaningful
  effect. At least three independent biological units is only a fail-closed minimum.

Therefore, "~100 nuclei per stratum" is acceptable as a rough pilot planning number only. It is not
the final sample-size justification because it ignores field clustering and unequal selection
probabilities.

## Method-transfer constraints

The raw threshold `440.76596787901417` is tied to the frozen acquisition/preprocessing intensity
scale. New confocal intensities cannot be compared directly with that raw number unless acquisition
and preprocessing are matched or a calibration rule is frozen. The clean comparison is the
original 2-D production score—or a matched projection processed by the frozen algorithm—against
the independently read 3-D truth.

The selector must expose whether original nuclei can actually be relocated. If stage registration
is unavailable, it must not pretend that pixel coordinates alone guarantee targeted reacquisition;
the alternative is prospectively selected matched z-stack fields followed by the frozen 2-D
projection analysis.

## Fold-change fallback correction

Failure of the absolute-percentage gate does not automatically authorize fold-change reporting.
Fold changes may remain exploratory only after a prespecified sensitivity analysis shows that
condition-dependent misclassification cannot reverse or materially distort the relative effect.
No treatment claim is permitted from the current same-plate wells.

## Existing 561 nm channel

Codex inspected the source ND2 metadata directly. Channel 0 is recorded only as `561`, with a
561-nm excitation line, 605/52 emission filter, and 500-ms exposure. The file contains no stain,
antibody, reporter, or biological identity. It therefore cannot be classified as MyHC or another
orthogonal marker from wavelength alone. The operator must resolve its identity from the
acquisition/staining record before any biological use.

Even if it is a useful second marker, a 2-D 561 image does not resolve the z-axis ambiguity by
itself. It may support marker specificity or threshold work, but Module A remains required for V1.

## Operator decisions still required

1. Identify the 561-nm channel from the laboratory record.
2. Confirm whether matched confocal z-stacks can be acquired and how fields/nuclei can be relocated.
3. Confirm whether MyHC or another validated marker is available.
4. Confirm whether a Desmin-negative control is available.
5. Confirm whether the variance-pilot field acquisition is feasible.

Until those decisions and the acquisition exist, Tier A remains internally reproducible but not
scientifically released.
