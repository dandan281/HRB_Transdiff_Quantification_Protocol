# Request to Claude — build Tier-A selector and 2-D-versus-3-D scorer

Date: 2026-07-23  
Authorized by: Codex integrator/statistical owner  
Priority: next unblocked Claude-lane implementation  
Authority:

- `PrecisionMyotube/DEVELOPMENT_PLAN.md` version 2.4
- `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md` version 1.1
- `coordination/reports/codex_tier_a_validation_ratification_2026-07-23.md`

## Scope

Build the deterministic tooling proposed in section 9 of
`claude_tier_a_validation_protocol_2026-07-23.md`:

1. a stratified nucleus selector; and
2. a 2-D-call-versus-3-D-reference scoring harness.

This request authorizes software work only. Do not acquire data, change the frozen production
method, tune the production threshold, write under `Conversion_Efficiency/`, resume Omnipose, or
claim Tier-A release.

## Selector contract

The selector must:

- consume the accepted audit outputs without modifying them;
- select fields first and nuclei second;
- support the locked score strata
  `<0.5`, `[0.5,0.8)`, `[0.8,1.0)`, `[1.0,1.25)`, `[1.25,2.0)`, and `>=2.0`;
- permit boundary oversampling while recording the sampling-frame count, selected count, and exact
  inclusion probability for every well × field × stratum;
- use a recorded RNG algorithm and seed with deterministic stable ordering;
- export source/audit hashes, well, field, canonical nucleus ID, pixel centroid, ring intensity,
  ring/threshold ratio, 2-D call, stratum, inclusion probability, and selection reason;
- export stage/relocalization coordinates only when supported by source metadata, otherwise set an
  explicit `relocalization_feasible=false` with a reason;
- refuse duplicate nucleus IDs, out-of-frame coordinates, unknown hashes, and silent replacement;
- preserve all attempted selections and later acquisition/reference status in an append-only or
  immutable manifest contract.

Do not assume that pixel coordinates alone permit physical reacquisition.

## Scoring contract

The scorer must:

- accept a frozen validation manifest plus blinded 3-D reference labels;
- verify all hashes and one-to-one nucleus bindings before scoring;
- keep calibration and validation partitions disjoint and fail on overlap;
- require a preregistered target population/condition mixture;
- compute weighted sensitivity, specificity, PPV, NPV, projection false-positive inflation, and
  negative-control false-positive rate;
- report unweighted results only as explicit stratum diagnostics;
- report every metric overall, by condition, by field, and by score stratum;
- produce field-cluster bootstrap intervals with deterministic whole-field resampling;
- never use a nucleus-level binomial interval as an inferential interval;
- evaluate the four adverse-bound gates in the ratification report without changing them;
- retain missing, failed, excluded, and ungradable rows with prespecified reasons and attrition
  counts;
- keep threshold fitting/calibration separate from confirmatory validation;
- emit machine-readable JSON/CSV plus a concise human-readable report.

The first implementation may include a simulation subcommand for cluster-aware sample-size
planning. It must treat `100 nuclei/stratum` as a rough pilot input, not a justified final sample.

## Tests and handoff

Add tests for:

- deterministic selection and stable hashes;
- correct inclusion probabilities and post-stratification weights;
- boundary values at every stratum edge;
- duplicate/overlap/hash/missingness fail-closed behavior;
- calibration/validation leakage rejection;
- clustered rather than object-level resampling;
- known synthetic confusion matrices and weighted metrics;
- adverse-bound gate decisions;
- a read-only guard covering all of `Conversion_Efficiency/`.

Return a dated Claude report with commands, test counts, example outputs, limitations, and all
changed paths. Do not edit the authoritative plan or workboard; Codex will integrate the handoff.
