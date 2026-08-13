# Statistical analysis plan

Version: 1.1  
Effective date: 2026-07-22  
Owner: Codex canonical integration/statistics lane  
Status: binding for T03, prospective validation, and scientific release

## Purpose

This plan prevents pseudoreplication and separates three different questions:

1. Did the image-analysis system measure a field accurately?
2. How variable is model performance across held-out wells or fields?
3. Is there a biological treatment effect across independent experiments?

Thousands of nuclei or myotubes improve measurement detail inside a well. They do not create
thousands of independent biological replicates. Every analysis must declare the experimental unit
before looking at group results.

## Unit hierarchy

Use the most specific metadata available:

`object (nucleus/myotube) -> field -> well -> plate/differentiation batch -> experiment`

- Objects are observational subsamples.
- Multiple images from one well are technical subsamples.
- Wells may be technical replicates or experimental units depending on how treatment was assigned.
- A plate is not automatically a biological replicate. The true biological unit is the independently
  prepared culture/differentiation batch to which treatment could independently have been assigned.
- Treatment inference requires independent biological units, not merely different image files.

The analysis manifest must name `biological_unit_key` and, when present, `technical_unit_key`.
Technical observations are collapsed before biological comparisons. The output reports raw object
count, technical-unit count, and biological-unit count separately; only the last is reported as
inferential `n`.

## Current data limitation

The present six reviewed wells all come from Plate 23. They support:

- descriptive field/well summaries;
- internal six-fold model evaluation;
- identification of failure modes;
- variability across the six held-out wells.

They do not support a general treatment-effect claim across biological experiments. No p-value,
narrow object-level confidence interval, or large nucleus count can remove this limitation.

## Required summaries

For every outcome and condition report:

- exact raw object count;
- exact technical-unit count;
- exact independent biological-unit count;
- all biological-unit summary values or a machine-readable table containing them;
- mean and median;
- between-biological-unit standard deviation when at least two units exist;
- an unstandardized effect size for every planned comparison;
- a 95% confidence interval resampling or modeling the independent biological units;
- all exclusions and whether they were specified before analysis;
- missing fields, failed runs, and attrition without silent deletion.

Plots must show biological-unit points. Object-level distributions may be shown as secondary detail
but must not visually imply that objects are independent replicates.

## Confidence intervals

- Binary counts within one field may use a Wilson interval, explicitly labeled as descriptive
  counting uncertainty rather than biological uncertainty.
- Group estimates use whole-biological-unit resampling or an appropriate hierarchical model.
- T03 model-performance intervals resample entire held-out fields/wells, never individual matched
  myotubes.
- If fewer than three independent units are available, inferential intervals are withheld and the
  result is labeled descriptive only.
- Three units is a software fail-closed minimum, not proof of adequate power. The prospective study
  requires an a priori precision/power justification using a scientifically meaningful effect and
  variance estimated from independent pilot experiments.

## Treatment comparisons

- Prefer estimation: absolute effect, scientifically interpretable relative effect when appropriate,
  and 95% confidence interval.
- If the same biological batch contains multiple treatment conditions, use a paired comparison or a
  mixed model with batch as a random/blocking effect.
- For independent groups, compare biological-unit summaries or fit a model whose covariance reflects
  the declared hierarchy.
- Do not select paired versus unpaired analysis after inspecting which gives a better result.
- Do not run hypothesis tests on technical replicates as though they were biological replicates.

## Multiple outcomes and comparisons

The primary outcome and primary comparison must be declared before prospective labels are revealed.
Secondary outcomes and exploratory subgroup analyses must be labeled as such.

If confirmatory hypothesis tests are used:

- define the family of tests in advance;
- report exact unadjusted and adjusted p-values;
- use Holm control for a small family where family-wise error is required, or Benjamini-Hochberg when
  false-discovery-rate control is the declared objective;
- always report the effect and confidence interval alongside the p-value;
- never use `p < 0.05` alone as a release criterion.

The canonical code provides Benjamini-Hochberg adjustment but does not automatically invent a test
family or p-values.

## Outcome-specific rules

### Conversion efficiency and fusion index

Compute the numerator and denominator per field/well, then summarize within the declared biological
unit. Do not pool all nuclei across treatment groups for inference. A pooled plate total is allowed
only as a descriptive measurement total. For multi-plate inference, use biological-unit proportions
or an appropriately checked binomial mixed model with plate/batch effects.

### Length, width, and nuclei per myotube

Object distributions are usually skewed. Report median, quartiles, and the complete biological-unit
summary distribution. Treatment effects must be estimated across independent biological units.
Object-level mixed models may be used only if the nesting structure and residual diagnostics are
recorded; they do not change the biological replicate count.

### Nucleus totals

Report total and valid nucleus counts per field plus segmentation sensitivity analyses. Biological
comparisons use independent-unit summaries, not every nucleus as `n`.

### Orthogonal validation of conversion efficiency

Validation of a 2-D Desmin-ring call must use an orthogonal reference capable of resolving its
failure mode. Another review of the same 2-D Desmin projection is not an independent reference.
Confocal z localization, a biologically validated second marker, and a true-negative control are
acceptable complementary axes when their definitions are locked before scoring.

The validation sample may deliberately oversample nuclei near the decision threshold, but every
record must retain its sampling stratum and known inclusion probability. Population sensitivity,
specificity, predictive values, conversion prevalence, and projection inflation must be estimated
with inverse-probability or equivalent post-stratification weights. Unweighted boundary-enriched
metrics are stratum diagnostics only. The target population and condition mixture for every
weighted estimate must be declared before reference labels are revealed; predictive value cannot
be transported to a population with a different prevalence without recalculation.

Nuclei from the same image are clustered technical observations. Confidence intervals and
sample-size calculations must preserve the field and biological-unit hierarchy. A field-cluster
bootstrap or a prespecified hierarchical binary model is acceptable; an object-level binomial
interval is not. The initial control/high-converter variance pilot may estimate nuisance quantities
but cannot establish a treatment effect or a universal fields-per-well requirement.

The pilot begins with at least eight prespecified, spatially distributed fields in each of a control
and a high-converter well. Final fields per well are chosen from the observed between-field
variance for a 95% well-level conversion interval with no more than 5 percentage points absolute
half-width (3 points may be reported as a higher-precision design). The calculation must use
cluster-aware simulation or a checked beta-binomial/hierarchical model, preserve unequal field
nucleus counts, and be rounded up. The field pilot determines technical sampling only. Treatment
power must instead use between-independent-batch variance, a prespecified meaningful effect, and
at least three independent biological units; three remains a fail-closed minimum, not an adequate
sample-size justification.

Threshold calibration and threshold validation must use disjoint biological material or a
prospectively locked nested split. A z-stack ROC or negative-control distribution used to choose a
threshold cannot also provide its confirmatory performance estimate. The threshold-selection loss
or constraint must be declared in advance; an undefined post-hoc "optimal ROC band" is not a
release rule.

For descriptive absolute conversion percentage, the three co-primary method gates are conjunctive:

- the lower bound of a two-sided 95% interval for specificity is at least 0.95;
- the lower bound of a two-sided 95% interval for sensitivity is at least 0.90; and
- the upper bound of a two-sided 95% interval for projection false-positive inflation
  (`1 - positive predictive value`) is at most 0.10.

The negative-control calibration gate is an upper two-sided 95% confidence bound on false-positive
rate no greater than 0.05. All primary estimates use the target-population weights and
field/biological-unit clustering. Because all co-primary gates must pass, this is a conjunctive
intersection-union release rule; each adverse 95% bound must pass. Metrics are also reported by
sampling stratum and condition so pooled performance cannot conceal a boundary or condition
failure.

After the pilot estimates prevalence, intrafield correlation, and stratum yields, final validation
sample size is selected by reproducible Monte Carlo simulation of the complete clustered,
stratified analysis. It must provide at least 80% probability that all adverse-bound gates pass at
the prespecified anticipated performance and meet at most 10 percentage points interval half-width
within each boundary stratum. A flat rule such as 100 nuclei per stratum is only an initial
independent-binomial approximation and is not the final sample-size justification.

The existing raw-intensity threshold is valid only under its frozen acquisition and preprocessing
scale. Validation against newly acquired z-stacks compares the original 2-D production score, or a
prospectively matched projection processed by the frozen method, with the 3-D reference. Transfer
to a different instrument or acquisition batch requires a locked acquisition SOP or an explicit
intensity-calibration rule evaluated on independent validation data.

If the absolute-percentage gates fail, fold change is not automatically released. Relative results
remain exploratory unless a prespecified threshold/misclassification sensitivity analysis shows
that differential error across conditions cannot reverse or materially distort the relative
effect. Failure narrows the claim and never triggers threshold relaxation on the validation set.

### T03 model performance

For every candidate report both:

- micro estimates based on summed TP/GT/prediction counts;
- macro mean/median across held-out wells;
- 95% intervals obtained by resampling whole held-out wells;
- all six fold values and all failed folds;
- precision, recall, F1, matched IoU, split/merge rates, length/width MdAPE, and automatic coverage.

Candidate selection uses the adverse confidence bound when available: the lower bound for minimum
requirements such as precision, and the upper bound for maximum-error requirements. Current T03
results remain internal model-evaluation evidence because all six wells are from one plate.

## Missingness and exclusions

- Predefine image, object, and integrity exclusions.
- Preserve failed fields in the manifest with their failure reason.
- Report counts before and after exclusions by condition and biological unit.
- Never exclude an outlier solely because it weakens a result.
- Any post-hoc sensitivity exclusion must be reported alongside the primary analysis.
- Binding annotation exclusions apply identically to training and every derived synthetic record.

## Prospective release gate

Authoritative instance-metric release requires all of the following:

- experimental unit declared;
- at least three independent biological replicates, with the final number justified prospectively;
- technical/object measurements not counted as independent `n`;
- analysis and exclusions locked before prospective labels are revealed;
- every release metric accompanied by a 95% confidence interval;
- the adverse confidence bound, not only the point estimate, passes its predefined threshold;
- complete reporting of failed fields and missing data.

Failure retains field metrics and manual-QC-only instance reporting. It does not relax thresholds.

## Canonical implementation

`precision_myotube.statistics` provides:

- hierarchical technical-to-biological-unit aggregation;
- deterministic biological-unit bootstrap intervals;
- paired or independent unstandardized mean differences;
- descriptive Wilson intervals for binary counts;
- Benjamini-Hochberg adjustment for a pre-specified test family;
- explicit descriptive-only output when independent replication is insufficient.

Run a manifest with:

```powershell
python -m precision_myotube statistics-summary `
  --manifest statistics_manifest.json --out statistics_result.json
```

The example `statistics_manifest.example.json` documents the required design metadata.

## Reporting standards used

- ARRIVE 2.0 principles for declaring the experimental unit, exact `n`, exclusions, design, and
  analysis: https://arriveguidelines.org/arrive-guidelines
- SAMPL guidance for effect estimates, confidence intervals, exact statistical methods, and avoiding
  reliance on p-values alone: https://www.equator-network.org/reporting-guidelines/sampl/
- Nature guidance distinguishing biological and technical replicates:
  https://www.nature.com/documents/Biological_and_technical_replicates_guidelines.pdf
- Recommended fluorescence-microscopy metadata and validation reporting:
  https://www.nature.com/articles/s41592-021-01156-w
