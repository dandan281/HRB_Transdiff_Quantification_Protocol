# Codex ruling — G-SO2 labelling shift, T03 sensitivity, and Tier-A relocalization

Date: 2026-08-12 America/Los_Angeles  
Owner: Codex integrator/statistical lane  
Request: `coordination/requests/codex/2026-08-11-labelling-shift-and-tier-a-relocalization.md`

## 1. G-SO2 disclosure is binding

The single-operator corpus shows a material between-well certification shift. Review order is
inferred from `decisions.json` filesystem mtimes, not logged review order, and must be called a
proxy.

| proxy order | well | triaged candidates | complete before exclusions | fraction |
|---:|---|---:|---:|---:|
| 1 | 19_B06_act104_trka | 240 | 120 | 0.500 |
| 2 | 22_B03_act104_egfrc | 237 | 61 | 0.257 |
| 3 | 29_C05_br223_egfrc | 241 | 59 | 0.245 |
| 4 | 32_C08_br223_igf1r | 225 | 54 | 0.240 |
| 5 | 33_C09_br223_trka | 235 | 48 | 0.204 |
| 6 | 23_B02_ctrl | 69 | 35 | 0.507 |

Well 1's ordinary length/intensity distribution and short ambiguous remainder are consistent with
the operator settling on a stricter standard after the first well. That is an inference, not a
proven cause. Well 1 has a unique treatment combination, so biology cannot be excluded. The control
was last and is confounded with order; it is not evidence against a settling effect.

One numeric correction to the request is important: B06 had 120 objects certified complete, but
one binding G-SO1 exclusion leaves **119 of the 375 authoritative masks (31.7%)**, not 120/375.
This does not remove the imbalance.

G-SO2 must disclose the table, the mtime proxy, the treatment/order confounding, and the fact that
the reference set is single-operator and may pool non-uniform certification standards. This is a
claim limitation; no relabelling is authorized.

## 2. T03 statistical ruling

The predeclared all-six-well pooled statistic remains the primary. Removing or reweighting B06
after discovering the certification shift would change the estimand post hoc. The existing
statistical plan already requires every post-hoc exclusion sensitivity to appear beside, never
instead of, the primary.

A **complete drop-one-whole-well sensitivity is now mandatory** in every T03 candidate assessment.
It reports remaining false-split count and rate after each omission. Once Omnipose exists, the
candidate comparison must additionally report paired candidate deltas under every same-well
omission. No candidate may choose which omission to feature based on which one improves its result.

For the sealed classical floor:

- primary: 52/375 = 0.1387;
- omit suspected B06: 34/256 = 0.1328;
- all six omission rates range from 0.1206 to 0.1519.

The B06 omission changes the classical floor rate by about -0.0059 and does not reverse its
interpretation. This does not prove the standards were uniform; it shows the current floor's pooled
false-split result is not numerically driven by that single omission. The sensitivity is post-hoc
and descriptive, never a replacement primary.

## 3. Stage metadata finding

The statement “stage metadata is unavailable” is too broad. It is absent from derived caches and
the current selector, but present in every original Plate-23 ND2. Read-only inspection with
`nd2 0.11.3` found per-file event coordinates:

| well | event X µm | event Y µm | event Z µm |
|---|---:|---:|---:|
| 19_B06_act104_trka | 4109.7 | -23423.1 | 7450.28 |
| 22_B03_act104_egfrc | 31109.7 | -23423.1 | 6986.84 |
| 23_B02_ctrl | 40109.7 | -23423.4 | 6831.54 |
| 29_C05_br223_egfrc | 13109.5 | -14423.4 | 7077.18 |
| 32_C08_br223_igf1r | -13890.1 | -14423.3 | 7472.76 |
| 33_C09_br223_trka | -22890.3 | -14423.4 | 7587.22 |

All six also contain calibrated XY pixel size 0.649268834 µm and the same camera rotation matrix.
However, `pixelToStageTransformationMatrix` is absent. Worse, the ND2 frame-metadata accessor
returns the same XYZ triplet for all six files while the event table returns distinct positions
with the expected 9-mm well spacing. Therefore the event coordinates are credible field-centre
metadata, but a pixel-centroid-to-stage affine is **not yet certified**. Removal/reseating of the
sample would add an unknown transform even if the original affine were reconstructed.

## 4. TA03b/TA03c amendment

Direct targeting of individual selected nuclei is not presently authorized. The existing selector
correctly remains `relocalization_feasible=false`; raw stage metadata alone must not flip it true.

The targeted biological sampling design is retained, but its acquisition/matching implementation
is superseded as follows:

1. Reacquire the original **field or a registered field mosaic** as a z-stack using the ND2 event
   field-centre coordinates only as a navigation aid.
2. Include a DAPI/nucleus channel in both images and acquire an overview suitable for registration.
3. Estimate and lock the original-2D-to-new-3D-projection transform from the nucleus channel,
   independently of the Desmin truth call. Record transform model, residuals, source hashes, and
   every failed field.
4. Match nuclei one-to-one after registration using a prespecified centroid-distance plus mask-
   overlap rule. Preserve unmatched, duplicate, split, and ungradable selected nuclei as attrition;
   do not silently replace them.
5. Apply the selector's existing inclusion probabilities to successfully matched selected nuclei.
   If only part of a field is reacquired, field/area inclusion probabilities must also be declared;
   otherwise primary population weighting is invalid.
6. Score the frozen original 2-D call against z-resolved truth. Registration uses DAPI geometry;
   Desmin may not be used to choose the transform or resolve match ambiguity.

Claude may build the scorer against this field-registration/post-hoc-matching contract. A direct
per-nucleus stage-target mode may be added later only after a calibration target or retained-sample
fiducial test validates the full pixel-to-stage transform and a maximum reacquisition error is
predeclared. Acquisition itself remains unauthorized until the operator confirms that the physical
sample exists and can be returned to the microscope.

## 5. Carried-over items

- The classical T03 artifact was already regenerated as assessment v1.1 on 2026-08-04; this
  session regenerates it as v1.2 to add the mandatory labelling-standard sensitivity. Metrics
  remain unchanged. Current SHA-256 is
  `cc42b25bd0266119cc26a5780b8384d59b172fdfff8ef181dd57d2aaf8636bcf`.
- `linked_candidate.py` does not rely on the changed finder default. It explicitly passes a
  `require_axis_agreement` value at prediction and training-pair reconstruction call sites:
  sealed v1 resolves to `False`; the new constrained-v2 identity resolves to `True`.

No GPU job, relabelling, acquisition, threshold tuning, or `Conversion_Efficiency/**` write was
performed by Codex.

Verification: full CPU suite **429 passed**, with 16 existing Pydantic deprecation warnings. Both
assessment-v1.2 artifacts pass integrity; no metric changed.
