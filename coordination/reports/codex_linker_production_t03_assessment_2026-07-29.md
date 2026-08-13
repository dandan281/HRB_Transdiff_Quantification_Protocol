# Codex assessment — fragment-linker production wiring and official T03

> **Posthoc safety correction, 2026-07-30 PT / 2026-07-31 UTC:** the run and pooled
> summary below remain reproducible, but `over_merge_count=3` is only the number of
> sparse-reference-examinable flags. It is not the total over-merge count and
> `3/3807` is not a linker safety rate. A blinded two-pass raw-image review subsequently
> called 6/12 unflagged control merges `different_myotubes`, and link confidence was
> anti-correlated with the human verdict (`AUC=0.107`). The automatic linker is not
> released; retain the locked `P>=0.90` output for reproducibility and manual QC only.
> The authoritative current ruling is
> `coordination/reports/codex_linker_release_ruling_2026-07-31.md`.

Date: 2026-07-29  
Integrator: Codex  
Execution: CPU only  
Disposition: **wired as a separate reproducible candidate; not promoted to automatic default**

## Executive ruling

The fragment linker is now available through the canonical CLI at the predeclared
`P >= 0.90` operating point and has a complete six-fold, hash-bound candidate run. The sealed
`classical_ridge_graph/v1` floor was not modified.

The official T03 assessment passes artifact integrity but does **not** support promoting the linker
to the automatic production default. It improves false splitting, but the primary non-circular
25-mask corrected subset loses two true positives and has worse matched IoU:

| Primary corrected subset | Sealed floor | Floor + linker 0.9 | Change |
|---|---:|---:|---:|
| GT masks / evaluable wells | 25 / 2 | 25 / 2 | unchanged |
| True positives | 20 | 18 | **-2** |
| Recall | 0.8000 | 0.7200 | **-0.0800** |
| Mean matched IoU | 0.6667 | 0.6509 | **-0.0158** |
| False splits | 3 | 1 | **-2** |
| False-split rate | 0.1200 | 0.0400 | improved |
| Over-merges | 0 | 0 | unchanged |
| Length MdAPE | 0.3169 | 0.2984 | modest improvement |
| Width MdAPE | 0.0779 | 0.0779 | unchanged |

False-split improvement does not justify losing 10% of the previously matched corrected objects,
especially when over-merge is an operator-named high-cost error and the evidence spans only two
wells. The linked candidate remains useful as a manual-QC proposal generator or experimental
postprocessor, not an unattended default.

## Check a — sealed-floor invariance

The existing `test_default_decider_is_the_classical_rule` is a useful unit guard, but by itself it
tests only one synthetic crossing. Codex therefore performed the stronger repository-level check:

1. load each of the six sealed fold parameter sets;
2. rerun `trace_fibers_parameterised(..., junction_decider=None)` on the sealed territory cache;
3. rerun territory assignment and filtering; and
4. compare every regenerated foreground-position set with its sealed prediction mask.

All **5,279 of 5,279 masks across six wells were exactly identical**. Each sealed prediction file
also still matches its run-manifest SHA-256. The hook has not changed the default floor.

## Check b — LOWO honesty

### Junction integration

The pair model, branch-point model, and gate threshold are fit from the other five wells inside
each fold. `decision_accuracy` intersects predictions with only the supplied training-junction
keys, so passing the full truth dictionary does not expose held-out truth during threshold choice.

The junction feature family, including the 15-pixel direction window, was developed on all-well
junction labels before the instance-level run. Therefore the negative instance result is not a
fully nested estimate of architecture selection. This does not weaken the shelving decision: any
such development optimism would favor the junction model, yet its measured instance effect was
still negative.

### Linker integration

For every fold, Codex independently checked that:

- the held-out well is absent from `train_wells`;
- the held-out well is absent from the serialized model's `training_wells`; and
- every serialized training row comes from another well.

Fold training-row counts are 198, 159, 217, 160, 159, and 192. The scaler and logistic
coefficients are genuinely LOWO-refit.

Threshold 0.9 is **not selected per fold**. It is the predeclared high-confidence `P >= 0.90`
policy documented in the 2026-07-23 round-2 and round-3 reports before instance-level integration.
That is preferable to selecting a threshold on held-out T03 results.

The feature family (`bridge_axis`) and candidate window were nevertheless developed earlier from
labels across all six wells. The official run and T03 assessment now explicitly report that only
the scaler/coefficients are nested LOWO; architecture/window/operating-point development is not.

## Check c — the over-merge trade

On all 375 proposal-conditioned masks, the linked result exactly reproduces Claude's A/B harness:

| All proposal-conditioned GT | Sealed floor | Floor + linker 0.9 |
|---|---:|---:|
| Predictions | 5,279 | 3,807 |
| True positives | 348 | 349 |
| Micro recall | 0.9280 | 0.9307 |
| False splits | 52 | 41 |
| False-split rate | 0.1387 | 0.1093 |
| Over-merges | 0 | 3 |
| Over-merge / predictions | 0 | 0.00079 |
| Over-merge / sparse GT | 0 | 0.0080 |
| Mean matched IoU | 0.9146 | 0.9120 |

The macro recall gain in the handoff (`+0.0149`) is driven mainly by the small, difficult control
well. The micro gain is only one additional matched object (`+0.0027` recall). In two wells, B06
and B03, recall decreases and all three newly detected over-merges occur. More importantly, those
are the only two wells with accepted corrected masks, and their combined corrected recall falls
from 0.80 to 0.72.

Ruling: the `0 -> 3` over-merge change is not accepted as an unattended-production trade when the
primary corrected subset simultaneously loses recall and IoU. No post-hoc threshold search on
these 25 masks is authorized.

## Check d — sparse-GT precision and F1

Confirmed. The all-GT comparison has only 375 reviewed-complete masks against thousands of field
predictions, and the corrected comparison has 25 masks against 1,047 linked predictions. Unmatched
predictions are not ordinary false detections because the GT is not a complete field census.

The canonical T03 output now explicitly writes:

- `precision_interpretable: false`; and
- `f1_interpretable: false`.

Recall, matched IoU, false-split/over-merge events touching the sparse GT, and measurement error
remain descriptive diagnostics. Precision and F1 must not be quoted as detector performance.

## Check e — junction classifier status

The junction classifier should be recorded as **built, measured, and shelved** at the next
authorized plan reconciliation:

- junction-decision accuracy: 64.5% versus the classical rule's 23.8%;
- instance-level recall delta: -0.0118;
- reached only 893 of 49,594 pairing decisions (1.8%);
- labeling and tested feature/model alternatives are saturated.

It should not be deleted and should not be resumed unless upstream fragmentation is materially
reduced and junction decisions become consequential. Per the session constraint, Codex did not edit
`DEVELOPMENT_PLAN.md` or `coordination/WORKBOARD.md`.

## Implementation

New canonical command:

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m precision_myotube `
  linked-candidate-run `
  --base-run model_labs/classical/_runs/v1 `
  --pairs PrecisionMyotube/annotation_work/links_active_r3/banked/combined_pairs_r123.jsonl `
  --out PrecisionMyotube/runs/t02/classical_linker_v1 `
  --threshold 0.9 --gap-um 80 --cos-min 0.70
```

The run contains six serialized fold models, six full decision ledgers, six unreviewed canonical
prediction sets and sidecars, copied sealed grid evidence, source/input/environment hashes, exact
LOWO training rows, and a run manifest. It refuses to overwrite a non-empty run directory.

Official assessment:

```powershell
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m precision_myotube `
  t03-assess `
  --run PrecisionMyotube/runs/t02/classical_linker_v1 `
  --out PrecisionMyotube/runs/t03/classical_linker_v1/assessment.json `
  --bootstrap-resamples 10000 --seed 20260723
```

T03 integrity passes. T03 remains incomplete; G-SO2 does not pass; no candidate is selected; the
disposition is `retain_as_reproducible_floor; manual_QC_only`.

## Verification and hashes

- Resume baseline before implementation: 276 passed.
- Focused integration checks: 28 passed, then 9 passed after reporting amendments.
- Final combined CPU suite: **284 passed**, 16 Pydantic deprecation warnings.
- Production output reproduces every per-well linker@0.9 harness metric to `1e-12`.
- All base prediction hashes remain unchanged.

| Artifact | SHA-256 |
|---|---|
| `PrecisionMyotube/precision_myotube/linked_candidate.py` | `4de4f383ddf79a91aca70fdeb70121bb01a67ad19d2bd7aff5850ce3393a351f` |
| `PrecisionMyotube/precision_myotube/t03.py` | `f934e995c42e7d845510ec738a1bef51c868bbafb77d1a41a40a192c6ad8e974` |
| `PrecisionMyotube/runs/t02/classical_linker_v1/run_manifest.json` | `c1cfc824ff5da2a8bd1e78f8d18373a53b61c97895d9f0a79e06eefe934e6a29` |
| `PrecisionMyotube/runs/t03/classical_linker_v1/assessment.json` | `ebab62bbdd647962a8010e394c8b8d90e0e40a1575fdce009191a3901b2b0e9d` |
| Sealed base run manifest | `420dc2996f93621641ca0e09e6a763f8cc1300110b098b0be6962f2db0ab0e07` |
| Claude A/B linker artifact | `d03d4b24c9d2ac680f5f49942728adf8c1e909f6b7f9b7ca0afcddab0d64943d` |

No GPU work, Omnipose work, Tier-A work, or `Conversion_Efficiency/**` write occurred. Nothing was
committed.
