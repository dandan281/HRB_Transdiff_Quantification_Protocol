# Fragment linker: instance-level integration measurement — POSITIVE at the confident threshold

2026-07-28. Companion to `claude_junction_integration_measurement_2026-07-28.md`,
same harness and protocol. The linker reports LOWO AUC 0.902 on its own pair
task but had never been scored on the instance-level readout. Fragmentation is
the error the project has repeatedly called dominant, so this is the
measurement that says whether fixing it moves the science.

**Result: yes, at the confident operating point.** At threshold 0.9 —
the "auto-merge the confident minority" point the linker was always
characterised as — recall **+0.015**, F1 **+0.050**, false-split rate
**−0.018 (−12% relative)**, instance count −28%, at the cost of a newly
non-zero but tiny over-merge rate (0.0009).

## Correction to an earlier read in this session

A single-well smoke test (33_C09) showed recall 0.917 → 0.979 at threshold 0.7
and I reported that as the headline. **It did not generalise.** Across all six
wells, threshold 0.7 gives recall **0.900 vs the floor's 0.915 — a loss**. The
one-well number was not representative and should be disregarded; the six-well
means below are the result.

## Instance-level result (means over 6 LOWO folds)

| metric | classical | linked@0.5 | linked@0.7 | **linked@0.9** |
|---|---:|---:|---:|---:|
| n_pred | 879.8 | 479.8 | 566.3 | **634.5** |
| recall | 0.9149 | 0.8437 | 0.8998 | **0.9298** |
| f1 | 0.1244 | 0.1973 | 0.1835 | **0.1739** |
| precision | 0.0677 | 0.1149 | 0.1053 | 0.0984 |
| mean_matched_iou | 0.9174 | 0.9108 | 0.9144 | 0.9118 |
| false_split_rate | 0.1476 | 0.1251 | 0.1232 | **0.1294** |
| over_merge_rate | 0.0000 | 0.0020 | 0.0014 | **0.0009** |
| n_merges | – | 984.0 | 735.3 | 548.8 |

**Threshold 0.9 is the only operating point that improves recall.** Lower
thresholds buy a better F1 by merging aggressively — F1 peaks at 0.5 — but
that F1 gain is precision-driven, and precision here is dominated by the
sparse-GT effect the sealed run already documents (62.5 GT masks against ~880
predictions; the reviewed `complete` set is a small subset of each field's
fibre-like structure). **Precision and F1 are therefore not trustworthy in
either arm.** The columns that mean something are recall, false_split_rate,
over_merge_rate and matched IoU — and on those, 0.9 is the only setting that
is better than the floor on the first and third while still improving the
second.

Per well at threshold 0.5 (the aggressive end), showing why the mean recall
falls:

| well | n_pred | recall | false-split rate |
|---|---|---|---|
| 19_B06_act104_trka | 842 → 430 | 0.941 → 0.790 | 0.151 → 0.076 |
| 22_B03_act104_egfrc | 853 → 399 | 0.967 → 0.800 | 0.100 → 0.050 |
| 23_B02_ctrl | 1030 → 636 | 0.771 → **0.829** | 0.314 → 0.343 |
| 29_C05_br223_egfrc | 789 → 423 | 0.949 → 0.898 | 0.068 → 0.051 |
| 32_C08_br223_igf1r | 873 → 519 | 0.944 → 0.870 | 0.148 → 0.148 |
| 33_C09_br223_trka | 892 → 472 | 0.917 → 0.875 | 0.104 → 0.083 |

`23_B02_ctrl` is the informative outlier: the most fragmented well (false-split
0.314, recall only 0.771) and the **only** one where aggressive merging *helps*
recall. Where fragmentation genuinely dominates, the linker earns its keep even
at a loose threshold; elsewhere loose merging destroys real objects.

## Comparison with the junction classifier

Same harness, same protocol, same day:

| | reaches | recall | f1 | false_split_rate |
|---|---:|---:|---:|---:|
| junction classifier | 1.8% of pairing decisions | −0.0118 | −0.0026 | −0.0003 |
| **fragment linker @0.9** | 4,058–11,982 candidate pairs/well | **+0.0149** | **+0.0495** | **−0.0182** |

The junction classifier is *more accurate at its own task* (2.7x the classical
rule) and moves the science by zero. The linker is the one that reaches enough
of the pipeline to matter. This is a **scope** difference, not a quality one —
recorded as a durable lesson.

## Caveats, stated plainly

- **Domain shift.** The linker was trained on pairs drawn from the annotation
  packages' `starting_labels.tif` proposals and is applied here to
  classical-floor instances. Both are ridge-style segmentations of the same
  fields and the features are generic, but they are not the same masks. The
  bias direction is unknown; a *positive* result under a domain shift is the
  more trustworthy direction for it to point, but this should be re-measured
  if the linker is ever retrained on classical-floor instances directly.
- **34 banked pairs did not re-match the candidate finder** at the wide window
  and were excluded from training (217 of 251 usable pairs used). Pre-existing
  behaviour of `recompute_training_pairs`, reported not silenced.
- **A new error class appears.** Over-merge was exactly 0.0000 in the floor and
  is 0.0009 after linking (~0.5 objects per well at threshold 0.9). Small, but
  over-merge is one of the operator's three named error classes and is arguably
  worse than under-merge for counting. It is a real cost, not a rounding error.
- Instance count falling 880 → 635 cannot be validated against GT count: the
  eval GT is a sparse reviewed subset (62.5 masks/well), not a complete census.
  `false_split_rate` is the metric that actually measures fragmentation against
  GT, and it improved.

## Recommendation

**Wire the linker into the pipeline at threshold ~0.9**, and treat that as the
default operating point rather than 0.5. It is the first change measured this
session that improves the instance-level readout on more than one axis. The
integration harness (`model_labs/classical/run_linker_folds.py`) is reusable
for re-measuring after any linker change.

## Learning curve — measured, and it is flat too

Before recommending more fragment labels, the curve was run (same method as the
junction classifier's; train on N pairs from the other wells, LOWO by well,
20 seeds):

| n_pairs | LOWO AUC | sd |
|---:|---:|---:|
| 40 | 0.8810 | 0.0142 |
| 60 | 0.8877 | 0.0074 |
| 80 | 0.8920 | 0.0046 |
| 100 | 0.8937 | 0.0039 |
| 120 | 0.8952 | 0.0026 |
| 140 | 0.8954 | 0.0027 |
| 159 (max) | 0.8949 | 0.0010 |

**Saturated at ~100-120 pairs; slope −0.0008 AUC per 100 additional pairs.**
So more labels *of the same kind* will not improve the scorer. Third saturated
learning curve measured this session.

### But the curve is on the wrong distribution — checked, and the shift is moderate

The banked pairs come from 65 operator-confirmed fragments in a narrow window.
At deployment the model scores **4,130 candidate pairs in one well alone** —
everything the classical floor emits. A flat curve on the banked set therefore
does not automatically settle what *deployment-sampled* labels would buy, so
the shift was measured rather than assumed:

| feature | train mean | deploy mean | train p5–p95 | % of deploy outside |
|---|---:|---:|---|---:|
| bridge_over_bg | 1.141 | 1.466 | [0.85, 1.80] | 18.2% |
| axis_cos | 0.920 | 0.855 | [0.54, 1.00] | 12.5% |
| displacement_along_axis | 0.861 | 0.775 | [0.28, 1.00] | 10.5% |

Train-vs-deploy discriminability: **AUC 0.639** (0.5 = identical
distributions). That is a real but **moderate** shift — the model is being
extrapolated somewhat, not wildly. 12.3% of deployment candidates score ≥0.9,
which is exactly the ~549 merges/well the fold run produced.

**Reading:** more labeling is probably low-value for the linker too, though
less definitively than for the junction classifier. If a labeling round is run
anyway, it should **sample candidates from the classical-floor deployment pool
rather than the banked fragment set** — that is where the 10–18% out-of-range
mass lives, and it is the only version of "more labels" this evidence leaves
open.

Code: `model_labs/classical/run_linker_folds.py`. Output:
`model_labs/classical/_runs/linker_instance_v1.json`.

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence.
