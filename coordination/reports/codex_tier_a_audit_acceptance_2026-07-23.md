# Integrator acceptance — Tier-A conversion audit

Integrator: Codex  
Date: 2026-07-23  
Source audit: `model_labs/tier_a_audit/`  
Claude report: `coordination/reports/claude_tier_a_audit_results_2026-07-23.md`

## Ruling

The read-only Tier-A audit passes independent integration review. The declared conversion method is
reproducible, internally consistent, plate-wide, and provenance-bound. It is adopted as the
project-canonical internal conversion estimator.

This acceptance is not a scientific release and does not establish biological correctness.
Conversion efficiency and Desmin territory remain held pending orthogonal validation, a declared
field-sampling design, and prospective biological replication.

## Independent verification

- A fresh audit run reproduced the declared pooled-Otsu threshold exactly:
  `440.76596787901417`.
- All six well counts reproduced exactly:

| Well | Positive | Valid nuclei | Percent |
|---|---:|---:|---:|
| B02 control | 1,166 | 7,635 | 15.27% |
| C09 | 1,616 | 8,947 | 18.06% |
| C05 | 1,750 | 7,210 | 24.27% |
| B06 | 2,748 | 8,440 | 32.56% |
| C08 | 3,341 | 10,114 | 33.03% |
| B03 | 3,749 | 9,524 | 39.36% |

- The operating point is one pooled log-uniform Otsu threshold over all six wells. There is no
  per-well threshold or tuning.
- The two C08 mask files are byte-identical and share SHA-256
  `ca15dffda03f52ff310381212d45e6067066dc5f1a8666a6fac45f062a0d47c9`.
- The C08 count lineage is 10,588 labels, 10,562 at the MyoFuse `>=30 px` floor, and 10,114 under
  the canonical `[50,500] µm²` physical-area filter.
- An independent rerun of the MyoFuse premise script reproduced its 10,560 measured nuclei,
  confirming that its downstream ring-validity logic removes two more objects.
- A fresh output matched Claude's reproduced JSON byte for byte by SHA-256.
- A full before/after metadata snapshot found no file change anywhere under
  `Conversion_Efficiency/`.

## Evidence hashes

| Artifact | SHA-256 |
|---|---|
| `model_labs/tier_a_audit/audit.py` | `ebb4b7d4b3a761e3c1afb09c442ae0974230b4085f1817ff7e2f5aed971f7bc2` |
| `model_labs/tier_a_audit/_audit/audit_manifest.json` | `5794b9705d1d1ad5fc0a44e392b2878b230d754056b00e81000eca37f39b9321` |
| `model_labs/tier_a_audit/_audit/reproduced_visualize_final.json` | `2134a38a6fc142328f43ab332115a97c6fd616c972bd202d73ef71335b1d65be` |
| `coordination/reports/claude_tier_a_audit_results_2026-07-23.md` | `62c740b39933f28078930a604b65d97993a6626a4a819b1463d044db99b0e25f` |

## Canonical method and superseded result

The internal canonical estimator is:

- 10 µm perinuclear ring;
- canonical nucleus-area filter `[50,500] µm²`;
- one plate-wide pooled log-uniform Otsu threshold, `440.76596787901417`;
- no per-well threshold;
- same-plate results interpreted descriptively, not as treatment effects.

The traced-fiber result `670 / 10,114 = 6.6245%` is retained as superseded provenance and must not
be cited as the current conversion estimate. The absolute-threshold `k` sweeps are diagnostic
sensitivity analyses, not operating points. This ruling does not canonize Desmin territory.

## Remaining release gates

Reproducibility answers whether the declared computation can be repeated. It does not resolve
whether a 2-D Desmin signal is inside the nucleus plane or above/below it. Scientific release
therefore still requires orthogonal evidence such as confocal z-stacks, a validated additional
marker, and/or a Desmin-negative control well. Field sampling, independent biological replication,
prospective declarations, and the binding statistical plan also remain required.

Total and valid nuclei may be reported as descriptive single-plate measurements after freezing the
mask source and hashes. No treatment claim is authorized from these six same-plate wells.

## Verification totals

- Tier-A focused audit suite: 13 passed.
- Canonical PrecisionMyotube suite: 44 passed.
- Annotation/model-lab suite: 160 passed.
- Combined project checkpoint: 204 passed.

No production method was changed, no upstream Cellpose run was repeated, and nothing was committed.
