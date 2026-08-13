# Acquisition brief for the 2026-08-19 z-stack session — how many fields, and why

Date: 2026-08-12
Lane: Claude (model laboratories)
For: the operator, before acquisition. **This is planning, not authorization** — the
ruling holds that acquisition remains unauthorized until the physical sample is confirmed
to exist and be returnable to the microscope.

**The one number: acquire at least 2 fields per well (12 total), not 1. Three per well
(18) if the session allows.**

---

## 1. Why one field per well is not enough

The ratified contract requires each boundary stratum's two-sided 95% interval to have a
half-width of no more than 10 percentage points, and requires the interval to be clustered
by field rather than computed over nuclei. Current data has **one field per well**, so a
straight reacquisition of what exists gives six clusters.

Simulating the exact bootstrap the analysis will use, at an assumed sensitivity of 0.90:

| fields | nuclei/field/stratum | ICC | median half-width | p90 half-width | share meeting target |
|--:|--:|--:|--:|--:|--:|
| 6 | 25 | 0.05 | 0.057 | 0.090 | 97% |
| **6** | **25** | **0.10** | **0.070** | **0.117** | **86%** |
| 12 | 25 | 0.05 | 0.042 | 0.060 | 100% |
| **12** | **25** | **0.10** | **0.053** | **0.077** | **100%** |
| 20 | 25 | 0.05 | 0.034 | 0.047 | 100% |
| 20 | 25 | 0.10 | 0.045 | 0.059 | 100% |

At six fields and a moderate intracluster correlation of 0.10, roughly **one run in seven
misses the required half-width** — and the p90 half-width, 0.117, is outside the target
outright. Twelve fields clears it in every replicate at both ICC values tried, with real
margin.

Note the analytic design-effect calculation is more optimistic: it says five fields suffice
at ICC 0.10. It is wrong in the direction that matters. A percentile bootstrap over six
clusters is coarse and unstable, and the analytic Wald approximation does not know that.
Where the two disagree, the simulation is the one that describes what the analysis will
actually do, and it is the basis for the recommendation.

## 2. Nuclei per field is not the binding constraint — fields are

This is the part that is easy to get backwards. With clustered data the design effect is
`1 + (m - 1) x ICC`, so as you grade more nuclei in a single field the effective sample
size per field approaches a ceiling of `1 / ICC`. At ICC 0.10 that ceiling is **ten
effective observations per field no matter how many nuclei you grade in it.**

The practical consequence: grading 100 nuclei per stratum in one field is worth barely more
than grading 25, and cannot substitute for a second field. Budget the session around
**more fields, fewer nuclei each**.

**25–50 nuclei per boundary stratum per field is sufficient.** Beyond that the return is
close to nil.

## 3. Supply is not a problem — the strata are well populated

The selector was run against the real plate for the first time. Its sampling frame totals
**51,869 nuclei, exactly matching the accepted audit's `pooled_cells` of 51,869** — the
selector reproduces the audit's population rather than approximating it.

Per-well counts in the two boundary strata, which are the ones that must clear the gate:

| well | frame | `0.8_1.0` | `1.0_1.25` |
|---|--:|--:|--:|
| 23_B02_ctrl | 7,635 | 214 | 150 |
| 33_C09_br223_trka | 8,947 | 428 | 334 |
| 29_C05_br223_egfrc | 7,210 | 399 | 339 |
| 19_B06_act104_trka | 8,440 | 446 | 470 |
| 32_C08_br223_igf1r | 10,113 | 660 | 586 |
| 22_B03_act104_egfrc | 9,524 | 594 | 601 |
| **pooled** | **51,869** | **2,741** | **2,480** |

Even the thinnest well — the control, at 214 and 150 — holds far more boundary-stratum
nuclei than the 25–50 per field the plan calls for. Nothing needs to be acquired to make
the sample size reachable; the constraint is purely how many distinct fields get imaged.

At 3,636 px and 0.6493 µm/px a field is about 2.36 mm across, so two or three
non-overlapping fields per well is geometrically straightforward.

## 4. What each acquired field must carry

From §4 of the 2026-08-12 ruling, and all of it is needed for the scorer to run at all:

- a **DAPI/nucleus channel in both the original and the new image**, plus an overview
  suitable for registration — registration is estimated from nucleus geometry, and Desmin
  may not be used to choose the transform or break a matching tie;
- enough overlap/context to estimate and lock the 2-D-to-3-D-projection transform, with
  **transform model, residuals and source hashes recorded**, and every failed field recorded
  rather than dropped;
- if only part of a field is reacquired, a **declared field/area inclusion probability** —
  without it the population weighting is invalid and the primary result cannot be computed.

## 5. Two things that will invalidate the run regardless of field count

**A negative control.** The fourth adverse-bound gate — negative-control false-positive rate,
upper 95% bound ≤ 0.05 — cannot be evaluated without Desmin-negative material. The harness
reports it as `not_evaluable` and fails it rather than passing it by default. All three
co-primary gates can pass and the absolute-percentage claim still fails without this. If
negative-control material is not in the 8/19 session, it needs its own session.

**Attrition.** Unmatched, duplicate, split and ungradable nuclei are retained and counted,
never replaced. A low match rate invalidates the population weighting whatever the metrics
say, because the nuclei that fail to match are exactly the ones that were hard. Budget for
the possibility that registration fails on some fields and acquire margin accordingly.

## 6. Caveats on these numbers

- **ICC is assumed, not measured.** No pilot exists, so the table spans 0.05 and 0.10 rather
  than asserting one value. If the true ICC is above 0.20, twelve fields is not enough
  either. The first fields acquired should be used to estimate it before committing the
  rest of the session.
- **Planning at a sensitivity of 0.90 is optimistic.** The half-width is widest at 0.5, so a
  method performing worse than assumed needs more fields, not fewer.
- **This sizes one stratum.** Both boundary strata must clear the target independently; a
  pooled result may not conceal a boundary failure.

## 7. Reproducing

```bash
python -c "
from tier_a_audit import planning as pl
r = pl.build_plan_report([2,4,6,8,12,20], [10,25,50,100], rate=0.90,
      simulate_at=[(6,25,0.05),(6,25,0.10),(12,25,0.05),(12,25,0.10),(20,25,0.05),(20,25,0.10)],
      n_rep=120, n_boot=300, seed=0)
print(pl.human_readable(r))"
```

Run from `model_labs/` in `pm-annotate`. The stratum-yield table comes from
`selector.build_frame` over the six wells at threshold `440.76596787901417`, executed inside
`selector.read_only_guard()`; nothing under `Conversion_Efficiency/` was modified.

Tests: **496 passed** (483 before this module, 13 added).
