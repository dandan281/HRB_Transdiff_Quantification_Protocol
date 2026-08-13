# Tier-A release handoff — decision package for the integrator (v2, corrected)

**From:** Claude annotation/model lane
**Date:** 2026-07-23
**Companion:** `claude_tier_a_findings_2026-07-23.md` (the two findings)
**Revision:** v2 corrects v1 after Codex integrator review. v1's central claim —
"the conversion number is unfrozen and swings 5–31% by parameter" — **was wrong**:
it was read off the robustness-sweep diagnostics (`conversion_v2.json`,
`absolute_desmin.json`, both listed under *Diagnostics/robustness* in the package
README) instead of the **declared final method** at
`New_Quantif_P23/README.md` §"Operating point" / `visualize_final.json`. Corrections
are recorded in §11. Codex's release ruling is adopted in §8.

**Status:** decision surface for Codex/integrator. I cannot edit `DEVELOPMENT_PLAN.md`,
`WORKBOARD.md`, or the `Conversion_Efficiency/` lane. Numbers are read from the
Conversion_Efficiency output files; the Conversion_Efficiency owner must confirm them.

---

## 1. What Tier A is
Per `DEVELOPMENT_PLAN.md` §12, Tier A is the mature, field-level product: **total /
valid nucleus count**, **Desmin territory %**, and **conversion efficiency %**. It
does not depend on the single-myotube (Tier B) segmentation work and can proceed on
its own.

## 2. The blocking issue: two declared conversion methods conflict and must be reconciled
The newer Conversion_Efficiency package **does** declare a single, frozen,
data-driven operating point — `New_Quantif_P23/README.md` §"Operating point (fixed,
plate-wide, data-driven)", realized in `visualize_final.json`:

| stage | parameter | value |
|---|---|---|
| nuclei | Cellpose-SAM `cellprob_threshold` | 0 (plateau) |
| nuclei | area cut | 50–500 µm² |
| Desmin | background removal | white top-hat, disk r=40, raw camera units |
| Desmin | per-cell readout | mean bg-subtracted Desmin in a **10 µm cytoplasmic ring** |
| Desmin | positivity threshold | **440.8 raw units** = Otsu on the **pooled** per-cell distribution |

Headline (`visualize_final.json`): control `23_B02` = **15.27 %**; `32_C08` =
**33.03 %** (3,341 / 10,114); folds 1.0–2.58× across wells.

The conflict is with the **plan's** number. `DEVELOPMENT_PLAN.md` §8 records the
Tier-A baseline as **6.62 %** for C08 — but that is the older **traced-fiber /
fusion-index** method, which the package README explicitly lists as **"Superseded —
DO NOT use for reporting"** (it required nuclei inside long traced fibres and
discards mononuclear converted cells, so it reads too low). So the situation is not
"no operating point"; it is **the plan still cites the superseded method (6.62 %)
while the declared current method reads 33.03 %.**

**Status distinction (per Codex integrator review):** the 10 µm ring / pooled-Otsu
method is **frozen within the Conversion_Efficiency package**, but it is **not yet
project-canonical** — the development plan still records the superseded 6.6245 %
method as the Tier-A baseline. Frozen-in-package ≠ authoritative.

**Decision required — the authoritative next step:** run the **method-reconciliation
audit** of the newer declared method (README §13), ratify (or revise) its operating
point against orthogonal evidence (§7), then **update the plan to a single canonical
conversion definition and number**. Until that plan update lands, the release hold
in §8 remains correct. This is integrator / Conversion_Efficiency-owner work: I
cannot edit the plan or that lane. The
`conversion_v2`/`absolute_desmin` k-sweeps are **robustness diagnostics** (k is a
background-noise threshold multiplier for an absolute-threshold sweep, **not** a
smoothing kernel and **not** competing operating points); they show how sensitive
the absolute % is, which motivates §3.

## 3. The absolute conversion % has real, two-directional uncertainty
The README's own caveat: the per-cell Desmin distribution is **unimodal with no
valley** (`percell_desmin.png`), so the Otsu threshold sits on a shoulder. That
makes the **absolute %** genuinely uncertain, while **fold-changes are far more
stable** — the package's robust claim is "B03/C08/B06 convert ~2.1–2.6× above
control," not the absolute levels.

**Correction to v1 (per Codex):** do **not** label the current result an "upper
bound." MyoFuse demonstrated that a **2D pixel-mask-overlap** method can overestimate
fusion, but that does not establish the **direction or magnitude** of bias for this
**ring-based** method. The honest statement is: the absolute conversion % carries
uncertainty of unestablished sign and size; fold-changes are the more trustworthy
readout.

## 4. Sampling adequacy is a planning hypothesis, not a derived requirement
Both methods use a **single 2-D field per well**. The MyoFuse "~6 fields/well for a
±5 % CI" arithmetic is a **planning hypothesis** for how many fields might be needed
— not a requirement derived for our ring method or our variance. Treat it as a
prompt to estimate our own field-to-field variance, which one field per well cannot
provide.

## 5. Binding constraint from the statistical plan: descriptive only
`STATISTICAL_ANALYSIS_PLAN.md` is binding: all six wells are **one plate**; they
support **descriptive field/well summaries and internal model evaluation only**;
nested nuclei are subsamples, **not** biological replicates; an inferential release
needs **≥3 independent biological replicates**, locked analysis, and the **adverse
95 % bound** passing. So no treatment-effect claim (e.g. "act104 raises conversion")
can ship on Plate 23, regardless of how clean the number is.

## 6. Provenance items to reconcile before any release
- **Which conversion method is canonical** (§2) — plan 6.62 % (superseded) vs
  declared 33.03 %.
- **Nucleus-count provenance:** the canonical pipeline reports **10,114** valid C08
  nuclei (`visualize_final.json`); the local MyoFuse analysis used **10,560**. The
  exact mask source + integrity hashes must be frozen so one count is authoritative.
- **Desmin territory %** shares the per-image-normalization history (README "The bug
  this package fixed") and must be tied to the same frozen mask source.

## 7. Validation needs orthogonal evidence — a 2-D Desmin review cannot do it
**Correction to v1 (per Codex):** v1 proposed a 2-D operator "nucleus in/out on
Desmin" labelling as the hinge experiment. That **cannot** resolve the real question
— whether a nucleus lies **inside** the myotube vs **above/below** it in projection —
because a human viewing 2-D Desmin faces the **same missing-z-axis ambiguity** as the
algorithm. Proper validation needs an **orthogonal reference**:
- **confocal z-stacks** (resolve the axial position directly), and/or
- **an additional validated marker** (e.g. a membrane / MyHC co-stain), and/or
- the package README's own highest-value proposal: a **Desmin-negative control well**
  (secondary-only / unconverted fibroblasts), which would turn the pooled-Otsu
  threshold from a data-driven guess into a measured one.

All three are acquisition / wet-lab decisions, not code changes. This is the same
conclusion the biggest-risk analysis reached: the ceiling here may be the imaging,
not the software.

## 8. Adopted release ruling (Codex integrator verdict)
- **Hold conversion efficiency and Desmin territory** from release pending the §2
  audit and §7 orthogonal validation.
- **Total / valid nucleus counts** may release **only as descriptive single-plate
  measurements**, and only after freezing the exact mask source and integrity hashes.
- **No treatment-effect claims.**
- **T02 / T03 development continues** — this Tier-A issue does not block it.

## 9. Integrator / Conversion_Efficiency-owner action list
1. Audit the newer declared final method (`New_Quantif_P23/README.md` §"Operating
   point"), ratify or revise the operating point, and reconcile it against the plan's
   superseded 6.62 % so one canonical conversion definition is authoritative (§2).
2. Freeze the exact mask source + integrity hashes for nuclei and Desmin territory;
   reconcile the 10,114 vs 10,560 count (§6).
3. Plan orthogonal validation (z-stacks / added marker / Desmin-negative control
   well) before any conversion or territory claim (§7).
4. Confirm descriptive-only, no-treatment-claim framing (§5, §8).
5. Rerun all six wells at the ratified operating point once frozen.

## 10. Recommended near-term scope
Release **only** total/valid nuclei as descriptive single-plate measurements (after
§6 freezing). Hold conversion and territory. Report fold-changes, if shown at all, as
the more stable readout with explicit descriptive-only, single-plate caveats — never
as a biological effect. Keep the pipeline moving toward the §7 orthogonal validation,
which is the real unlock.

## 11. Corrections in this revision + provenance
Corrected after Codex integrator review (verified against the files):
- **Central reframe:** v1 "unfrozen, 5–31 % swing" → v2 "two declared methods
  conflict (plan 6.62 % superseded vs declared 33.03 %)." The 5–31 % came from
  reading the k-sweep **diagnostics** as operating points.
- **`k`** is a background-noise **threshold multiplier**, not a smoothing kernel.
- **`absolute_desmin.json` attribution fixed:** the 15.0/8.3 values are **C09**, not
  C08; C08 is **26.63 / 14.20** (verified in the file).
- **"Upper bound" removed** — bias direction/magnitude for the ring method is not
  established by MyoFuse's 2-D-mask result (§3).
- **Hinge experiment corrected** — a 2-D Desmin review cannot resolve the z-axis;
  orthogonal validation required (§7).
- **"Six fields"** demoted to a planning hypothesis (§4).
- **Nucleus-count provenance** (10,114 vs 10,560) added (§6).

I did **not** audit or modify the Conversion_Efficiency pipeline; numbers are read
from `New_Quantif_P23/visualize_final.json`, `absolute_desmin.json`, `README.md`, and
`DEVELOPMENT_PLAN.md` §8. Nothing is committed; no plan/workboard edit is made. All
Plate-23 evidence is single-operator, retrospective, proposal-conditioned.
