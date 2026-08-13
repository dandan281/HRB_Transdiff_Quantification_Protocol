# Tier-A orthogonal validation & field-sampling protocol — design proposal

> **STATUS: DEFERRED (operator decision, 2026-07-23).** The operator has no
> Desmin-negative control well and is not pursuing new acquisition now, so this
> whole thread — control-well threshold calibration, z-stacks, and the field-sampling
> pilot — is **future work**. The design below is complete and ready to execute.
> **Resume trigger:** a Desmin-negative condition (undifferentiated / secondary-only
> cells) becomes available, or a decision is made to acquire one. Until then, Tier-A
> conversion/territory remain **held** (per the audit + handoff), and MyHC (561 nm) is
> set aside as a 2-D marker that would not resolve the z-question anyway.

**From:** Claude model lane
**Date:** 2026-07-23
**Prereq:** `claude_tier_a_audit_results_2026-07-23.md` (method reproduced, plate-wide,
provenance-pinned) and `claude_tier_a_release_handoff_2026-07-23.md` (release held).

**Status / governance.** This is a **design proposal**, not an authorization to
acquire data or a released protocol. It is written in the Claude lane
(`coordination/reports/claude_*`) for ratification by the **operator** (biological /
acquisition decisions) and **Codex** (statistical authority: sample size and
acceptance criteria must be reconciled with `STATISTICAL_ANALYSIS_PLAN.md`). Nothing
here changes the production method, releases Tier A, or edits the plan/workboard.

---

## 1. What must be validated (and what the audit already settled)
The audit proved the conversion number is **reproducible, plate-wide, and
provenance-pinned**. It did **not** establish that the number is **correct**. Three
distinct claims remain unvalidated:

- **V1 — the positivity call is biologically true.** A nucleus is called "converted"
  when its 10 µm 2-D Desmin ring exceeds threshold. But a bright 2-D ring can arise
  because the nucleus is *inside* a myotube **or** because a Desmin+ structure passes
  *above/below* it in projection. The missing z-axis makes these indistinguishable
  in 2-D. (This is why a 2-D Desmin manual review **cannot** validate — Codex's
  correction, restated as a design constraint.)
- **V2 — the threshold is correctly placed.** Pooled-Otsu = 440.8 raw units sits on a
  **shoulder** of a unimodal (no-valley) per-cell distribution. Whether it is the
  right cut is unmeasured.
- **V3 — one field/well is adequate.** Between-field variance is unknown; "~6
  fields/well" is a planning hypothesis, not a derived number.

## 2. Design principle: orthogonality
Each claim is validated against evidence on an **independent axis** from the 2-D
Desmin ring — the z-axis (3-D), a second marker, or a true-negative population.
Anything that is still a 2-D Desmin readout (including expert manual review) is
**excluded** as non-orthogonal for V1.

## 3. Module A — Confocal z-stack reference (core; resolves V1, informs V2)
The primary orthogonal reference. Resolves the in-vs-above/below ambiguity directly.

**Acquisition (operator decision).** Confocal z-stacks (or z-stack widefield +
deconvolution) on a subset of fields, channels DAPI + Desmin (+ MyHC, Module B),
z-step ≤ ~0.5 µm through the cell layer, lateral pixel matched to the production
0.6493 µm/px (or recorded if different). Record the exact optics in the manifest.

**3-D ground-truth definition (pre-register before scoring).** A nucleus is "in a
myotube" iff, in 3-D, it is enveloped by contiguous Desmin+ (and, per Module B,
MyHC+) cytoplasm co-localized in z — *not* merely overlapping a Desmin structure at
a different z. The exact rule (contiguity, z-tolerance, minimum envelope fraction)
is fixed in advance and applied blind to the 2-D call.

**Stratified nucleus sampling — oversample the boundary.** Random sampling wastes
effort on easy calls. Sample by the 2-D ring-to-threshold ratio; the real
distribution (from the audit's ring intensities) shows where the uncertainty is:

| ring / threshold | control (23_B02) | high converter (32_C08) | validation priority |
|---|---|---|---|
| < 0.5 (clearly negative) | 62.2 % | 37.6 % | low (confirm specificity floor) |
| 0.5–0.8 | 19.7 % | 22.9 % | low–medium |
| **0.8–1.0 (just below)** | **2.8 %** | **6.5 %** | **HIGH — the boundary** |
| **1.0–1.25 (just above)** | **2.0 %** | **5.8 %** | **HIGH — the boundary** |
| 1.25–2.0 | 3.4 % | 10.6 % | medium |
| ≥ 2.0 (clearly positive) | 9.9 % | 16.7 % | low (confirm sensitivity ceiling) |

The **boundary zone [0.8, 1.25)×threshold** — where a small threshold shift flips the
call — is ~5 % of control but ~12 % of C08 nuclei. Draw the validation sample to
**over-represent that zone** across ≥2 wells (a low and a high converter), with
enough per stratum to estimate a proportion to a target half-width (e.g. ~100
nuclei/stratum for ±10 % at 95 %).

**Analysis.** Per stratum and overall, a 2×2 of 2-D call vs 3-D truth →
sensitivity, specificity, PPV, NPV. The key deliverable is the **projection
false-positive inflation**: the fraction of 2-D "converted" nuclei that 3-D shows are
*not* inside a myotube. That finally **measures the direction and magnitude of bias**
the handoff could not state. A 2-D-ring ROC against 3-D truth then shows whether
440.8 is in the optimal band or biased (V2).

## 4. Module B — MyHC (or equivalent) differentiation marker
Desmin is not fully myotube-specific; MyHC marks differentiated muscle specifically
(and is MyoFuse's marker). Acquire it on the **same z-stack fields** so the 3-D truth
uses both Desmin envelope and MyHC+ syncytium membership. This checks that "Desmin+
ring" means "genuinely differentiated," not just "near Desmin."

**Open question that may reduce acquisition:** the production nd2 already has a
**561 nm channel (ch0)** whose identity is unrecorded in the package (README lists
only 488=Desmin, 405=DAPI). **If 561 is a differentiation/conversion marker or
reporter, Module B may be partially answerable on existing acquisitions.** Operator
to confirm what ch0/561 is — this is the cheapest possible orthogonal lead.

## 5. Module C — Desmin-negative control (calibrates V2, measures specificity)
The package README's own highest-value proposal. A population that should **not**
convert (undifferentiated cells / secondary-antibody-only / non-myogenic control)
gives the **true-negative** per-cell ring distribution. Uses:
- set the threshold to a **measured target false-positive rate** (e.g. 5 % of true
  negatives called positive) instead of pooled-Otsu on a shoulder — turning the
  threshold from a data-driven guess into a measurement;
- report the method's specificity floor directly.

## 6. Field-sampling protocol (V3)
Three phases, the first two of which produce the *measured* field count that replaces
the "~6 fields" hypothesis.

1. **Variance pilot.** Acquire K ≥ 8–10 fields per well on ≥2 representative wells
   (control + high converter). Compute per-field conversion %, the between-field SD,
   and any spatial gradient (edge vs centre).
2. **Sample-size calculation.** From the between-field variance, compute
   fields-per-well for a target CI half-width (propose ±5 % absolute, or ±3 % if
   feasible) using the plan's whole-unit bootstrap or a beta-binomial (overdispersed)
   model. **Codex ratifies the method and target against `STATISTICAL_ANALYSIS_PLAN.md`.**
3. **Biological replication.** ≥3 independent differentiation batches for any
   treatment claim, with a priori power from the pilot variance and a scientifically
   meaningful effect size — per the statistical plan's prospective gate.

## 7. Pre-registered analysis & acceptance criteria
Fixed **before** acquisition (ARRIVE/SAMPL, matching the statistical plan):
- **Method accepted for descriptive absolute %** iff, against 3-D truth: specificity
  ≥ a pre-set bound, sensitivity ≥ a pre-set bound, and projection false-positive
  inflation ≤ a pre-set bound (so absolute % can be reported or bias-corrected).
- **Threshold accepted** iff pooled-Otsu 440.8 lies within the 3-D-ROC optimal band;
  otherwise adopt the Module-C-calibrated threshold.
- **Fallback on failure:** report **fold-changes only** (empirically more stable than
  absolute %), and/or promote the 3-D/MyHC method to the reference. Failure narrows
  the claim; it never relaxes a threshold.

## 8. Freeze / Tier-C linkage
Before any **prospective** validation plate is acquired: freeze the method, threshold,
ring size, area gate, and code — the audit's SHA-256 manifest supplies the hashes.
Keep the new plate outside the workspace until predictions are sealed, per
`DEVELOPMENT_PLAN.md` §12 Tier C.

## 9. What can proceed now vs what needs acquisition
**Now (Claude lane, no new imaging):**
- finalize this protocol and pre-register definitions + acceptance criteria;
- **build the tooling**: a stratified-nucleus **selector** (draw the Module-A sample
  from the audit's ring intensities, export nucleus IDs + coordinates for targeted
  z-imaging) and a **2-D-vs-3-D scoring harness** (confusion, inflation, ROC) ready
  for when 3-D data arrives — both are deterministic, testable, in my lane;
- if the 561 channel turns out to be a usable marker, a preliminary check on existing
  data.

**Needs acquisition (operator / wet-lab):** confocal z-stacks + MyHC on a subset; a
Desmin-negative control; multiple fields per well.

## 10. Decision points to ratify
**Operator (biology/acquisition):**
1. What is the **561 nm channel**? (Could be a free orthogonal marker.)
2. Can you acquire **confocal z-stacks** on a subset of fields (+ MyHC)?
3. Is a **Desmin-negative control** available or acquirable?
4. Can you acquire **≥8 fields** on a control + a high-converter well for the variance
   pilot?

**Codex (statistics):**
5. Ratify the **sample-size method and CI target** (§6.2) and the **acceptance
   criteria** (§7) against `STATISTICAL_ANALYSIS_PLAN.md`.

## 11. Constraints honored
Read-only w.r.t. `Conversion_Efficiency/**`, the plan, and the workboard. No Omnipose
training, no linker/annotation round, no treatment-effect claim on same-plate folds.
Orthogonal validation is required and a 2-D Desmin review cannot substitute. This
document proposes; it does not authorize acquisition or declare Tier A released.

**Handoff:** to the operator and Codex for ratification. On ratification, I can build
the §9 tooling immediately.
