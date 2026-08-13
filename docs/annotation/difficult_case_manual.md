# PrecisionMyotube — Difficult-Case Annotation Manual

**Status: DRAFT — pending biological reviewer approval (CL04 / G1).**
Claude drafts and organizes this manual but **cannot approve biological
definitions alone**. Two biological reviewers must sign off (see
[Approval record](#approval-record)) before it governs production annotation.

Version: 0.1 (draft) · Lane: Claude Code (Annotation documentation) · Scope: `docs/annotation/**`

---

## 0. The one rule everything serves

> **Precision before coverage.** If an independent myotube cannot be identified
> from the pixels in front of you, mark it **ambiguous**. Never split or merge by
> guesswork. An ambiguous object never enters length, width, or multinucleation
> statistics — but it is *not* discarded, and Desmin territory / conversion
> efficiency still use it.

The annotation target is a **full-area mask of one independently traceable
myotube**, plus a status and (for reviewed complete objects) a reviewer. It is
not a centerline and not a connected Desmin blob.

---

## 1. Statuses (the only four)

These are the frozen schema statuses. Assign exactly one per object.

| Status | Meaning | Enters length/width/multinucleation? |
|---|---|---|
| `complete` | The whole cell body is visible and its identity as one myotube is unambiguous. | **Yes**, only once `reviewed=true`. |
| `border_truncated` | A real myotube that runs off the field edge (identity clear, extent censored). | No (length would be censored). |
| `occluded` | Identity is clear but part of the body is hidden by an overlapping structure. | No. |
| `ambiguous` | Identity or boundary cannot be resolved from the available evidence. | No. |

Only `complete` + `reviewed=true` is **authoritative**. A convenient UI action can
never grant this in bulk — status and review are set one object at a time.

---

## 2. The complete case (the easy positive)

Annotate an object as `complete` when **all** hold:

1. The full Desmin+ body is inside the field (no edge contact — otherwise
   `border_truncated`; the analysis also auto-demotes border-touching masks).
2. Its two ends are visible and you can trace one continuous body between them.
3. No unresolved contact forces a guess about where this cell ends and a
   neighbour begins.
4. Nuclei are used as *context only* (see §5), not to define the boundary.

**Completion check (CL04.1):** two reviewers independently agree on the mask
extent and the endpoints for the shared complete examples.

---

## 3. Status decision tree

```
Is the whole Desmin+ body of ONE cell visible and its identity unambiguous?
├─ YES → does it touch the field border?
│        ├─ YES → border_truncated
│        └─ NO  → is part of the body hidden by an overlapping structure?
│                 ├─ YES → occluded
│                 └─ NO  → complete   (then review → reviewed=true to make authoritative)
└─ NO  → can you resolve identity/boundary from VISIBLE evidence alone?
         ├─ YES → resolve it (draw the supported split/merge), then re-enter the tree
         └─ NO  → ambiguous        (never guess a boundary that is not visible)
```

**Completion check (CL04.2):** reviewers assign the same status to the pilot
examples; visible incompleteness (`border_truncated`/`occluded`) is kept separate
from uncertain identity (`ambiguous`) so censored lengths never contaminate the
complete-cell distribution.

---

## 4. Contact and crossing evidence — when a split is allowed

A **split** (declaring two independent myotubes where Desmin is continuous) is
permitted **only** when the pixels support it. Acceptable evidence:

- A visible boundary / membrane gap between the two bodies.
- A clear intensity valley separating two distinct fiber cores that each has its
  own continuous body and independent endpoints.
- At a **crossing**, two fibers pass through each other with directions that
  continue consistently on both sides (each body is traceable straight through).
  Annotate **both** as separate overlap-safe masks that share the crossing pixels
  — the tool preserves this; a flat label image cannot.

**Not** acceptable evidence for a split:

- "There must be two because there are two nuclei." (Nuclei are context, §5.)
- A thin waist with no boundary and no directional evidence.
- Symmetry or expectation. If you are inferring an invisible boundary → `ambiguous`.

A **merge** (joining Desmin into one cell) is permitted only when one continuous
body with consistent direction and no separating boundary is visible. Do not
merge across a dark gap.

**Completion check (CL04.3):** no rule in this section asks a reviewer to guess
through an absent boundary.

---

## 5. DAPI / nucleus use

- Nuclei provide **context**: they help you see where cells are and support the
  separate nucleus-per-myotube measurement.
- Nuclei do **not** define cell boundaries. Do not draw a mask boundary to match
  an expected nucleus count, and do not split a body because it "should" have one
  nucleus per cell.
- Examples in this manual deliberately include **nuclei near but outside** a
  myotube territory, and multinucleated complete cells, so reviewers internalize
  that nucleus position ≠ boundary.

**Completion check (CL04.4):** the worked examples include nuclei near but
outside territory and are annotated without letting nucleus count drive the mask.

---

## 6. Adjudication escalation

When two annotators disagree:

1. **Categorize** the disagreement (feeds the correct lane, CL04/H02):
   - *tool problem* → request to Claude (annotation-tool path);
   - *unclear rule* → revise this manual (with reviewer approval);
   - *genuinely missing evidence* → resolve as ambiguous, not by fiat.
2. **Adjudicate with raw channels.** An expert adjudicator inspects the native
   fiber + DAPI (and any annotations) and decides: consensus `complete`/other, or
   retained `ambiguous`.
3. **Every unresolved disagreement has an allowed final state** — consensus or
   `ambiguous`. No object is forced to `complete` to break a tie.
4. Corrections flow into a new **versioned** annotation snapshot; prior snapshots
   are never overwritten.

**Completion check (CL04.5):** every unresolved pilot disagreement ends in an
allowed final state (consensus or ambiguous).

---

## 7. What this manual must never do

- Never ask a reviewer to infer a boundary that is not visible.
- Never let a proposal/prompt be treated as truth (prompts are visually distinct
  and non-authoritative in the tool).
- Never expose **Plate 26** examples during development — it is the locked test
  set. All worked examples here come from development plates only.

---

## 8. Worked examples (to be attached before approval)

The following illustrated cases must be added (with images from development
plates) and agreed before this manual governs production:

- [ ] Isolated complete fiber (easy positive) — mask extent + endpoints.
- [ ] Border-truncated fiber vs a genuinely short complete fiber.
- [ ] Occluded fiber (identity clear, body partly hidden).
- [ ] Resolvable crossing (two overlap-safe masks sharing crossing pixels).
- [ ] Unresolvable contact (→ ambiguous): touching bodies, no boundary.
- [ ] Dense field: several contacts, mix of statuses.
- [ ] Multinucleated complete cell + nuclei near-but-outside territory.
- [ ] Dim/thin fiber at the visibility limit.

---

## Approval record

This manual is **not in force** until both entries are signed.

| Reviewer | Role | Approved version | Date |
|---|---|---|---|
| _pending_ | Biological reviewer 1 | — | — |
| _pending_ | Biological reviewer 2 | — | — |

Unclear rules found during the pilot (H01/H02) are reported back here **before**
scale-up (D02).
