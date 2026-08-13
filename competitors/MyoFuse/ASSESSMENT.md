# MyoFuse — assessment: what we borrow, what stays ours

**Paper:** Lair et al., *MyoFuse is a fully AI-based workflow for automated
quantification of skeletal muscle cell fusion in vitro*, Scientific Reports (2026)
16:9387. https://doi.org/10.1038/s41598-026-40047-y
**Code:** https://github.com/BenLair/MyoFuse — MIT, pinned at
`273a2f4dafdfc88c47eb3a63d2dbf6f2e86757a4` (2026-04-13), vendored in `upstream/`.
**Assessed:** 2026-07-23. **Verdict:** *borrow the workflow shape and the
validation discipline for conversion efficiency; do not use their model; nothing
here touches individual-myotube measurement.*

---

## 1. One-line summary of what they built

Cellpose (retrained) segments nuclei → a small CNN (21 k parameters, trained via
the Svetlana Napari plugin) classifies **each nucleus** as inside or outside a
myotube, using the **myotube channel only**. Output is the fusion index:
`nuclei inside ÷ total nuclei`.

**They never segment individual myotubes.** No length, no width, no instance
count, no separation of touching or crossing fibres. Their unit of analysis is
the nucleus.

---

## 2. The decisive test we ran before adopting anything

Their entire discriminative signal is that a nucleus in the cytoplasm **displaces
MyHC** and leaves a *dark hole*, while a myoblast lying above or below the
myotube does not. That is what lets them beat the mask method.

We stain **Desmin**, an intermediate filament that forms a cage *around* nuclei.
`evidence/test_desmin_premise.py` measured Desmin intensity inside every nucleus
against its own surrounding cytoplasm ring, on the validated C08 well
(10,560 nuclei):

| population | median inside ÷ ring | showing a hole (<0.8) |
|---|---|---|
| counted converted by our 50% rule (n=701) | **1.60** | **4.9%** |
| not converted (n=9,859) | 1.03 | 2.9% |

**The premise is inverted for our stain.** Our myonuclei are *brighter* than
surrounding cytoplasm, not darker — perinuclear Desmin enrichment, exactly as the
biology predicts. Their pretrained classifier keys on the opposite sign and
cannot transfer. This is not something retraining trivially repairs: the feature
they rely on is not weakly present in Desmin, it is reversed.

(Caveat recorded honestly: the ring sits only ~42% inside Desmin territory for
thin fibres, so the ratio is a conservative estimate of contrast. A Gaussian
mixture does prefer two components among converted nuclei — means 1.24 and 2.03 —
but *both are above 1*, so this is "bright vs very bright", not "hole vs no hole".)

---

## 3. What we BORROW

| # | What | Why | Where it lands |
|---|---|---|---|
| B1 | **The critique of the mask method** | Their core result: classifying a nucleus by overlap with a myotube mask counts myoblasts lying above/below, inflating FI. They show this holds even at a **90%** overlap threshold. | `Conversion_Efficiency/` — our rule is literally `nucleus counted inside if >=50% of its area overlaps Desmin mask`, i.e. *more permissive than the threshold they already showed to be biased*. |
| B2 | **Per-nucleus classification as the workflow shape** | Replacing "overlap test" with "learned per-nucleus decision" is the right architecture, independent of which feature carries the signal. | Future Tier-A work: Cellpose nuclei → per-nucleus classifier, rather than a geometric overlap rule. |
| B3 | **Their validation design** | Single-nucleus confusion matrix + ROC/AUC, *plus* external validation on a cell type absent from training (vastus lateralis, acc 0.944). That is the standard our Tier-A claims should meet. | Tier-A release gate. |
| B4 | **The selection-bias quantification** (Fig. 7) | Free and directly actionable — see §5. | Field-count planning for every conversion-efficiency claim. |
| B5 | **Cellpose for nuclei** | Confirms our existing choice. They used Cellpose 2.0 retrained; we already run Cellpose-SAM 4.2.1.1, which is newer. | Nothing to change — this is corroboration, not a borrowing. |
| B6 | **Retraining recipe** (`upstream/How to re-train the classifier.md`) | If we ever train a per-nucleus classifier, their documented protocol (Svetlana, 200-px patches, min-max normalisation, class weights, full augmentation) is a sane starting configuration. | Reference only. |

## 4. What we do NOT take

| # | What | Why not |
|---|---|---|
| N1 | **`Models/Svetlana/MyoFuse.pth`** (their trained classifier) | Trained on MyHC dark holes. Our Desmin signal is inverted (§2). It would fail, and silently — it outputs a probability either way. |
| N2 | **Their annotation criterion** ("nucleus associated with a decrease in MyHC signal is inside") | Directly inapplicable. Adopting the *wording* into our protocol would encode a rule that is false for Desmin. |
| N3 | **Their FI definition as our headline metric** | Their FI is nuclei-based. Our Tier-A product is *conversion efficiency* over a transdifferentiation experiment at ~6.6%, not myoblast fusion at 30–50%. Related, not interchangeable. |
| N4 | **Anything for T02 / individual myotubes** | They do not segment myotubes at all. Our tracer, gap-linker, and Omnipose candidate address a problem this paper does not attempt. |
| N5 | **Their nuclei model** | Superseded by our validated Cellpose-SAM run (10,588 nuclei on C08, GPU, 16 s). |

## 5. What stays entirely OURS (no overlap with any published tool we know of)

- **Individual myotube instance segmentation** — the whole T02 effort: the
  deterministic ridge/graph floor, the parameterised crossing tracer, the
  operator-confirmed **gap-bridging linker** (28 confirmed joins, bridge-signal
  AUC 0.82), and the Omnipose candidate.
- **Per-myotube measurements** — length, width, instance count, nuclei per
  myotube. MyoFuse produces none of these.
- **The overlap-aware `InstanceSet` contract** — crossing myotubes kept as two
  masks sharing pixels. Their flat per-nucleus output has no equivalent concept.
- **The single-operator evidence discipline** — proposal-conditioned labelling,
  binding exclusions, leave-one-well-out folds, the circularity audit, blind
  repeatability. Their validation is strong but conventional; ours has to survive
  having one annotator.
- **The transdifferentiation context** — low conversion (~6.6%) changes which
  errors matter (see §6).

## 6. Two findings from their paper that change what we should do

**6a — the mask-method bias is probably *worse* for us than for them.**
At their FI of 30–50%, most nuclei are genuinely fused. At our ~6.6% conversion,
~93% of nuclei are non-converted and therefore *available* to be spuriously
counted when they happen to lie over a fibre. The same absolute contamination is
a far larger relative error. Our 6.6% should be treated as an upper bound until
measured.

**6b — we may not have enough fields per well.**
They report ~66 tiles needed for a ±5% FI confidence interval.

| | tile / field | area |
|---|---|---|
| MyoFuse tile | 1168 × 1005 px @ 0.645 µm | 0.49 mm² |
| our field | 3636 × 3636 px @ 0.6493 µm | 5.57 mm² |

One of our fields ≈ **11.4** of their tiles, so ±5% needs ≈ **6 of our fields per
well**. We currently analyse **one**. Per-well conversion-efficiency numbers
therefore carry a wider CI than we have been reporting. This is a claim-quality
issue for Tier A and should go to the integrator.

*(Sampling note: their optics are 10×/0.3 with a 6.45 µm-pixel ORCA-R2 → 0.645
µm/px; ours is 0.6493 µm/px. Effectively identical, so their tile arithmetic
transfers directly.)*

## 7. The uncomfortable implication

If Desmin cannot separate "nucleus *in* the myotube" from "myoblast lying on
top", that is an **acquisition** limit, not a software one — the same conclusion
the plan already reached for dense-cluster separation. The two fixes are a **MyHC
co-stain** (which would hand us their exact signal, and then their model and
weights become genuinely reusable) or **z-stacks**. Worth settling before
investing in a Desmin-only per-nucleus classifier.

## 8. Suggested next actions (not yet authorised)

1. Quantify 6a: have the operator label a stratified sample of nuclei in/out on
   Desmin, and measure how inflated the 50%-overlap rule actually is.
2. Route 6b to the integrator — it affects Tier-A release claims, and Tier A is
   the product closest to release.
3. Decide on 7 before building any Desmin per-nucleus classifier.
4. T02 continues unchanged; this paper has no bearing on it.
