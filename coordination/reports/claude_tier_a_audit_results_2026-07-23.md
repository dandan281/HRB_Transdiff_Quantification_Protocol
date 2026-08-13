# Tier-A conversion audit — results (read-only)

**From:** Claude model lane
**Date:** 2026-07-23
**Code:** `model_labs/tier_a_audit/` (`audit.py`, `README.md`); tests
`model_labs/tests/test_tier_a_audit.py` (13, all pass)
**Output:** `model_labs/tier_a_audit/_audit/` (`reproduced_visualize_final.json`,
`audit_manifest.json`)
**Companion:** `claude_tier_a_release_handoff_2026-07-23.md`

**Nature of this document:** a read-only reproduction + provenance audit. It does
**not** change the production method, does **not** declare Tier A released, and does
**not** make the ring method project-canonical — that is the integrator's
canonical-plan reconciliation. `Conversion_Efficiency/**`, `DEVELOPMENT_PLAN.md`, and
`WORKBOARD.md` were treated as read-only; a test asserts the audit writes nothing
under `Conversion_Efficiency/`.

---

## 1. Reproduction — the declared method reproduces EXACTLY
The declared frozen operating point (`New_Quantif_P23/README.md` §"Operating point":
10 µm cytoplasmic ring, pooled log-Otsu threshold, one plate-wide value, nucleus
area 50–500 µm²) was re-executed from the package's cached intermediates
(`dbs_cache/*_dbs.npy` + `plate23_nuclei/*_masks.npy`), with `ring_intensity` /
`classify` transcribed verbatim. **Pooled-Otsu threshold reproduced = 440.77 raw
units** (declared 440.77). All six wells match `visualize_final.json` to the count:

| well | condition | Desmin+ / valid | conversion | fold | matches declared |
|---|---|---|---|---|---|
| 23_B02_ctrl | control | 1,166 / 7,635 | 15.27 % | 1.00× | ✅ |
| 33_C09_br223_trka | br223/trka | 1,616 / 8,947 | 18.06 % | 1.18× | ✅ |
| 29_C05_br223_egfrc | br223/egfrc | 1,750 / 7,210 | 24.27 % | 1.59× | ✅ |
| 19_B06_act104_trka | act104/trka | 2,748 / 8,440 | 32.56 % | 2.13× | ✅ |
| 32_C08_br223_igf1r | br223/igf1r | 3,341 / 10,114 | 33.03 % | 2.16× | ✅ |
| 22_B03_act104_egfrc | act104/egfrc | 3,749 / 9,524 | 39.36 % | 2.58× | ✅ |

The declared numbers are reproducible from the stated inputs and method. **Fold
values are descriptive, single-plate, and are NOT treatment effects** (see §6).

## 2. One plate-wide operating point — verified
A single pooled-Otsu threshold is computed once over the kept cells of all six wells
and applied to every well. Checks all pass: the declared `threshold_method` is
`pooled_otsu_log_uniform`; the reproduced threshold matches the declared to < 0.5
raw units; no per-well threshold field exists. A unit test also shows the pooled
threshold is genuinely from the concatenated distribution, not any single well's
Otsu — i.e. no per-well tuning.

## 3. C08 nucleus reconciliation — PROVEN
The 10,114 (canonical) vs 10,560 (MyoFuse-local) discrepancy is fully explained:

- **Same masks.** `Conversion_Efficiency/cp_c08_full/cellpose_masks.npy` and
  `plate23_nuclei/32_C08_br223_igf1r_masks.npy` are **byte-identical arrays**
  (`np.array_equal` True; same shape 3636×3636; 10,588 labels). It is one Cellpose
  run referenced from two paths — **not** a mask-source discrepancy.
- **The difference is the size gate**, on those identical masks:

  | filter | count | used by |
  |---|---|---|
  | all labels | 10,588 | — |
  | ≥ 30 px, no upper bound | 10,562 | MyoFuse `test_desmin_premise.py` (`MIN_NUCLEUS_PX=30`) |
  | area ∈ [50, 500] µm² | **10,114** | canonical `visualize_final.py` |

  MyoFuse's 10,560 = the 10,562 at ≥30 px minus 2 nuclei dropped by its extra
  ring-validity filters (`ring≥20 px`, `around>0`). The canonical gate instead
  removes 474 more nuclei (small 30–119 px and large > 1,186 px).

The audit **fails closed** (`proven=False`) if the masks were not identical or the
canonical count did not reproduce 10,114; here `proven=True`.

## 4. Method distinction — kept explicitly separate
| result | C08 | method | status |
|---|---|---|---|
| **declared ring** | 3,341 / 10,114 = **33.03 %** | 10 µm ring + pooled log-Otsu | reproduced here |
| **superseded traced-fiber** | 670 / 10,114 = **6.6245 %** | 50 % traced-fiber/territory overlap | plan §8; README marks "DO NOT use" |
| robustness sweeps | `conversion_v2` / `absolute_desmin` k-sweeps | absolute-threshold diagnostics | **not** operating points |

The ring (33 %) and traced-fiber (6.6 %) numbers are different definitions on the
**same 10,114 valid nuclei**. The k-sweeps are robustness diagnostics and must not be
shown as competing results (this corrects the error in handoff v1).

## 5. Provenance manifest (SHA-256, first 16 hex)
`_audit/audit_manifest.json` hashes every input. Selected:

| artifact | sha256… |
|---|---|
| source image `32_C08…igf1r.nd2` | `d7902b3c0103f542` |
| nucleus mask `32_C08…_masks.npy` | `ca15dffda03f52ff` |
| Desmin cache `32_C08…_dbs.npy` | `76cd48e8bac6e069` |
| per-cell cache `percell_values_r10.0.npz` | `1ff4f517fd141b1e` |
| script `visualize_final.py` | `0c52b1454f9b7b6f` |
| declared `visualize_final.json` | `2d84b2b2a221890f` |
| parameter configuration | `9fec331a85f377e4` |
| reproduced output JSON | `2134a38a6fc14232` |

All six nd2, six masks, six Desmin caches, and the scripts are hashed; see the
manifest for the complete set.

## 6. Unresolved — required before any conversion/territory release
The audit proves the number is **reproducible and internally consistent**. It does
**not** establish that the number is **correct**, and these remain open (integrator /
Conversion_Efficiency owner):

1. **Canonical-plan reconciliation.** The method is frozen *in the package* but not
   *project-canonical* — the plan still records the superseded 6.6245 %. The plan
   must be updated to one canonical definition.
2. **Orthogonal validation of the ring positivity call.** A 2-D Desmin readout cannot
   determine whether a nucleus is inside vs above/below a myotube (missing z-axis).
   Validation requires confocal z-stacks, an added marker (e.g. MyHC), and/or the
   package README's own top proposal — a **Desmin-negative control well** to turn the
   pooled-Otsu threshold from a data-driven guess into a measurement. A 2-D Desmin
   review cannot substitute.
3. **Absolute-level uncertainty.** The per-cell distribution is unimodal with no
   valley, so the threshold sits on a shoulder; fold-changes are more stable than
   absolute %. Direction/magnitude of any bias is **not** established for this
   ring method (do not label it an "upper bound").
4. **Sampling.** One 2-D field per well; between-well field variance is unmeasured.
   "~6 fields/well" is a planning hypothesis, not a derived requirement.
5. **Statistical-plan constraint.** One plate → descriptive only; no treatment-effect
   claim; inference needs ≥3 biological replicates + a prospective plate.

## 7. Recommendation (unchanged from the handoff, now evidence-backed)
The declared ring method is **reproducible, plate-wide, and provenance-pinned** — a
sound basis for the integrator's canonical-plan reconciliation. **Hold** conversion
efficiency and Desmin territory from release pending §6.1–6.2; total/valid nuclei may
release as descriptive single-plate measurements once the mask source + hashes are
frozen (this audit supplies those hashes). T02/T03 development is unaffected.

**Handoff to Codex:** adopt for integrator review and canonical-plan reconciliation.
Nothing here is committed; no plan/workboard edit was made.
