# New_Quantif_P44 — conversion efficiency, PLATE 44 (Tdiffs)

40 wells (B02–B11, C02–C11, D02–D11, E02–E11), 739,280 cells quantified.
Run from `Conversion_Efficiency/`. Every stage is resumable.

```
cpenv/Scripts/python.exe New_Quantif_P44/pilot_cellprob.py      # operating point
cpenv/Scripts/python.exe New_Quantif_P44/run_nuclei.py --cellprob 0
cpenv/Scripts/python.exe New_Quantif_P44/build_dbs.py
cpenv/Scripts/python.exe New_Quantif_P44/percell_desmin.py
cpenv/Scripts/python.exe New_Quantif_P44/confound_checks.py
```

---

## Headline

**The quantification is sound. The conversion readout is NOT interpretable as
treatment effects on this plate** — the same verdict Plate 9 received, for the
same reason, plus a missing plate map.

| check | result | verdict |
|---|---|---|
| density confound | Pearson r = **+0.21** (p = 0.2) | clean — not the P28 artifact (r = 0.99) |
| plate-position trend | column r = −0.29, row r = +0.27 | mild, below the 0.4 flag |
| spread across all 40 wells | 4.5 – 20.3 % (**4.5×**); excluding the one dead well, 9.4 – 20.3 % (**2.1×**) | **within noise** |
| **yardstick** | Plate 9's **six control wells alone** spanned 11.3 – 46.7 % = **4.1×** with no treatment difference | the entire P44 range is *smaller* than that |
| assumed control 23_B02 | 15.7 %, **rank 27 of 40** (67th percentile) | 26 of 39 other wells fall *below* it |

The last row is the decisive one. On PLATE_32 — the plate whose result was
trusted — the control was the **lowest** well and a clear outlier below the
density trend. Here the assumed control sits two-thirds of the way up the
ranking, so either the treatments largely do not work, or well-to-well variance
dominates, or 23_B02 is not this plate's control. All three mean the same thing:
**do not report a responder list from this plate.**

---

## Two blockers, both external

1. **No plate map.** P44 filenames carry no treatment token (`23_B02.nd2`, not
   `23_B02_ctrl.nd2` as on every other plate), and no layout sheet for this plate
   exists in the repo. **No condition has been invented for any well** — the same
   rule `Plate9_C6C2_QTFCs` applied to its blank G09/G10. Wells are reported
   individually by id. Supply the sheet and the treatment grouping, replicate
   averaging, and fold-changes all follow immediately from the numbers below.
2. **Control identity is assumed, not known.** `CTRL = "23_B02"` is a *position
   convention* — position 23 = B02 is the control on PLATE_23/26/28/32. It is
   flagged `control_is_assumed_by_position: true` in the JSON. Every absolute
   per-well percentage is independent of this choice; only the `fold` column
   depends on it.

---

## This plate is a different acquisition — three traps

Verified read-only from OME metadata on **all 40** files; every well agrees.

| | PLATE_23/26/28/32 | **PLATE 44** |
|---|---|---|
| channel 0 | 561 (receptor) | **DAPI 429 nm (nuclei)** |
| channel 1 | 488 Desmin | 488 Desmin *(same)* |
| channel 2 | 405 DAPI (nuclei) | **AF546 571 nm (receptor)** |
| frame | 3636² | **1818²** |
| pixel | 0.650017 µm | **1.724571 µm** (2.65× coarser) |
| field | 2.36 mm (5.59 mm²) | **3.14 mm (9.83 mm²)**, 1.76× the area |
| depth | 16-bit | **12-bit** (max 4095) |

1. **Channels 0 and 2 are swapped.** Desmin is channel 1 on both, so running the
   old `--nuclei-ch 2` default would have produced a perfectly plausible myotube
   mask while segmenting the *receptor* channel as nuclei. Nothing in the output
   would have looked wrong. (Plate 9 shares this DAPI-first order.)
2. **Pixel size differs by 2.65×.** Every pixel-valued parameter is re-derived
   from its physical size in `p44_layout.py`, never copied: top-hat 26 µm =
   **disk(15 px)** here vs disk(40 px) there; ring 10 µm = **6 px** vs 15 px.
   Copying `40` across would have applied a 69 µm top-hat.
3. **Raw units are not cross-plate comparable** (12-bit, different acquisition),
   which is why the Desmin threshold is always re-derived from this plate's own
   pooled distribution. Standing rule, not a P44 exception. **Compare folds
   across plates, never absolute percentages.**

### Resolution was checked, not assumed
A 50–500 µm² nucleus is only 17–168 px here (118–1185 px on PLATE_2x), so
segmenting at native resolution needed justification. Cross-check on the control
well at the chosen operating point (`pilot_cellprob.json`):

| | valid nuclei | median area |
|---|--:|--:|
| native 1.72 µm/px | 17,759 | 157.6 µm² |
| resampled to 0.65 µm/px | 18,292 | 151.7 µm² |
| difference | **+3.0 %** | −3.7 % |

3 % agreement, and native is 5× faster with no interpolation → **native is used.**

---

## Operating point (fixed, plate-wide, data-driven)

Cellpose-SAM `cellprob = 0` · nucleus area 50–500 µm² · Desmin top-hat 26 µm
(disk 15 px, raw units) · 10 µm cytoplasmic ring (6 px) · pooled-Otsu threshold
**118.9 raw units** from P44's own 739,280 cells.

`cellprob = 0` was chosen by a 5-point sweep on four pilot wells spanning the
plate (B02, C07, D05, E11). **All four plateaued at 0, unanimously** — relative
neighbour sensitivity 0.002–0.008. Applied unchanged to all 40 wells; per-well
tuning is forbidden, and no threshold was chosen to make a number come out.

---

## Per-well results

739,400 valid nuclei segmented → 739,280 with a measurable cytoplasmic ring.
Sorted by conversion. **`fold` is against the *assumed* control.**

| well | id | cells | Desmin+ | conversion | fold |
|---|---|--:|--:|--:|--:|
| 14_B11 | B11 | 14,474 | 653 | **4.5 %** | 0.29× |
| 16_B09 | B09 | 17,135 | 1,616 | 9.4 % | 0.60× |
| 58_E10 | E10 | 15,225 | 1,636 | 10.8 % | 0.68× |
| 51_E03 | E03 | 22,865 | 2,599 | 11.4 % | 0.72× |
| 34_C10 | C10 | 18,186 | 2,130 | 11.7 % | 0.74× |
| 38_D11 | D11 | 16,060 | 1,928 | 12.0 % | 0.76× |
| 15_B10 | B10 | 16,584 | 2,027 | 12.2 % | 0.78× |
| 35_C11 | C11 | 15,445 | 2,029 | 13.1 % | 0.83× |
| 45_D04 | D04 | 21,369 | 2,844 | 13.3 % | 0.85× |
| 33_C09 | C09 | 24,441 | 3,301 | 13.5 % | 0.86× |
| 27_C03 | C03 | 17,096 | 2,312 | 13.5 % | 0.86× |
| 52_E04 | E04 | 22,365 | 3,099 | 13.9 % | 0.88× |
| 21_B04 | B04 | 15,641 | 2,198 | 14.1 % | 0.89× |
| 41_D08 | D08 | 15,502 | 2,240 | 14.4 % | 0.92× |
| 40_D09 | D09 | 17,367 | 2,519 | 14.5 % | 0.92× |
| 46_D03 | D03 | 22,534 | 3,274 | 14.5 % | 0.92× |
| 19_B06 | B06 | 24,615 | 3,577 | 14.5 % | 0.92× |
| 39_D10 | D10 | 15,689 | 2,289 | 14.6 % | 0.93× |
| 22_B03 | B03 | 17,516 | 2,578 | 14.7 % | 0.94× |
| 53_E05 | E05 | 20,607 | 3,131 | 15.2 % | 0.97× |
| 29_C05 | C05 | 14,537 | 2,217 | 15.2 % | 0.97× |
| 28_C04 | C04 | 14,591 | 2,259 | 15.5 % | 0.98× |
| 30_C06 | C06 | 14,716 | 2,281 | 15.5 % | 0.98× |
| 18_B07 | B07 | 23,130 | 3,601 | 15.6 % | 0.99× |
| 50_E02 | E02 | 21,580 | 3,366 | 15.6 % | 0.99× |
| 44_D05 | D05 | 21,217 | 3,321 | 15.7 % | 0.99× |
| **23_B02** | **B02** | 17,758 | 2,795 | **15.7 %** | **1.00×** ← assumed control |
| 43_D06 | D06 | 15,077 | 2,399 | 15.9 % | 1.01× |
| 17_B08 | B08 | 16,832 | 2,686 | 16.0 % | 1.01× |
| 20_B05 | B05 | 22,975 | 3,676 | 16.0 % | 1.02× |
| 59_E11 | E11 | 14,738 | 2,409 | 16.4 % | 1.04× |
| 54_E06 | E06 | 19,864 | 3,382 | 17.0 % | 1.08× |
| 47_D02 | D02 | 15,951 | 2,773 | 17.4 % | 1.10× |
| 42_D07 | D07 | 16,707 | 2,908 | 17.4 % | 1.11× |
| 31_C07 | C07 | 20,875 | 3,763 | 18.0 % | 1.15× |
| 26_C02 | C02 | 17,066 | 3,223 | 18.9 % | 1.20× |
| 56_E08 | E08 | 19,313 | 3,664 | 19.0 % | 1.21× |
| 55_E07 | E07 | 21,160 | 4,091 | 19.3 % | 1.23× |
| 32_C08 | C08 | 21,582 | 4,364 | 20.2 % | 1.28× |
| 57_E09 | E09 | 18,895 | 3,830 | 20.3 % | 1.29× |

**Maximum effect is 1.29×.** For scale, PLATE_23's strongest wells reached
2.1–2.6× and PLATE_32's 1.67×.

### One technical outlier
**14_B11 (4.5 %) is almost certainly a failed well, not a biological result.**
Its background-subtracted Desmin p99 is **329** against 1,066–2,331 for every
other well on the plate — the Desmin channel is essentially empty. Its nucleus
count and background are normal, so this is a staining/acquisition failure in one
channel, not a dead well. Exclude it, or re-image.

---

## Files
- `p44_layout.py` — **the single source** of channel indices, pixel size and
  derived pixel parameters. Import it; never re-type a constant.
- `pilot_cellprob.py` → `pilot_cellprob.json` — operating-point sweep + the
  native-vs-resampled resolution cross-check
- `run_nuclei.py` → `nuclei/` (masks, overlays, `nuclei_results.json`)
- `build_dbs.py` → `dbs_cache/` (+ `dbs_manifest.json`)
- `percell_desmin.py` → `percell_desmin.png`, `.json`
- `confound_checks.py` → `confound_checks.png`, `.json`

Masks, the dbs cache and per-well overlays are gitignored (regenerable, ~GB).
Scripts, JSON and the two summary figures are tracked.

## What would make this plate interpretable
1. **The layout sheet** — enables treatment grouping and replicate averaging,
   which is the only way to separate effect from well-to-well variance.
2. **Confirmation of which well is the control**, and ideally **several control
   wells** — with one control there is no estimate of control variability, which
   is precisely what sank the Plate 9 conversion readout.
3. **Re-image or exclude 14_B11.**
