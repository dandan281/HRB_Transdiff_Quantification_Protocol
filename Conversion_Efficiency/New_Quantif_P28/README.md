# New_Quantif_P28 — conversion efficiency, PLATE_28

Same pipeline, code, and parameters as `../New_Quantif_P23/` and
`../New_Quantif_P26/`, applied to PLATE_28 (5 wells). Run **from
`Conversion_Efficiency/`**.

Wells: `23_B02_ctrl` (control), `17_B08_BMPR2_HER2mb`, `21_B04_br223_egfrc`,
`56_E08_br223_egfrc`, `58_E10_br223_igf1r`.

## ⚠️ RESULT IS NOT INTERPRETABLE AS A TREATMENT EFFECT

**Plate 28 shows no differential conversion.** All five per-cell Desmin
distributions overlap (`percell_desmin.png`, right panel), and the apparent
"conversion %" tracks **nucleus density almost perfectly** (Pearson r = 0.99,
Spearman = 1.00; see `density_confound.png`). The control is the densest well and
scores highest; the least-dense well (E10) scores lowest — the ranking is a density
artifact, not biology. In dense fields the 10 µm cytoplasmic ring picks up
neighbouring cytoplasm/background, inflating per-cell Desmin. On P23 the same
density correlation is only 0.61 because a real treatment effect dominates.

Do **not** report P28 as "control converts best / treatments reduce conversion" —
that is the confound, and it is biologically implausible (the same `br223_igf1r`
gave 2.2–2.4× on P23/P26; here E10_br223_igf1r visibly has MORE Desmin fibres than
the control yet scores lowest). Most likely this plate failed to differentiate, or
these wells did not induce. Check the plate before using these numbers.

## Numbers as computed (uninterpreted)

Operating point: Cellpose `cellprob=0` · area 50–500 µm² · top-hat r=40 · 10 µm ring
· pooled-Otsu threshold **365.6 raw units** (P28's own data).

| well | total nuclei (raw) | valid nuclei | Desmin+ nuclei | conversion | fold |
|---|--:|--:|--:|--:|--:|
| 23_B02_ctrl | 14,654 | 14,121 | 2,446 | 17.3 % | 1.00× |
| 17_B08_BMPR2_HER2mb | 13,764 | 12,391 | 1,896 | 15.3 % | 0.88× |
| 56_E08_br223_egfrc | 12,495 | 11,598 | 1,559 | 13.4 % | 0.78× |
| 21_B04_br223_egfrc | 12,015 | 11,593 | 1,477 | 12.7 % | 0.74× |
| 58_E10_br223_igf1r | 10,164 | 9,347 | 895 | 9.6 % | 0.55× |

Plate: 63,092 raw nuclei → 59,050 valid. **Folds are the density ranking, not treatment.**

## What would be needed to rescue an interpretation
- Confirm the plate/wells (labeling, differentiation success).
- If the effect is real but weak, it is below the density confound here; a
  density-controlled readout (e.g. Desmin+ *area fraction* per field, or matched
  cell densities) would be required. Not attempted — would not honor "same
  parameters across the plate," and there is no effect visible to rescue.

## Files
- `visualize_final.py` → `conversion_summary_bar.png`, 5× `*_percell_classified.png`,
  `visualize_final.json`
- `percell_desmin.py` → `percell_desmin.png` (the all-overlapping distributions), `.json`
- `density_confound.py` → `density_confound.png` (P28 vs P23 density-vs-conversion)
- `build_dbs.py` → `dbs_cache/`; nuclei masks: `../plate28_nuclei/`
