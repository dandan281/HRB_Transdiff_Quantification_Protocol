# New_Quantif_P32 — conversion efficiency, PLATE_32

Same pipeline, code, and parameters as the other plates, applied to PLATE_32
(10 wells: control + 9 two-receptor combination treatments). Run **from
`Conversion_Efficiency/`**.

## Operating point (fixed, plate-wide, data-driven)
Cellpose `cellprob=0` · nucleus area 50–500 µm² · Desmin top-hat r=40 (raw units)
· 10 µm cytoplasmic ring · pooled-Otsu threshold **237.2 raw units** (P32's own data).

## Result — REAL, modest treatment effect

| well | condition | total nuclei | valid nuclei | Desmin+ nuclei | conversion | fold |
|---|---|--:|--:|--:|--:|--:|
| 23_B02_ctrl | control | 13,400 | 12,356 | 3,178 | **25.7 %** | 1.00× |
| 27_C03_egfrc_act104 | egfrc+act104 | 14,079 | 12,284 | 3,942 | 32.1 % | 1.25× |
| 47_D02_trka_bmpr211m2 | trka+bmpr2 | 12,792 | 10,563 | 3,452 | 32.7 % | 1.27× |
| 29_C05_trka_act104 | trka+act104 | 13,102 | 11,309 | 3,846 | 34.0 % | 1.32× |
| 40_D09_trka_br223 | trka+br223 | 14,395 | 13,201 | 4,800 | 36.4 % | 1.41× |
| 45_D04_her2mb_br223m2 | her2mb+br223 | 19,099 | 15,704 | 5,821 | 37.1 % | 1.44× |
| 26_C02_fgfr_act104 | fgfr+act104 | 16,064 | 14,265 | 5,365 | 37.6 % | 1.46× |
| 35_C11_igf1r_bmpr211m2 | igf1r+bmpr2 | 13,550 | 11,843 | 4,546 | 38.4 % | 1.49× |
| 41_D08_igf1r_br223 | igf1r+br223 | 16,055 | 14,344 | 6,026 | 42.0 % | 1.63× |
| 38_D11_her2mb_br223 | her2mb+br223 | 13,226 | 12,003 | 5,141 | 42.8 % | **1.67×** |

Plate: 145,762 raw nuclei → 127,872 valid; 46,117 Desmin+.

**This result is trustworthy** (unlike P28): the control is the LOWEST well and a
clear outlier below the density trend (`density_confound.png`); density–conversion
correlation is only r = 0.34 (P28 was 0.99); and the control has the thinnest
high-Desmin shoulder in `percell_desmin.png`. All 9 treatments exceed control,
1.25–1.67×. Top responders: **her2mb+br223 (1.67×), igf1r+br223 (1.63×)**.

## Cross-plate notes
- Effect is **modest** (1.25–1.67×) vs P23's strongest (2.1–2.6×). These are
  two-receptor combinations, a different panel from P23/P26.
- `igf1r+br223` = 1.63×; on P23/P26 single `br223_igf1r` gave 2.2–2.4× — not directly
  comparable (combination vs single, different plate).
- **Absolute % not comparable across plates** (acquisition differs: P32 dbs median
  ~115, threshold 237, vs P23 threshold 441). Compare FOLDS. Same caveat: per-cell
  distribution unimodal/no valley → absolute level uncertain, folds robust.

## Files
- `visualize_final.py` → `conversion_summary_bar.png`, 10× `*_percell_classified.png`,
  `visualize_final.json`
- `percell_desmin.py` → `percell_desmin.png`, `.json`
- `density_confound.py` → `density_confound.png` (r=0.34, confirms not a density artifact)
- `build_dbs.py` → `dbs_cache/`; nuclei masks: `../plate32_nuclei/`
