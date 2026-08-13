# New_Quantif_P26 — conversion efficiency, PLATE_26

Same pipeline, code, and parameters as `../New_Quantif_P23/`, applied to PLATE_26
(3 wells: control + the two strongest P23 responders). Run **from
`Conversion_Efficiency/`**, e.g. `cpenv/Scripts/python.exe New_Quantif_P26/visualize_final.py`.

Wells: `23_B02_ctrl` (control), `19_B06_actv104_trka`, `32_C08_br223_igf1r`.
Channels: ch1=Desmin, ch2=DAPI. Pixel 0.6493 µm.

## Operating point (fixed, plate-wide, data-driven — identical method to P23)
Cellpose `cellprob=0` · nucleus area 50–500 µm² · Desmin white top-hat r=40 (raw
units) · 10 µm cytoplasmic ring · positivity threshold = **351.8 raw units**
(Otsu on **PLATE_26's own** pooled per-cell distribution — recomputed per plate,
because the threshold is an output of the data, not a fixed constant).

## Result

| well | condition | total nuclei (raw) | valid nuclei | Desmin+ nuclei | conversion | fold |
|---|---|--:|--:|--:|--:|--:|
| 23_B02_ctrl | control | 8,670 | 8,430 | 816 | **9.7 %** | 1.00× |
| 19_B06_actv104_trka | act104/trka | 10,536 | 9,807 | 1,457 | 14.9 % | 1.54× |
| 32_C08_br223_igf1r | br223/igf1r | 9,607 | 9,192 | 2,158 | 23.5 % | 2.43× |

Plate: 28,813 raw nuclei → 27,429 valid; 4,431 Desmin+.

## Cross-plate notes (vs New_Quantif_P23)
- **C08 (br223/igf1r) reproduces**: 2.43× here vs 2.16× in P23 — same strong response.
- **B06 (act104/trka) is weaker here**: 1.54× vs 2.13× in P23. Replicate/plate-level
  variability, worth confirming.
- **Fold-change is threshold-robust**: at P23's threshold (440.8) the P26 folds are
  identical (1.54×, 2.44×); only the absolute % shift (control 9.7 % → 8.3 %).
- **Acquisition not identical across plates**: P26 negative-peak = 142 vs P23 = 168
  raw units (~15 % dimmer). So each plate must use its **own** threshold; do NOT
  compare absolute % across plates without accounting for this. Folds are comparable.
- Same caveat as P23: the per-cell distribution is **unimodal, no valley**
  (`percell_desmin.png`) — absolute levels have real uncertainty; folds are stable.

## Files
- `visualize_final.py` → `conversion_summary_bar.png`, 3× `*_percell_classified.png`
  (magenta = Desmin+, cyan = negative, green = Desmin), `visualize_final.json`
- `percell_desmin.py` → `percell_desmin.png` (distribution + no-valley check), `.json`
- `build_dbs.py` → `dbs_cache/` (background-subtracted Desmin, raw units)
- `percell_values_r10.0.npz` — cached per-cell ring intensities
- nuclei masks + total-nuclei counts: `../plate26_nuclei/`
