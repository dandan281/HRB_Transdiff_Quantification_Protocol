# New_Quantif_P23 — conversion efficiency, PLATE_23

Nuclei inside Desmin+ myotubes (conversion efficiency) for the 6 wells of
`../../Q_PLATES/Q_Plates/PLATE_23/*.nd2`. Self-contained package: scripts, cached
intermediates, result JSONs, and figures. All scripts are run **from
`Conversion_Efficiency/`** (their parent), e.g.
`cpenv/Scripts/python.exe New_Quantif_P23/visualize_final.py`.

Channel order: `ch0`=561, `ch1`=488 **Desmin**, `ch2`=405 **DAPI**. Pixel = 0.6493 µm.

---

## Operating point (fixed, plate-wide, data-driven)

**One parameter set applied identically to every well** — no per-well tuning, no
calibration to any expected value. Within a plate the artifacts/irreducible error
are assumed constant, so a single shared operating point is valid and every
between-well difference is biology.

| stage | parameter | value |
|---|---|---|
| nuclei | Cellpose-SAM `cellprob_threshold` (plateau) | 0 |
| nuclei | area boundary | 50–500 µm² |
| Desmin | background removal | white top-hat, disk r=40, **raw camera units** |
| Desmin | per-cell readout | mean bg-subtracted Desmin in a **10 µm cytoplasmic ring** |
| Desmin | positivity threshold | **440.8 raw units** = Otsu on the **pooled** per-cell distribution |

---

## TRUE result (this is the headline)

| well | condition | valid nuclei | Desmin+ nuclei | conversion | fold |
|---|---|--:|--:|--:|--:|
| 23_B02_ctrl | control | 7,635 | 1,166 | **15.3 %** | 1.00× |
| 33_C09_br223_trka | br223/trka | 8,947 | 1,616 | 18.1 % | 1.18× |
| 29_C05_br223_egfrc | br223/egfrc | 7,210 | 1,750 | 24.3 % | 1.59× |
| 19_B06_act104_trka | act104/trka | 8,440 | 2,748 | 32.6 % | 2.13× |
| 32_C08_br223_igf1r | br223/igf1r | 10,114 | 3,341 | 33.0 % | 2.16× |
| 22_B03_act104_egfrc | act104/egfrc | 9,524 | 3,749 | 39.4 % | 2.58× |

- **B03, C08, B06 convert ~2.1–2.6× above control** — robust across 4 methods and
  every hyperparameter sweep (see `robustness_sweeps.png`).
- **C05, C09 are weak** (1.6×, 1.2×); by distribution-level measures C09 is
  essentially at control. Reported as-is.
- Data-driven control = **15.3 %**. The lab reference (~20 %) was used only to
  sanity-check the range — it did **not** set any threshold.

### Caveats (not fixable by parameter choice)
1. **Absolute levels have real uncertainty**: the per-cell Desmin distribution is
   unimodal with no valley (`percell_desmin.png`), so the threshold sits on a
   shoulder. Fold-changes are far more stable than absolute %.
2. **A Desmin-negative control well** (secondary-only / unconverted fibroblasts)
   would turn the threshold into a measurement rather than a data-driven guess —
   the highest-value next experiment.
3. Single 2-D field per well; no membrane marker.

---

## The bug this package fixed

The production detector (`../myotube_detect.py`) normalised **per image** (percentile
rescale → CLAHE → percentile hysteresis → percentile intensity gate). The gate took
the brightest ~10 % of *every* image, pinning Desmin coverage to ~11 % in all wells
and flattening fold-change to 1.0–1.1×. Measured: the gate was ~3.9σ in the control
but ~12σ in B03. Fix = absolute raw-unit thresholds with a shared noise scale. See
`diagnosis_the_bug.png`.

---

## File guide

**Final result**
- `visualize_final.py` → `conversion_summary_bar.png`, 6× `*_percell_classified.png`
  (spatial QC: magenta = Desmin+, cyan = negative), `visualize_final.json`
- `percell_desmin.py` → `percell_desmin.png` (the no-valley distribution), `.json`

**Diagnostics / robustness**
- `viz_diagnostics.py` → `diagnosis_the_bug.png`, `robustness_sweeps.png`
- `absolute_desmin.py`/`.json` — per-image-vs-absolute coverage (the diagnosis)
- `conversion_v2.py`/`.json` — absolute-threshold k-sweep (pixel-overlap)
- `ring_sweep.py`/`.json` — ring-size saturation
- `amin_sweep.py`/`amin_sweep_g30.json` — nucleus area-cut sweep

**Cached intermediates** (delete to force recompute; regenerated on next run)
- `dbs_cache/` — background-subtracted Desmin per well (raw units, uint16)
- `percell_values_r*.npz` — per-cell ring intensities at each ring size
- `nuclei/`, `myotube/` — B02 Cellpose + ridge masks/overlays

**Multinucleation — nuclei per INDIVIDUAL myotube** (gate ≥50 µm, 50% overlap)
- `plate23_multinucleation.py` → `plate23_multinucleation_summary.png` (per-well +
  pooled composition + maturity), 6× `{well}_multinuc_overlay.png` (each myotube
  coloured by nucleus count), `plate23_multinucleation.json`
- `b02_multinucleation.py` → B02-only version with the full gate × overlap sensitivity
  (`b02_multinucleation_hist.png`, `_overlay.png`, `.json`)
- Individual myotubes = traced fibres through crossings (from `real_fusion`); a
  nucleus is assigned to its dominant host myotube if ≥50% of it overlaps that
  myotube's territory. Distribution reported over NUCLEATED myotubes (≥1 nucleus).

**Superseded — kept for provenance, DO NOT use for reporting**
- `b02_new_quantif.py`, `b02_results.json`, `23_B02_ctrl_fusion_g50_ov25.png` —
  fusion-index method (required nuclei inside long traced fibres; discards
  mononuclear converted cells → too low, B02=391).
- `percell_calibrate.py`, `percell_calibrate.json` — control-**anchored** threshold
  (made the control an input, not a measurement). Replaced by pooled-Otsu.
