# Benchmark conversion efficiency — 25-image set

- Input: `Benchmark/1..25.tif`, 8-bit RGB ImageJ exports, 1040x1392; green = Desmin, blue = DAPI, red empty.
- Pixel size **measured from the burned-in scale bar**: 50 um = 96 px -> **0.521 um/px**.

## Pipeline (one shared, data-driven parameter set for all images)
- Nuclei: Cellpose-SAM, cellprob plateau **cp=+0** (pooled sweep 6187-6317 over cp -2..+2 — flat).
- Nucleus area filter: [50, 500] um^2 from the pooled histogram (median 208 um^2; keeps 5,856/6,317 = 92.7%).
- Myotube territory: background-surface subtraction (8x8 masked-tile median), shared sigma = 6.27, hysteresis threshold 15.68 -> 25.09 8-bit units (plateaus k=4, k_low=2.5); min object 180 px; hole-filled.

## Results (pooled over 25 images)

| readout | >=25% overlap | >=50% overlap |
|---|---|---|
| **All Desmin+ territory (primary)** | **1732 / 5856 = 29.6%** | **1295 / 5856 = 22.1%** |
| Traced fibres >=50 um only | 745 = 12.7% | 464 = 7.9% |
| Area band [90,500] um^2 (sensitivity) | 1625 / 5482 = 29.6% | 1209 = 22.1% |

- Density confound (ov25 vs nuclei/image): Pearson r=-0.65 (p=0.000), Spearman rho=-0.70.
- Per-image range: 7.9-55.8% (ov25); densest image 459 nuclei, sparsest 88.

## Files
- `nuclei/` masks + sweep; `myotube2/` masks + labeled overlays;
- `fusion/` per-image 3-class overlays + fusion_results.json;
- `summary/` figures + this file.
