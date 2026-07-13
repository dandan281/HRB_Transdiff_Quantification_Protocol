# Conversion Efficiency — nuclei inside myotubes

Counts how many DAPI nuclei sit **inside** Desmin+ myotubes, from the two
fluorescence channels of a single confocal `.nd2` field (already pixel-registered).

## Run
```bash
python conversion_efficiency.py \
    --nd2 "../Q_PLATES/Q_Plates/PLATE_23/32_C08_br223_igf1r.nd2" \
    --nuclei-ch 2 --myotube-ch 1 --outdir outputs
```
Channel map for these nd2 files: `0`=561 (receptor), `1`=488 Desmin (myotube),
`2`=405 DAPI (nuclei).

## What it computes

**Matrices.**
- `M1` nuclei (DAPI): Gaussian smooth + Li threshold — keeps dim nuclei so they
  can be counted.
- `M2` myotubes (Desmin): **NOT a global threshold** — thin/dim fibers need a
  ridge detector. `myotube_detect.py` replicates the lab's validated Fiji recipe
  (Subtract Background → CLAHE local contrast → Ridge/tubeness filter →
  hysteresis threshold). This recovers the faint tendrils a global threshold
  drops: Otsu captured ~2.85% of the field, Triangle ~6.2%, but the true fiber
  network (matching the raw Desmin image) is **~19%**. Verified against the raw
  channel with the overlays in `outputs/tune/`.

**Two readouts:**

1. **Pixel classes.** Instead of `M1 - M2` (which collides: `1-1=0` *and* `0-0=0`),
   use the bijective code `2*M1 + M2`:

   | code | (M1,M2) | meaning |
   |---|---|---|
   | 0 | (0,0) | background |
   | 1 | (0,1) | myotube only  (= your −1) |
   | 2 | (1,0) | nucleus only  (= your +1) |
   | 3 | (1,1) | **overlap: nucleus pixel inside myotube** |

   Background and overlap are reported separately (both would be `0` under subtraction).

2. **Fusion index (headline).** Individual nuclei are segmented (intensity-peak
   seeding + watershed, robust to touching nuclei in dense fields). A nucleus is
   "inside" if ≥50% of its area overlaps the myotube mask.
   `fusion_index = nuclei_inside / nuclei_total`.

## Outputs (`outputs/`)
- `labeled_nuclei_full.png` — **full-field map: every nucleus marked**, cyan =
  outside myotube, red = inside; myotube mask in green, with legend + scale bar
- `results.json` — all counts and the fusion index
- `matrix1_nuclei.npy`, `matrix2_myotube.npy` — downsampled binary matrices
- `outcome_matrix.npy` — the literal M1−M2 map; `class_code_map.npy` — the 2·M1+M2 map
- `overlay_masks.png` — magenta nuclei / green myotube / white overlap
- `qc/nuclei_segmentation_crop.png` — zoomed per-nucleus check (cyan=outside, red=inside)

## Results (PLATE_23)
| well | myotube area | nuclei total | inside | fusion index |
|---|---|---|---|---|
| 32_C08_br223_igf1r | 11.4% | 11,268 | 1,326 | 11.8% |
| 23_B02_ctrl        |  8.6% | 10,693 | 1,789 | 16.7% |

## Detector history (why the myotube mask matters most)
The fusion index is dominated by how faithfully the myotube mask matches the real
fibers. Two failure modes were found and fixed:
1. **Under-segmentation** (global Otsu/Triangle): captured only bright cores,
   dropped thin/dim fibers → fusion index far too low.
2. **Over-dilation** (tubeness + hysteresis, no intensity gate): mask ballooned
   3–4× past the true fiber width into the dim halo → fusion index too high.
   Caught by testing on the sparse **B02 control**, where fat blobs were obvious.

Current detector = ridge/tubeness detection (finds dim fibers) **gated by
background-subtracted intensity** (width tracks real signal). Tuning overlays that
justify the parameters are in `outputs/tune/` and `outputs/tune2/`.

## Caveats
- Detector knobs are at the top of `myotube_detect.py` (`low_pct`, `high_pct`,
  `gate_pct`, `sigmas`); retune against the `tune*/` overlays if staining changes.
- Absolute nucleus counts depend on the DAPI threshold — treat as estimates.
- **Do not over-read the C08-vs-B02 fusion difference.** These are single 2-D
  fields; "ctrl" here is a control *perturbation* in a differentiation screen (both
  wells are differentiated), not an undifferentiated control. A real comparison
  needs several fields per well and biological replicates.
- No membrane marker, so "inside" is a projection-overlap call, not true 3-D.
