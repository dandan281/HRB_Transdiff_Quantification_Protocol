# Conversion Efficiency — Myotube Fusion Index Pipeline

**Goal.** From a two-channel confocal `.nd2` field (Desmin = myotubes, DAPI = nuclei),
count how many nuclei sit **inside real myotubes** and report a **fusion index**
(= nuclei inside myotubes / total nuclei). Built and validated on **Plate 23** (6 wells).

> **Scope rule for this project:** all code + outputs live in `Conversion_Efficiency/`.
> Reading source images from `../Q_PLATES/` is fine; nothing is written outside this folder.

---

## 0. Data

- Source: `../Q_PLATES/Q_Plates/PLATE_23/*.nd2` (6 wells).
- Channel order (confirmed): `ch0` = 561, `ch1` = 488 **Desmin (myotube)**, `ch2` = 405 **DAPI (nuclei)**.
- Pixel size: **0.6493 µm/px**. Field ≈ 3600² px ≈ 2337 µm across.
- Wells: `19_B06_act104_trka`, `22_B03_act104_egfrc`, `23_B02_ctrl`, `29_C05_br223_egfrc`,
  `32_C08_br223_igf1r`, `33_C09_br223_trka`.

**Why hybrid (two different detectors):** nuclei are blobby → a learned instance model
(Cellpose-SAM) excels; myotubes are thin elongated fibers → Cellpose fails on them, so a
ridge/tubeness detector is used instead. Each tool is applied to its strength.

---

## 1. Nuclei — Cellpose-SAM + cellprob plateau sweep

**Problem with the first approach.** An intensity-peak **watershed over-counted ~1.8×**
(chromatin texture splits one nucleus into 2–3). Head-to-head on an 800² crop: watershed 982
vs Cellpose 540; full field 13,329 vs 10,588. → **adopt Cellpose-SAM** (`cpsam` model), which
gives one mask per nucleus.

**No brightness/contrast/threshold knob.** Cellpose normalizes each tile internally
(1st–99th pct), so the lab's manual "adjust B&C + threshold until the count stabilizes" has no
direct equivalent — **except `cellprob_threshold`** (raise = stricter = fewer). So we do the
lab's *plateau* trick on that knob:

- **Sweep** `cellprob_threshold ∈ {-2,-1,0,+1,+2}`, count nuclei at each.
- **Pick the plateau** = interior point where neighbors change least (most stable), tie-break
  toward 0. This is the Cellpose-native version of "the setting where the count barely moves."
- Result: **every well plateaued at cp = 0** (count peaks there, falls off symmetrically).

**Script:** `count_well.py` (runs in `cpenv`, GPU). Per well → sweep plot, labeled overlay,
`{well}_masks.npy`, appends to `plate23_nuclei/plate_results.jsonl`.
**QC:** `zoom_crop.py` (raw | colored masks | red outlines, high-res crop), `compare_viz.py`
(watershed vs Cellpose).

**Result — total nuclei, Plate 23 = 54,229**
| well | nuclei | | well | nuclei |
|---|--:|---|---|--:|
| 19_B06_act104_trka | 8,949 | | 29_C05_br223_egfrc | 7,521 |
| 22_B03_act104_egfrc | 10,051 | | 32_C08_br223_igf1r | 10,588 |
| 23_B02_ctrl | 7,905 | | 33_C09_br223_trka | 9,215 |

**Area boundary (`nuclei_filter.py`).** Keep only nuclei with **50 ≤ area ≤ 500 µm²**
(1 px = 0.4216 µm²). Removes sub-nuclear debris/bleed-through (<50) and merged doublets/giant
artefacts (>500). The area histogram is a clean unimodal peak at ~150 µm² sitting entirely inside
the band, with a small separate debris bump <50 µm². Impact: removes **2,359 / 54,229 (4.4 %)**
(~2,319 small + ~40 large) → **51,870 valid nuclei**. This boundary is applied as the denominator
and to the inside-counts in `real_fusion.py` (`--amin-um2 50 --amax-um2 500`).

---

## 2. Myotubes — ridge detector + threshold plateau sweep

**Why not Cellpose / global threshold.** Cellpose is blobby-object shaped and misses thin
fibers. A single global threshold (Otsu/Triangle) under-counts dim/thin fibers. → a **ridge
(tubeness) pipeline** mirroring the lab's Fiji recipe (Subtract Background → CLAHE → Ridge
Detection), fully automated in scikit-image.

**`detect_myotubes()` in `myotube_detect.py`:**
`white_tophat` (remove haze) → `CLAHE` (local contrast so dim fibers appear) → `Sato` tubeness
(enhances elongated ridges at several widths) → **hysteresis threshold** (keeps dim tendrils
connected to bright ridges) **AND an intensity gate** (mask must also exceed a bg-subtracted
intensity pct, so it hugs the fiber body instead of ballooning into the dim halo — this fixed
an earlier 3–4× over-dilation seen on sparse wells).

**Threshold plateau sweep (`myotube_sweep.py`).** Same robustness idea as the nuclei:
the expensive preprocessing (top-hat + Sato) is **computed once**, then the cheap hysteresis
threshold is swept and distinct-fiber count recorded; the plateau (most stable count) is the
operating point.
- **Key finding:** `detect_myotubes` DEFAULT `high_pct = 94` is already on the **collapse** of
  the curve (fibers being destroyed). The **plateau is ~89–92** (more coverage). → the first
  fusion index (which used the default) under-measured myotube territory. Use the plateau masks
  (`plate23_myotube/{well}_myotube_mask.npy`) downstream.

---

## 3. Myotube LENGTH distribution — trace fibers through crossings

**Why needed.** Raw "number of myotubes" (connected components) mixes real fibers with tiny
Desmin **fragments**. A length distribution separates them (fragments pile at small sizes,
fibers form the tail).

**Three definitions were tried (this matters):**
1. **Connected-component skeleton length** → *wrong*: touching fibers fuse into one mesh whose
   skeleton sums to **physically impossible 15,000–30,000 µm** "fibers".
2. **Split at every junction** → *wrong the other way*: shatters each crossing into 5 µm twigs;
   N explodes to 5–14k, median collapses to ~5 µm.
3. **Trace through crossings** (chosen) → at each junction, pair the two branch-ends whose local
   directions are most **anti-parallel** (dot < −0.5 = a fiber passing straight through), walk
   the resulting chains, sum branch lengths = one length per whole fiber. Terminal spurs
   < 10 µm pruned. Uses `skan` (geodesic branch lengths) + `networkx`.

**Script:** `myotube_length_dist.py` → 3-panel figure (linear | log-y | CCDF) +
`myotube_length_stats.json`.

**Result — mature fibers (>100 µm) per well:** control **B02 = 83** (median 10 µm) vs treated
**~276–343** (median ~14–34 µm). The control is short-fiber/fragment dominated.

---

## 4. Fusion index — nuclei inside myotubes

**Overlap rule.** A nucleus is **inside** if **≥ 50 %** of its pixels fall in the myotube
territory. Desmin is cytoplasmic and leaves a **void where the nucleus sits**, so the myotube
mask is **hole-filled** first (`binary_fill_holes`) — otherwise an enclosed nucleus would read
as "outside."

**First pass (`fusion_index.py`)** used the *default-threshold* masks and *no* fragment gating.
It produced a **paradox: the control B02 had the HIGHEST fusion index (15.2%)** — because nuclei
near short Desmin **fragments** were counted as "inside myotubes."

**Real-myotube fusion (`real_fusion.py`) — the correct version.**
Fragments must be removed at the **traced-fiber level**, NOT the connected-object level
(a per-object length gate fails: B02's short fragments are wired into big **meshes** that survive
it — verified, B02 stayed highest). So:
1. trace whole fibers through crossings (§3);
2. for a length **gate**, keep only skeleton pixels of fibers ≥ gate;
3. **rebuild fiber bodies**: assign every Desmin pixel to its nearest skeleton pixel
   (`distance_transform_edt` indices), keep those whose nearest fiber is real; hole-fill →
   "real myotube" territory;
4. count nuclei inside (≥50%).

**Result — Plate 23 nuclei inside real myotubes (of 51,870 area-filtered nuclei):**
| fibre-length gate | nuclei inside | fusion index |
|---|--:|--:|
| ≥ 0 µm (all traced, spurs pruned) | 5,065 | 9.8 % |
| ≥ 30 µm | 3,517 | 6.8 % |
| **≥ 50 µm (real myotubes)** | **2,909** | **5.6 %** |
| ≥ 100 µm (mature) | 1,986 | 3.8 % |

(Area boundary 50–500 µm² barely shifts the *percentage* — it removes debris from both numerator
and denominator — confirming the ratio is robust; it mainly cleans the absolute counts.)

**Paradox resolved:** at gate ≥0 the control looks highest (fragments), but once real fibers are
required it drops to the **bottom** (B02 = 2.1 % at ≥50 µm, 1.0 % at ≥100 µm) while act104/igf1r
wells lead. The pipeline now tracks biology, not artifact.

### 4a. Joint distribution — Desmin+ nuclei (`nuclei_3d.py`)

For each nucleus inside a myotube, record (nuclear area µm², **host** myotube length µm). Host fibre
is found by propagating each traced fibre's ID to its territory (nearest-skeleton) and taking the
dominant ID under the nucleus. 3D surface: x = nuclear area, y = myotube length, z = nucleus count
(+ 2D heatmap). Finding: nuclear area is ~constant (~137 µm², independent of fibre length) while the
**host-myotube length** separates control (median 32 µm) from treated (96–137 µm). N = 5,027 Desmin+
nuclei. Raw pairs saved to `plate23_real_fusion/nuclei_3d_data.npz`.

---

## 5. How to run (all from `Conversion_Efficiency/`, `cpenv` python)

```
# 1. nuclei per well (GPU) -> plate23_nuclei/
cpenv/Scripts/python.exe count_well.py --nd2 <file.nd2> --outdir plate23_nuclei
# 2. myotube threshold plateau + masks -> plate23_myotube/
cpenv/Scripts/python.exe myotube_sweep.py --nd2 <file.nd2> --outdir plate23_myotube
# 3. myotube length distribution
cpenv/Scripts/python.exe myotube_length_dist.py
# 4. nuclei inside REAL myotubes (gated) -> plate23_real_fusion/
cpenv/Scripts/python.exe real_fusion.py --gates 0,30,50,100 --primary 50
# summaries: summary_plate.py, summary_fusion.py, summary_real_fusion.py
```

**Env `cpenv/` (py3.13):** cellpose 4.2.1.1, torch/torchvision **cu128** (RTX 5070 Ti = Blackwell,
install torch from the cu128 index LAST or `pip install cellpose` clobbers it with CPU torch),
nd2, scikit-image, scipy, matplotlib, **skan + networkx** (installing skan downgraded numpy
2.5.1 → 2.4.6; if cellpose/torch later break, reinstall numpy). Model `cpsam_v2` in
`~/.cellpose/models/` (fetch with `curl -L -C - --retry 8` — the built-in downloader hangs).

---

## 6. Script / output map

| script | role |
|---|---|
| `count_well.py` | nuclei: cellprob sweep + plateau, per well (GPU) |
| `nuclei_filter.py` | nucleus area boundary [50,500] um^2 report + histogram |
| `cellpose_nuclei.py` | single-image Cellpose wrapper / QC |
| `compare_viz.py`, `zoom_crop.py` | nuclei QC overlays |
| `myotube_detect.py` | ridge/tubeness Desmin detector (`detect_myotubes`) |
| `myotube_sweep.py` | myotube threshold plateau sweep + masks |
| `myotube_length_dist.py` | whole-fiber length distribution (traced) |
| `fusion_index.py` | first fusion pass (default masks, no gating) — superseded |
| `real_fusion.py` | **nuclei inside REAL (fiber-length-gated) myotubes** + nucleus area boundary |
| `nuclei_3d.py` | 3D joint dist of Desmin+ nuclei: area x host-myotube length x count |
| `summary_plate.py` / `summary_fusion.py` / `summary_real_fusion.py` | plate montages/bars |
| `conversion_efficiency.py` | original watershed pipeline — nuclei part superseded by Cellpose |

| output dir | contents |
|---|---|
| `plate23_nuclei/` | nuclei masks, sweep plots, labeled overlays, `plate_results.jsonl` |
| `plate23_myotube/` | myotube masks, threshold sweeps, `myotube_length_distribution.png`, stats |
| `plate23_fusion/` | first (superseded) fusion overlays + results |
| `plate23_real_fusion/` | real-myotube fusion overlays, `real_fusion_results.json`, summary |

---

## 7. Open items / next steps

- **Pick the myotube length gate** that defines a "real myotube" for reporting (currently 50 µm
  primary; 30/100 also computed). Biological convention (≥2–3 nuclei, or length) can set this.
- **Validate** against the Q_Plates hand-labels (do NOT use the lab result CSVs — flagged wrong).
- **Nuclei artifact filters:** area boundary 50–500 µm² is DONE (`nuclei_filter.py`). Optional
  further filters (shape/solidity, mean-DAPI intensity, border-touching) remain if needed.
- Generalize the per-well drivers to any plate (currently Plate-23 paths hard-coded in batch
  shell loops `run_fusion.sh` / `run_myotube.sh`).
