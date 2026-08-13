"""Benchmark myotube (Desmin) detection on the GREEN channel of the 25 RGB
benchmark TIFFs — the lab's validated ridge recipe (white top-hat -> CLAHE ->
Sato tubeness -> hysteresis, intensity-gated), with one critical change:

    ALL thresholds are ABSOLUTE values computed from the POOLED distribution
    across the 25 images, applied identically to every image.

Per-image percentile thresholds are the known normalization bug (they pin
coverage per image and destroy between-image differences). The green channel
is consistent across this set (bg peak 39-61, no saturation), so pooled
absolute thresholds are valid; the top-hat removes the residual bg offset.

Two passes:
  A. per image: d_bs = white_tophat(green/255, disk(40));
     clahe = equalize_adapthist(d_bs); tub = sato(clahe, sigmas=(1,2,4)).
     tub/d_bs cached as float32 npy (expensive part, computed once).
  B. pool subsampled nonzero tub + d_bs values -> shared gate threshold
     (pooled p90 of d_bs>0) and shared hysteresis thresholds for each swept
     high_pct (pooled tub percentiles). Sweep high_pct, count distinct fibres
     summed over the set, pick the global PLATEAU, save masks + labeled viz.

Run:  cpenv/Scripts/python.exe bench_myotube.py
"""
from __future__ import annotations
import os, json, glob, time
import numpy as np
import tifffile
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import (white_tophat, disk, binary_dilation,
                                remove_small_objects, remove_small_holes)
from skimage.exposure import equalize_adapthist
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image

BENCH = r"C:\Users\liqig\Documents\HRB_Transdiff\Benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "myotube")
WORK = os.path.join(HERE, "myotube", "work")
TOPHAT_R = 40
CLAHE_CLIP = 0.01
SIGMAS = (1, 2, 4)
HIGHS = [86.0, 89.0, 92.0, 94.0, 96.0]
BAND = 14            # low_pct = high_pct - BAND (plate-pipeline convention)
GATE_PCT = 90        # pooled percentile of nonzero d_bs
MIN_OBJ = 180        # px


def image_files():
    return sorted(glob.glob(os.path.join(BENCH, "*.tif")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()

    # ---- pass A: expensive filters, once per image ----
    stems = []
    for p in image_files():
        stem = os.path.splitext(os.path.basename(p))[0]
        stems.append(stem)
        f_tub = os.path.join(WORK, f"{stem}_tub.npy")
        f_dbs = os.path.join(WORK, f"{stem}_dbs.npy")
        if os.path.exists(f_tub) and os.path.exists(f_dbs):
            continue
        g = tifffile.imread(p)[..., 1].astype(np.float32) / 255.0   # green
        d_bs = white_tophat(g, disk(TOPHAT_R))
        clahe = equalize_adapthist(d_bs, kernel_size=g.shape[0] // 16,
                                   clip_limit=CLAHE_CLIP)
        tub = sato(clahe, sigmas=SIGMAS, black_ridges=False)
        np.save(f_tub, tub.astype(np.float32))
        np.save(f_dbs, d_bs.astype(np.float32))
        print(f"preprocessed {stem} ({time.time()-t0:.0f}s)", flush=True)

    # ---- pool subsampled nonzero values across the whole set ----
    tub_pool, dbs_pool = [], []
    for stem in stems:
        tub = np.load(os.path.join(WORK, f"{stem}_tub.npy"))[::3, ::3]
        dbs = np.load(os.path.join(WORK, f"{stem}_dbs.npy"))[::3, ::3]
        tub_pool.append(tub[tub > 0])
        dbs_pool.append(dbs[dbs > 0])
    tub_pool = np.concatenate(tub_pool)
    dbs_pool = np.concatenate(dbs_pool)
    gate_thr = float(np.percentile(dbs_pool, GATE_PCT))
    hyst = {h: (float(np.percentile(tub_pool, h - BAND)),
                float(np.percentile(tub_pool, h))) for h in HIGHS}
    print(f"pooled gate thr (d_bs p{GATE_PCT}) = {gate_thr:.5f}", flush=True)
    for h, (lo, hi) in hyst.items():
        print(f"  high_pct {h:.0f}: hyst = ({lo:.5f}, {hi:.5f})")

    # ---- pass B: cheap shared-threshold sweep ----
    counts = {h: 0 for h in HIGHS}
    covs = {h: [] for h in HIGHS}
    per_image = {s: {} for s in stems}
    for stem in stems:
        tub = np.load(os.path.join(WORK, f"{stem}_tub.npy"))
        dbs = np.load(os.path.join(WORK, f"{stem}_dbs.npy"))
        gate = binary_dilation(dbs > gate_thr, disk(1))
        for h in HIGHS:
            lo, hi = hyst[h]
            m = apply_hysteresis_threshold(tub, lo, hi) & gate
            m = remove_small_objects(m, min_size=MIN_OBJ)
            m = remove_small_holes(m, area_threshold=MIN_OBJ)
            n = int(label(m).max())
            counts[h] += n
            covs[h].append(100 * m.mean())
            per_image[stem][f"{h:.0f}"] = {"fibres": n,
                                           "cov_pct": round(100 * m.mean(), 2)}
        print(f"swept {stem}: " + "  ".join(
            f"h{h:.0f}={per_image[stem][f'{h:.0f}']['fibres']}" for h in HIGHS),
            flush=True)

    tot = [counts[h] for h in HIGHS]
    best_i, best_key = None, None
    for i in range(1, len(HIGHS) - 1):
        key = (abs(tot[i - 1] - tot[i + 1]), -tot[i])   # flattest, tie->peakier
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    op_h = HIGHS[best_i]
    print("pooled fibre totals: "
          + "  ".join(f"h{h:.0f}={t}" for h, t in zip(HIGHS, tot)))
    print(f"-> global plateau high_pct = {op_h:.0f}", flush=True)

    # ---- save masks + labeled overlays at the operating point ----
    lo, hi = hyst[op_h]
    for p in image_files():
        stem = os.path.splitext(os.path.basename(p))[0]
        tub = np.load(os.path.join(WORK, f"{stem}_tub.npy"))
        dbs = np.load(os.path.join(WORK, f"{stem}_dbs.npy"))
        gate = binary_dilation(dbs > gate_thr, disk(1))
        m = apply_hysteresis_threshold(tub, lo, hi) & gate
        m = remove_small_objects(m, min_size=MIN_OBJ)
        m = remove_small_holes(m, area_threshold=MIN_OBJ)
        np.save(os.path.join(OUT, f"{stem}_myotube_mask.npy"), m)
        g = tifffile.imread(p)[..., 1].astype(np.float32) / 255.0
        lab = label(m)
        over = label2rgb(lab, image=np.clip(g / max(g.max(), 1e-6), 0, 1),
                         bg_label=0, alpha=0.55, image_alpha=1, bg_color=(0, 0, 0))
        Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"{stem}_myotube_labeled.png"))

    with open(os.path.join(OUT, "myotube_results.json"), "w") as fh:
        json.dump({"highs": HIGHS, "band": BAND, "gate_pct": GATE_PCT,
                   "gate_thr_abs": gate_thr,
                   "hyst_abs": {f"{h:.0f}": v for h, v in hyst.items()},
                   "operating_high_pct": op_h,
                   "pooled_totals": {f"{h:.0f}": t for h, t in zip(HIGHS, tot)},
                   "per_image": per_image,
                   "seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"MYOTUBE_DONE  op_high={op_h:.0f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
