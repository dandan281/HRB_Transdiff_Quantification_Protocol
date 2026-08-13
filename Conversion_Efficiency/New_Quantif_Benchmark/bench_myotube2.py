"""Benchmark myotube (Desmin) territory, v2 — built for the BROAD, filled
ribbons in this set, which the thin-fibre ridge detector traces only at the
rims (see myotube/1_myotube_labeled.png for the failure).

Method (uniform across all 25 images, all thresholds pooled/absolute):
  1. Background SURFACE per image: the field is center-bright (tile-p15
     corners ~25-39 vs center ~42-54), so a scalar offset is not enough.
     Estimate a coarse 8x8 tile background from NON-foreground pixels only
     (foreground = > mode + 2*sigma, dilated) — ribbons are excluded, so broad
     structures are NOT subtracted (the flaw of the 40px top-hat). Fill empty
     tiles from neighbours, smooth, bilinear-upsample, subtract, clip >= 0.
  2. Noise sigma per image from the residual background (1.4826*MAD), then one
     SHARED sigma = median over the 25 images (the desmin-bug fix pattern).
  3. Threshold: sub > k * sigma_shared, identical everywhere. k is swept
     (2,3,4,5,6,8) and chosen at the pooled PLATEAU of the object count.
  4. HYSTERESIS extension: the dimmest broad-ribbon interiors sit below the
     plateau cut (visible in the first k=4 overlay), so the final mask is
     apply_hysteresis_threshold(sub, k_low*sigma, k_plateau*sigma): dim pixels
     join only when connected to a confident seed. k_low is swept the same
     pooled-plateau way.
  5. remove_small_objects/holes at 180 px, scale-bar box zeroed.

Run:  cpenv/Scripts/python.exe bench_myotube2.py
"""
from __future__ import annotations
import os, json, glob, time
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, zoom, binary_dilation as nd_dilate
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image

BENCH = r"C:\Users\liqig\Documents\HRB_Transdiff\Benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "myotube2")
GRID = (8, 8)
KS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
K_LOWS = [1.0, 1.5, 2.0, 2.5, 3.0]
MIN_OBJ = 180
SCALEBAR = (slice(1013, 1040), slice(1280, 1392))   # burned-in ruler box


def image_files():
    return sorted(glob.glob(os.path.join(BENCH, "*.tif")),
                  key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))


def robust_sigma(vals):
    med = np.median(vals)
    return 1.4826 * np.median(np.abs(vals - med))


def bg_surface(g):
    """Foreground-masked tile background, upsampled to full frame."""
    hist = np.bincount(g.astype(np.uint8).ravel(), minlength=256)
    mode = int(np.argmax(hist))
    resid = g[g <= mode + 20] - mode          # sample around the bg peak
    s0 = max(robust_sigma(resid), 1.0)
    fg = g > mode + 2 * s0
    fg = nd_dilate(fg, iterations=5)
    fg[SCALEBAR] = True                        # ruler never counts as bg
    H, W = g.shape
    th, tw = H // GRID[0], W // GRID[1]
    tiles = np.full(GRID, np.nan)
    for i in range(GRID[0]):
        for j in range(GRID[1]):
            sl = (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
            bg = g[sl][~fg[sl]]
            if bg.size > 0.15 * th * tw:
                tiles[i, j] = np.median(bg)
    if np.isnan(tiles).any():                  # fill dense tiles from neighbours
        m = np.nanmedian(tiles)
        for _ in range(3):
            nan = np.isnan(tiles)
            if not nan.any():
                break
            pad = np.pad(tiles, 1, mode="edge")
            neigh = np.nanmean(np.stack([pad[:-2, 1:-1], pad[2:, 1:-1],
                                         pad[1:-1, :-2], pad[1:-1, 2:]]), axis=0)
            tiles[nan] = neigh[nan]
        tiles[np.isnan(tiles)] = m
    tiles = gaussian_filter(tiles, 0.8)
    B = zoom(tiles, (H / GRID[0], W / GRID[1]), order=1)
    return B, s0


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    stems, subs, sigmas = [], {}, {}
    for p in image_files():
        stem = os.path.splitext(os.path.basename(p))[0]
        stems.append(stem)
        g = tifffile.imread(p)[..., 1].astype(np.float32)
        B, s0 = bg_surface(g)
        sub = np.clip(g - B, 0, None)
        sub[SCALEBAR] = 0
        # sigma of the flattened background (residual noise after subtraction)
        flat_bg = (g - B)[(g - B) < 2 * s0]
        sigmas[stem] = robust_sigma(flat_bg)
        subs[stem] = sub
    sig_shared = float(np.median(list(sigmas.values())))
    print(f"per-image sigma: min={min(sigmas.values()):.2f} "
          f"median={sig_shared:.2f} max={max(sigmas.values()):.2f}", flush=True)

    counts = {k: 0 for k in KS}
    covs = {k: {} for k in KS}
    for stem in stems:
        sub = subs[stem]
        for k in KS:
            m = sub > k * sig_shared
            m = remove_small_objects(m, min_size=MIN_OBJ)
            m = remove_small_holes(m, area_threshold=MIN_OBJ)
            counts[k] += int(label(m).max())
            covs[k][stem] = round(100 * m.mean(), 2)
    tot = [counts[k] for k in KS]
    print("pooled object totals: "
          + "  ".join(f"k{k:g}={t}" for k, t in zip(KS, tot)))
    print("mean coverage %: "
          + "  ".join(f"k{k:g}={np.mean(list(covs[k].values())):.1f}" for k in KS),
          flush=True)

    best_i, best_key = None, None
    for i in range(1, len(KS) - 1):
        key = (abs(tot[i - 1] - tot[i + 1]), KS[i])   # flattest; tie -> lower k
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    op_k = KS[best_i]
    print(f"-> global plateau k = {op_k:g}  "
          f"(threshold = {op_k * sig_shared:.1f} 8-bit units above local bg)",
          flush=True)

    # hysteresis low-threshold sweep (high fixed at the plateau k)
    lcounts = {kl: 0 for kl in K_LOWS}
    lcovs = {kl: [] for kl in K_LOWS}
    for stem in stems:
        sub = subs[stem]
        for kl in K_LOWS:
            m = apply_hysteresis_threshold(sub, kl * sig_shared,
                                           op_k * sig_shared)
            m = remove_small_objects(m, min_size=MIN_OBJ)
            m = remove_small_holes(m, area_threshold=MIN_OBJ)
            lcounts[kl] += int(label(m).max())
            lcovs[kl].append(100 * m.mean())
    ltot = [lcounts[kl] for kl in K_LOWS]
    print("hysteresis low-k totals: "
          + "  ".join(f"kl{kl:g}={t}" for kl, t in zip(K_LOWS, ltot)))
    print("hysteresis mean coverage %: "
          + "  ".join(f"kl{kl:g}={np.mean(lcovs[kl]):.1f}" for kl in K_LOWS),
          flush=True)
    best_i, best_key = None, None
    for i in range(1, len(K_LOWS) - 1):
        key = (abs(ltot[i - 1] - ltot[i + 1]), -K_LOWS[i])  # flattest; tie -> lower
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    op_kl = K_LOWS[best_i]
    print(f"-> plateau k_low = {op_kl:g}  (hysteresis "
          f"{op_kl * sig_shared:.1f} -> {op_k * sig_shared:.1f} units)", flush=True)

    per_image = {}
    for p in image_files():
        stem = os.path.splitext(os.path.basename(p))[0]
        sub = subs[stem]
        m = apply_hysteresis_threshold(sub, op_kl * sig_shared,
                                       op_k * sig_shared)
        m = remove_small_objects(m, min_size=MIN_OBJ)
        m = remove_small_holes(m, area_threshold=MIN_OBJ)
        np.save(os.path.join(OUT, f"{stem}_myotube_mask.npy"), m)
        per_image[stem] = {"objects": int(label(m).max()),
                           "cov_pct": round(100 * m.mean(), 2),
                           "sigma": round(float(sigmas[stem]), 2)}
        g = tifffile.imread(p)[..., 1].astype(np.float32) / 255.0
        over = label2rgb(label(m), image=np.clip(g / max(g.max(), 1e-6), 0, 1),
                         bg_label=0, alpha=0.45, image_alpha=1, bg_color=(0, 0, 0))
        Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"{stem}_myotube_labeled.png"))

    with open(os.path.join(OUT, "myotube2_results.json"), "w") as fh:
        json.dump({"ks": KS, "operating_k": op_k,
                   "k_lows": K_LOWS, "operating_k_low": op_kl,
                   "sigma_shared": sig_shared,
                   "threshold_abs_high": round(op_k * sig_shared, 2),
                   "threshold_abs_low": round(op_kl * sig_shared, 2),
                   "pooled_totals": {f"{k:g}": t for k, t in zip(KS, tot)},
                   "hyst_totals": {f"{kl:g}": t for kl, t in zip(K_LOWS, ltot)},
                   "coverage_by_k": {f"{k:g}": covs[k] for k in KS},
                   "per_image": per_image,
                   "seconds": round(time.time() - t0, 1)}, fh, indent=2)
    print(f"MYOTUBE2_DONE  op_k={op_k:g}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
