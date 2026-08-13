"""
Conversion Efficiency: count nuclei located INSIDE myotubes.

Two channels of the SAME confocal field (already pixel-registered):
  - Matrix1 = DAPI  (nuclei)     -> binary nucleus mask
  - Matrix2 = Desmin (myotubes)  -> binary myotube mask

We produce two readouts:

1) Pixel matrix subtraction (exactly as specified by the user)
      outcome = Matrix1 - Matrix2
   per pixel:
      both 1  ( 1 - 1 = 0 )  -> OVERLAP   (nucleus pixel inside myotube)
      +1      ( 1 - 0 )      -> nucleus, no myotube
      -1      ( 0 - 1 )      -> myotube, no nucleus
      both 0  ( 0 - 0 = 0 )  -> BACKGROUND (reported separately from overlap)

2) Nuclei-count fusion index (headline biological metric)
      segment individual DAPI nuclei, then classify each nucleus as
      "inside a myotube" if the majority of its area sits on the Desmin mask.
      fusion_index = nuclei_inside / nuclei_total

Usage:
  python conversion_efficiency.py \
      --nd2 "../Q_PLATES/Q_Plates/PLATE_23/32_C08_br223_igf1r.nd2" \
      --nuclei-ch 2 --myotube-ch 1 --outdir outputs
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import nd2
from skimage.filters import (threshold_otsu, threshold_triangle,
                             threshold_li, gaussian)
from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
from skimage.segmentation import watershed
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from PIL import Image
from myotube_detect import detect_myotubes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _save_png(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr).save(path)


# ---------------------------------------------------------------- I/O

def load_channels(nd2_path: str, nuclei_ch: int, myotube_ch: int):
    with nd2.ND2File(nd2_path) as f:
        arr = f.asarray()          # (C, Y, X) uint16
        vox = f.voxel_size()       # microns per pixel (x, y, z)
    dapi = arr[nuclei_ch].astype(np.float32)
    desmin = arr[myotube_ch].astype(np.float32)
    um_per_px = float(vox.x)
    return dapi, desmin, um_per_px


# ---------------------------------------------------------------- masks

def binarize_fluor(img: np.ndarray, sigma: float, min_obj_px: int,
                   close_radius: int = 0, method: str = "otsu"):
    """Robust binary mask for a fluorescence channel over a large dark
    background. Threshold on the Gaussian-smoothed image, then size cleanup.

    method='otsu'     -> good for compact bright objects (nuclei)
    method='triangle' -> better for thin/dim structures over dark background
                         (recovers full myotube bodies that Otsu fragments)"""
    sm = gaussian(img, sigma=sigma, preserve_range=True)
    thr = threshold_triangle(sm) if method == "triangle" else threshold_otsu(sm)
    mask = sm > thr
    if close_radius > 0:
        mask = binary_closing(mask, disk(close_radius))
    mask = remove_small_objects(mask, min_size=min_obj_px)
    return mask, float(thr)


# ---------------------------------------------------------------- nuclei

def segment_nuclei(dapi_sm: np.ndarray, mask: np.ndarray, min_area_px: int,
                   min_distance_px: int):
    """Count/segment nuclei in a dense DAPI field.

    Seeds are INTENSITY peaks of the smoothed DAPI (one per nucleus center),
    which resolve touching nuclei that a distance-transform seed would merge
    into a single confluent blob. Watershed then partitions the mask."""
    mask = remove_small_holes(mask, area_threshold=min_area_px)
    coords = peak_local_max(
        dapi_sm, min_distance=min_distance_px, labels=mask,
        footprint=np.ones((3, 3)), exclude_border=False,
    )
    peaks = np.zeros(mask.shape, dtype=bool)
    peaks[tuple(coords.T)] = True
    markers = label(peaks)
    labels = watershed(-dapi_sm, markers, mask=mask)
    # drop tiny fragments
    out = np.zeros_like(labels)
    keep = 0
    for r in regionprops(labels):
        if r.area >= min_area_px:
            keep += 1
            out[labels == r.label] = keep
    return out


# ---------------------------------------------------------------- shading

def flatfield_correct(img, sigma_frac=0.09):
    """Remove smooth illumination shading (microscope vignetting: center bright,
    corners dim). Estimate the low-frequency illumination field by heavy
    Gaussian blur and divide it out, so a nucleus in a dim corner ends up as
    bright as one in the center and a SINGLE threshold works field-wide.

    This is the automated, seam-free equivalent of manually thresholding the
    image quarter-by-quarter."""
    sigma = max(60, int(round(img.shape[0] * sigma_frac)))
    illum = gaussian(img, sigma=sigma, preserve_range=True)
    floor = np.percentile(illum, 20) * 0.5          # avoid blow-up in dark voids
    illum = np.maximum(illum, floor)
    corrected = img / illum * illum.mean()
    return corrected.astype(np.float32), illum


# ---------------------------------------------------------------- main

def run(nd2_path, nuclei_ch, myotube_ch, outdir, downsample_to=256):
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "qc"), exist_ok=True)
    dapi, desmin, um_px = load_channels(nd2_path, nuclei_ch, myotube_ch)
    H, W = dapi.shape

    # micron-calibrated cleanup sizes
    px_area_um = um_px * um_px
    nucleus_min_area_px = int(round(20.0 / px_area_um))    # >~20 um^2 object
    nucleus_min_dist_px = max(3, int(round(6.0 / um_px)))  # ~6 um between peaks
    myotube_min_area_px = int(round(80.0 / px_area_um))    # drop small specks
    dapi_min_area_px = int(round(15.0 / px_area_um))

    # ---- binary matrices
    # nuclei: flat-field correct FIRST (removes the ~2.3x center-to-corner
    # illumination gradient), then one Li threshold is valid everywhere -- dim
    # corner nuclei are no longer missed. Corrected image also drives seeding.
    dapi_ff, illum = flatfield_correct(dapi)
    dapi_sm = gaussian(dapi_ff, sigma=1.0, preserve_range=True)
    thr1 = float(threshold_li(dapi_sm))
    M1 = remove_small_objects(dapi_sm > thr1, min_size=dapi_min_area_px)
    # myotubes: thin/dim fibers need a ridge detector, not a global threshold.
    # Replicates the lab's Fiji recipe (bg-subtract -> CLAHE -> ridge ->
    # hysteresis) so faint tendrils are recovered, not just the bright cores.
    M2, m2dbg = detect_myotubes(desmin, min_obj_um2_px=myotube_min_area_px)
    thr2 = m2dbg["coverage_pct"]

    # ---- (1) pixel matrix subtraction  AND  bijective 4-class code
    # subtraction (M1 - M2) collides: (1,1) and (0,0) both give 0.
    # code = 2*M1 + M2 is bijective -> 0 bg, 1 myotube-only, 2 nucleus-only,
    # 3 overlap; bincount gives all four exactly with no ambiguity.
    M1i = M1.astype(np.int8)
    M2i = M2.astype(np.int8)
    outcome = M1i - M2i
    code = (2 * M1.astype(np.uint8) + M2.astype(np.uint8))
    counts = np.bincount(code.ravel(), minlength=4)
    total_px = H * W
    background_px = int(counts[0])   # code 0  (0,0)
    minus1_px = int(counts[1])       # code 1  myotube only  (0,1)  -> M1-M2 = -1
    plus1_px = int(counts[2])        # code 2  nucleus only  (1,0)  -> M1-M2 = +1
    overlap_px = int(counts[3])      # code 3  overlap        (1,1)  -> M1-M2 =  0

    # ---- (2) nuclei-count fusion index
    labels = segment_nuclei(dapi_sm, M1, nucleus_min_area_px, nucleus_min_dist_px)
    cy_list, cx_list, inside_list = [], [], []
    for r in regionprops(labels):
        ys, xs = r.coords[:, 0], r.coords[:, 1]
        frac_on_myotube = M2[ys, xs].mean()   # fraction of nucleus area on Desmin
        cy_list.append(r.centroid[0])
        cx_list.append(r.centroid[1])
        inside_list.append(frac_on_myotube >= 0.5)
    cys = np.array(cy_list)
    cxs = np.array(cx_list)
    inside = np.array(inside_list, dtype=bool)
    n_total = int(inside.size)
    n_inside = int(inside.sum())
    fusion_index = (n_inside / n_total) if n_total else 0.0

    # ---- save downsampled binary matrices + outcome (compact, inspectable)
    def down(mat):
        step_y = max(1, H // downsample_to)
        step_x = max(1, W // downsample_to)
        return mat[::step_y, ::step_x]
    np.save(os.path.join(outdir, "matrix1_nuclei.npy"), down(M1).astype(np.uint8))
    np.save(os.path.join(outdir, "matrix2_myotube.npy"), down(M2).astype(np.uint8))
    np.save(os.path.join(outdir, "outcome_matrix.npy"), down(outcome).astype(np.int8))
    np.save(os.path.join(outdir, "class_code_map.npy"), down(code).astype(np.uint8))

    # ---- shading-correction QC: raw DAPI | illumination field | flat-fielded
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    for a, im, ttl in [
        (ax[0], down(dapi), "raw DAPI (center-bright shading)"),
        (ax[1], down(illum), "estimated illumination field"),
        (ax[2], down(dapi_ff), "flat-field corrected (uniform)")]:
        vlo, vhi = np.percentile(im, 1), np.percentile(im, 99)
        a.imshow(im, cmap="magma", vmin=vlo, vmax=vhi)
        a.set_title(ttl, fontsize=13); a.axis("off")
    fig.savefig(os.path.join(outdir, "qc", "shading_correction.png"),
                dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- overlay PNG: magenta=nucleus, green=myotube, white=overlap
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[..., 1] = (M2 * 180).astype(np.uint8)               # green myotube
    rgb[..., 0] = (M1 * 180).astype(np.uint8)               # red   nucleus
    rgb[..., 2] = (M1 * 180).astype(np.uint8)               # blue  nucleus -> magenta
    overlap = (M1 & M2)
    rgb[overlap] = [255, 255, 255]                          # white overlap
    _save_png(os.path.join(outdir, "overlay_masks.png"), down(rgb))

    # ---- RGB base for figures: dim raw DAPI (blue) + myotube mask (green)
    lo, hi = np.percentile(dapi, 1), np.percentile(dapi, 99.5)
    dapi8 = np.clip((dapi - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    base = np.zeros((H, W, 3), np.uint8)
    base[..., 2] = (dapi8 * 0.9).astype(np.uint8)           # blue nuclei
    base[..., 1] = np.maximum(base[..., 1], (M2 * 70).astype(np.uint8))  # green myotube

    # ---- FULL-FIELD labeled figure: every nucleus marked
    #      cyan = nucleus outside myotube, red = nucleus inside myotube
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.imshow(base)
    out_m = ~inside
    ax.scatter(cxs[out_m], cys[out_m], s=6, c="#00e5ff", linewidths=0,
               label=f"nucleus outside myotube  (n={n_total - n_inside:,})")
    ax.scatter(cxs[inside], cys[inside], s=10, c="#ff2d2d", linewidths=0,
               label=f"nucleus INSIDE myotube  (n={n_inside:,})")
    ax.set_title(
        f"{os.path.basename(nd2_path)}   |   total nuclei = {n_total:,}   |   "
        f"inside = {n_inside:,}   |   fusion index = {fusion_index*100:.1f}%",
        fontsize=16)
    leg = ax.legend(loc="upper right", fontsize=14, framealpha=0.85,
                    markerscale=3)
    leg.get_frame().set_facecolor("black")
    for t in leg.get_texts():
        t.set_color("white")
    # scale bar: 200 microns
    bar_px = 200.0 / um_px
    ax.plot([W * 0.03, W * 0.03 + bar_px], [H * 0.97, H * 0.97],
            c="white", lw=4)
    ax.text(W * 0.03, H * 0.955, "200 um", color="white", fontsize=13)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    fig.savefig(os.path.join(outdir, "labeled_nuclei_full.png"),
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    # ---- zoomed QC crop (same colour scheme) for close inspection
    y0, x0, s = min(1400, H - 800), min(1400, W - 800), 800
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(base[y0:y0 + s, x0:x0 + s])
    sel = (cys >= y0) & (cys < y0 + s) & (cxs >= x0) & (cxs < x0 + s)
    so, si = sel & out_m, sel & inside
    ax.scatter(cxs[so] - x0, cys[so] - y0, s=40, facecolors="none",
               edgecolors="#00e5ff", linewidths=1.2)
    ax.scatter(cxs[si] - x0, cys[si] - y0, s=55, facecolors="none",
               edgecolors="#ff2d2d", linewidths=1.6)
    ax.set_title("QC crop — cyan=outside, red=inside", fontsize=13)
    ax.axis("off")
    fig.savefig(os.path.join(outdir, "qc", "nuclei_segmentation_crop.png"),
                dpi=140, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    results = {
        "source": os.path.basename(nd2_path),
        "channels": {"nuclei(DAPI)": nuclei_ch, "myotube(Desmin)": myotube_ch},
        "image_px": [H, W],
        "um_per_px": round(um_px, 4),
        "dapi_threshold": thr1,
        "myotube_coverage_pct": round(thr2, 2),
        "pixel_classes_2M1_plus_M2": {
            "code0_background": background_px,
            "code1_myotube_only": minus1_px,
            "code2_nucleus_only": plus1_px,
            "code3_overlap": overlap_px,
            "note": "bijective encoding; code3 is the true nucleus-in-myotube overlap",
        },
        "pixel_matrix_subtraction": {
            "total_px": total_px,
            "overlap_both1_px": overlap_px,
            "plus1_nucleus_only_px": plus1_px,
            "minus1_myotube_only_px": minus1_px,
            "background_both0_px": background_px,
            "overlap_pct_of_image": round(100 * overlap_px / total_px, 3),
            "nucleus_area_inside_myotube_pct": round(
                100 * overlap_px / max(1, overlap_px + plus1_px), 2),
            "myotube_area_pct": round(
                100 * (overlap_px + minus1_px) / total_px, 2),
        },
        "fusion_index_count": {
            "nuclei_total": n_total,
            "nuclei_inside_myotube": n_inside,
            "nuclei_outside_myotube": n_total - n_inside,
            "fusion_index": round(fusion_index, 4),
            "rule": "nucleus counted inside if >=50% of its area overlaps Desmin mask",
        },
    }
    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--nuclei-ch", type=int, default=2, help="DAPI channel index")
    ap.add_argument("--myotube-ch", type=int, default=1, help="Desmin channel index")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--downsample-to", type=int, default=256)
    a = ap.parse_args()
    res = run(a.nd2, a.nuclei_ch, a.myotube_ch, a.outdir, a.downsample_to)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
