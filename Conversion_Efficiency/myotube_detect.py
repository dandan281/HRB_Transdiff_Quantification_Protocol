"""Myotube (Desmin) detector tuned to capture thin/dim fibers.

Mirrors the lab's validated Fiji recipe (Subtract Background -> CLAHE ->
Ridge Detection) using scikit-image so it is fully automatable:

    background subtraction (white top-hat)
    -> CLAHE local contrast
    -> Sato tubeness filter (enhances elongated fibers at several widths)
    -> hysteresis threshold (dim tendrils kept if connected to a bright ridge)

Exposes detect_myotubes(desmin) -> (mask, debug dict).
Run directly to dump tuning overlays on an 800x800 crop for visual QC.
"""
from __future__ import annotations
import numpy as np
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import (white_tophat, disk, remove_small_objects,
                                binary_dilation, remove_small_holes)
from skimage.exposure import equalize_adapthist, rescale_intensity


def _preprocess(desmin, tophat_radius=40, clahe_clip=0.01):
    """Return (d_bs, clahe): background-subtracted intensity (real signal, used
    to constrain mask WIDTH) and its CLAHE version (used for ridge detection so
    dim fibers become detectable)."""
    lo, hi = np.percentile(desmin, 1), np.percentile(desmin, 99.9)
    d = rescale_intensity(desmin, in_range=(lo, hi), out_range=(0.0, 1.0))
    d_bs = white_tophat(d, disk(tophat_radius))          # subtract slow haze
    clahe = equalize_adapthist(d_bs, kernel_size=desmin.shape[0] // 16,
                               clip_limit=clahe_clip)
    return d_bs, clahe


def detect_myotubes(desmin, sigmas=(1, 2, 4), low_pct=80, high_pct=94,
                    gate_pct=90, min_obj_um2_px=180):
    """Binary myotube mask that captures thin/dim fibers WITHOUT over-dilating.

    Ridge (tubeness) detection on the contrast-enhanced image finds fibers
    including faint ones; a hysteresis threshold links dim tendrils to bright
    ridges. The result is then GATED by real background-subtracted intensity so
    the mask hugs the true fiber body instead of ballooning into the dim halo
    around each fiber (which inflated area, especially in sparse/control wells).

    low_pct/high_pct: hysteresis seeds on the tubeness response.
    gate_pct: intensity percentile (of non-zero bg-subtracted signal) a pixel
              must also exceed -> controls mask width."""
    d_bs, clahe = _preprocess(desmin)
    tub = rescale_intensity(sato(clahe, sigmas=sigmas, black_ridges=False),
                            out_range=(0.0, 1.0))
    nz = tub[tub > 0]
    ridge = apply_hysteresis_threshold(tub, np.percentile(nz, low_pct),
                                       np.percentile(nz, high_pct))
    gate = binary_dilation(d_bs > np.percentile(d_bs[d_bs > 0], gate_pct),
                           disk(1))                      # width = real signal
    mask = ridge & gate
    mask = remove_small_objects(mask, min_size=min_obj_um2_px)
    mask = remove_small_holes(mask, area_threshold=min_obj_um2_px)
    return mask, {"gate_pct": gate_pct, "coverage_pct": float(100 * mask.mean())}


if __name__ == "__main__":
    import nd2, os
    from PIL import Image
    f = "../Q_PLATES/Q_Plates/PLATE_23/32_C08_br223_igf1r.nd2"
    with nd2.ND2File(f) as x:
        des = x.asarray()[1].astype(np.float32)
    os.makedirs("outputs/tune", exist_ok=True)

    def stretch(a, p=99.5):
        lo, hi = np.percentile(a, 1), np.percentile(a, p)
        return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)

    # tune on two crops (dense + sparse) for speed
    crops = {"c1": (1400, 1400), "c2": (300, 2600)}
    param_sets = {
        "A_hi94": dict(low_pct=80, high_pct=94),
        "B_hi90": dict(low_pct=72, high_pct=90),
        "C_hi86": dict(low_pct=65, high_pct=86),
    }
    for cname, (y0, x0) in crops.items():
        s = 800
        sub = des[y0:y0 + s, x0:x0 + s]
        raw = stretch(sub)
        Image.fromarray(raw).save(f"outputs/tune/{cname}_raw.png")
        for pname, kw in param_sets.items():
            m, dbg = detect_myotubes(sub, **kw)
            rgb = np.zeros((s, s, 3), np.uint8)
            rgb[..., 1] = raw                       # green raw
            rgb[m, 0] = 120                          # red tint = mask fill
            rgb[m, 2] = 120
            cov = 100 * m.mean()
            Image.fromarray(rgb).save(
                f"outputs/tune/{cname}_{pname}_cov{cov:.1f}.png")
        print(f"{cname} done")
    print("tuning overlays in outputs/tune/")
