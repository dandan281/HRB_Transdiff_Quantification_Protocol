"""Myotube (Desmin) threshold sweep -- the same robustness trick used for nuclei,
applied to the ridge detector.

The expensive preprocessing (background subtraction + Sato ridge filter) is
IDENTICAL for every threshold, so it is computed ONCE per well; only the cheap
hysteresis threshold + gate + object count is swept. For each threshold level we
count the number of distinct Desmin+ myotubes; we pick the PLATEAU (the level
where the count is least sensitive to the knob) as the operating point and also
mark the raw PEAK.

Writes {stem}_myotube_sweep.png (count curve) + {stem}_myotube_labeled.png
(each fiber a distinct colour) + a myotube mask .npy, and appends to
myotube_results.jsonl.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import nd2
from skimage.filters import sato, apply_hysteresis_threshold
from skimage.morphology import (disk, binary_dilation, remove_small_objects,
                                remove_small_holes)
from skimage.exposure import rescale_intensity
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from myotube_detect import _preprocess

MIN_OBJ = 180        # px, same as detect_myotubes default
BAND = 14            # hysteresis band: low_pct = high_pct - BAND
GATE_PCT = 90        # intensity ("brightness") gate, held fixed


def stretch(a, p=99.5):
    lo, hi = np.percentile(a, 1), np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--desmin-ch", type=int, default=1)
    ap.add_argument("--highs", default="86,89,92,94,96")   # ridge threshold sweep
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.nd2))[0]
    highs = [float(v) for v in a.highs.split(",")]

    with nd2.ND2File(a.nd2) as x:
        desmin = x.asarray()[a.desmin_ch].astype(np.float32)

    import time
    t0 = time.time()
    # ---- expensive part, ONCE ----
    d_bs, clahe = _preprocess(desmin)
    tub = rescale_intensity(sato(clahe, sigmas=(1, 2, 4), black_ridges=False),
                            out_range=(0.0, 1.0))
    nz = tub[tub > 0]
    gate = binary_dilation(d_bs > np.percentile(d_bs[d_bs > 0], GATE_PCT), disk(1))
    pre_s = time.time() - t0

    # ---- cheap threshold sweep ----
    counts, covs, mask_by_h = [], [], {}
    for h in highs:
        ridge = apply_hysteresis_threshold(tub, np.percentile(nz, h - BAND),
                                           np.percentile(nz, h))
        m = ridge & gate
        m = remove_small_objects(m, min_size=MIN_OBJ)
        m = remove_small_holes(m, area_threshold=MIN_OBJ)
        lab = label(m)
        counts.append(int(lab.max()))
        covs.append(round(100 * m.mean(), 2))
        mask_by_h[h] = m
        print(f"  ridge_thresh(high_pct)={h:.0f}  myotubes={counts[-1]}  cov={covs[-1]}%")
    secs = time.time() - t0

    c = np.array(counts)
    peak_i = int(np.argmax(c))
    # plateau = interior point where neighbours differ least (most stable)
    best_i, best_key = None, None
    for i in range(1, len(highs) - 1):
        sens = abs(int(c[i - 1]) - int(c[i + 1]))
        key = (sens, -c[i])            # tie-break toward the higher (peakier) count
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    if best_i is None:
        best_i = peak_i
    op_h, op_count = highs[best_i], int(c[best_i])
    print(f"  -> plateau high_pct={op_h:.0f} count={op_count} | peak={int(c[peak_i])}"
          f" @ {highs[peak_i]:.0f}  (pre {pre_s:.0f}s / total {secs:.0f}s)")

    # ---- viz 1: count-vs-threshold sweep curve ----
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(highs, counts, "-o", color="#16a34a")
    ax.plot(op_h, op_count, "*", ms=22, color="#ef4444",
            label=f"plateau: thr={op_h:.0f}, n={op_count}")
    if peak_i != best_i:
        ax.plot(highs[peak_i], c[peak_i], "D", ms=10, color="#f59e0b",
                label=f"peak: n={int(c[peak_i])}")
    ax.set_xlabel("ridge threshold (hysteresis high percentile)  -> stricter")
    ax.set_ylabel("distinct myotubes counted")
    ax.set_title(f"{stem}\nmyotube-count threshold sweep")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(a.outdir, f"{stem}_myotube_sweep.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- viz 2: each myotube a distinct colour at the operating threshold ----
    m = mask_by_h[op_h]
    lab = label(m)
    base = stretch(desmin)
    over = label2rgb(lab, image=base, bg_label=0, alpha=0.55, image_alpha=1,
                     bg_color=(0, 0, 0))
    im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1800, 1800))
    im.save(os.path.join(a.outdir, f"{stem}_myotube_labeled.png"))
    np.save(os.path.join(a.outdir, f"{stem}_myotube_mask.npy"), m)

    rec = {"well": stem, "operating_high_pct": op_h, "myotubes": op_count,
           "peak_myotubes": int(c[peak_i]), "peak_high_pct": highs[peak_i],
           "coverage_pct": covs[best_i],
           "sweep": {f"{h:.0f}": n for h, n in zip(highs, counts)},
           "sweep_cov": {f"{h:.0f}": v for h, v in zip(highs, covs)},
           "seconds": round(secs, 1)}
    # Re-running a well must replace its old record, not append a second one --
    # the summary scripts sum every line, so duplicates double-count the plate.
    path = os.path.join(a.outdir, "myotube_results.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r.get("well") != rec["well"]] + [rec]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("MYOTUBE_DONE", stem, op_count)


if __name__ == "__main__":
    main()
