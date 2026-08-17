"""Visualise an instance-label image -- from Omnipose, from Fiji, or from the GT.

Deliberately standalone and dependency-light (numpy / scipy / skimage / PIL), so
it runs on the workstation against labels copied back from klone. Segmentation
needs `cellpose_omni` and a GPU; looking at the answer should not.

Four views, because they fail differently:

``instances``  every object a different colour -- shows IDENTITY. A fibre broken
               into three pieces is obvious here and invisible in a binary mask.
``outlines``   boundaries only, over the raw image -- shows BOUNDARY ACCURACY.
               The filled view hides whether an edge sits on the real fibre.
``length``     objects shaded by measured length -- shows the SCIENTIFIC READOUT
               directly, so fragmentation reads as "everything is pale".
``panel``      all three beside the raw field, for a single figure.

Run from the repo root::

    python model_labs/omnipose_lab/visualize_labels.py \\
        --labels <labels.tif> --image <image_fiber.tif> --out <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PIXEL_UM = 0.650017
# Sequential ramp, one hue light->dark (never a rainbow: rainbow makes equal
# steps in value look like unequal steps in colour).
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
        "#0d366b"]


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0


def ramp(t: np.ndarray) -> np.ndarray:
    """Interpolate the blue ramp at t in [0, 1]."""
    stops = np.stack([_hex(c) for c in BLUE])
    x = np.clip(t, 0, 1) * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[:, None]
    return stops[i] * (1 - f) + stops[i + 1] * f


def grey(image: np.ndarray, dim: float = 1.0) -> np.ndarray:
    lo, hi = np.percentile(image, 1), np.percentile(image, 99.5)
    g = np.clip((image.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.repeat((g * dim)[:, :, None], 3, axis=2)


def view_instances(image, labels, seed=7):
    rgb = grey(image, 0.5)
    n = int(labels.max())
    if n:
        rng = np.random.default_rng(seed)
        lut = np.zeros((n + 1, 3))
        lut[1:] = rng.integers(80, 255, size=(n, 3)) / 255.0
        m = labels > 0
        rgb[m] = lut[labels[m]]
    return rgb


def view_outlines(image, labels):
    """Boundaries only. The fill hides whether the edge is on the real fibre."""
    from scipy import ndimage as ndi
    rgb = grey(image, 1.0)
    mx = ndi.maximum_filter(labels, size=3)
    mn = ndi.minimum_filter(labels, size=3)
    edge = (mx != mn) & (labels > 0)
    rgb[edge] = [1.0, 0.25, 0.0]
    return rgb


def view_length(image, labels, lengths_um, vmax=None):
    rgb = grey(image, 0.35)
    if not lengths_um:
        return rgb, 0.0
    vmax = vmax or float(np.percentile(list(lengths_um.values()), 95))
    n = int(labels.max())
    lut = np.zeros((n + 1, 3))
    t = np.array([min(lengths_um.get(i, 0.0) / max(vmax, 1e-6), 1.0)
                  for i in range(1, n + 1)])
    if n:
        lut[1:] = ramp(t)
    m = labels > 0
    rgb[m] = lut[labels[m]]
    return rgb, vmax


def measure(labels, pixel_um):
    from scipy import ndimage as ndi
    from precision_myotube.geometry import measure_mask

    h, w = labels.shape
    rows, lengths = [], {}
    for lid, box in enumerate(ndi.find_objects(labels), start=1):
        if box is None:
            continue
        sub = labels[box] == lid
        if sub.sum() < 4:
            continue
        try:
            g = measure_mask(sub, pixel_um)
        except ValueError:
            continue
        ys, xs = box
        lengths[lid] = g.length_um
        rows.append({"instance": lid, "area_um2": round(g.area_um2, 2),
                     "length_um": round(g.length_um, 2),
                     "width_median_um": round(g.width_median_um, 3),
                     "aspect_ratio": round(g.length_um / g.width_median_um, 2)
                     if g.width_median_um > 0 else None,
                     "touches_border": bool(ys.start <= 0 or xs.start <= 0
                                            or ys.stop >= h or xs.stop >= w)})
    return rows, lengths


def save(rgb, path, max_px=2400):
    from PIL import Image
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((max_px, max_px))
    im.save(path)


def main(argv=None) -> int:
    import tifffile

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True, help="instance-label tif/npy")
    ap.add_argument("--image", required=True, help="raw Desmin field")
    ap.add_argument("--out", default="model_labs/omnipose_lab/_runs/viz")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-px", type=int, default=2400)
    ap.add_argument("--no-panel", action="store_true")
    args = ap.parse_args(argv)

    lp = Path(args.labels)
    labels = (np.load(lp) if lp.suffix == ".npy"
              else tifffile.imread(lp)).astype(np.int32)
    image = tifffile.imread(args.image)
    if labels.shape != image.shape:
        raise SystemExit(f"labels {labels.shape} != image {image.shape}")

    tag = args.tag or lp.stem.replace("__labels", "")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, lengths = measure(labels, args.pixel_um)
    L = np.array([r["length_um"] for r in rows]) if rows else np.array([0.0])
    n_border = sum(r["touches_border"] for r in rows)
    print(f"{tag}: {int(labels.max())} labels, {len(rows)} measured")
    print(f"  length um  median {np.median(L):7.1f}  p90 {np.percentile(L,90):7.1f}"
          f"  max {L.max():7.1f}")
    print(f"  touching border: {n_border}  "
          f"({100*n_border/max(len(rows),1):.1f}%) -- lengths are lower bounds")

    save(view_instances(image, labels), out / f"{tag}__instances.png", args.max_px)
    save(view_outlines(image, labels), out / f"{tag}__outlines.png", args.max_px)
    lrgb, vmax = view_length(image, labels, lengths)
    save(lrgb, out / f"{tag}__length.png", args.max_px)

    if rows:
        with open(out / f"{tag}__instances.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    if not args.no_panel:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap

        def small(a, k=4):
            return a[::k, ::k]

        fig, ax = plt.subplots(2, 2, figsize=(15, 15), facecolor="#fcfcfb")
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        ax[0, 0].imshow(small(grey(image)))
        ax[0, 0].set_title("raw Desmin", fontsize=12)
        ax[0, 1].imshow(small(view_instances(image, labels)))
        ax[0, 1].set_title(f"instances — {len(rows)} objects, one colour each",
                           fontsize=12)
        ax[1, 0].imshow(small(view_outlines(image, labels)))
        ax[1, 0].set_title("outlines over raw — boundary accuracy", fontsize=12)
        im = ax[1, 1].imshow(small(lrgb))
        ax[1, 1].set_title(f"shaded by length (median {np.median(L):.0f} µm)",
                           fontsize=12)
        cmap = LinearSegmentedColormap.from_list("blue", BLUE)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=0, vmax=vmax))
        cb = fig.colorbar(sm, ax=ax[1, 1], fraction=0.046, pad=0.02)
        cb.set_label("length (µm)", fontsize=10)
        fig.suptitle(tag, fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out / f"{tag}__panel.png", dpi=110,
                    bbox_inches="tight", facecolor="#fcfcfb")

    (out / f"{tag}__summary.json").write_text(json.dumps(
        {"labels": str(lp), "image": str(args.image), "pixel_um": args.pixel_um,
         "n_labels": int(labels.max()), "n_measured": len(rows),
         "n_touching_border": n_border,
         "length_um": {"median": float(np.median(L)),
                       "p10": float(np.percentile(L, 10)),
                       "p90": float(np.percentile(L, 90)),
                       "max": float(L.max())}}, indent=2), encoding="utf-8")
    print(f"-> {out}/{tag}__{{instances,outlines,length,panel}}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
