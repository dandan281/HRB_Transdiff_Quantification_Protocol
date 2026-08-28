"""Validate ridge-offset yardsticks on synthetic fibres BEFORE real data.

The question that needs answering: does the operator's traced line sit on the
image's intensity ridge, and by how much does it wander? The answer decides
whether the network's 12 px-FWHM blur is partly irreducible (noisy targets)
or entirely the model's fault.

Two yardsticks have already been pointed at that question and RETRACTED
(commit ed6b75f): one was a no-op behind a redefined metric, the other's
control -- the tracing shifted 4 px sideways -- separated by only ~0.9 px, so
it could not have detected the effect it claimed to measure. The lesson is
the harness in this file: a yardstick may be used on real data ONLY after it
recovers known shifts on synthetic fibres whose centreline is known exactly.

Synthetic conditions mimic what real myotubes actually look like (measured
profiles, this plate): ~8 px width, intensity varying along the fibre,
multiplicative speckle, flat-topped cross-sections, a parallel neighbour one
fibre-width away, and shot noise. A yardstick passes a condition when its
median reported offset tracks an injected shift of 0..4 px within 25% and
0.35 px absolute.

    python model_labs/tracer_lab/ridge_yardstick.py            # validate
    python model_labs/tracer_lab/ridge_yardstick.py --measure  # real wells,
                                                               # passing
                                                               # yardsticks only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# synthetic fibres with exactly known centrelines
# ---------------------------------------------------------------------------

def make_fibre_field(rng, *, n_fibres=6, shape=(512, 512), width_px=4.0,
                     flat_top=False, speckle=0.0, neighbour_px=0.0,
                     noise=0.02):
    """-> (image [0,1], list of (n,2) float centrelines, tangents list).

    Fibres are smooth curves; intensity falls off from the TRUE centreline,
    optionally flat-topped (plateau of ~width/2 before the falloff),
    modulated along the fibre, specked multiplicatively, and a parallel
    neighbour can be laid `neighbour_px` away to contaminate profiles the way
    a bundle does.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    H, W = shape
    img = np.zeros(shape, np.float64)
    lines, tangents = [], []

    for i in range(n_fibres):
        y0 = (i + 0.7) * H / (n_fibres + 1) + rng.uniform(-8, 8)
        x = np.arange(30.0, W - 30.0, 1.0)
        amp = rng.uniform(6, 18)
        per = rng.uniform(150, 300)
        ph = rng.uniform(0, 2 * np.pi)
        y = y0 + amp * np.sin(2 * np.pi * x / per + ph)
        pts = np.column_stack([y, x])
        d = np.gradient(pts, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        lines.append(pts)
        tangents.append(d)

        curves = [pts]
        if neighbour_px > 0:
            n_vec = np.column_stack([-d[:, 1], d[:, 0]])
            curves.append(pts + neighbour_px * n_vec)

        for c in curves:
            tree = cKDTree(c)
            r0 = max(int(c[:, 0].min() - 12), 0)
            r1 = min(int(c[:, 0].max() + 13), H)
            rr, cc = np.mgrid[r0:r1, 0:W]
            dist, idx = tree.query(np.column_stack([rr.ravel(), cc.ravel()]),
                                   workers=-1)
            dist = dist.reshape(rr.shape)
            idx = idx.reshape(rr.shape)
            if flat_top:
                eff = np.clip(dist - width_px / 2.0, 0.0, None)
                prof = np.exp(-(eff ** 2) / (2.0 * (width_px / 2.0) ** 2))
            else:
                prof = np.exp(-(dist ** 2) / (2.0 * width_px ** 2))
            along = 0.75 + 0.25 * np.sin(idx / rng.uniform(40, 90)
                                         + rng.uniform(0, 6.3))
            img[r0:r1] = np.maximum(img[r0:r1],
                                    rng.uniform(0.5, 1.0) * prof * along)

    if speckle > 0:
        tex = ndimage.gaussian_filter(rng.normal(0, 1, shape), 2.0)
        tex = 1.0 + speckle * tex / np.abs(tex).std()
        img *= np.clip(tex, 0.2, None)
    img = ndimage.gaussian_filter(img, 0.8)
    img += rng.normal(0, noise, shape)
    img -= img.min()
    img /= img.max()
    return img.astype(np.float32), lines, tangents


# ---------------------------------------------------------------------------
# yardstick candidates: (image, points, tangents) -> per-point offset (px)
# ---------------------------------------------------------------------------

_SMOOTH_CACHE: list = [None, None]   # [source array, smoothed array]


def _profiles(image, pts, tans, half=8.0, step=0.5):
    from scipy import ndimage
    # smoothing a 3636^2 field costs ~0.5 s; the snap calls this thousands of
    # times per well on the SAME image, so the last smoothed field is kept.
    # Identity is checked on the array object itself, not id(), so a freed
    # id being reused cannot alias a stale cache entry.
    if _SMOOTH_CACHE[0] is not image:
        _SMOOTH_CACHE[0] = image
        _SMOOTH_CACHE[1] = ndimage.gaussian_filter(
            image.astype(np.float64), 1.0)
    im = _SMOOTH_CACHE[1]
    offs = np.arange(-half, half + 1e-9, step)
    n = np.column_stack([-tans[:, 1], tans[:, 0]])
    sample = pts[:, None, :] + offs[None, :, None] * n[:, None, :]
    vals = ndimage.map_coordinates(im, [sample[..., 0].ravel(),
                                        sample[..., 1].ravel()], order=1)
    return offs, vals.reshape(len(pts), len(offs))


def yard_peak(image, pts, tans):
    """Sub-pixel argmax of the perpendicular profile."""
    offs, P = _profiles(image, pts, tans)
    k = np.argmax(P, axis=1)
    ok = (k > 0) & (k < P.shape[1] - 1)
    y0 = P[np.arange(len(P)), np.clip(k - 1, 0, None)]
    y1 = P[np.arange(len(P)), k]
    y2 = P[np.arange(len(P)), np.minimum(k + 1, P.shape[1] - 1)]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    est = offs[k] + sub * (offs[1] - offs[0])
    return est[ok]


def yard_centroid(image, pts, tans):
    """Background-subtracted intensity centroid of the profile."""
    offs, P = _profiles(image, pts, tans)
    base = P.min(axis=1, keepdims=True)
    w = np.clip(P - base, 0, None) ** 2
    s = w.sum(axis=1)
    ok = s > 1e-9
    return (w @ offs)[ok] / s[ok]


def yard_matched(image, pts, tans, sigma=4.0):
    """Cross-correlation with the expected fibre profile, sub-pixel shift."""
    offs, P = _profiles(image, pts, tans)
    lags = np.arange(-12, 13)
    tmpl = np.exp(-(offs ** 2) / (2.0 * sigma ** 2))
    tmpl -= tmpl.mean()
    Pc = P - P.mean(axis=1, keepdims=True)
    corr = np.stack([np.roll(Pc, -l, axis=1) @ tmpl for l in lags], axis=1)
    k = np.argmax(corr, axis=1)
    ok = (k > 0) & (k < len(lags) - 1)
    y0 = corr[np.arange(len(corr)), np.clip(k - 1, 0, None)]
    y1 = corr[np.arange(len(corr)), k]
    y2 = corr[np.arange(len(corr)), np.minimum(k + 1, len(lags) - 1)]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    return ((lags[k] + sub) * (offs[1] - offs[0]))[ok]


def yard_peak3(image, pts, tans):
    """Sub-pixel argmax restricted to a +-3 px basin.

    The wide yardsticks all FAIL the neighbour condition: with a parallel
    fibre 9 px away, a +-8 px search jumps to the wrong ridge (measured:
    a 2 px shift reported as -6.9 px). A +-3 px basin cannot see the
    neighbour at all, at the cost of only measuring misalignments up to
    ~2.5 px -- which is the regime the annotation question lives in. Points
    whose profile peaks at the basin edge are dropped as unresolvable rather
    than trusted.
    """
    offs, P = _profiles(image, pts, tans, half=3.0, step=0.25)
    k = np.argmax(P, axis=1)
    ok = (k > 0) & (k < P.shape[1] - 1)          # edge peak = ambiguous, drop
    y0 = P[np.arange(len(P)), np.clip(k - 1, 0, None)]
    y1 = P[np.arange(len(P)), k]
    y2 = P[np.arange(len(P)), np.minimum(k + 1, P.shape[1] - 1)]
    den = y0 - 2 * y1 + y2
    sub = np.where(np.abs(den) > 1e-12, 0.5 * (y0 - y2) / den, 0.0)
    est = offs[k] + sub * (offs[1] - offs[0])
    return est[ok]


def yard_trace_mean(image, pts, tans):
    """ONE offset per trace: average all its profiles, then find the peak.

    Per-point yardsticks fail under speckle -- texture bumps inside the
    search window capture the argmax and a 2.5 px shift reads as ~1.2
    (measured above). Speckle is independent ALONG the fibre, so averaging
    the perpendicular profiles over the whole trace cancels it, while a
    systematic line-vs-ridge offset survives the average untouched. The
    search stays in a +-4 px basin so a 9 px neighbour remains invisible.

    This deliberately answers a narrower question: the per-TRACE systematic
    offset, not per-point wobble. That is the damaging kind for training --
    correlated target error cannot be averaged away by any loss.
    """
    offs, P = _profiles(image, pts, tans, half=5.0, step=0.25)
    prof = P.mean(axis=0)
    basin = np.abs(offs) <= 4.0
    idx = np.where(basin)[0]
    k = idx[np.argmax(prof[idx])]
    if k == idx[0] or k == idx[-1]:
        return np.array([])              # peak at basin edge: unresolvable
    y0, y1, y2 = prof[k - 1], prof[k], prof[k + 1]
    den = y0 - 2 * y1 + y2
    sub = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    return np.array([offs[k] + sub * (offs[1] - offs[0])])


YARDSTICKS = {"peak": yard_peak, "centroid": yard_centroid,
              "matched": yard_matched}
# validated separately on the shifts they are built for
BASIN_YARDSTICKS = {"peak3": yard_peak3, "trace_mean": yard_trace_mean}
BASIN_SHIFTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)

CONDITIONS = {
    "clean":     dict(),
    "flat_top":  dict(flat_top=True),
    "speckle":   dict(speckle=0.35),
    "neighbour": dict(neighbour_px=9.0),
    "noisy":     dict(noise=0.06, speckle=0.35),
    "hard":      dict(flat_top=True, speckle=0.35, neighbour_px=9.0,
                      noise=0.04),
}
SHIFTS = (0.0, 1.0, 2.0, 3.0, 4.0)


def validate(seed=0, subsample=6):
    """Every yardstick x condition x shift. A cell passes when the median
    recovered offset is within max(0.35, 0.25*shift) of the injected shift."""
    rng = np.random.default_rng(seed)
    results = {}
    print(f"{'yardstick':<10}{'condition':<11}"
          + "".join(f"{s:>8.0f}px" for s in SHIFTS) + "   verdict")
    print("-" * 78)
    for cond, kw in CONDITIONS.items():
        img, lines, tans = make_fibre_field(rng, **kw)
        for name, fn in YARDSTICKS.items():
            meds = []
            for s in SHIFTS:
                est_all = []
                for pts, tan in zip(lines, tans):
                    n = np.column_stack([-tan[:, 1], tan[:, 0]])
                    moved = (pts + s * n)[::subsample]
                    est_all.append(fn(img, moved, tan[::subsample]))
                e = np.concatenate(est_all)
                meds.append(float(np.median(-e)) if len(e) else np.nan)
                # NOTE the sign: moving the line +s along its normal puts the
                # ridge at -s in the line's own frame; report the recovered
                # line-to-ridge distance as a positive shift
            ok = all(abs(m - s) <= max(0.35, 0.25 * s)
                     for m, s in zip(meds, SHIFTS))
            results[(name, cond)] = {"medians": meds, "pass": ok}
            print(f"{name:<10}{cond:<11}"
                  + "".join(f"{m:>10.2f}" for m in meds)
                  + f"   {'PASS' if ok else 'FAIL'}")
    wide_passing = sorted({n for (n, c), r in results.items()
                           if all(results[(n, cc)]["pass"]
                                  for cc in CONDITIONS)})
    print(f"\nwide yardsticks passing EVERY condition: "
          f"{wide_passing or 'NONE'}\n")

    # the basin yardstick, on the small shifts it is built for
    print(f"{'yardstick':<10}{'condition':<11}"
          + "".join(f"{s:>8.1f}px" for s in BASIN_SHIFTS) + "   verdict")
    print("-" * 78)
    for cond, kw in CONDITIONS.items():
        img, lines, tans = make_fibre_field(np.random.default_rng(seed + 1),
                                            **kw)
        for name, fn in BASIN_YARDSTICKS.items():
            meds = []
            for s in BASIN_SHIFTS:
                est_all = []
                for pts, tan in zip(lines, tans):
                    n = np.column_stack([-tan[:, 1], tan[:, 0]])
                    moved = (pts + s * n)[::subsample]
                    est_all.append(fn(img, moved, tan[::subsample]))
                e = np.concatenate(est_all)
                meds.append(float(np.median(-e)) if len(e) else np.nan)
            ok = all(abs(m - s) <= max(0.3, 0.2 * s)
                     for m, s in zip(meds, BASIN_SHIFTS))
            results[(name, cond)] = {"medians": meds, "pass": ok}
            print(f"{name:<10}{cond:<11}"
                  + "".join(f"{m:>10.2f}" for m in meds)
                  + f"   {'PASS' if ok else 'FAIL'}")
    passing = sorted({n for n in BASIN_YARDSTICKS
                      if all(results[(n, cc)]["pass"] for cc in CONDITIONS)})
    print(f"\nbasin yardsticks passing EVERY condition: {passing or 'NONE'}")
    return results, passing


# ---------------------------------------------------------------------------
# the real measurement -- passing yardsticks only
# ---------------------------------------------------------------------------

def measure_wells(passing, wells=("D04", "B02"), subsample=7):
    from tracer_lab.train_tracer import load_well

    out = {}
    for well in wells:
        image, gt, _ = load_well(well)
        for name in passing:
            fn = BASIN_YARDSTICKS[name]
            est = []
            for pts, tan in zip(gt["traces"], gt["tangents"]):
                good = np.linalg.norm(tan, axis=1) > 0.5
                p, t = pts[good][::subsample], tan[good][::subsample]
                if len(p) < 4:
                    continue
                est.append(fn(image, p, t))
            e = np.concatenate(est)
            e = e[np.abs(e) <= 8.0]
            out[(well, name)] = e
            print(f"{well} / {name}: n={len(e)}  median {np.median(e):+.2f} px"
                  f"  SD {e.std():.2f} px  |off|>2px {(np.abs(e) > 2).mean():.0%}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--measure", action="store_true",
                    help="after validation, measure the real wells with the "
                         "yardsticks that passed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    results, passing = validate(seed=a.seed)
    rec = {"passing": passing,
           "cells": {f"{n}/{c}": r for (n, c), r in results.items()}}
    if a.measure:
        if not passing:
            print("NO yardstick validated -- measuring real data is refused.")
            return 1
        wells = measure_wells(passing)
        rec["real"] = {f"{w}/{n}": {"n": int(len(e)),
                                    "median_px": float(np.median(e)),
                                    "sd_px": float(e.std()),
                                    "frac_gt2px": float((np.abs(e) > 2).mean())}
                       for (w, n), e in wells.items()}
    out = ROOT / "model_labs/tracer_lab/_runs/ridge_yardstick.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
