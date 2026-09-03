"""Conversion efficiency for ANY plate -- one command, one env (fijiconv).

    conda run -n fijiconv python Conversion_Efficiency/quantify_ce_plate.py \\
        --plate "Q_PLATES/Q_Plates/PLATE_26" [--layout wells.csv] [--wells B02 ...]

A folder of .nd2 files in; per well out: nuclei, desmin-positive fraction
(perinuclear), and the nucleus-in-myotube conversion efficiency at BOTH the
25 % and 50 % overlap conventions, plus treatment summaries with a
permutation test, the knob-sweep curves that justify every cut, and a QC
overlay per well. wells.csv / summary.json / sweeps.png / <well>_overlay.png
under model_labs/tracer_lab/_runs/ce_plates/<PLATE>/.

WHAT IT WRAPS (the lane's current, ground-truth-backed method) -- nothing
here is a new readout:

* nuclei: the Plate-44 Fiji recipe (Gaussian blur -> rolling-ball ->
  absolute DAPI cut -> fill holes -> open -> watershed -> area gate), run
  through ImageJ in-process (scyjava) exactly as `extract_per_nucleus.py`.
* desmin per nucleus: mean of rolling-ball-subtracted desmin in a thin
  shell just OUTSIDE each nucleus (perinuclear) -- the cytoplasmic-marker
  readout; nucleus-pixel overlap is invalid for desmin (2026-08-20).
* conversion efficiency: nucleus >= 25 % / 50 % inside the FILLED
  absolute-threshold desmin territory (k * one shared plate sigma) -- the
  form validated against the PLATE_32 per-nucleus fusion ground truth
  (rho +0.89 at k=5, `Plate32_Fusion_GT/eval_v2_candidate.py`).

THE THREE RULES IT OBEYS:

1. one plate = ONE parameter set, identical for every well, chosen from
   the plate's own pooled data -- never per well, never from expectations;
2. every cut is taken at the PLATEAU of its knob curve (min |d ln y / d ln x|),
   swept over the full range and plotted, for THIS plate -- nothing is
   inherited from another plate;
3. both overlap conventions are always reported, and treatment differences
   get a label-shuffle permutation test before anyone reads a ranking.

Geometry constants are in MICROMETRES and converted per plate from the
nd2 pixel size (the Plate-44 recipe's pixel constants at 1.72 um/px would
gate away real nuclei on a 0.65 um/px plate).

Channels: inferred (DAPI = the channel yielding the most nucleus-sized
objects; desmin = the remaining channel with the most fibre-like coverage),
recorded per well, and shown in channels.png -- check it once per plate,
and use --dapi-ch / --desmin-ch when it is wrong or a channel is saturated.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
FIJI = r"C:\Users\liqig\AppData\Local\FijiConvCounter\Fiji.app"
JAVA_HOME = r"C:\Users\liqig\anaconda3\envs\fijiconv\Library\lib\jvm"
OUT_ROOT = ROOT / "model_labs/tracer_lab/_runs/ce_plates"
WELL_RE = re.compile(r"^[A-P]\d{2}$")

# ---- the recipe, in micrometres (Plate-44 operating point at 1.7246 um/px)
BLUR_UM = 1.72            # Gaussian sigma, was 1 px
RB_DAPI_UM = 43.0         # rolling ball, was 25 px
RB_DES_UM = 259.0         # rolling ball, was 150 px
SHELL_UM = 10.3           # perinuclear shell, was 6 px
NUC_LO_UM2, NUC_HI_UM2 = 24.0, 2080.0     # area gate, was 8-700 px
MIN_OBJ_UM2 = 76.0        # territory-mask small-object floor, was 180 px @0.65
FRACS = (0.25, 0.5)
K_GRID = (2, 3, 4, 5, 6, 8, 10, 12)
PLATEAU_TOL = 1.5
INST_FLOOR = 0.10         # |d ln y / d ln x| below this = "barely changes"
N_PERM = 5000

_C: dict = {}


# ----------------------------------------------------------------- ImageJ
def cls(n):
    if n not in _C:
        from scyjava import jimport
        _C[n] = jimport(n)
    return _C[n]


def ij_start():
    os.environ["JAVA_HOME"] = JAVA_HOME
    import scyjava
    scyjava.config.add_option("-Xmx6g")
    import imagej
    ij = imagej.init(FIJI, mode="headless")
    cls("ij.Prefs").blackBackground = True
    cls("ij.IJ").run("Options...", "iterations=1 count=1 black do=Nothing")
    return ij


def to_imp(name, arr):
    from jpype.types import JArray, JShort
    h, w = arr.shape
    ja = JArray(JShort)(np.ascontiguousarray(arr, dtype=np.uint16)
                        .view(np.int16).ravel())
    imp = cls("ij.ImagePlus")(name, cls("ij.process.ShortProcessor")(
        w, h, ja, None))
    imp.setCalibration(cls("ij.measure.Calibration")())
    return imp


def to_np(imp):
    ip = imp.getProcessor()
    px = np.asarray(ip.getPixels())
    w, h = int(ip.getWidth()), int(ip.getHeight())
    return (px.view(np.uint16) if px.dtype == np.int16
            else px.view(np.uint8)).reshape(h, w).copy()


def shut(imp):
    imp.changes = False
    imp.close()


def preprocess(dapi_raw, des_raw, px):
    """Fiji: blur + rolling-ball on DAPI, rolling-ball on desmin."""
    IJ = cls("ij.IJ")
    imp = to_imp("d", dapi_raw)
    IJ.run(imp, "Gaussian Blur...", f"sigma={px['blur']:.2f}")
    IJ.run(imp, "Subtract Background...", f"rolling={px['rb_dapi']}")
    dpre = to_np(imp)
    shut(imp)
    imp = to_imp("m", des_raw)
    IJ.run(imp, "Subtract Background...", f"rolling={px['rb_des']}")
    des = to_np(imp)
    shut(imp)
    return dpre, des


def segment(dpre, T, px):
    """Fiji: absolute cut -> fill -> open -> watershed; numpy: label + gate.
    Returns (label map, kept ids, areas[ids])."""
    IJ, IP = cls("ij.IJ"), cls("ij.process.ImageProcessor")
    imp = to_imp("d2", dpre)
    imp.getProcessor().setThreshold(float(T) + 0.5, 65535.0, IP.NO_LUT_UPDATE)
    IJ.run(imp, "Convert to Mask", "")
    if imp.getProcessor().isInvertedLut():
        IJ.run(imp, "Invert LUT", "")
        IJ.run(imp, "Invert", "")
    IJ.run(imp, "Fill Holes", "")
    IJ.run(imp, "Open", "")
    IJ.run(imp, "Watershed", "")
    nm = to_np(imp)
    shut(imp)
    lab, _ = ndi.label(nm > 0, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    ids = np.where((sizes >= px["nuc_lo"]) & (sizes <= px["nuc_hi"]))[0]
    keep = np.zeros(sizes.size, bool)
    keep[ids] = True
    lab = np.where(keep[lab], lab, 0)
    return lab, ids, sizes[ids]


# ----------------------------------------------------------------- helpers
def well_token(stem):
    for part in stem.split("_"):
        if WELL_RE.match(part.upper()):
            return part.upper()
    return stem


def treatment_from_stem(stem, well):
    parts = [p for p in stem.split("_") if p.upper() != well
             and not p.isdigit()]
    return "_".join(parts) if parts else ""


def px_constants(pixel_um):
    a = pixel_um ** 2
    return {"blur": BLUR_UM / pixel_um,
            "rb_dapi": max(int(round(RB_DAPI_UM / pixel_um)), 1),
            "rb_des": max(int(round(RB_DES_UM / pixel_um)), 1),
            "shell": max(int(round(SHELL_UM / pixel_um)), 1),
            "nuc_lo": max(int(round(NUC_LO_UM2 / a)), 1),
            "nuc_hi": int(round(NUC_HI_UM2 / a)),
            "min_obj": max(int(round(MIN_OBJ_UM2 / a)), 1)}


def robust_sigma(a):
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med)) + 1e-12


def plateau_pick(xs, ys, lo=None, hi=None, smooth=1):
    """min |d ln y / d ln x| on [lo, hi]; returns (x*, x_lo, x_hi, inst)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    inst = np.abs(np.gradient(np.log(np.maximum(ys, 1e-9)), np.log(xs)))
    if smooth > 1:
        inst = np.convolve(inst, np.ones(smooth) / smooth, mode="same")
    win = np.ones(len(xs), bool)
    if lo is not None:
        win &= xs >= lo
    if hi is not None:
        win &= xs <= hi
    i = int(np.argmin(np.where(win, inst, np.inf)))
    # the reported plateau RANGE: within PLATEAU_TOL of the minimum, or under
    # an absolute "barely changes" floor (y moves < 10 % per e-fold of x) --
    # without the floor a unimodal curve's zero-derivative peak reports a
    # one-point plateau (measured: "1293..1293")
    on = win & (inst <= max(inst[i] * PLATEAU_TOL, INST_FLOOR))
    a = b = i
    while a > 0 and on[a - 1]:
        a -= 1
    while b < len(xs) - 1 and on[b + 1]:
        b += 1
    return float(xs[i]), float(xs[a]), float(xs[b]), inst


def infer_channels(arr, px, ij_ready=True):
    """DAPI = most nucleus-sized objects at a mid cut; desmin = remaining
    channel with the most fibre-like coverage after background removal."""
    scores = {}
    for c in range(arr.shape[0]):
        ch = arr[c].astype(np.float32)
        sat = float(np.mean(ch >= 0.99 * ch.max()))
        # cheap nucleus proxy: blur, cut at p90, count blobs in the gate
        b = ndi.gaussian_filter(ch, px["blur"])
        m = b > np.percentile(b, 90)
        lab, n = ndi.label(m)
        sizes = np.bincount(lab.ravel())[1:]
        n_nuc = int(((sizes >= px["nuc_lo"]) & (sizes <= px["nuc_hi"])).sum())
        # fibre proxy: fraction of pixels far above the robust background
        cov = float(np.mean(ch > np.median(ch) + 5 * robust_sigma(ch)))
        scores[c] = {"nucleus_like": n_nuc, "coverage": round(cov, 4),
                     "saturated_frac": round(sat, 4)}
    dapi = max(scores, key=lambda c: scores[c]["nucleus_like"])
    rest = [c for c in scores if c != dapi]
    desmin = max(rest, key=lambda c: scores[c]["coverage"])
    return dapi, desmin, scores


# ----------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plate", required=True)
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--layout", default=None,
                    help="CSV with columns well,treatment (else parsed from "
                         "file names)")
    ap.add_argument("--dapi-ch", type=int, default=None)
    ap.add_argument("--desmin-ch", type=int, default=None)
    ap.add_argument("--n-t", type=int, default=16, help="DAPI cut grid size")
    ap.add_argument("--no-overlay", action="store_true")
    ap.add_argument("--out", default=str(OUT_ROOT))
    a = ap.parse_args(argv)

    import nd2
    plate_dir = Path(a.plate)
    files = sorted(plate_dir.glob("*.nd2"))
    wells = []
    for f in files:
        w = well_token(f.stem)
        if a.wells and w not in a.wells:
            continue
        wells.append((w, f))
    if not wells:
        raise SystemExit(f"no .nd2 files in {plate_dir}")
    names = [w for w, _ in wells]
    dup = sorted({w for w in names if names.count(w) > 1})
    if dup:
        raise SystemExit(f"well token collision {dup}; pass --wells")
    out = Path(a.out) / plate_dir.name.replace(" ", "_")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)

    layout = {}
    if a.layout:
        for r in csv.DictReader(open(a.layout)):
            layout[r["well"].strip().upper()] = r["treatment"].strip()

    t_all = time.time()
    ij = ij_start()
    print(f"ImageJ {ij.getVersion()} | plate {plate_dir.name}: "
          f"{len(wells)} wells", flush=True)

    # ---- pass 0: read, infer channels, preprocess, cache, pooled histogram
    meta = {}
    hist_dpre = np.zeros(65536, np.int64)
    for i, (w, f) in enumerate(wells, 1):
        t0 = time.time()
        with nd2.ND2File(str(f)) as h:
            arr = np.squeeze(np.asarray(h.asarray()))
            pixel_um = float(h.voxel_size().x)
        if arr.ndim == 2:
            arr = arr[None]
        px = px_constants(pixel_um)
        if a.dapi_ch is not None and a.desmin_ch is not None:
            dapi_c, des_c, scores = a.dapi_ch, a.desmin_ch, {}
            basis = "override"
        else:
            dapi_c, des_c, scores = infer_channels(arr, px)
            basis = "inferred"
            if a.dapi_ch is not None:
                dapi_c = a.dapi_ch
            if a.desmin_ch is not None:
                des_c = a.desmin_ch
        dpre, des = preprocess(arr[dapi_c], arr[des_c], px)
        np.save(cache / f"{w}_dpre.npy", dpre)
        np.save(cache / f"{w}_des.npy", des)
        hist_dpre += np.bincount(dpre.ravel(), minlength=65536)[:65536]
        sat = [c for c, s in scores.items() if s["saturated_frac"] > 0.02]
        meta[w] = {"nd2": f.name, "pixel_um": pixel_um, "px": px,
                   "dapi_ch": int(dapi_c), "desmin_ch": int(des_c),
                   "channel_basis": basis, "channel_scores": scores,
                   "saturated_channels": sat, "shape": list(dpre.shape),
                   "treatment": layout.get(w, treatment_from_stem(f.stem, w))}
        if i == 1:
            _channels_png(arr, out / "channels.png", dapi_c, des_c)
        flag = f"  ** ch{sat} saturated **" if sat else ""
        print(f"  [{i:>2}/{len(wells)}] {w}: dapi=ch{dapi_c} desmin=ch{des_c} "
              f"({basis}) {pixel_um:.4f} um/px  ({time.time() - t0:.0f}s){flag}",
              flush=True)

    # ---- pass 1: DAPI cut sweep (pooled), plateau rule
    cdf = np.cumsum(hist_dpre) / hist_dpre.sum()
    p50 = int(np.searchsorted(cdf, 0.50))
    p999 = int(np.searchsorted(cdf, 0.999))
    T_grid = sorted({int(round(v)) for v in
                     np.exp(np.linspace(np.log(max(p50, 1)),
                                        np.log(max(p999, p50 + 1)), a.n_t))})
    counts = np.zeros((len(wells), len(T_grid)))
    print(f"\nDAPI cut sweep: {len(T_grid)} cuts in [{T_grid[0]}, "
          f"{T_grid[-1]}] (pooled p50..p99.9 of the pre-processed DAPI)",
          flush=True)
    for wi, (w, _) in enumerate(wells):
        dpre = np.load(cache / f"{w}_dpre.npy")
        t0 = time.time()
        for ti, T in enumerate(T_grid):
            _, ids, _ = segment(dpre, T, meta[w]["px"])
            counts[wi, ti] = len(ids)
        print(f"  {w}: n(T) = {counts[wi].astype(int).tolist()}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    pooled = counts.sum(0)
    ok = pooled >= 0.5 * pooled.max()
    T_star, T_lo, T_hi, inst_T = plateau_pick(
        np.array(T_grid)[ok], pooled[ok])
    T_star = int(T_star)
    print(f"  -> T* = {T_star} (plateau {int(T_lo)}..{int(T_hi)}; "
          f"count peak at {T_grid[int(np.argmax(pooled))]})", flush=True)

    # ---- pass 2: per-nucleus tables at T*, desmin territory masks per k
    sig = {}
    per = {}
    for w, _ in wells:
        des = np.load(cache / f"{w}_des.npy").astype(np.float32)
        sig[w] = robust_sigma(des)
    sig_ref = float(np.median(list(sig.values())))
    for w, _ in wells:
        t0 = time.time()
        px = meta[w]["px"]
        dpre = np.load(cache / f"{w}_dpre.npy")
        des = np.load(cache / f"{w}_des.npy").astype(np.float32)
        lab, ids, areas = segment(dpre, T_star, px)
        dist, (iy, ix) = ndi.distance_transform_edt(lab == 0,
                                                    return_indices=True)
        shell = np.where((dist > 0) & (dist <= px["shell"]), lab[iy, ix], 0)
        peri = np.nan_to_num(np.atleast_1d(np.asarray(
            ndi.mean(des, shell, index=ids), float)))
        dnuc = np.nan_to_num(np.atleast_1d(np.asarray(
            ndi.mean(dpre.astype(np.float32), lab, index=ids), float)))
        cen = ndi.center_of_mass(lab > 0, lab, ids) if len(ids) else []
        flat = lab.ravel()
        area_all = np.bincount(flat).astype(np.float64)
        frac_k = {}
        cov_k = {}
        for k in K_GRID:
            m = des > k * sig_ref
            ml, _ = ndi.label(m)
            sz = np.bincount(ml.ravel())
            sz[0] = 0
            m = sz[ml] >= px["min_obj"]
            m = ndi.binary_fill_holes(m)
            inside = np.bincount(flat, weights=m.ravel().astype(np.float64),
                                 minlength=area_all.size)
            with np.errstate(invalid="ignore", divide="ignore"):
                fr = np.where(area_all > 0, inside / area_all, 0.0)
            frac_k[k] = fr[ids].astype(np.float32)
            cov_k[k] = float(m.mean())
        per[w] = {"ids": ids, "area": areas.astype(np.int32),
                  "peri": peri.astype(np.float32),
                  "dapi": dnuc.astype(np.float32),
                  "y": np.array([c[0] for c in cen], np.float32),
                  "x": np.array([c[1] for c in cen], np.float32),
                  "frac_k": frac_k, "cov_k": cov_k, "lab": lab}
        print(f"  {w}: {len(ids)} nuclei at T*  ({time.time() - t0:.0f}s)",
              flush=True)

    # ---- desmin cut (perinuclear), plateau rule on the pooled distribution.
    # The fraction-positive curve is flat by construction at BOTH ends
    # (f -> 1 as the cut -> 0, f -> 0 as the cut -> max); those are trivial
    # plateaus, not the shoulder the rule is after. Measured on Plate 44:
    # unrestricted, the rule picked cut 22 -> 95 % "positive". The search is
    # therefore confined to cuts calling 5-80 % positive -- an exclusion of
    # the degenerate flats, not a calibration to any expected value.
    peri_all = np.concatenate([per[w]["peri"] for w, _ in wells])
    lo, hi = np.percentile(peri_all[peri_all > 0], [1, 99.5])
    cuts = np.exp(np.linspace(np.log(max(lo, 1)), np.log(hi), 160))
    f_all = np.array([(peri_all > c).mean() for c in cuts])
    band = (f_all <= 0.80) & (f_all >= 0.05)
    if band.any():
        c_lo, c_hi = float(cuts[band][0]), float(cuts[band][-1])
    else:
        c_lo, c_hi = None, None
    cut_star, cut_lo, cut_hi, inst_c = plateau_pick(cuts, f_all, lo=c_lo,
                                                    hi=c_hi, smooth=7)
    print(f"\ndesmin cut* = {cut_star:.0f} raw units (plateau "
          f"{cut_lo:.0f}..{cut_hi:.0f}) -> {100 * (peri_all > cut_star).mean():.1f}% "
          f"desmin+ pooled", flush=True)

    # ---- k for the territory mask, plateau rule on pooled CE25 vs k
    n_tot = sum(len(per[w]["ids"]) for w, _ in wells)
    ce_k = {f: [] for f in FRACS}
    cov_mean = []
    for k in K_GRID:
        for f in FRACS:
            ce_k[f].append(100 * sum(float((per[w]["frac_k"][k] >= f).sum())
                                     for w, _ in wells) / max(n_tot, 1))
        cov_mean.append(np.mean([per[w]["cov_k"][k] for w, _ in wells]))
    k_star, k_lo, k_hi, inst_k = plateau_pick(K_GRID, ce_k[0.25])
    k_star = int(k_star)
    print(f"territory k* = {k_star} (plateau {int(k_lo)}..{int(k_hi)}; "
          f"sigma {sig_ref:.1f}; coverage "
          f"{100 * cov_mean[K_GRID.index(k_star)]:.1f}%)", flush=True)

    # ---- per-well rows
    rows = []
    for w, f in wells:
        p = per[w]
        n = len(p["ids"])
        r = {"well": w, "nd2": f.name, "treatment": meta[w]["treatment"],
             "pixel_um": round(meta[w]["pixel_um"], 5),
             "dapi_ch": meta[w]["dapi_ch"], "desmin_ch": meta[w]["desmin_ch"],
             "nuclei": n,
             "desmin_pos": int((p["peri"] > cut_star).sum()),
             "desmin_pos_pct": round(100 * float((p["peri"] > cut_star).mean()), 3)
             if n else 0.0,
             "ce25_pct": round(100 * float((p["frac_k"][k_star] >= 0.25).mean()), 3)
             if n else 0.0,
             "ce50_pct": round(100 * float((p["frac_k"][k_star] >= 0.5).mean()), 3)
             if n else 0.0,
             "territory_cov_pct": round(100 * p["cov_k"][k_star], 2),
             "peri_median": round(float(np.median(p["peri"])), 1) if n else 0.0,
             "saturated_channels": meta[w]["saturated_channels"]}
        rows.append(r)
        if not a.no_overlay:
            _overlay(np.load(cache / f"{w}_des.npy"), p["lab"], p["ids"],
                     p["peri"] > cut_star, out / f"{w}_overlay.png")

    # ---- treatments + permutation test (only if >= 2 groups with >= 2 wells)
    treat = {}
    groups = {}
    for r in rows:
        groups.setdefault(r["treatment"], []).append(r)
    if len([g for g in groups.values() if len(g) >= 2]) >= 2 and \
            len(groups) >= 2:
        rng = np.random.default_rng(0)
        y_keys = ("desmin_pos_pct", "ce25_pct", "ce50_pct")
        g = np.array([r["treatment"] for r in rows])
        for key in y_keys:
            y = np.array([r[key] for r in rows], float)

            def eta2(yv, gv):
                gm = yv.mean()
                ssb = sum(((yv[gv == t].mean() - gm) ** 2) * (gv == t).sum()
                          for t in np.unique(gv))
                return ssb / max(((yv - gm) ** 2).sum(), 1e-12)
            obs = eta2(y, g)
            null = np.array([eta2(y, rng.permutation(g))
                             for _ in range(N_PERM)])
            treat[key] = {
                "per_treatment": {t: {"n_wells": len(v),
                                      "mean": round(float(np.mean(
                                          [r[key] for r in v])), 3),
                                      "sem": round(float(np.std(
                                          [r[key] for r in v], ddof=1)
                                          / np.sqrt(len(v))), 3)
                                      if len(v) > 1 else 0.0}
                                  for t, v in groups.items()},
                "eta2": round(float(obs), 4),
                "eta2_chance": round(float(null.mean()), 4),
                "perm_p": round(float((null >= obs).mean()), 4)}

    # ---- outputs
    with (out / "wells.csv").open("w", newline="") as fh:
        keys = [k for k in rows[0] if k != "saturated_channels"]
        wr = csv.DictWriter(fh, fieldnames=keys + ["saturated_channels"])
        wr.writeheader()
        for r in rows:
            wr.writerow({**r, "saturated_channels": ";".join(
                f"ch{c}" for c in r["saturated_channels"])})
    np.savez_compressed(out / "per_nucleus.npz", **{
        f"{w}_{k}": per[w][k] for w, _ in wells
        for k in ("x", "y", "area", "peri", "dapi")},
        **{f"{w}_frac_k{k_star}": per[w]["frac_k"][k_star] for w, _ in wells})
    summary = {
        "plate": plate_dir.name, "n_wells": len(wells),
        "recipe_um": {"blur": BLUR_UM, "rb_dapi": RB_DAPI_UM,
                      "rb_des": RB_DES_UM, "shell": SHELL_UM,
                      "nuc_area": [NUC_LO_UM2, NUC_HI_UM2],
                      "min_obj_area": MIN_OBJ_UM2},
        "chosen": {"T_dapi": T_star, "T_plateau": [int(T_lo), int(T_hi)],
                   "T_grid": T_grid, "pooled_counts": pooled.astype(int).tolist(),
                   "desmin_cut_raw": round(cut_star, 1),
                   "desmin_cut_plateau": [round(cut_lo, 1), round(cut_hi, 1)],
                   "k": k_star, "k_plateau": [int(k_lo), int(k_hi)],
                   "shared_sigma": round(sig_ref, 2),
                   "ce25_vs_k": dict(zip(map(str, K_GRID),
                                         [round(v, 2) for v in ce_k[0.25]])),
                   "ce50_vs_k": dict(zip(map(str, K_GRID),
                                         [round(v, 2) for v in ce_k[0.5]]))},
        "rules": ["one parameter set per plate, from pooled data",
                  "every cut at the plateau of its own sweep on this plate",
                  "both overlap conventions reported; permutation test "
                  "before ranking"],
        "wells": rows, "meta": meta, "treatments": treat}
    (out / "summary.json").write_text(json.dumps(summary, indent=2,
                                                 default=str))
    _sweeps_png(T_grid, pooled, T_star, T_lo, T_hi, cuts, f_all, cut_star,
                cut_lo, cut_hi, K_GRID, ce_k, k_star, out / "sweeps.png")

    print(f"\n{'well':<5}{'treatment':<16}{'nuclei':>8}{'desmin+%':>10}"
          f"{'CE25%':>8}{'CE50%':>8}{'cov%':>7}")
    for r in rows:
        print(f"{r['well']:<5}{r['treatment'][:15]:<16}{r['nuclei']:>8}"
              f"{r['desmin_pos_pct']:>10.2f}{r['ce25_pct']:>8.2f}"
              f"{r['ce50_pct']:>8.2f}{r['territory_cov_pct']:>7.1f}")
    for key, tr in treat.items():
        print(f"\n{key}: eta2 {tr['eta2']:.3f} (chance {tr['eta2_chance']:.3f})"
              f"  permutation p = {tr['perm_p']:.4f}")
        for t, v in tr["per_treatment"].items():
            print(f"   {t or '(none)':<20} n={v['n_wells']}  "
                  f"{v['mean']:.2f} +/- {v['sem']:.2f}")
    print(f"\n-> {out / 'wells.csv'}\n-> {out / 'summary.json'}\n"
          f"-> {out / 'sweeps.png'}  (the three knob curves + plateaus)\n"
          f"-> {out / 'channels.png'}  (check channel roles ONCE per plate)\n"
          f"-> {out}/<well>_overlay.png\n"
          f"done in {(time.time() - t_all) / 60:.1f} min", flush=True)
    sys.stdout.flush()
    os._exit(0)       # JVM teardown otherwise hangs the interpreter


# ----------------------------------------------------------------- figures
def _stretch(a, lo=1, hi=99.7):
    a = a.astype(np.float32)
    l, h = np.percentile(a, [lo, hi])
    return np.clip((a - l) / max(h - l, 1e-6), 0, 1)


def _channels_png(arr, path, dapi_c, des_c):
    from PIL import Image
    panels = []
    for c in range(arr.shape[0]):
        p = _stretch(arr[c])[::2, ::2]
        tag = " DAPI" if c == dapi_c else " DESMIN" if c == des_c else ""
        p[:14, :] = 1.0 if tag else p[:14, :]          # a white bar marks a chosen channel
        panels.append(p)
    gap = np.ones((panels[0].shape[0], 8)) * 0.5
    row = panels[0]
    for p in panels[1:]:
        row = np.concatenate([row, gap, p], axis=1)
    Image.fromarray((row * 255).astype(np.uint8)).save(path)


def _overlay(des, lab, ids, pos, path):
    """Desmin image with nucleus outlines: orange = desmin+, blue = negative."""
    from PIL import Image
    base = _stretch(des)
    rgb = np.stack([base * 0.7] * 3, -1)
    edge = (lab > 0) & (ndi.minimum_filter(lab, size=3) != lab)
    pos_lab = np.zeros(lab.max() + 1, bool)
    pos_lab[ids[pos]] = True
    rgb[edge & pos_lab[lab]] = [0.92, 0.41, 0.20]
    rgb[edge & ~pos_lab[lab]] = [0.16, 0.47, 0.84]
    h2, w2 = (rgb.shape[0] // 2) * 2, (rgb.shape[1] // 2) * 2
    r = rgb[:h2, :w2]
    half = (r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2] + r[1::2, 1::2]) / 4
    Image.fromarray((np.clip(half, 0, 1) * 255).astype(np.uint8)).save(path)


def _sweeps_png(T_grid, pooled, T_star, T_lo, T_hi, cuts, f_all, cut_star,
                cut_lo, cut_hi, K, ce_k, k_star, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a60"
    fig, ax = plt.subplots(1, 3, figsize=(13, 4), facecolor="white")
    ax[0].plot(T_grid, pooled, "-o", color=BLUE, ms=4)
    ax[0].axvspan(T_lo, T_hi, color=ORANGE, alpha=0.15)
    ax[0].axvline(T_star, color=ORANGE, lw=1.5, label=f"T* = {T_star}")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("DAPI cut (pre-processed units)")
    ax[0].set_ylabel("nuclei, pooled")
    ax[0].set_title("A  nucleus count vs DAPI cut")
    ax[1].plot(cuts, 100 * f_all, color=BLUE)
    ax[1].axvspan(cut_lo, cut_hi, color=ORANGE, alpha=0.15)
    ax[1].axvline(cut_star, color=ORANGE, lw=1.5, label=f"cut* = {cut_star:.0f}")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("perinuclear desmin cut (raw units)")
    ax[1].set_ylabel("desmin+ nuclei, pooled (%)")
    ax[1].set_title("B  desmin+ fraction vs cut")
    ax[2].plot(K, ce_k[0.25], "-o", color=BLUE, ms=4, label="CE 25%")
    ax[2].plot(K, ce_k[0.5], "-o", color=ORANGE, ms=4, label="CE 50%")
    ax[2].axvline(k_star, color=INK, lw=1.2, ls="--", label=f"k* = {k_star}")
    ax[2].set_xlabel("territory threshold k (x plate sigma)")
    ax[2].set_ylabel("conversion efficiency, pooled (%)")
    ax[2].set_title("C  CE vs territory threshold")
    for x in ax:
        x.legend(frameon=False)
        x.spines[["top", "right"]].set_visible(False)
        x.spines[["left", "bottom"]].set_color(MUTED)
        x.tick_params(colors=MUTED)
        x.grid(color="#e8e7df", lw=0.8)
        x.set_axisbelow(True)
    fig.suptitle("Every cut taken at the plateau of its own sweep on THIS plate",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")


if __name__ == "__main__":
    raise SystemExit(main())
