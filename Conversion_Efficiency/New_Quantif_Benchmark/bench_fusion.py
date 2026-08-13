"""Benchmark conversion efficiency: nuclei inside Desmin+ myotube territory.

Counts, per image and pooled over the 25-image set:
  - total valid nuclei (Cellpose masks, area-filtered)
  - Desmin-positive nuclei = nuclei inside the hole-filled myotube territory,
    at BOTH >=25% and >=50% pixel-overlap (lab convention: always report both)
  - the same at a >=50 um traced-fibre gate (sensitivity check: territory
    restricted to real fibres, fragments removed at the traced-fibre level)

Pixel size: MEASURED from the burned-in scale bar in the bottom-right corner
of every image (50 um ruler = 96 px -> 0.521 um/px). Area bounds are set from
THIS set's pooled area histogram (see nuclei_area_hist.png) and applied
identically to every image.

Run:  cpenv/Scripts/python.exe bench_fusion.py [--amin-um2 50 --amax-um2 500]
"""
from __future__ import annotations
import os, json, glob, argparse
import numpy as np
import tifffile
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skan import Skeleton, summarize
import networkx as nx
from PIL import Image

BENCH = r"C:\Users\liqig\Documents\HRB_Transdiff\Benchmark"
HERE = os.path.dirname(os.path.abspath(__file__))
NUC_DIR = os.path.join(HERE, "nuclei")
MYO_DIR = os.path.join(HERE, "myotube2")   # v2 broad-ribbon detector
OUT = os.path.join(HERE, "fusion")

UM = 0.521                    # um/px, measured from the burned-in 50um scale bar
UM2 = UM * UM
SPUR_UM = 10.0
STRAIGHT_DOT = -0.5
FRACS = (0.25, 0.5)
GATES_UM = (0.0, 50.0)        # 0 = all Desmin territory; 50 = real fibres only


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def trace_fibres(myo_mask):
    """(skel, nearest_idx, fibres[(length_um, pixel_rc)]) — whole fibres traced
    straight through crossings; terminal spurs < SPUR_UM pruned."""
    skel = skeletonize(myo_mask > 0)
    idx = distance_transform_edt(~skel, return_distances=False,
                                 return_indices=True)
    if skel.sum() < 3:
        return skel, idx, []
    sk = Skeleton(skel)
    df = summarize(sk, separator="-")
    dcol = [c for c in df.columns if "branch" in c and "distance" in c][0]
    tcol = [c for c in df.columns if "branch" in c and "type" in c][0]
    scol = [c for c in df.columns if "node" in c and "src" in c][0]
    ecol = [c for c in df.columns if "node" in c and "dst" in c][0]
    Lpx = df[dcol].to_numpy(dtype=np.float64)
    T = df[tcol].to_numpy()
    src = df[scol].to_numpy(); dst = df[ecol].to_numpy()

    coords = [np.asarray(sk.path_coordinates(i), float) for i in range(len(Lpx))]
    G = nx.Graph()
    node_ends = {}
    for i in range(len(Lpx)):
        if T[i] == 1 and Lpx[i] * UM < SPUR_UM:
            continue
        G.add_edge(("b", i, "s"), ("b", i, "d"), w=float(Lpx[i]) * UM, bi=i)
        c = coords[i]
        step = min(3, len(c) - 1)
        ds = _unit(c[step] - c[0]) if len(c) > 1 else np.zeros(2)
        dd = _unit(c[-1 - step] - c[-1]) if len(c) > 1 else np.zeros(2)
        node_ends.setdefault(int(src[i]), []).append((i, "s", ds))
        node_ends.setdefault(int(dst[i]), []).append((i, "d", dd))
    for ends in node_ends.values():
        if len(ends) < 2:
            continue
        pairs = []
        for a in range(len(ends)):
            for b in range(a + 1, len(ends)):
                pairs.append((float(np.dot(ends[a][2], ends[b][2])), a, b))
        pairs.sort()
        used = set()
        for dot, a, b in pairs:
            if dot > STRAIGHT_DOT:
                break
            if a in used or b in used:
                continue
            used.update((a, b))
            ia, ea, _ = ends[a]; ib, eb, _ = ends[b]
            G.add_edge(("b", ia, ea), ("b", ib, eb), w=0.0)

    fibres = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        bis, tot = set(), 0.0
        for _, _, d in sub.edges(data=True):
            tot += d["w"]
            if "bi" in d:
                bis.add(d["bi"])
        if tot <= 0:
            continue
        pix = np.concatenate([coords[i] for i in bis]).round().astype(int)
        fibres.append((tot, pix))
    return skel, idx, fibres


def real_territory(myo_mask, skel, idx, fibres, gate_um):
    if gate_um <= 0:
        return binary_fill_holes(myo_mask > 0), len(fibres)
    kept = np.zeros_like(skel, dtype=bool)
    n_real = 0
    for length_um, pix in fibres:
        if length_um >= gate_um:
            kept[pix[:, 0], pix[:, 1]] = True
            n_real += 1
    nearest_kept = kept[idx[0], idx[1]]
    real = (myo_mask > 0) & nearest_kept
    return binary_fill_holes(real), n_real


def nuclei_inside(nuc, territory, amin_um2, amax_um2, frac):
    flat = nuc.ravel()
    area = np.bincount(flat).astype(np.float64)
    inside = np.bincount(flat, weights=territory.ravel().astype(np.float64),
                         minlength=area.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(area > 0, inside / area, 0.0)
    area_um2 = area * UM2
    valid = (area_um2 >= amin_um2) & (area_um2 <= amax_um2)
    valid[0] = False
    return (fr >= frac) & valid, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amin-um2", type=float, default=50.0)
    ap.add_argument("--amax-um2", type=float, default=500.0)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fkey = lambda f: f"overlap_{int(round(100 * f))}pct"      # noqa: E731

    stems = sorted((os.path.splitext(os.path.basename(p))[0]
                    for p in glob.glob(os.path.join(BENCH, "*.tif"))), key=int)
    results = {}
    totals = {g: {fkey(f): {"inside": 0} for f in FRACS} for g in GATES_UM}
    for g in GATES_UM:
        totals[g]["total"] = 0

    for stem in stems:
        nuc = np.load(os.path.join(NUC_DIR, f"{stem}_masks.npy"))
        myo = np.load(os.path.join(MYO_DIR, f"{stem}_myotube_mask.npy"))
        skel, idx, fibres = trace_fibres(myo)

        rec = {}
        for g in GATES_UM:
            terr, n_real = real_territory(myo, skel, idx, fibres, g)
            gr = {"n_fibres": n_real,
                  "territory_cov_pct": round(100 * terr.mean(), 2)}
            for f in FRACS:
                is_in, valid = nuclei_inside(nuc, terr, a.amin_um2, a.amax_um2, f)
                nin, ntot = int(is_in.sum()), int(valid.sum())
                gr[fkey(f)] = {"inside": nin,
                               "pct": round(100 * nin / ntot, 2) if ntot else 0}
                totals[g][fkey(f)]["inside"] += nin
                if f == FRACS[0]:
                    gr["total_valid"] = ntot
                    totals[g]["total"] += ntot
                if g == 0.0:
                    if f == 0.25:
                        is_in_25, valid_v = is_in, valid
                    else:
                        is_in_50 = is_in
                    terr_0 = terr
            rec[f"gate{int(g)}um"] = gr
        results[stem] = rec

        # 3-class overlay at gate 0: green=territory, magenta=inside(>=25%),
        # orange=inside only at 25% not 50%, blue=outside (valid nuclei only)
        inside25 = is_in_25[nuc]
        inside50 = is_in_50[nuc]
        outside = valid_v[nuc] & (nuc > 0) & ~inside25
        rgb = np.zeros((*nuc.shape, 3), np.float32)
        rgb[..., 1] = 0.30 * terr_0
        rgb[inside25] = [1.0, 0.55, 0.1]          # orange: 25-50%
        rgb[inside50] = [1.0, 0.1, 0.9]           # magenta: >=50%
        rgb[outside, 2] = 1.0; rgb[outside, 0] = 0.1
        Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(OUT, f"{stem}_fusion_overlay.png"))
        g0 = rec["gate0um"]
        print(f"img {stem:>3s}: total={g0['total_valid']:4d}  "
              f"in25={g0[fkey(0.25)]['inside']:4d} ({g0[fkey(0.25)]['pct']:5.1f}%)  "
              f"in50={g0[fkey(0.5)]['inside']:4d} ({g0[fkey(0.5)]['pct']:5.1f}%)  "
              f"cov={g0['territory_cov_pct']:5.1f}%", flush=True)

    summary = {}
    for g in GATES_UM:
        t = totals[g]["total"]
        summary[f"gate{int(g)}um"] = {
            "total_valid_nuclei": t,
            **{fkey(f): {"inside": totals[g][fkey(f)]["inside"],
                         "pct": round(100 * totals[g][fkey(f)]["inside"] / t, 2)
                         if t else 0} for f in FRACS}}
    with open(os.path.join(OUT, "fusion_results.json"), "w") as fh:
        json.dump({"um_per_px_assumed": UM,
                   "nucleus_area_um2": [a.amin_um2, a.amax_um2],
                   "overlap_fracs": list(FRACS), "gates_um": list(GATES_UM),
                   "per_image": results, "pooled": summary}, fh, indent=2)

    print("-" * 78)
    for g in GATES_UM:
        s = summary[f"gate{int(g)}um"]
        print(f"POOLED gate>={int(g):>2d}um: {s['total_valid_nuclei']} valid nuclei | "
              + "  ".join(f"ov{int(100*f)}%: {s[fkey(f)]['inside']} "
                          f"({s[fkey(f)]['pct']:.2f}%)" for f in FRACS))
    print("FUSION_DONE")


if __name__ == "__main__":
    main()
