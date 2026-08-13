"""Fusion index using ONLY REAL myotubes -- fragments removed at the TRACED-FIBRE
level (not the connected-object level, which fails because short fragments are
wired into big meshes that survive a per-object length gate).

Per well:
  1. load PLATEAU-threshold Desmin mask + Cellpose nuclei masks.
  2. skeletonise; trace whole fibres THROUGH crossings (straightest-through
     pairing at junctions) -> each fibre has a true length and a set of pixels.
  3. for a length GATE, keep only the skeleton pixels of fibres >= gate, then
     rebuild the fibre BODIES by assigning every Desmin pixel to its nearest
     skeleton pixel and keeping those whose nearest fibre is a real one. Hole-fill
     -> "real myotube" territory.
  4. a nucleus is INSIDE if >= OVERLAP FRACTION of its pixels fall in that
     territory. This is a KEY HYPERPARAMETER: both 25% and 50% are reported at
     every gate (lab convention), so the sensitivity to it is always visible.

Sweeps gates so the sensitivity is explicit; saves overlays + JSON.
"""
from __future__ import annotations
import os, json, argparse
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.stats import gaussian_kde  # noqa (kept for parity, unused)
from skan import Skeleton, summarize
import networkx as nx
from PIL import Image

UM = 0.6493
UM2 = UM * UM                 # um^2 per pixel
SPUR_UM = 10.0
STRAIGHT_DOT = -0.5
NUC_DIR = "plate23_nuclei"
MYO_DIR = "plate23_myotube"


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def trace_fibres(myo_mask):
    """Return (skel, nearest_idx, fibres). fibres = list of (length_um, pixel_rc
    array of that whole fibre's skeleton pixels). nearest_idx maps every image
    pixel to its nearest skeleton pixel (for body reconstruction)."""
    skel = skeletonize(myo_mask > 0)
    if skel.sum() < 3:
        idx = distance_transform_edt(~skel, return_distances=False,
                                     return_indices=True)
        return skel, idx, []
    sk = Skeleton(skel)                       # pixel units; length = px * UM
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
        if T[i] == 1 and Lpx[i] * UM < SPUR_UM:        # prune spur
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
    idx = distance_transform_edt(~skel, return_distances=False,
                                 return_indices=True)
    return skel, idx, fibres


def real_territory(myo_mask, skel, idx, fibres, gate_um):
    kept = np.zeros_like(skel, dtype=bool)
    n_real = 0
    for length_um, pix in fibres:
        if length_um >= gate_um:
            kept[pix[:, 0], pix[:, 1]] = True
            n_real += 1
    nearest_kept = kept[idx[0], idx[1]]          # nearest-skeleton propagation
    real = (myo_mask > 0) & nearest_kept
    return binary_fill_holes(real), n_real


def nuclei_inside(nuc, territory, amin, amax, frac=0.5):
    """Nuclei INSIDE the territory, restricted to the area boundary [amin,amax] um^2.
    Returns (is_in per label, total valid nuclei, valid per label)."""
    flat = nuc.ravel()
    area = np.bincount(flat).astype(np.float64)                 # px per label
    inside = np.bincount(flat, weights=territory.ravel().astype(np.float64),
                         minlength=area.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        fr = np.where(area > 0, inside / area, 0.0)
    area_um2 = area * UM2
    valid = (area_um2 >= amin) & (area_um2 <= amax)             # area boundary
    valid[0] = False
    is_in = (fr >= frac) & valid
    return is_in, int(valid.sum()), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", default="0,30,50,100")
    ap.add_argument("--primary", type=float, default=50.0)
    ap.add_argument("--fracs", default="0.25,0.5",
                    help="nucleus-in-myotube overlap fractions to report (both by convention)")
    ap.add_argument("--primary-frac", type=float, default=0.5,
                    help="which overlap fraction fills the top-level inside/fusion_pct keys")
    ap.add_argument("--amin-um2", type=float, default=50.0)   # nucleus area boundary
    ap.add_argument("--amax-um2", type=float, default=500.0)
    ap.add_argument("--outdir", default="plate23_real_fusion")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    gates = [float(g) for g in a.gates.split(",")]
    fracs = [float(f) for f in a.fracs.split(",")]
    if a.primary_frac not in fracs:
        fracs.append(a.primary_frac)
    # The overlay block below reads terr_p/is_in_p/valid_p, which are only bound
    # on the (primary gate, primary frac) iteration — so the primary gate must be
    # in the sweep, same as the primary frac above.
    if a.primary not in gates:
        gates.append(a.primary)
    fkey = lambda f: f"overlap_{int(round(100 * f))}pct"      # noqa: E731

    wells = sorted(f.replace("_myotube_mask.npy", "") for f in os.listdir(MYO_DIR)
                   if f.endswith("_myotube_mask.npy"))

    results = {}
    totals = {g: {"inside": 0, "total": 0,
                  **{fkey(f): {"inside": 0, "total": 0} for f in fracs}} for g in gates}
    for w in wells:
        myo = np.load(os.path.join(MYO_DIR, f"{w}_myotube_mask.npy"))
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        skel, idx, fibres = trace_fibres(myo)

        per_gate = {}
        for g in gates:
            terr, n_real = real_territory(myo, skel, idx, fibres, g)
            rec = {"n_real_fibres": n_real,
                   "coverage_pct": round(100 * terr.mean(), 2)}
            for f in fracs:
                is_in, ntot, valid = nuclei_inside(nuc, terr, a.amin_um2, a.amax_um2,
                                                   frac=f)
                nin = int(is_in.sum())
                rec[fkey(f)] = {"inside": nin, "total": ntot,
                                "fusion_pct": round(100 * nin / ntot, 2) if ntot else 0}
                totals[g][fkey(f)]["inside"] += nin
                totals[g][fkey(f)]["total"] += ntot
                if f == a.primary_frac:
                    # legacy top-level keys mirror the primary fraction
                    rec.update({"inside": nin, "total": ntot,
                                "fusion_pct": rec[fkey(f)]["fusion_pct"]})
                    totals[g]["inside"] += nin
                    totals[g]["total"] += ntot
                    if g == a.primary:
                        terr_p, is_in_p, valid_p = terr, is_in, valid
            per_gate[g] = rec
        results[w] = per_gate

        inside_pix = is_in_p[nuc]
        outside_pix = valid_p[nuc] & (nuc > 0) & ~inside_pix   # area-filtered nuclei only
        rgb = np.zeros((*nuc.shape, 3), np.float32)
        rgb[..., 1] = 0.30 * terr_p
        rgb[inside_pix] = [1.0, 0.1, 0.9]
        rgb[outside_pix, 2] = 1.0; rgb[outside_pix, 0] = 0.1
        im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        im.thumbnail((1700, 1700))
        im.save(os.path.join(a.outdir,
                             f"{w}_realfusion_g{int(a.primary)}"
                             f"_ov{int(round(100*a.primary_frac))}.png"))
        print(f"{w:26s}" + "".join(
            f"  g{int(g)}=" + "/".join(f"{results[w][g][fkey(f)]['inside']}" for f in fracs)
            for g in gates))

    with open(os.path.join(a.outdir, "real_fusion_results.json"), "w") as fh:
        json.dump({"gates": gates, "primary": a.primary,
                   "overlap_fracs": fracs, "primary_frac": a.primary_frac,
                   "nucleus_area_um2": [a.amin_um2, a.amax_um2],
                   "per_well": results, "totals": totals}, fh, indent=2)

    print("-" * 70)
    print(f"nucleus area boundary = [{a.amin_um2:.0f}, {a.amax_um2:.0f}] um2"
          f"   | overlap fractions reported: "
          + ", ".join(f"{100*f:.0f}%" for f in fracs)
          + f"  (primary {100*a.primary_frac:.0f}%)")
    for g in gates:
        cells = "  ".join(
            f"ov{int(round(100*f)):>2d}%: {totals[g][fkey(f)]['inside']:6d}"
            f" ({100*totals[g][fkey(f)]['inside']/totals[g][fkey(f)]['total']:5.2f}%)"
            for f in fracs)
        print(f"PLATE gate>={int(g):>3d}um  {cells}"
              f"  (of {totals[g][fkey(fracs[0])]['total']} valid nuclei)")


if __name__ == "__main__":
    main()
