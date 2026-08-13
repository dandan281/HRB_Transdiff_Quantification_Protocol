"""Myotube LENGTH distribution.

For every Desmin+ myotube (connected component in the plateau-threshold mask),
measure its length by skeletonising it and counting skeleton pixels (geodesic
fibre length, robust to curvature), converted to microns. Then draw a continuous
distribution curve: x = myotube length (um), y = number of myotubes of that size.

Reads the *_myotube_mask.npy files written by myotube_sweep.py.
"""
from __future__ import annotations
import os, glob, json
import numpy as np
from skimage.morphology import skeletonize
from scipy.stats import gaussian_kde
from skan import Skeleton, summarize
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UM = 0.6493          # micron / pixel (confocal, same as pipeline)
BINW = 10.0          # um; curve is scaled to "myotubes per BINW um"
D = "plate23_myotube"


SPUR_UM = 10.0        # terminal branches shorter than this = spurs, pruned
STRAIGHT_DOT = -0.5   # pair branch-ends whose dirs are <-0.5 (angle >120 deg)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def lengths_um(mask):
    """Whole-fibre length (um): trace each myotube THROUGH crossings.

    skan builds the skeleton graph; each branch has a true geodesic length. At
    every junction we pair the two branch-ends that continue straightest (most
    anti-parallel local directions = a fibre passing through the crossing) and
    leave sharp/unmatched ends as fibre terminations. Walking the resulting
    chains gives one length per whole fibre, summed across the crossings it
    passes. Terminal spur branches (<SPUR_UM) are pruned first."""
    skel = skeletonize(mask > 0)
    if skel.sum() < 3:
        return np.array([])
    sk = Skeleton(skel, spacing=UM)
    df = summarize(sk, separator="-")
    dcol = [c for c in df.columns if "branch" in c and "distance" in c][0]
    tcol = [c for c in df.columns if "branch" in c and "type" in c][0]
    scol = [c for c in df.columns if "node" in c and "src" in c][0]
    ecol = [c for c in df.columns if "node" in c and "dst" in c][0]
    L = df[dcol].to_numpy(dtype=np.float64)
    T = df[tcol].to_numpy()
    src = df[scol].to_numpy(); dst = df[ecol].to_numpy()

    G = nx.Graph()
    node_ends = {}                     # node id -> [(branch, end, dir_into_branch)]
    for i in range(len(L)):
        if T[i] == 1 and L[i] < SPUR_UM:            # prune terminal spur
            continue
        G.add_edge(("b", i, "s"), ("b", i, "d"), w=float(L[i]))
        c = np.asarray(sk.path_coordinates(i), dtype=np.float64)
        step = min(3, len(c) - 1)
        ds = _unit(c[step] - c[0]) if len(c) > 1 else np.zeros(2)
        dd = _unit(c[-1 - step] - c[-1]) if len(c) > 1 else np.zeros(2)
        node_ends.setdefault(int(src[i]), []).append((i, "s", ds))
        node_ends.setdefault(int(dst[i]), []).append((i, "d", dd))

    for ends in node_ends.values():                 # straightest-through pairing
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

    out = []
    for comp in nx.connected_components(G):
        tot = sum(d["w"] for _, _, d in G.subgraph(comp).edges(data=True))
        if tot > 0:
            out.append(tot)
    return np.array(out)


def main():
    files = sorted(glob.glob(os.path.join(D, "*_myotube_mask.npy")))
    wells, data = [], {}
    for f in files:
        stem = os.path.basename(f).replace("_myotube_mask.npy", "")
        L = lengths_um(np.load(f))
        L = L[L > 0]
        wells.append(stem)
        data[stem] = L
        print(f"{stem:26s} n={L.size:5d}  median={np.median(L):6.1f}um  "
              f"max={L.max():7.1f}um  >100um={int((L>100).sum())}")

    allL = np.concatenate([data[w] for w in wells])
    xmax = np.percentile(allL, 99.5)
    xs = np.linspace(0, xmax, 500)
    colors = plt.cm.tab10(np.linspace(0, 1, len(wells)))
    # treated wells drawn thick, control dashed+black so it reads as the baseline
    order = sorted(wells, key=lambda w: np.median(data[w]))

    fig, ax = plt.subplots(1, 3, figsize=(24, 7))

    def curve(w):
        L = data[w]
        return gaussian_kde(L)(xs) * L.size * BINW

    # panel 1: the requested distribution — number of myotubes vs length (linear)
    for w, c in zip(wells, colors):
        style = dict(color="k", lw=2.5, ls="--") if "ctrl" in w else dict(color=c, lw=2)
        ax[0].plot(xs, curve(w), label=f"{w.split('_',1)[1]} (n={data[w].size})", **style)
    ax[0].set_title("Length distribution — per well (linear)")
    ax[0].set_ylim(bottom=0)

    # panel 2: same on log-y so the mature-fibre tails separate
    for w, c in zip(wells, colors):
        style = dict(color="k", lw=2.5, ls="--") if "ctrl" in w else dict(color=c, lw=2)
        ax[1].plot(xs, np.clip(curve(w), 0.5, None), **style,
                   label=f"{w.split('_',1)[1]} (n={data[w].size})")
    ax[1].set_yscale("log")
    ax[1].set_title("Same, log y — long-fibre tail visible")

    for a in ax[:2]:
        a.set_xlabel("myotube length (um)")
        a.set_ylabel(f"number of myotubes (per {BINW:.0f} um)")
        a.legend(fontsize=8); a.grid(alpha=0.3, which="both"); a.set_xlim(0, xmax)
        a.axvline(100, ls=":", color="#9ca3af")

    # panel 3: CCDF — how many fibres are AT LEAST X long (the treatment signal)
    xc = np.linspace(0, xmax, 400)
    for w, c in zip(wells, colors):
        L = np.sort(data[w])
        ccdf = L.size - np.searchsorted(L, xc, side="left")
        style = dict(color="k", lw=2.5, ls="--") if "ctrl" in w else dict(color=c, lw=2)
        ax[2].plot(xc, ccdf, **style, label=f"{w.split('_',1)[1]} (>100um: {(data[w]>100).sum()})")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("myotube length (um)")
    ax[2].set_ylabel("number of myotubes >= length")
    ax[2].set_title("Cumulative — # fibres at least this long")
    ax[2].legend(fontsize=8, title="mature fibres/well"); ax[2].grid(alpha=0.3, which="both")
    ax[2].set_xlim(0, xmax); ax[2].axvline(100, ls=":", color="#9ca3af")

    fig.suptitle("Plate 23 — Desmin+ myotube length distribution (whole fibres, traced through crossings)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(D, "myotube_length_distribution.png")
    fig.savefig(out, dpi=120, bbox_inches="tight", facecolor="white")

    stats = {w: {"n": int(data[w].size),
                 "median_um": round(float(np.median(data[w])), 1),
                 "mean_um": round(float(np.mean(data[w])), 1),
                 "max_um": round(float(data[w].max()), 1),
                 "n_over_50um": int((data[w] > 50).sum()),
                 "n_over_100um": int((data[w] > 100).sum())} for w in wells}
    with open(os.path.join(D, "myotube_length_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
