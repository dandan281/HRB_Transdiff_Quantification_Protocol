"""v3c -- does the CYTOPLASMIC RING SIZE explain the residual fold compression?

In a dense field a fixed ring around a nucleus can pick up the neighbouring
cell's Desmin. That contaminates negative cells with their positive neighbours'
signal, which inflates the control and COMPRESSES the treated/control ratio --
exactly the residual symptom after the normalisation fix.

If ring size is the culprit, a tight ring (less neighbour bleed) should widen the
fold-change and a wide ring should narrow it. If folds are flat across ring size,
neighbour contamination is not the limiter and the overlap is in the biology or
the staining -- which would mean further tuning is not worth pursuing.

Control is re-anchored at --ctrl-target for every ring size, so the comparison is
like-for-like.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/ring_sweep.py
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from percell_desmin import ring_intensity, CACHE, NUC_DIR, AMIN, AMAX, CTRL  # noqa
from real_fusion import UM, UM2  # noqa: E402

RINGS_UM = [1.5, 3.0, 6.0]     # default; override with --rings

# Result (control anchored at 20%): folds INCREASE monotonically with ring size
# (B06 1.66->1.96, B03 2.08->2.37 over 1.5->6 um). So neighbour bleed is NOT the
# limiter -- the reverse: converted cells have extended cytoplasm, so a wider ring
# samples more of their real Desmin while negative cells stay negative.


def measure(ring_px):
    out = {}
    wells = sorted(f.replace("_dbs.npy", "") for f in os.listdir(CACHE)
                   if f.endswith("_dbs.npy"))
    for w in wells:
        dbs = np.load(os.path.join(CACHE, f"{w}_dbs.npy")).astype(np.float32)
        nuc = np.load(os.path.join(NUC_DIR, f"{w}_masks.npy"))
        area = np.bincount(nuc.ravel(), minlength=int(nuc.max()) + 1) * UM2
        valid = (area >= AMIN) & (area <= AMAX); valid[0] = False
        mean, cnt = ring_intensity(nuc, dbs, ring_px)
        out[w] = mean[valid[:mean.size] & (cnt > 0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-target", type=float, default=20.0)
    ap.add_argument("--rings", default=",".join(str(v) for v in RINGS_UM))
    a = ap.parse_args()
    rings = [float(v) for v in a.rings.split(",")]

    res = {}
    for r_um in rings:
        r_px = max(1, int(round(r_um / UM)))
        cf = os.path.join(HERE, f"percell_values_r{r_um}.npz")
        if os.path.exists(cf):
            z = np.load(cf); V = {k: z[k] for k in z.files}
        else:
            V = measure(r_px)
            np.savez_compressed(cf, **V)
        thr = float(np.percentile(V[CTRL], 100 - a.ctrl_target))
        wells = sorted(V)
        row = {}
        for w in wells:
            if w == CTRL:
                continue
            e = 100 * float((V[w] > thr).mean())
            row[w] = {"conversion_pct": round(e, 2),
                      "fold": round(e / a.ctrl_target, 2)}
        res[r_um] = {"ring_px": r_px, "threshold": round(thr, 1), "wells": row}
        print(f"ring={r_um:>4} um ({r_px:>2} px)  thr={thr:>7.1f}  " +
              "  ".join(f"{w.split('_',1)[1][:9]}={row[w]['fold']:.2f}x"
                        for w in wells if w != CTRL))

    folds = {w: [res[r]["wells"][w]["fold"] for r in rings]
             for w in res[rings[0]]["wells"]}
    print(f"\ncontrol anchored at {a.ctrl_target:.0f}% for every ring size")
    print("fold spread across ring sizes (max-min):")
    for w, f in folds.items():
        print(f"  {w:<24} {min(f):.2f} - {max(f):.2f}   spread={max(f)-min(f):.2f}")

    with open(os.path.join(HERE, "ring_sweep.json"), "w") as fh:
        json.dump({"ctrl_target_pct": a.ctrl_target,
                   "results": {str(k): v for k, v in res.items()}}, fh, indent=2)
    print("\n-> NEW_Quantif/ring_sweep.json")


if __name__ == "__main__":
    main()
