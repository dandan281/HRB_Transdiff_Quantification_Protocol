"""Benchmark nuclei counting: Cellpose-SAM on the BLUE (DAPI) channel of the
25 RGB benchmark TIFFs, with the standard cellprob_threshold plateau sweep.

Input format differs from the plate pipeline: 8-bit RGB ImageJ exports
(1040x1392x3), red empty, green = Desmin, blue = DAPI. DAPI display scaling
varies per image (bg peak 0-87, some saturation) — fine for Cellpose, which
normalizes internally; DAPI is only used to FIND nuclei, never to measure them.

Uniform-hyperparameter rule: ONE cellprob for the whole set, picked as the
GLOBAL plateau of the pooled count-vs-cellprob curve (flattest interior point,
tie toward cp=0). Masks are saved at cp=0 during the sweep; if the global
plateau lands elsewhere a second pass re-evals and overwrites.

Run:  cpenv/Scripts/python.exe bench_nuclei.py
"""
from __future__ import annotations
import os, json, glob, time
import numpy as np
import tifffile

BENCH = r"C:\Users\liqig\Documents\HRB_Transdiff\Benchmark"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nuclei")
CELLPROBS = [-2.0, -1.0, 0.0, 1.0, 2.0]
SAVE_CP = 0.0                       # masks cached at this cp during the sweep


def images():
    files = sorted(glob.glob(os.path.join(BENCH, "*.tif")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    for p in files:
        stem = os.path.splitext(os.path.basename(p))[0]
        yield stem, tifffile.imread(p)[..., 2].astype(np.float32)   # blue = DAPI


def main():
    os.makedirs(OUT, exist_ok=True)
    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    print(f"GPU={gpu}", flush=True)

    sweep = {}                       # stem -> {cp: count}
    t0 = time.time()
    for stem, dapi in images():
        counts = {}
        for cp in CELLPROBS:
            masks, _, _ = model.eval(dapi, cellprob_threshold=cp)
            counts[cp] = int(masks.max())
            if cp == SAVE_CP:
                np.save(os.path.join(OUT, f"{stem}_masks.npy"),
                        masks.astype(np.int32))
        sweep[stem] = counts
        print(f"img {stem:>3s}: " + "  ".join(
            f"cp{cp:+.0f}={n}" for cp, n in counts.items()), flush=True)

    # global plateau on the pooled curve
    totals = [sum(sweep[s][cp] for s in sweep) for cp in CELLPROBS]
    best_i, best_key = None, None
    for i in range(1, len(CELLPROBS) - 1):
        key = (abs(totals[i - 1] - totals[i + 1]), abs(CELLPROBS[i]))
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    op_cp = CELLPROBS[best_i]
    print(f"pooled sweep totals: "
          + "  ".join(f"cp{cp:+.0f}={t}" for cp, t in zip(CELLPROBS, totals)))
    print(f"-> global plateau cellprob = {op_cp:+.1f}", flush=True)

    if op_cp != SAVE_CP:             # re-eval at the chosen operating point
        print(f"re-evaluating all images at cp={op_cp:+.1f}", flush=True)
        for stem, dapi in images():
            masks, _, _ = model.eval(dapi, cellprob_threshold=op_cp)
            np.save(os.path.join(OUT, f"{stem}_masks.npy"),
                    masks.astype(np.int32))

    secs = time.time() - t0
    with open(os.path.join(OUT, "nuclei_sweep.json"), "w") as fh:
        json.dump({"cellprobs": CELLPROBS, "operating_cellprob": op_cp,
                   "pooled_totals": dict(zip(map(str, CELLPROBS), totals)),
                   "per_image": {s: {str(c): n for c, n in d.items()}
                                 for s, d in sweep.items()},
                   "count_at_operating": {s: sweep[s][op_cp] for s in sweep},
                   "seconds": round(secs, 1), "gpu": gpu}, fh, indent=2)
    print(f"NUCLEI_DONE  total@op={sum(sweep[s][op_cp] for s in sweep)}"
          f"  ({secs:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
