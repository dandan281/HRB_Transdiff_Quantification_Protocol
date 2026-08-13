"""Count nuclei in one well with Cellpose-SAM, using a cellprob_threshold sweep
and picking the PLATEAU (operating point where the count is least sensitive to
the knob) -- the Cellpose-native version of the user's 'adjust until the count
barely changes' rule.

Runs in the cpenv (GPU). Writes per-well viz + appends one line to a jsonl.
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import nd2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.color import label2rgb


def stretch(a, p=99.5):
    lo, hi = np.percentile(a, 1), np.percentile(a, p)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2", required=True)
    ap.add_argument("--nuclei-ch", type=int, default=2)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cellprobs", default="-2,-1,0,1,2")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.nd2))[0]
    cps = [float(v) for v in a.cellprobs.split(",")]

    with nd2.ND2File(a.nd2) as x:                     # keep DAPI channel only
        dapi = x.asarray()[a.nuclei_ch].astype(np.float32)

    import torch, time
    from cellpose import models, utils
    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    counts, mask_by_cp, t0 = [], {}, time.time()
    for cp in cps:                                   # the sweep
        masks, _, _ = model.eval(dapi, cellprob_threshold=cp)
        counts.append(int(masks.max()))
        mask_by_cp[cp] = masks
        print(f"  cellprob={cp:+.1f}  nuclei={counts[-1]}")
    secs = time.time() - t0

    # plateau = interior sweep point where neighbours differ least (flattest);
    # tie-break toward cellprob nearest 0 (Cellpose default).
    c = np.array(counts)
    best_i, best_key = None, None
    for i in range(1, len(cps) - 1):
        sens = abs(int(c[i - 1]) - int(c[i + 1]))
        key = (sens, abs(cps[i]))
        if best_key is None or key < best_key:
            best_key, best_i = key, i
    if best_i is None:                               # <3 points -> default cp=0
        best_i = int(np.argmin([abs(v) for v in cps]))
    op_cp, op_count = cps[best_i], int(c[best_i])
    print(f"  -> plateau cellprob={op_cp:+.1f}  count={op_count}  ({secs:.1f}s)")

    # ---- viz 1: sweep curve with the chosen plateau marked
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cps, counts, "-o", color="#3b82f6")
    ax.plot(op_cp, op_count, "*", ms=22, color="#ef4444",
            label=f"plateau: cp={op_cp:+.0f}, n={op_count}")
    ax.set_xlabel("cellprob_threshold  (raise = stricter = fewer)")
    ax.set_ylabel("nuclei counted")
    ax.set_title(f"{stem}\nnucleus-count sweep")
    ax.legend(); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(a.outdir, f"{stem}_sweep.png"), dpi=120,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- viz 2: labeled overlay at the chosen operating point
    masks = mask_by_cp[op_cp]
    raw = stretch(dapi)
    over = label2rgb(masks, image=raw, bg_label=0, alpha=0.5, image_alpha=1)
    from PIL import Image
    im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
    im.thumbnail((1800, 1800))                       # keep file size sane
    im.save(os.path.join(a.outdir, f"{stem}_labeled.png"))
    np.save(os.path.join(a.outdir, f"{stem}_masks.npy"), masks.astype(np.int32))

    rec = {"well": stem, "operating_cellprob": op_cp, "nuclei": op_count,
           "sweep": dict(zip([f"{v:+.0f}" for v in cps], counts)),
           "seconds": round(secs, 1), "gpu": gpu}
    # Re-running a well must replace its old record, not append a second one --
    # the summary scripts sum every line, so duplicates double-count the plate.
    path = os.path.join(a.outdir, "plate_results.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
    rows = [r for r in rows if r.get("well") != rec["well"]] + [rec]
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print("WELL_DONE", stem, op_count)


if __name__ == "__main__":
    main()
