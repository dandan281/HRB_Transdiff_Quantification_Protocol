"""Synthesize (wrong-proposal -> correct-mask) training pairs from reviewed-complete fibres.

The real correction pairs are dominated by the `too_short` error (machine under-traces
fibre length). We have 377 human-confirmed *complete* fibres; each one is a free source of
labeled length-error examples:

  * TRUNCATE  -> chop a fraction of the fibre length along its principal axis. The chopped
                 mask is a fake machine "proposal"; the full mask is the target. (too_short)
  * MERGE     -> union two nearby complete fibres into one blob. The union is a fake
                 over-merged proposal; the two separate fibres are the target. (over_merge)

Output matches the `corrections/` npz layout so real + synthetic pools train together, with
the 40 REAL pairs held out for validation.

    python model_labs/synth_length_errors.py --instances <well>.qc.instances.json \
        --package <annotation_work/well> --out <annotation_work/synth> [--per-fibre 6]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _read_tif(path: Path):
    import tifffile
    return np.asarray(tifffile.imread(str(path)))


def _norm8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def _axis_projection(mask: np.ndarray):
    """Return (t, tmin, tmax) where t is each fg pixel's coord along the principal axis.
    Closed-form 2x2 eigenvector — no LAPACK (avoids the delay-load DLL fault)."""
    ys, xs = np.nonzero(mask)
    x = xs.astype(np.float64); y = ys.astype(np.float64)
    x -= x.mean(); y -= y.mean()
    a = (x * x).mean(); b = (x * y).mean(); d = (y * y).mean()
    theta = 0.5 * math.atan2(2 * b, a - d)
    t = x * math.cos(theta) + y * math.sin(theta)
    return ys, xs, t


# (keep_fraction, which_end): deterministic variant set for reproducibility
_VARIANTS = [(0.55, "hi"), (0.6, "lo"), (0.7, "hi"), (0.45, "mid"), (0.65, "mid"), (0.8, "lo")]


def _truncate(mask: np.ndarray, keep: float, end: str) -> np.ndarray:
    ys, xs, t = _axis_projection(mask)
    tmin, tmax = t.min(), t.max()
    L = tmax - tmin
    if end == "hi":
        sel = t <= tmin + keep * L
    elif end == "lo":
        sel = t >= tmax - keep * L
    else:  # keep the middle `keep` fraction, chop both tips
        pad = (1 - keep) / 2 * L
        sel = (t >= tmin + pad) & (t <= tmax - pad)
    out = np.zeros_like(mask)
    out[ys[sel], xs[sel]] = True
    return out


def _crop_window(bbox, H, W, pad_frac=0.25):
    r0, c0, r1, c1 = bbox
    ph = int((r1 - r0) * pad_frac) + 2
    pw = int((c1 - c0) * pad_frac) + 2
    return max(0, r0 - ph), max(0, c0 - pw), min(H, r1 + ph), min(W, c1 + pw)


def _fit(arr, max_dim, order):
    from PIL import Image
    h, w = arr.shape[:2]
    if max(h, w) <= max_dim:
        return arr
    s = max_dim / max(h, w)
    im = Image.fromarray(arr)
    return np.asarray(im.resize((max(1, int(w * s)), max(1, int(h * s))), order))


def main(argv=None):
    from PIL import Image
    from _shared.schema_bridge import InstanceSet   # noqa: E402  (model_labs on path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-fibre", type=int, default=6)
    ap.add_argument("--max-dim", type=int, default=340)
    ap.add_argument("--merge-dist", type=float, default=140.0)
    args = ap.parse_args(argv)

    pkg = Path(args.package)
    stem = pkg.name
    readme = json.loads((pkg / "README.json").read_text()) if (pkg / "README.json").is_file() else {}
    stem = readme.get("image_id", stem)
    fiber = _read_tif(pkg / "fiber_raw16.tif")
    dapi = _read_tif(pkg / "dapi_raw16.tif") if (pkg / "dapi_raw16.tif").is_file() else None
    inst = InstanceSet.load(args.instances)
    H, W = inst.image_shape
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    completes = [(rec, bbox, crop) for rec, bbox, crop in inst.cropped_masks()
                 if rec.reviewed and rec.status == "complete"]

    rows, n_trunc, n_merge = [], 0, 0

    # --- truncation (too_short) ---
    for rec, bbox, crop in completes:
        full = crop
        if full.sum() < 60:
            continue
        r0, c0, r1, c1 = _crop_window(bbox, H, W)
        fib = _fit(_norm8(fiber[r0:r1, c0:c1]), args.max_dim, Image.BILINEAR)
        dap = _fit(_norm8(dapi[r0:r1, c0:c1]), args.max_dim, Image.BILINEAR) if dapi is not None else None
        # place the fibre mask into the padded window
        wfull = np.zeros((r1 - r0, c1 - c0), bool)
        wfull[bbox[0] - r0:bbox[2] - r0, bbox[1] - c0:bbox[3] - c0] = full
        for keep, end in _VARIANTS[:args.per_fibre]:
            prop = _truncate(wfull, keep, end)
            if prop.sum() < 40 or prop.sum() >= wfull.sum() * 0.97:
                continue
            tgt8 = _fit((wfull.astype(np.uint8)), args.max_dim, Image.NEAREST)
            prop8 = _fit((prop.astype(np.uint8)), args.max_dim, Image.NEAREST)
            name = f"{stem}__{rec.id}__trunc_{end}{int(keep*100)}.npz"
            payload = {"fiber": fib, "proposal": prop8, "corrected": tgt8}
            if dap is not None:
                payload["dapi"] = dap
            np.savez_compressed(out / name, **payload)
            rows.append({"stem": stem, "id": rec.id, "kind": "trunc", "reason": "too_short",
                         "keep_frac": keep, "end": end, "synthetic": True,
                         "added_px": int((wfull & ~prop).sum()), "removed_px": 0, "n_labels": 1,
                         "npz": name})
            n_trunc += 1

    # --- merge (over_merge / too_long) ---
    cents = [(np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), rec, b, c) for rec, b, c in completes]
    used = set()
    for i, (ci, ri, bi, cri) in enumerate(cents):
        if ri.id in used:
            continue
        for j in range(i + 1, len(cents)):
            cj, rj, bj, crj = cents[j]
            if rj.id in used:
                continue
            if np.linalg.norm(ci - cj) > args.merge_dist:
                continue
            r0 = min(bi[0], bj[0]); c0 = min(bi[1], bj[1]); r1 = max(bi[2], bj[2]); c1 = max(bi[3], bj[3])
            r0, c0, r1, c1 = _crop_window((r0, c0, r1, c1), H, W)
            if max(r1 - r0, c1 - c0) > 700:
                continue
            lab = np.zeros((r1 - r0, c1 - c0), np.uint8)
            lab[bi[0] - r0:bi[2] - r0, bi[1] - c0:bi[3] - c0][cri] = 1
            lab[bj[0] - r0:bj[2] - r0, bj[1] - c0:bj[3] - c0][crj] = 2
            prop = (lab > 0)
            fib = _fit(_norm8(fiber[r0:r1, c0:c1]), args.max_dim, Image.BILINEAR)
            dap = _fit(_norm8(dapi[r0:r1, c0:c1]), args.max_dim, Image.BILINEAR) if dapi is not None else None
            tgt8 = _fit(lab, args.max_dim, Image.NEAREST)
            prop8 = _fit(prop.astype(np.uint8), args.max_dim, Image.NEAREST)
            name = f"{stem}__{ri.id}_{rj.id}__merge.npz"
            payload = {"fiber": fib, "proposal": prop8, "corrected": tgt8}
            if dap is not None:
                payload["dapi"] = dap
            np.savez_compressed(out / name, **payload)
            rows.append({"stem": stem, "id": f"{ri.id}+{rj.id}", "kind": "merge", "reason": "over_merge",
                         "added_px": 0, "removed_px": 0, "n_labels": 2, "synthetic": True, "npz": name})
            used.add(ri.id); used.add(rj.id); n_merge += 1
            break

    (out / f"{stem}.synth.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(json.dumps({"out": str(out), "complete_sources": len(completes),
                      "truncation_pairs": n_trunc, "merge_pairs": n_merge, "total": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
