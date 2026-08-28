"""Attribute every unwalked GT-spine pixel to one of three mechanisms.

  field_hole   no NMS crest within 2 px — the field never proposed a ridge
  unseedable   crest exists, but its connected crest component's max centre
               is below seed_thresh 0.4 — no walk can START there; the
               segment is invisible at the frozen seed threshold even though
               support (0.3) would have carried a walk across it
  walk_fail    crest exists and its component contains seed-level pixels —
               a walk could have started and covered it, and did not

The split between the three decides where the fix lives: the network, the
seeding rule, or the walk mechanics.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("PrecisionMyotube", "annotation_tools", "model_labs"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import tifffile
from scipy import ndimage
from skimage.morphology import skeletonize

from annotation_tools.relabel.raster import polyline_pixels
from precision_myotube.schema import InstanceSet
from tracer_lab.infer_trace import predict_fields, fields_for_walk
from tracer_lab.oracle_trace import TraceParams, trace_field

BOOT = ROOT / "PrecisionMyotube/annotation_work/bootstrap_v1"
EVAL_GT = ROOT / "model_labs/tracer_lab/_runs/eval_bootstrap_v1/eval_gt"
CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"
WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)


def spines_bootstrap(well):
    s = InstanceSet.load(EVAL_GT / f"{well}.eval_gt.instances.json")
    out = []
    for record, bbox, mask in s.cropped_masks():
        sk = skeletonize(mask)
        rr, cc = np.nonzero(sk)
        if len(rr) >= 5:
            out.append(np.stack([rr + bbox[0], cc + bbox[1]], 1))
    return out


def run(name, image, spines, ckpt):
    pred = predict_fields(image, ckpt)
    wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                         prep="nms")
    res = trace_field(wf, TraceParams(**WALK))
    H, W = image.shape

    walked = np.zeros((H, W), dtype=bool)
    for path in res["paths"]:
        p = np.asarray(path, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2:
            continue
        r, c = polyline_pixels(p)
        walked[np.clip(np.round(r).astype(int), 0, H - 1),
               np.clip(np.round(c).astype(int), 0, W - 1)] = True
    yy, xx = np.ogrid[-4:5, -4:5]
    disc = (yy ** 2 + xx ** 2) <= 16
    walked_near = ndimage.binary_dilation(walked, disc)

    crest = wf["centre"] > 0
    crest_near = ndimage.maximum_filter(crest.astype(np.uint8), size=5) \
        .astype(bool)
    lab, ncomp = ndimage.label(crest, structure=np.ones((3, 3)))
    comp_max = np.zeros(ncomp + 1, np.float32)
    mx = ndimage.maximum(wf["centre"], lab, range(1, ncomp + 1))
    comp_max[1:] = np.atleast_1d(mx)
    # nearest crest component per pixel (for spine px just OFF the crest)
    _, (ir, ic) = ndimage.distance_transform_edt(~crest, return_indices=True)
    near_lab = lab[ir, ic]
    seedable = comp_max >= WALK["seed_thresh"]

    tot = unw = hole = unseed = wfail = 0
    for S in spines:
        Si = np.clip(S.astype(int), 0, [H - 1, W - 1])
        w = walked_near[Si[:, 0], Si[:, 1]]
        cn = crest_near[Si[:, 0], Si[:, 1]]
        cl = near_lab[Si[:, 0], Si[:, 1]]
        tot += len(Si)
        u = ~w
        unw += int(u.sum())
        hole += int((u & ~cn).sum())
        on_crest_u = u & cn
        unseed += int((on_crest_u & ~seedable[cl]).sum())
        wfail += int((on_crest_u & seedable[cl]).sum())

    out = {"well": name, "spine_px": tot,
           "unwalked_frac": round(unw / tot, 4),
           "of_unwalked": {
               "field_hole": round(hole / max(unw, 1), 4),
               "unseedable_crest": round(unseed / max(unw, 1), 4),
               "walk_fail_seedable": round(wfail / max(unw, 1), 4)},
           "n_crest_components": int(ncomp),
           "frac_components_seedable": round(float(seedable[1:].mean()), 4)}
    print(json.dumps(out, indent=2), flush=True)
    return out


def main():
    results = []
    for well in ("23_B02_ctrl", "19_B06_act104_trka"):
        img = tifffile.imread(BOOT / well / "image_fiber.tif") \
            .astype(np.float32)
        lo, hi = np.percentile(img, [1.0, 99.9])
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) \
            .astype(np.float32)
        print(f"\n=== {well} ===", flush=True)
        results.append(run(well, norm, spines_bootstrap(well),
                           CV / "B02" / "best.pt"))

    from tracer_lab.train_tracer import load_well
    image, gt, _ = load_well("C05")
    spines = []
    for t in gt["traces"]:
        r, c = polyline_pixels(np.asarray(t, dtype=float))
        spines.append(np.stack([np.round(r), np.round(c)], 1).astype(int))
    print("\n=== C05 (PLATE_32 reference) ===", flush=True)
    results.append(run("C05_plate32", image, spines, CV / "C05" / "best.pt"))

    out = ROOT / ("model_labs/tracer_lab/_runs/eval_bootstrap_v1/"
                  "unwalked_attribution.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
