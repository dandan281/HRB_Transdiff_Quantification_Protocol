"""Fragmentation census, cross-plate: is it the crest, the walk, or the domain?

For each GT fibre spine (bootstrap: skeleton of the certified mask; PLATE_32:
the operator's polyline at 1 px density) measure, with the SAME frozen config
the T03 run used:

  walk_cov    fraction of spine within 4 px of any walked path
  pieces      how many walked objects each cover >= max(10 px, 10%) of it
  walk gaps   sizes of uncovered spine runs (the missing middles)
  crest_cov   fraction of spine within 2 px of the NMS crest (field-level:
              could ANY walk have covered it?)
  crest gaps  sizes of crest holes along the spine
  centre p50  raw predicted centre on the spine (field brightness)
  img p50     normalized image intensity on the spine (stain/exposure shift)
  crossing%   fraction of spine flagged crossing at 0.4

If walk_cov << crest_cov the walk is dying on a crest that exists (walk
problem). If crest_cov is low and its holes match the walk gaps, the field
is the limiter (the PLATE_32 diagnosis). If PLATE_23 centre/image stats sit
below the PLATE_32 reference well, the cross-plate domain is the new fact.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in ("PrecisionMyotube", "annotation_tools", "model_labs"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import tifffile
from scipy import ndimage
from scipy.spatial import cKDTree
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
COVER_R = 4.0


def spine_points_bootstrap(well):
    s = InstanceSet.load(EVAL_GT / f"{well}.eval_gt.instances.json")
    spines = []
    for record, bbox, mask in s.cropped_masks():
        sk = skeletonize(mask)
        rr, cc = np.nonzero(sk)
        if len(rr) >= 5:
            spines.append(np.stack([rr + bbox[0], cc + bbox[1]], 1))
    return spines


def spine_points_plate32(well):
    from tracer_lab.train_tracer import load_well
    image, gt, _ = load_well(well)
    spines = []
    for t in gt["traces"]:
        r, c = polyline_pixels(np.asarray(t, dtype=float))
        spines.append(np.stack([np.round(r), np.round(c)], 1).astype(int))
    return image, spines


def run_well(name, image, spines, ckpt):
    pred = predict_fields(image, ckpt)
    wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                         prep="nms")
    res = trace_field(wf, TraceParams(**WALK))

    # dense walked points, tagged by final object
    pts, oids = [], []
    for pid, path in enumerate(res["paths"], start=1):
        p = np.asarray(path, dtype=float)
        if p.ndim != 2 or p.shape[0] < 2:
            continue
        r, c = polyline_pixels(p)
        pts.append(np.stack([r, c], 1))
        oids.append(np.full(len(r), res["object_of"][pid]))
    P = np.concatenate(pts)
    O = np.concatenate(oids)
    tree = cKDTree(P)

    H, W = image.shape
    crest = wf["centre"] > 0
    crest_near = ndimage.maximum_filter(crest.astype(np.uint8), size=5)
    xing = pred["crossing"] >= 0.4
    xing_near = ndimage.maximum_filter(xing.astype(np.uint8), size=5)

    rows = []
    gap_walk_all, gap_crest_all = [], []
    for S in spines:
        Si = np.clip(S.astype(int), 0, [H - 1, W - 1])
        n = len(Si)
        near = tree.query_ball_point(Si.astype(float), r=COVER_R)
        covered = np.array([len(ix) > 0 for ix in near])
        cnt = Counter()
        for ix in near:
            for o in set(O[ix]):
                cnt[o] += 1
        need = max(10, 0.10 * n)
        pieces = sum(1 for k in cnt.values() if k >= need)
        ccov = crest_near[Si[:, 0], Si[:, 1]].astype(bool)

        def gap_sizes(cov):
            canvas_r = Si[:, 0] - Si[:, 0].min()
            canvas_c = Si[:, 1] - Si[:, 1].min()
            canvas = np.zeros((canvas_r.max() + 1, canvas_c.max() + 1),
                              dtype=bool)
            canvas[canvas_r[~cov], canvas_c[~cov]] = True
            lab, k = ndimage.label(canvas, structure=np.ones((3, 3)))
            return [int(v) for v in
                    ndimage.sum(canvas, lab, range(1, k + 1))]

        gw = gap_sizes(covered) if not covered.all() else []
        gc = gap_sizes(ccov) if not ccov.all() else []
        gap_walk_all += gw
        gap_crest_all += gc
        rows.append({
            "n_px": n,
            "walk_cov": float(covered.mean()),
            "crest_cov": float(ccov.mean()),
            "pieces": int(pieces),
            "centre_p50": float(np.median(pred["centre"][Si[:, 0], Si[:, 1]])),
            "img_p50": float(np.median(image[Si[:, 0], Si[:, 1]])),
            "crossing_frac": float(xing_near[Si[:, 0], Si[:, 1]].mean()),
        })

    r = {k: np.array([x[k] for x in rows]) for k in rows[0]}
    frag = r["pieces"]
    out = {
        "well": name,
        "n_gt": len(rows),
        "walk_cov_mean": float(r["walk_cov"].mean()),
        "crest_cov_mean": float(r["crest_cov"].mean()),
        "pieces_hist": {str(k): int((frag == k).sum()) for k in range(0, 4)}
        | {"4+": int((frag >= 4).sum())},
        "frac_fragmented(pieces>=2)": float((frag >= 2).mean()),
        "walk_gap_px_p50_p75_p90": [float(np.percentile(gap_walk_all, q))
                                    for q in (50, 75, 90)] if gap_walk_all
        else None,
        "crest_gap_px_p50_p75_p90": [float(np.percentile(gap_crest_all, q))
                                     for q in (50, 75, 90)] if gap_crest_all
        else None,
        "n_walk_gaps": len(gap_walk_all),
        "n_crest_gaps": len(gap_crest_all),
        "centre_on_spine_p50": float(np.median(r["centre_p50"])),
        "img_on_spine_p50": float(np.median(r["img_p50"])),
        "crossing_frac_mean": float(r["crossing_frac"].mean()),
        "walk_cov_p25": float(np.percentile(r["walk_cov"], 25)),
        "crest_cov_p25": float(np.percentile(r["crest_cov"], 25)),
    }
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
        print(f"\n=== {well} (PLATE_23, fold-B02 ckpt) ===", flush=True)
        results.append(run_well(well, norm, spine_points_bootstrap(well),
                                CV / "B02" / "best.pt"))

    print("\n=== C05 (PLATE_32 reference, fold-C05 ckpt, never-seen) ===",
          flush=True)
    image, spines = spine_points_plate32("C05")
    results.append(run_well("C05_plate32", image, spines,
                            CV / "C05" / "best.pt"))

    out = ROOT / "model_labs/tracer_lab/_runs/eval_bootstrap_v1/frag_diag.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
