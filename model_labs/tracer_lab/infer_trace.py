"""Predict the three fields on a held-out well, trace them, score them.

The third leg of the lane: `oracle_trace` proved the walk on perfect fields;
`train_tracer` produced a checkpoint; this runs the SAME walk -- same code,
same frozen plateau parameters -- on the *predicted* fields, and scores with
the same polyline metrics against the same GT. The three numbers to read
side by side per well:

    classical floor   0.3169 length_mdape (bootstrap_v1, context only)
    trained candidate (this script)
    oracle ceiling    (oracle_trace on the same well)

The trained-to-oracle gap is what a better network can still win; the
oracle-to-truth gap is what it never can.

The predicted orientation is normalised to unit length before tracing (the
walk decodes angles, so only direction matters), and `orient_valid` -- a
TARGET-side construct -- is derived for the walk as "predicted centre support
AND not predicted crossing", since at inference nothing else exists.

    python model_labs/tracer_lab/infer_trace.py --well B02 \
        --ckpt model_labs/tracer_lab/_runs/net_v1/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def predict_fields(image: np.ndarray, ckpt_path: Path, *, tile=512,
                   overlap=64, device=None) -> dict:
    """Sliding-window inference; overlapping strips are averaged.

    Returns centre (prob-like, clipped to [0,1]), orient (unit), crossing
    (sigmoid prob) -- full field, float32.
    """
    import torch
    from tracer_lab.net import TracerNet

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TracerNet(base=ck.get("base", 32)).to(device)
    # Non-strict: checkpoints from before the offset head exist and must stay
    # comparable. Missing heads are reported rather than silently ignored --
    # a head absent from the weights is untrained, and any field derived from
    # it is noise.
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    heads_missing = sorted({m.split(".")[0] for m in missing
                            if m.startswith("head_")})
    if heads_missing:
        print(f"  note: {ckpt_path.parent.name} has no {heads_missing} -- "
              "those outputs are untrained noise, do not use them")
    model.eval()

    H, W = image.shape
    out = {"centre": np.zeros((H, W), np.float32),
           "orient": np.zeros((2, H, W), np.float32),
           "crossing": np.zeros((H, W), np.float32)}
    weight = np.zeros((H, W), np.float32)
    step = tile - overlap
    rs = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    cs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if rs[-1] + tile < H:
        rs.append(H - tile)
    if cs[-1] + tile < W:
        cs.append(W - tile)

    with torch.no_grad():
        for r in rs:
            for c in cs:
                patch = image[r:r + tile, c:c + tile]
                x = torch.from_numpy(patch[None, None]).to(device)
                y = model(x)
                out["centre"][r:r + tile, c:c + tile] += \
                    y["centre"][0, 0].cpu().numpy()
                out["orient"][:, r:r + tile, c:c + tile] += \
                    y["orient"][0].cpu().numpy()
                out["crossing"][r:r + tile, c:c + tile] += \
                    torch.sigmoid(y["crossing"][0, 0]).cpu().numpy()
                if "offset" in y:
                    if "offset" not in out:
                        out["offset"] = np.zeros((2, H, W), np.float32)
                    out["offset"][:, r:r + tile, c:c + tile] += \
                        y["offset"][0].cpu().numpy()
                weight[r:r + tile, c:c + tile] += 1.0
    if "offset" in out:
        out["offset"] = (out["offset"] / weight) * OFFSET_BAND_PX
    out["centre"] = np.clip(out["centre"] / weight, 0.0, 1.0)
    out["crossing"] = out["crossing"] / weight
    o = out["orient"] / weight
    norm = np.sqrt((o ** 2).sum(0)).clip(min=1e-6)
    out["orient"] = (o / norm).astype(np.float32)
    return out


OFFSET_BAND_PX = 12.0


def centre_from_offset(pred: dict, sigma: float = 2.0) -> np.ndarray:
    """Reconstruct a SHARP centreline map from the predicted offset field.

    The offset head predicts, per pixel, the displacement to its nearest
    centreline point. Two independent signals say "a centreline is here":

    * ``|offset| ~ 0`` -- the pixel is already on it;
    * the offset field **converges** there -- neighbours on both sides point
      inward, so its divergence is strongly negative. Convergence is the
      sharp one: it survives even when the magnitudes are systematically off,
      because it depends on the field's shape rather than its scale.

    Both are computed from predictions only. The magnitude term is turned into
    the same ``exp(-d^2/2 sigma^2)`` shape the target used, so thresholds keep
    their meaning across the two field preparations.
    """
    from scipy import ndimage

    off = pred["offset"]
    mag = np.sqrt((off ** 2).sum(0))
    by_mag = np.exp(-(mag ** 2) / (2.0 * sigma ** 2))

    div = (np.gradient(off[0], axis=0) + np.gradient(off[1], axis=1))
    conv = np.clip(-ndimage.gaussian_filter(div, 1.0), 0.0, None)
    hi = np.percentile(conv[conv > 0], 99.0) if (conv > 0).any() else 1.0
    by_conv = np.clip(conv / max(hi, 1e-9), 0.0, 1.0)

    return (by_mag * by_conv).astype(np.float32)


def steerable_ridge(pred: dict, *, smooth: float = 1.5,
                    floor: float = 0.30) -> np.ndarray:
    """Sharpen the centre map into a crest, steered by the orientation head.

    Measured on net_v3 (D04 and B02, ~20k perpendicular cuts): the predicted
    centre peaks in the right place but is **12 px FWHM against a 4 px
    target**, with 0.08 of contrast between crest and 8 px off-fibre. MSE
    against a peaked target under positional uncertainty pays for a wide low
    bump, so no threshold on that map can find a centreline -- which is
    exactly what both sweeps showed.

    The orientation head, however, is good (7 deg median axial error). So the
    crest is recovered analytically instead of by more training: the second
    derivative of the centre map ALONG THE FIBRE NORMAL is strongly negative
    on a ridge and ~0 on a plateau, and the normal comes from the orientation
    field. This is a steerable ridge filter with the steering supplied by a
    head that already works.

    Returns a [0,1] ridge score, zero where the response is not a crest.
    """
    from scipy import ndimage

    c = ndimage.gaussian_filter(pred["centre"], smooth)
    theta = 0.5 * np.arctan2(pred["orient"][1], pred["orient"][0])
    # tangent (sin t, cos t) in (row, col); normal is perpendicular to it
    nr, nc = np.cos(theta), -np.sin(theta)

    grr = ndimage.gaussian_filter(pred["centre"], smooth, order=(2, 0))
    gcc = ndimage.gaussian_filter(pred["centre"], smooth, order=(0, 2))
    grc = ndimage.gaussian_filter(pred["centre"], smooth, order=(1, 1))
    d2n = nr * nr * grr + 2.0 * nr * nc * grc + nc * nc * gcc

    ridge = np.clip(-d2n, 0.0, None)
    hi = np.percentile(ridge[ridge > 0], 99.5) if (ridge > 0).any() else 1.0
    ridge = np.clip(ridge / max(hi, 1e-9), 0.0, 1.0)
    # a crest of the WRONG thing is still not a fibre: require the underlying
    # centre map to be at least `floor`, which the halo satisfies but glass
    # does not
    return (ridge * (c >= floor)).astype(np.float32)


def fields_for_walk(pred: dict, *, crossing_thresh: float,
                    valid_thresh: float, nms_radius: float = 2.5,
                    nms_tol: float = 0.02, ridge_floor: float = 0.15,
                    prep: str = "nms") -> dict:
    """Assemble the dict `trace_field` expects, from predictions alone.

    The centre field is non-max suppressed first. Measured on net_v2 / D04:
    on-ridge p50 0.57 against an off-ridge p99 of 0.65 -- the distributions
    overlap, so NO absolute threshold separates fibre from halo, and the first
    sweep's walks roamed the halo into objects 45x too long. What separates
    them is shape: a ridge is a LOCAL MAXIMUM within `nms_radius`, a halo is a
    slope. Pixels not within `nms_tol` of their local max (or under
    `ridge_floor` outright) are zeroed, so the walk's support/seed thresholds
    act on crest membership rather than raw brightness. Predictions only --
    nothing target-side is consumed.
    """
    from scipy import ndimage

    if prep == "offset":
        centre_o = centre_from_offset(pred)
        crossing = pred["crossing"] >= crossing_thresh
        valid = (centre_o >= valid_thresh) & ~crossing
        return {"centre": centre_o, "orient": pred["orient"],
                "crossing": crossing, "orient_valid": valid}

    if prep == "steer":
        centre_s = steerable_ridge(pred)
        crossing = pred["crossing"] >= crossing_thresh
        valid = (centre_s >= valid_thresh) & ~crossing
        return {"centre": centre_s, "orient": pred["orient"],
                "crossing": crossing, "orient_valid": valid}

    centre = pred["centre"]
    r = int(np.ceil(nms_radius))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (yy ** 2 + xx ** 2) <= nms_radius ** 2
    crest = (centre >= ndimage.maximum_filter(centre, footprint=disc)
             - nms_tol) & (centre >= ridge_floor)
    # one-pixel dilation keeps the corridor wide enough to walk with the
    # lateral snap; without it the crest is a 1 px thread the 3 px steps miss
    crest = ndimage.binary_dilation(
        crest, np.ones((3, 3), dtype=bool))
    centre_nms = np.where(crest, centre, 0.0).astype(np.float32)

    crossing = pred["crossing"] >= crossing_thresh
    valid = crest & (centre_nms >= valid_thresh) & ~crossing
    return {"centre": centre_nms, "orient": pred["orient"],
            "crossing": crossing, "orient_valid": valid}


def main(argv=None) -> int:
    from tracer_lab.train_tracer import load_well
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well", default="B02")
    ap.add_argument("--ckpt", default="model_labs/tracer_lab/_runs/net_v1/best.pt")
    ap.add_argument("--crossing-thresh", type=float, default=0.5)
    ap.add_argument("--valid-thresh", type=float, default=0.3)
    ap.add_argument("--seed-thresh", type=float, default=None,
                    help="override the walk's seed threshold (predicted "
                         "centre is dimmer than a perfect Gaussian)")
    ap.add_argument("--prep", default="nms",
                    choices=("nms", "steer", "offset"))
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs/net_v1")
    a = ap.parse_args(argv)

    t0 = time.time()
    image, gt_fields, meta = load_well(a.well)
    t1 = time.time()
    pred = predict_fields(image, ROOT / a.ckpt)
    t2 = time.time()

    walk_fields = fields_for_walk(
        pred, crossing_thresh=a.crossing_thresh, valid_thresh=a.valid_thresh,
        prep=a.prep)
    # scoring needs the GT lookup fields; the walk never reads them
    walk_fields["instance"] = gt_fields["instance"]
    walk_fields["traces"] = gt_fields["traces"]

    overrides = {} if a.seed_thresh is None else {"seed_thresh": a.seed_thresh}
    prm = TraceParams(**overrides)
    res = trace_field(walk_fields, prm)
    t3 = time.time()
    sc = score_against_gt(res, walk_fields)
    sc["well"] = a.well
    sc["ckpt"] = str(a.ckpt)
    sc["thresholds"] = {"crossing": a.crossing_thresh,
                        "valid": a.valid_thresh,
                        "seed": prm.seed_thresh, "prep": a.prep}
    sc["params"] = prm.to_dict()
    sc["timing_s"] = {"load": round(t1 - t0, 1),
                      "predict": round(t2 - t1, 1),
                      "trace+score": round(t3 - t2, 1)}

    # field-quality diagnostics, separate from the walk outcome: if the walk
    # fails, these say whether the fields or the walk failed
    gc = gt_fields["centre"]
    sc["field_diag"] = {
        "centre_mae": float(np.abs(pred["centre"] - gc).mean()),
        "centre_on_ridge_mean": float(pred["centre"][gc >= 0.9].mean()),
        "centre_off_ridge_mean": float(pred["centre"][gc < 0.1].mean()),
        "crossing_recall@t": float(
            (pred["crossing"] >= a.crossing_thresh)[gt_fields["crossing"]]
            .mean()),
    }

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rec = out / f"trace_{a.well}.json"
    rec.write_text(json.dumps(sc, indent=2))
    print(json.dumps(sc, indent=2))
    print(f"\nwritten: {rec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
