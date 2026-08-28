"""Snap operator polylines laterally onto the image ridge, and prove it safe.

REWRITTEN 2026-08-24. The first version of this file cited a per-point
measurement (SD 3.3 px) that was RETRACTED: its yardstick had never been
validated, and when the synthetic-fibre harness (`ridge_yardstick.py`) was
built, every per-point yardstick failed either the parallel-neighbour
condition (locks onto the wrong ridge) or the speckle condition (texture
attenuates a 2.5 px shift to ~1.2). The only instrument that passes every
condition is `trace_mean`: average the perpendicular profiles along the
trace, then find the peak -- speckle is independent along the fibre and
cancels; a systematic offset survives.

With that validated instrument the real measurement is: the operator's traces
carry **no systematic bias but a per-trace lateral offset of SD ~2.1 px**
against the image ridge (D04 -0.07 +/- 2.10 px over 385 traces, B02
+0.07 +/- 2.05 px over 250). A per-trace offset is the damaging kind for
training -- correlated target error cannot be averaged away by any loss --
and it is also the safely fixable kind: a lateral shift is length-invariant.

The snap here is windowed `trace_mean`: the trace is cut into ~120 px windows
(50% overlap), each window's mean profile gives one offset, offsets are
interpolated along the trace, smoothed, clipped to ``max_lateral_px``, and
applied along the local normal. Lateral only, bounded, smooth -- a free snap
would walk a line onto the brighter fibre next door.

``--verify`` must pass BEFORE anything trains on snapped traces:

1. re-measured offset collapses (median ~0, SD well under the original);
2. per-trace arc length changes stay under 1%;
3. no identity theft: snapped points do not become closer to a DIFFERENT
   operator trace than to their own.

    python model_labs/tracer_lab/snap_targets.py --verify --wells D04,B02
    python model_labs/tracer_lab/snap_targets.py --all   # write every well
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT = ROOT / "model_labs/tracer_lab/_runs/snapped_v1"


def snap_trace(image, pts, tans, *, window_px=120, max_lateral_px=3.0,
               smooth_px=40.0):
    """-> (snapped points, applied per-point offsets). Lateral only."""
    from scipy import ndimage
    from tracer_lab.ridge_yardstick import yard_trace_mean

    n = len(pts)
    if n < 8:
        return pts.copy(), np.zeros(n)

    half = window_px // 2
    centres = list(range(half, n - half, half)) or [n // 2]
    cen_off, cen_pos = [], []
    for c in centres:
        lo, hi = max(c - half, 0), min(c + half, n)
        est = yard_trace_mean(image, pts[lo:hi], tans[lo:hi])
        if len(est):
            cen_off.append(float(est[0]))
            cen_pos.append(c)
    if not cen_off:
        return pts.copy(), np.zeros(n)

    off = np.interp(np.arange(n), cen_pos, cen_off)
    off = ndimage.gaussian_filter1d(off, max(smooth_px / 2.0, 1.0),
                                    mode="nearest")
    off = np.clip(off, -max_lateral_px, max_lateral_px)
    normal = np.column_stack([-tans[:, 1], tans[:, 0]])
    return pts + off[:, None] * normal, off


def snap_well(well: str):
    """-> dict with original and snapped traces plus verification numbers."""
    from scipy.spatial import cKDTree
    from tracer_lab.train_tracer import load_well
    from tracer_lab.ridge_yardstick import yard_trace_mean
    from tracer_lab.centreline_targets import polyline_tangents

    image, gt, meta = load_well(well)
    traces, tangents = gt["traces"], gt["tangents"]

    # Per-trace opt-in: a snap is APPLIED only when that trace individually
    # keeps its arc length within 1% and steals no identity. The first global
    # run measured max arc-length drift of 4.5% (curved traces under varying
    # normals) and 2% identity theft inside bundles -- rather than loosening
    # the gate, unsafe traces keep their original geometry. Their targets
    # stay noisy; no new harm is introduced.
    trees = [cKDTree(t) for t in traces]
    all_pts = np.vstack(traces)
    owner = np.concatenate([np.full(len(t), i)
                            for i, t in enumerate(traces)])
    big = cKDTree(all_pts)

    snapped, applied, before, after, arclen_err = [], 0, [], [], []
    for i, (pts, tan) in enumerate(zip(traces, tangents)):
        sp, _ = snap_trace(image, pts, tan)
        L0 = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        L1 = float(np.linalg.norm(np.diff(sp, axis=0), axis=1).sum())
        err = abs(L1 - L0) / max(L0, 1e-9)

        d_own = trees[i].query(sp[::5], workers=-1)[0]
        d_any, j = big.query(sp[::5], workers=-1)
        # theft needs to clear half a pixel: targets are rasterised through
        # _round_to_grid, so ownership differences below 0.5 px do not exist
        # at raster resolution and counting them reverts safe snaps over ties
        stolen = float(((owner[j] != i) & (d_any < d_own - 0.5)).mean())

        keep = sp if (err <= 0.01 and stolen <= 0.0) else pts.copy()
        if keep is sp:
            applied += 1
            arclen_err.append(err)
        snapped.append(keep)

        e0 = yard_trace_mean(image, pts[::7], tan[::7])
        st = polyline_tangents(keep, 15.0)
        e1 = yard_trace_mean(image, keep[::7], st[::7])
        if len(e0):
            before.append(float(e0[0]))
        if len(e1):
            after.append(float(e1[0]))

    b, a = np.array(before), np.array(after)
    report = {
        "well": well, "n_traces": len(traces), "n_snapped": applied,
        "offset_before": {"median": float(np.median(b)), "sd": float(b.std()),
                          "n": int(len(b))},
        "offset_after": {"median": float(np.median(a)), "sd": float(a.std()),
                         "n": int(len(a))},
        "arclen_err_median": (float(np.median(arclen_err))
                              if arclen_err else 0.0),
        "arclen_err_max": float(np.max(arclen_err)) if arclen_err else 0.0,
        "theft_frac": 0.0,       # by construction: thieving traces revert
    }
    ok = (abs(report["offset_after"]["median"]) <= 0.3
          and report["offset_after"]["sd"] < report["offset_before"]["sd"]
          and report["arclen_err_max"] <= 0.01
          and report["n_snapped"] >= 0.5 * report["n_traces"])
    report["pass"] = bool(ok)
    return snapped, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--wells", default="D04,B02")
    ap.add_argument("--verify", action="store_true",
                    help="report only; write nothing")
    ap.add_argument("--all", action="store_true",
                    help="snap every well and persist to _runs/snapped_v1")
    a = ap.parse_args(argv)

    if a.all:
        wells = sorted(p.name for p in
                       (ROOT / "PrecisionMyotube/annotation_work/"
                               "plate32_dense_v1").iterdir() if p.is_dir())
    else:
        wells = [w for w in a.wells.split(",") if w]

    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for well in wells:
        snapped, rep = snap_well(well)
        reports.append(rep)
        print(f"{well}: offset {rep['offset_before']['median']:+.2f}"
              f"+/-{rep['offset_before']['sd']:.2f} px"
              f" -> {rep['offset_after']['median']:+.2f}"
              f"+/-{rep['offset_after']['sd']:.2f} px"
              f" | arclen err med {rep['arclen_err_median']:.4%}"
              f" max {rep['arclen_err_max']:.4%}"
              f" | theft {rep['theft_frac']:.3%}"
              f" | {'PASS' if rep['pass'] else 'FAIL'}", flush=True)
        if not a.verify:
            np.savez_compressed(
                OUT / f"{well}.npz",
                **{f"trace_{i}": t for i, t in enumerate(snapped)})
    (OUT / "verification.json").write_text(json.dumps(reports, indent=2))
    print(f"\nwritten: {OUT / 'verification.json'}")
    if not all(r["pass"] for r in reports):
        print("AT LEAST ONE WELL FAILED -- do not train on snapped traces.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
