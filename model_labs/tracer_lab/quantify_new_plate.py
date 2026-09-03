"""Quantify ANY plate with the deployed tracer stack -- the product entry point.

Everything before this script was written to answer one question about one
dataset, so each runner had its plate baked in. This one takes a folder of
``.nd2`` files and produces, per well: fibre count, total length, median
length, and the operator's metric of record -- the share of myotubes per
length class -- with a comparison to operator ROIs only if they exist.

Acquisition handling, from metadata rather than constants:

* **channel**: `precision_myotube.io.resolve_roles` (the project's existing
  rule: ch1 = desmin on validated Q-plates, morphology otherwise; override
  with ``--fiber-ch``). The chosen channel is recorded per well.
* **pixel size**: read from the nd2. The network was trained at
  0.650 um/px; a plate acquired at another scale is RESAMPLED to that
  scale before inference and lengths are reported in um. This is the
  mechanically correct thing to do and it is UNVALIDATED -- the tracer has
  only been measured at 0.65 um/px. Any plate rescaled by more than 2% is
  flagged loudly, and its numbers must not be trusted until a validation
  pass exists for that acquisition (project rule: sweep every knob per
  dataset, never inherit).

Two stages, because the nd2 reader lives in a different env from torch:

    # 1. extract fibre channels (needs the `nd2` package -- env `base`)
    conda run -n base python model_labs/tracer_lab/quantify_new_plate.py \\
        --plate "Q_PLATES/Q_Plates/PLATE_26" --extract

    # 2. trace + quantify (GPU env)
    conda run -n pm-omnipose python model_labs/tracer_lab/quantify_new_plate.py \\
        --plate "Q_PLATES/Q_Plates/PLATE_26"

Deployed stack = frozen walk -> junction weld -> identity repair
(``--mode repair``); ``--mode weld`` and ``--mode loop`` are available.
Outputs go to ``_runs/plates/<PLATE>/``: ``wells.csv``, ``summary.json``,
``length_classes.png``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools",
           ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TARGET_UM = 0.650017                     # the training acquisition
DEFAULT_CKPT = "model_labs/tracer_lab/_runs/net_cv/B02/best.pt"
MIN_UM = 50.0
RESCALE_TOL = 0.02                       # beyond this, resample + warn
WELL_RE = re.compile(r"^[A-P]\d{2}$")
WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
            rescue_window_steps=1)
WELD = dict(weld_dist_px=14.0, weld_deg=12.5, crossing_gate_px=12.0)


# ----------------------------------------------------------------- discovery
def well_token(stem: str) -> str:
    """'19_B06_actv104_trka' -> 'B06'; falls back to the whole stem."""
    for part in stem.split("_"):
        if WELL_RE.match(part.upper()):
            return part.upper()
    return stem


def discover(plate_dir: Path, wells_filter=None) -> list[tuple[str, Path]]:
    out = []
    for nd in sorted(plate_dir.glob("*.nd2")):
        w = well_token(nd.stem)
        if wells_filter and w not in wells_filter:
            continue
        out.append((w, nd))
    if not out:
        raise SystemExit(f"no .nd2 files in {plate_dir}")
    names = [w for w, _ in out]
    dup = sorted({w for w in names if names.count(w) > 1})
    if dup:
        raise SystemExit(f"well token collision {dup} -- pass --wells or "
                         "rename; refusing to guess")
    return out


def find_rois(plate_dir: Path, well: str) -> Path | None:
    cands = [z for z in plate_dir.glob("*.zip")
             if z.name.upper().startswith(well.upper() + "_")]
    return cands[0] if cands else None


def out_dir_for(args) -> Path:
    plate_name = Path(args.plate).name.replace(" ", "_")
    d = ROOT / args.out / plate_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------- stage 1
def extract(args) -> int:
    try:
        import nd2  # noqa: F401  (load_nd2 imports lazily; fail early, clearly)
        from precision_myotube.io import load_nd2, resolve_roles
    except ModuleNotFoundError:
        raise SystemExit(
            "the `nd2` package is not in this env. Run this stage in one "
            "that has it, e.g.\n  conda run -n base python "
            "model_labs/tracer_lab/quantify_new_plate.py --plate "
            f"\"{args.plate}\" --extract")

    plate_dir = Path(args.plate)
    out = out_dir_for(args) / "cache"
    out.mkdir(exist_ok=True)
    for well, nd in discover(plate_dir, args.wells):
        dst = out / f"{well}_fiber.npy"
        meta_p = out / f"{well}_meta.json"
        if dst.exists() and meta_p.exists() and not args.force:
            print(f"{well}: cached")
            continue
        channels, pixel_um, sizes = load_nd2(nd)
        roles, scores = resolve_roles(channels, args.fiber_ch, args.dapi_ch)
        fiber = channels[roles["fiber"]]
        np.save(dst, fiber)
        meta = {"well": well, "nd2": nd.name, "pixel_um": float(pixel_um),
                "fiber_channel": int(roles["fiber"]),
                "dapi_channel": int(roles["dapi"]),
                "n_channels": int(channels.shape[0]),
                "shape": list(fiber.shape),
                "channel_scores": {str(k): {kk: float(vv)
                                            for kk, vv in v.items()}
                                   for k, v in scores.items()}}
        # A saturated channel scores ZERO on every morphology feature (one
        # giant region, no blobs), which silently derails `resolve_roles`:
        # measured on PLATE_44 B07, ch0 (DAPI) at p97.5 = 4095 scored
        # nuclei=0, so DAPI was assigned to ch2 and the fibre channel was
        # decided by the ch1 prior alone. The prior happened to be right
        # there; on a plate where it is not, this is the only warning.
        sat = [int(k) for k, v in scores.items()
               if v["p975"] >= 0.99 * float(channels.max())
               and v["nuclei"] == 0 and v["fiber"] == 0]
        meta["saturated_channels"] = sat
        meta["fiber_choice"] = ("override" if args.fiber_ch is not None
                                else "ch1 prior" if roles["fiber"] == 1
                                else "morphology")
        meta_p.write_text(json.dumps(meta, indent=2))
        flags = []
        if abs(pixel_um / TARGET_UM - 1.0) > RESCALE_TOL:
            flags.append(f"pixel {pixel_um:.4f} um != trained "
                         f"{TARGET_UM:.4f} -- will be resampled "
                         f"x{pixel_um / TARGET_UM:.2f}; UNVALIDATED")
        if sat:
            flags.append(f"ch{sat} SATURATED (p97.5 at max) -- channel "
                         "roles are not trustworthy; confirm with "
                         "--fiber-ch")
        flag = ("   ** " + " | ".join(flags) + " **") if flags else ""
        print(f"{well}: {nd.name}  ch{roles['fiber']}=fiber "
              f"({meta['fiber_choice']}) ch{roles['dapi']}=dapi  "
              f"{tuple(fiber.shape)}  {pixel_um:.4f} um/px{flag}")
    return 0


# ----------------------------------------------------------------- stage 2
def _lengths(res, um, smoothed):
    from tracer_lab.centreline_targets import resample_polyline
    from tracer_lab.length_classes import arc_um
    per_obj: dict[int, float] = {}
    for pid, path in enumerate(res["paths"], start=1):
        p = np.asarray(path, float)
        if p.ndim != 2 or len(p) < 2:
            continue
        d = resample_polyline(p, 1.0)
        oid = res["object_of"][pid]
        per_obj[oid] = per_obj.get(oid, 0.0) + arc_um(d, um, smoothed=smoothed)
    return np.array([v for v in per_obj.values() if v >= MIN_UM])


def quantify(args) -> int:
    from scipy import ndimage
    from tracer_lab.centreline_targets import build_targets
    from tracer_lab.decompose_retrace import apply_repair, loop_pipeline
    from tracer_lab.infer_trace import fields_for_walk, predict_fields
    from tracer_lab.length_classes import (
        LABELS, arc_um, class_shares, format_shares)
    from tracer_lab.oracle_trace import (
        TraceParams, score_against_gt, trace_field, weld_objects)

    plate_dir = Path(args.plate)
    out = out_dir_for(args)
    cache = out / "cache"
    ckpt = ROOT / args.ckpt
    wells = discover(plate_dir, args.wells)
    missing = [w for w, _ in wells if not (cache / f"{w}_fiber.npy").exists()]
    if missing:
        raise SystemExit(f"no extracted channel for {missing}: run --extract "
                         "first (in an env with the `nd2` package)")

    print(f"plate {plate_dir.name}: {len(wells)} wells  mode={args.mode}  "
          f"ckpt={args.ckpt}")
    rows, warned = [], []
    for well, nd in wells:
        t0 = time.time()
        meta = json.loads((cache / f"{well}_meta.json").read_text())
        img = np.load(cache / f"{well}_fiber.npy").astype(np.float32)
        pixel_um = meta["pixel_um"]
        s = pixel_um / TARGET_UM
        rescaled = abs(s - 1.0) > RESCALE_TOL
        if rescaled:
            img = ndimage.zoom(img, s, order=1)
            warned.append(well)
        um = TARGET_UM if rescaled else pixel_um

        lo, hi = np.percentile(img, [1.0, 99.9])
        norm = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) \
            .astype(np.float32)
        pred = predict_fields(norm, ckpt)
        wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                             prep="nms")
        res = trace_field(wf, TraceParams(**WALK))
        res = weld_objects(res, wf, **WELD)
        if args.mode == "repair":
            res, _ = apply_repair(res, norm, ckpt)
        elif args.mode == "loop":
            res, _ = loop_pipeline(res, norm, ckpt)

        L = _lengths(res, um, smoothed=True)
        L_raw = _lengths(res, um, smoothed=False)
        rec = {"well": well, "nd2": nd.name, "pixel_um": pixel_um,
               "fiber_channel": meta["fiber_channel"],
               "rescaled_by": round(s, 4) if rescaled else 1.0,
               "n": int(len(L)),
               "total_mm": round(float(L.sum() / 1000), 3),
               "total_mm_rawarc": round(float(L_raw.sum() / 1000), 3),
               "median_um": round(float(np.median(L)), 1) if len(L) else 0.0,
               "length_classes": class_shares(L),
               "lengths_um": [round(float(v), 1) for v in L]}

        # optional operator comparison -- only when ROIs exist for the well
        zp = find_rois(plate_dir, well)
        human_polys_scaled = None
        if zp is not None:
            from annotation_tools.relabel.fiji_roi import (
                LINE_TYPES, read_roi_set)
            polys = [np.asarray(r["points"], float)
                     for r in read_roi_set(zp)
                     if r["type"] in LINE_TYPES and len(r["points"]) >= 2]
            h_sm = np.array([arc_um(p, pixel_um, smoothed=True)
                             for p in polys])
            h_raw = np.array([arc_um(p, pixel_um) for p in polys])
            h_sm, h_raw = h_sm[h_sm >= MIN_UM], h_raw[h_raw >= MIN_UM]
            human_polys_scaled = [p * s for p in polys] if rescaled else polys
            gt = build_targets(norm.shape, human_polys_scaled)
            wf["instance"], wf["traces"] = gt["instance"], gt["traces"]
            sc = score_against_gt(res, wf)
            rec["human"] = {
                "roi_zip": zp.name, "n": int(len(h_sm)),
                "total_mm": round(float(h_sm.sum() / 1000), 3),
                "total_mm_rawarc": round(float(h_raw.sum() / 1000), 3),
                "length_classes": class_shares(h_sm),
                "length_classes_rawarc": class_shares(h_raw),
                "trace_recall": round(sc["recall_traces"], 3),
                "false_splits": sc["false_split_count"],
                "false_merges": sc["false_merge_count"],
                "identity_through_crossing":
                    round(sc["identity_through_crossing"], 3),
                "length_mdape": round(sc["length_mdape"], 3)}
        rows.append(rec)
        if not args.no_overlay:
            _overlay(norm, res, out / f"{well}_overlay.png",
                     human_polys_scaled)

        hum = ""
        if "human" in rec:
            h = rec["human"]
            hum = (f" | you n {h['n']:>4} mm {h['total_mm']:>6.1f} "
                   f"recall {h['trace_recall']:.2f}")
        scale_note = f" x{s:.2f}" if rescaled else "      "
        print(f"{well:<5} ch{meta['fiber_channel']} {pixel_um:.3f}um"
              f"{scale_note}  n {rec['n']:>4}  mm {rec['total_mm']:>6.1f}  "
              f"med {rec['median_um']:>5.0f}  "
              f"{format_shares(rec['length_classes'])}{hum}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- plate-level outputs
    has_h = any("human" in r for r in rows)
    with (out / "wells.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        head = ["well", "nd2", "pixel_um", "fiber_channel", "rescaled_by",
                "n", "total_mm", "median_um"] + \
               [f"share_{l}" for l in LABELS]
        if has_h:
            head += ["human_n", "human_total_mm", "trace_recall",
                     "false_splits", "identity_through_crossing"] + \
                    [f"human_share_{l}" for l in LABELS]
        wr.writerow(head)
        for r in rows:
            line = [r["well"], r["nd2"], r["pixel_um"], r["fiber_channel"],
                    r["rescaled_by"], r["n"], r["total_mm"], r["median_um"]] \
                + [r["length_classes"][l] for l in LABELS]
            if has_h:
                h = r.get("human")
                if h:
                    line += [h["n"], h["total_mm"], h["trace_recall"],
                             h["false_splits"],
                             h["identity_through_crossing"]] \
                        + [h["length_classes"][l] for l in LABELS]
                else:
                    line += [""] * (5 + len(LABELS))
            wr.writerow(line)

    pooled = class_shares(np.concatenate([r["lengths_um"] for r in rows]))
    summary = {"plate": plate_dir.name, "mode": args.mode, "ckpt": args.ckpt,
               "target_um": TARGET_UM, "min_um": MIN_UM,
               "walk": WALK, "weld": WELD,
               "rescaled_wells_UNVALIDATED": warned,
               "pooled_length_classes": pooled, "wells": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _chart(rows, pooled, plate_dir.name, out / "length_classes.png")

    print(f"\nPLATE {plate_dir.name}: {sum(r['n'] for r in rows)} fibres, "
          f"{sum(r['total_mm'] for r in rows):.1f} mm   pooled mix: "
          f"{format_shares(pooled)}")
    if warned:
        print(f"\n** {len(warned)} well(s) were RESAMPLED from a non-training "
              f"pixel size: {warned}. The tracer is validated at "
              f"{TARGET_UM:.3f} um/px only; treat these as unvalidated until "
              "a sweep on this acquisition exists. **")
    print(f"-> {out / 'wells.csv'}\n-> {out / 'summary.json'}\n"
          f"-> {out / 'length_classes.png'}"
          + ("" if args.no_overlay else
             f"\n-> {out}/<well>_overlay.png  (per-well QC: one colour per "
             "traced object; operator ROIs as the left panel where present)"))
    return 0


def _overlay(norm, res, out_path, human_polys=None):
    """Per-well QC visual: every traced object in its own color on the
    image (half resolution, whole well), and -- when ROIs exist -- the
    operator's traces as a left panel in the same style. A colour change
    along one fibre is a cut; a fibre with no colour is a miss. This is the
    picture that decides whether a well's numbers are believed."""
    import colorsys
    import imageio.v2 as imageio
    from tracer_lab.centreline_targets import resample_polyline

    H, W = norm.shape

    def base():
        return np.stack([np.clip(norm * 0.55, 0, 1)] * 3, -1)

    def color(i):
        h = (i * 0.6180339887) % 1.0
        return np.array(colorsys.hsv_to_rgb(h, 0.85 if i % 2 == 0 else 0.65,
                                            1.0 if i % 3 else 0.8))

    def draw(rgb, pts, col):
        d = resample_polyline(np.asarray(pts, float), 1.0)
        r = np.clip(np.round(d[:, 0]).astype(int), 0, H - 1)
        c = np.clip(np.round(d[:, 1]).astype(int), 0, W - 1)
        for dr in (0, 1):
            for dc in (0, 1):
                rgb[np.clip(r + dr, 0, H - 1), np.clip(c + dc, 0, W - 1)] = col

    tracer = base()
    oid_col: dict[int, np.ndarray] = {}
    for pid, poly in enumerate(res["paths"], start=1):   # NOT `path`: it
        p = np.asarray(poly, float)                        # shadowed out_path
        if p.ndim != 2 or len(p) < 2:
            continue
        oid = res["object_of"][pid]
        if oid not in oid_col:
            oid_col[oid] = color(len(oid_col))
        draw(tracer, p, oid_col[oid])

    def half(rgb):
        h2, w2 = (H // 2) * 2, (W // 2) * 2
        r = rgb[:h2, :w2]
        return (r[0::2, 0::2] + r[1::2, 0::2] + r[0::2, 1::2]
                + r[1::2, 1::2]) / 4.0

    panel = half(tracer)
    if human_polys:
        human = base()
        for i, p in enumerate(human_polys):
            if len(p) >= 2:
                draw(human, p, color(i))
        gap = np.ones((panel.shape[0], 12, 3)) * 0.15
        panel = np.concatenate([half(human), gap, panel], axis=1)
    imageio.imwrite(out_path, (np.clip(panel, 0, 1) * 255).astype(np.uint8))


def _chart(rows, pooled, plate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tracer_lab.length_classes import LABELS

    BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#6b6a60"
    x = np.arange(len(LABELS))
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="white")
    tv = np.array([pooled[l] for l in LABELS]) * 100
    hrows = [r for r in rows if "human" in r]
    if hrows:
        # pool operator shares by fibre count across the wells that have ROIs
        hn = sum(r["human"]["n"] for r in hrows)
        hv = np.array([sum(r["human"]["length_classes"][l] * r["human"]["n"]
                           for r in hrows) / max(hn, 1) for l in LABELS]) * 100
        w = 0.38
        b1 = ax.bar(x - w / 2 - 0.02, hv, w, color=BLUE,
                    label=f"operator ROIs (n={hn}, {len(hrows)} wells)")
        b2 = ax.bar(x + w / 2 + 0.02, tv, w, color=ORANGE,
                    label=f"tracer (n={pooled['n']}, {len(rows)} wells)")
        bars = [(b1, hv), (b2, tv)]
    else:
        b2 = ax.bar(x, tv, 0.55, color=ORANGE,
                    label=f"tracer (n={pooled['n']}, {len(rows)} wells)")
        bars = [(b2, tv)]
    for bs, vals in bars:
        for rect, v in zip(bs, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.6, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, color=INK)
    ax.set_xlabel("myotube length (um)", color=INK)
    ax.set_ylabel("share of myotubes (%)", color=INK)
    ax.set_title(f"{plate}: length mix (smoothed convention, >=50 um)",
                 fontsize=11, color=INK)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color="#e8e7df", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plate", required=True, help="folder of .nd2 files")
    ap.add_argument("--extract", action="store_true",
                    help="stage 1: extract fibre channels (needs `nd2`)")
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--mode", default="repair",
                    choices=("weld", "repair", "loop"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--fiber-ch", type=int, default=None,
                    help="override the fibre channel (default: project rule)")
    ap.add_argument("--dapi-ch", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="re-extract")
    ap.add_argument("--no-overlay", action="store_true",
                    help="skip the per-well QC overlay PNGs")
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs/plates")
    args = ap.parse_args(argv)
    return extract(args) if args.extract else quantify(args)


if __name__ == "__main__":
    raise SystemExit(main())
