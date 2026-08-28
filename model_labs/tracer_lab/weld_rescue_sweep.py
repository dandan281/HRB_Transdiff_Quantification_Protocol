"""Sweep the two crossing-cut fixes on the TUNE wells; claim on TEST wells.

Motivated by the 2026-08-27 break-point attribution: ~1 in 5 fibres is cut
into >=2 pieces, breaks sit at predicted crossings 2-3x above base rate,
~1/3 of them abut. Two walk-level mechanisms, no retraining:

* **junction weld** (`weld_objects`): post-walk identity merge of co-linear
  pieces meeting at a predicted crossing. Validated on synthetic geometry
  (5/5 contract cases) before touching real data.
* **rescue widening**: `rescue_window_steps` / `rescue_reach` were frozen at
  1 / 2.0 on PERFECT fields, where crossing exits are clean; predicted
  crossing zones are wider and messier.

Discipline (predeclared before any result was computed):

* Sweep runs ONLY on the tune wells C02 C03 C05 C11 D02, each scored by its
  own never-seen CV fold. Test wells B02 D04 D08 D09 D11 are untouched until
  one config is frozen.
* PRIMARY metric: pooled ``false_split_count`` (= GT fibres covered by >=2
  substantial pieces — the "one myotube traced as several" count).
* Selection rule: maximize pooled net repairs =
  (baseline_splits - splits) - max(0, merges - baseline_merges),
  subject to (a) pooled mean-of-well-median mdape <= baseline + 0.01 and
  (b) pooled identity_through_crossing >= baseline - 0.005.
  On a plateau (net repairs within 2 of the best), prefer the smallest
  (rescue_window, rescue_reach, weld_dist) — the least-intervention config.
* The claim run (``--claim``) scores ONLY the frozen baseline and the frozen
  chosen config on the test wells, with a drop-one-well check.

    python model_labs/tracer_lab/weld_rescue_sweep.py            # tune sweep
    python model_labs/tracer_lab/weld_rescue_sweep.py --claim    # test claim
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

TUNE = ["C02", "C03", "C05", "C11", "D02"]
TEST = ["B02", "D04", "D08", "D09", "D11"]
CV = ROOT / "model_labs/tracer_lab/_runs/net_cv"
CACHE = ROOT / "model_labs/tracer_lab/_runs/sweep_cache"
OUT = ROOT / "model_labs/tracer_lab/_runs/weld_rescue_sweep.json"
OUT_CLAIM = ROOT / "model_labs/tracer_lab/_runs/weld_rescue_claim.json"

BASE_WALK = dict(seed_thresh=0.4, support_thresh=0.3, claim_radius_px=3.5,
                 rescue_window_steps=1)          # the frozen CV config
RESCUE_GRID = [(1, 2.0), (2, 2.0), (4, 2.0), (8, 2.0), (4, 4.0)]
WELD_DIST = [0.0, 6.0, 10.0, 14.0]
WELD_DEG = [12.5, 20.0]
CROSSING_GATE_PX = 12.0


def get_fields(well):
    """Predicted fields for `well` from its never-seen fold, disk-cached."""
    from tracer_lab.infer_trace import predict_fields
    from tracer_lab.train_tracer import load_well

    CACHE.mkdir(parents=True, exist_ok=True)
    npz = CACHE / f"{well}.npz"
    image, gt, _ = load_well(well)
    if npz.exists():
        z = np.load(npz)
        pred = {k: z[k] for k in ("centre", "orient", "crossing")}
    else:
        pred = predict_fields(image, CV / well / "best.pt")
        np.savez_compressed(npz, **{k: pred[k] for k in
                                    ("centre", "orient", "crossing")})
    return pred, gt


def score_config(res, wf, weld_dist, weld_deg):
    from tracer_lab.oracle_trace import score_against_gt, weld_objects

    r = weld_objects(res, wf, weld_dist_px=weld_dist, weld_deg=weld_deg,
                     crossing_gate_px=CROSSING_GATE_PX) \
        if weld_dist > 0 else res
    sc = score_against_gt(r, wf)
    return {"false_split": sc["false_split_count"],
            "false_merge": sc["false_merge_count"],
            "identity_x": sc["identity_through_crossing"],
            "mdape": sc["length_mdape"],
            "recall": sc["recall_traces"],
            "n_welds": len(r.get("weld_events", []))}


def run_wells(wells, configs, tag):
    """-> rows[(rescue, weld)][well] = metrics; configs = list of
    ((window, reach), (weld_dist, weld_deg))."""
    from tracer_lab.infer_trace import fields_for_walk
    from tracer_lab.oracle_trace import TraceParams, trace_field

    rescue_set = sorted({rc for rc, _ in configs})
    weld_by_rescue = {rc: sorted({wc for r2, wc in configs if r2 == rc})
                      for rc in rescue_set}
    rows: dict[str, dict] = {}
    for well in wells:
        t0 = time.time()
        pred, gt = get_fields(well)
        wf = fields_for_walk(pred, crossing_thresh=0.4, valid_thresh=0.2,
                             prep="nms")
        wf["instance"] = gt["instance"]
        wf["traces"] = gt["traces"]
        for (win, reach) in rescue_set:
            prm = TraceParams(**{**BASE_WALK,
                                 "rescue_window_steps": win,
                                 "rescue_reach": reach})
            res = trace_field(wf, prm)
            for (wd, wdeg) in weld_by_rescue[(win, reach)]:
                key = f"r{win}-{reach:g}_w{wd:g}-{wdeg:g}"
                m = score_config(res, wf, wd, wdeg)
                rows.setdefault(key, {})[well] = m
                print(f"  [{tag}] {well} {key:<18} splits {m['false_split']:>3}"
                      f"  merges {m['false_merge']:>3}"
                      f"  idx {m['identity_x']:.3f}"
                      f"  mdape {m['mdape']:.3f}"
                      f"  welds {m['n_welds']:>3}", flush=True)
        print(f"  [{tag}] {well} done in {time.time() - t0:.0f}s", flush=True)
    return rows


def pool(rows_for_cfg, wells):
    ms = [rows_for_cfg[w] for w in wells]
    return {"false_split": sum(m["false_split"] for m in ms),
            "false_merge": sum(m["false_merge"] for m in ms),
            "identity_x": float(np.mean([m["identity_x"] for m in ms])),
            "mdape": float(np.mean([m["mdape"] for m in ms])),
            "recall": float(np.mean([m["recall"] for m in ms])),
            "n_welds": sum(m["n_welds"] for m in ms)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--claim", action="store_true",
                    help="run frozen baseline + chosen config on TEST wells")
    a = ap.parse_args(argv)

    if not a.claim:
        configs = [(rc, (wd, wdeg)) for rc in RESCUE_GRID
                   for wd in WELD_DIST for wdeg in WELD_DEG
                   if not (wd == 0.0 and wdeg != WELD_DEG[0])]
        print(f"TUNE sweep: {len(RESCUE_GRID)} rescue x "
              f"{len(WELD_DIST) * len(WELD_DEG) - 1} weld configs on {TUNE}")
        rows = run_wells(TUNE, configs, "tune")

        base_key = f"r1-2_w0-{WELD_DEG[0]:g}"
        base = pool(rows[base_key], TUNE)
        table = []
        for key, per_well in rows.items():
            p = pool(per_well, TUNE)
            net = (base["false_split"] - p["false_split"]) \
                - max(0, p["false_merge"] - base["false_merge"])
            ok = (p["mdape"] <= base["mdape"] + 0.01
                  and p["identity_x"] >= base["identity_x"] - 0.005)
            table.append({"key": key, **p, "net_repairs": net,
                          "guards_ok": bool(ok)})
        table.sort(key=lambda r: -r["net_repairs"])
        best_net = max(r["net_repairs"] for r in table if r["guards_ok"])

        def knob_size(key):
            r_part, w_part = key.split("_")
            win, reach = r_part[1:].split("-")
            wd = w_part[1:].split("-")[0]
            return (int(win), float(reach), float(wd))

        plateau = [r for r in table
                   if r["guards_ok"] and r["net_repairs"] >= best_net - 2]
        chosen = min(plateau, key=lambda r: knob_size(r["key"]))

        print("\n=== pooled over tune wells (baseline first) ===")
        hdr = (f"{'config':<20}{'splits':>8}{'merges':>8}{'net':>6}"
               f"{'idx':>7}{'mdape':>8}{'recall':>8}{'welds':>7}{'ok':>4}")
        print(hdr)
        print("-" * len(hdr))
        for r0 in [next(r for r in table if r["key"] == base_key)] + \
                [r for r in table if r["key"] != base_key]:
            print(f"{r0['key']:<20}{r0['false_split']:>8}"
                  f"{r0['false_merge']:>8}{r0['net_repairs']:>6}"
                  f"{r0['identity_x']:>7.3f}{r0['mdape']:>8.3f}"
                  f"{r0['recall']:>8.3f}{r0['n_welds']:>7}"
                  f"{'y' if r0['guards_ok'] else 'n':>4}")
        print(f"\nCHOSEN (plateau rule): {chosen['key']}  "
              f"net repairs {chosen['net_repairs']} vs baseline")

        OUT.write_text(json.dumps({
            "tune_wells": TUNE, "base_key": base_key,
            "chosen_key": chosen["key"],
            "selection_rule": ("max net_repairs s.t. mdape<=base+0.01 and "
                               "identity_x>=base-0.005; plateau within 2 -> "
                               "smallest knobs"),
            "pooled": table, "per_well": rows}, indent=2))
        print(f"-> {OUT}")
        return 0

    # ---- claim on TEST wells: frozen baseline + frozen chosen config ----
    sweep = json.loads(OUT.read_text())
    base_key, chosen_key = sweep["base_key"], sweep["chosen_key"]

    def parse(key):
        r_part, w_part = key.split("_")
        win, reach = r_part[1:].split("-")
        wd, wdeg = w_part[1:].split("-")
        return ((int(win), float(reach)), (float(wd), float(wdeg)))

    configs = [parse(base_key), parse(chosen_key)]
    print(f"CLAIM on {TEST}: baseline {base_key} vs chosen {chosen_key}")
    rows = run_wells(TEST, configs, "claim")

    out = {"test_wells": TEST, "base_key": base_key,
           "chosen_key": chosen_key, "per_well": rows,
           "pooled": {k: pool(rows[k], TEST) for k in rows}}
    b, c = out["pooled"][base_key], out["pooled"][chosen_key]
    print("\n=== TEST WELLS (never used for tuning) ===")
    print(f"baseline: splits {b['false_split']}  merges {b['false_merge']}  "
          f"idx {b['identity_x']:.3f}  mdape {b['mdape']:.3f}")
    print(f"chosen  : splits {c['false_split']}  merges {c['false_merge']}  "
          f"idx {c['identity_x']:.3f}  mdape {c['mdape']:.3f}")
    drop = {}
    for w in TEST:
        rest = [x for x in TEST if x != w]
        pb, pc = pool(rows[base_key], rest), pool(rows[chosen_key], rest)
        drop[w] = {"net": (pb["false_split"] - pc["false_split"])
                   - max(0, pc["false_merge"] - pb["false_merge"])}
    out["drop_one_well_net_repairs"] = drop
    print("drop-one-well net repairs:", {k: v["net"] for k, v in drop.items()})
    OUT_CLAIM.write_text(json.dumps(out, indent=2))
    print(f"-> {OUT_CLAIM}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
