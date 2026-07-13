"""CLI: score the benchmark.

  python -m benchmark --all              score every well with a completed prediction
  python -m benchmark --well P23_C08_BR223_IGF1R
  python -m benchmark --all --sweep      also report F1 sensitivity to dilate radius {3,5,8}
  python -m benchmark --status           list which wells have GT / predictions (no scoring)
"""
from __future__ import annotations

import argparse

from . import config as C
from . import run as R


def cmd_status():
    print(f"{'well_id':28s} {'plate':9s} {'group':13s} {'GT':4s} {'pred':5s} run_stem")
    n_pred = 0
    for w in C.WELLS:
        gt = "ok" if w["gt_roi_ok"] else "BAD"
        if w["gt_geom_partial"]:
            gt = "part"
        has = C.has_prediction(w)
        n_pred += bool(has)
        print(f"{w['well_id']:28s} {w['plate']:9s} {w['group']:13s} {gt:4s} "
              f"{'YES' if has else '-':5s} {w['run_stem'] or ''}")
    print(f"\n{len(C.WELLS)} registry wells; {n_pred} have completed predictions (scoreable now).")


def cmd_sweep(wells):
    print("\n--- dilate-radius sensitivity (F1) ---")
    print(f"{'well_id':28s}  r=3      r=5      r=8")
    for w in wells:
        cells = []
        for r in (3, 5, 8):
            res = R.score_well(w, radius=r, verbose=False)
            t1 = res["tier1"]
            cells.append(f"{t1['f1']:.3f}" if t1 else "n/a  ")
        print(f"{w['well_id']:28s}  " + "    ".join(cells))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmark")
    ap.add_argument("--all", action="store_true", help="score all wells with predictions")
    ap.add_argument("--well", help="score a single well by well_id")
    ap.add_argument("--sweep", action="store_true", help="report F1 vs dilate radius")
    ap.add_argument("--status", action="store_true", help="list GT/prediction availability, no scoring")
    a = ap.parse_args(argv)

    if a.status:
        cmd_status()
        return

    if a.well:
        wells = [C.well_by_id(a.well)]
    else:
        wells = [w for w in C.WELLS if C.has_prediction(w)]
    if not wells:
        print("No scoreable wells (none have completed predictions). Try --status.")
        return

    results = [R.score_well(w) for w in wells]
    R.write_outputs(results)
    if a.sweep:
        cmd_sweep([w for w in wells if w["gt_roi_ok"]])


if __name__ == "__main__":
    main()
