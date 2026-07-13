"""Stage 5 -- Measure & Store (validation + index extraction).

The actual measurement (ImageJ "M") is done by ../common/trace_render_measure.ijm, which the
orchestrator runs once per FIGURE into this folder, emitting <fig>_rois.zip and <fig>_results.csv
as SEPARATE files. This script then, for each figure:
  * verifies the two files exist and are separate,
  * verifies ROI count == CSV row count,
  * verifies length_um ~= length_px * pixel_um,
  * writes <fig>_index.csv (id -> mid_x, mid_y, length_um): the overlay indices stored on their own.

Usage: python store.py --out <stage5 dir> --figures final,bright,dim
"""
from __future__ import annotations
import os
import sys
import csv
import json
import zipfile
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from iohelpers import load_config, write_json  # noqa: E402


def check_figure(out_dir, fig, pixel_um, drawn_labels=None):
    roi = os.path.join(out_dir, f"{fig}_rois.zip")
    csvp = os.path.join(out_dir, f"{fig}_results.csv")
    info = {"figure": fig, "rois_zip": os.path.basename(roi), "results_csv": os.path.basename(csvp)}
    if not os.path.exists(csvp):
        info.update(status="MISSING_CSV", n=0)
        return info
    rows = list(csv.DictReader(open(csvp, encoding="utf-8-sig")))
    n_csv = len(rows)
    n_roi = None
    if os.path.exists(roi):
        with zipfile.ZipFile(roi) as z:
            n_roi = len([x for x in z.namelist() if x.endswith(".roi")])
    elif n_csv > 0:
        info.update(status="MISSING_ROI", n=n_csv)
        return info

    # length_um consistency
    bad = 0
    for r in rows:
        try:
            lp, lu = float(r["length_px"]), float(r["length_um"])
            if lp > 0 and abs(lu - lp * pixel_um) / (lp * pixel_um) > 0.02:
                bad += 1
        except (KeyError, ValueError):
            bad += 1

    # write the index file (overlay ids on their own) + your label for any trace you drew
    drawn_labels = drawn_labels or {}
    idxp = os.path.join(out_dir, f"{fig}_index.csv")
    with open(idxp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "mid_x", "mid_y", "length_um", "user_label"])
        for r in rows:
            w.writerow([r.get("id"), r.get("mid_x"), r.get("mid_y"), r.get("length_um"),
                        drawn_labels.get(str(r.get("id")), "")])

    status = "OK"
    if n_roi is not None and n_roi != n_csv:
        status = f"COUNT_MISMATCH(roi={n_roi},csv={n_csv})"
    elif bad:
        status = f"LENGTH_UM_OFF({bad})"
    info.update(status=status, n=n_csv, n_roi=n_roi, length_um_bad=bad,
                index_csv=os.path.basename(idxp))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures", default="final,bright,dim")
    a = ap.parse_args()
    pixel_um = load_config()["pixel_um"]
    figs = [f.strip() for f in a.figures.split(",") if f.strip()]
    drawn = {}
    dl = os.path.join(a.out, "..", "stage4_qc", "drawn_labels.json")   # your per-myotube labels
    if os.path.exists(dl):
        try:
            with open(dl, encoding="utf-8") as fh:
                drawn = json.load(fh)
        except (ValueError, OSError):
            drawn = {}
    results = [check_figure(a.out, f, pixel_um, drawn if f == "final" else {}) for f in figs]
    write_json(os.path.join(a.out, "store_summary.json"), {"figures": results})
    for r in results:
        print(f"  {r['figure']:8s} n={r.get('n')}  roi={r.get('n_roi')}  status={r['status']}  "
              f"-> {r['rois_zip']} + {r['results_csv']}")
    if any(r["status"] not in ("OK",) and not r["status"].startswith("LENGTH_UM_OFF") for r in results):
        print("WARNING: some figures failed validation (see store_summary.json)")


if __name__ == "__main__":
    main()
