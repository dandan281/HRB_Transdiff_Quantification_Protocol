"""Append a completed review's (features -> your decision) records to the accumulating dataset.

Runs during --resume, AFTER decisions.json is applied. One CSV per case type accumulates across
every well you review (learning/data/{split,merge,occluded}.csv). Deduped on (stem, case_id) so
re-resuming the same image never double-counts.
"""
from __future__ import annotations
import os
import sys
import csv
import json
import math
import argparse


def _plen(poly):
    return sum(math.hypot(poly[i][0] - poly[i - 1][0], poly[i][1] - poly[i - 1][1])
               for i in range(1, len(poly)))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import FEATURE_KEYS, label_for, DATA_DIR  # noqa: E402


def _redraw_label(ctype, case, decision):
    """Infer a binary label from a hand redraw when the geometry makes the intent clear.

    The review page gives us both overlays in coordinates:
      * case["trace_polys"] = the pipeline's proposed trace(s)
      * decision["polys"] = the myotube(s) the reviewer drew instead

    These are still conservative labels. Shape-only corrections become negative split examples
    rather than a new "extend" class, because the current served models only know split/merge/
    occluded review defaults.
    """
    orig_count = len(case.get("trace_polys", []))
    drawn_count = len(decision.get("polys", []))
    if drawn_count <= 0:
        return None, None

    if ctype == "split":
        if drawn_count > orig_count:
            return 1, "redraw_split"
        if drawn_count == orig_count:
            return 0, "redraw_keep"
        return 0, "redraw_reject"

    if ctype == "merge":
        if drawn_count == 1:
            return 1, "redraw_merge"
        if drawn_count > 1:
            return 0, "redraw_separate"

    if ctype == "occluded":
        return 1, "redraw_restore"

    return None, None


def existing_keys(path):
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                keys.add((row.get("stem"), row.get("case_id")))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="stage4_qc dir of the reviewed run")
    ap.add_argument("--stem", required=True)
    a = ap.parse_args()

    flags_p = os.path.join(a.out, "flags.json")
    dec_p = os.path.join(a.out, "decisions.json")
    if not os.path.exists(dec_p):
        print("no decisions.json -> nothing to learn from (auto-only run)")
        return
    if not os.path.exists(flags_p):
        print("no flags.json -> cannot map decisions to features; skipping")
        return
    try:
        with open(flags_p, encoding="utf-8") as fh:
            flags = json.load(fh)
        with open(dec_p, encoding="utf-8") as fh:
            decisions = json.load(fh).get("decisions", {})
    except (ValueError, OSError) as e:
        print(f"malformed flags/decisions json -> skipping ({e})")
        return
    flags.setdefault("cases", [])

    os.makedirs(DATA_DIR, exist_ok=True)
    added = {"split": 0, "merge": 0, "occluded": 0}
    inferred = {"split": 0, "merge": 0, "occluded": 0}
    redraw_rows = []
    for ctype in FEATURE_KEYS:
        path = os.path.join(DATA_DIR, f"{ctype}.csv")
        seen = existing_keys(path)
        new_rows = []
        for c in flags["cases"]:
            if c["type"] != ctype:
                continue
            d = decisions.get(c["id"])
            if not d:                                   # case absent from decisions -> skip
                continue
            if d.get("skip_learn"):                     # abandoned/no-op decision -> not a label
                continue
            key = (a.stem, c["id"])
            if key in seen:
                continue
            feats = c.get("features", {})
            action = d.get("action", "")
            lab = label_for(ctype, action)
            if lab is None:                             # e.g. 'redraw' -> not a binary label
                lab, inferred_action = _redraw_label(ctype, c, d)
                # capture the DIRECTION of your correction (the 3 error classes the model must learn):
                #  len_ratio>1 -> my trace was TOO SHORT (#1);  drawn_count>orig_count -> I joined
                #  separate myotubes into one (#2/#3);  drawn_count<orig_count -> I over-split.
                orig = c.get("trace_polys", [])
                drawn = d.get("polys", [])
                o_len = sum(_plen(p) for p in orig)
                d_len = sum(_plen(p) for p in drawn)
                redraw_rows.append({
                    "stem": a.stem, "case_id": c["id"], "type": ctype, "action": action,
                    "orig_count": len(orig), "drawn_count": len(drawn),
                    "orig_len_px": round(o_len, 1), "drawn_len_px": round(d_len, 1),
                    "len_ratio": round(d_len / o_len, 3) if o_len > 0 else 0.0,
                    "labels": "|".join(str(x) for x in d.get("labels", []))})
                if lab is None:
                    seen.add(key)
                    continue
                action = inferred_action
                inferred[ctype] += 1
            row = {"stem": a.stem, "case_id": c["id"], "action": action, "label": lab}
            row.update({k: feats.get(k, 0.0) for k in FEATURE_KEYS[ctype]})
            new_rows.append(row)
            seen.add(key)
        if new_rows:
            header = ["stem", "case_id", "action", "label"] + FEATURE_KEYS[ctype]
            write_header = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=header)
                if write_header:
                    w.writeheader()
                w.writerows(new_rows)
            added[ctype] = len(new_rows)

    if redraw_rows:                                  # record your hand-drawn corrections separately
        rpath = os.path.join(DATA_DIR, "redraws.csv")
        seen_r = existing_keys(rpath)
        rows = [r for r in redraw_rows if (r["stem"], r["case_id"]) not in seen_r]
        if rows:
            wh = not os.path.exists(rpath)
            with open(rpath, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["stem", "case_id", "type", "action", "orig_count",
                                                   "drawn_count", "orig_len_px", "drawn_len_px",
                                                   "len_ratio", "labels"])
                if wh:
                    w.writeheader()
                w.writerows(rows)

    print(f"feedback logged: split+{added['split']} merge+{added['merge']} "
          f"occluded+{added['occluded']} redraw+{len(redraw_rows)} "
          f"(redraw_inferred: split+{inferred['split']} merge+{inferred['merge']} "
          f"occluded+{inferred['occluded']})")


if __name__ == "__main__":
    main()
