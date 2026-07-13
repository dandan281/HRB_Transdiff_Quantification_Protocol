"""Stage 4 -- reconcile flags + human decisions into the final trace set.

Reads `combined_traces.txt`, `flags.json`, and (if present) `decisions.json`, then applies:
  * merge  -> union the involved traces and concatenate into one
  * split  -> cut a trace at the chosen points (or N equal pieces)
  * reject -> drop the trace
Cases absent from decisions.json fall back to their flag's `auto` action; if no decisions.json
exists at all, only auto cases are applied. Writes `final_traces.txt` (spatially ordered) and
`reconcile_summary.json`.

Usage: python reconcile.py --out <stage4 dir>
"""
from __future__ import annotations
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from iohelpers import read_traces, write_traces, write_json  # noqa: E402
from geometry import split_at_points, equal_split_points, chain_merge, spatial_order  # noqa: E402


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def resolve_action(case, decisions):
    """The action to apply for a case: explicit decision wins, else the flag's auto/safe default."""
    d = decisions.get(case["id"])
    if d is not None:
        return d
    if case["type"] == "occluded":
        return {"action": "drop"}                       # default: stay excluded
    if case["auto"]:
        if case["type"] == "merge":
            return {"action": "merge"}
        # auto-split applies ONLY dark-gap-derived points, never kink points
        return {"action": "split", "points": case.get("gap_splits", case.get("proposed_splits", []))}
    return {"action": "separate" if case["type"] == "merge" else "keep"}


def _cut_points(op, trace):
    """Resolve a split_ops entry to concrete cut coordinates for `trace`."""
    return equal_split_points(trace, op[1]) if isinstance(op, tuple) else op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    combined = read_traces(os.path.join(a.out, "combined_traces.txt"))
    flags = json.load(open(os.path.join(a.out, "flags.json"), encoding="utf-8"))
    dpath = os.path.join(a.out, "decisions.json")
    decisions = {}
    if os.path.exists(dpath):
        decisions = json.load(open(dpath, encoding="utf-8")).get("decisions", {})

    n = len(combined)
    reject = set()
    split_ops = {}     # trace_index -> list of cut points (or ('n', k))
    redraw_polys = []  # user-drawn replacement traces (full coords)
    redraw_labels = []  # parallel: the user's per-myotube label for each redraw poly
    uf = UF(n)
    n_merge = n_split = n_reject = n_occlude_drop = n_restore = n_redraw = 0

    for case in flags["cases"]:
        act = resolve_action(case, decisions)
        a_ = act.get("action")
        if a_ == "redraw":
            # the user rejected my proposed trace(s) and drew their own labelled myotubes -> replace
            reject.update(case["trace_indices"])               # drop the proposed trace(s)
            labels = act.get("labels", [])
            for k, poly in enumerate(act.get("polys", [])):
                pts = [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]
                if len(pts) >= 2:
                    redraw_polys.append(pts)
                    redraw_labels.append(labels[k] if k < len(labels) else "")
            n_redraw += 1
            continue
        if case["type"] == "merge":
            if a_ == "merge":
                i, j = case["trace_indices"]
                uf.union(i, j); n_merge += 1
        elif case["type"] == "occluded":
            i = case["trace_indices"][0]
            if a_ in ("restore", "keep"):
                n_restore += 1                          # leave it in (becomes a real fibre)
            else:
                reject.add(i); n_occlude_drop += 1      # default drop
        elif case["type"] == "split":
            i = case["trace_indices"][0]
            if a_ == "reject":
                reject.add(i); n_reject += 1
            elif a_ == "split_n" and act.get("n", 0) > 1:
                split_ops[i] = ("n", int(act["n"])); n_split += 1
            elif a_ == "split":
                pts = act.get("points") or case.get("gap_splits") or case.get("proposed_splits", [])
                if pts:
                    split_ops[i] = [tuple(p) for p in pts]; n_split += 1

    # merge groups (only multi-member groups, excluding rejected members)
    groups = {}
    for i in range(n):
        if i in reject:
            continue
        groups.setdefault(uf.find(i), []).append(i)

    final = []
    final_labels = []                           # parallel: user label or None per final trace
    consumed = set()
    conflicts = []     # traces that were BOTH merged and split: split applied to the merged chain

    def add(traces_list, label=None):
        for t in traces_list:
            final.append(t)
            final_labels.append(label)

    for root, members in groups.items():
        members = [m for m in members if m not in reject]
        if len(members) > 1:
            chain = chain_merge([combined[m] for m in members])
            chain_cuts = []
            for m in members:
                if m in split_ops:                      # honour a split on a merged member...
                    op = split_ops[m]
                    if isinstance(op, tuple):           # split_n: divide the MERGED CHAIN into N
                        chain_cuts.extend(equal_split_points(chain, op[1]))
                    else:                               # absolute points project onto the chain
                        chain_cuts.extend(op)
                    conflicts.append(m)
            add(split_at_points(chain, chain_cuts) if chain_cuts else [chain])
            consumed.update(members)

    for i in range(n):
        if i in reject or i in consumed:
            continue
        t = combined[i]
        if i in split_ops:
            add(split_at_points(t, _cut_points(split_ops[i], t)))
        else:
            add([t])

    for poly, lab in zip(redraw_polys, redraw_labels):   # the myotubes you drew yourself
        final.append(poly); final_labels.append(lab or None)

    order = spatial_order(final)
    final = [final[k] for k in order]
    final_labels = [final_labels[k] for k in order]
    nfin = write_traces(os.path.join(a.out, "final_traces.txt"), final)
    # id -> your label, for the traces you drew (id matches the overlay number + the CSV id)
    id_label = {str(i + 1): lab for i, lab in enumerate(final_labels) if lab}
    write_json(os.path.join(a.out, "drawn_labels.json"), id_label)

    summary = dict(stem=flags.get("stem"), n_combined=n, n_merge_cases=n_merge,
                   n_split_cases=n_split, n_reject=n_reject,
                   n_occlude_drop=n_occlude_drop, n_restore=n_restore,
                   n_redraw_cases=n_redraw, n_redraw_traces=len(redraw_polys),
                   n_merge_split_conflicts=len(conflicts), n_final=nfin,
                   used_decisions=os.path.exists(dpath))
    write_json(os.path.join(a.out, "reconcile_summary.json"), summary)
    print(f"combined={n} -> final={nfin}  (merges={n_merge}, splits={n_split}, rejects={n_reject}, "
          f"occlude_drop={n_occlude_drop}, restore={n_restore}, redraw={n_redraw}(+{len(redraw_polys)} traces), "
          f"conflicts={len(conflicts)}, decisions={'yes' if os.path.exists(dpath) else 'auto-only'})")


if __name__ == "__main__":
    main()
