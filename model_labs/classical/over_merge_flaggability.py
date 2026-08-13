"""How many accepted merges can the over-merge rule even SEE?

Motivation
----------
`linker_instance_v1.json` reports `over_merge_count = 3` at the locked threshold and
I recommended the linker partly on the strength of that number being small. But the
rule needs **two or more** reviewed reference masks each covering >=20% of a
prediction, and the eval GT is a sparse reviewed subset (~60 masks per well against
~880 predictions). A merge with zero or one overlapping reference **cannot be
flagged no matter how wrong it is**.

So "3" is only a rate if the denominator is the number of merges the rule could
examine. This computes that denominator directly from the extraction artifacts.

Result on the two wells that contain every flagged case (2026-07-29): of **216**
accepted merges, **213 (98.6%) are ineligible** -- 0 or 1 reference at >=20%. Only
**3 (1.4%) are eligible, and all 3 were flagged.** The detector's flag rate among
merges it can see is 3/3, so `over_merge_count = 3` is a **ceiling, not a
measurement**, and `over_merge_rate = 3/3807 predictions` is meaningless.

Usage::

    $env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
    python model_labs/classical/over_merge_flaggability.py \
      --cases model_labs/classical/_runs/over_merges_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

COVERAGE_THRESHOLD = 0.2        # precision_myotube.benchmark default


def eligibility(merged: np.ndarray, kept_only: np.ndarray, references,
                coverage_threshold: float = COVERAGE_THRESHOLD) -> dict:
    """Per accepted merge, how many reference masks cover >= the threshold of it.

    An "accepted merge" is a merged component that absorbed more than one original
    fragment. Returns counts bucketed at 0 / 1 / >=2 references, where **>=2 is the
    only bucket the over-merge rule can flag**.
    """
    labels, counts = np.unique(merged[merged > 0], return_counts=True)
    area = {int(label): int(count) for label, count in zip(labels, counts)}

    hits = {label: 0 for label in area}
    for ref in references:
        r0, c0, r1, c1 = ref["bbox"]
        window = merged[r0:r1, c0:c1]
        inside = window[ref["mask"]]
        present, overlap = np.unique(inside[inside > 0], return_counts=True)
        for label, inter in zip(present, overlap):
            if int(inter) / area[int(label)] >= coverage_threshold:
                hits[int(label)] += 1

    merges = [label for label in area
              if int(np.unique(kept_only[merged == label]).size) >= 2]
    buckets = {0: 0, 1: 0, 2: 0}
    for label in merges:
        buckets[min(hits[label], 2)] += 1
    return {"n_merges": len(merges), "buckets": buckets,
            "n_eligible": buckets[2],
            "n_ineligible": buckets[0] + buckets[1],
            "eligible_fraction": (buckets[2] / len(merges)) if merges else None}


def load_references(src: Path, well: str) -> list[dict]:
    meta = json.loads((src / f"{well}.references.json").read_text(encoding="utf-8"))
    packed = np.load(src / f"{well}.references.npz")
    out = []
    for rid, entry in meta.items():
        r0, c0, r1, c1 = entry["bbox"]
        shape = (r1 - r0, c1 - c0)
        bits = np.unpackbits(packed[rid])[:shape[0] * shape[1]]
        out.append({"id": rid, "bbox": (r0, c0, r1, c1),
                    "mask": bits.reshape(shape).astype(bool)})
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", default="model_labs/classical/_runs/over_merges_v1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    src = ROOT / args.cases if not Path(args.cases).is_absolute() else Path(args.cases)
    cases = json.loads((src / "cases.json").read_text(encoding="utf-8"))
    flagged_per_well = cases["published_over_merge_counts"]

    per_well, totals = [], {0: 0, 1: 0, 2: 0}
    for entry in cases["wells"]:
        well = entry["well"]
        arrays = np.load(src / f"{well}.arrays.npz")
        result = eligibility(arrays["merged"], arrays["kept_only"],
                             load_references(src, well))
        result["well"] = well
        result["n_flagged"] = flagged_per_well.get(well)
        per_well.append(result)
        for k, v in result["buckets"].items():
            totals[k] += v
        print(f"{well}")
        print(f"  accepted merges (>1 fragment)   : {result['n_merges']}")
        print(f"    0 references at >={COVERAGE_THRESHOLD:.0%}          : "
              f"{result['buckets'][0]}")
        print(f"    1 reference  at >={COVERAGE_THRESHOLD:.0%}          : "
              f"{result['buckets'][1]}")
        print(f"    >=2 -> ELIGIBLE to be flagged : {result['n_eligible']} "
              f"({result['eligible_fraction']:.1%})")
        print(f"    actually flagged              : {result['n_flagged']}")
        del arrays

    n_merges = sum(totals.values())
    n_flagged = sum(v for v in flagged_per_well.values())
    summary = {
        "coverage_threshold": COVERAGE_THRESHOLD,
        "n_accepted_merges": n_merges,
        "n_ineligible": totals[0] + totals[1],
        "ineligible_fraction": (totals[0] + totals[1]) / n_merges if n_merges else None,
        "n_eligible": totals[2],
        "eligible_fraction": totals[2] / n_merges if n_merges else None,
        "n_flagged": n_flagged,
        "flag_rate_among_eligible": (n_flagged / totals[2]) if totals[2] else None,
        "reading": ("over_merge_count is a CEILING set by how many merges have >=2 "
                    "reviewed reference masks, not a rate over predictions"),
    }
    print()
    print(f"BOTH WELLS: {n_merges} accepted merges; {summary['n_ineligible']} "
          f"({summary['ineligible_fraction']:.1%}) cannot be flagged at all; "
          f"{totals[2]} eligible; {n_flagged} flagged "
          f"-> flag rate among eligible = {summary['flag_rate_among_eligible']:.2f}")
    payload = {"per_well": per_well, "summary": summary,
               "evidence_class": "development_bootstrap_single_operator_"
                                 "proposal_conditioned_retrospective"}
    if args.out:
        out_path = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
