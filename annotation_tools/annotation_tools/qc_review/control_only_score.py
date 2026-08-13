"""Score a control-only accepted-merge safety round.

The previous two rounds asked "did the reviewer agree with the flag?", and their
scorer is built around that flagged-versus-control contrast. This round has no
flagged cases by construction, so that contrast does not exist and the reading
rules are different: the question is the population rate at which the linker's
accepted merges join distinct myotubes.

Everything the primary estimate does was fixed before the reviewer saw the page
and is carried in the key under ``control_only_round``. This module reads that
spec rather than restating it, so a scorer that drifts from what was promised
fails loudly instead of quietly reporting something else.

One thing was **not** predeclared: what to do with a case the reviewer left
undecided. That is disclosed in the output rather than silently resolved, and the
sensitivity bounds below cover every possible assignment of it.
"""
from __future__ import annotations

import numpy as np


RESOLVED = ("same_myotube", "different_myotubes")
UNRESOLVED = "ambiguous_2d"


def _rate(verdicts: list[int]) -> float | None:
    return float(np.mean(verdicts)) if verdicts else None


def weighted_population_rate(strata: list[dict]) -> float | None:
    """sum_w (population_w * rate_w) / sum_w population_w, over wells with data.

    Equal draws per well over unequal wells: the unweighted mean of the six well
    rates would weight a 70-merge well like a 109-merge one. A well with no
    resolved verdict contributes nothing and is dropped from the denominator too,
    which is why the caller reports how many wells carried the estimate.
    """
    num = den = 0.0
    for stratum in strata:
        if stratum["rate"] is None:
            continue
        num += stratum["population"] * stratum["rate"]
        den += stratum["population"]
    return num / den if den else None


def stratified_bootstrap(strata: list[dict], *, resamples: int, seed: int,
                         percentiles=(2.5, 97.5)) -> dict:
    """Resample whole wells, then merges within each drawn well.

    Two levels because both are real sources of variance: which wells you happened
    to have, and which merges you happened to draw inside them. With only six
    wells the outer level dominates and the interval is wide -- that width is the
    honest consequence of six wells, not a defect to be tuned away.
    """
    usable = [s for s in strata if s["verdicts"]]
    if not usable:
        return {"lower": None, "upper": None, "resamples": 0}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        picked = rng.integers(0, len(usable), len(usable))
        num = den = 0.0
        for index in picked:
            stratum = usable[int(index)]
            verdicts = stratum["verdicts"]
            inner = rng.integers(0, len(verdicts), len(verdicts))
            num += stratum["population"] * float(np.mean(
                [verdicts[int(j)] for j in inner]))
            den += stratum["population"]
        draws.append(num / den)
    low, high = np.percentile(draws, percentiles)
    return {"lower": round(float(low), 4), "upper": round(float(high), 4),
            "resamples": len(draws)}


def score_control_only(key: dict, export: dict) -> dict:
    """Join a control-only export against its key under the predeclared spec."""
    spec = key.get("control_only_round")
    if not spec:
        raise SystemExit(
            "this key carries no control_only_round block, so no estimator was "
            "predeclared for it. Score it with score-over-merge-review instead, or "
            "rebuild the packet with --control-only.")
    if export.get("batch_id") != key.get("batch_id"):
        raise SystemExit("batch_id mismatch between export and key")
    if export.get("threshold") != key.get("threshold"):
        raise SystemExit(f"threshold mismatch: export {export.get('threshold')} vs key "
                         f"{key.get('threshold')}; these are not the same packet")
    if key.get("n_over_merge_cases"):
        raise SystemExit(
            f"{key['n_over_merge_cases']} flagged case(s) in this packet; a control-only "
            "population rate must not be computed over a flag-enriched sample")

    entries, decisions = key["key"], export["decisions"]
    populations = {s["well"]: s["accepted_merges_in_well"] for s in spec["strata"]}

    by_well: dict[str, dict] = {
        well: {"well": well, "population": population, "verdicts": [],
               "same_myotube": 0, "different_myotubes": 0,
               "unresolved": 0, "undecided": 0}
        for well, population in populations.items()
    }
    undecided, unresolved, missing = [], [], []
    for uid, meta in entries.items():
        got = decisions.get(uid)
        if got is None:
            missing.append(uid)
            continue
        stratum = by_well[meta["well"]]
        verdict = got.get("decision")
        if verdict in RESOLVED:
            stratum["verdicts"].append(1 if verdict == "different_myotubes" else 0)
            stratum[verdict] += 1
        elif verdict == UNRESOLVED:
            stratum["unresolved"] += 1
            unresolved.append(uid)
        else:
            stratum["undecided"] += 1
            undecided.append(uid)
    if missing:
        raise SystemExit(f"{len(missing)} key uid(s) absent from the export ({missing[:3]}); "
                         "the export does not cover this packet")

    strata = []
    for well in sorted(by_well):
        stratum = dict(by_well[well])
        stratum["n_resolved"] = len(stratum["verdicts"])
        stratum["rate"] = _rate(stratum["verdicts"])
        strata.append(stratum)

    point = weighted_population_rate(strata)
    interval = stratified_bootstrap(
        strata, resamples=_int_from(spec["interval"], "resamples", 10000),
        seed=_int_from(spec["interval"], "seed", 20260731))

    population_total = sum(populations.values())
    n_resolved = sum(s["n_resolved"] for s in strata)
    n_different = sum(s["different_myotubes"] for s in strata)

    # Every assignment of the cases the predeclared rate excludes. If the whole
    # conclusion survives both extremes, the exclusion rule is not load-bearing and
    # nobody has to take the handling of `ambiguous_2d` on trust.
    n_excluded = len(unresolved) + len(undecided)
    floor_strata, ceiling_strata = [], []
    for stratum in strata:
        extra = stratum["unresolved"] + stratum["undecided"]
        floor_strata.append({**stratum, "verdicts": stratum["verdicts"] + [0] * extra,
                             "rate": _rate(stratum["verdicts"] + [0] * extra)})
        ceiling_strata.append({**stratum, "verdicts": stratum["verdicts"] + [1] * extra,
                               "rate": _rate(stratum["verdicts"] + [1] * extra)})

    return {
        "batch_id": key["batch_id"],
        "reviewer": export.get("reviewer"),
        "threshold": key["threshold"],
        "threshold_status": key.get("threshold_status"),
        "predeclared": spec,
        "counts": {
            "n_cases": len(entries),
            "n_resolved": n_resolved,
            "n_different_myotubes": n_different,
            "n_same_myotube": n_resolved - n_different,
            "n_unresolved_ambiguous_2d": len(unresolved),
            "n_undecided": len(undecided),
            "undecided_uids": undecided,
        },
        "primary": {
            "estimator": spec["estimator"],
            "population_over_merge_rate": (round(point, 4) if point is not None else None),
            "ci95_stratified_bootstrap": interval,
            "wells_contributing": sum(1 for s in strata if s["rate"] is not None),
            "accepted_merges_across_six_wells": population_total,
            "implied_over_merges": (round(point * population_total)
                                    if point is not None else None),
            "naive_pooled_rate": (round(n_different / n_resolved, 4)
                                  if n_resolved else None),
            "naive_pooled_note": (
                "unweighted pool of all resolved verdicts; reported only to show the "
                "weighting is not doing the work, never as the population estimate"),
        },
        "sensitivity_to_excluded_cases": {
            "n_excluded": n_excluded,
            "handling": spec["unresolved_handling"],
            "undecided_handling_was_predeclared": False,
            "undecided_handling_note": (
                "The predeclared spec covered ambiguous_2d but not a case left with no "
                "decision at all. Undecided cases are excluded here on the same "
                "reasoning, and the bounds below cover every alternative."),
            "rate_if_all_excluded_were_same_myotube": _round(
                weighted_population_rate(floor_strata)),
            "rate_if_all_excluded_were_different": _round(
                weighted_population_rate(ceiling_strata)),
        },
        "per_well": [
            {k: v for k, v in s.items() if k != "verdicts"} | {"rate": _round(s["rate"])}
            for s in strata
        ],
    }


def _round(value, digits: int = 4):
    return None if value is None else round(float(value), digits)


def _int_from(text: str, label: str, default: int) -> int:
    """Pull the predeclared resample count / seed out of the spec's prose.

    The interval was promised in words in the key. Parsing it back keeps the scorer
    honest to the promise instead of carrying a second copy that can drift.
    """
    import re
    if label == "resamples":
        found = re.search(r"(\d+)\s*resamples", text)
    else:
        found = re.search(r"seed\s*(\d+)", text)
    return int(found.group(1)) if found else default
