"""TA03b part 2 - 2-D-call versus 3-D-reference scoring harness.

Built against the **2026-08-12 relocalization amendment** in
`coordination/reports/codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md` section 4, which
superseded the original per-nucleus stage-targeting design. Direct targeting of individual
nuclei is not authorized: the pixel-to-stage affine is absent and frame/event metadata
conflict, so `selector.relocalization_feasible` stays False and raw stage metadata alone
must not flip it. What is authorized is whole-field or registered-mosaic 3-D reacquisition,
DAPI-based registration, and prespecified one-to-one post-hoc nucleus matching.

Three rules from that amendment are enforced here rather than documented:

**Desmin may not touch the match.** Registration and matching are DAPI/nucleus geometry
only. Letting the Desmin channel choose a transform or break a matching tie would let the
quantity under test select its own reference, which is the one way this harness could
manufacture agreement. :func:`match_nuclei` never receives a Desmin value, and
:func:`score` verifies the transform record declares a nucleus-channel basis.

**Attrition is preserved, never replaced.** Unmatched, duplicate, split and ungradable
selected nuclei are retained with reasons and counted. A selected nucleus that vanishes
silently would bias the weighted estimate by exactly the amount that made it hard.

**Intervals are field-clustered, never nucleus-binomial.** Nuclei within a field are
correlated; a nucleus-level binomial interval would be inferentially wrong and is refused
outright rather than offered as an option.

Nothing here fits a threshold. Calibration and validation partitions must be disjoint and
the run fails if they overlap.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field as dc_field
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .selector import STRATA, SelectionError, sha256_file

SCHEMA_VERSION = "tier_a_scoring/1.0"
RNG_ALGORITHM = "numpy.random.Generator(PCG64)"

# The four adverse two-sided 95% bounds, ratified 2026-07-23. Not tunable here: the
# harness evaluates them, it does not get to choose them.
GATES = {
    "specificity":        {"side": "lower", "threshold": 0.95, "co_primary": True},
    "sensitivity":        {"side": "lower", "threshold": 0.90, "co_primary": True},
    "fp_inflation":       {"side": "upper", "threshold": 0.10, "co_primary": True},
    "negative_control_fpr": {"side": "upper", "threshold": 0.05, "co_primary": False},
}

ATTRITION_REASONS = ("unmatched_no_candidate", "unmatched_below_overlap",
                     "unmatched_beyond_distance", "duplicate_candidate",
                     "split_reference", "ungradable_reference",
                     "field_registration_failed", "field_not_reacquired")


class ScoringError(RuntimeError):
    """Fail-closed condition."""


# ------------------------------------------------------------------------- matching


@dataclass(frozen=True)
class MatchRule:
    """Prespecified and recorded, per amendment item 4. Both criteria must pass."""
    max_centroid_distance_px: float
    min_mask_overlap: float          # IoU of the 2-D nucleus mask against the projected 3-D mask

    def validate(self) -> None:
        if not (self.max_centroid_distance_px > 0):
            raise ScoringError("max_centroid_distance_px must be positive")
        if not (0.0 < self.min_mask_overlap <= 1.0):
            raise ScoringError("min_mask_overlap must be in (0, 1]")


@dataclass
class MatchOutcome:
    matched: list[dict] = dc_field(default_factory=list)
    attrition: list[dict] = dc_field(default_factory=list)

    def attrition_counts(self) -> dict[str, int]:
        out = {r: 0 for r in ATTRITION_REASONS}
        for a in self.attrition:
            out[a["reason"]] = out.get(a["reason"], 0) + 1
        return out


def match_nuclei(selected: list[dict], candidates: dict[tuple[str, str, int], list[dict]],
                 rule: MatchRule) -> MatchOutcome:
    """One-to-one match of selected 2-D nuclei onto registered 3-D reference nuclei.

    `candidates[(well, field, nucleus_id)]` is the list of registered reference nuclei
    near that selected nucleus, each a dict with `reference_id`, `distance_px`,
    `mask_overlap`, and `truth_positive` / `ungradable`. Distances and overlaps must have
    been computed from the nucleus channel; no Desmin value is accepted here at all.

    One-to-one is enforced in both directions. A reference nucleus already claimed by a
    closer selected nucleus cannot be reused -- that would let one 3-D object certify two
    2-D calls. Ties on distance are broken by higher overlap, then by reference id, so the
    result does not depend on dict ordering.
    """
    rule.validate()
    out = MatchOutcome()
    claimed: dict[str, tuple[str, str, int]] = {}

    def sort_key(c):
        return (c["distance_px"], -c["mask_overlap"], str(c["reference_id"]))

    # Process selected nuclei in a stable order so claiming is deterministic.
    ordered = sorted(selected, key=lambda s: (s["well"], s["field"], int(s["nucleus_id"])))
    # Global best-first assignment: sort all (selected, candidate) pairs by quality.
    pairs = []
    for s in ordered:
        key = (s["well"], s["field"], int(s["nucleus_id"]))
        for c in candidates.get(key, []):
            if "desmin" in json.dumps(c).lower():
                raise ScoringError(
                    f"candidate for {key} carries a Desmin field; registration and "
                    "matching must use nucleus geometry only")
            pairs.append((sort_key(c), key, s, c))
    pairs.sort(key=lambda p: (p[0], p[1]))

    taken_sel: set[tuple[str, str, int]] = set()
    for _, key, s, c in pairs:
        if key in taken_sel:
            continue
        if c["distance_px"] > rule.max_centroid_distance_px:
            continue
        if c["mask_overlap"] < rule.min_mask_overlap:
            continue
        rid = str(c["reference_id"])
        if rid in claimed:
            out.attrition.append({**_ident(s), "reason": "duplicate_candidate",
                                  "detail": f"reference {rid} already claimed by "
                                            f"{claimed[rid]}"})
            continue
        if c.get("ungradable"):
            out.attrition.append({**_ident(s), "reason": "ungradable_reference",
                                  "detail": f"reference {rid} marked ungradable"})
            taken_sel.add(key)
            continue
        if c.get("split_reference"):
            out.attrition.append({**_ident(s), "reason": "split_reference",
                                  "detail": f"reference {rid} is a split object"})
            taken_sel.add(key)
            continue
        claimed[rid] = key
        taken_sel.add(key)
        out.matched.append({**_ident(s),
                            "inclusion_probability": float(s["inclusion_probability"]),
                            "stratum": s["stratum"],
                            "call_2d": bool(s["call_2d"]),
                            "truth_3d": bool(c["truth_positive"]),
                            "reference_id": rid,
                            "distance_px": float(c["distance_px"]),
                            "mask_overlap": float(c["mask_overlap"])})

    for s in ordered:
        key = (s["well"], s["field"], int(s["nucleus_id"]))
        if key in taken_sel:
            continue
        cands = candidates.get(key, [])
        if not cands:
            reason = "unmatched_no_candidate"
        elif all(c["distance_px"] > rule.max_centroid_distance_px for c in cands):
            reason = "unmatched_beyond_distance"
        else:
            reason = "unmatched_below_overlap"
        out.attrition.append({**_ident(s), "reason": reason,
                              "detail": f"{len(cands)} candidate(s) considered"})
    return out


def _ident(s: dict) -> dict:
    return {"well": s["well"], "field": s["field"], "nucleus_id": int(s["nucleus_id"])}


# -------------------------------------------------------------------------- metrics


def _cells(rows: list[dict], weights: list[float]) -> dict[str, float]:
    tp = fp = tn = fn = 0.0
    for r, w in zip(rows, weights):
        if r["truth_3d"] and r["call_2d"]:
            tp += w
        elif r["truth_3d"] and not r["call_2d"]:
            fn += w
        elif (not r["truth_3d"]) and r["call_2d"]:
            fp += w
        else:
            tn += w
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _ratio(num: float, den: float) -> float | None:
    return None if den <= 0 else num / den


def metrics_from_cells(c: dict[str, float]) -> dict[str, float | None]:
    sens = _ratio(c["tp"], c["tp"] + c["fn"])
    spec = _ratio(c["tn"], c["tn"] + c["fp"])
    ppv = _ratio(c["tp"], c["tp"] + c["fp"])
    npv = _ratio(c["tn"], c["tn"] + c["fn"])
    return {"sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "fp_inflation": None if ppv is None else 1.0 - ppv,
            **{k: c[k] for k in ("tp", "fp", "tn", "fn")}}


def design_weights(rows: list[dict], population_mixture: dict[str, float] | None
                   ) -> list[float]:
    """Horvitz-Thompson design weights, optionally post-stratified to a target mixture.

    The base weight is 1/inclusion_probability, which undoes the deliberate boundary
    oversampling. `population_mixture` is the **preregistered** target share per stratum;
    without it the estimate describes the sampled mixture, which is not the population and
    must not be reported as a primary result. :func:`score` requires it.
    """
    base = []
    for r in rows:
        p = float(r["inclusion_probability"])
        if not (0.0 < p <= 1.0):
            raise ScoringError(f"inclusion probability {p!r} outside (0,1] for {_ident(r)}")
        base.append(1.0 / p)
    if population_mixture is None:
        return base
    known = {s for s, _, _ in STRATA}
    unknown = set(population_mixture) - known
    if unknown:
        raise ScoringError(f"population_mixture names unknown strata: {sorted(unknown)}")
    total = sum(population_mixture.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ScoringError(f"population_mixture must sum to 1.0, got {total!r}")
    by_stratum: dict[str, float] = {}
    for r, w in zip(rows, base):
        by_stratum[r["stratum"]] = by_stratum.get(r["stratum"], 0.0) + w
    out = []
    for r, w in zip(rows, base):
        target = population_mixture.get(r["stratum"], 0.0)
        denom = by_stratum.get(r["stratum"], 0.0)
        out.append(0.0 if denom <= 0 else w * target / denom)
    return out


# ------------------------------------------------------------------------ intervals


def field_cluster_bootstrap(rows: list[dict], population_mixture: dict[str, float] | None,
                            *, n_boot: int, seed: int,
                            ) -> dict[str, tuple[float | None, float | None]]:
    """Percentile CIs by resampling whole fields with replacement.

    Fields are the cluster, not nuclei: nuclei within a field share staining, focus and
    registration error, so a nucleus-level interval understates uncertainty. Resampling is
    deterministic given the seed, and fields are resampled as intact blocks so a field
    drawn twice contributes all of its nuclei twice.
    """
    if n_boot < 1:
        raise ScoringError("n_boot must be >= 1")
    fields = sorted({(r["well"], r["field"]) for r in rows})
    if len(fields) < 2:
        raise ScoringError(
            f"field-cluster bootstrap needs >=2 fields, got {len(fields)}. A "
            "nucleus-level interval is not an acceptable substitute.")
    by_field: dict[tuple[str, str], list[dict]] = {f: [] for f in fields}
    for r in rows:
        by_field[(r["well"], r["field"])].append(r)

    keys = ("sensitivity", "specificity", "ppv", "npv", "fp_inflation")
    draws: dict[str, list[float]] = {k: [] for k in keys}
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(n_boot):
        pick = rng.integers(0, len(fields), size=len(fields))
        sample = [r for i in pick for r in by_field[fields[i]]]
        if not sample:
            continue
        try:
            w = design_weights(sample, population_mixture)
        except ScoringError:
            continue
        m = metrics_from_cells(_cells(sample, w))
        for k in keys:
            if m[k] is not None:
                draws[k].append(m[k])
    out = {}
    for k in keys:
        v = sorted(draws[k])
        if len(v) < 2:
            out[k] = (None, None)
        else:
            lo = v[max(0, int(math.floor(0.025 * (len(v) - 1))))]
            hi = v[min(len(v) - 1, int(math.ceil(0.975 * (len(v) - 1))))]
            out[k] = (lo, hi)
    return out


def evaluate_gates(point: dict, ci: dict, negative_control: dict | None) -> dict:
    """Apply the four ratified adverse bounds. Missing evidence fails, never passes."""
    res = {}
    for name, spec in GATES.items():
        if name == "negative_control_fpr":
            if negative_control is None:
                res[name] = {"status": "not_evaluable", "passed": False,
                             "reason": "no negative-control material supplied",
                             **spec}
                continue
            bound = negative_control.get("upper95")
            val = negative_control.get("point")
        else:
            lo, hi = ci.get(name, (None, None))
            bound = lo if spec["side"] == "lower" else hi
            val = point.get(name)
        if bound is None:
            res[name] = {"status": "not_evaluable", "passed": False,
                         "reason": "interval unavailable", "point": val, **spec}
            continue
        ok = bound >= spec["threshold"] if spec["side"] == "lower" else bound <= spec["threshold"]
        res[name] = {"status": "evaluated", "passed": bool(ok), "point": val,
                     "adverse_bound": bound, **spec}
    co = [k for k, v in GATES.items() if v["co_primary"]]
    res["_co_primary_all_passed"] = all(res[k]["passed"] for k in co)
    res["_all_passed"] = all(res[k]["passed"] for k in GATES)
    return res


# --------------------------------------------------------------------------- report


def score(selection_manifest: dict, matched: list[dict], attrition: list[dict], *,
          population_mixture: dict[str, float], transform_record: dict,
          calibration_ids: set[tuple[str, str, int]] | None = None,
          negative_control: dict | None = None,
          n_boot: int = 2000, seed: int = 0) -> dict:
    """Confirmatory scoring. Fails closed on every contract violation it can detect."""
    if not matched:
        raise ScoringError("no matched nuclei; nothing to score")

    basis = str(transform_record.get("basis", "")).lower()
    if "dapi" not in basis and "nucle" not in basis:
        raise ScoringError(
            f"transform basis {transform_record.get('basis')!r} is not a nucleus channel; "
            "the amendment forbids choosing the transform from Desmin")
    for k in ("model", "residual_px", "source_hashes"):
        if k not in transform_record:
            raise ScoringError(f"transform_record is missing required field {k!r}")

    ids = [(m["well"], m["field"], int(m["nucleus_id"])) for m in matched]
    if len(set(ids)) != len(ids):
        raise ScoringError("matched set contains duplicate selected nuclei")
    refs = [m["reference_id"] for m in matched]
    if len(set(refs)) != len(refs):
        raise ScoringError("matched set reuses a reference nucleus; one-to-one violated")

    if calibration_ids:
        overlap = sorted(set(ids) & set(calibration_ids))
        if overlap:
            raise ScoringError(
                f"calibration/validation overlap on {len(overlap)} nuclei, e.g. "
                f"{overlap[:3]}. Threshold fitting and confirmatory validation must be "
                "disjoint.")

    weights = design_weights(matched, population_mixture)
    point = metrics_from_cells(_cells(matched, weights))
    ci = field_cluster_bootstrap(matched, population_mixture, n_boot=n_boot, seed=seed)
    gates = evaluate_gates(point, ci, negative_control)

    def subgroup(key: str) -> dict:
        out = {}
        for val in sorted({m[key] for m in matched}):
            rows = [m for m in matched if m[key] == val]
            w = design_weights(rows, population_mixture)
            out[str(val)] = {"n": len(rows), "weighted": metrics_from_cells(_cells(rows, w)),
                             "unweighted_diagnostic": metrics_from_cells(
                                 _cells(rows, [1.0] * len(rows)))}
        return out

    att = {r: 0 for r in ATTRITION_REASONS}
    for a in attrition:
        att[a["reason"]] = att.get(a["reason"], 0) + 1

    report = {
        "schema_version": SCHEMA_VERSION,
        "task": "TA03b", "stage": "scoring",
        "contract": "2026-08-12 relocalization amendment; whole-field/mosaic 3-D "
                    "reacquisition, DAPI registration, one-to-one post-hoc matching",
        "selection_manifest_sha256": selection_manifest.get("manifest_sha256"),
        "transform": transform_record,
        "population_mixture": population_mixture,
        "n_selected": selection_manifest.get("n_selected"),
        "n_matched": len(matched),
        "attrition": att,
        "attrition_total": len(attrition),
        "match_rate": len(matched) / max(1, len(matched) + len(attrition)),
        "primary_weighted": point,
        "confidence_intervals_95": {k: list(v) for k, v in ci.items()},
        "interval_method": "percentile bootstrap over whole fields, "
                           f"{n_boot} draws, {RNG_ALGORITHM} seed {seed}",
        "gates": gates,
        "by_stratum": subgroup("stratum"),
        "by_field": subgroup("field"),
        "by_well": subgroup("well"),
        "unweighted_note": "unweighted figures are stratum diagnostics only and are not "
                           "inferential; primary results are weighted",
        "limitations": [
            "single plate, single operator, proposal-conditioned",
            "direct per-nucleus stage targeting is not authorized; this is field "
            "registration with post-hoc matching",
            "attrition is retained, not replaced; a low match rate invalidates the "
            "population weighting regardless of the metric values",
        ],
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, default=str).encode()).hexdigest()
    return report


def write_report(out_dir: Path, report: dict, matched: list[dict],
                 attrition: list[dict]) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "scoring_report.json"
    jp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    for name, rows in (("matched.csv", matched), ("attrition.csv", attrition)):
        if not rows:
            continue
        with (out_dir / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "scoring_report.txt").write_text(human_readable(report), encoding="utf-8")
    return {"json": str(jp), "report_sha256": report["report_sha256"]}


def human_readable(report: dict) -> str:
    p, g = report["primary_weighted"], report["gates"]
    ci = report["confidence_intervals_95"]
    # ASCII only: this is printed to a console, and Windows cp1252 mangles or raises on
    # non-ASCII. The JSON report is UTF-8 and unaffected.
    L = ["Tier-A 2-D-versus-3-D validation - weighted, field-clustered", "=" * 62,
         f"matched {report['n_matched']} of {report['n_selected']} selected "
         f"(match rate {report['match_rate']:.3f})", ""]
    for k in ("sensitivity", "specificity", "ppv", "npv", "fp_inflation"):
        lo, hi = ci.get(k, (None, None))
        v = p.get(k)
        L.append(f"  {k:16s} {'n/a' if v is None else f'{v:.4f}'}   "
                 f"95% CI [{'n/a' if lo is None else f'{lo:.4f}'}, "
                 f"{'n/a' if hi is None else f'{hi:.4f}'}]")
    L += ["", "adverse-bound gates (ratified 2026-07-23, not tunable):"]
    for name, spec in GATES.items():
        r = g[name]
        mark = "PASS" if r["passed"] else "FAIL"
        L.append(f"  [{mark}] {name:22s} {spec['side']} bound vs {spec['threshold']}"
                 f"{'' if r['status'] == 'evaluated' else '  (' + r.get('reason', '') + ')'}")
    L += ["", f"co-primary gates all passed: {g['_co_primary_all_passed']}",
          f"all gates passed:            {g['_all_passed']}", "",
          "attrition:"]
    for k, v in report["attrition"].items():
        if v:
            L.append(f"  {k:28s} {v}")
    return "\n".join(L) + "\n"
