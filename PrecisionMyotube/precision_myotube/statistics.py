"""Hierarchical statistical summaries with explicit experimental units.

The project measures many nuclei and myotubes inside wells, but those objects are
not automatically independent biological replicates.  This module collapses raw
observations to technical units and then to user-declared biological units before
estimating group effects.  It deliberately does not manufacture p-values from a
single plate or from thousands of nested objects.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260722
MIN_INDEPENDENT_UNITS = 3


def _finite_float(value, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return number


def _validate_confidence(confidence: float) -> float:
    confidence = _finite_float(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return confidence


def _aggregate(values: Sequence[float], method: str) -> float:
    if not values:
        raise ValueError("cannot aggregate an empty sequence")
    if method == "mean":
        return float(np.mean(values))
    if method == "median":
        return float(np.median(values))
    raise ValueError("aggregation must be 'mean' or 'median'")


def wilson_interval(successes: int, total: int,
                    confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """Wilson score interval for a descriptive binary proportion.

    This interval describes counting uncertainty within the supplied unit.  It
    must not be presented as biological-replicate uncertainty when objects are
    clustered within wells or plates.
    """
    confidence = _validate_confidence(confidence)
    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("successes and total must be integer counts")
    if int(successes) != successes or int(total) != total:
        raise ValueError("successes and total must be integer counts")
    successes, total = int(successes), int(total)
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return {
        "estimate": p,
        "lower": max(0.0, center - half),
        "upper": min(1.0, center + half),
        "confidence": confidence,
        "method": "Wilson score interval",
        "unit_warning": (
            "descriptive count interval only; nested objects are not biological replicates"
        ),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted p-values in original order.

    The function is available for pre-specified families of confirmatory tests.
    The canonical default remains estimation with effect sizes and confidence
    intervals; this function does not decide which hypotheses belong to a family.
    """
    values = [_finite_float(value, "p-value") for value in p_values]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("p-values must lie in [0, 1]")
    n = len(values)
    if not n:
        return []
    order = sorted(range(n), key=lambda index: values[index])
    adjusted = [0.0] * n
    running = 1.0
    for rank_index in range(n - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1
        running = min(running, values[original_index] * n / rank)
        adjusted[original_index] = min(1.0, running)
    return adjusted


def _percentile_interval(samples: Sequence[float], confidence: float) -> dict:
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(np.asarray(samples, dtype=float), [alpha, 1.0 - alpha])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "method": "percentile bootstrap over biological units",
    }


def _bootstrap_one_sample(values: Sequence[float], statistic: Callable,
                          confidence: float, resamples: int,
                          rng: np.random.Generator) -> dict | None:
    if len(values) < MIN_INDEPENDENT_UNITS:
        return None
    array = np.asarray(values, dtype=float)
    estimates = []
    for _ in range(resamples):
        sample = array[rng.integers(0, len(array), size=len(array))]
        estimates.append(float(statistic(sample)))
    return _percentile_interval(estimates, confidence)


def _bootstrap_difference(a: Sequence[float], b: Sequence[float], paired: bool,
                          confidence: float, resamples: int,
                          rng: np.random.Generator) -> dict | None:
    if paired:
        if len(a) != len(b):
            raise ValueError("paired groups must have the same number of units")
        if len(a) < MIN_INDEPENDENT_UNITS:
            return None
        differences = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
        estimates = []
        for _ in range(resamples):
            sample = differences[rng.integers(0, len(differences), size=len(differences))]
            estimates.append(float(np.mean(sample)))
    else:
        if len(a) < MIN_INDEPENDENT_UNITS or len(b) < MIN_INDEPENDENT_UNITS:
            return None
        aa, bb = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        estimates = []
        for _ in range(resamples):
            sa = aa[rng.integers(0, len(aa), size=len(aa))]
            sb = bb[rng.integers(0, len(bb), size=len(bb))]
            estimates.append(float(np.mean(sb) - np.mean(sa)))
    result = _percentile_interval(estimates, confidence)
    result["method"] = (
        "paired percentile bootstrap over biological units" if paired
        else "independent percentile bootstrap over biological units"
    )
    return result


def _collapse_records(records: Sequence[Mapping], *, value_key: str,
                      condition_key: str, biological_unit_key: str,
                      technical_unit_key: str | None,
                      technical_aggregation: str) -> tuple[dict, int]:
    """Collapse raw rows -> technical units -> biological-unit/condition values."""
    raw_by_technical: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for index, record in enumerate(records):
        missing = [key for key in (value_key, condition_key, biological_unit_key)
                   if key not in record]
        if missing:
            raise ValueError(f"record {index} is missing required keys: {missing}")
        condition = str(record[condition_key])
        biological = str(record[biological_unit_key])
        technical = (str(record[technical_unit_key]) if technical_unit_key else
                     f"row_{index}")
        raw_by_technical[(condition, biological, technical)].append(
            _finite_float(record[value_key], f"record {index}.{value_key}"))

    technical_values = {
        key: _aggregate(values, technical_aggregation)
        for key, values in raw_by_technical.items()
    }
    by_biological: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (condition, biological, _technical), value in technical_values.items():
        by_biological[(condition, biological)].append(value)
    collapsed: dict[str, dict[str, float]] = defaultdict(dict)
    for (condition, biological), values in by_biological.items():
        collapsed[condition][biological] = _aggregate(values, technical_aggregation)
    return dict(collapsed), len(technical_values)


def summarize_hierarchical(records: Sequence[Mapping], *, value_key: str,
                           condition_key: str, biological_unit_key: str,
                           technical_unit_key: str | None = None,
                           comparisons: Sequence[Mapping] = (),
                           technical_aggregation: str = "mean",
                           confidence: float = DEFAULT_CONFIDENCE,
                           bootstrap_resamples: int = DEFAULT_RESAMPLES,
                           seed: int = DEFAULT_SEED) -> dict:
    """Summarize nested observations without treating objects as independent n.

    Each condition is represented by one value per declared biological unit.
    Technical observations are collapsed first.  Comparisons report unstandardized
    mean differences (``b - a``); paired comparisons match biological-unit IDs.
    """
    confidence = _validate_confidence(confidence)
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if not records:
        raise ValueError("statistics manifest contains no records")
    collapsed, n_technical = _collapse_records(
        records, value_key=value_key, condition_key=condition_key,
        biological_unit_key=biological_unit_key,
        technical_unit_key=technical_unit_key,
        technical_aggregation=technical_aggregation)
    rng = np.random.default_rng(seed)
    groups = {}
    for condition in sorted(collapsed):
        unit_map = collapsed[condition]
        values = [unit_map[key] for key in sorted(unit_map)]
        n = len(values)
        ci = _bootstrap_one_sample(values, np.mean, confidence,
                                   bootstrap_resamples, rng)
        groups[condition] = {
            "n_biological_units": n,
            "n_is_biological_units_not_objects": True,
            "biological_unit_values": unit_map,
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "sd_between_biological_units": (
                float(np.std(values, ddof=1)) if n >= 2 else None),
            "mean_confidence_interval": ci,
            "inference_eligible": n >= MIN_INDEPENDENT_UNITS,
            "precision_warning": (
                "fewer than 3 independent biological units; descriptive only" if n < 3
                else "fewer than 5 independent biological units; interval is low precision"
                if n < 5 else None),
        }

    comparison_results = []
    for index, comparison in enumerate(comparisons):
        a, b = str(comparison.get("a", "")), str(comparison.get("b", ""))
        if a not in collapsed or b not in collapsed:
            raise ValueError(f"comparison {index} references an unknown condition")
        paired = bool(comparison.get("paired", False))
        if paired:
            ids_a, ids_b = set(collapsed[a]), set(collapsed[b])
            unit_ids = sorted(ids_a & ids_b)
            unmatched_a, unmatched_b = sorted(ids_a - ids_b), sorted(ids_b - ids_a)
            a_values = [collapsed[a][unit] for unit in unit_ids]
            b_values = [collapsed[b][unit] for unit in unit_ids]
            n_a = n_b = len(unit_ids)
        else:
            overlap = sorted(set(collapsed[a]) & set(collapsed[b]))
            if overlap:
                raise ValueError(
                    f"comparison {index} declares independent groups but biological unit IDs "
                    f"occur in both conditions: {overlap}; declare paired=true or use unique IDs")
            a_values = [collapsed[a][unit] for unit in sorted(collapsed[a])]
            b_values = [collapsed[b][unit] for unit in sorted(collapsed[b])]
            unit_ids = None
            unmatched_a = unmatched_b = []
            n_a, n_b = len(a_values), len(b_values)
        eligible = min(n_a, n_b) >= MIN_INDEPENDENT_UNITS
        effect = float(np.mean(b_values) - np.mean(a_values)) if a_values and b_values else None
        ci = _bootstrap_difference(a_values, b_values, paired, confidence,
                                   bootstrap_resamples, rng)
        comparison_results.append({
            "a": a, "b": b, "paired": paired,
            "paired_biological_unit_ids": unit_ids,
            "unmatched_biological_unit_ids_a": unmatched_a,
            "unmatched_biological_unit_ids_b": unmatched_b,
            "n_biological_units_a": n_a,
            "n_biological_units_b": n_b,
            "effect": "unstandardized mean difference (b - a)",
            "estimate": effect,
            "confidence_interval": ci,
            "inference_eligible": eligible,
            "p_value": None,
            "p_value_note": (
                "No default hypothesis test. Pre-specify a model/test and multiplicity family; "
                "report an adjusted p-value only alongside the effect and interval."
            ),
        })

    all_biological_ids = {
        unit for condition in collapsed.values() for unit in condition
    }
    return {
        "schema_version": "1.0",
        "analysis_type": "hierarchical_estimation",
        "value_key": value_key,
        "condition_key": condition_key,
        "biological_unit_key": biological_unit_key,
        "technical_unit_key": technical_unit_key,
        "technical_aggregation": technical_aggregation,
        "n_raw_observations": len(records),
        "n_technical_units": n_technical,
        "n_distinct_biological_units": len(all_biological_ids),
        "confidence": confidence,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "groups": groups,
        "comparisons": comparison_results,
        "guardrails": {
            "objects_are_not_counted_as_independent_n": True,
            "technical_units_are_collapsed_before_inference": True,
            "minimum_independent_units_for_interval": MIN_INDEPENDENT_UNITS,
            "single_plate_treatment_inference_allowed": False,
            "default_emphasis": "effect sizes and confidence intervals, not p-values",
        },
    }


def analyze_statistics_manifest(path: str | Path) -> dict:
    """Load a versioned JSON statistics manifest and return its analysis."""
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = ("value_key", "condition_key", "biological_unit_key", "records")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"statistics manifest missing required keys: {missing}")
    result = summarize_hierarchical(
        payload["records"], value_key=payload["value_key"],
        condition_key=payload["condition_key"],
        biological_unit_key=payload["biological_unit_key"],
        technical_unit_key=payload.get("technical_unit_key"),
        comparisons=payload.get("comparisons", []),
        technical_aggregation=payload.get("technical_aggregation", "mean"),
        confidence=float(payload.get("confidence", DEFAULT_CONFIDENCE)),
        bootstrap_resamples=int(payload.get("bootstrap_resamples", DEFAULT_RESAMPLES)),
        seed=int(payload.get("seed", DEFAULT_SEED)),
    )
    result["analysis_id"] = payload.get("analysis_id", manifest_path.stem)
    result["outcome_label"] = payload.get("outcome_label", payload["value_key"])
    result["pre_specified"] = bool(payload.get("pre_specified", False))
    result["exclusion_rule"] = payload.get("exclusion_rule")
    return result
