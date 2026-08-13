"""TA03b part 3 - cluster-aware sample-size planning for Tier-A validation.

The ratification requires each boundary stratum's two-sided 95% interval to have a
half-width no greater than 10 percentage points, and is explicit that "~100 nuclei per
stratum" is a rough pilot planning input rather than a justified final sample. This module
turns that into a number, and it exists to be run **before** acquisition: the quantity it
sizes is fields, and fields are decided at the microscope.

The thing that makes this non-obvious is that nuclei are not independent. Nuclei in one
field share staining, focus, registration error and local biology, so the effective sample
size is far below the nucleus count. Planning on nucleus count alone is the standard way to
acquire a study that cannot clear its own gates -- you get the nuclei you budgeted for and
an interval twice as wide as you needed.

Two estimates are produced and cross-checked:

**Analytic**, via the design effect ``Deff = 1 + (m - 1) * ICC`` on a Wald interval. Instant,
and adequate for scanning a grid.

**Simulated**, via a beta-binomial field-level random effect scored through the same
:func:`scorer.field_cluster_bootstrap` the real analysis will use. Slower and honest about
the bootstrap's behaviour at small cluster counts, which is exactly the regime this study
is in.

Neither is a substitute for a pilot. Intracluster correlation is assumed here, not measured,
so the output is a range across plausible ICC rather than a single number. The ICC that
actually applies is a pilot deliverable, and the ratification says so.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import numpy as np

from .scorer import ScoringError, field_cluster_bootstrap

Z95 = 1.959963984540054

# Ratified: no more than 10 percentage points half-width inside a boundary stratum.
TARGET_HALF_WIDTH = 0.10
# Plausible range for imaging data. Reported across, never collapsed to one value.
ICC_GRID = (0.01, 0.05, 0.10, 0.20)


@dataclass(frozen=True)
class PlanRow:
    n_fields: int
    nuclei_per_field_per_stratum: int
    icc: float
    true_rate: float
    design_effect: float
    effective_n: float
    analytic_half_width: float
    meets_target: bool

    def as_dict(self) -> dict:
        return asdict(self)


def design_effect(m: float, icc: float) -> float:
    """Deff = 1 + (m - 1) * ICC, the standard cluster inflation."""
    if m <= 0:
        raise ScoringError("cluster size must be positive")
    if not (0.0 <= icc < 1.0):
        raise ScoringError(f"icc must be in [0, 1), got {icc!r}")
    return 1.0 + (m - 1.0) * icc


def analytic_half_width(n_fields: int, m: int, icc: float, rate: float) -> tuple[float, float, float]:
    """Wald half-width after cluster inflation. Returns (half_width, deff, n_eff).

    `rate` is the proportion being estimated -- sensitivity within its stratum, say. The
    worst case is 0.5; a rate near 1.0 gives a narrower interval, which is why planning at
    the expected sensitivity rather than at 0.5 is optimistic and is flagged in the report.
    """
    if n_fields < 1 or m < 1:
        raise ScoringError("n_fields and m must be >= 1")
    if not (0.0 <= rate <= 1.0):
        raise ScoringError(f"rate must be in [0, 1], got {rate!r}")
    total = n_fields * m
    deff = design_effect(m, icc)
    n_eff = total / deff
    hw = Z95 * math.sqrt(max(rate * (1.0 - rate), 1e-12) / n_eff)
    return hw, deff, n_eff


def plan_grid(field_counts, per_field_counts, *, icc_grid=ICC_GRID, rate: float = 0.90,
              target: float = TARGET_HALF_WIDTH) -> list[PlanRow]:
    """Analytic sweep. Cheap enough to scan the whole space before simulating."""
    rows = []
    for nf in field_counts:
        for m in per_field_counts:
            for icc in icc_grid:
                hw, deff, neff = analytic_half_width(nf, m, icc, rate)
                rows.append(PlanRow(nf, m, icc, rate, deff, neff, hw, hw <= target))
    return rows


def minimum_fields(per_field: int, icc: float, rate: float = 0.90,
                   target: float = TARGET_HALF_WIDTH, cap: int = 500) -> int | None:
    """Fewest fields meeting the half-width target. None if `cap` is not enough."""
    for nf in range(1, cap + 1):
        hw, _, _ = analytic_half_width(nf, per_field, icc, rate)
        if hw <= target:
            return nf
    return None


# ------------------------------------------------------------------------ simulation


def _beta_params(p: float, icc: float) -> tuple[float, float]:
    """Beta prior whose binomial mixture has the requested intracluster correlation.

    For a beta-binomial, rho = 1 / (1 + a + b), so a + b = (1 - rho) / rho and the mean
    p splits it. As ICC goes to 0 the concentration diverges and the field effect vanishes.
    """
    if icc <= 0:
        return float("inf"), float("inf")
    conc = (1.0 - icc) / icc
    return max(p * conc, 1e-9), max((1.0 - p) * conc, 1e-9)


def simulate_half_width(n_fields: int, m: int, icc: float, rate: float, *,
                        n_rep: int = 200, n_boot: int = 400, seed: int = 0,
                        target: float = TARGET_HALF_WIDTH) -> dict:
    """Simulate the real pipeline: clustered data scored by the field-cluster bootstrap.

    Each replicate draws a per-field true rate from the beta prior, draws nucleus outcomes,
    and runs the same bootstrap the analysis will run. What comes back is the distribution
    of realised half-widths, plus the share of replicates that would clear the target --
    which is the number worth planning against, because a design that clears it on average
    but fails 40% of the time is not a design.
    """
    if n_fields < 2:
        raise ScoringError(
            "the field-cluster bootstrap needs >= 2 fields; a single-field design cannot "
            "produce an inferential interval at all")
    rng = np.random.Generator(np.random.PCG64(seed))
    a, b = _beta_params(rate, icc)
    widths, met = [], 0
    for _ in range(n_rep):
        field_rate = (np.full(n_fields, rate) if math.isinf(a)
                      else rng.beta(a, b, size=n_fields))
        rows = []
        for f in range(n_fields):
            hits = rng.random(m) < field_rate[f]
            for i, h in enumerate(hits):
                # truth positive throughout: this sizes sensitivity within one stratum,
                # where the estimand is P(call | truth positive).
                rows.append({"well": "w", "field": f"f{f}", "nucleus_id": f * m + i,
                             "inclusion_probability": 1.0, "stratum": "0.8_1.0",
                             "call_2d": bool(h), "truth_3d": True,
                             "reference_id": f"r{f}_{i}"})
        ci = field_cluster_bootstrap(rows, None, n_boot=n_boot, seed=int(rng.integers(1 << 31)))
        lo, hi = ci["sensitivity"]
        if lo is None or hi is None:
            continue
        hw = (hi - lo) / 2.0
        widths.append(hw)
        met += hw <= target
    if not widths:
        raise ScoringError("no replicate produced a usable interval")
    w = np.array(widths)
    return {"n_fields": n_fields, "nuclei_per_field": m, "icc": icc, "true_rate": rate,
            "n_replicates": len(widths),
            "median_half_width": float(np.median(w)),
            "p90_half_width": float(np.quantile(w, 0.90)),
            "share_meeting_target": met / len(widths),
            "target_half_width": target}


def build_plan_report(field_counts, per_field_counts, *, rate: float = 0.90,
                      icc_grid=ICC_GRID, simulate_at: list[tuple[int, int, float]] | None = None,
                      n_rep: int = 200, n_boot: int = 400, seed: int = 0) -> dict:
    grid = [r.as_dict() for r in plan_grid(field_counts, per_field_counts,
                                           icc_grid=icc_grid, rate=rate)]
    minima = {f"m={m},icc={icc}": minimum_fields(m, icc, rate)
              for m in per_field_counts for icc in icc_grid}
    sims = [simulate_half_width(nf, m, icc, rate, n_rep=n_rep, n_boot=n_boot, seed=seed)
            for nf, m, icc in (simulate_at or [])]
    return {
        "schema_version": "tier_a_planning/1.0",
        "task": "TA03b", "stage": "sample_size_planning",
        "estimand": "sensitivity within one boundary stratum",
        "target_half_width": TARGET_HALF_WIDTH,
        "assumed_rate": rate,
        "icc_grid": list(icc_grid),
        "analytic_grid": grid,
        "minimum_fields_analytic": minima,
        "simulations": sims,
        "limitations": [
            "ICC is assumed, not measured; a pilot is the only way to fix it, and the "
            "ratification already requires one",
            "planning at the expected rate rather than at 0.5 is optimistic; a rate "
            "further from 1.0 needs more fields than this grid shows",
            "the analytic column is a Wald approximation and is unreliable near rate 1.0, "
            "where the simulated column should be preferred",
            "this sizes one stratum; the boundary strata are the binding ones and each "
            "must clear the target on its own",
        ],
    }


def human_readable(report: dict) -> str:
    L = ["Tier-A sample-size planning - cluster-aware", "=" * 62,
         f"estimand: {report['estimand']}",
         f"assumed rate: {report['assumed_rate']}   "
         f"target half-width: {report['target_half_width']}", "",
         "minimum fields required (analytic):", ""]
    L.append(f"  {'nuclei/field':>13s} " + "".join(f"{'ICC ' + str(i):>12s}"
                                                   for i in report["icc_grid"]))
    per = sorted({int(k.split(",")[0].split("=")[1])
                  for k in report["minimum_fields_analytic"]})
    for m in per:
        cells = []
        for icc in report["icc_grid"]:
            v = report["minimum_fields_analytic"].get(f"m={m},icc={icc}")
            cells.append(f"{'>cap' if v is None else v:>12}")
        L.append(f"  {m:>13d} " + "".join(cells))
    if report["simulations"]:
        L += ["", "simulated (same bootstrap the analysis uses):", ""]
        L.append(f"  {'fields':>7s}{'m':>6s}{'ICC':>7s}{'median hw':>11s}"
                 f"{'p90 hw':>9s}{'% meeting':>11s}")
        for s in report["simulations"]:
            L.append(f"  {s['n_fields']:>7d}{s['nuclei_per_field']:>6d}{s['icc']:>7.2f}"
                     f"{s['median_half_width']:>11.4f}{s['p90_half_width']:>9.4f}"
                     f"{s['share_meeting_target'] * 100:>10.0f}%")
    L += ["", "limitations:"] + [f"  - {x}" for x in report["limitations"]]
    return "\n".join(L) + "\n"


def write_plan(out_dir: Path, report: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_size_plan.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "sample_size_plan.txt").write_text(human_readable(report), encoding="utf-8")
    return {"json": str(out_dir / "sample_size_plan.json")}
