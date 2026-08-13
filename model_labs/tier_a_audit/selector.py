"""TA03b part 1 - deterministic stratified nucleus selector for Tier-A validation.

Draws the Module-A sample from the accepted audit's ring intensities and exports
nucleus IDs plus coordinates for targeted z-imaging. Authorized by
`coordination/requests/claude/2026-07-23-tier-a-selector-and-scorer.md` and governed
by `coordination/reports/codex_tier_a_validation_ratification_2026-07-23.md`.

What this is NOT allowed to do, and does not do: acquire data, change the frozen
production method, tune the production threshold, or write anywhere under
`Conversion_Efficiency/`. The last is enforced rather than promised -- see
:func:`read_only_guard`, which stats the whole tree before and after and raises if
anything moved.

Two design decisions worth stating, because both were choices:

**Ring intensity is imported from `audit.py`, never reimplemented.** That function is
transcribed verbatim from the production package and is the reason the audit reproduces
exactly. A second copy here would be a second method, and the first time they drifted the
selector would be sampling from a population the audit never measured.

**`relocalization_feasible` defaults to False.** The ratification is explicit that the
selector must expose whether original nuclei can actually be relocated, and that pixel
coordinates alone do not permit physical reacquisition. Nothing in the current pipeline
carries stage registration, so the honest export says so with a reason rather than
emitting centroids that look actionable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from . import audit

ROOT = audit.ROOT
CE = audit.CE

# Locked by the ratified contract. Half-open [lo, hi); the first is (-inf, 0.5) and
# the last [2.0, inf). These may not be re-cut without re-ratification.
STRATA: tuple[tuple[str, float, float], ...] = (
    ("lt_0.5",    float("-inf"), 0.5),
    ("0.5_0.8",   0.5,           0.8),
    ("0.8_1.0",   0.8,           1.0),
    ("1.0_1.25",  1.0,           1.25),
    ("1.25_2.0",  1.25,          2.0),
    ("ge_2.0",    2.0,           float("inf")),
)
BOUNDARY_STRATA = ("0.8_1.0", "1.0_1.25")   # the call boundary; oversampling target

RNG_ALGORITHM = "numpy.random.Generator(PCG64)"
SCHEMA_VERSION = "tier_a_selection/1.0"


class SelectionError(RuntimeError):
    """Fail-closed condition. Never downgraded to a warning."""


def stratum_of(ratio: float) -> str:
    """Map a ring/threshold ratio onto a locked stratum name.

    Boundaries are half-open upward, so a ratio of exactly 1.0 is a positive call in
    `1.0_1.25` rather than a negative one in `0.8_1.0`. NaN is an error, not a bucket:
    a nucleus with no ring pixels must be excluded upstream and counted as attrition.
    """
    if ratio != ratio:                                    # NaN
        raise SelectionError("ratio is NaN; exclude upstream and record attrition")
    for name, lo, hi in STRATA:
        if lo <= ratio < hi:
            return name
    raise SelectionError(f"ratio {ratio!r} fell through every stratum")


@dataclass(frozen=True)
class NucleusRecord:
    """One row of the sampling frame. Every field the contract requires to be exported."""
    well: str
    field: str
    nucleus_id: int                 # canonical label id in the well's mask array
    centroid_row: float
    centroid_col: float
    ring_intensity: float
    ring_ratio: float               # ring_intensity / production threshold
    call_2d: bool                   # the frozen method's Desmin-positive call
    stratum: str
    valid_by_area: bool

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.well, self.field, self.nucleus_id)


@dataclass(frozen=True)
class SelectedNucleus:
    record: NucleusRecord
    inclusion_probability: float
    selection_reason: str

    def row(self) -> dict:
        d = asdict(self.record)
        d["inclusion_probability"] = self.inclusion_probability
        d["selection_reason"] = self.selection_reason
        return d


# ----------------------------------------------------------------------------- guard


def _tree_state(root: Path) -> dict[str, tuple[int, int]]:
    if not root.is_dir():
        return {}
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in root.rglob("*") if p.is_file()}


class read_only_guard:
    """Assert nothing under `Conversion_Efficiency/` changed while we ran.

    The request forbids writing there. This makes that checkable rather than trusted:
    size and mtime for every file, before and after. Cheap (stat only, no hashing) and
    it catches truncation, rewrite and deletion alike.
    """

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else CE
        self._before: dict[str, tuple[int, int]] = {}

    def __enter__(self):
        self._before = _tree_state(self.root)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            return False
        after = _tree_state(self.root)
        added = sorted(set(after) - set(self._before))
        removed = sorted(set(self._before) - set(after))
        changed = sorted(k for k in set(after) & set(self._before)
                         if after[k] != self._before[k])
        if added or removed or changed:
            raise SelectionError(
                "read-only guard tripped: Conversion_Efficiency/ was modified. "
                f"added={added[:5]} removed={removed[:5]} changed={changed[:5]}")
        return False


# ------------------------------------------------------------------- sampling frame


def _centroids(nuc: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Label centroids by bincount; index 0 (background) is meaningless and unused."""
    flat = nuc.ravel()
    rows = np.repeat(np.arange(nuc.shape[0], dtype=np.int64), nuc.shape[1])
    cols = np.tile(np.arange(nuc.shape[1], dtype=np.int64), nuc.shape[0])
    cnt = np.bincount(flat, minlength=n + 1).astype(np.float64)
    rsum = np.bincount(flat, weights=rows, minlength=n + 1)
    csum = np.bincount(flat, weights=cols, minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(cnt > 0, rsum / np.maximum(cnt, 1), np.nan),
                np.where(cnt > 0, csum / np.maximum(cnt, 1), np.nan))


def build_frame(well: str, threshold: float, *, field: str = "field_00",
                nuc: np.ndarray | None = None, dbs: np.ndarray | None = None
                ) -> tuple[list[NucleusRecord], dict]:
    """Enumerate every eligible nucleus in one well as a sampling-frame row.

    `nuc`/`dbs` are injectable so the logic is testable on synthetic arrays without the
    plate. When omitted they are read from the canonical audit inputs, read-only.

    Nuclei failing the frozen area gate, or with no ring pixels, are excluded here and
    reported in the returned attrition dict rather than silently dropped -- the contract
    requires the sampling-frame count to be recorded, and a frame that quietly shrinks
    makes every inclusion probability wrong.
    """
    if nuc is None:
        nuc = np.load(audit.NUC_DIR / f"{well}_masks.npy")
    if dbs is None:
        dbs = np.load(audit.DBS_DIR / f"{well}_dbs.npy").astype(np.float32)
    if nuc.shape != dbs.shape:
        raise SelectionError(f"{well}: mask {nuc.shape} vs desmin {dbs.shape}")
    if threshold <= 0:
        raise SelectionError(f"threshold must be positive, got {threshold!r}")

    rp = audit.ring_px()
    mean, cnt = audit.ring_intensity(nuc, dbs, rp)
    valid = audit.valid_by_area(nuc)[:mean.size]
    n = int(nuc.max())
    crow, ccol = _centroids(nuc, n)

    records, excluded_area, excluded_ring = [], 0, 0
    for lab in range(1, mean.size):
        if not valid[lab]:
            excluded_area += 1
            continue
        if cnt[lab] <= 0:
            excluded_ring += 1
            continue
        ratio = float(mean[lab]) / threshold
        records.append(NucleusRecord(
            well=well, field=field, nucleus_id=lab,
            centroid_row=float(crow[lab]), centroid_col=float(ccol[lab]),
            ring_intensity=float(mean[lab]), ring_ratio=ratio,
            call_2d=bool(mean[lab] > threshold), stratum=stratum_of(ratio),
            valid_by_area=True))
    attrition = {"labels_in_mask": n, "excluded_failed_area_gate": excluded_area,
                 "excluded_no_ring_pixels": excluded_ring,
                 "sampling_frame_size": len(records)}
    return records, attrition


# ------------------------------------------------------------------------ selection


def _validate(records: list[NucleusRecord], image_shape: dict[tuple[str, str], tuple[int, int]] | None) -> None:
    seen: set[tuple[str, str, int]] = set()
    for r in records:
        if r.key in seen:
            raise SelectionError(f"duplicate nucleus {r.key}")
        seen.add(r.key)
        if r.nucleus_id <= 0:
            raise SelectionError(f"non-positive nucleus id {r.key}")
        if r.centroid_row != r.centroid_row or r.centroid_col != r.centroid_col:
            raise SelectionError(f"NaN centroid for {r.key}")
        shape = (image_shape or {}).get((r.well, r.field))
        if shape is not None:
            h, w = shape
            if not (0 <= r.centroid_row < h and 0 <= r.centroid_col < w):
                raise SelectionError(
                    f"out-of-frame centroid for {r.key}: "
                    f"({r.centroid_row:.1f}, {r.centroid_col:.1f}) vs {shape}")


def select(records: list[NucleusRecord], per_stratum: dict[str, int] | int, *,
           seed: int, image_shape: dict[tuple[str, str], tuple[int, int]] | None = None,
           ) -> list[SelectedNucleus]:
    """Stratified sample without replacement, fields first and nuclei second.

    `per_stratum` is a target count per (field, stratum); pass a dict to oversample the
    boundary strata. Where a stratum holds fewer nuclei than requested, all of them are
    taken and the inclusion probability is 1.0 -- taking fewer would silently bias the
    boundary, and taking more is impossible.

    Determinism: rows are sorted by (well, field, nucleus_id) before anything is drawn,
    and each (field, stratum) cell draws from a generator seeded on that cell rather than
    from one shared stream, so adding a well or changing one stratum's count cannot
    reshuffle the others.
    """
    _validate(records, image_shape)
    targets = ({s: per_stratum for s, _, _ in STRATA} if isinstance(per_stratum, int)
               else dict(per_stratum))
    unknown = set(targets) - {s for s, _, _ in STRATA}
    if unknown:
        raise SelectionError(f"unknown stratum name(s): {sorted(unknown)}")

    cells: dict[tuple[str, str, str], list[NucleusRecord]] = {}
    for r in sorted(records, key=lambda x: x.key):
        cells.setdefault((r.well, r.field, r.stratum), []).append(r)

    out: list[SelectedNucleus] = []
    for (well, field, stratum) in sorted(cells):
        frame = cells[(well, field, stratum)]
        want = int(targets.get(stratum, 0))
        if want <= 0:
            continue
        take = min(want, len(frame))
        # Per-cell stream: the cell key is hashed into the seed so cells are independent.
        cell_seed = int.from_bytes(
            hashlib.sha256(f"{seed}|{well}|{field}|{stratum}".encode()).digest()[:8],
            "big")
        rng = np.random.Generator(np.random.PCG64(cell_seed))
        idx = sorted(rng.choice(len(frame), size=take, replace=False).tolist())
        prob = take / len(frame)
        reason = ("boundary_oversample" if stratum in BOUNDARY_STRATA
                  else "stratified_random")
        if take < want:
            reason += "_frame_exhausted"
        for i in idx:
            out.append(SelectedNucleus(frame[i], prob, reason))
    return out


# --------------------------------------------------------------------------- export


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_manifest(selected: list[SelectedNucleus], frames: dict[str, dict], *,
                   seed: int, threshold: float, source_hashes: dict[str, str],
                   relocalization_feasible: bool = False,
                   relocalization_reason: str = (
                       "no stage registration is carried through the current pipeline; "
                       "pixel centroids alone do not permit physical reacquisition"),
                   ) -> dict:
    """Immutable selection manifest. Every attempted selection is retained."""
    per_cell: dict[str, dict] = {}
    for s in selected:
        k = f"{s.record.well}|{s.record.field}|{s.record.stratum}"
        c = per_cell.setdefault(k, {"selected": 0, "inclusion_probability": s.inclusion_probability})
        c["selected"] += 1
        if abs(c["inclusion_probability"] - s.inclusion_probability) > 1e-12:
            raise SelectionError(f"inconsistent inclusion probability within cell {k}")
    for well, att in frames.items():
        for name, _, _ in STRATA:
            k = f"{well}|{att.get('field', 'field_00')}|{name}"
            per_cell.setdefault(k, {"selected": 0, "inclusion_probability": 0.0})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task": "TA03b", "stage": "selection",
        "rng": {"algorithm": RNG_ALGORITHM, "seed": seed,
                "per_cell_seed": "sha256(seed|well|field|stratum)[:8]"},
        "production_threshold": threshold,
        "threshold_method": audit.THRESHOLD_METHOD,
        "strata": [{"name": n, "lo": lo, "hi": hi} for n, lo, hi in STRATA],
        "boundary_strata": list(BOUNDARY_STRATA),
        "source_hashes": source_hashes,
        "sampling_frame": frames,
        "per_cell": per_cell,
        "n_selected": len(selected),
        "relocalization_feasible": relocalization_feasible,
        "relocalization_reason": None if relocalization_feasible else relocalization_reason,
        "acquisition_status": "not_attempted",
        "limitations": [
            "single operator, single plate, proposal-conditioned",
            "one field per well in the current data; the field axis exists so that "
            "multi-field acquisition does not change the schema",
            "selection is not authorization to acquire; TA03c gates that",
        ],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest


def write_selection(out_dir: Path, selected: list[SelectedNucleus], manifest: dict) -> dict:
    """Write manifest JSON plus a flat CSV. Refuses to overwrite an existing selection."""
    out_dir = Path(out_dir)
    mp, cp = out_dir / "selection_manifest.json", out_dir / "selection.csv"
    for p in (mp, cp):
        if p.exists():
            raise SelectionError(
                f"{p} exists; a selection manifest is append-only evidence and is not "
                "silently replaced. Write to a new directory or remove it deliberately.")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = [s.row() for s in selected]
    with cp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           [f.name for f in NucleusRecord.__dataclass_fields__.values()]
                           + ["inclusion_probability", "selection_reason"])
        w.writeheader()
        w.writerows(rows)
    return {"manifest": str(mp), "csv": str(cp),
            "manifest_sha256": manifest["manifest_sha256"]}
