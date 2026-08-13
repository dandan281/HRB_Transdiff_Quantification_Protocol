"""CL02 -- prove overlap-safe annotation round-trips.

A single mutually exclusive label image cannot represent two independently
visible crossing myotubes that share projected pixels: at the crossing, one arm
must overwrite the other. The ``InstanceSet`` JSON keeps a full per-instance
mask, so both arms survive. This module builds that exact known case and
verifies it survives export/re-import, that editing one instance does not touch
its neighbour, and it quantifies the information a flat TIFF would lose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import masks as M
from .model import AnnotationSession
from ._schema_bridge import InstanceSet


def make_synthetic_crossing(shape: tuple[int, int] = (64, 64),
                            thickness: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Two elongated bars forming an 'X' that share pixels at the crossing."""
    h, w = shape
    half = thickness // 2
    a = np.zeros(shape, dtype=bool)
    b = np.zeros(shape, dtype=bool)
    # Bar A: a broad diagonal (top-left -> bottom-right).
    for r in range(h):
        c = int(round(r * (w - 1) / (h - 1)))
        a[r, max(0, c - half):min(w, c + half + 1)] = True
    # Bar B: the opposite diagonal (top-right -> bottom-left).
    for r in range(h):
        c = int(round((h - 1 - r) * (w - 1) / (h - 1)))
        b[r, max(0, c - half):min(w, c + half + 1)] = True
    return a, b


@dataclass
class RoundTripReport:
    passed: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def require(self, name: str, condition: bool, **detail) -> None:
        self.checks[name] = bool(condition)
        if detail:
            self.details[name] = detail

    def finalize(self) -> "RoundTripReport":
        self.passed = all(self.checks.values())
        return self


def _iou_full(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    uni = int(np.logical_or(a, b).sum())
    return inter / uni if uni else 1.0


def run_overlap_roundtrip(out_dir: str | Path,
                          shape: tuple[int, int] = (64, 64)) -> RoundTripReport:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = RoundTripReport()

    # 1. Synthetic crossing with a known shared region.
    a_mask, b_mask = make_synthetic_crossing(shape)
    shared = int(np.logical_and(a_mask, b_mask).sum())
    report.require("shared_pixels_nonzero", shared > 0, shared=shared)

    # 2. Import both as independent, overlap-safe instances.
    session = AnnotationSession(shape, "synthetic_crossing")
    id_a = session.create(a_mask, status="complete", reviewer="pilot", source="manual")
    id_b = session.create(b_mask, status="complete", reviewer="pilot", source="manual")
    session.set_reviewed(id_a, True, reviewer="pilot")
    session.set_reviewed(id_b, True, reviewer="pilot")

    # Selecting either object shows its full mask through the crossing.
    full_a0 = session.instances[id_a].mask.full()
    full_b0 = session.instances[id_b].mask.full()
    report.require("both_full_through_crossing",
                   int(np.logical_and(full_a0, full_b0).sum()) == shared)
    hash_a0 = session.instances[id_a].mask.content_hash()

    # 3. Edit ONLY instance B (erase a small patch away from the crossing).
    erase = np.zeros(shape, dtype=bool)
    erase[0, :] = b_mask[0, :]           # trim B's first row only
    n_erased = int(erase.sum())
    session.erase(id_b, erase, reviewer="pilot")
    report.require("edit_is_nonempty", n_erased > 0, n_erased=n_erased)
    report.require("untouched_hash_stable",
                   session.instances[id_a].mask.content_hash() == hash_a0)

    # 4. Export authoritative JSON + re-import.
    save_info = session.save(out_dir / "crossing.instances.json")
    reloaded = InstanceSet.load(out_dir / "crossing.instances.json")
    session2 = AnnotationSession.from_instance_set(reloaded, as_prompts=False)
    report.require("ids_preserved", set(session2.instances) == {id_a, id_b})
    report.require("statuses_preserved",
                   all(session2.instances[i].status == "complete" for i in (id_a, id_b)))

    # Per-instance IoU: untouched A == 1.0; edited B reflects exactly the edit.
    iou_a = _iou_full(session2.instances[id_a].mask.full(), full_a0)
    edited_b_expected = b_mask & ~erase
    iou_b = _iou_full(session2.instances[id_b].mask.full(), edited_b_expected)
    report.require("untouched_iou_is_1", iou_a == 1.0, iou_a=iou_a)
    report.require("edited_iou_matches_intended_edit", iou_b == 1.0, iou_b=iou_b)
    # And the neighbour really is different from the edit target.
    report.require("edit_actually_changed_b",
                   _iou_full(edited_b_expected, b_mask) < 1.0)

    # Overlap survived the JSON round-trip.
    shared2 = int(np.logical_and(session2.instances[id_a].mask.full(),
                                 session2.instances[id_b].mask.full()).sum())
    report.require("overlap_survived_json", shared2 > 0, shared2=shared2)

    # 5. Document the TIFF limitation: a flat label image collides at the cross.
    flat = np.zeros(shape, dtype=np.int32)
    flat[a_mask] = 1
    flat[edited_b_expected] = 2           # B overwrites A at the crossing
    a_from_flat = flat == 1
    lost = int((a_mask & ~a_from_flat).sum())      # A pixels destroyed by the flatten
    report.require("flat_tiff_loses_overlap", lost > 0, pixels_lost=lost)
    report.details["tiff_limitation"] = {
        "message": ("A single label TIFF is mutually exclusive: at the crossing, "
                    "one arm overwrites the other, destroying %d shared pixels. "
                    "Authoritative export is JSON/RLE; a flat TIFF is a "
                    "convenience view only." % lost),
        "authoritative_export": save_info["instances"],
    }

    return report.finalize()
