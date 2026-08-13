"""Headless annotation session -- the testable core of the assisted UI.

The napari/micro-sam GUI (``napari_app.py``) is a thin view over this model, so
every scientific guardrail is unit-testable without a display:

* Instances are overlap-safe (each keeps a full mask; crossings share pixels).
* Automated proposals load as *prompt* instances, visually and structurally
  distinct from accepted annotations, and are never exported as authoritative.
* There is deliberately **no** "mark everything complete" operation; status and
  review state are set one instance at a time.
* Export refuses any accepted instance that is missing a status, and any
  reviewed instance that is missing a reviewer -- a task never closes on a mask
  alone.

Authoritative output is the canonical ``InstanceSet`` JSON (validated by Codex's
frozen schema) plus a companion ``review_log.jsonl`` capturing reviewer/action
provenance that the compact schema record does not carry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from . import masks as M
from ._schema_bridge import (
    VALID_STATUSES, AUTHORITATIVE_STATUSES, InstanceRecord, InstanceSet,
    decode_rle_cropped,
)

# ``source`` values are free text in the schema; the lane standardises these.
SOURCE_MANUAL = "manual"
SOURCE_PROMPT = "proposal_prompt"          # automated proposal, not accepted
SOURCE_ACCEPTED_PROMPT = "reviewed_from_proposal"


class AnnotationError(RuntimeError):
    """Raised when an operation would violate a scientific guardrail."""


@dataclass
class Instance:
    id: str
    mask: M.SparseMask
    status: str = "ambiguous"
    reviewed: bool = False
    source: str = SOURCE_MANUAL
    confidence: float | None = None
    notes: str = ""
    reviewer: str = ""
    is_prompt: bool = False

    def to_record(self) -> InstanceRecord:
        return InstanceRecord(
            id=self.id, status=self.status, rle=self.mask.to_rle(),
            source=self.source, confidence=self.confidence,
            reviewed=self.reviewed, notes=self.notes,
        )


@dataclass
class AnnotationSession:
    image_shape: tuple[int, int]
    image_id: str
    pixel_um: float | None = None
    reviewer: str = ""                       # default reviewer for new edits
    instances: dict[str, Instance] = field(default_factory=dict)
    review_log: list[dict] = field(default_factory=list)
    _counter: int = 0

    # ------------------------------------------------------------------ helpers
    def _next_id(self) -> str:
        self._counter += 1
        candidate = f"myotube_{self._counter:04d}"
        while candidate in self.instances:
            self._counter += 1
            candidate = f"myotube_{self._counter:04d}"
        return candidate

    def _require(self, instance_id: str) -> Instance:
        if instance_id not in self.instances:
            raise AnnotationError(f"unknown instance {instance_id!r}")
        return self.instances[instance_id]

    def _as_mask(self, mask) -> M.SparseMask:
        if isinstance(mask, M.SparseMask):
            if mask.image_shape != self.image_shape:
                raise AnnotationError("mask image_shape does not match the field")
            return mask
        return M.SparseMask.from_full(np.asarray(mask), self.image_shape)

    def _log(self, action: str, instance_id: str, **extra) -> None:
        # Ordered by a monotonic sequence, not wall-clock, so exports stay
        # reproducible/hashable (needed for the CL02 round-trip checks).
        entry = {"seq": len(self.review_log), "action": action, "id": instance_id}
        entry.update(extra)
        self.review_log.append(entry)

    # ------------------------------------------------------------- prompt layer
    def load_prompt(self, mask, *, source: str = SOURCE_PROMPT,
                    confidence: float | None = None, instance_id: str | None = None) -> str:
        """Add an automated proposal as a non-authoritative prompt instance."""
        iid = instance_id or self._next_id()
        if iid in self.instances:
            raise AnnotationError(f"instance {iid!r} already exists")
        self.instances[iid] = Instance(
            id=iid, mask=self._as_mask(mask), status="ambiguous", reviewed=False,
            source=source, confidence=confidence, is_prompt=True,
        )
        self._log("load_prompt", iid, source=source)
        return iid

    def accept_prompt(self, instance_id: str, *, status: str | None = None,
                      reviewer: str | None = None) -> None:
        """Promote a prompt to an editable annotation. It is still not authoritative
        until a reviewer assigns a status (and reviews it, for 'complete')."""
        inst = self._require(instance_id)
        if not inst.is_prompt:
            raise AnnotationError(f"{instance_id!r} is not a prompt")
        inst.is_prompt = False
        inst.source = SOURCE_ACCEPTED_PROMPT
        if status is not None:
            self.set_status(instance_id, status)
        if reviewer is not None:
            inst.reviewer = reviewer
        self._log("accept_prompt", instance_id)

    # ------------------------------------------------------------ create / edit
    def create(self, mask, *, status: str = "ambiguous", source: str = SOURCE_MANUAL,
               reviewer: str | None = None, confidence: float | None = None,
               notes: str = "", instance_id: str | None = None) -> str:
        if status not in VALID_STATUSES:
            raise AnnotationError(f"invalid status {status!r}")
        iid = instance_id or self._next_id()
        if iid in self.instances:
            raise AnnotationError(f"instance {iid!r} already exists")
        self.instances[iid] = Instance(
            id=iid, mask=self._as_mask(mask), status=status, source=source,
            confidence=confidence, notes=notes,
            reviewer=self.reviewer if reviewer is None else reviewer,
        )
        self._log("create", iid, status=status)
        return iid

    def refine(self, instance_id: str, new_mask, *, reviewer: str | None = None) -> None:
        inst = self._require(instance_id)
        inst.mask = self._as_mask(new_mask)
        if reviewer is not None:
            inst.reviewer = reviewer
        self._log("refine", instance_id)

    def erase(self, instance_id: str, erase_mask, *, reviewer: str | None = None) -> None:
        inst = self._require(instance_id)
        inst.mask = M.subtract(inst.mask, self._as_mask(erase_mask))
        if reviewer is not None:
            inst.reviewer = reviewer
        self._log("erase", instance_id)

    def merge(self, id_a: str, id_b: str, *, reviewer: str | None = None,
              status: str | None = None, notes: str = "") -> str:
        a, b = self._require(id_a), self._require(id_b)
        merged_mask = M.union(a.mask, b.mask)
        del self.instances[id_a]
        del self.instances[id_b]
        new_id = self._next_id()
        self.instances[new_id] = Instance(
            id=new_id, mask=merged_mask, status=status or "ambiguous",
            source=SOURCE_MANUAL, notes=notes,
            reviewer=self.reviewer if reviewer is None else reviewer,
        )
        self._log("merge", new_id, merged_from=[id_a, id_b])
        return new_id

    def split(self, instance_id: str, mask_a, mask_b, *,
              reviewer: str | None = None) -> tuple[str, str]:
        self._require(instance_id)
        del self.instances[instance_id]
        rev = self.reviewer if reviewer is None else reviewer
        id_a = self._next_id()
        self.instances[id_a] = Instance(id=id_a, mask=self._as_mask(mask_a),
                                        source=SOURCE_MANUAL, reviewer=rev)
        id_b = self._next_id()
        self.instances[id_b] = Instance(id=id_b, mask=self._as_mask(mask_b),
                                        source=SOURCE_MANUAL, reviewer=rev)
        self._log("split", instance_id, split_into=[id_a, id_b])
        return id_a, id_b

    def remove(self, instance_id: str) -> None:
        self._require(instance_id)
        del self.instances[instance_id]
        self._log("remove", instance_id)

    # ---------------------------------------------------------- status / review
    def set_status(self, instance_id: str, status: str) -> None:
        if status not in VALID_STATUSES:
            raise AnnotationError(f"invalid status {status!r}")
        inst = self._require(instance_id)
        if inst.is_prompt:
            raise AnnotationError("accept the prompt before assigning a status")
        inst.status = status
        self._log("set_status", instance_id, status=status)

    def set_reviewed(self, instance_id: str, reviewed: bool, *, reviewer: str | None = None) -> None:
        inst = self._require(instance_id)
        if inst.is_prompt:
            raise AnnotationError("accept the prompt before reviewing it")
        rev = reviewer if reviewer is not None else (inst.reviewer or self.reviewer)
        if reviewed and not rev:
            raise AnnotationError(
                f"{instance_id!r}: a reviewer is required to mark an instance reviewed")
        inst.reviewed = bool(reviewed)
        if rev:
            inst.reviewer = rev
        self._log("set_reviewed", instance_id, reviewed=bool(reviewed), reviewer=rev)

    def set_notes(self, instance_id: str, notes: str) -> None:
        self._require(instance_id).notes = notes
        self._log("set_notes", instance_id)

    def set_reviewer(self, instance_id: str, reviewer: str) -> None:
        self._require(instance_id).reviewer = reviewer
        self._log("set_reviewer", instance_id, reviewer=reviewer)

    # NOTE: there is intentionally no bulk "mark all complete/reviewed" method.
    # Authority is granted one instance at a time so a convenient UI action can
    # never turn proposals into biological truth.

    # ------------------------------------------------------------------- audit
    def authoritative_ids(self) -> list[str]:
        return [i.id for i in self.instances.values()
                if i.reviewed and i.status in AUTHORITATIVE_STATUSES and not i.is_prompt]

    def accepted_instances(self) -> list[Instance]:
        """Instances eligible for export (everything that is not a raw prompt)."""
        return [i for i in self.instances.values() if not i.is_prompt]

    def unresolved_prompt_ids(self) -> list[str]:
        return [i.id for i in self.instances.values() if i.is_prompt]

    def _export_problems(self) -> list[str]:
        problems: list[str] = []
        for inst in self.accepted_instances():
            if inst.status not in VALID_STATUSES:
                problems.append(f"{inst.id}: missing/invalid status {inst.status!r}")
            if inst.reviewed and not inst.reviewer:
                problems.append(f"{inst.id}: reviewed but no reviewer recorded")
            if inst.status in AUTHORITATIVE_STATUSES and inst.reviewed and not inst.reviewer:
                problems.append(f"{inst.id}: authoritative-complete without a reviewer")
        return problems

    # ------------------------------------------------------------------ export
    def to_instance_set(self, *, include_prompts: bool = False) -> InstanceSet:
        """Build a canonical, validated ``InstanceSet``.

        By default raw prompts are excluded so an automated proposal cannot be
        published as an accepted mask. Set ``include_prompts=True`` only to
        round-trip a proposal package unchanged (they stay status=ambiguous,
        reviewed=False).
        """
        problems = self._export_problems()
        if problems:
            raise AnnotationError("cannot export; unresolved review state:\n  " +
                                  "\n  ".join(problems))
        chosen = list(self.instances.values()) if include_prompts else self.accepted_instances()
        records = [i.to_record() for i in chosen]
        result = InstanceSet(tuple(self.image_shape), self.image_id, records)
        result.validate()
        return result

    def save(self, instances_path: str | Path, *, review_log_path: str | Path | None = None,
             include_prompts: bool = False) -> dict:
        """Write the authoritative InstanceSet JSON and a companion review log."""
        instances_path = Path(instances_path)
        instance_set = self.to_instance_set(include_prompts=include_prompts)
        instance_set.save(instances_path)

        if review_log_path is None:
            review_log_path = instances_path.with_suffix(".review_log.jsonl")
        review_log_path = Path(review_log_path)
        lines = []
        # Per-instance review state (reviewer lives here, not in the compact record).
        for inst in (self.instances.values() if include_prompts else self.accepted_instances()):
            lines.append(json.dumps({
                "type": "instance", "id": inst.id, "status": inst.status,
                "reviewed": inst.reviewed, "reviewer": inst.reviewer,
                "source": inst.source, "is_prompt": inst.is_prompt,
                "notes": inst.notes,
            }, sort_keys=True))
        for entry in self.review_log:
            lines.append(json.dumps({"type": "action", **entry}, sort_keys=True))
        review_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"instances": str(instances_path), "review_log": str(review_log_path),
                "n_exported": len(instance_set.instances),
                "n_authoritative": len(self.authoritative_ids())}

    # ------------------------------------------------------------------ import
    @classmethod
    def from_instance_set(cls, instance_set: InstanceSet, *, pixel_um: float | None = None,
                          as_prompts: bool = False) -> "AnnotationSession":
        session = cls(tuple(instance_set.image_shape), instance_set.image_id, pixel_um=pixel_um)
        for rec in instance_set.instances:
            bbox, crop = decode_rle_cropped(rec.rle)
            mask = M.SparseMask((bbox[0], bbox[1]), crop, tuple(instance_set.image_shape))
            session.instances[rec.id] = Instance(
                id=rec.id, mask=mask, status=rec.status, reviewed=rec.reviewed,
                source=rec.source, confidence=rec.confidence, notes=rec.notes,
                is_prompt=as_prompts,
            )
        session._counter = _max_counter(session.instances)
        return session


def _max_counter(instances: dict[str, Instance]) -> int:
    best = 0
    for iid in instances:
        tail = iid.rsplit("_", 1)[-1]
        if tail.isdigit():
            best = max(best, int(tail))
    return best
