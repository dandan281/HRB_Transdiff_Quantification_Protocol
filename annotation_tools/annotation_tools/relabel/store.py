"""Append-only persistence for relabelling traces.

Every committed trace is written to disk immediately. A session that crashes, or
a browser tab closed by accident, loses nothing -- which is the failure the
download-a-JSON-at-the-end pattern has and the reason this one does not use it.

Records are never mutated. An edit or a delete appends a NEW record referencing
the earlier `trace_id`; the current state is the result of replaying the log in
order. That makes the annotation history auditable after the fact -- what was
drawn, what was reconsidered, and when -- which a mutable store throws away.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

SCHEMA_VERSION = 1
KINDS = ("add", "edit", "delete", "reject_existing", "note")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TraceStore:
    """One JSONL log per well."""

    def __init__(self, root: str | Path, well: str):
        self.root = Path(root)
        self.well = well
        self.dir = self.root / well
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "traces.jsonl"

    # ------------------------------------------------------------------ write
    def append(self, record: dict) -> dict:
        kind = record.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}; have {KINDS}")
        record = {
            "schema": SCHEMA_VERSION,
            "trace_id": record.get("trace_id") or uuid.uuid4().hex[:12],
            "well": self.well,
            "created_utc": _utc(),
            **record,
        }
        # Flush + fsync per record: a browser crash or a power cut must not cost
        # the operator work that the UI already told them was saved.
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    # ------------------------------------------------------------------- read
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
        return out

    def current(self) -> dict:
        """Replay the log into the live state.

        Returns ``{"traces": {id: record}, "rejected_existing": {label_id: rec}}``.
        """
        traces: dict[str, dict] = {}
        rejected: dict[int, dict] = {}
        for r in self.records():
            kind = r["kind"]
            if kind == "add":
                traces[r["trace_id"]] = r
            elif kind == "edit":
                target = r.get("replaces") or r["trace_id"]
                traces[target] = {**r, "trace_id": target}
            elif kind == "delete":
                traces.pop(r.get("replaces") or r["trace_id"], None)
            elif kind == "reject_existing":
                lid = int(r["source_label"])
                if r.get("undo"):
                    rejected.pop(lid, None)
                else:
                    rejected[lid] = r
        return {"traces": traces, "rejected_existing": rejected}

    def stats(self) -> dict:
        state = self.current()
        traces = list(state["traces"].values())
        return {
            "well": self.well,
            "n_traces": len(traces),
            "n_rejected_existing": len(state["rejected_existing"]),
            "n_records": len(self.records()),
            "reviewers": sorted({t.get("reviewer") for t in traces
                                 if t.get("reviewer")}),
            "log": str(self.path),
        }


def all_stats(root: str | Path, wells: list[str]) -> dict:
    root = Path(root)
    return {w: TraceStore(root, w).stats() for w in wells}
