"""Dense myotube relabelling: trace centrelines, snap widths, rebuild the corpus.

The QC-review tool answers "is this proposal a myotube?". This one answers the
question that turned out to matter more: **"what else in this field is a myotube
that nothing ever proposed?"**

Why it exists
-------------
Measured over the six bootstrap wells, `complete` targets cover 1.409% of field
area while ignored (`ambiguous` / `border_truncated` / overlap) covers 1.745%.
More real myotube pixels contribute nothing to training than contribute as
targets, and unproposed fibres are taught as background outright -- the ground
truth is conditioned on what the classical detector happened to propose.

No amount of triage fixes that, because triage can only rank an existing proposal
list. Dense relabelling can.

Design
------
A myotube is a ribbon: median width 5.4 um against median length 140 um. So the
annotation primitive is a **centreline polyline plus a width**, not a brush --
one click per bend instead of painting several hundred pixels, and the resulting
mask is smooth rather than hand-jittered. `snap` then grows the ribbon and
intersects it with the local signal, so the mask hugs the real fibre edge instead
of being a constant-width band.

Everything is append-only. Traces land in JSONL the moment they are committed,
never at the end of a session, and `apply` writes a NEW corpus version --
`bootstrap_v1` is sealed and is never modified in place.
"""
from __future__ import annotations

__all__ = ["store", "raster", "page", "server", "apply_traces"]
