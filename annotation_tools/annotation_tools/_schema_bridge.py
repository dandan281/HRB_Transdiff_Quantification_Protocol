"""Read-only bridge to the canonical, frozen ``precision_myotube.schema``.

The Claude annotation lane must never define its own copy of the InstanceSet
contract (that would defeat the frozen-schema guarantee in Phase 0 / P0.2). It
consumes Codex's schema as the single source of truth. This module locates that
package whether or not it is pip-installed:

1. If ``import precision_myotube.schema`` already works, use it.
2. Otherwise add the sibling ``PrecisionMyotube`` checkout to ``sys.path`` and
   retry, so the tool runs directly from a repository clone.

We only ever *import* the canonical package; we never write into it.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load_canonical_schema():
    try:
        import precision_myotube.schema as schema  # type: ignore
        return schema
    except ModuleNotFoundError:
        pass

    # Walk up from this file looking for a `PrecisionMyotube/precision_myotube`
    # canonical checkout, then make it importable read-only.
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "PrecisionMyotube"
        if (candidate / "precision_myotube" / "schema.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            import precision_myotube.schema as schema  # type: ignore
            return schema

    raise ModuleNotFoundError(
        "Could not locate the canonical `precision_myotube.schema`. Install the "
        "package with `pip install -e PrecisionMyotube`, or run from a checkout "
        "that contains the PrecisionMyotube/ directory."
    )


schema = _load_canonical_schema()

# Re-export the exact canonical objects the annotation lane is allowed to use.
InstanceRecord = schema.InstanceRecord
InstanceSet = schema.InstanceSet
encode_rle = schema.encode_rle
decode_rle = schema.decode_rle
encode_sparse_positions = schema.encode_sparse_positions
rle_foreground_positions = schema.rle_foreground_positions
decode_rle_cropped = schema.decode_rle_cropped
from_label_image = schema.from_label_image
VALID_STATUSES = schema.VALID_STATUSES
AUTHORITATIVE_STATUSES = schema.AUTHORITATIVE_STATUSES

__all__ = [
    "schema", "InstanceRecord", "InstanceSet", "encode_rle", "decode_rle",
    "encode_sparse_positions", "rle_foreground_positions", "decode_rle_cropped",
    "from_label_image", "VALID_STATUSES", "AUTHORITATIVE_STATUSES",
]
