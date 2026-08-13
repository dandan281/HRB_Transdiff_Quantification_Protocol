"""Read-only locator for the canonical ``precision_myotube.schema``.

Model laboratories consume the frozen InstanceSet contract; they never define or
edit it. This mirrors ``annotation_tools._schema_bridge`` so the labs run from a
bare checkout without installing the canonical package.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _load():
    try:
        import precision_myotube.schema as schema  # type: ignore
        return schema
    except ModuleNotFoundError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "PrecisionMyotube"
        if (candidate / "precision_myotube" / "schema.py").is_file():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            import precision_myotube.schema as schema  # type: ignore
            return schema
    raise ModuleNotFoundError(
        "Could not locate precision_myotube.schema; install `pip install -e "
        "PrecisionMyotube` or run from a checkout containing PrecisionMyotube/.")


schema = _load()
InstanceRecord = schema.InstanceRecord
InstanceSet = schema.InstanceSet
encode_rle = schema.encode_rle
encode_sparse_positions = schema.encode_sparse_positions
from_label_image = schema.from_label_image

__all__ = ["schema", "InstanceRecord", "InstanceSet", "encode_rle",
           "encode_sparse_positions", "from_label_image"]
