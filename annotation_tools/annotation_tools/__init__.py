"""PrecisionMyotube assisted annotation tooling (Claude Code lane, CL01/CL02).

This package builds the machinery that *creates* reviewed full-area myotube
masks. It consumes Codex's frozen ``precision_myotube.schema`` read-only and
never edits the canonical pipeline. The GUI (napari/micro-sam) is a thin view
over :class:`~annotation_tools.model.AnnotationSession`, which is fully testable
headless.
"""
from __future__ import annotations

from .model import AnnotationSession, AnnotationError, Instance
from .masks import SparseMask
from .package import load_annotation_package, LoadedPackage
from .roundtrip import run_overlap_roundtrip, make_synthetic_crossing

__all__ = [
    "AnnotationSession", "AnnotationError", "Instance", "SparseMask",
    "load_annotation_package", "LoadedPackage",
    "run_overlap_roundtrip", "make_synthetic_crossing",
]

__version__ = "0.1.0"
