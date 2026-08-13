"""Deterministic classical candidate for T02 (the reproducible floor)."""
from .ridge_graph import (  # noqa: F401
    FieldTrace, FilterParams, PARAM_GRID, TracerParams,
    assign_and_filter, assign_territory, filter_assigned,
    semantic_territory_cached, trace_fibers_parameterised,
)
