"""QC review loop for PrecisionMyotube proposals (the 'confirm, don't trace' path).

Instead of hand-tracing every myotube, you review proposals: a small, interpretable
logistic-regression QC model learns your accept/reject pattern from hand-crafted
features and pre-sets each proposal's default (shown transparently as ``p=``), so
over time you confirm rather than correct. Mirrors the old stage-4 learning loop,
adapted to full-area instance proposals.

Pieces:
- ``pipeline`` : per-proposal features + crop thumbnails + case assembly.
- ``model``    : class-balanced LogisticRegression accept/reject (scikit-learn).
- ``page``     : serverless review.html (embedded crops, learned defaults, download).
"""
from __future__ import annotations

__all__ = ["build_cases", "FEATURE_KEYS"]

from .pipeline import build_cases
from .model import FEATURE_KEYS
