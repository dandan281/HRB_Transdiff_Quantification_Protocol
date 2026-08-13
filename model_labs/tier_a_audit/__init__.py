"""Read-only audit of the Tier-A conversion-efficiency pipeline.

This package reproduces, hashes, and cross-checks the Conversion_Efficiency
`New_Quantif_P23` result **without modifying it**. It exists so the integrator can
adopt (or reject) the newer ring/Otsu method as project-canonical on verified
evidence rather than on a lane report's say-so.

Hard constraints (enforced by never opening these for write):
- `Conversion_Efficiency/**`, `PrecisionMyotube/DEVELOPMENT_PLAN.md`, and
  `coordination/WORKBOARD.md` are READ-ONLY inputs.
- The audit does not change the production method, does not declare Tier A
  released, and does not re-run Cellpose-SAM or touch `cpenv`. It reproduces the
  deterministic downstream (ring readout -> pooled-Otsu -> per-cell classification)
  from the package's own cached intermediates, and hashes the upstream it cannot
  re-execute.

See `README.md` for scope and `audit.py:run_audit` for the entry point.
"""
