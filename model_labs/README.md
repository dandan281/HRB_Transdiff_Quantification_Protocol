# PrecisionMyotube — Model Laboratories (Claude Code lane, CL03)

Isolated, reproducible experimental environments for the elongated-object model
bake-off. This lane owns `model_labs/omnipose/**` and `model_labs/microsam/**`
(plus `model_labs/_shared/**`, the framework-agnostic helpers). It consumes
`precision_myotube.schema` read-only and hands normalized predictions to Codex's
common benchmark via C02.

> **P0.4 PRIMARY FAILURE TO AVOID:** never install Omnipose or micro-sam into
> `Conversion_Efficiency/cpenv`. Each laboratory is a separate conda environment
> so no experiment can change the validated Cellpose/nucleus pipeline (the known
> C08 run must keep reproducing 10,114 valid nuclei).

## Layout

```
model_labs/
  _shared/            framework-agnostic, GPU-free, fully tested
    schema_bridge.py    read-only locator for precision_myotube.schema
    predict_export.py   normalize predictions -> unreviewed InstanceSet + provenance
    channel_config.py   frozen desmin_only / desmin_dapi configs (CL03.3)
    synthetic.py        synthetic elongated-object fixtures
    smoke.py            end-to-end plumbing smoke (CL03.2)
  omnipose/           pinned env, smoke_test.py, channel_config.json, resource_note.md
  microsam/           pinned env, smoke_test.py, channel_config.json, resource_note.md
  tests/              pytest for the shared core
```

## Definition of done (CL03)

- **Isolation:** each lab has its own `environment.yml`; none imports or modifies
  `cpenv`. (Enforced by convention + the header warnings in each env file.)
- **Smoke predictions pass the canonical adapter:** `smoke_test.py` runs a short
  inference (real when the framework is installed, deterministic fallback
  otherwise) and the exported `InstanceSet` passes `InstanceSet.validate()`.
- **Versions, seeds, hardware recorded:** every export carries a
  `ModelProvenance` (architecture, checkpoint/env/data hashes, seed, channels,
  `used_prompts`) written to a `*.prediction_manifest.json`.

## Handoff to Codex (C02)

Predictions are written per-model/version/image (`<model>/<version>/<image_id>.instances.json`)
so no model overwrites another. Every record is `reviewed=False`, `source=<model>`
(C02.4) with full provenance (C02.5). A convenience label TIFF is emitted too but
is explicitly non-authoritative — crossings collapse in a flat raster, so the
InstanceSet JSON is the truth. **Connected-component count is never treated as an
independent-myotube count** (C02 PRIMARY FAILURE TO AVOID).

## Development posture

The project owner has authorized immediate exploratory model development. Candidate training may
consume `PrecisionMyotube/annotation_work/bootstrap_v1` while preserving its single-operator,
proposal-conditioned limitations. Evidence gates constrain scientific claims and release, not
engineering progress. The validated nucleus environment remains isolated and must not be modified.

## Run

```powershell
$env:PYTHONPATH = "PrecisionMyotube;model_labs"
python -m pytest model_labs/tests -q
python model_labs/omnipose/smoke_test.py --out model_labs/omnipose/_smoke
python model_labs/microsam/smoke_test.py --out model_labs/microsam/_smoke
```
