# PrecisionMyotube — Assisted Annotation Tools (Claude Code lane)

Machinery that **creates** reviewed full-area myotube masks, feeding the
canonical pipeline owned by Codex. This package implements Wave 1 tasks
**CL01** (assisted annotation interface) and **CL02** (overlap-safe round-trip),
plus the shared schema bridge used by the model laboratories.

> **Ownership (per the Parallel Execution Manual §2.1).** This lane owns
> `annotation_tools/**`. It consumes `precision_myotube.schema` **read-only** as
> the single source of truth for the frozen `InstanceSet` contract and never
> edits `precision_myotube/**` or `tests/**`. Any needed core change goes through
> `coordination/requests/claude/`.

## Why write `InstanceSet` JSON, not a label TIFF

A single mutually exclusive label image cannot represent two independently
visible **crossing** myotubes that share projected pixels — at the crossing, one
arm must overwrite the other. The annotation lane stores each instance as its
own overlap-safe mask and exports canonical RLE JSON, so both arms survive.
`annotation-tools verify-roundtrip` proves this and quantifies exactly how many
pixels a flat TIFF would destroy.

## Architecture

| Module | Role |
|---|---|
| [`_schema_bridge.py`](annotation_tools/_schema_bridge.py) | Locates and re-exports the canonical `precision_myotube.schema` (read-only). |
| [`masks.py`](annotation_tools/masks.py) | `SparseMask`: overlap-safe, memory-proportional instance masks (tight bbox, not full field). |
| [`model.py`](annotation_tools/model.py) | `AnnotationSession`: the headless, fully tested core — create/split/merge/erase/refine, status/review, prompt layer, export guardrails. |
| [`package.py`](annotation_tools/package.py) | Loads a Codex `annotation-package` directory (channels + proposals as distinct prompt layers). |
| [`roundtrip.py`](annotation_tools/roundtrip.py) | CL02 overlap round-trip proof + TIFF-limitation documentation. |
| [`napari_app.py`](annotation_tools/napari_app.py) | Thin napari/micro-sam GUI over the model (lazy-imported; needs the isolated env). |
| [`cli.py`](annotation_tools/cli.py) | `launch`, `verify-roundtrip`, `validate`. |

The GUI is a thin shell: **all scientific guardrails live in `model.py` and are
unit-tested headless**, so they hold regardless of the display layer.

## Scientific guardrails (definition of done)

- **Prompt vs truth.** Automated proposals load as `is_prompt` instances, shown
  in a low-opacity, distinctly-coloured layer named "PROMPTS - not truth", and
  are **excluded from export by default**. Deleting all prompts never touches raw
  channels. A prompt cannot receive a status until explicitly accepted.
- **No bulk authority.** There is deliberately no "mark all complete/reviewed"
  operation; status and review are set one instance at a time.
- **Export refuses incomplete review state.** Export blocks any accepted instance
  with a missing/invalid status, and any reviewed instance with no reviewer.
- **Overlap-safe.** Crossing instances keep two full masks; JSON round-trip
  preserves shared pixels and per-instance IoU.
- **Authoritative output** = `InstanceSet` JSON (validated by Codex's schema)
  **plus** a `review_log.jsonl` capturing reviewer/action provenance.

## Install & run

Headless core / tests (any env with numpy, scipy, tifffile):

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools"
python -m pytest annotation_tools/tests -q
python -m annotation_tools verify-roundtrip --out annotation_tools/_roundtrip
```

Interactive GUI (isolated environment — never into `cpenv`):

```powershell
conda env create -f annotation_tools/environment.yml
conda activate pm-annotate
pip install -e PrecisionMyotube --no-deps
pip install -e annotation_tools
annotation-tools launch --package PrecisionMyotube/annotation_work/32_C08_smoke
```

## Handoff artifacts (per manual §10.1)

- Commit + this README + launch command above.
- Demo artifact: `annotation-tools verify-roundtrip` report (11/11 checks).
- Schema-validation result: `annotation-tools validate --instances <exported.json>`
  and Codex's `InstanceSet.load()` both accept the export unchanged
  (`tests/test_model.py::test_create_and_export_roundtrips_through_canonical_schema`).
- Any missing core capability is filed under `coordination/requests/claude/`.
