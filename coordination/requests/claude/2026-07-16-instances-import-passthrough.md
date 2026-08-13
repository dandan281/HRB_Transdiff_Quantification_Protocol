# Cross-lane change request (O01)

- **Request ID:** `2026-07-16-instances-import-passthrough`
- **From lane:** Claude Code
- **To lane / owner:** Codex (core)
- **Related task IDs:** CL01, CL02, C02, P0.2
- **Target path (owner edits):** `precision_myotube/cli.py` (and/or C02 adapters)

## 1. Missing capability (observable behavior)

The annotation lane exports **overlap-safe `InstanceSet` JSON** directly (per
CL02: two crossing myotubes keep two full masks sharing pixels). Today the only
CLI ingest path is `import-labels`, which goes through `from_label_image` on a
**mutually exclusive** label TIFF — that path structurally cannot represent
crossings (one arm overwrites the other at the intersection).

`analyze --instances <file.json>` already consumes the JSON directly, so there is
**no functional blocker** for the annotation → analysis flow. The request is a
small ergonomic + safety passthrough so operators do not reach for the
overlap-lossy TIFF route by habit:

- A CLI verb (e.g. `import-instances`) that validates an externally produced
  `InstanceSet` JSON against the frozen schema and copies it into the run, **or**
- A documented note in `import-labels --help` and the README that
  annotation-tool JSON is authoritative and should not be round-tripped through a
  flat label TIFF.

## 2. Minimal failing artifact

`annotation_tools verify-roundtrip --out <dir>` writes
`crossing.instances.json` with two overlapping instances (16 shared pixels).
Passing its equivalent through a flat label TIFF loses those pixels — quantified
in the same report (`details.tiff_limitation`).

## 3. Acceptance checks (objective pass/fail)

- [ ] A two-crossing-instance `InstanceSet` JSON is ingested with both instances
      and their shared pixels intact (per-instance IoU = 1.0 vs source).
- [ ] The flat-TIFF path is either avoided for this case or documented as
      non-authoritative in `import-labels` help/README.

## 4. Resolution (filled by the owner)

- Owner commit:
- Tests:
- Requester read-only verification:
