# Cross-lane change request (O01)

> Filed by the requesting lane; **implemented by the owning lane**. The requester
> does read-only verification only and never edits the target path.

- **Request ID:** `YYYY-MM-DD-<short-slug>`
- **From lane:** Claude Code
- **To lane / owner:** Codex (core) | Human/data manager | …
- **Related task IDs:** e.g. CL02, C02
- **Target path (owner edits):** e.g. `precision_myotube/…`

## 1. Missing capability (observable behavior, not an implementation)

_State what should be observable, not how to build it._

## 2. Minimal failing artifact

_Path to a fixture / sample that reproduces the need. Must contain no
unauthorized production labels and no Plate 26 data._

## 3. Acceptance checks (objective pass/fail)

- [ ] …

## 4. Resolution (filled by the owner)

- Owner commit:
- Tests:
- Requester read-only verification:
