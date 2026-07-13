# Stage 1 — Threshold Builder (contract)

> Read `../conventions.md` first. The orchestrator passes every path explicitly.
> This stage may **write only** into `runs/<stem>/stage1_threshold/`.
> It may **read only** the source `.nd2`.

## Responsibility
Extract the 3 channels, decide which channel is the fiber/trace target ("primary") vs the
overlap marker vs DAPI **per image** (never hardcoded), and choose the display brightness/contrast
on the primary channel that gives maximum myotube visibility. The chosen scaling is the **single
source of truth** reused by every later stage (renders + composite).

## Inputs
- `--src` : the `.nd2` file.

## Steps (run by orchestrator)
1. `extract.ijm`  (Fiji `-batch`)  → `ch0_raw16.tif ch1_raw16.tif ch2_raw16.tif` + `ch*_preview.png`.
2. `threshold.py` (Python)         → `metadata.json` + `bc_contactsheet.png`.
   - channel roles via nuclei/fiber scoring (same logic as the validated `analyze_channels.py`);
   - B&C: evaluates display-max candidates (p95/p97.5/p99/p99.5), scores each by fiber crispness ×
     visible-fraction, penalising >3% saturation; records all candidates + the chosen one.
3. `adjust_primary.ijm` (Fiji `-batch`) → `ch{primary}_adjusted8.tif` (the shared "duplicate")
   and `signal.png` (bg-subtracted primary @ chosen max — used by merge + Stage-4 flagging).

## Outputs (this folder only)
`ch{0,1,2}_raw16.tif`, `ch*_preview.png`, `metadata.json`, `bc_contactsheet.png`,
`ch{primary}_adjusted8.tif`, `signal.png`, `*_log.txt`.

## Human-in-the-loop (optional)
`bc_contactsheet.png` shows each candidate scaling with [CHOSEN] marked. The orchestrator can pause
for the user to override the max (`--force-max`) before stages 2/3 run. Channel roles can be
overridden with `--force-primary`.

## Must NOT
- Detect, trace, merge, or measure anything (that is Stages 2–5).
- Read or write any other stage's folder.
