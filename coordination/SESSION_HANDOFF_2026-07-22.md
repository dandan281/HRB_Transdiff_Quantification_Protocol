# Session handoff - 2026-07-22

## Start here

Read in this order:

1. `PrecisionMyotube/DEVELOPMENT_PLAN.md` - authoritative plan and full continuation record.
2. `coordination/WORKBOARD.md` - compact live task state.
3. `coordination/requests/claude/2026-07-21-t02-start.md` - exact next model-lane contract.
4. `PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json` - frozen T02 input.
5. `PrecisionMyotube/HUMAN/g_so1_result.json` - repeatability evidence status and limitations.
6. `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md` - binding statistics and replication rules.

Older dual-annotation plans and the 100-task pilot are historical. Do not reactivate them.
The annotation protocol is human-approved. The duplicate Codex `HUMAN/pilot_site` was intentionally
removed; use Claude's `annotation_tools/annotation_tools/qc_review/` workflow for any future UI.

## Current state in one paragraph

Wave 0 reconciliation and T01 are complete. Six Plate-23 wells contain 1,800 single-operator
decisions: 377 complete before exclusions, 31 border-truncated, 839 ambiguous, and 553 rejected.
Two complete masks that flipped to ambiguous are binding exclusions, leaving 375 real bootstrap
masks. T01 contains those 375 masks, 40 separate real correction pairs, and 2,290 eligible
synthetic pairs, split by whole well. G-SO1 numeric metrics passed, but historical repeat-export
provenance is incomplete; the owner explicitly made its 10-case recovery optional and nonblocking.
No human action is required. The next task is T02 real classical and Omnipose implementation.

## Frozen identifiers

| Artifact | SHA-256 |
|---|---|
| Six-well annotation snapshot | `5171286b5bcb153ad45cfe5db7ae532c4f6158a4f6fcaebbc25ad2649dc36994` |
| Triage CSV | `a1781a4cdae5c881a0179ce57575c5a2107dc7314c1a717ff85173dae50c2fe4` |
| Training exclusions | `b15492c167c8555dd8d306db5285792eea5ca6447cdc935268aa160d7ff847fb` |
| Bootstrap v1 manifest | `44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94` |
| Correction manifest | `6d6bd461b38c90c6c61879a3193633d13623c5b107611cdf3e43e38d1ae10e98` |
| Synthetic manifest | `3d8b7ad632dd94feac8bac78ed5f9f44eb76a8e75e0c1ae5a66dee7e20410a99` |

## Exact restart task

Implement T02 from `bootstrap_v1`:

- classical ridge/graph reproducible floor;
- real Omnipose train/inference harness in its own environment;
- six whole-well leave-one-well-out folds;
- synthetic pairs for pretraining/augmentation only;
- 40 real corrections kept separate, with tuning use disclosed;
- canonical overlap-aware predictions, always `reviewed=false`;
- run manifests containing commands, hashes, seeds, thresholds, fold, timing, and failures.

Then give sealed predictions to Codex for T03 common scoring. Existing Omnipose/micro-sam smoke
fallbacks and the stub checkpoint are plumbing only; they are not trained candidates.

## Do not do

- Do not wait for the optional 10-case recheck.
- Do not ask the user to perform the retired 100-task dual-annotation pilot.
- Do not recreate the removed duplicate `HUMAN/pilot_site` unless the ownership plan explicitly
  changes.
- Do not commit, push, clean, or create worktrees without explicit authorization.
- Do not split object crops from a well across folds.
- Do not train on border, ambiguous, rejected, occluded, or excluded masks.
- Do not call Plate 26 sealed or the current labels consensus/prospective truth.
- Do not install candidate frameworks into `Conversion_Efficiency/cpenv`.
- Do not delete the ignored `bootstrap_v1` directory; preserve its manifest hash.
- Do not ask for the source plates again; Plates 23, 26, 28, and 32 are under `Q_PLATES/Q_Plates`.

## Verification checkpoint

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python -m pytest PrecisionMyotube/tests -q --basetemp tmp/pytest_precision_handoff
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_labs_handoff
Get-FileHash PrecisionMyotube/annotation_work/bootstrap_v1/bootstrap_manifest.json -Algorithm SHA256
```

Expected: 41 canonical tests and 49 annotation/model-lab tests pass. The bootstrap hash is
`44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94`.

T03 must resample whole
held-out wells, report macro and micro metrics, and treat current same-plate results as internal
model evidence rather than biological treatment inference.

## Repository condition

HEAD is `0322ebf534fa5c279f109c1145ce2da39fa69fe4` on `main`. There is one worktree. `.gitignore` is
modified and most active project paths are untracked. Those files contain current user/agent work;
preserve them. R01 cleanup and intentional versioning are deferred and do not block T02.
