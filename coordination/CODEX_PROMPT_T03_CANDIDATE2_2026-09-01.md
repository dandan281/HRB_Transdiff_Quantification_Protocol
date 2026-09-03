# Codex prompt — T03 sealed benchmark, candidate #2 ruling

Copy everything below the line into a fresh Codex session. It is
self-contained; no other document is required.

---

You are Codex, the ruling authority for the **T03 sealed benchmark** of the
PrecisionMyotube project (repo `c:\Users\liqig\Documents\HRB_Transdiff`,
branch `cleanup-2026-08`, synced with `main` at `430b796`). Your role:
authorize (or refuse) a second candidate's one-shot run against the sealed
test set, then rule on the result. You never tune anything, and the
candidate's author (Claude, T04 lane) never scores its own candidate into
a ruling — you own the ruling.

## The benchmark (unchanged since its predeclaration)

- **Test set**: `PrecisionMyotube/annotation_work/bootstrap_v1` — 6 wells,
  375 reviewed-complete instances, PLATE_23, proposal-conditioned single
  operator. Sealed: no config may ever be tuned on it.
- **Predeclared metrics, in order**, against the classical floors:
  1. `length_mdape` — floor **0.3169**
  2. `false_split_count` — floor **52** / 375
  3. pooled `recall` — floor **0.928**
- Precision/F1 are NOT interpretable (sparse certified GT; a dense-trained
  model finds far more fibres than were certified) and are reported for
  completeness only.
- Standing caveat that travels with any recall comparison: the GT was
  triaged from the CLASSICAL pipeline's own proposals, so the 0.928 recall
  floor was set by the candidate whose proposals defined the GT.
- Leakage status (verified 2026-08-27): no bootstrap image was seen by any
  training fold — the recurring nd2 filenames across plate folders are
  per-plate acquisition indexes; pixel correlations ~0.001.

## Candidate #1 — already run (2026-08-27), for reference

Frozen nms walk (seed 0.4 / support 0.3 / claim 3.5 / rescue 1, crossing
0.4 / valid 0.2), fold-B02 checkpoint
(`model_labs/tracer_lab/_runs/net_cv/B02/best.pt`), polylines stamped as
8 px ribbons, primary row nms/≥50 µm:

| metric | candidate #1 | floor | verdict |
|---|---|---|---|
| length_mdape | **0.0864** | 0.3169 | better (3.7×) |
| false_split_count | **6** | 52 | better (8.7×) |
| pooled recall | **0.557** | 0.928 | worse |

Full artifacts: `model_labs/tracer_lab/_runs/eval_bootstrap_v1/`
(`eval_summary.json`, per-config exports, `width_cap_diagnostic.json` —
GT median width 8.2 px, ribbon recall ceiling 0.952, so the recall gap is
real coverage, not the ribbon convention).

## Candidate #2 — what is being submitted

The same frozen walk plus two identity mechanisms, **both frozen on
PLATE_32 tune wells (C02 C03 C05 C11 D02) and claimed on PLATE_32 test
wells (B02 D04 D08 D09 D11) before this submission; nothing was ever run
on PLATE_23 with them**:

1. **Junction weld** (`weld_objects` in
   `model_labs/tracer_lab/oracle_trace.py`): post-walk identity merge of
   co-linear pieces meeting at a predicted crossing; connector-angle +
   crossing gates; merges bookkeeping only (no fabricated arc). Frozen:
   dist 14 px / deg 12.5 / gate 12 px. PLATE_32 test-well claim: splits
   776→707, +12 merges, identity 0.379→0.417.
2. **Decompose-retrace identity repair** (`apply_repair` in
   `model_labs/tracer_lab/decompose_retrace.py`): first-pass objects are
   grouped so transverse crossers separate; each group is re-rendered as a
   sparse masked image and re-traced; re-traces act as WITNESSES only — a
   re-trace running ≥40 px along each of ≥2 first-pass pieces merges their
   identities; geometry stays first-pass. Frozen: witness 40 px, mask ext
   20 px, detect ext 90 px, contact 6 px, conflict 30°, bg pct 30.
   PLATE_32 test-well claim: splits 707→593 (every well improved),
   identity 0.417→0.474, +44 merges, mdape +0.022.

Expected effect on T03 metrics: `false_split_count` down (the mechanisms'
target), `length_mdape` roughly flat (weld/repair add no geometry; merged
sums move toward full-fibre length but wrong merges inflate), `recall`
possibly up slightly at IoU 0.5 (merged pieces form bigger masks). The
author predicts no metric crosses a floor in the wrong direction but has
NOT run it — that is what your authorization gates.

## The run command (one shot, GPU env)

    conda run -n pm-omnipose python model_labs/tracer_lab/eval_tracer_on_bootstrap.py --candidate 2

Writes to `model_labs/tracer_lab/_runs/eval_bootstrap_candidate2/`
(candidate #1's outputs are untouched). The script prints the same
per-well table and floors comparison as candidate #1; provenance records
version `cv_foldB02_weld_repair_v2` and the checkpoint sha256.

## What you must do

1. Confirm the submission is legal: configs frozen on PLATE_32 only
   (evidence: `_runs/weld_rescue_sweep.json`, `_runs/weld_rescue_claim.json`,
   `_runs/decompose_v1/results.json`, session report
   `coordination/reports/claude_tracer_t03_session_2026-08-27.md`
   §7b/§7g), and no prior candidate-2 run exists on PLATE_23.
2. Authorize and execute (or direct the operator to execute) the command
   above, exactly once.
3. Rule, in the predeclared metric order, against the floors AND against
   candidate #1 — labeling this "candidate #2 of 2 submitted"; multiple
   submissions are a multiplicity cost you should account for explicitly.
4. Record the ruling in `coordination/` and state which candidate (if
   either) is the project's standing T03 result.

Rules that bind everyone: the test set is read-only; no threshold may be
changed after seeing any candidate-2 number; a crashed run may be fixed
and re-launched, a completed run is final; report pooled and per-well.

---
