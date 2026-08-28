# Handoff — T04 tracer lane, post-CV. Paste into a fresh Claude Code session.

Copy everything below the line.

---

You are picking up the **T04 centreline-tracer lane** of the PrecisionMyotube
project in `c:\Users\liqig\Documents\HRB_Transdiff`, branch `cleanup-2026-08`.
The full session record is
`coordination/reports/claude_tracer_lane_session_2026-08-23.md` (§1–§12) —
read it before changing anything. This prompt is the operational summary.

The operator runs all cluster commands and pastes output back; you have no
SSH. Label every copy-paste block **WINDOWS** or **TILLICUM** — mixing them
has cost this project multiple wasted runs.

## State of the lane, established by measurement — do not re-litigate

1. **The tracer works and is cross-validated.** Pipeline: a 4-head U-Net
   (1.93 M params, `net.py`: centre/orient/crossing/offset) predicts fields
   from the raw image; a deterministic walk (`oracle_trace.py`) traces them.
   Leave-one-well-out CV (10 folds, `_runs/net_cv/<WELL>/best.pt`): every
   well scored by a network that never saw it. **nms config: total length
   0.95× the operator's, length rank ρ +0.90, count ρ +0.95, both surviving
   drop-one-well** (`_runs/plate32_cv_report.json`, figure
   `plate32_cv_final.png`, per-well overlays `_runs/net_cv/overlay_cv_*.png`).
2. **"Perfect" is measured, not 1.0.** The operator blind re-traced a D04
   window (`coordination/retrace_check/`): human self-consistency is
   **recall 0.71, count 1.72×, per-fibre length error 0.096, identity through
   crossings 0.27**. Judge every metric against this row, never against an
   implied 1.0. The original annotation is *selective* — fresh eyes found 72%
   more fibres — so recall-against-annotation understates every candidate.
3. **The only gap outside human noise: matched per-fibre length, 0.32 vs
   0.096.** Its cause is diagnosed (§11): split fibres are **missing
   middles** — the two traced pieces sit a median of 90 px apart (92/114
   lateral/far), where the walk lost the ridge for tens of px.
4. **The oracle ceiling equals human repeatability** (mdape 0.085 vs 0.096;
   identity 0.97 on perfect fields). The walk is not the limiter.
5. **Nine mechanisms measured and rejected for the length gap** — do NOT
   retry them without new information: ridge-weighted MSE alone (v2 helped
   brightness, not width), augmentation (v3), offset-vector head (v4,
   sign-ambiguity collapse), soft Dice (v5), 300-epoch cosine (v6),
   snapped targets (v7 — snap validated and kept, ridge unchanged), analytic
   steerable ridge, path smoothing, endpoint stitching, orientation bridging
   (v1 non-selective: +160 merges; v2 with mutual + end-landing guards:
   1.8 repairs/merge but 9% coverage and **mdape never improved**).
6. **The predicted centre ridge is 12 px FWHM vs a 4 px target** in every
   version; an 8-tile memorisation test reaches 7 px, so capacity is not the
   limiter. The halo keeps raw centre >0.25 near fibres, so support
   thresholds cannot discriminate (this killed the bridge's floor axis).
7. **Instruments are validated before use** (`ridge_yardstick.py`): every
   per-point ridge yardstick FAILS on synthetic fibres (neighbour capture or
   speckle attenuation); only `trace_mean` (profile-average per trace)
   passes. With it: the operator's traces sit ~2.1 px SD off the image ridge
   (no bias); the validated snap (`snap_targets.py`, per-trace opt-in,
   arc-length ≤1%, zero theft) is applied to training targets and kept.

## The decision the user made

Ship the well-level product; pursue the sealed benchmark next; the
history-conditioned stepping head only if the benchmark shows per-fibre
length blocks a real use. Omnipose is **benchmark-only** (user decision —
no further development).

## Your actions, in order

1. **Ask the operator, then commit** the uncommitted lane work (git status
   will show: `bridge.py`, `stitcher.py`, `cv_report.py`, `quantify_plate.py`,
   `ridge_yardstick.py`, rewritten `snap_targets.py`, `benchmark_versions.py`,
   modified `train_tracer.py`/`infer_trace.py`/`net.py`, the report, the
   retrace_check folder). `_runs/` stays untracked (gitignored). Never
   commit without asking.
2. **Build the trace→mask bridge for the sealed T03 benchmark**: convert
   walked polylines to instance masks (stamp each object's paths at the
   corpus `width_px: 8.0`) and score via
   `model_labs/omnipose_lab/eval_on_bootstrap.py` on `bootstrap_v1`
   (PLATE_23 — you must run inference on PLATE_23 images; the pipeline so
   far only ran PLATE_32). Predeclared metrics, in order: `length_mdape` vs
   floor **0.3169**, `false_split_count` vs **52/375**, pooled recall vs
   **0.928**. Report numbers to Codex; Codex rules. Never score your own
   candidate into a ruling.
3. **Preserve the Omnipose checkpoint before the purge** (sha256
   `5250ee87…`, on `/gpfs/scrubbed/danlovuw/precision_myotube` — that
   filesystem is purged on a schedule). Give the operator a **WINDOWS** scp
   block, verify md5 on arrival. Then run the same plate-32 well-level
   comparison for Omnipose masks (`measure_mask` in
   `PrecisionMyotube/precision_myotube/geometry.py` is the mask-length
   convention) so the user's "Omnipose as benchmark" row exists.
4. **Only if the user asks**: the history-conditioned stepping head — the
   one untried mechanism (predict the next walk step from the last few
   patches, DBT's sequential idea on our backbone). Pre-declare its gate on
   tune wells before training anything.

## Conventions that bind you

- Tune/test well split for ANY post-hoc knob: tune on C02 C03 C05 C11 D02,
  freeze, claim on B02 D04 D08 D09 D11. Sweep full range, take the plateau,
  state the metric beside the threshold.
- Report pooled AND per-well with a drop-one-well check; never mean-of-wells
  alone. Present every metric next to the human-ceiling row.
- Sealed artifacts are read-only (`PrecisionMyotube/runs/`,
  `annotation_work/bootstrap_v1`, `model_labs/classical/_runs/`,
  `Conversion_Efficiency` results).
- Validate any new measuring instrument on synthetic data with known truth
  before pointing it at real data (`ridge_yardstick.py` is the template and
  the cautionary tale — two retracted measurements preceded it).
- Environments: torch+CUDA lives in conda env `pm-omnipose` (name is
  historical — it is just the GPU env); CPU-only target building and tests
  run in `pm-annotate`. Tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
  model_labs/tests/test_oracle_trace.py model_labs/tests/test_tracer_targets.py -q`
  (15 green as of handoff).

## Traps this lane has already paid for

- **Never filter a log you might need** — a grep in the CV driver discarded
  the traceback of a crashed fold. Full per-fold logs, filter the view.
- **The laptop is shared with another Claude session** that runs 11+ GB
  jobs; the machine's commit ceiling (~65 GB) cannot hold both. Standing
  policy from the user: the other session has priority — pause, watch for
  its python to exit, resume (the CV script `_runs/run_cv.sh` is resumable;
  fold = complete iff `manifest.json` exists).
- **Windows sleep kills the CUDA context silently** — training hangs, GPU
  0%, process alive. Keep the machine awake for long runs; check
  utilization, not process existence.
- argparse defaults can silently override swept class defaults — CLI flags
  in this lane default to None meaning "use the frozen plateau".
- Walk claims/coverage are painted at 1 px density (step-spaced points leave
  gaps that create duplicate traces and 4× coverage under-reads).
- The dihedral augmentation's orientation fixup is verified by `--augcheck`
  (median must be 0.00; the max statistic is poisoned by equidistant ties).
- A "plateau" can be a floor: probe the loss floor (GT scored as the
  prediction) before believing any training curve; BCE on soft labels floors
  at the label entropy (0.082, measured).

## Files

| path | what |
|---|---|
| `coordination/reports/claude_tracer_lane_session_2026-08-23.md` | the full measured record, §1–§12 |
| `model_labs/tracer_lab/oracle_trace.py` | the walk + selftest + scorer (`score_against_gt`) |
| `model_labs/tracer_lab/net.py`, `train_tracer.py` | model, probe/augcheck/overfit/train |
| `model_labs/tracer_lab/infer_trace.py` | field prediction, nms/offset preps |
| `model_labs/tracer_lab/cv_report.py` | the cross-validated plate table |
| `model_labs/tracer_lab/ridge_yardstick.py` | validated-instrument harness |
| `model_labs/tracer_lab/snap_targets.py` | validated target snap (applied) |
| `model_labs/tracer_lab/stitcher.py`, `bridge.py` | refuted mechanisms, kept as the record |
| `model_labs/tracer_lab/_runs/net_cv/` | 10 fold checkpoints + per-well overlays |
| `model_labs/tracer_lab/_runs/plate32_cv_report.json` | the headline numbers |
| `coordination/retrace_check/` | the human-ceiling measurement |

`_runs/` is gitignored and local-only — warn the user before anything that
could delete it.

Nothing beyond commit `ed6b75f` is committed. Ask before committing.

---
