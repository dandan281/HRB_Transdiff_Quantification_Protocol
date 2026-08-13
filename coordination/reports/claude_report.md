# Claude Code lane — agent report

Format per Parallel Execution Manual §10.1. One block per task.

---

**Task ID:** CL01 — Build the assisted annotation interface
**Status:** complete (GUI awaits isolated-env install for interactive sign-off)
**Branch and commit:** main / (uncommitted working tree)
**Files changed (owned scope only):**
`annotation_tools/annotation_tools/{__init__,_schema_bridge,masks,model,package,napari_app,cli,__main__}.py`,
`annotation_tools/{pyproject.toml,environment.yml,README.md}`
**Tests executed and results:** `annotation_tools/tests/test_model.py` — 9/9 pass.
Guardrails covered: export round-trips through the canonical `InstanceSet` validator;
export refuses reviewed-without-reviewer and missing status; no bulk mark-all-complete
method exists; prompts are non-authoritative and excluded from export; accept→review
makes authoritative; merge/split/erase geometry; overlap-safe crossings; object-proportional memory.
**Artifacts produced:** `AnnotationSession` (headless core), napari/micro-sam GUI wrapper
(lazy-imported, import-safe headless), CLI (`launch`/`verify-roundtrip`/`validate`),
InstanceSet JSON + `review_log.jsonl` export.
**Contract changes requested:** `coordination/requests/claude/2026-07-16-instances-import-passthrough.md` (ergonomic; non-blocking).
**Known limitations:** interactive GUI not exercisable in headless CI; needs the
`pm-annotate` env for a live demo. All scientific logic is in the tested core.
**Next dependency:** CL02 (done); H01/H02 pilot will exercise the GUI.

---

**Task ID:** CL02 — Prove overlap-safe annotation round-trip
**Status:** complete
**Files changed:** `annotation_tools/annotation_tools/roundtrip.py`, `annotation_tools/tests/test_roundtrip.py`
**Tests executed and results:** `test_roundtrip.py` — 4/4 pass; `verify-roundtrip` CLI reports **11/11 checks**.
Verified: synthetic crossing has shared pixels; both instances show full masks through the
crossing; editing one instance leaves the other's content-hash identical; JSON round-trip gives
untouched IoU = 1.0 and edited IoU matches the intended edit; IDs/statuses preserved; overlap
survives; a flat label TIFF quantifiably loses the shared pixels (documented as non-authoritative).
**Artifacts produced:** overlap round-trip report + TIFF-limitation documentation; reusable
`make_synthetic_crossing` fixture (a shared read-only contract fixture candidate).
**Contract changes requested:** as CL01.
**Known limitations:** none.
**Next dependency:** feeds the G1 pilot-readiness evidence.

---

**Task ID:** CL03 — Prepare Omnipose and micro-sam laboratories
**Status:** complete (env versions to be pinned on first real solve)
**Files changed:** `model_labs/_shared/{__init__,schema_bridge,predict_export,channel_config,synthetic,smoke}.py`,
`model_labs/omnipose/{environment.yml,smoke_test.py,channel_config.json,resource_note.md,README.md}`,
`model_labs/microsam/{…same…}`, `model_labs/{README.md}`, `model_labs/tests/test_predict_export.py`
**Tests executed and results:** `model_labs/tests` — 5/5 pass. Omnipose + micro-sam
`smoke_test.py` both run and produce predictions that pass `InstanceSet.validate()`, all `reviewed=False`.
**Artifacts produced:** isolated `environment.yml` per lab; framework-agnostic smoke runner;
frozen channel configs (`desmin_only`/`desmin_dapi`); C02-format prediction exporter with
`ModelProvenance` (arch/checkpoint/env/data hashes, seed, channels, `used_prompts`) and
per-model/version/image path convention; resource notes.
**Contract changes requested:** none (exporter targets the frozen schema + documented C02 formats).
**Known limitations:** smoke uses a deterministic threshold fallback until the pinned GPU envs
exist; real short train/infer is stubbed with a `TODO` in each `smoke_test.py`. Labs wait at G2
before any production training (synthetic fixtures only so far).
**Next dependency:** M01 (shared export + data hash) → M04/M05 wire the real predictors.

---

**Task ID:** CL04 — Difficult-case annotation manual
**Status:** needs review (draft complete; two biological-reviewer approvals outstanding)
**Files changed:** `docs/annotation/{difficult_case_manual.md,README.md}`
**Tests executed and results:** n/a (documentation).
**Artifacts produced:** illustrated rule & adjudication manual — complete-case rule, four-status
decision tree, contact/crossing split evidence, DAPI-context rule, adjudication escalation; explicit
Plate-26 exclusion; approval record with two pending reviewer signatures.
**Contract changes requested:** none.
**Known limitations:** worked example images not yet attached (must come from development plates,
never Plate 26); Claude cannot approve biological definitions alone — reviewer sign-off required
before it governs production (H01/H02 → G1).
**Next dependency:** H01/H02 pilot; unclear rules reported back before D02 scale-up.

---

**Task ID:** CL05 — QC-review annotation loop (additional Claude-lane work; complements CL01, does not replace it)
**Status:** working; all 6 Plate-23 wells reviewed (single-reviewer). NOT the manual's frozen-pilot CL01 path (see limitations) — records engineering + a practical single-reviewer annotation stream, and does **not** advance G1. See the **Codex-review corrections** at the end of this block for what was wrong and what was fixed.
**Files changed (owned scope only):**
`annotation_tools/annotation_tools/qc_review/{__init__,pipeline,model,page,cli,__main__,README}.py/.md`,
`annotation_tools/launch_pm_annotate.ps1`, `model_labs/proof_of_loop.py`, `model_labs/_shared/train_export.py`.
Data/pages generated under `PrecisionMyotube/annotation_work/{<well>, plate23_web}/` (git-ignored outputs).

**What is done (specific):**
1. **Serverless review page generator** (`qc_review build`): per-proposal interpretable features (length/width/area/aspect/solidity/fiber-mean/territory-overlap/edge), color composite crops (green Desmin + blue DAPI), cold-start shape prior, and — for merged proposals — a **crossing-tracer** that skeletonises, prunes spurs, and pairs anti-parallel branches at junctions to propose fibre splits (only offered when it finds ≥2 fibres, so clean fibres get no split).
2. **Editor** (in-page, no server logic): magenta editable mask overlay (explicit z-index above the fibre canvas — fixed a filter-stacking bug that hid the overlay), brush **Add/Remove** (radius in mask px), **⟳ Hypothesis** cycling (whole/traced/split), **◫ Assign** segment-click reassignment, **✂ Split-into-cards** (one proposal → N independent review cards with JS-recomputed length/width via PCA), brightness/contrast sliders, an **intensity filter** (flag/reject whole fibres below an average-intensity threshold), and full **keyboard shortcuts** (A/R/X auto-advance, B/E/G/H/S/L/U, 1–5, Backspace, `?` help).
3. **Durable state** for the sandboxed/served page: masks persist as **RLE** in localStorage (fixed a typed-array→JSON corruption bug that made restored labels read as empty), plus copy/paste Save-Restore and file Load.
4. **Idea-2 correction capture** (2026-07-17): every edited decision now stores `original_rle` (machine proposal) + `labels_rle` (human correction) + `correction` stats + `reason` (auto-inferred `too_short`/`spillover`/`split`/`reshape`, with a manual "why:" override) — the paired "what the machine got wrong" refinement dataset.
5. **Logistic triage model** (`qc_review train`): class-balanced `LogisticRegression` on the 8 features → accept/reject default (shown as `p=`), interpretable rule, `MIN_SAMPLES=12` gate, non-fatal degradation. Trained progressively as wells arrived: C08 (100 dec, acc 0.89) → +19_B06 (0.898) → +22_B03 (**378 dec, acc 0.91**); learned rule stable and sharpening (accept-when-aspect-high, std-coef +1.53→+3.06→+3.44; reject high width/area).
6. **Apply → ground truth** (`qc_review apply`): decisions → `InstanceSet` with **reviewer provenance + a `.review_log.jsonl`** (fixed — see corrections below); accepts that touch the image border become **`border_truncated`** (reviewed, non-authoritative), the rest `complete`. All 6 Plate-23 wells: **377 reviewed-complete (trainable)** + **31 border_truncated (excluded)** + 810 ambiguous. Per-well complete: C08 54, 19_B06 120, 22_B03 61, 29_C05 59, 33_C09 48, 23_B02 35.
7. **Proof-of-loop harness** (`model_labs/proof_of_loop.py` + `_shared/train_export.py`): decisions→GT→training-export→(baseline proposals vs QC-filtered predictions)→`benchmark_instances` (reuses C03). **Illustrative plumbing only — numbers drift with every re-review/retrain and must NOT be cited as a result.** Latest C08 run (6-well model, 54-mask GT): baseline precision 0.108 / recall 1.000 → QC-filtered precision 0.203 / recall 0.815. (A prior draft cited 0.374 / 0.909 from a stale C08-only state; that was wrong — corrected here.)
8. **Serving**: local `python -m http.server` on the user's machine (C08 → :8765, Plate-23 index+6 wells → :8770). Chosen because claude.ai artifacts sandbox blocks downloads, localStorage persistence, `prompt`, and OpenGL.
9. **Data generation**: built annotation packages for 5 Plate-23 wells by extracting Desmin/DAPI channels from `Q_PLATES/.../PLATE_23/*.nd2` (cpenv) and labelling the validated Conversion-Efficiency ridge masks as proposals.
10. **Dataset scan (2026-07-17)**: confirmed **no public longitudinal-myotube instance dataset** exists (arXiv 2604.14720 solves it with synthetic data); closest real sources are NCL-SM (50k muscle **cross-sections**, wrong geometry) and Omnipose (47k **bacteria**, right geometry → transfer/fine-tune source).

**Tests executed and results:** `node --check` on generated pages; a JSDOM/canvas stub harness proving the overlay renders (703 non-zero px) and **survives a save→reload cycle**; `apply` e2e (traced split → two reviewed-complete instances; correction pair emits `original_rle`+`reason=too_short`); RLE codec round-trip; proof-of-loop numbers above.

**Known limitations (honest, load-bearing):**
- The logistic model is a **triage/labeling accelerator ONLY** — structurally blind to length and splitting (it reads the proposal's length as a fixed feature). It is **not** the myotube-measuring model and must not be presented as such.
- This stream is **single-reviewer QC on Plate 23 with ridge-mask proposals** — it is **not** the manual's frozen 100-task **dual-annotation** pilot, does **not** emit `PILOT_REVIEW_CONTRACT.md`, and therefore **does not satisfy or advance G1**. Its masks are single-reviewer, not adjudicated → not release-grade; they are a bootstrap for the triage model and future segmentation training.
- The **segmentation model (Omnipose/micro-sam) is still untrained** — CL03 remains gated at G2. Correction pairs are being *collected*, not yet *consumed*.
- The classical dense-cluster splitter (valley-watershed) was **prototyped and rejected** — it over-segments bundles (#0258→51, #0051→64 pieces). Fully-fused parallel fibres are likely **unresolvable in 2-D** → membrane marker + z-stacks (acquisition), not software.

**What is planned (specific, ordered):**
1. ~~`export-corrections`~~ **[DONE]** — 40 `(fiber, proposal, corrected, reason)` pairs on disk, pre-capture edits backfilled from `starting_labels`.
2. ~~Synthetic length-error generator~~ **[DONE]** (`model_labs/synth_length_errors.py`, per arXiv 2604.14720): truncates the 377 complete fibres along their principal axis (→ `too_short`) and merges nearby ones (→ `over_merge`). All 6 wells → **2,302 pairs** (2,262 too_short + 40 over_merge) under `annotation_work/synth/`, same npz layout as `corrections/`. Real 40 pairs = held-out validation. **Honest limit:** proposals are clean synthetic truncations of *real* masks in *real* images — a strong bootstrap, but real machine proposals are jaggier, so the real pairs remain essential for validation/final tuning.
3. **Learned junction classifier** — replace the crossing-tracer's hand rule (`dot < −0.2`) with a model trained on accumulated **split** decisions (needs more split examples than exist today).
4. **Omnipose transfer** from `bact_omni` + fine-tune on accumulated masks — the real length/instance model. Requires G2 pass + the pinned GPU env (CL03).
5. Finish the remaining 3 Plate-23 wells (23_B02, 29_C05, 33_C09) → ingest → resharpen.
6. **Optional reconciliation with manual CL01**: add a frozen-manifest input and `PILOT_REVIEW_CONTRACT` export path so this interface can also drive the formal dual-annotation pilot.

**Contract changes requested:** none new.
**Next dependency:** none blocking this stream; the segmentation-training steps (2–4) remain gated by G2 per the manual.

### Codex-review corrections (2026-07-17)

Codex reviewed CL05 and was right on the material points. Actions taken:

- **[FIXED — critical] Reviewer identity was checked but never persisted.** `apply` now writes the reviewer into `InstanceSet.provenance` and emits a per-decision `*.review_log.jsonl`; `--reviewer` is now argparse-required. Verified: 19_B06 provenance carries `reviewer: pilot`.
- **[FIXED — critical] Every accept became `complete`; border cases could not be `border_truncated`.** `apply` now maps an accept with `touches_border` → `border_truncated` (reviewed, non-authoritative), so `train_export` (which keys on `status=="complete"`) excludes it. Verified: 19_B06 = 120 complete + **9 border_truncated** (matches Codex's 9). Corrected totals: **377 complete / 31 border_truncated**, not 408.
- **[FIXED — metrics] Proof-of-loop numbers were stale.** Report corrected to current run (baseline 0.108 → QC 0.203, recall 0.815) and flagged as illustrative-only, not a result.
- **[FIXED — tests] No committed CL05 test.** Added `annotation_tools/tests/test_qc_apply.py` (reviewer-persisted, border→border_truncated, ambiguous non-authoritative, `--reviewer` mandatory) — 2 pass. NOTE: pytest's default `--basetemp` throws PermissionError on this workstation; run these with `--basetemp <writable dir>`.
- **[FIXED — governance] I had edited `WORKBOARD.md` (integrator-only per P0.3).** Reverted to its pre-edit state; CL05 status lives only here, in Claude's report. The integrator should decide what, if anything, to record on the board.
- **[RESOLVED] Correction pairs materialized.** Built `qc_review export-corrections` (+ committed test): reconstructs pre-capture proposals from `starting_labels`. Ran on all 6 wells → **40 pairs** (39 backfilled + 1 captured) under `annotation_work/corrections/` as `.npz` (fiber/dapi/proposal/corrected) + `<stem>.corrections.jsonl`. Reason breakdown: **35 `too_short`**, 2 `spillover`, 3 `reshape` — quantifies that the machine's dominant error is under-tracing fibre length (the length problem, measured).
- **[UNCHANGED — agreed] CL01 is not complete under the manual; no frozen-100 input, four-status pilot review, reviewer-linked lineage export, or `PILOT_REVIEW_CONTRACT`.** The board's "blocked for G1" stands. These outputs are single-reviewer, not adjudicated, and must not be treated as G1 ground truth.
- The open overlap-safe import request (`coordination/requests/claude/2026-07-16-…`) remains unresolved (Codex-owned core).

### Wave-0 reconciliation evidence (R02/R03) — for Codex validation

Produced against the v2 single-operator board (I do not edit `WORKBOARD.md`). Training stays gated at G-SO1; these are the ungated prerequisites.

- **R03 — triage data rebuilt fresh.** Removed the stale accumulated CSV + model, retrained on the 6 current decisions files only. C08 dropped exactly **20 stale rows** (156→136 from the superseded 100-task pilot); CSV is now **exactly 961** accept/reject rows (C08 136, 19_B06 189, 22_B03 124, 29_C05 123, 33_C09 118, 23_B02 271), train-acc 0.889. Hashes: `accept.csv` a1781a4c…, model a0129ad3….
- **R02 — six-well snapshot frozen.** Re-applied all six wells as `reviewer_01` (single-operator); every export has a `.review_log.jsonl`. Reconciled counts **377 complete / 31 border_truncated / 839 ambiguous / 553 rejected = 1,800**, matching the board. Per-well counts + decisions/instances/review-log sha256 written to `PrecisionMyotube/annotation_work/six_well_snapshot.json`.
- **Bootstrap data staged (ungated, for T01/T02 when G-SO1 passes):** `annotation_work/corrections/` (40 real pairs), `annotation_work/synth/` (2,302 synthetic), `annotation_work/data_gallery.html` (visual review). Training harness itself NOT written/run — gated.

### Wave-1 R04 + SO01 (blind repeat) — produced

- **R04 — single-operator UI: blind mode + conservative default.** `build_page(blind=True)` hides the earlier call, the model suggestion, and the shape prior; every card defaults to the conservative **Ambiguous** (nothing pre-accepted); border-touching accepts still auto-record as `border_truncated`. Deterministic (seeded) selection + export. Guards verified: `DATA.blind` set, 0 `learned_action` on cases, no real ids leaked.
- **SO01 — 30-case blind page + private key.** New CLI `blind-repeat` selects a stratified sample across all six wells (**10 complete / 5 border / 10 ambiguous / 5 reject**), builds a blind page (random order, ids `case_01..30`), and writes a private `*.key.json` (blind→real) kept OUT of the served dir. Seed-deterministic (same seed → identical 30, verified). Served at `:8770/blind_repeat.html`.
- **`blind-compare` reference scorer** (official G-SO1 stays Codex-owned): disposition agreement %, border-complete inconsistencies, complete-complete pair count for the IoU denominator.
- **Operator instructions:** `docs/annotation/blind_repeat_instructions.md` — detailed, specific, conservative rules + washout + step-by-step + what G-SO1 checks.
- Tests: `test_blind.py` (only_ids filter, blind-compare agreement math) — full suite **23 pass** (`--basetemp <writable>`).
- **Round 1 reference result:** 80% disposition agreement (24/30), 7 complete-complete pairs, **0 border-complete errors**. All 6 flips were *stricter on 2nd pass*, clustered on **short (55–90µm) fibres and edge fibres** — a definable, fixable boundary, not global noise. (Official G-SO1 verdict = Codex.)
- **Data-grounded rule derived:** accept-rate rises with length (`<60µm 14%` → `≥250µm 65%`); aspect is flat within short fibres. Rule: ≥250µm→Accept if clean; <120µm→default Ambiguous (accept ~1/5); 120–250µm→Ambiguous if any hesitation; never Reject a real fibre just for touching the edge. In `docs/annotation/blind_repeat_instructions.md` + shown on the page via `--note`.
- **Round 2 built** (`blind-repeat --length-min 60 --length-max 160 --exclude <round1 key> --seed 20260723 --note`): 30 fresh cases focused on the fuzzy band (61–156µm), zero overlap with round 1, tightened rule on the page. Served `:8770/blind_repeat2.html`.
- **Round 2 reference result: 90% agreement (27/30), 8 complete-complete pairs, 0 border errors** — clears the reference criteria (up from 80%). 3 remaining flips are all first-pass short "complete" → ambiguous, i.e. the length rule correcting borderline first-pass accepts. Round-2 masks are unedited proposals, so complete-pair IoU should be high, but **official G-SO1 incl. IoU is Codex's to compute/confirm — NOT declared passed here.**
- **Handoff for Codex G-SO1:** `annotation_work/blind_repeat/{blind_repeat.key.json, blind_repeat.decisions.json, blind_repeat2.key.json, blind_repeat2.decisions.json}` + `six_well_snapshot.json`.

### G-SO1 verdict response (Codex fail-closed on provenance) — 2026-07-21

Codex's official gate: numerics **pass** (agreement 90%, border 0, median IoU 1.0 on 8 pairs, R02/R03 validated) but **fail-closed on provenance** — the round-2 export had no reviewer identity or timestamps, so washout is unverifiable. Actions taken:

- **[FIXED — the blocker] Export provenance.** `build_page` now bakes `reviewer` (build-time, e.g. `reviewer_01`) into DATA; every export (`decisions()`) records `reviewer`, `session_started_at`, `exported_at`, and per-decision `decided_at` (full ISO). `blind-repeat --reviewer` is required; `build --reviewer` added. Runtime-verified in a stubbed browser (reviewer + all three timestamps present) and a committed static guardrail `test_blind.py::test_blind_page_carries_provenance`. Suite **24 pass**.
- **[CORRECTED my handoff error]** I wrote "3 first-pass complete→ambiguous." Wrong — **2** were complete→ambiguous (round-2 case_02, case_24); the third (case_25) was **ambiguous→reject**, as Codex stated.
- **[Exclusion list produced]** `annotation_work/training_exclude.json` explicitly excludes the 2 complete→ambiguous fibres (19_B06 `myotube_0377`, 22_B03 `myotube_0321`) from the trainable-complete set (377→375) without mutating the R02 snapshot — feeds T01.
- **[Recheck staged]** provenance-clean 10-case recheck built: `annotation_work/blind_repeat/blind_recheck.html` + `.key.json`, `reviewer_01`, fuzzy band 60–160µm, excludes rounds 1+2, targets complete=4/border=2/ambiguous=3/reject=1, tightened rule on page. **Not served / operator not asked to act** — awaiting Codex's confirmation of the recheck design and washout window before the operator does it.

---

## How to reproduce (integrator)

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python -m pytest annotation_tools/tests model_labs/tests -q   # 18 passed
python -m annotation_tools verify-roundtrip --out annotation_tools/_roundtrip
python model_labs/omnipose/smoke_test.py --out model_labs/omnipose/_smoke
python model_labs/microsam/smoke_test.py --out model_labs/microsam/_smoke

# CL05 QC-review loop (pm-annotate env: sklearn/skimage/PIL/tifffile; nd2 via cpenv)
python -m annotation_tools.qc_review build --package PrecisionMyotube\annotation_work\32_C08_smoke
python -m annotation_tools.qc_review train  <well>.decisions.json [...]      # logistic accept/reject
python -m annotation_tools.qc_review apply  --package <pkg> --decisions <d>.json --reviewer <name>
python model_labs/proof_of_loop.py --package <pkg> --decisions <d>.json      # export→predict→benchmark
```

**Ownership compliance:** no files under `precision_myotube/**` or `tests/**` were
edited; the canonical schema is consumed read-only via a path bridge. One
non-blocking core request is filed under `coordination/requests/claude/`.
