# Learning layer — adapt to your review patterns (contract)

> Read `../conventions.md` first. Classical ML only (scikit-learn). **No deep learning / pytorch** —
> the data is small (tens of ambiguous cases per well) and the learned rule must stay interpretable.

## What it does
You review only the **ambiguous** cases; the system draws everything. Each reviewed case is a
training example: hand-crafted **features → your decision**. A small model per case type
(split / merge / occluded) learns your pattern and, on the next run, **pre-sets each case's default
to your likely choice** — so over time you confirm rather than correct.

## The 3 error classes the model must learn to avoid (user-stated targets)
1. **Trace too SHORT** → inaccurate myotube length; the real fibre is longer. Signal: a `redraw`
   with `len_ratio > 1` (you drew it longer), or accepting a `merge` that extends the trace.
   NOTE: a purely-too-short trace is not always flagged today (it isn't a split/merge/occluded
   case) — the planned **`extend` flag** (endpoint sits on continuing fibre signal) will surface
   these for review so the model can learn them proactively.
2. **Two SEPARATE myotubes seen as ONE kinky myotube** → must split. Signal: you `split` (or
   `redraw` into `drawn_count > 1`) a high-`max_bend` trace; the split model learns bend → split.
3. **Several short FRAGMENTS wrongly joined into one** → must keep separate. Signal: you `separate`
   a merge candidate, or `split`/`redraw` an over-joined trace; merge+split models learn the
   short/low-continuity → don't-connect boundary.

`data/redraws.csv` records `orig_count/drawn_count/orig_len_px/drawn_len_px/len_ratio`. Clear
count-changing redraws are also converted into binary training labels: one proposed trace redrawn as
multiple myotubes is a split-positive example, one proposed trace redrawn as one myotube is a
split-negative shape/length correction, one merge case redrawn as one myotube is merge-positive,
and one occluded case redrawn as a myotube is restore-positive. Ambiguous redraws stay only in
`redraws.csv`.

## Files
- `model.py` — feature schema (`FEATURE_KEYS`), label mapping, `fit()` (StandardScaler +
  LogisticRegression, class-balanced), `predict_default()`, interpretable rule text.
- `log_feedback.py` — after a review, append `(features, your action)` from `flags.json` +
  `decisions.json` to `data/{split,merge,occluded}.csv`. Clear redraw-vs-proposal count
  differences are converted into synthetic actions such as `redraw_split` / `redraw_keep` and used
  as labels. Deduped on `(stem, case_id)`.
- `train.py` — fit one model per type from the accumulated CSVs → `models/{type}.joblib` +
  `model_summary.json` (n, accuracy, ranked std-coefficients, a plain-English rule).
- `data/` — accumulating labeled feedback (one CSV per type). `models/` — fitted pipelines.

## Loop (driven by orchestrator)
1. `stage4_qc/flag.py` attaches a `features` block to every case and, if a model exists and has
   ≥ `MIN_SAMPLES` (12) both-class examples, sets `learned_default` + `learned_proba`.
2. `stage4_qc/build_review_html.py` uses `learned_default` as the pre-selected action (shown with
   `p=…` so it's transparent); the conservative default is the fallback.
3. On `--resume` (after your `decisions.json` is applied + measured), the orchestrator runs
   `log_feedback.py` then `train.py`. Disable with `--no-learn`.

## Guarantees / guardrails
- Until ≥12 both-class examples of a type exist, the model does NOT influence defaults
  (conservative defaults stand). Under-data → `train.py` removes any stale model for that type.
- The model only sets **review defaults** — it never auto-applies an edit. You always see and can
  override every case. (`reconcile.py` still only auto-applies the strict `auto` flag, not learned.)
- L2-regularized + class-balanced logistic regression; standardized features; CPU, milliseconds.

## Must NOT
- Install or use pytorch / GPU / neural nets. Auto-apply edits without your confirmation.
- Read or write outside `learning/` (except reading the reviewed run's `flags.json`/`decisions.json`
  via `log_feedback.py --out`).
