# Proposal QC review + learning loop

The "confirm, don't trace" path. Instead of hand-drawing every myotube, you triage
the automated **proposals**, and a small interpretable model learns your judgment
and pre-decides the next batch — the same loop as the old stage-4 QC page, adapted
to full-area instance proposals.

## The loop

```
build  → review.html (serverless page: crops + features + suggestions)
review → you Accept / Reject / Ambiguous each card → download <stem>.decisions.json
train  → LogisticRegression learns accept-vs-reject from your decisions
build  → next page shows the model's suggestions with p=, so you confirm
apply  → decisions → canonical InstanceSet JSON (accepted = reviewed complete)
```

## Commands

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools"
$py = "C:\Users\liqig\anaconda3\envs\pm-annotate\python.exe"

# 1. build the review page from an annotation package
& $py -m annotation_tools.qc_review build --package PrecisionMyotube\annotation_work\32_C08_smoke

# 2. (open review.html, triage, download <stem>.decisions.json)

# 3. teach the model your pattern (needs scikit-learn + joblib)
& $py -m annotation_tools.qc_review train 32_C08_br223_igf1r.decisions.json

# 4. rebuild — now suggestions are the model's, shown with probabilities
& $py -m annotation_tools.qc_review build --package PrecisionMyotube\annotation_work\32_C08_smoke

# 5. turn accepted proposals into canonical reviewed-complete masks
& $py -m annotation_tools.qc_review apply --package PrecisionMyotube\annotation_work\32_C08_smoke `
      --decisions 32_C08_br223_igf1r.decisions.json --reviewer <your-name>
```

## Why this is the automation payoff

- **Cold start:** suggestions come from a transparent shape prior (elongated,
  in-territory, non-edge proposals rank first). You confirm the obvious ones fast.
- **After ≥12 accept/reject decisions:** a class-balanced `LogisticRegression`
  (standardized features, milliseconds, CPU) sets each card's default and shows
  its probability. The learned rule is printed in plain English.
- The model only sets **review defaults** — it never auto-accepts. `apply` still
  requires `--reviewer` to promote a proposal to reviewed-complete, so authority
  stays human. `ambiguous` decisions are kept but never used as training labels.

## Files

| File | Role |
|---|---|
| `pipeline.py` | per-proposal interpretable features + crop thumbnails + cold-start prior |
| `model.py` | binary accept/reject `LogisticRegression` (fit / predict_default / rule text) |
| `page.py` | serverless `review.html` (embedded crops, learned defaults, localStorage, download) |
| `cli.py` | `build` / `train` / `apply` |
| `data/`, `models/` | accumulated decisions + fitted pipeline (git-ignored) |
