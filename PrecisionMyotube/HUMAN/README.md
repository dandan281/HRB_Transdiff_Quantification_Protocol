# Single-operator human review handoff

## What changed

The project has one human operator, so the original two-annotator 100-task pilot is retired. Its
manifest, templates, and failed `g1_result.json` remain in this folder as historical artifacts; they
are not tasks the user must complete.

The active workflow uses the 1,800 decisions the user already completed across six Plate-23 wells.
The blinded repeat of 30 selected cases has also been completed. Its numeric checks passed, but
the historical export lacked sufficient provenance; the optional 10-case recovery is deferred.
There is no current human task.

## Work already completed

- 1,800 proposal reviews across six Plate-23 wells.
- 377 reviewed complete masks.
- 31 border-truncated masks, excluded from complete-object measurements/training.
- 839 ambiguous retained instances, excluded from authoritative measurements/training.
- 553 rejected proposals.
- 40 correction pairs.

These labels are single-operator and began from machine proposals. They are useful development
labels, but they are not independent consensus ground truth.

The operator approved `../ANNOTATION_PROTOCOL.md`, including the explicit definitions and decision
order for `complete`, `border_truncated`, `occluded`, and `ambiguous`. Those definitions are the
source of truth for any later review.

The JSON pilot manifest, two reviewer templates, checklist, and old G1 output retained here are
historical reproducibility artifacts. The duplicate Codex pilot HTML site was intentionally
removed to avoid overlapping Claude's annotation interface; `HUMAN/pilot_site` is not an active
or expected directory.

## Current gate result and next human action

The 30-case round-2 review exists and its numerical checks pass: 90% disposition agreement, zero
unsafe border/complete transitions, and median mask IoU 1.0 over eight complete/complete pairs.
The official [G-SO1 result](g_so1_result.json) nevertheless failed as repeatability evidence because the original
browser export did not record reviewer identity or UTC review time. The export is now fixed, the
two disputed complete masks are applied through the training exclusion manifest (377 to 375), and the targeted 10-case
[recheck protocol](G_SO1_RECHECK_PROTOCOL.md) is approved.

The project owner has directed implementation to continue immediately. The recheck is therefore
optional and is not being served now. There is no current human action. If repeatability evidence
is wanted later, use the dated protocol and then:

1. Review the 10 randomized cases without seeing prior decisions or model suggestions.
2. Explicitly click a disposition for every case, even when retaining `Ambiguous`.
3. Export the session with stable reviewer ID `reviewer_01`; do not edit the JSON.

## How disagreements are handled

Because both sessions come from the same person, disagreement is not adjudicated by pretending a
second expert exists. Any inconsistent case becomes `ambiguous` or is excluded from training. The
report will explicitly state that inter-rater agreement was not measured.

## G-SO1 pass conditions

- all 30 tasks have traceable reviewer/time/status records;
- at least 85 percent disposition agreement;
- no unsafe border-to-complete classification;
- median mask IoU at least 0.80 among at least eight cases called complete twice;
- every disagreement is excluded or ambiguous.

If the audit fails, only 10 cases from the affected category are repeated after the rule/tool is
fixed. The old 100-task dual pilot is not restarted.

## What remains agent-owned

- Claude: begin T02 candidate implementation from `annotation_work/bootstrap_v1`.
- Codex: enforce field-level splits and run T03 benchmarking when predictions exist.
- Human: no action during current development.

See `../DEVELOPMENT_PLAN.md` for the complete version-2 plan.
