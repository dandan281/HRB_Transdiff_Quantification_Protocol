# Pilot review export contract

> **Legacy contract.** The project has one human operator, so this two-reviewer contract is no
> longer an active requirement. It remains for reproducibility of the original plan. See
> `../DEVELOPMENT_PLAN.md` and `SINGLE_OPERATOR_CHECKLIST.md` for the active workflow.

This contract connects Claude's overlap-safe annotation exports to Codex's frozen 100-task pilot.
It does not change Claude's interface ownership or biological decisions.

Each annotator produces:

1. one canonical `InstanceSet` JSON and companion `review_log.jsonl` per field;
2. one reviewer decision JSON containing exactly the 100 frozen task IDs;
3. a unique, non-empty reviewer ID that differs from the other annotator.

## Per-task decision

Every task records:

- `task_id`, `field_key`, and `source_object_id` copied unchanged from the frozen manifest;
- `disposition`:
  - `instance` when the target corresponds to one or more retained myotube instances;
  - `not_an_instance` when the proposal is debris, noise, or otherwise not a biological instance;
- `status`: one of `complete`, `ambiguous`, `occluded`, or `border_truncated` for an instance;
- `final_instance_ids`: the retained IDs after refinement, split, or merge;
- `notes`: an explanation when useful.

`not_an_instance` has a null status and no final instance IDs. This is a proposal disposition, not
a fifth biological instance status.

## Validation

Codex validation requires every retained final instance to:

- exist in the field's canonical `InstanceSet`;
- have `reviewed=true`;
- carry the status recorded for the task;
- have matching reviewer identity in Claude's companion review log.

Run:

```powershell
python -m precision_myotube pilot-review-validate `
  --manifest HUMAN/pilot_manifest.json `
  --review HUMAN/annotator_a/pilot_review.json `
  --out HUMAN/annotator_a/validation.json
```

After both reviews validate:

```powershell
python -m precision_myotube pilot-review-compare `
  --manifest HUMAN/pilot_manifest.json `
  --review-a HUMAN/annotator_a/pilot_review.json `
  --review-b HUMAN/annotator_b/pilot_review.json `
  --out HUMAN/adjudication/disagreements.json
```

The comparison reports disposition disagreements, status disagreements, and union-mask IoU below
the configured threshold. Every reported disagreement remains pending until a human adjudicator
resolves it or retains it as ambiguous.
