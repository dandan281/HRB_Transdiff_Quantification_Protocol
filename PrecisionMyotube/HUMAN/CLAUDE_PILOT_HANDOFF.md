# Claude pilot-interface handoff

> **Superseded 2026-07-21.** This handoff documents the retired two-annotator pilot. The active
> Claude task is R04 blind-repeat mode for the 30-case single-operator audit described in
> `../DEVELOPMENT_PLAN.md`.

Owner: Claude Code annotation-tools lane (CL01/CL02)

Codex has frozen and hashed six annotation packages in `pilot_package_index.json`. All 100 target
proposal IDs load unchanged through Claude's existing package loader.

Before independent pilot annotation begins, the Claude-owned interface needs observable support
for the frozen task workflow:

- load `pilot_manifest.json` or `pilot_package_index.json`;
- navigate directly among the selected target IDs rather than requiring reviewers to search among
  4,973 proposal prompts;
- show progress through exactly 100 tasks;
- record an explicit disposition for every task (`instance` or `not_an_instance`);
- preserve source task lineage when a proposal is refined, split, or merged;
- write the reviewer decision JSON defined in `PILOT_REVIEW_CONTRACT.md`;
- keep the existing prompt-vs-truth, overlap-safe, per-instance status, and reviewer guardrails.

Acceptance checks:

1. Two reviewer IDs can independently complete the same frozen task list.
2. Every task has one disposition and traceable final instance IDs when applicable.
3. Codex `pilot-review-validate` accepts each export unchanged.
4. Codex `pilot-review-compare` produces a complete disagreement queue.

Codex must not implement these GUI changes under `annotation_tools/**`.
