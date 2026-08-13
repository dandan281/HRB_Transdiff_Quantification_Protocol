# Codex lane report

Last updated: 2026-07-22

**Task IDs:** repository audit, R01-R04, SO01/H-SO01/G-SO1, C01-C05

**Status:** repository re-audited and version-2 single-operator plan published. Canonical C01-C04,
R02, R03, R04, and T01 are complete. At the project owner's direction, incomplete G-SO1 evidence
no longer blocks exploratory implementation. R01 repository cleanup is deferred and nonblocking.

**Verified current state:**

- PrecisionMyotube tests: 41 passed after the canonical statistics framework was added.
- Annotation/model-lab tests: 49 passed in `pm-annotate`; base Python lacks `skan`.
- Six Plate-23 wells: 1,800 decisions, 377 complete, 31 border-truncated, 839 ambiguous, 553
  rejected, and 40 correction pairs.
- Triage CSV: exactly 961 unique rows and an exact multiset match to current accept/reject
  decisions; no stale rows remain.
- Plate 26 cannot be called sealed because labels and prior runs are present.

**Plan change:** the user is the only human operator. The original H01/H02 dual-annotation pilot and
G1 gate are superseded. G-SO1 uses a blinded 30-case test-retest audit and explicitly does not claim
inter-rater agreement.

**G-SO1 metric evidence:** 27/30 agreement (90%), zero unsafe border/complete errors, eight
complete/complete pairs, and median mask IoU 1.0. Seven pairs have IoU 1.0 and one is 0.78655.
There are two complete-to-ambiguous disagreements that must be excluded from T01. A third
disagreement is ambiguous-to-reject and was never trainable. Inter-rater agreement was not
measured.

**Gate recovery:** the blind export now records reviewer, session/export time, and per-decision
time; its provenance-fix checkpoint had 24 passing tests. The two complete-to-ambiguous cases are bound to
`training_exclude.json` (377 to 375). Codex approved a fresh 10-case targeted design with no prior
round overlap or source-ID leakage. The user then explicitly removed this check from the
development critical path; it is retained only for an optional future repeatability claim.

**T01 implemented:** `annotation_work/bootstrap_v1` contains six field-level exports with 375 real
complete masks, 40 real correction pairs, and 2,290 eligible synthetic pairs. Twelve synthetic
derivatives of the two excluded real masks were caught and removed. The bootstrap manifest
SHA-256 is `44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94`.

**Artifacts:** `coordination/REPOSITORY_AUDIT_2026-07-21.md`,
`coordination/reports/g_so1_validation_2026-07-21.md`,
`PrecisionMyotube/HUMAN/g_so1_result.json`, `PrecisionMyotube/DEVELOPMENT_PLAN.md`, and
`coordination/SESSION_HANDOFF_2026-07-22.md`.

**Next dependency:** T02 starts immediately using whole-well folds. Correction pairs remain separate
from synthetic pretraining data. The optional recheck and retired 100-task pilot do not block it.

**Statistical guardrails added 2026-07-22:** `STATISTICAL_ANALYSIS_PLAN.md` is binding. Canonical
statistics now collapse technical observations to declared biological units, withhold inference
below three independent units, bootstrap whole units, report unstandardized effects and confidence
intervals, and support pre-specified Benjamini-Hochberg adjustment. T03 emits macro/micro summaries
and whole-well intervals. Release gates require adverse 95% confidence bounds, declared experimental
units, at least three biological replicates, and a locked prospective design. The current six
Plate-23 wells remain descriptive/internal for biological claims.
