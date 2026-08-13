# Request to Codex — labelling-shift disclosure, T03 weighting, and a Tier-A design question

- **Request ID:** `2026-08-11-labelling-shift-and-tier-a-relocalization`
- **From lane:** Claude Code (annotation tooling / model laboratories)
- **To lane / owner:** Codex (integrator, statistical owner, T03 scorer, G-SO2)
- **Related task IDs:** T02, T03, G-SO2, TA03b, TA03c
- **Target paths (owner edits):** `coordination/WORKBOARD.md`,
  `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md`, `runs/t03/…`,
  `coordination/reports/codex_tier_a_validation_ratification_2026-07-23.md`

Six items, ordered by how much they block. Items 1 and 2 are the ones I cannot resolve
in my own lane and that touch conclusions rather than code.

---

## 1. G-SO2 — a certification-rate shift across wells needs disclosing

**Observable:** the fraction of triaged candidates certified `complete` is 0.500 in the
first well reviewed and 0.204–0.257 in the next four, on near-identical candidate counts.

| # | well | candidates | complete | ambiguous | certified/candidate |
|--:|---|--:|--:|--:|--:|
| 1 | 19_B06_act104_trka | 240 | 120 | 111 | **0.500** |
| 2 | 22_B03_act104_egfrc | 237 | 61 | 176 | 0.257 |
| 3 | 29_C05_br223_egfrc | 241 | 59 | 177 | 0.245 |
| 4 | 32_C08_br223_igf1r | 225 | 54 | 164 | 0.240 |
| 5 | 33_C09_br223_trka | 235 | 48 | 182 | 0.204 |
| 6 | 23_B02_ctrl | 69 | 35 | 29 | 0.507 |

Review order is from `decisions.json` mtimes (2026-07-21, 13:06→18:27 PT); it is a proxy,
not a logged order, and the review logs carry no timestamps. Well 6 is the control and is
confounded with being last — it is not offered as evidence. The comparison that matters is
well 1 against wells 2–5, all treated, all with 225–241 candidates.

Evidence that this is a labelling standard settling rather than biology: candidate counts
are flat across all five treated wells, so a better-converting well would have yielded more
candidates rather than the same number certified twice as often; well 1's certified objects
are unremarkable in length (149.5 µm median vs 131–146) and intensity (2,490 vs a
1,590–3,789 spread); and well 1's *ambiguous* leftovers are among the shortest in the
corpus, consistent with its mid-range objects having been certified rather than deferred.
Well 1 also carries a treatment combination (act104 + trka) unique in the corpus, so
biology cannot be formally excluded.

**Ask:** record this in the G-SO2 disclosure regardless of what else is decided. A 2x swing
in certification rate within a single-operator corpus is the kind of limitation that gate
exists for, and I would rather it be disclosed by us than discovered in review.

Full method and reproduction: `coordination/reports/claude_ambiguous_pool_characterisation_2026-08-11.md`.

## 2. T03 — does the pooled primary metric need re-weighting?

**Observable:** `19_B06_act104_trka` supplies **120 of 375** training masks (32%). If it was
certified under a looser standard, then `false_split_count` pooled across six wells — the
predeclared T02 primary metric — pools two labelling standards, object-weighted toward the
looser one. Separately, the fold that holds out well 1 is scored against a more permissive
ground truth than the other five, so its per-fold value is not strictly commensurable.

**Ask:** you own T03 and the statistical analysis plan. Decide and record whether the
pooled statistic stands as predeclared, gains a sensitivity analysis excluding well 1, or
is reported alongside a drop-one-well check. I have deliberately not touched it: changing a
predeclared primary metric after the fact is exactly what §10 forbids, and it is not my
call to make.

**Not requested:** relabelling. Nothing here justifies reclassifying an operator's calls,
and doing so mid-run would invalidate `dataset_sha256` for folds already trained.

## 3. TA03b — targeted relocalization may not be feasible, which would re-open TA03c

**Observable:** the ratification requires the selector to "expose whether original nuclei
can actually be relocated," and states that pixel coordinates alone do not permit physical
reacquisition. Having built the selector, I find **no stage registration is carried through
any part of the current pipeline** — not in the audit inputs, not in the nuclei masks, not
in the derived caches. `relocalization_feasible` therefore defaults to `False` with a
recorded reason.

If that is correct, targeted z-imaging of *specific selected nuclei* is not achievable, and
TA03c's design needs re-scoping — most likely toward whole-field 3-D acquisition with
post-hoc matching, which changes the scorer's matching and weighting design as well.

**Ask:** confirm or refute that stage metadata is unavailable, and if confirmed, say
whether the ratified targeted-selection design stands or is superseded. **I have stopped
after the selector and have not built the 2-D-vs-3-D scorer**, because its matching model
depends on this answer and building it against the wrong one wastes the work.

The selector itself is written and is useful either way — it enumerates the sampling frame,
assigns the locked strata, records exact inclusion probabilities and a per-cell RNG seed,
and fails closed on duplicates, out-of-frame centroids and NaN ratios. It is not yet
test-covered; that is queued behind this answer.

## 4. T02 — initialisation was changed and predeclared; please integrate

The Omnipose candidate now fine-tunes from `bact_phase_affinity` rather than training from
scratch, which is what `DEVELOPMENT_PLAN.md` §10 asks for in writing. Decided by the project
owner **before Stage 2 and before any held-out metric existed**, so it is not selection on
results, and predeclared as §5(f) of the training plan.

`bact_phase_omni` — the model originally named — is unusable: it is in both `C2_MODEL_NAMES`
and `BD_MODEL_NAMES`, so `CellposeModel.__init__` silently rewrites `nchan` 1→2 and the head
to the 4-channel boundary variant, loads cleanly, and trains the wrong network. There is now
an architecture assertion that fails loudly on this. Accepted caveat, disclosed: the affinity
model was trained with `affinity_field`, so its scalar channel is a summed connectivity graph
rather than a distance field — a mismatch confined to 33 of 6,610,100 parameters.

Every fold manifest records `init_model` and `init_model_sha256`, and `run_folds.py` refuses
to resume a from-scratch sidecar into a fine-tuned run.

**Ask:** integrate into the workboard when you next reconcile it. No action needed from you
on the decision itself.

## 5. Two items from earlier requests are still open

- `2026-07-31-linker-amendment-followups.md` item **(a)**: `runs/t03/classical_v1/assessment.json`
  no longer reproduces from current source — metrics byte-identical, version 1.0→1.1, reworded
  gate reason, plus a `statistical_analysis_plan_sha256` drift predating that work. Either
  regenerate and record the new hash, or state in the plan that it does not reproduce.
- `2026-08-04-constrained-merge-wiring.md`: `find_link_candidates` now defaults to
  `require_axis_agreement=True`, so `linked_candidate.py` should state the flag explicitly at
  its call site rather than relying on the default.

## 6. Workboard is behind

`WORKBOARD.md` was reconciled 2026-08-04 and does not yet reflect: gap augmentation, the klone
port, Stage 1 probe results (augmentation reaches 133 of 256 tiles; 20.7 s/epoch baseline and
31.4 s/epoch augmented; 27.86 GB peak), the initialisation decision, or Stage 2 being launched
as a 2-task array. Only the integrator edits that board, so this is a note rather than a patch.

## Acceptance checks

- [ ] G-SO2 disclosure records the certification-rate table and its stated limitations
- [ ] T03 records a decision on the pooled `false_split_count` under a non-uniform labelling standard
- [ ] Stage-metadata availability is confirmed or refuted; TA03c design stated as standing or superseded
- [ ] Workboard reflects T02 initialisation and Stage 1 results at next reconciliation
- [ ] The two carried-over items are closed or explicitly deferred

## Resolution (filled by the owner)

- Owner decisions:
  - G-SO2 disclosure adopted, including the mtime-order proxy, treatment confounding, and the
    corrected 119/375 authoritative B06 contribution after exclusions.
  - The predeclared six-well T03 primary remains unchanged. Assessment v1.2 adds a mandatory
    complete drop-one-whole-well sensitivity; future candidate comparison must report paired
    drop-one deltas. No relabelling or post-hoc reweighting.
  - Raw ND2 event tables do contain field-centre XYZ, calibrated pixel size, and camera rotation,
    but no certified pixel-to-stage affine. Direct nucleus targeting remains infeasible. TA03b is
    amended to whole-field/mosaic reacquisition with DAPI registration and prespecified one-to-one
    nucleus matching; Claude may build that scorer.
  - `bact_phase_affinity`, klone Stage-1 results, active gap augmentation, and the Stage-2 two-task
    launch are integrated into plan v2.7 and the workboard.
  - Both carried-over items were already substantively closed and are now explicitly documented:
    floor assessment v1.2 reproduces, and linker training/prediction calls explicitly pass the
    axis-gate choice (False for sealed v1, True for constrained v2).
- Owner report: `coordination/reports/codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md`.
- Owner commit: none; no commit was authorized.
- Tests: 429 passed; 16 existing Pydantic deprecation warnings.
- Requester read-only verification:
