# Handoff / step-back review prompt — Precision Myotube, 2026-08-12

Paste this to the incoming agent. It states where the project is, what is actually
blocking it, why the current model decision was made, where to verify all of it, and what
a sceptical review should attack. It is written to be useful even if every conclusion in
it turns out to be wrong.

---

## 1. What the project is trying to do

Quantify myotube conversion from immunofluorescent images of transdifferentiated cells
(Plate 23, six wells, one human operator). Two tracks run in parallel:

- **Tier A** — field-level conversion efficiency (nuclei-based, Desmin ring intensity).
  Method is frozen and internally reproduced; blocked on orthogonal biological validation.
- **T01-T03 / model track** — instance segmentation of individual myotubes, benchmarked
  leave-one-well-out. This is where nearly all recent effort has gone.

The contract governing everything is `PrecisionMyotube/DEVELOPMENT_PLAN.md` (v2.7) and
`PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md`. Three lanes: **Codex** is integrator,
statistical owner and scorer; **Claude** builds annotation tooling and model laboratories;
a **human operator** does review and acquisition. Claude does not score its own candidate —
that lane split is what makes the comparison worth anything.

## 2. What is blocking, stated plainly

**The nominal blocker: T02 needs a second candidate.** The classical ridge floor is
candidate 1 and is sealed. T03 cannot select anything with one candidate, and everything
downstream — G-SO2, workflow freeze, prospective plate, any scientific release — queues
behind T03. The fragment linker was tried as candidate 2 and **failed hard**: a uniform
60-merge safety round measured its population over-merge rate at 0.649 (95% CI
0.450-0.832), roughly 350 wrong merges of 540 accepted against 11 false splits recovered.
Withdrawn from all use. Omnipose is the only remaining path.

**The immediate blocker: compute.** Omnipose Stage 2 (12 fold-trainings) has failed twice
on UW Hyak `klone`, burning 16+ GPU-hours for **zero** completed folds. Three separate
causes, detailed in `claude_session_state_2026-08-06.md` §4. The live decision is whether
to build checkpoint/resume machinery for a preemptible partition or pay ~$25-90 for
uninterrupted GPU time.

**The blocker a reviewer should take most seriously, and which nobody has called a
blocker yet: the benchmark may not be able to distinguish candidates.** Evidence gathered
in the last two days:

- Ground truth is **proposal-conditioned** — the classical detector proposed objects and
  the human triaged them, so GT is conditioned on candidate 1's output.
- **Single operator**, and the certification standard was not uniform: the first well
  reviewed certified 0.500 of its candidates, the next four certified 0.204-0.257, on
  near-identical candidate counts. That well supplies 119 of 375 training masks.
- **Precision and F1 are explicitly not interpretable** against a sparse reviewed GT
  (binding project-wide). **Recall moves 0.0027 per object**, so anything under ~5 objects
  is flat. That leaves `false_split_count` pooled over six wells as the only usable
  primary metric.
- 839 of 1,800 reviewed objects are `ambiguous` and excluded from supervision, with **no
  recorded rationale anywhere** — no reason codes, no timestamps. At most ~290 look
  recoverable; half the pool is dimmer or shorter than the certified 10th percentile, and
  the ambiguous population's median Desmin intensity (1,639) is statistically
  indistinguishable from the *rejected* population (1,645) against certified (2,455).

So: a second candidate is being produced at real cost, to be scored on a single
object-count statistic, against a reference that is single-operator, proposal-conditioned,
non-uniformly certified, and sparse. **Ask whether that comparison can return a meaningful
answer before spending more on producing the candidate.**

## 3. Why fine-tuning Omnipose — the decision chain

Each step is defensible on its own; judge the chain.

1. **Omnipose over Cellpose** because Cellpose predicts flows toward a cell centre, which
   is meaningless for a 400 µm fibre with no centre. Omnipose uses a distance field and
   flows along the medial axis, and its published wins are on bacteria — elongated,
   touching, variable length. Plain Cellpose was tried on myotubes and produced blobby
   output.
2. **It was parked** in 2026-07 because the dominant failure (signal-gap fragmentation) is
   labelled `ambiguous` and therefore excluded from training targets.
3. **Unparked** when measurement showed 173 of 374 certified masks already contain an
   internal near-background gap (longest 79.2 µm). `gap_augment` widens that supervision
   from certified masks without inventing any — it attenuates a Desmin band and leaves the
   mask untouched. Verified reaching the data: 133 synthetic tiles from 256 originals.
4. **Transfer rather than from-scratch** because `DEVELOPMENT_PLAN.md` §10 asks in writing
   for "real Omnipose transfer/fine-tuning", 375 instances is small for random
   initialisation, and the field's current default for small datasets is fine-tuning.
   Decided by the project owner before Stage 2 and before any held-out metric existed, and
   predeclared as §5(f) of the training plan, so it is not selection on results.
5. **`bact_phase_affinity` specifically** because it is the only shipped bacterial model
   that is architecturally a drop-in: 311/311 tensors, zero shape mismatches, `nchan=1`
   and the 3-channel head preserved. `bact_phase_omni` — the model originally named in the
   plan — silently rewrites `nchan` 1→2 and the head to the 4-channel boundary variant,
   loads cleanly, and trains the wrong network. There is now an assertion that fails loudly.
6. **Disclosed caveat:** the affinity model was trained with `affinity_field`, so its
   scalar channel is a summed connectivity graph rather than a distance field. The mismatch
   is confined to 33 parameters of 6,610,100; the encoder and both flow channels transfer
   normally.

**What this run explicitly cannot answer** (training plan §9): z-overlap. Omnipose is 2-D,
and 15 of 36 wrong linker merges were "dimension problems" — no 2-D architecture can see
which of two overlapping fibres is on top. If the dominant residual error is z-overlap,
this candidate cannot fix it and neither can any 2-D successor.

## 4. Where to look

**Contract and status**
- `PrecisionMyotube/DEVELOPMENT_PLAN.md` (v2.7) — the binding contract
- `PrecisionMyotube/STATISTICAL_ANALYSIS_PLAN.md` — how T03 is allowed to be scored
- `coordination/WORKBOARD.md` — task-by-task status; only the integrator edits it
- `coordination/reports/codex_g_so2_t03_and_tier_a_ruling_2026-08-12.md` — latest ruling:
  G-SO2 disclosure, T03 drop-one-well sensitivity, Tier-A relocalization amendment

**The model track**
- `coordination/reports/claude_omnipose_training_plan_2026-08-05.md` — predeclared design;
  §4 and §5 are binding and may not be revised after seeing a held-out metric
- `coordination/reports/claude_omnipose_initialisation_decision_2026-08-06.md` — the
  fine-tuning decision with the weight-shape evidence
- `coordination/reports/claude_session_state_2026-08-06.md` — running state; **§4 has the
  three Stage 2 failure modes and is the most useful single section for a reviewer**
- `model_labs/omnipose_lab/` — `train_fold.py`, `run_folds.py`, `data.py`, `gap_augment.py`
- `model_labs/omnipose/klone_*.slurm` — cluster scripts

**Data quality**
- `coordination/reports/claude_ambiguous_pool_characterisation_2026-08-11.md` — the 839
  ambiguous objects and the certification-rate shift
- `coordination/reports/claude_control_only_round_results_2026-08-04.md` — the linker's
  population safety measurement, and the best example in this repo of how to kill an idea

**Tier A**
- `model_labs/tier_a_audit/` — `audit.py` (frozen method), `selector.py`, `scorer.py`,
  `planning.py`
- `coordination/reports/claude_ta03c_acquisition_brief_2026-08-12.md` — z-stacks arrive
  **2026-08-19**; this says acquire ≥2 fields per well, not 1

**Verify rather than trust**
```powershell
# Corrected 2026-08-12: the previous single-line combined command
# (`pytest model_labs/tests annotation_tools/tests PrecisionMyotube -q`) does NOT
# work — it yields 342 passed / 154 collection errors. Only this two-invocation
# form (DEVELOPMENT_PLAN.md §13) reproduces the full suite: 57 + 439 = 496.
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
python -m pytest PrecisionMyotube/tests -q --basetemp tmp/pytest_pm      # 57 expected
python -m pytest annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_labs  # 439 expected
```
Run in the `pm-annotate` conda env. *(Updated 2026-08-12: work is now committed on the
`cleanup-2026-08` branch — see `cleanup_2026-08/ACTION_LOG.md`; `main` remains at
`0322ebf`.)* **Do not commit without asking.**

## 5. What the review should attack

Honest candidates for "we could have done this better", offered so the reviewer does not
have to be polite about finding them:

1. **Three Stage 2 failures share one shape:** each was invisible in the artifact meant to
   catch it. The probe looked healthy but did not share flags with the run it sized.
   `--requeue` looked like insurance but does not cover `TIMEOUT`. Per-fold sidecars looked
   like resume but checkpoint at fold granularity, useless when preemption is finer-grained
   than a fold. Is there a general practice that would have caught all three?
2. **The linker took three review rounds to kill.** The decisive artifact was a uniformly
   sampled 60-case population estimate. Rounds 1 and 2 were density-matched and structurally
   hid the tail. Should the uniform population estimate simply have come first?
3. **Effort was committed to producing a candidate before checking the benchmark could
   discriminate.** See §2. Is `false_split_count` on this GT capable of separating two
   detectors at all? What is its minimum detectable difference?
4. **Inference time has never been measured** and is paid 12 times per run. It could be a
   large fraction of the budget and nobody knows.
5. **839 ambiguous labels carry no reason codes.** One dropdown at review time would have
   made the largest untapped data pool diagnosable. It still is not implemented.
6. **All six wells were reviewed before anyone checked labelling consistency across them.**
   The 2x certification-rate shift was found a month later, by accident, while answering a
   different question.
7. **Is Omnipose still the right second candidate?** It was chosen before Cellpose-SAM,
   CellSAM and µSAM matured. Plain Cellpose failed here, but that is not evidence that a
   fine-tuned SAM-backbone generalist would. Benchmarks in 2026 report no single model
   winning everywhere, so this is an empirical question, not a settled one.
8. **Neurite tracing is the closest prior art and has not been mined.** Myotubes and
   neurites are the same geometric problem — thin curvilinear structures where connectivity
   across signal gaps is the hard part — and that field has two decades more experience
   with fragmentation and false joins.

## 6. The immediate open decisions

1. **Compute:** build checkpoint/resume for `ckpt-g2`, or pay ~$25-90 for ~25-30
   uninterrupted GPU-hours. Interrupted training is not merely slower — it confounds the
   ablation (gaps-ON folds are longer, so preempted more often), restarts the RNG stream so
   folds see differently-weighted samples, and forfeits reproducibility, which is a G-SO2
   disclosure item.
2. **Epochs:** 300 was chosen when the plan was from-scratch. With transfer it may be 2x
   more than needed. The probe's 100 uninterrupted epochs are on disk and settle it; reading
   *training* loss involves no held-out metric and so carries no selection risk.
3. **Tier A:** z-stacks arrive 2026-08-19. The acquisition brief asks for ≥2 fields per
   well and a Desmin-negative control, without which the fourth adverse-bound gate cannot be
   evaluated and no absolute conversion percentage can be claimed.

## 7. Ground rules that are not negotiable

- The T02 candidate is the **gaps-OFF** arm, predeclared. If gaps-ON scores better that is
  a finding to report, not a licence to swap.
- `false_split_count` pooled across six wells is the predeclared primary, now with a
  mandatory drop-one-well sensitivity. No post-hoc reweighting or relabelling.
- No threshold search on held-out results.
- Claude does not score its own candidate; Codex owns T03.
- Nothing under `Conversion_Efficiency/` gets written by model-lab code.
- Do not commit without asking.
