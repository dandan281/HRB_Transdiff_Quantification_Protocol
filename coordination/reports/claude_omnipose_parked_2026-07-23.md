# T02 candidate 2 (Omnipose) — built, validated, PARKED

**Decision (operator, 2026-07-23):** do not train Omnipose now. Move it to future
potential work. The harness is complete and validated; only the GPU training run
is deferred.

## Why parked
- Dominant project error is `fragment_too_short` = genuine **signal-gap
  fragmentation** in Desmin. A per-image segmentation model sees the same gap, and
  the gap-bridging cases are labelled `ambiguous` → **excluded from Omnipose's
  training set**. So Omnipose is unlikely to fix the failure mode that matters.
- Best realistic outcome was a candidate slightly cleaner on continuous fibres but
  no better on fragmentation — a ~20 GPU-hour spend to likely confirm a negative.
- Workstation is blue-screening (`HYPERVISOR_ERROR` / `DPC_WATCHDOG`); an
  `nvlddmkm` error coincided with the batch-8 OOM. Not a machine to run 20h on.

## State at parking (nothing wasted)
- `model_labs/omnipose_lab/` — `ignore_policy.py`, `data.py`, `train_fold.py`,
  `infer_fold.py`, `run_folds.py`; shared `model_labs/_shared/eval_gt.py`.
- Ignore-mask decision settled by measurement (paint-out, with A/B ablation arm).
- Orchestration smoke passed: both arms train+infer+seal+ablate, 5.3 GB peak,
  per-fold resumable. 20 new tests; **173 pass total**.
- `run_folds.py` is ready to run unchanged when resumed. Suggested first real check:
  one fold, one arm, 100 epochs (~1–2 h) before committing to all 12.

## To resume (future)
1. Update NVIDIA driver (32.0.15.7627, Apr 2025) + confirm stability; re-run
   `verify_env.py` and the cpenv check.
2. `python model_labs/omnipose_lab/run_folds.py --wells <one> --arms paint_out --epochs 100`
3. If quality justifies it, run full 6 folds × 2 arms, then hand sealed
   predictions to Codex for T03.

For the integrator: this does **not** close T02. Candidate 2 remains a required
deliverable per DEVELOPMENT_PLAN §10 whenever training resumes.
