# Omnipose laboratory — resource note (CL03.5)

Harness at `model_labs/omnipose_lab/` (named to avoid shadowing the installed
`omnipose` package — see `omnipose_lab/__init__.py`). Values below are **measured**
on the pinned `pm-omnipose` environment against real 3636×3636 16-bit fields,
unless marked as an estimate.

| Item | Planned / measured |
|---|---|
| GPU | RTX 5070 Ti Laptop (Blackwell, sm_120, cu128) — **separate** from cpenv |
| Tile size (train) | **per-instance**, bbox + 96px margin, floored 256 / capped 1024. A fixed 1024 tile left a median instance at 3.5% of its own crop and Omnipose's crop sampler died "Sparse or over-dense image detected"; sizing to the instance fixed it |
| Full-field inference tiling | `eval(tile=True, bsize=224, tile_overlap=0.1)` — stitches the **flow field** across tiles and reconstructs masks once over the whole field, so seam-crossing objects are never split (completion check below satisfied by construction) |
| Normalization | percentile (1.0 / 99.5) at **whole-field** scope, `normalize=False` downstream. Omnipose's per-tile normalization stretched fibre-free tiles' noise to full range on this ~98%-background field |
| Precision | mixed precision **off** — `cellpose_omni` calls `autocast()` with no `device_type`, which torch 2.11 rejects. Left stock rather than patched |
| Dataloader | in-process (`dataloader=False`) — the worker path calls `set_start_method('fork')`, absent on Windows |
| Runtime (train) | ~22 s/epoch at batch 4 / tyx 224, ~100–256 tiles/fold (CPU flow recompute dominates; GPU work is ~0.3 s/batch) |
| Runtime (infer, full field) | **31 s** per 3636² field |
| Peak GPU memory (train) | **5.3 GB** at batch 4 / tyx 224; 7.2 GB at batch 2 / tyx 256; batch 8 / tyx 384 **OOMs a 12 GB card** and torch reports it as `CUDA error: unknown error`, not a clean OOM |
| Peak GPU memory (infer) | **1.2 GB** |

> **Memory budget note (2026-07-23):** the workstation has been blue-screening
> (`HYPERVISOR_ERROR`/`DPC_WATCHDOG`), and an `nvlddmkm` error was logged at the
> exact minute the batch-8 OOM occurred. Keep the full run at **batch 4 (5.3 GB
> peak)** with headroom, not near the card limit, until the NVIDIA driver
> (32.0.15.7627, Apr 2025) is updated and stability confirmed. See
> `coordination/reports/claude_resume_state.md` §crash-triage.

**Full-run cost estimate (not yet run):** 300 epochs × 6 folds × 2 ablation arms
≈ **12 fold-trainings**. At ~22 s/epoch that is very roughly **20+ GPU-hours**
continuous. `run_folds.py` writes a per-fold sidecar and resumes, so a crash costs
one fold, not the run. Consider fewer epochs or a single arm first.

**Tiling rule (CL03.5 completion check):** satisfied — inference stitches in flow
space and reconstructs masks once over the whole field, so no object is split at a
seam and none is dropped. Training tiles are built per-well and per-instance, and
`instance_tiles` asserts every reviewed instance is whole in at least one tile.

**Seeds/hardware (CL03.4):** every training result records GPU, seed,
environment hash, and data hash (M01). These flow into `ModelProvenance`.
