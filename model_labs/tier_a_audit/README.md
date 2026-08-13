# Tier-A conversion audit (read-only)

Reproduces and pins the Conversion_Efficiency `New_Quantif_P23` result so the
integrator can decide whether the ring/Otsu method becomes project-canonical on
verified evidence. **It changes nothing** — `Conversion_Efficiency/**`, the plan,
and the workboard are read-only inputs; the audit writes only under its own
`_audit/` output directory.

## What it does
1. **Reproduces** `visualize_final.json` for all six wells from the package's cached
   intermediates (`dbs_cache/*_dbs.npy` + `plate23_nuclei/*_masks.npy`), using the
   declared frozen operating point (10 µm cytoplasmic ring, pooled log-Otsu
   threshold, one plate-wide value, nucleus area 50–500 µm²). `ring_intensity` and
   `classify` are transcribed verbatim from the package scripts. Reproduction is
   **exact** (threshold 440.77; C08 3,341/10,114 = 33.03 %).
2. **Verifies one plate-wide operating point** — a single pooled-Otsu threshold for
   all wells; fails if any per-well threshold appears or the declared method is not
   `pooled_otsu_log_uniform`.
3. **Reconciles** the C08 nucleus discrepancy (10,114 canonical vs 10,560
   MyoFuse-local) and **fails closed** if it cannot prove the cause.
4. **Distinguishes** the declared ring method (33.03 %) from the superseded
   traced-fiber method (6.6245 %) and from the robustness sweeps (diagnostics).
5. **Hashes** every input (nd2 source images, nucleus masks, Desmin caches, per-cell
   cache, scripts, parameter config, reproduced output) into a SHA-256 manifest.

## Scope boundary
The upstream `nd2 → Desmin channel → white top-hat → dbs` and the Cellpose-SAM
segmentation are **not** re-executed — they need `cpenv` (read-only) and a GPU. They
are pinned by content hash instead. The audit reproduces everything downstream of
segmentation.

## Run
```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m tier_a_audit.audit `
  --out model_labs/tier_a_audit/_audit
python -m pytest model_labs/tests/test_tier_a_audit.py -q --basetemp tmp/pytest_audit
```
Outputs `_audit/reproduced_visualize_final.json` and `_audit/audit_manifest.json`.

## What it does NOT do
Does not change the production method, does not declare Tier A released, does not
re-run Omnipose or start any linker/annotation round, and does not treat same-plate
fold changes as treatment effects. Orthogonal validation (confocal z-stacks / an
added marker / a Desmin-negative control well) remains required and cannot be
replaced by a 2-D Desmin review.
