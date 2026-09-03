# T03 Candidate 2 pre-run authorization — 2026-09-01

Authorization time: `2026-09-01T12:37:19-07:00`

## Ruling before any Candidate 2 PLATE_23 result was observed

Candidate 2 is **authorized for one sealed run**. This is candidate **#2 of 2
submitted**, so the final comparison must disclose the multiplicity cost. No
configuration may change after this authorization, and a completed run is final.

The authorized command is exactly:

```powershell
conda run -n pm-omnipose python model_labs/tracer_lab/eval_tracer_on_bootstrap.py --candidate 2
```

The required output directory,
`model_labs/tracer_lab/_runs/eval_bootstrap_candidate2/`, was absent at
authorization. Repository and run-artifact search found no prior export carrying
version `cv_foldB02_weld_repair_v2` and no prior Candidate 2 result on PLATE_23.

## Legality check

- Junction-weld selection used PLATE_32 tune wells C02/C03/C05/C11/D02.
- Its frozen configuration was claimed once on separate PLATE_32 wells
  B02/D04/D08/D09/D11.
- Decompose-retrace identity repair used the same tune/test separation. Witness
  length 40 px was selected on the tune wells and then claimed once on the test
  wells.
- Neither selection used the sealed PLATE_23 bootstrap.
- The PLATE_23 bootstrap remains read-only. The evaluator writes only under the
  new Candidate 2 output directory.
- Seven junction-weld contract tests passed immediately before authorization.
- The `pm-omnipose` environment reports PyTorch 2.11.0+cu128, CUDA available,
  NVIDIA GeForce RTX 5070 Ti Laptop GPU.

## Frozen hashes

| Artifact | SHA-256 |
|---|---|
| `model_labs/tracer_lab/decompose_retrace.py` | `e794b52f6f850c71e59f7531d40f49c9009d4151a392949308d10e6c0c920c97` |
| `model_labs/tracer_lab/eval_tracer_on_bootstrap.py` | `44fa110d405d1d81ced66fda16df7530dd46d84a072dae2120e3ac4cfb6ba7b7` |
| `model_labs/tracer_lab/oracle_trace.py` | `cd9063095cfe596fa2da7fc8e85b0d65091c83bd88ffa22efdb5fa836f3142d7` |
| `model_labs/tracer_lab/infer_trace.py` | `e64568abfaabcf5635120c8e17a3f5ec43b19e231ac696ec25c7efcf0c4451cb` |
| fold-B02 checkpoint | `5725f8c1f61e85e74148cda3307bfda91eeb8b8f43d9f5d7ed577cd58f8c0fba` |
| sealed bootstrap manifest | `44e381142ca31ef650d202f7d2edbb53895602a13b3e4f6959437504d667de94` |
| weld tune sweep | `a312b3efb21430faadff2ebf8b05614408a638a7909c0105d278d6e1470e5f95` |
| weld PLATE_32 claim | `a54ec350c3863a4000f03e7348c6d6b7eb40344fb3457ac8ca2847640da7dff1` |
| identity-repair PLATE_32 claim | `fbecfce6b697e0ec5814677f57647768ac16af386c3e863a789b4dce21aaf948` |

## Frozen decision rule

The primary row is `nms_min50`. Metrics are ruled in this order:

1. `length_mdape` against 0.3169;
2. `false_split_count` against 52/375;
3. pooled recall against 0.928.

The result must also be compared with Candidate 1 (0.0864, 6, 0.557 in the
same order). Precision and F1 remain non-interpretable against the sparse,
proposal-conditioned certified ground truth.
