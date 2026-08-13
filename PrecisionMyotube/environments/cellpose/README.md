# Validated Cellpose environment

This folder is the P0.4 fingerprint for the existing
`Conversion_Efficiency/cpenv` environment. `fingerprint.json` records Python, operating system,
Torch/CUDA visibility, a stable environment hash, and the C08 validation check.
`requirements.freeze.txt` is the exact installed package snapshot.

Regenerate from the repository root with the validated interpreter:

```powershell
$env:PYTHONPATH = "PrecisionMyotube"
& "Conversion_Efficiency\cpenv\Scripts\python.exe" -m precision_myotube `
  fingerprint-environment `
  --out "PrecisionMyotube\environments\cellpose" `
  --label "validated-cellpose-c08" `
  --validation-summary "PrecisionMyotube\runs\32_C08_smoke\analysis_summary.json" `
  --expected-total-nuclei 10114
```

The command fails if the validation summary no longer reports exactly 10,114 valid nuclei.
