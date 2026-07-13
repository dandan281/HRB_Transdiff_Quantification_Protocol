"""Phase-0 benchmark harness: score pipeline predictions against the Q_Plates hand-traced ground truth.

Model-agnostic. Given a well's predicted centerlines + lengths and its ground-truth ImageJ ROIs +
Results.csv, computes Tier-1 detection metrics (precision/recall/F1 + the 3 error classes) and
Tier-2 scientific-endpoint metrics (fibre count, %below/above 300 um, boundary-weighted length error).

Standalone: needs only numpy / scipy / scikit-image / roifile (base anaconda). No napari, no torch.
Run:  cd MyotubePipeline && C:/Users/liqig/anaconda3/python.exe -m benchmark --all
"""
