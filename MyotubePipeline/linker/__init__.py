"""Learned fragment-linker: replace merge()'s hand-tuned geometric join rules with a classifier
trained on the Q_Plates ground truth.

Auto-labels candidate fragment-endpoint pairs (join=1 if both raw Ridge fragments lie on the SAME
GT fibre) and trains a join/no-join classifier. No pipeline change and no stage5 dependency — trains
on raw stage-2 fragments (bright_segments.txt) + GT ROIs.

Standalone (base anaconda). Reuses benchmark/ for geometry/overlap and learning/ for the fit scaffold.
"""
