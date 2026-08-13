# Candidate prediction adapter contract

All candidate frameworks remain outside the canonical environment. They may export either:

- a 2-D mutually exclusive TIFF/NPY label image, optionally with a same-shape confidence map; or
- JSON containing `image_id`, `image_shape: [height, width]`, and `instances`.

Each JSON instance must contain an `id` plus either uncompressed COCO `rle`, one flat/Nx2
`polygon`, or a list of `polygons`. Polygon coordinates use `(x, y)` pixel coordinates. Overlapping
objects are intentionally preserved as separate RLE records.

Normalize an export with:

```powershell
python -m precision_myotube adapt-prediction --input predictions.json --format json `
  --image-id 32_C08_br223_igf1r --height 2048 --width 2048 `
  --architecture omnipose --checkpoint model.pt --environment environment.json `
  --thresholds '{"mask_threshold": 0.0}' --out candidate.instances.json
```

Every converted object is `reviewed=false`. Predicted objects can be marked `status=complete` for
benchmark matching, but they are never authoritative biological instances without expert review.
The output provenance records architecture, checkpoint SHA-256, environment, thresholds, input
format, and the review policy.
