# Dense myotube relabelling

Draw the myotubes nothing ever proposed, and make partial coverage safe to train on.

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools"
$py = "C:\Users\liqig\anaconda3\envs\pm-annotate\python.exe"

& $py -m annotation_tools.relabel serve          # annotate at 127.0.0.1:8777
& $py -m annotation_tools.relabel stats          # what you've done so far
& $py -m annotation_tools.relabel apply --reviewer <your-name>
```

---

## Why

Measured over the six bootstrap wells:

| | share of field |
|---|--:|
| labelled as myotube (`complete`) | **1.1 – 2.6 %** |
| fibre-like according to the ridge detector | **~10 %** |

So roughly **8 % of every field looks like a myotube and carried no label**, and the
old pipeline handed those pixels to the loss as *background*. Omnipose regresses a
distance field and a flow field, and the background target is not neutral — it is
"distance 0, flow 0". Unlabelled fibres were therefore teaching the network to
suppress exactly what it is meant to detect. The `ambiguous` pool (1.7 %) was
already ignored correctly; this is the other 8 %, and it was invisible.

Triage cannot fix it, because triage only ranks an existing proposal list. The GT
is conditioned on what the classical detector proposed. Drawing is the only way out.

## How you annotate

A myotube is a ribbon — median width 5.4 µm against median length 140 µm — so the
primitive is a **centreline plus a width**, not a brush. One click per bend, then
`snap` recovers the edges from the image.

| | |
|---|---|
| add point | click |
| commit trace | <kbd>Enter</kbd> or double-click |
| cancel | <kbd>Esc</kbd> |
| width | <kbd>[</kbd> <kbd>]</kbd> |
| pan / zoom | drag with <kbd>Space</kbd> / scroll |
| fit to window | <kbd>F</kbd> |
| hide existing labels | <kbd>H</kbd> |
| select an existing instance | <kbd>Alt</kbd>+click it |
| reject the selected one | <kbd>X</kbd> |
| undo your last | <kbd>Ctrl</kbd>+<kbd>Z</kbd> |

**Every commit writes to disk immediately.** There is no save button, because a save
button is a thing you forget to press. Traces land in
`PrecisionMyotube/annotation_work/relabel/<well>/traces.jsonl`, append-only: an edit
or delete appends a record referencing the earlier one, so the full annotation
history stays auditable.

## Mask fitting

`snap` (default) grows the drawn ribbon into a search band ~2.2× wider, thresholds
it with **Otsu over the band's own intensities**, and keeps the component touching
your spine. It refuses and falls back to the plain ribbon in three cases, each of
which would otherwise corrupt `width_median_um` without any visible symptom:

- no contrast in the band — nothing to hug;
- the threshold kept nearly the whole band — it discriminated nothing, and
  accepting it would return a mask 2.2× too wide;
- the threshold erased the fibre — wrong threshold for this object.

A fixed percentile was tried first and is wrong: the band is mostly background by
design, so any fixed percentile lands in background for a thin fibre and inside the
fibre for a fat one. The unit tests pin all three fallbacks.

## `apply` — what it writes

Emits **`bootstrap_v2`** beside the sealed corpus, same layout, so
`omnipose_lab.data` and the benchmark consume it unchanged. **`bootstrap_v1` is never
modified** — every ruling, hash and T03 artifact in the project references it.

Two rules hold by construction:

- **Existing certified masks win.** Where a new trace overlaps one, the existing
  instance keeps its pixels. A relabelling pass can add supervision; it cannot
  quietly rewrite a mask a human already certified.
- **Unlabelled fibre-like territory becomes `ignore`, not background** — the fix
  above. A 6 px halo around every label deliberately stays background, because the
  fibre→background transition is exactly what the distance field must learn.

`--exhaustive` turns the ignore mask off and asserts background everywhere
unlabelled. Only correct if every fibre in every field really is drawn; it prints a
warning, because asserting that wrongly is the original bug.

## Honest limits

- **Length is the trustworthy readout.** It comes from the spine you place while
  looking at the fibre.
- **Width is a convention.** The mask edge comes from the snap threshold, so
  training on it teaches the model that heuristic and benchmarking width afterwards
  measures how faithfully it was copied. Hand-draw ~20 boundaries and compare before
  quoting a width result.
- **Benchmarking against your own annotations measures reproduction, not
  correctness.** Same operator, same session, same convention. The 40 held-out
  correction pairs remain the only non-circular reference.
- **Crossings stay ambiguous in 2-D.** A flat raster holds one identity per pixel,
  and no 2-D model can tell which fibre is on top. Overlap pixels are ignored.
- v1 and v2 instances are **not exchangeable for provenance**: v1 objects were
  proposal-conditioned, v2 additions were drawn directly. That is the point, and the
  manifest records it.
