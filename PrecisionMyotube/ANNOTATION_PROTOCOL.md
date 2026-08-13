# Independent-myotube annotation protocol

This protocol defines the biological ground truth. Model predictions and old centerline ROIs are
prompts only.

## Identity

- Draw the full visible Desmin-positive body of one independently traceable myotube.
- Separate touching structures only when a visible boundary, gap, independent endpoint, or clear
  continuation supports the decision.
- Mark a contact `ambiguous` when the 2-D pixels do not contain enough evidence. Never infer a
  boundary solely to increase the object count.
- Use separate masks for independently visible crossing myotubes; overlapping mask pixels are valid.
- Mark instances `complete`, `border_truncated`, `occluded`, or `ambiguous`.
- Mark `reviewed=true` only after checking the raw fiber and DAPI channels at native resolution.

## Status definitions

Assign one status to every proposed instance. Status describes whether the object's identity and
full measurable extent are supported by the pixels; it does not describe whether the mask looks
generally plausible.

### `complete`

Use `complete` only when all of the following are true:

- one independently traceable myotube can be followed without guessing its connectivity;
- its full visible body and both observable ends are contained within the image;
- the boundary needed for area and width measurement is visible around the entire object;
- no hidden or unresolved segment could materially change its length, width, or nucleus assignment;
- the mask has been checked at native resolution.

A touching or crossing myotube can still be `complete` when visible evidence supports its
continuation and separate overlapping masks can be drawn confidently. `Complete` does not mean
"probably complete" or "the model produced a clean mask."

### `border_truncated`

Use `border_truncated` when the myotube reaches an outer image boundary and appears to continue
beyond the captured field. The visible portion may be perfectly clear, but the image does not
contain the full object.

- Any mask touching the outer image border is treated as `border_truncated` by the canonical
  analysis, even if it was accidentally labeled `complete`.
- Do not invent an endpoint at the image edge.
- Do not use the object for authoritative length, width, or nuclei-per-myotube measurements because
  an unknown portion lies outside the image.

Contact with an internal crop, tile seam, or missing-data boundary should also be recorded as
truncated and documented in the notes.

### `occluded`

Use `occluded` when the myotube's identity is still sufficiently clear, but part of its body or
boundary is hidden inside the field by another structure, saturated signal, debris, a dense
overlap, or another imaging obstruction.

- The annotator must be able to identify the same myotube on both sides of the hidden region without
  choosing among multiple plausible continuations.
- Draw only the supported visible area; do not interpolate a hidden boundary as ground truth.
- Use `occluded` even when both endpoints lie inside the field if the missing region could bias
  length, width, or nucleus assignment.

If more than one continuation is plausible, use `ambiguous`, not `occluded`.

### `ambiguous`

Use `ambiguous` when the available 2-D pixels do not support one defensible identity, connectivity,
or boundary assignment.

Examples include:

- a contact where two or more split/merge interpretations are plausible;
- a crossing where incoming and outgoing branches cannot be paired confidently;
- dense Desmin-positive territory that cannot be partitioned into independent cells;
- a faint or fragmented structure that may be one myotube or several objects;
- uncertainty about which myotube owns a shared region or nucleus.

Preserve the visible semantic territory, record the uncertainty, and do not force a split or merge.
Ambiguous territory may contribute to field-level conversion efficiency but not authoritative
single-myotube measurements.

## Status decision order

1. Does the object leave the captured image? Use `border_truncated`.
2. If it remains inside the image, is its identity or connectivity uncertain? Use `ambiguous`.
3. If identity is clear, is a material part of its body or boundary hidden? Use `occluded`.
4. Only if none of the above applies and the full object is measurable, use `complete`.

When two limitations apply, use the status that best explains why identity or measurement fails
and record the other limitation in `notes`. In particular, unresolved identity should remain
`ambiguous` rather than being softened to `occluded`.

## Review

- This project currently has one human operator. `reviewed=true` means that operator inspected the
  raw fiber and DAPI channels; it does not mean independent consensus.
- Measure test-retest consistency with a blinded 30-case repeatability sample. Hide prior decisions
  and model defaults during the repeat session.
- If the two sessions disagree, exclude the case from training or retain it as `ambiguous`; do not
  resolve disagreement by choosing the more convenient answer.
- Record `review_mode=single_operator`, a stable reviewer ID, timestamp, tool version, and decision
  hash in each frozen annotation snapshot.
- Plate 26 is not locked: its labels are already committed and prior code has processed it. Treat it
  as retrospective only. A new plate held outside the workspace is required for prospective testing.

## Measurement eligibility

- Only `complete` and `reviewed` instances enter authoritative length, width, and nuclei-per-myotube
  statistics.
- Border-truncated, occluded, ambiguous, disconnected, or unreviewed objects remain visible in QC
  outputs and are counted separately.
- Ambiguous contact territory may still contribute to field-level conversion efficiency.
