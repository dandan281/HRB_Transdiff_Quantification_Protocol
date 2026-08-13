# The three candidate over-merges: extracted, and a blinded review packet built

2026-07-29. Acts on the open item from
`claude_linker_per_well_correction_2026-07-29.md` §3(c): the fragment linker at
the locked threshold **0.90** introduces **3 candidate over-merges** across the
six-well corpus, and three objects is few enough to settle by hand rather than by
argument. This report describes what was extracted, the packet built for a human
to judge it, and — importantly — **how the packet is to be read afterwards**,
written down *before* any verdict exists.

**Nothing is decided here.** No review has been run. This is the instrument.

---

## 1. What an over-merge is, in the benchmark's own terms

`precision_myotube.benchmark` builds `pred_links[j] = {i : inter/area(pred_j) >=
coverage_threshold}` with `coverage_threshold = 0.2`, and counts
`len(pred_links[j]) >= 2`. So a prediction is flagged when **two or more distinct
reviewed+`complete` reference masks each account for at least 20% of that
prediction's area**.

Note what that does *not* say: it does not say the linker was wrong. A flag is
raised equally by the linker joining two real myotubes and by the **reference set
having split one real myotube into two masks**. Which of those happened is exactly
the question a human has to answer, and it cannot be answered from the metric.

The extraction reproduces this rule directly from the label arrays rather than
through the scratch `InstanceSet`, and **asserts it recovers the published
per-well counts** — 2 in `19_B06_act104_trka`, 1 in `22_B03_act104_egfrc`. It
does. A mismatch aborts, because a packet built on a pipeline that disagrees with
the measurement it audits would be worthless.

Code: `model_labs/classical/extract_over_merges.py`.
Output: `model_labs/classical/_runs/over_merges_v1/`.

## 2. The three cases

Recorded here for the record — and these numbers are in the **key file**, not in
anything the reviewer sees.

| # | well | merged label | fragments | link P | gap µm | reference masks claimed | share of prediction | share of reference |
|---|---|---:|---|---:|---|---|---|---|
| 1 | 19_B06_act104_trka | 711 | 711 + 769 | 0.9198 | 13.1 | `myotube_0210` | 0.542 | 0.723 |
| | | | | | | `myotube_0228` | 0.524 | **1.000** |
| 2 | 19_B06_act104_trka | 1185 | 1185 + 1243 | 0.9038 | 34.1 | `myotube_0450` | 0.719 | 0.976 |
| | | | | | | `myotube_0417` | 0.295 | 0.817 |
| 3 | 22_B03_act104_egfrc | 530 | 453 + 530 + 540 | **1.0000**, **1.0000** | 28.7, 9.0 | `myotube_0126` | 0.527 | 0.308 |
| | | | | | | `myotube_0176` | 0.472 | 0.973 |

Three observations worth flagging before anyone looks at pixels:

- **Case 1 and case 3 split the prediction almost in half** (0.54/0.52 and
  0.53/0.47). Two references each claiming ~half of one prediction is the
  signature of either a genuine two-object merge or a reference mask boundary
  drawn across a continuous fibre.
- **`fraction_of_reference` is the more diagnostic column.** In cases 1 and 3 one
  reference is swallowed essentially whole (1.000, 0.973) while the other is only
  partly covered (0.723, 0.308). A prediction that entirely contains one reference
  *and* most of another is a stronger over-merge candidate than one that clips a
  neighbour.
- **Case 3 was merged at P = 1.0000, twice.** The linker was maximally confident
  on both pairs of a three-fragment chain. If a human calls this one
  `different_myotubes`, that is a more serious finding than a marginal
  0.90-something merge going wrong — it says confidence is not tracking
  correctness at the top of the range. Worth stating in advance so it is not
  rationalised afterwards.

## 3. The packet

`PrecisionMyotube/annotation_work/over_merge_r1/over_merge_review.html`
(blinded) plus `over_merge_r1.key.json` (the answers — **must not be opened by the
reviewer before they export**).

Built by `qc-review build-over-merge-page`. Five panels per object, switchable
with `1`–`5`:

1. **Desmin** — raw fluorescence, no overlay. The ground of appeal.
2. **Fragments** — the pre-merge pieces, one colour each.
3. **Proposed link** — the bridge the linker asserted, endpoint to endpoint.
4. **Linked mask** — the merged object as a single instance.
5. **Reference masks** — every reviewed reference mask intersecting the crop.

Decisions are `same_myotube` / `different_myotubes` / `ambiguous_2d`, with a free
note. The export carries `reviewer`, `session_started_at`, `exported_at`, and a
**per-case `decided_at`** timestamp; undecided cases export as
`decision: null` rather than a guess.

Brightness/contrast controls are present and are not a nicety: mean green in these
crops is **~10/255**, so judging whether stain continues across a gap is
impossible at native brightness.

### Blinding, and the three leaks that had to be closed

The reviewer is not told the well, the merged label, the link probability, or
which objects are the real flags; order is shuffled. **12 controls** — ordinary
merges accepted at the same locked threshold that do *not* trip the over-merge
rule — are interleaved with the 3 real cases, 4 per case.

Withholding the label turned out to be the easy part. Three separate leaks were
found and fixed, and each one would have made the review worthless:

1. **Fragment count.** The first control sample ran to 6- and 8-fragment
   components while every real case is 2–3. Controls are now drawn only from
   components whose fragment count matches a real case in the same well.
2. **Reference density.** Real flags sit by construction where two reference masks
   meet, which is a densely annotated neighbourhood: the flagged cases had
   **7, 7, 4** reference masks in view against the controls' **0–3** — a perfect
   ranking. Controls are now matched to each case on this count, and the count is
   no longer displayed at all.
3. **Missing pair fields on controls.** Controls were extracted without
   `gap_um` or endpoints, so the page would have shown a gap chip and drawn a
   bridge line for the real cases *and neither for the controls*. Controls now
   carry the identical pair fields.

Leaks 1–3 all passed the original "no forbidden key in the payload" check. So
that check is no longer trusted alone: `assert_no_separating_field` now runs at
build time over every displayed field and **fails the build** if a field's values
perfectly separate the two groups, *or* if a field is uniformly present in one
group and absent from the other. Leak 2 is caught by the first rule and leak 3 by
the second. The build prints the surviving ranges.

State of the shipped packet, verified after build:

| | flagged (3) | controls (12) |
|---|---|---|
| payload fields shown | `uid`, `n_fragments`, `gaps_um` | identical |
| `n_fragments` | 2, 2, 3 | 2×8, 3×4 |
| `gaps_um` (max) | 13.1, 28.7, 34.1 | 3.3 … 43.3, interleaved |
| reference masks in view (**not shown**) | 7, 7, 4 | 1 … 8 |
| bridge drawn in the link panel | yes | yes |
| all five panels present | yes | yes |

An independent scan of the built HTML finds no well name, no probability, no
reference id and no label. The only things the reviewer is given are the pictures,
the fragment count and the gap.

This is recorded at length because it is the transferable part: **a blinded packet
is not blinded because you withheld the label. It is blinded when no displayed
quantity, and no rendering difference, tracks the label.** Two of the three leaks
here were rendering differences, not data fields.

## 4. How to read the result — written before the result exists

Fixed in advance so the reading cannot drift to fit the answer.

- **Any `different_myotubes`** → a confirmed over-merge. That reinforces
  **manual-QC-only** use of the linked output. It does not by itself unwind the
  linker: the trade is still 3 candidate over-merges against 11 recovered false
  splits, but "confirmed" changes the character of the cost from hypothetical to
  real.
- **`ambiguous_2d` is unresolved.** It is **not** evidence the merge was safe, and
  must never be pooled with `same_myotube`. If a case is ambiguous in 2-D, the
  honest statement is that this benchmark cannot settle it.
- **Even if all three are `same_myotube`** — i.e. all three flags are reference-set
  artefacts, not linker errors — **the linker is not promoted.** Per the integrator,
  the independent corrected subset still declines from **20 to 18 matches and loses
  IoU** (their measurement, not reproduced here). That is a separate line of
  evidence which this review does not touch and cannot overturn.
- **The controls are the calibration.** If the reviewer calls a large share of the
  12 controls `different_myotubes`, they are not discriminating and the verdicts on
  the 3 real cases carry little weight. Report the control verdicts alongside the
  case verdicts, always. With 3 cases there is no statistical power here whatever
  the answer — this is case evidence, not a rate.

### The headline this does not change

> **11 fewer false splits, three candidate over-merges, and essentially flat
> recall (348 → 349 of 375).**

That remains the correct summary of the linker at 0.90 before and after this
review. The review can turn "three candidate over-merges" into "three confirmed"
or "three reference-set artefacts"; it cannot move the other two clauses.

## 5. Threshold discipline

**0.90 is locked.** The extractor pins it as a module constant, the packet builder
asserts the extraction's threshold matches `--expect-threshold`, and both the page
and the key carry `threshold_status: LOCKED`. **These three cases must never be
used to select or tune the threshold** — they are the objects the chosen threshold
produced, so tuning on them would be selecting an operating point on its own
errors. If the threshold ever moves, this packet is void and the extraction must be
re-run.

## 6. Also corrected in this session

The recall framing in `claude_linker_per_well_correction_2026-07-29.md` was
overstated in the other direction and has been fixed. Sparse GT makes
**precision and F1 invalid** — their denominator is all predictions, most of which
no reviewed mask could match. It does **not** do the same to **recall**, whose
denominator *is* the reviewed set; `tp/n_gt` is a valid descriptive statistic for
that reviewed, proposal-conditioned subset. What fails is the *inference*: +1
object in 375, reversing sign if one of six wells is dropped, cannot support a
recall-benefit claim. Descriptive, not promotable.

Also closed: **GT-centred neighbourhood scoring**, which I had floated as a way to
rescue precision. It would only estimate *conditional local matching performance*,
not field-level precision, because the denominator is still set by where the
annotation happened to be. The real fix is **independently sampled, fully
annotated tiles**. Not started.

## 7. Verification

```powershell
$env:PYTHONPATH = "PrecisionMyotube;annotation_tools;model_labs"
Remove-Item -Recurse -Force tmp/pytest_resume -ErrorAction SilentlyContinue
& "C:/Users/liqig/anaconda3/envs/pm-annotate/python.exe" -m pytest `
  PrecisionMyotube/tests annotation_tools/tests model_labs/tests -q --basetemp tmp/pytest_resume
```

**330 passed** (284 at the start of this assignment; +46). New tests:
`annotation_tools/tests/test_over_merge_page.py` (35 — blinding contract, both
separation rules, crop/count agreement with the renderer, panel rendering,
timestamped export), `model_labs/tests/test_extract_over_merges.py` (11 — locked
threshold, the benchmark's coverage rule reproduced including the false-split and
below-coverage-floor cases, published counts pinned to the source run).

Rebuild the packet (CPU-only):
```
python model_labs/classical/extract_over_merges.py --out model_labs/classical/_runs/over_merges_v1 --controls 60
python -m annotation_tools.qc_review.cli build-over-merge-page \
  --cases model_labs/classical/_runs/over_merges_v1 \
  --out PrecisionMyotube/annotation_work/over_merge_r1/over_merge_review.html \
  --key PrecisionMyotube/annotation_work/over_merge_r1/over_merge_r1.key.json \
  --reviewer reviewer_01 --batch-id over_merge_r1
```

All results remain exploratory, single-operator, proposal-conditioned,
retrospective development evidence — not consensus, not inter-rater agreement,
not prospective validation.
