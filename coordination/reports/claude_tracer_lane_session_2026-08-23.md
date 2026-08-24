# T04 tracer lane — session 2026-08-23

One session, all local, $0 of cluster spend. The lane went from "handoff
prompt" to: DeepBranchTracer digested, the oracle gate **passed on three
wells**, the three-head net built, probed clean, and training. This file
records what was measured; `model_labs/tracer_lab/deepbranchtracer_notes.md`
records the method mapping.

## 1. DeepBranchTracer, read in full (paper + code)

AAAI-24, arXiv:2402.01187, github.com/CSDLLab/DeepBranchTracer. Three findings
from the *code* that the paper does not advertise:

- **No junction logic.** An R-tree collision check STOPS any trace touching an
  already-traced region, and seeds near existing traces are discarded. On
  crossing myotubes that is a false split at every contested crossing — the
  exact failure this lane exists to fix. Adopt the iterative formulation,
  replace the bookkeeping.
- **No pretrained weights are published at all.** The zero-shot question for
  DBT is moot; that thread closes at zero cost.
- **Their full-quality variant traces from gold-standard seeds** (GT SWC,
  5 px upsampling). Any future comparison to their published numbers must say
  so. Our candidate seeds only from the predicted centre map.

Component verdicts (full table in the notes file): adopt the
predictor–corrector loop, bidirectional seeds, U-turn guard, prune-on-stop;
replace LSTM point-wise direction with the dense angle-doubled field; reject
the radius head (no width label exists — corpus ribbons are fixed 8 px), the
boundary head, gold seeds, and the uninstrumented multi-term loss.

## 2. Stage 0 — the oracle gate: PASSED

`model_labs/tracer_lab/oracle_trace.py`: the walk on perfect GT fields.
Seeds on centre-ridge maxima → bidirectional march (step 3 px along the
orientation field, lateral-only snap to the ridge) → crossings passed by
dead-reckoning (orientation is masked there by construction) → contact with a
claimed identity split by angle: transverse = pass through, co-linear
(sustained, <15°) = same fibre, merge. Union-find joins multi-seed fibres.

Scored against the operator's own polylines (`instance` as the point→GT
lookup; coverage counted at 1 px density):

| well | n GT | identity through crossings | length_mdape | false splits | false merges | recall |
|---|---|---|---|---|---|---|
| B02 (swept) | 356 | **0.969** | **0.081** | 12 | 31 | 1.000 |
| C05 (frozen config) | 472 | **0.957** | 0.099 | 16 | 58 | 1.000 |
| D04 (frozen config) | 626 | **0.973** | 0.066 | 19 | 84 | 1.000 |

Gate: identity ≥95% everywhere; mdape 3–5× under the classical floor 0.3169
(bootstrap_v1, different GT — comparable in intent, not a substitute).
Runtime ~6 s per well on CPU. 76 configs swept (scratchpad
`oracle_sweep*.log`); defaults in `TraceParams` are the plateau.

What the sweeps and diagnosis established, in order:

1. **Claims and coverage must be painted/counted at 1 px density.** Walk
   points are `step_px` apart; discs around them leave gaps through which a
   second seed on the same fibre escapes suppression (selftest caught a
   duplicated fibre), and per-point votes under-read coverage 4×.
2. **Rescue probe** (angle-gated cone search when support dies just after a
   crossing): false splits 28 → ~10. Ungated it bridged end-to-end fibre
   gaps — the old fragments-joined class — so it only fires within
   `rescue_window_steps` of a crossing.
3. **The follow gate** (refuse a field direction >25° axial off the incoming
   tangent): a walk exiting a crossing can land on the OTHER fibre's ridge
   with good support, and plain field-following silently adopts the wrong
   identity — 25 of 49 false-merged pairs sat at 26–87°, reachable no other
   way.
4. **In-crossing snap rejected by the selftest**: it bought 3 splits on B02
   but drags a fibre that *ends* at a junction sideways onto the transverse
   fibre. Identity errors outrank corridor splits; snap stays 0 in the mask.
5. **Residual failures are classified, not mysterious**: remaining splits
   live in near-parallel overlap corridors (one split trace had 461 crossing
   px ON it — a bundle, not an X); remaining merges are dominated by
   shallow/parallel pairs (<25°, dmin ≤5 px) that are genuinely ambiguous
   locally. Fibre-count-proportional merges on D04 (87 at 626 fibres) are the
   density tax.
6. **A CLI-default bug scored three wells on stale knobs** (argparse defaults
   overriding the swept `TraceParams`); caught because the numbers moved when
   nothing should have. CLI flags now default to None = class defaults.

## 3. Stage 1 — the net, instrumented before trained

`model_labs/tracer_lab/net.py`: U-Net depth 4, base 32, **1.93 M params**
(vs Omnipose's 6.6 M), heads centre (MSE) / orient (masked component MSE) /
crossing (BCE, pos_weight 25).

`model_labs/tracer_lab/train_tracer.py` runs the Omnipose lesson in the right
order, before any spend:

- **Loss-floor probe**: ground truth scored as the prediction. It caught a
  real one immediately: centre as BCE floors at the soft label's own entropy,
  **0.082** — a constant that would have read as a mysterious plateau in
  every curve. Centre switched to MSE; floors now 0.000 / 0.000 / 0.0004.
- **Trunk gradient shares** (measured, not assumed): centre 44% / orient 45% /
  crossing 11%. No dead heads.
- **Overfit-one-batch**: 8 tiles memorised, total 2.27 → 0.012 in 600 steps.

Training: 9 wells (B02 held out), random 384² crops (half biased to fibre
content), batch 8, AdamW 1e-3, 60×100 steps, every head logged every epoch on
FIXED train and held-out batches (`_runs/net_v1/log.jsonl`), local RTX 5070 Ti.

## 4. Stage 3 scaffold

`model_labs/tracer_lab/infer_trace.py`: sliding-window field prediction →
the SAME walk, unchanged → the same scorer, plus field-quality diagnostics
(centre on/off ridge, crossing recall at threshold) so a walk failure on
predicted fields is attributable to fields vs walk. `orient_valid` at
inference = predicted support AND not predicted crossing.

## 5. Rules compliance

- Sealed artifacts untouched; `bootstrap_v1` not consumed (the sealed T03 run
  comes later, via `eval_on_bootstrap`, after threshold freezing on training
  wells only).
- No numbers scored into a ruling; Codex owns T03.
- Oracle knobs swept full-range and frozen; the predicted-field thresholds
  (`--crossing-thresh`, `--valid-thresh`, seed) are NOT yet swept — that
  sweep happens on training wells, never on B02.
- Omnipose lane untouched (other session owns it). The local fixed-batch
  replication logs from earlier today are in this session's scratchpad.

## 5b. First training run (net_v1): fields learned, centre too weak to walk on

60 epochs, best held-out total 0.2137 (~epoch 30; clear overfit after — train
kept falling while held-out rose to 0.315 by epoch 59). Field quality on D04
with `best.pt`, measured head by head:

- `orient`: **good** — 7.2° median axial error on the ridge (p90 20.9°).
- `centre`: **not walkable** — on-ridge median 0.235 vs off-ridge p99 0.347;
  the distributions overlap and the first threshold sweep found literally
  zero traces at every config (the whole grid sat above the whole field).
- `crossing`: FP-heavy — off-crossing p99 0.791 vs on-crossing median 0.676;
  pos_weight 25 bought recall at hopeless precision.

Diagnosis: plain MSE let the ~94% background dominate the centre head.
Fix (net_v2 retrain): ridge-weighted centre MSE (`1 + 10·target`; floor still
exactly 0 — re-probed: gradient shares moved to centre 79% / orient 19% /
crossing 3%), crossing pos_weight 25 → 8, 40 epochs. Both changes at once,
recorded as such. The sweep grid was also extended downward and `support`
added to it — the walk's thresholds are in the field's own brightness units,
a fact the oracle's perfect-brightness fields had hidden.

## 5c. Second training run (net_v2) and the walk on predicted fields

40 epochs with the reweighted loss, best held-out total 0.2352. Fields on D04
with `best.pt`: centre on-ridge median **0.235 → 0.570** (off-ridge p99
0.648 — better, still not oracle-clean), orient unchanged good (6.9°),
crossing now under-confident (on-crossing p50 0.386) after the pos_weight
retreat. A `max_seed_candidates` cap (500k, brightest first) was added to the
walk after the first predicted-field sweep attempt: a low seed threshold on a
dim field admits millions of halo pixels and turns the per-seed claim check
into the entire runtime — a failure mode perfect fields can never exhibit.

## 5d. Where the candidate actually stands (end of session)

Predicted-field walks on D04 (a TRAINING well — held-out numbers would be
worse), best configs per regime:

| field prep | identity_x | mdape | splits | merges | recall | diagnosis |
|---|---|---|---|---|---|---|
| oracle (perfect fields) | 0.973 | 0.066 | 19 | 84 | 1.000 | the ceiling |
| raw v2 fields | 0.762 | 45.2 | 202 | 111 | 0.979 | walks roam the halo: ridge p50 0.57 vs halo p99 0.65 overlap, no absolute threshold separates them |
| NMS v2 fields | 0.498 | 1.77 | 273 | 175 | 0.919 | crest is discontinuous; walks die at prediction dips and fragment |

**Verdict: the walk is proven and the fields are not yet walkable.** The
trained-to-oracle gap is the entire problem, and it is a field-quality
problem, not a threshold problem — both sweeps' axes went inert once the
regime was set. B02 (held out) was deliberately NOT consumed: no config
earned freezing, and the first B02 run should spend its one clean shot on a
candidate that passed the training wells.

What the next iteration should try, in cheapest-decisive-first order:

1. **Gap-tolerant walking on NMS fields**: the rescue probe is currently
   gated to crossings; on NMS crest gaps it is the exact mechanism needed.
   Un-gate it for NMS fields (or gate on "crest ended but orientation still
   confident") and re-measure — could recover much of the 273-split
   fragmentation for free.
2. **Train the centre head to be sharper, not just brighter**: higher
   ridge weight, more epochs at lower lr, and augmentation (flips/rotations
   are exact symmetries of every target). The overfit gap (train 0.19 vs
   held 0.10 centre at ep39) says capacity is not the constraint yet.
3. **Deep supervision of the crest**: an auxiliary loss on the NMS-style
   property itself (lateral profile peakedness) rather than pixel MSE.
4. If fields stay fuzzy: the walk could consume the orientation field as the
   primary signal (it is already good at 6.9°) with centre only as a weak
   prior — a different tracer, same representation.

## 5e. Four hypotheses tested on the fuzzy-ridge failure; three refuted

The candidate's single blocking defect is measurable in one number: the
**perpendicular FWHM of the predicted centre map, 12 px against a 4 px
target**, contrast crest-to-8-px-off only 1.2x. Everything downstream (walks
wandering inside the band, duplicate paths, mega-merges) follows from it. Four
hypotheses, each with its own run and its own measurement:

| # | hypothesis | change | result | verdict |
|---|---|---|---|---|
| 1 | class imbalance starves the ridge | ridge-weighted MSE (net_v2) | ridge brighter (p50 0.24 -> 0.57), FWHM unchanged | partial |
| 2 | overfitting / too little data variety | dihedral augmentation, lower lr, 80 ep (net_v3) | no divergence at all, FWHM **12 px** | **refuted** |
| 3 | the representation is wrong: a peaked target cannot be regressed | offset-to-centreline vector head (net_v4) | head collapsed to a near-constant 1.9 px magnitude (should be 0 on the ridge, 6 at 6 px out); FWHM 11 px | **refuted** |
| 4 | MSE rewards hedging; need a scale-sensitive loss | + soft Dice on centre (net_v5) | contrast 1.2x -> 1.4x, FWHM **12 px** | **refuted** |

An analytic route was also tried and failed: a steerable ridge filter (second
derivative along the fibre normal, steered by the orientation head, which is
good at 7 deg) returned FWHM 13 px. A 12 px bump with 0.08 of contrast has no
crest to recover -- the information is not there to sharpen.

**What finally located the fault: an 8-tile memorisation test.** Given data it
can memorise, the same architecture reproduces the ridge at **FWHM 7 px
against a 6 px target** and drives the centre term to 0.004. So capacity and
loss can both express a sharp ridge. Re-reading the logs with that in hand,
every run had **held-out loss BELOW training loss** (net_v5: 0.53 vs 0.71 on
its own training batch) -- the signature of **underfitting**, not
overfitting. 80 epochs x 100 steps was simply not enough optimisation, and
three of the four "fixes" above were treating a symptom of that.

net_v6 is therefore 300 epochs with cosine decay (1e-3 -> 2e-5) and no other
change. If the ridge sharpens, the earlier representation and loss changes
should be **ablated** before any of them is credited.

## 5f. End-to-end benchmark of every version (D04, a TRAINING well)

Same walk, same well, one row per network. `raw` = predicted centre used
directly; `nms` = non-max suppressed first.

| version | prep | identity | mdape | splits | merges | recall | total mm (GT 167.2) |
|---|---|---|---|---|---|---|---|
| ORACLE | - | 0.975 | 0.085 | 19 | 82 | 1.000 | 158.4 |
| net_v1 | raw | 0.015 | 0.342 | 4 | 4 | 0.021 | 13.0 |
| net_v2 | raw | **0.776** | 22.55 | 154 | 91 | **0.949** | 364.8 |
| net_v2 | nms | 0.440 | 0.576 | 229 | 146 | 0.823 | 227.7 |
| net_v3 | raw | 0.734 | 65.74 | 119 | 90 | 0.877 | 337.3 |
| net_v4 | raw | 0.694 | 30.19 | 122 | 87 | 0.869 | 326.3 |
| net_v5 | nms | 0.277 | **0.316** | 155 | 80 | 0.508 | 138.2 |
| net_v6* | raw | 0.635 | 2.69 | 110 | 122 | 0.789 | 229.9 |

*net_v6 was stopped at ~epoch 200 of 300 (flat since 50) and is a
mid-training snapshot.

**The structure of the failure, stated properly**: coverage and cleanliness
trade off and are never both present. `raw` fields give recall 0.95 and
identity 0.78 with lengths 2.2x too long -- the walk re-walks each fibre
several times inside the fuzzy band (visible directly in
`_runs/version_comparison_D04.png`). `nms` fields give near-floor mdape with
half the fibres missing. A sharp ridge is what would allow one configuration
to have both, which is why the ridge width is the whole problem.

Also tested and **refuted**: that the 2.2x over-length was double counting in
the scorer (an object's length is the sum of its member paths, so a
twice-walked fibre reports twice). Recomputing length as the union of covered
pixels moves net_v2/raw 22.55 -> 21.91 and net_v6/raw 2.69 -> 2.59, i.e. ~3%.
The over-length is real geometry. The metric default was left unchanged
rather than redefined for a 3% effect.

## 5g. The annotation-alignment hypothesis: RAISED, then RETRACTED

A first measurement (perpendicular cut, brightest intensity peak within
+-8 px) put the operator's line **SD 3.3-4.3 px from the image ridge**, which
would have explained ~half the 12 px blur as irreducible target noise and
would have redirected the whole lane toward target snapping. It was reported
as a finding. **It does not survive validation and is withdrawn.**

What killed it: the claim needs a definition of "the image ridge" that does
not move when the method moves, and two attempts failed.

1. A snap using the *nearest* local max appeared to pass its pre-flight
   (offset SD "0.39 -> 0.44 px", identity theft 0.65%) -- but that run had
   changed the MEASUREMENT to nearest-peak as well, which is ~0 by
   construction. The snap moved points a median of 0.09 px: a no-op wearing a
   redefined metric. The *brightest* variant did move points (median 1.7 px)
   but stole 2.4% of them onto neighbouring traces, failing its own bar.
2. A Steger/Hessian crest map was built as a fixed yardstick, with the right
   control: compare the operator's line against a copy shifted 4 px sideways.
   **Separation was only ~0.9 px at every scale tested (sigma 1.5-5.0)**, with
   absolute median distances of 4-10 px. A yardstick that barely tells the
   operator's tracing apart from a 4 px-displaced copy cannot measure
   annotation accuracy, so no number from it is usable in either direction.

The honest state: **it is unknown** how well the operator's centrelines align
with the image's own ridges, and therefore unknown how much of the 12 px blur
is learnable. Nothing in this section licenses either "the annotation is the
ceiling" or "the annotation is fine".

**How to settle it next time, and the discipline that was missing**: validate
the yardstick on SYNTHETIC fibres first -- render ribbons of known width and
noise around known centrelines, confirm the ridge detector puts them within
<1 px and that a 4 px-shifted control lands ~4 px away, and only then point it
at real data. Building a measuring instrument and its subject at the same time
is what produced two unusable answers here.

## 6. Open items, in order

1. Field-quality iteration (§5d list) until a config passes the training
   wells; only then spend B02, frozen thresholds, one run.
2. The bundle corridors (residual oracle splits) and shallow-parallel merges
   are the known ceiling costs; if they matter at T03 scale, the fix is
   global (graph assignment over the whole field), not more local rules.
3. PLATE_23 / bootstrap_v1 scoring path for polyline candidates (trace →
   instance masks at proposal sites) — needed before any T03 number exists.
4. Nothing committed yet; ask the operator.

## 7. Artifacts

| path | what |
|---|---|
| `model_labs/tracer_lab/deepbranchtracer_notes.md` | DBT method mapping |
| `model_labs/tracer_lab/oracle_trace.py` | the walk + oracle gate + selftest |
| `model_labs/tracer_lab/net.py` | 1.93 M three-head U-Net + instrumented loss |
| `model_labs/tracer_lab/train_tracer.py` | probe / overfit / train, per-head logs |
| `model_labs/tracer_lab/infer_trace.py` | field prediction + NMS + walk + scoring |
| `model_labs/tracer_lab/sweep_infer.py` | threshold sweep (training wells only) |
| `model_labs/tests/test_oracle_trace.py` | 4 walk-contract tests (15/15 green with targets) |
| `model_labs/tracer_lab/_runs/oracle/` | oracle JSONs, B02/C05/D04 |
| `model_labs/tracer_lab/_runs/net_v1/`, `_runs/net_v2/` | checkpoints, logs, manifests, sweeps |
