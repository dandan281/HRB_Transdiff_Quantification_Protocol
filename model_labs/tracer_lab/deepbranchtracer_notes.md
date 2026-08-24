# DeepBranchTracer → T04: what we take, what we change, what we reject

*2026-08-23. Sources: the AAAI-24 paper (arXiv:2402.01187) and the released code
(github.com/CSDLLab/DeepBranchTracer — `train_2D.py`,
`tools/tracing/tracing_tools_2D.py`), read directly, not from the abstract.*

DeepBranchTracer (DBT) is the closest published relative of this lane: it
formulates curvilinear reconstruction as **iterative geometric attribute
estimation** — predict position, direction and radius, step along the fibre —
instead of pixel partitioning. That formulation is why the lane exists, so we
adopt it. Almost every design decision *downstream* of the formulation we
change, and each change traces to a measured property of our data or a rule of
this project. Component by component:

## 1. Formulation — ADOPT

DBT steps `c(t+1) = c(t) + r(t)·ξ(t)`: a trace is a sequence of points, not a
region. Two traces may pass through the same image location because nothing in
the output is a label map. This is exactly the property Omnipose cannot have
(one instance per pixel vs 54.5% of our traces touching a crossing), and it is
the reason a tracer is the T04 candidate at all.

## 2. Their crossing behaviour — the critical gap we must NOT copy

This is the most important finding from reading their tracing code rather than
the paper. DBT has **no junction logic**. Two mechanisms stand in for it:

- a stopping rule: *"this branch is traced"* — an R-tree collision check halts
  a trace when it comes within `node_range` of an already-traced edge;
- seed skipping: a seed within 2 px of an existing trace is discarded.

At an X-crossing, whichever fibre is traced first claims the junction; the
second fibre's trace **terminates at the crossing**. On our data that is a
false split at every contested crossing — the *same* failure mode as the label
map, produced by the tracer's bookkeeping instead of the representation. DBT
gets away with it because vessels and neurites *branch* (Y), and a tree that
stops at a junction is still a correct tree; myotubes *cross* (X), and a trace
that stops at a junction is a broken fibre.

**Our replacement, and it is the heart of the candidate:** an explicit
`crossing` head (supervised — the label a segmentation corpus must throw away,
we train on directly), and a pass-through rule at crossings: continue on the
exit whose tangent best matches the incoming tangent. Two refinements the
collision rule should keep, made angle-aware:

- **transverse** proximity to an existing trace (large crossing angle) → pass
  through, pixels shared, both identities kept;
- **co-linear** overlap with an existing trace (angle ≈ 0, sustained longer
  than ~a fibre width) → same fibre reached from another seed → merge, stop.

The angle threshold between those two cases is not free: myotubes never branch
at ~90° (`myotube-no-orthogonal-branching`), so high-angle contact is *always*
two objects. The constraint is a decision rule here, not a soft feature. Sweep
the threshold on training wells like every other knob; do not inherit one.

## 3. Direction: their LSTM vs our dense orientation field — REPLACE

DBT predicts direction **at the current tracing point only**, with an LSTM over
the last ~3 patches, as per-component classification over K bins plus a cosine
term. Notably their cosine loss is `1 − |ξ·ξ̂|/…` — the absolute value makes it
sign-invariant, which is the same insight as our angle-doubling: the clicking
direction is arbitrary. They then resolve the ± ambiguity with trace history
(the LSTM knows where it came from).

We predict a **dense per-pixel `(cos 2θ, sin 2θ)` field** and put the
history-keeping in the tracer (the walk carries the incoming tangent). Reasons,
in order of weight:

1. **Supervision density.** 5,004 instances is small. A dense field is
   supervised at every centreline pixel in every tile; an LSTM head is
   supervised only along sampled trajectories, and needs trajectory-sampling
   machinery we would have to build and debug.
2. **Auditability.** T03 counts false splits and merges. A deterministic graph
   walk over a fixed field gives reproducible counts and an inspectable failure
   at every junction. A recurrent stepping net gives neither.
3. **The oracle test requires it.** `oracle_trace.py` feeds the tracer perfect
   fields with no network. That experiment is only possible if the tracer is
   separable from the predictor — DBT's design, where direction only exists as
   a network output at a stepped point, cannot run it. (It is also this week's
   Omnipose lesson applied: validate the downstream consumer before paying for
   training.)
4. Their two-stage training (train backbone + image heads → freeze → train
   LSTM heads on trajectories) exists *because* of the recurrent heads. All-
   dense heads train in one stage with one optimizer.

What we keep from their direction design: the sign-invariance principle (ours
by construction via angle-doubling) and the idea that direction is the primary
steering signal with the centreline map as corrector (§4).

## 4. The tracing loop mechanics — ADOPT the predictor–corrector

Their best strategy (`anglecenterline`) is: step along the predicted direction,
then **snap the candidate to the nearest centreline point with ŷc > 0.5**.
Direction steers; centreline corrects drift. Also worth keeping verbatim:

- **bidirectional tracing from every seed** (`direction_state` 0/1) — seeds
  land mid-fibre, so trace both ways and join;
- **U-turn guard** — flip the predicted step if its cosine against the incoming
  tangent is negative (their `cos_sim < 0` check); with an angle-doubled field
  this is not optional, it is how the ± of the field is resolved;
- **stopping on support**, not on collision: they stop when a rolling mean (2
  steps) of an existence score drops below threshold and then prune the last 2
  nodes. Our equivalent support signal is the `centre` map value along the
  step. The prune-on-stop detail is worth copying — the last steps before a
  stop are the least trustworthy and length is our primary metric.
- step size: theirs is 2×radius. Our width barely varies (corpus
  `width_px: 8.0`, median 5.81 µm), so a **fixed step ≈ half-width (~4 px)**
  replaces the radius-scaled step.

Our formulation is a graph walk (nodes on ridge points, edges scored by
orientation continuity) rather than their sequential march; at oracle-trace
level these should behave identically on unambiguous stretches and the graph
form makes the junction decision explicit and testable.

## 5. Radius head — REJECT (unsupervisable here)

DBT regresses radius per step (MSE, λ=100) because vessel/neurite radius varies
and their step size depends on it. Our labels **contain no width**: the
operator traces centrelines, and the corpus ribbons were synthesized at a fixed
`width_px: 8.0`. A radius head would train on a constant. Fixed step, no head.
(If per-fibre width is ever wanted as a *measurement*, it comes at inference
from the image — e.g. profile fitting normal to the traced centreline — not
from a trained head.)

## 6. Boundary head — REJECT as a head, keep the function

Their boundary map's role in tracing is the stop signal. Our `centre` map
already serves that role (§4), and our corpus has no true boundary label to
train one on (see §5). Three heads stay three: `centre`, `orient`, `crossing`.

## 7. Seeds — theirs is a leak we must not reproduce

Read from `train_2D.py`: the **full-quality variant traces from gold-standard
seeds** (SWC of the ground truth, upsampled every 5 px, shuffled); only the
"fast" variant seeds from the skeletonized predicted map. So their headline
numbers are partly conditioned on ground-truth seed placement. Two
consequences:

- when (if) we ever compare numbers against DBT's published tables, that
  protocol difference must be stated;
- our candidate seeds **only from the predicted `centre` map** (ridge maxima
  above a swept threshold, consumed in confidence order). Gold seeds at eval
  would leak GT into the candidate and void the T03 comparison. Gold seeds
  *are* legitimate inside `oracle_trace.py`, whose whole point is perfect
  inputs — but then the trained candidate must be scored seed-and-all.

## 8. Zero-shot — CLOSED, and more cheaply than expected

The repo publishes **no pretrained weights at all** (README documents training
from scratch; no weight files are mentioned or linked; license is academic-use
release of code only). The prior session's note ("their published weights are
vasculature/neurites") was too generous — there is nothing to download, so the
zero-shot question for DBT is moot without training their code ourselves, at
which point we are better off training our own smaller net on a representation
built for crossings. No further effort here.

## 9. Losses and the lesson of this week — instrument first

Their training recipe (weighted BCE on 1-px centreline labels with w=0.9;
λ_radius=100 to balance magnitudes) is exactly the pattern that just cost the
Omnipose lane four rounds of diagnosis: multi-term loss, hand weights, no
per-term gradient accounting. Before the first real training run of the
three-head net:

1. run a `loss_floor`-style probe — score the ground-truth fields as if they
   were the prediction; every term must reach ~0, and per-head gradient norms
   must show every head reachable;
2. log every head separately from step 0 (lane rule, already binding);
3. our soft-Gaussian `centre` target already avoids their 1-px-binary-target
   problem (a miss by 1 px and by 50 px score the same under their label; ours
   is graded).

## 10. Metrics — theirs are diagnostics, ours are the contract

DBT reports SSD-F1, Length-F1, ABL (average branch length), PE/RE. ABL is the
same instinct as our continuity concern, but the T03 contract does not move:
`length_mdape` vs 0.3169, `false_split_count` vs 52/375, pooled recall vs
0.928, on sealed `bootstrap_v1`, pooled + object-weighted + drop-one-well.
Their metrics may be computed as internal diagnostics if useful, never as the
headline.

## Summary table

| DBT component | verdict | why |
|---|---|---|
| Iterative trace, not label map | **adopt** | the reason the lane exists |
| Collision-stop at traced regions | **reject** | = false split at every contested crossing on our data |
| (no junction logic) | **replace** | explicit `crossing` head + tangent-continuity pass-through; angle decides pass-through vs merge |
| LSTM point-wise direction, binned | **replace** | dense angle-doubled field; supervision density, auditability, oracle-testability |
| Sign-invariant cosine loss | already ours | angle-doubling is the same invariance, in the target |
| `anglecenterline` predictor–corrector | **adopt** | step on `orient`, snap to `centre` ridge |
| Bidirectional seeds, U-turn guard, prune-on-stop | **adopt** | small, correct mechanics |
| Radius head, radius-scaled step | **reject** | no width label exists; fixed step ~4 px |
| Boundary head as stop signal | **fold into `centre`** | no boundary label; centre support stops |
| Gold-seed evaluation | **reject** | GT leak; seeds from predicted map only (oracle excepted) |
| Two-stage training | **drop** | only needed for their recurrent heads |
| Pretrained weights | **moot** | none are published |
| Hand-weighted multi-term loss | **instrument first** | loss-floor probe + per-head logging before any GPU spend |

## What this changes about the next concrete step

Nothing — it confirms it. `oracle_trace.py` *is* DBT's MFT loop run on perfect
fields, with the junction rule we claim fixes their gap. If the oracle carries
identity through ≳95% of crossings and beats length_mdape 0.3169, the
representation and the walk are sound and the three-head U-Net is worth
training. If it does not, the fault is in §2/§4 above and no training run would
have told us that. Cost: CPU seconds, on Windows, offline.
