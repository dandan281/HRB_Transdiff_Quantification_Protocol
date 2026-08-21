# Why the dense-corpus Omnipose fine-tune produces nothing — diagnosis session

Date: 2026-08-20
Lane: Claude (model laboratories)
Status: **failure localised to the loss function; the fix is untested**

---

## 1. One line

The 300-epoch run on `plate32_dense_v1` completed and produces zero masks; this
session eliminated every explanation except one and named it — `omnipose.core.loss`
divides each of its nine terms by that term's own magnitude before backprop, so
no term can ever lose weight and the objective never settles.

Nothing about the corpus, the target, the links, the initialisation or the
evaluation path is wrong. That is the substantive result, and it was expensive to
establish only in the sense that each check had to be built.

## 2. What was measured, in the order it was measured

| check | question | answer |
|---|---|---|
| LR sweep (job 250696) | is `lr=0.1` too large? | **no** — 0.1 and 0.01 plateau at the same value |
| `diagnose_alignment.py` | do the images predict the labels? | **yes** — AUC 0.800 vs 0.50 controls, offset (0,0) |
| `overfit_one_tile.py` | can the net memorise 4 tiles? | **no** — and the *untrained init* is good |
| `omnipose/core.py` source | what does the loss compute? | 9 terms, dynamically renormalised |
| `trace_loss.py` | what does the training step deliver? | the correct 7-channel target |

### 2.1 The learning rate is not the cause

Two rates a decade apart land on the same epoch loss (~3.85 on the 144-tile
fold), flat from epoch 5. Rates 0.001 and 0.0001 were queued in the same job but
their blocks were never read back — **unfinished, and no longer interesting**,
because the overfit test that followed is strictly more informative.

### 2.2 The corpus and the labels are sound

`diagnose_alignment.py`, six tiles of the fold as `build_dense_fold` hands them
to `model.train()`:

- labelled pixels average intensity **0.18–0.31**, unlabelled **0.039–0.048**
- AUC of intensity as a classifier of "labelled" = **0.791–0.848**, median 0.800
- null controls — the mask rolled by (211, 173), and another tile's image —
  **0.487–0.537**, i.e. exactly chance
- image↔mask cross-correlation peak at **(0, 0)** on every tile, z ≈ 25–29

So the fibres are where the labels say they are, at zero offset, with a strong
intensity separation. No shift, no transpose, no misalignment.

### 2.3 The initialisation is *good*, and fine-tuning destroys it

This is the finding with the widest consequences. Before a single gradient step,
`bact_phase_affinity` at native scale with no rescaling gives:

| tile | n_pred | n_gt | distance range | predicted fg | true fg |
|---|---|---|---|---|---|
| 0 | 297 | 183 | −5.02 … +16.21 | 8.02% | 7.27% |
| 1 | 275 | 208 | −5.02 … +14.96 | 7.78% | 6.29% |
| 2 | 183 | 118 | −5.02 … +13.61 | 4.83% | 5.47% |
| 3 | 131 | 65 | −5.01 … +13.24 | 3.73% | 5.33% |

Territory coverage lands within ~1–2 points of truth on every tile and the
distance field is well-formed. **Weights load, the architecture is right, and the
inference path works** — three explanations closed at once.

Note the tension with the 2026-08-17 zero-shot result (aspect ratio 1.8–2.7
against a true 38). That was measured on 1024 px crops and scored on instance
*shape*; this is measured on 1280 px tiles and scored on *coverage*. They are not
in contradiction — coverage says nothing about whether the objects are the right
objects — but the zero-shot negative deserves re-reading in light of it, and the
scale sweep it still owes (`--diameters`) is now more interesting, not less.

Then 400 epochs on four fixed tiles move the raw loss from 10.40 to ~10.15. Four
images cannot defeat 6.6M parameters. The "AFTER training" lines of that run were
never read back — **owed, and cheap**.

### 2.4 The target the training step delivers is correct

`trace_loss.py` wraps `omnipose.core.loss` and reports the tensor as it arrives:

```
lbl (4, 7, 384, 384)   y (4, 3, 384, 384)
 ch0 masks             [  +0.000, +694.000]  nonzero   5.31%
 ch1 thresholded mask  [  +0.000,   +1.000]  nonzero   5.31%
 ch2 boundary field    [  +0.000,   +1.000]  nonzero   2.62%
 ch3 smooth distance   [  -5.000,   +7.587]  nonzero 100.00%
 ch4 weights           [  +0.500,   +1.000]  nonzero 100.00%
 ch5 flow[0]           [  -5.123,   +5.112]  nonzero   5.27%
 ch6 flow[1]           [  -5.134,   +5.181]  nonzero   5.28%
```

Seven of seven, in the documented order, background distance at −5, flows at 5×.
**The links do not corrupt it.** Every remaining data-side hypothesis is closed.

### 2.5 What is left: the loss renormalises itself

```python
raw_loss = sum(losses).detach() / len(losses)
losses = [scale_to_tenths(l, max_gain=1e12) for l in losses]
return sum(losses), raw_loss
```

`scale_to_tenths(x)` multiplies `x` by `10**(-floor(log10(|x|))-1)` — it returns
the mantissa. Applied per term, with the gain cap set to 10¹² (effectively
absent), it means **any term that improves is amplified straight back to O(1)**.
The nine terms therefore hold fixed relative weight in the gradient no matter
what the network does, and there is no configuration the optimiser can move
toward. A raw loss pinned at 10 is what that looks like.

Supporting evidence that this is development code rather than a stable release:
two competing definitions of `scale_to_tenths` in the same file (the second
shadows the first), `bd_loss` computed as a BCE and then unconditionally
overwritten by a `DerivativeLoss` twelve lines later, large commented-out blocks
(`if 0:`), and the author's own note that `lossB` "is particularly instable
maybe, need to plot all these".

**This is a hypothesis with a named mechanism, not a proven cause.** The
experiment that tests it is written and not yet run (§4).

## 3. Mistakes

**Mine, in the order they cost time:**

1. **Built the LR sweep first.** It was derived from reading the flat epoch loss
   as "not learning", which was correct but under-determined — the same evidence
   was equally consistent with four other causes. The overfit test costs $0.45
   and 5 minutes and would have closed three of them on the first try. *Cheapest
   decisive test first, and "decisive" means it splits the hypothesis space, not
   that it tests my favourite branch.*
2. **Read the logged loss wrong, then corrected it wrong.** First I treated it as
   an absolute loss; then, on reading `scale_to_tenths`, I told the operator the
   logged number was the renormalised sum and therefore uninformative. It is
   `raw_loss` — step 1 logs 10.401753 and `raw_loss` is 10.401753. The original
   reading was right and the correction was wrong. *Root cause both times: I
   reasoned about a number instead of reading the function that produces it. The
   thing that finally worked was `inspect.getsource`.*
3. **Concluded "the target is scrambled" from my own harness.** `diagnose_target.py`
   and `diagnose_loss.py` both call `labels_to_flows` directly, which returns a
   5-channel stack; the loss documents 7. I wrote that comparison up as a probable
   root cause. Training does not use that path — it computes flows inside the step
   — and delivers the correct 7. *A diagnostic that does not go through the
   production path is evidence about the diagnostic.*
4. **`diagnose_loss.py` passed numpy where a device tensor was required**, wasting
   a GPU allocation ($0.23), and the fixed copy was never transferred, so the
   rerun tested the stale file and produced the identical traceback.
5. **Put the `scp` in the middle of a block of Tillicum commands**, so it was run
   in the wrong shell — twice. Local-vs-remote needs to be a heading, not a line.

**Structural, not personal:** four diagnostics in a row were needed because
`train_fold.py` records no per-epoch loss and no per-term breakdown. The third
time a missing artifact has hidden a failure on this project.

## 4. The untested fix, and how to test it

`trace_loss.py --no-rescale` neutralises `scale_to_tenths` by monkey-patching it
for the duration of a short `train()`, and the same wrapper records all nine
terms individually (it is called once per term, in source order). Written,
committed, **not yet run**:

```bash
srun --account=hrbomics --partition=gpu-h200 --gpus=h200:1 -c 4 --mem=32G --time=20:00 \
  bash -c "$PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.01 2>&1 | grep -v 'Train epoch\|models/trace'; \
           echo '=== NO RESCALE, lr 0.01 ==='; \
           $PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.01 --no-rescale 2>&1 | grep -v 'Train epoch\|models/trace'; \
           echo '=== NO RESCALE, lr 0.0001 ==='; \
           $PY model_labs/omnipose_lab/trace_loss.py --epochs 40 --lr 0.0001 --no-rescale 2>&1 | grep -v 'Train epoch\|models/trace'"
```

Removing the per-term normalisation changes the gradient magnitude entirely, so a
much smaller step size is tested alongside.

Outcomes and what each implies:

- **raw loss falls with `--no-rescale`** → cause confirmed. Fix is either pinning
  a stable `omnipose`/`cellpose_omni` release whose `loss` is the published one,
  or carrying a documented patch in `train_fold.py`. Pinning is preferable: a
  monkey-patch in the training path is a provenance liability for a T03 claim.
- **raw loss flat in all three** → the cause is one specific term; the per-term
  table names it. `AffinityLoss` is the first place to look, since it consumes
  flows, distance and the link structure together.
- Version identification is owed either way:
  `python -c "import cellpose_omni, omnipose; print(cellpose_omni.version, omnipose.__version__)"`

## 5. Commits

| sha | what |
|---|---|
| `5da601c` | LR sweep (superseded by §2.1 — keep for the record) |
| `10384b2` | `diagnose_alignment.py`, `overfit_one_tile.py` |
| `e16732e` | `diagnose_loss.py` — loss-floor probe |
| `918c3fd` | tensor fix for the above |
| `43f75ea` | `trace_loss.py` — live training step |
| `e057fbd` | per-term breakdown + `--no-rescale` |

## 6. What this does and does not change for T02

**Does not change:** no held-out metric exists. `length_mdape` against the
classical floor of 0.3169 is still unmeasured, `false_split_count` against 52 is
still unmeasured, and no claim about the candidate is available in either
direction.

**Does change:** the candidate's *data* is no longer under suspicion. The corpus,
the tiling, the links, the ignore policy and the target construction have each
been measured directly rather than argued about. If the loss hypothesis holds,
the run that produces a metric is a rerun of the existing pipeline with one
dependency change — not a rebuild.

**Worth flagging to the integrator:** if the fix requires patching or pinning the
segmentation library, that is a change to what the candidate *is*, and belongs in
the binding-settings table alongside `init_model` and `nclasses` rather than in a
script comment.
