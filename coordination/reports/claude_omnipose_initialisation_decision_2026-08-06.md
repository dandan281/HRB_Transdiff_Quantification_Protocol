# Omnipose initialisation — the from-scratch vs fine-tuning question, now with measurements

Date: 2026-08-06
Lane: Claude (model laboratories)
Status: **DECIDED 2026-08-10 — option (A), fine-tune from `bact_phase_affinity`.**
Taken by the project owner before Stage 2 and before any held-out metric existed, so it
is not selection on results. Now predeclared as §5(f) of the training plan and
implemented; see §6 below for what shipped and for the one thing this memo got wrong.

This resolves the open discrepancy in §3 of `claude_omnipose_training_plan_2026-08-05.md`
("An open discrepancy worth resolving before the full run"). That section said the swap
had not been made because `nchan=1` and `diam_mean=0.0` *may* be incompatible with the
pretrained weights. Both halves of that guess have now been checked against the installed
`cellpose_omni 1.1.4` / `omnipose 1.1.4` rather than assumed.

---

## 1. What was measured

Weight-shape comparison against the architecture `train_fold.py:140-142` builds
(`nchan=1, nclasses=2, dim=2, omni=True` → 3 output channels: 2 flow + 1 distance).
Raw `state_dict` tensor shapes were compared directly, not via `load_model()`, because
its GPU branch uses `strict=True` while its CPU branch uses `strict=False` inside a
`try/except` that only prints on failure (`resnet_torch.py:341-390`).

| | `bact_phase_omni` (the model §3 named) | `bact_phase_affinity` |
|---|---|---|
| in `C2_MODEL_NAMES` → forces `nchan=2` | **yes** | no |
| in `BD_MODEL_NAMES` → forces boundary head | **yes** | no |
| resolved `nchan` | **2** (ours silently overridden from 1) | **1** |
| resolved output channels | **4** (boundary variant) | **3** |
| resolved `diam_mean` | 0.0 (survives) | 0.0 (survives) |
| tensors | 311 / 311 | 311 / 311 |
| **shape mismatches vs our net** | **12** | **0** |
| weights verified loaded into `net` | yes | yes |

## 2. What that means

**`diam_mean=0.0` was never the problem.** The bacterial branch's `self.diam_mean = 0.`
is commented out at `models.py:449` and nothing else touches it, so the value we pass
survives both paths. Half the stated concern is unfounded.

**`nchan=1` was a real problem, for `bact_phase_omni` specifically — and the failure mode
is worse than a crash.** `models.py:453-461` rewrites `nclasses` and `nchan` to fit the
checkpoint *before* the network is built, so the weights load cleanly and nothing
complains. What you get is a model that demands 2-channel input while our tiles are
1-channel Desmin, carrying the 4-channel boundary head that `train_fold.py`'s own
docstring rejects on the grounds that `omnipose.core.loss` unconditionally overwrites
`bd_loss` with the derivative loss, so the boundary output receives no gradient of its
own. A silent config override, in a harness whose last two bugs were both silent no-ops.

**`bact_phase_affinity` is a drop-in.** 311 of 311 tensors match, zero shape mismatches,
first conv `(32,1,3,3)` and output head `(3,32,1,1)` identical to the from-scratch net.
`net_type` is unchanged at `..._abstract_nclasses_2_nchan_1_dim_2`. The change is one
line: `pretrained_model=False` → `model_type="bact_phase_affinity"`.

**Unverified, and it matters:** matching shapes are not matching semantics.
`bact_phase_affinity` is the sole entry in `C1_MODELS`, commented "This will be the
affinity seg models". The architecture is identical and it is an `omni` model, so the
three channels are almost certainly the same 2 flow + 1 distance our targets supply — but
that has not been confirmed, and an affinity-trained head could carry different
conventions. This should be established before it is trusted, not after.

## 3. The options

**(A) Fine-tune from `bact_phase_affinity`.** Satisfies `DEVELOPMENT_PLAN.md` §10's
"real Omnipose transfer/fine-tuning" as written. One line. Architecture verified
identical. Cost: confirm the channel semantics first. The domain gap is real — phase
contrast bacteria to fluorescent Desmin myotubes — but early convolutional features are
generic and 375 instances is small for random initialisation.

**(B) Keep from-scratch and amend §10's wording.** Defensible on the domain gap, adds no
new risk, costs nothing, and loses no time. Requires the owner to amend the contract,
which is the owner's call, not mine.

**(C) Run initialisation as a third axis.** Doubles Stage 2 from 12 fold-trainings to 24,
roughly 40 GPU-hours on a partition where every job is preemptible. It also creates a
selection hazard, see §4.

**Recommendation: (A), conditional on the semantics check.** It is what the contract asks
for, the compatibility objection that blocked it turns out not to apply to this model,
and it is a one-line change. If the semantics check fails, (B) — and then §10 needs its
wording amended either way.

## 4. The discipline point

Whichever way this goes, **the initialisation must be chosen on contract and
compatibility grounds, not on any score.** The probe trains on a fold with
`23_B02_ctrl` held out; reading accuracy off it to pick an initialisation would be
selection on a held-out metric, which §10 forbids and which is exactly the discipline the
linker's predeclared 0.90 threshold followed. Timing, memory and "does it train at all"
are safe to read from the probe. `false_split_count` is not.

This also touches the predeclaration in §5(a) of the training plan — the T02 candidate is
the gaps-OFF arm. Changing initialisation changes what that candidate *is*, so it must be
settled and written down before Stage 2 starts. §5 may not be revised afterwards.

## 5. Operational note for klone, whichever option wins

If (A) or (C): `cache_model_path` (`models.py:59-67`) downloads the weights at model
construction time from `raw.githubusercontent.com`, 25.4 MB, into
`~/.cellpose/models` by default. Two problems on klone, both pre-empted by staging on a
login node:

- **`$HOME` is 10 GB and ~80% full** (§10 of the training plan). Set
  `CELLPOSE_LOCAL_MODELS_PATH` to a `/gscratch` path; the env var is read at import.
- **`ckpt-g2` compute nodes may have no outbound internet.** If so the download fails at
  model construction, after the job has already queued and claimed a GPU.

Download once on a login node, point the env var at it, and the training job never
touches the network.

---

## 6. Outcome, and the caveat that turned out to be real

§2 flagged that matching weight *shapes* are not matching *semantics* and said this
"should be established before it is trusted." It was, and the concern was justified.

**`bact_phase_affinity` was trained with `affinity_field`.** At `omnipose/core.py:1094`
that flag replaces the scalar field `T` with the summed connectivity graph — the code's
own comment calls it "experimenting with using the connectivity graph to define the
scalar field prediction class" — instead of the eikonal distance field the standard omni
path computes. Our training targets supply a distance field. So the third output channel
of the pretrained model means something different from what we are asking it to predict.

**It does not overturn the decision, and here is the arithmetic.** The mismatch is
confined to the slice of the output convolution that produces that one channel: 32
weights plus 1 bias, 33 parameters out of 6,610,100 — 0.0005% of the model. The encoder,
where every transferable feature lives, is unaffected, and the two flow channels mean the
same thing under either flag. Thirty-three parameters relearning over 300 epochs is not a
risk worth abandoning transfer for.

The alternative would have been partial loading of `bact_phase_omni` — skipping its 12
incompatible tensors and keeping the encoder, which has correct distance-field semantics.
That was rejected because it requires custom load code, and §3 of the training plan
already ruled that forking upstream machinery makes T03 a statement about our fork rather
than about Omnipose. The same principle applies here.

## 7. What shipped

- `train_fold.py`: `init_model` in `DEFAULTS` (default `bact_phase_affinity`), a
  `--init-model` flag accepting `scratch`, weights resolved and hashed *before* the net is
  built so a network-less compute node fails early rather than after claiming a GPU, and
  an **architecture assertion** on the resolved `(nchan, nclasses)`. That guard is the
  load-bearing part: the wrong model here does not raise, it trains the wrong network.
  Verified — it accepts `bact_phase_affinity` and `scratch`, and rejects
  `bact_phase_omni` (nchan 2, 4 channels).
- `train_manifest.json` gains `init_model` and `init_model_sha256` at top level. A
  manifest that cannot distinguish transfer from random init cannot support a T03 claim.
- `run_folds.py`: `--init-model`, `_init_model` written to every sidecar, and the resume
  check extended so a from-scratch sidecar is never replayed into a fine-tuned run.
  Sidecars predating the field have no key and are treated as stale, not as matching.
- 428 tests pass, unchanged from the baseline.

## 8. Still required on klone before Stage 2

The weights are fetched from GitHub at model construction (25.4 MB). `$HOME` is 10 GB and
~80% full, and `ckpt-g2` compute nodes may have no outbound network. Pre-stage once on a
login node and point `CELLPOSE_LOCAL_MODELS_PATH` at `/gscratch`, or the first fold of
Stage 2 dies after queueing for a GPU.
