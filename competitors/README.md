# Competitors — published tools that overlap our problem

Tracked so we can answer three questions on demand, with evidence rather than
impression:

1. **Does it already solve part of what we are building?** If yes, adopting beats
   rebuilding.
2. **Do its assumptions hold on *our* data?** Published accuracy is measured on
   the authors' stain, cells and optics. Ours differ, so every load-bearing
   assumption gets tested here before adoption.
3. **What must we still solve ourselves?** This is what our claim to novelty
   rests on, and it should be stated precisely rather than assumed.

## Layout

```
competitors/
  <Tool>/
    upstream/     unmodified vendored source, pinned to a commit — never edit
    ASSESSMENT.md what we borrow / what we reject / what stays ours
    NOTICE.md     licence, attribution, adoption log
    evidence/     runnable tests of the tool's assumptions on OUR data
```

`evidence/` is the point of the exercise. A claim that a competitor's method does
or does not transfer must be reproducible, not asserted.

## Index

| Tool | Solves | Verdict | Assessed |
|---|---|---|---|
| [MyoFuse](MyoFuse/ASSESSMENT.md) | Fusion index — per-nucleus in/out classification (Cellpose + small CNN). **Does not segment individual myotubes.** | **Partial adopt, concepts only.** Their mask-method critique applies to our conversion-efficiency rule. Their trained model does **not** transfer: it keys on an MyHC dark hole that is *inverted* in our Desmin (median inside/ring 1.60 vs the <1 they rely on). No bearing on T02. | 2026-07-23 |

## Where our work does and does not overlap the literature

**Overlaps (Tier A — field-level conversion efficiency).** Nuclei segmentation
and classifying nuclei as inside/outside myotubes are well covered: MyoFuse,
MyoFInDer, SEPO-FI, Myotube Analyzer, MyoCount, ViaFuse. We should be borrowing
here, not inventing, and we should meet the validation bar these papers set.

**Does not overlap (Tier B — individual myotube instances).** Every tool above
stops at nuclei or at a semantic myotube *mask*. None separates individual
myotubes, traces one through a crossing, bridges a broken fibre, or reports
per-myotube length, width and nuclei count. That is the T02 problem and, as far
as this review goes, it is unclaimed.

Which also means: there is no external baseline to compare T02 against. Our
honest floor has to come from our own deterministic classical candidate, which is
why its **circularity audit** matters (`model_labs/classical/README.md`).

## Adding a competitor

1. `git clone --depth 1 <url> competitors/<Tool>/upstream` and record the commit
   in `NOTICE.md`.
2. Read the licence **before** copying anything. Code licence and article licence
   often differ — MyoFuse is MIT code under a CC BY-NC-ND article.
3. Identify the single assumption the method depends on, and test it on our data
   in `evidence/`. Write down the result even when it is inconvenient.
4. Fill in `ASSESSMENT.md` with an explicit borrow / reject / ours-alone split.
