# Third-party notice — MyoFuse

`upstream/` is an **unmodified vendored copy** of a third-party repository. It is
kept for evaluation and reproducibility. Do not edit anything inside it; if we
adopt code, copy it into our own tree and record the provenance there.

## Source

| | |
|---|---|
| Repository | https://github.com/BenLair/MyoFuse |
| Pinned commit | `273a2f4dafdfc88c47eb3a63d2dbf6f2e86757a4` |
| Commit date | 2026-04-13 |
| Retrieved | 2026-07-23 |
| Files | 11 (notebooks, Svetlana config + `MyoFuse.pth` 284 KB, conda envs, docs) |

One file, `MyoFuse : How to Use.md`, contains a colon and **cannot exist on
Windows**. `git clone` therefore succeeds but its checkout fails. The file is
preserved here as `upstream/MyoFuse - How to Use.md`, extracted with
`git show 'HEAD:MyoFuse : How to Use.md'`. Content is byte-identical; only the
name differs. Anyone re-cloning on Windows will hit the same error.

## Licence — two different licences apply

**Code and models: MIT.** `Copyright (c) 2025 INSERM and Université de Toulouse`
(`upstream/LICENSE`). Permits use, modification and redistribution provided the
copyright notice and permission notice are retained. The authors additionally
state it is distributed "without any restrictions to use by non-academics".

**The article itself: CC BY-NC-ND 4.0.** This is *more* restrictive than the
code and is easy to conflate:

- **BY** — attribution required;
- **NC** — non-commercial use only;
- **ND** — **no derivatives**: we may not publish adapted versions of the
  article's text or figures.

Practical consequence: we may freely reuse and modify **their code**, but we must
not reproduce or adapt **their figures or prose** in our own write-ups. Cite and
paraphrase instead. Any numbers quoted from the paper in our documents
(accuracies, the 66-tile figure) are cited facts, which is fine.

## Attribution to carry if we adopt code

> Portions derived from MyoFuse (https://github.com/BenLair/MyoFuse), MIT
> Licence, Copyright (c) 2025 INSERM and Université de Toulouse. Lair et al.,
> Sci Rep (2026) 16:9387.

## Current adoption status

**None.** As of 2026-07-23 no MyoFuse code, weights, or configuration has been
copied into our tree. See `ASSESSMENT.md` — their trained classifier keys on a
MyHC dark-hole signal that is *inverted* in our Desmin images
(`evidence/desmin_premise_result.json`), so it is not transferable. What we take
is conceptual (workflow shape, validation design, sampling arithmetic), which
carries no licence obligation but is cited anyway.

If that changes, record it here: file copied, from which upstream path, at which
commit, and where it now lives.

## Repository hygiene

`upstream/` retains its own `.git` directory, which makes it a nested repo.
Before any commit the integrator should decide whether to keep it as-is,
`.gitignore` it, or convert it to a submodule. Nothing here has been staged or
committed — no such authorisation exists.

*Resolved 2026-08-12 (cleanup, ACTION_LOG #002): `competitors/MyoFuse/upstream/` is
now in the root `.gitignore`, so the clone stays on disk for evaluation but can never
enter this repository's history as a phantom submodule. Provenance survives via the
pinned commit above; re-obtain with
`git clone https://github.com/BenLair/MyoFuse && git checkout 273a2f4dafdf`.*

Size: `Models/Svetlana/MyoFuse.pth` is 284 KB, so vendoring is cheap either way.
