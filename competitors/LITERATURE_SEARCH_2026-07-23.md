# Literature search — external benchmarks & annotator variability (2026-07-23)

**Why:** the operator asked whether observed labelling disagreement is *machine*
error or *their* error. Answering needs (A) external ground truth not conditioned
on our own proposals, and (B) a published reference for how much expert humans
disagree with each other on this task.

> **RUN STATUS — INCOMPLETE.** The multi-agent search hit the account session
> limit partway through: **48 of 100 agents completed, 52 failed**, and the final
> synthesis step never ran. Claims below are therefore tagged by how much
> verification each actually received. Several of the most important claims
> (NCL-SM, the arXiv myotube paper) reached **zero** completed verification votes
> because the voters died on the limit, not because they were doubted. **Re-run
> before citing any UNVERIFIED row externally.**

---

## Headline answers

**A — Instance-level, in-vitro myotube ground truth appears not to exist
publicly.** Every named tool stops at nuclei or at a *semantic* myotube mask.
This matches what the project already believed and makes our T02 set, as far as
this search goes, without a public equivalent. It also means **there is no
external baseline to score T02 against** — which is exactly why the circularity
audit on our own classical floor carries the weight it does.

**B — The best expert-vs-expert reference found is IoU ≈ 0.96**, from NCL-SM, on
*histology cross-sections* (round fibres — an easier delineation problem than our
long, crossing, broken fibres). For in-vitro work the only muscle-specific figure
is ICC > 0.75 across 5 of 6 parameters from **two** raters (Myotube Analyzer).

**What that says about the operator's own numbers.** Their blind-repeat mask
agreement was **median IoU 1.0** on complete/complete pairs — at or above the
published expert ceiling. Their 90% figure is on the harder *categorical*
disposition (complete / border / ambiguous / reject), which is a different and
noisier judgement than tracing a boundary. On the evidence available, their
boundary work is not the weak link.

---

## A — Datasets

| Dataset | Instance-level myotubes? | In vitro? | Downloadable | Confidence |
|---|---|---|---|---|
| **MyoFuse** (Zenodo `10.5281/zenodo.14731491`) | **No** — per-nucleus class labels only, no myotube masks of any kind | yes (C2C12 + human primary, MyHC) | **Yes**, 39.8 MB zip, CC-BY-4.0, deposited 2025-02-13 | record live: **CONFIRMED 3-0**; contents (20 × 1000² patches + Svetlana labels): **UNVERIFIED** |
| **NCL-SM** (`10.25405/data.ncl.24125391.v2`) | **Yes** — 50,434 individually delineated myofibres over 46 sections; `Mask_All_AM`, `Mask_AM_vs_NAM` | **No — histology cross-sections** (biopsy, IMC + IF) | claimed 5.9 GB, CC BY 4.0, no access request | **UNVERIFIED** (all 3 voters died on limit) |
| **Myotube Analyzer** (Noë 2022) | mask *format* is instance-level (one colour per myotube in PNG) — but **masks were never published** | yes | tool only; data availability disputed in-run | format: **CONFIRMED 3-0**; "no public data": **REFUTED 1-2** — recheck |
| **MyoFInDer** (Weisrock 2024) | **No** — binary semantic mask by threshold + blur + morphology; touching myotubes merge into one component | yes | code only (GPL-3.0); no data DOI | semantic-only: **CONFIRMED 2-0**; "ships no data": **REFUTED 0-3** — conflicting, recheck |
| **arXiv 2604.14720** (3D myotube synthesis) | authors state **no large annotated myotube dataset existed publicly**; their own real set = 17 volumes / 40 manual instances, no inter-annotator agreement, only the *synthesis pipeline* released | yes (3D) | pipeline only | **UNVERIFIED** — but consistent with the project's prior note |
| SEPO-FI, MyoCount, ViaFuse | not reached before the limit | — | — | **NOT SEARCHED** |
| IDR / BBBC / EMPIAR / BioStudies / Kaggle sweep | agent ran; results not carried into synthesis | — | — | **INCOMPLETE** |

## B — Annotator variability

| Source | Figure | Task | Confidence |
|---|---|---|---|
| **NCL-SM** | **mean IoU 0.96** between independent expert re-segmentations (53 IMC + 23 IF myofibres; rAoB 0.99, rAiB 0.77) | fibre boundary delineation, **histology cross-section** | **UNVERIFIED** |
| **Myotube Analyzer** | **ICC > 0.75** on 5 of 6 parameters, some **> 0.9**; **only 2 raters**, 19 image sets | in-vitro derived parameters (fusion index, coverage…), not masks | **CONFIRMED 2-0** |
| **MyoFuse** | **none reported** — validation is model-vs-expert only (acc 0.954 / 0.911; r 0.991 / 0.937) | — | **UNVERIFIED** |
| "threshold subjectivity is the dominant source of human variability" | — | — | **REFUTED 0-3** — do not repeat |

Broader multi-annotator bioimage literature (Cellpose / StarDist / LIVECell /
NeurIPS cell-seg challenge) and guidance on required annotator counts: **not
reached** before the limit.

---

## Caveats

- Two claims are **internally contradictory** on MyoFInDer's data availability
  (one agent says no data, verifiers refuted 0-3). Unresolved.
- The NCL-SM 0.96 IoU is on **round cross-sections**. Our objects are long,
  crossing, and broken. Treating 0.96 as our ceiling would be optimistic; the
  comparable number for longitudinal in-vitro fibres does not appear to exist.
- Nothing here was fetched into `competitors/` as data. No dataset has been
  downloaded.

## Re-run

Within the **original session only**, the run could have been resumed with
`Workflow({scriptPath: '…/deep-research-wf_9871ec66-003.js',
resumeFromRunId: 'wf_9871ec66-003'})`, replaying the 48 cached agents and
re-running only the 52 failures. **Workflow resume is same-session only, and that
session is closed** — so from any new session, launch a *fresh* `deep-research`
run.

Narrow it to what is still missing, so it fits inside the account limit:

1. **NCL-SM** — verify the instance-mask claim (50,434 myofibres, 46 sections)
   and the **inter-annotator mean IoU 0.96** figure. Highest value of the four.
2. **arXiv 2604.14720** — verify the "no large annotated myotube dataset exists
   publicly" statement, and their real validation set (17 volumes / 40 instances).
3. **SEPO-FI, MyoCount, ViaFuse** — never searched.
4. **IDR / BBBC / EMPIAR / BioStudies / Kaggle** sweep — agent ran but its results
   never reached synthesis.
5. Broader multi-annotator bioimage literature (Cellpose / StarDist / LIVECell /
   NeurIPS cell-seg challenge) for expert-vs-expert Dice — never reached.

The rows already marked **CONFIRMED** above do not need re-running.
