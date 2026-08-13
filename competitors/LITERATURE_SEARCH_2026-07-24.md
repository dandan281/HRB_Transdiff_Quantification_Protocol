# Literature search — targeted re-run of the five open gaps (2026-07-24)

**Scope.** This run closes the five gaps left open by
[`LITERATURE_SEARCH_2026-07-23.md`](LITERATURE_SEARCH_2026-07-23.md) plus the
one contradiction that file left unresolved (MyoFInDer data availability).
Claims already tagged **CONFIRMED** in the 07-23 file were *not* re-run.
That file is retained as evidence; nothing here overwrites it.

> **RUN STATUS — COMPLETE, with two named exceptions.**
> Five of six items were fully reached. The exceptions are stated plainly in
> [§7 What was not reached](#7-what-was-not-reached): **EMPIAR** (search API
> returned 404 on every endpoint tried — never queried) and **Kaggle**
> (searched only indirectly via web search, not by enumerating the site).
> No other item was truncated.

> **METHODOLOGY NOTE — this is not a `deep-research` multi-agent run.**
> The `deep-research` skill's `Workflow` executor was **not available** in this
> agent context (`ToolSearch` for `Workflow` returned no match), so the fan-out /
> adversarial-vote harness could not be launched. The same protocol was executed
> **manually**: independent searches per angle, primary-source fetches, and
> per-claim corroboration. **"Votes" below therefore mean *independent sources
> that corroborate or contradict a claim*, not independent verifier agents.**
> A `2-0` here is weaker evidence than a `2-0` in the 07-23 file. Single-source
> rows are marked `1-0` and flagged. Wherever possible the source is the
> *primary* document (paper full text, repo README, repository API), not a
> search snippet.

---

## 1 — Headline answer (the question that matters)

### Does an external, public, instance-level, **in-vitro** myotube segmentation baseline exist that we could score T02 against?

## **No.** — CONFIRMED 6-0

Six independent lines of evidence, none contradicting:

| Evidence | What it shows |
|---|---|
| **BBBC** full image-set index enumerated (54 sets) | Zero myotube / myoblast / C2C12 / skeletal-muscle sets |
| **IDR** search API, term `myotube` | **0 hits** across images, plates, projects, screens, wells |
| **IDR** control query `myoblast` | Returns hits → endpoint functional, so the 0 is real, not a broken query |
| **BioImage Archive** (`bioimages` collection), term `myotube` | **2 hits**, neither a segmentation dataset (circadian gene expression; cardiac EM montages) |
| **arXiv 2604.14720** (Apr 2026, the most recent people to look) | *"As no large annotated 3D myotube datasets, nor a myotube simulation pipeline exist…"* — and they did not release their own 40 real instances |
| **NCL-SM**, the one large instance-level muscle dataset | **Histology cross-sections from biopsies**, not cultured myotubes — wrong object geometry |

**Consequence for the project.** This is the same conclusion the 07-23 run
reached, but now it rests on repository-level negative evidence rather than on
tool-paper reading alone. There is **no external ground truth to score T02
against**, so the internal circularity audit of our classical floor remains the
only check available. The nearest thing to a usable external object is NCL-SM,
and its objects are round cross-sections — see §2 for why that is not a
substitute.

**Second-order finding worth noting.** A 2026 group with a 3D myotube dataset
in hand (arXiv 2604.14720) chose to build a *synthesis pipeline* rather than
annotate at scale, and released the pipeline but **not** their 40 real
annotated instances. That is independent corroboration that instance-level
myotube annotation is expensive enough that nobody has published one — and it
means a future collaboration/data-request to that group is the cheapest route
to an external set, if one is ever wanted.

---

## 2 — Item 1: NCL-SM dataset (highest value)

DOI **`10.25405/data.ncl.24125391`** ·
[data.ncl.ac.uk record](https://data.ncl.ac.uk/articles/dataset/Newcastle_Skeletal_Muscle_NCL-SM_A_Fully_Annotated_Dataset_of_Images_from_Human_Skeletal_Muscle_Biopsies/24125391) ·
papers [arXiv:2311.15113](https://arxiv.org/abs/2311.15113) and
[arXiv:2311.11099](https://arxiv.org/abs/2311.11099) ·
code [github.com/atifkhanncl/NCL-SM](https://github.com/atifkhanncl/NCL-SM)

| Claim | Verdict |
|---|---|
| **50,434 myofibres across 46 tissue sections** | **CONFIRMED 3-0** (arXiv 2311.15113 full text; arXiv 2311.11099 abstract *">50,000 … 46 sections"*; record description) |
| Breakdown: 30,794 analysable (AM); 18,102 not-analysable-due-to-shape; 1,538 not-analysable-due-to-freezing-damage; 405 folded-tissue-region annotations | **CONFIRMED 2-0** (arXiv 2311.15113; record description) |
| Modality split: **IMC 27 sections / 22,979 myofibres; IF 19 sections / 27,455 myofibres** | **CONFIRMED 1-0** (arXiv 2311.15113 full text — single source) |
| **Masks are instance-level**, one label per myofibre | **CONFIRMED 2-0** (GitHub README describes *"instance segmentation mask of all analysable myofibres"*; paper) |
| Mask sets named **`Mask_All_AM`** and **`Mask_AM_vs_NAM`** | **CONFIRMED 1-0** — GitHub README, verbatim: `Mask_All_AM` = *"instance segmentation mask of all analysable myofibres"*; `Mask_AM_vs_NAM` = *"class mask of analysable vs non-analysable myofibres"*. **Third set found that the 07-23 file did not list: `Mask_QA`** = *"segmentation mask of quality assurance duplicate annotations"* — this is the set the inter-annotator numbers come from, and it is downloadable. |
| **Histology cross-sections from human biopsies, NOT cultured in-vitro myotubes** | **CONFIRMED 3-0** (paper: *"cross-sections"* from biopsies, healthy controls + genetically diagnosed muscle pathology incl. mitochondrial disease; both arXiv records; GitHub) |
| Licence **CC BY 4.0** | **CONFIRMED 1-0** (paper) |
| Size **5.9 GB**, no access request needed | **UNVERIFIED** — the `data.ncl.ac.uk` record returned HTTP 403 / timeout on every fetch attempt. Not refuted, just not reached. |

### 2.1 The inter-annotator figure — **CONFIRMED 1-0, but the 07-23 file states it slightly wrong**

Published quality-assurance comparison (human re-segmentation vs original), from
arXiv 2311.15113:

| Modality | n myofibres re-segmented | rAoB | rAiB | **mean IoU** |
|---|---|---|---|---|
| IMC | **53** | **0.99** | **0.77** | **0.96** |
| IF | **23** | **0.92** | **0.94** | **0.96** |

- **The `mean IoU 0.96` figure is CONFIRMED** (1-0, primary source; the PDF
  re-fetch failed so there is no second independent read).
- **Correction to the 07-23 file.** That file records *"mean IoU 0.96 … (53 IMC
  + 23 IF myofibres; rAoB 0.99, rAiB 0.77)"*, which reads as though 0.99/0.77
  apply to the pooled 76 fibres. They do **not** — `rAoB 0.99 / rAiB 0.77` is
  the **IMC row only**. The IF row is `rAoB 0.92 / rAiB 0.94`. The 0.96 IoU
  happens to hold for both. **Cite it as two rows, not one.**
- **n = 76 total.** This is a small QA sample, and the paper does not report a
  distribution or CI. Treat 0.96 as a point estimate on a handful of fibres.

### 2.2 Why NCL-SM is still not our ceiling

Unchanged from 07-23, now with the numbers verified: 0.96 IoU is agreement on
**round, convex, non-touching cross-sections** — the easiest possible
delineation geometry. Our objects are long, crossing, branching and broken.
Quoting 0.96 as the expected human ceiling for T02 would be optimistic and the
report should say so whenever the number is used.

---

## 3 — Item 2: arXiv 2604.14720, *Data Synthesis Improves 3D Myotube Instance Segmentation*

[arXiv:2604.14720](https://arxiv.org/abs/2604.14720) ·
Exler, Friederich, Krüger, Jbeily, Vitacolonna, Rudolf, Mikut, Reischl ·
submitted 16 Apr 2026 · pipeline at `github.com/DavidExler/syn_myo`

| Claim | Verdict |
|---|---|
| The arXiv ID **2604.14720 resolves** and is the 3D myotube synthesis paper | **CONFIRMED 3-0** (search index, `/abs/`, `/html/`) |
| Authors state **no large annotated myotube dataset exists publicly** | **CONFIRMED 2-0**. Verbatim from the body: *"As no large annotated 3D myotube datasets, nor a myotube simulation pipeline exist, we tackle this challenge by presenting the first data synthesis pipeline for 3D myotube data."* Abstract: *"…fail to generalize to this domain due to the absence of large annotated myotube datasets."* **Nuance: the strong claim is scoped to 3D.** The abstract's phrasing is unscoped, the body's is "3D". Do not over-read it as a proof about 2D. |
| Real validation set = **17 volumes / 40 manual instances** | **CONFIRMED 1-0**. Verbatim: *"This dataset contains 17 images of shape (z, 1024, 1024), with z ranging from 24 to 128"*; *"n=40 individual myotube instances across 17 volumetric stacks."* |
| **No inter-annotator agreement reported** | **CONFIRMED 1-0** (full-text read found none) |
| **Only the synthesis pipeline is released, not the real annotated data** | **CONFIRMED 1-0**. Verbatim: *"The synthesis pipeline is publicly available at github.com/DavidExler/syn_myo."* No data DOI anywhere in the paper. |
| Synthetic training set = 200 instances across 30 volumes of 128×1024×1024 | **CONFIRMED 1-0** |
| Best result: 3D U-Net, self-supervised encoder pretraining, synthetic-only training, **mean IPQ 0.22** on real data, beating three zero-shot baselines | **CONFIRMED 2-0** (abstract; search summary) |

**Everything the 07-23 file assumed about this paper holds.** The only addition
is the scoping caveat on "3D", and a hard confirmation that their 40 real
instances are **not** downloadable.

**Practical note.** `mean IPQ 0.22` is a low number, and it is the state of the
art for 3D myotube instance segmentation as of April 2026. If T02 is anywhere
near that on a comparable 3D task, that is context worth having.

---

## 4 — Item 3: SEPO-FI, MyoCount, ViaFuse (never previously searched)

### SEPO-FI (Kim et al., *Comput Biol Med* 186, 2025) — nuclei, not myotubes
[PubMed 39862466](https://pubmed.ncbi.nlm.nih.gov/39862466/) · DOI `10.1016/j.compbiomed.2025.109706`

| Claim | Verdict |
|---|---|
| Output is **nuclei detection + classification (inside vs outside myotube)**, *not* myotube instance masks | **CONFIRMED 2-0** (PubMed abstract; independent search summary). Pipeline: colour-based pattern recognition detects nuclei → `SEPONet` CNN classifies each nucleus inside/outside a myotube → count → fusion index. Myotubes are never individuated. |
| Reports **F1 = 0.953** for nucleus detection/classification vs <0.5 for conventional object detection | **CONFIRMED 2-0** |
| Releases an annotated image dataset | **UNVERIFIED** — abstract says *"public software"* only; no DOI/repository named in the PubMed record. Full text is paywalled (ScienceDirect) and was not reached. **Leaning no, but not established.** |

### MyoCount (Murphy et al., *Wellcome Open Res* 2019) — threshold + connected components; **does publish data, but not masks**
[PMC6419977](https://pmc.ncbi.nlm.nih.gov/articles/PMC6419977/) · [PubMed 30906880](https://pubmed.ncbi.nlm.nih.gov/30906880/)

| Claim | Verdict |
|---|---|
| Myotube detection is **global threshold → binarise → fill/smooth → size filter**, i.e. connected components | **CONFIRMED 2-0**. Verbatim: *"Myotubes and their approximate borders are identified by normalisation and thresholding to give a binary image. The tool uses filling and smoothing followed by removal of noise—including any objects below the threshold size—to identify approximate borders of individual myotubes."* |
| It therefore yields **per-connected-component objects, but does not separate touching/overlapping myotubes** | **CONFIRMED 2-0** — the authors concede *"some errors persist"* on overlapping myotubes; a second read confirms the four QC output images show *"the estimated centroids and approximate borders of the **nuclei**"*, not of myotubes. **This is the important nuance: MyoCount is nominally instance-like (labelled blobs) but is a thresholding method, so it fails on exactly our hard case — two myotubes touching become one object.** It is not an instance-segmentation baseline. |
| **It does publish data** — validation `.tif` + `.csv` on OSF | **CONFIRMED 1-0**. Verbatim data availability: *"The data underlying the results presented in Figure 4 and Figure 5, (.csv and .tif files) are available as 'Myocount Validation Data' via OSF. DOI: https://doi.org/10.17605/OSF.IO/F5DXE. Data are available under the terms of the Creative Commons Zero 'No rights reserved' data waiver (CC0 1.0 Public domain dedication)."* **But these are raw images + measurement tables — no pixel-level ground-truth masks.** CC0, so freely reusable if raw in-vitro myotube images are ever wanted. |
| Human-vs-human variability: **mean CV between two blinded investigators = 14.3 ± 2.5% (myotube area)** and **17.6 ± 2.3% (NFI)**; investigator-mean-vs-MyoCount CV 13.3 ± 1.4% for NFI | **CONFIRMED 2-0**. Agreement with manual: area R² = 0.89, NFI R² = 0.87. Note this is a **coefficient of variation on derived scalars, not a mask overlap metric** — it is not comparable to IoU. Only **2 raters**. |

### ViaFuse (Hopkins et al., *Skeletal Muscle* 2021) — semantic binary mask only
[PMC8675483](https://pmc.ncbi.nlm.nih.gov/articles/PMC8675483/) · DOI `10.1186/s13395-021-00284-3` · code `github.com/tasneemsmacros/ViaFuse`

| Claim | Verdict |
|---|---|
| Produces a **binary/semantic myotube mask + nuclei counts**, **no** per-myotube instances | **CONFIRMED 2-0** (primary full text; search summary). Method verbatim: *"smoothing the image using the Gaussian Blur filter followed by applying a threshold to the image using the default method in Fiji to make it a binary image and then a median filter to reduce the noise and then filled the small holes."* The MYH binary mask is then subtracted from the DAPI image to find nuclei outside myotubes. Individual myotubes are never separately labelled. |
| **No annotated dataset published** | **CONFIRMED 1-0**. Verbatim: *"The datasets and materials used and/or analyzed during the current study are available from the corresponding author on reasonable request."* GitHub holds macros only. |
| Claimed advantages over MyoCount: distinguishes nuclear clumps, identifies myotube borders correctly, distinguishes nearby myoblasts | **CONFIRMED 1-0** (authors' own claim — not independently evaluated) |

**Summary of item 3:** all three are fusion-index tools. **None produces
instance-level myotube masks.** SEPO-FI works at nuclei level; ViaFuse produces
a semantic binary mask; MyoCount produces threshold-derived connected
components that merge touching myotubes. **None publishes ground-truth myotube
masks.** MyoCount alone publishes raw validation images (OSF, CC0).

---

## 5 — Item 4: public bioimage repository sweep

| Repository | Query | Result | Verdict |
|---|---|---|---|
| **BBBC** (Broad) | Enumerated the **full image-set index** (BBBC001–BBBC054, 54 sets) | **Zero** myotube / myoblast / C2C12 / skeletal-muscle / muscle-fibre sets. Cell types are HT29, Kc167, U2OS, MCF7, A549, C. elegans, HL60, Jurkat, macrophages, astrocytes, CHO, hepatocytes, microglia, CAD, kidney cortex, blood smears, embryos, synthetic. | **CONFIRMED 1-0** (direct index read — authoritative, this is the complete list) |
| **IDR** | `searchengine/api/v1/.../searchvalues/?value=myotube` | **0 results** across image / plate / project / screen / well | **CONFIRMED 2-0** — the second vote is a **control query**: `value=myoblast` returns hits (gene "myoblast city"), proving the endpoint works and the zero is genuine |
| **BioImage Archive / BioStudies** (`bioimages` collection) | `myotube` | **2 hits total**: `S-BIAD1317` (EZH1 circadian gene expression) and `S-BIAD1475` (2D TEM montages of left-ventricular heart muscle, RBM20-deficient mice). **Neither is a myotube segmentation dataset.** | **CONFIRMED 1-0** (repository API) |
| **BioStudies** (all collections) | `myotube` and `myotube segmentation` | 50 hits reviewed for `myotube` (all molecular-biology PMC records, no imaging ground truth); 346 hits for `myotube segmentation`, top 50 reviewed — the segmentation datasets present are nuclei, mitochondria, roots, embryos, E. coli, etc. **No muscle instance-mask dataset.** | **CONFIRMED 1-0** |
| **Kaggle** | web search for myotube/C2C12 segmentation masks | Nothing found. Nearest neighbours are the Sartorius neuronal-cell segmentation competition and generic "Medical Cells Image Segmentation" — neither is muscle. | **UNVERIFIED (absence of evidence)** — searched indirectly, the site was not enumerated. See §7. |
| **EMPIAR** | — | **NOT REACHED.** Every API/search endpoint tried returned 404. See §7. | **NOT REACHED** |

**Net:** across the three repositories that were properly enumerated (BBBC, IDR,
BioImage Archive), the answer is a clean **no**. Kaggle is a soft no. EMPIAR is
unqueried but is a raw-EM archive (cryo-EM/ET/volume EM), which is the wrong
modality for in-vitro fluorescence myotube instance masks — the residual risk
there is low but non-zero.

---

## 6 — Item 5: expert-vs-expert agreement in general bioimage segmentation

### 6.1 Cellpose — the only source with a real human-vs-human number

**Cellpose-SAM**, Pachitariu, Rariden & Stringer, bioRxiv 2025,
[10.1101/2025.04.28.651001](https://www.biorxiv.org/content/10.1101/2025.04.28.651001v1)
(preprint; **not** yet in a journal as of this search).

| Claim | Verdict |
|---|---|
| The Cellpose test set was **relabelled by a second, independent annotator** to measure inter-annotator variability | **CONFIRMED 2-0** |
| **Annotator2 error rate = 0.257** relative to Annotator1 | **CONFIRMED 2-0** (two independent retrievals of the full text agree) |
| **Human-consensus estimate = 0.128** (obtained by halving the inter-annotator error rate) | **CONFIRMED 2-0** — note this is an *assumption-based extrapolation by the authors*, not a measurement. They argue both annotators err, so a consensus would halve the rate. |
| Model error rates on the same benchmark: **Cellpose3 = 0.292, CellSAM = 0.328, Cellpose-SAM = 0.163** | **CONFIRMED 2-0** |
| Abstract framing | **CONFIRMED 1-0**, verbatim: *"Modern algorithms for biological segmentation can match inter-human agreement in annotation quality. This however is not a performance bound: a hypothetical human-consensus segmentation could reduce error rates in half. … The resulting Cellpose-SAM model substantially outperforms inter-human agreement and approaches the human-consensus bound."* |
| "Error rate" = 1 − AP at IoU 0.5, where AP = TP/(TP+FP+FN) | **UNVERIFIED / INFERRED.** This is the standard Cellpose metric (AP = TP/(TP+FP+FN), matching at IoU 0.5 — **CONFIRMED 1-0** from the Cellpose metric definition), and error rate is almost certainly its complement, but the Cellpose-SAM full text could not be fetched (403 on every biorxiv `.full` / `.pdf` route) to confirm the definition verbatim. **Do not quote the metric definition as established.** |

**This is the most directly usable number in the whole search.** Reframed for
the project: **two independent expert annotators of ordinary cells disagree
enough that one scores ~0.26 error against the other** — i.e. roughly
**74% agreement, not 96%**, on an instance-matching metric rather than a
boundary-overlap metric. That is a far more relevant reference point for our
categorical/instance disposition work than NCL-SM's 0.96 boundary IoU, because
it is *instance-level* and it counts missed and spurious objects, which is
exactly where our tracing errors live.

### 6.2 LIVECell — **no inter-annotator figure exists**

Edlund et al., *Nature Methods* 2021,
[PMC8440198](https://pmc.ncbi.nlm.nih.gov/articles/PMC8440198/) ·
data [figshare 10.6084/m9.figshare.14931555](https://doi.org/10.6084/m9.figshare.14931555)

| Claim | Verdict |
|---|---|
| **No inter-annotator / inter-observer agreement metric (IoU, Dice, F1) is reported anywhere in the paper** | **CONFIRMED 1-0** (primary full text) |
| Annotation was by a **managed professional team (CloudFactory)** trained by an experienced cell biologist, not by domain experts and not in duplicate | **CONFIRMED 1-0** |
| QA was **two-level review**, not duplicate annotation: annotation managers inspect every image and return faults with feedback; then an experienced cell biologist gives final approval. Verbatim: *"Images passing both rounds of approval were included in LIVECell. To assure that the annotation managers assessments stayed consistent, there were frequent follow-up calls where the cell biologist provided feedback on difficult cases directly to the annotation managers."* | **CONFIRMED 1-0** |
| Scale: **5,239 images, 1,686,352 cells, 8 cell types** | **CONFIRMED 2-0** |

**Reading for us:** the largest, most carefully built instance-segmentation
dataset in the field **did not measure human-vs-human agreement at all**. It
used a review hierarchy instead. That is a defensible precedent for our own
process — and it means "no published inter-annotator IoU" is the *norm*, not a
gap peculiar to muscle.

### 6.3 NeurIPS 2022 Cell Segmentation Challenge — **no inter-annotator figure either**

*The multimodality cell segmentation challenge: toward universal solutions*,
Ma et al., *Nature Methods* 21:1103–1113 (2024),
[10.1038/s41592-024-02233-6](https://www.nature.com/articles/s41592-024-02233-6) ·
[PMC11210294](https://pmc.ncbi.nlm.nih.gov/articles/PMC11210294/) ·
data [Zenodo 10719375](https://zenodo.org/records/10719375)

| Claim | Verdict |
|---|---|
| **No inter-annotator agreement metric reported**; no independent duplicate annotations | **CONFIRMED 1-0** (primary full text) |
| Annotation team = **two biologists with 10 years' experience**, whose role was *"ensuring compliance with annotation requirements"* — i.e. supervisors, not duplicate raters | **CONFIRMED 1-0** |
| Workflow was **model-assisted**: for unlabelled images, *"publicly available specialist models were initially employed to generate predictions. The resulting segmentation outcomes were subsequently subjected to manual revision"*; contributed annotations were *"thoroughly checked and revised as needed"* | **CONFIRMED 1-0** |
| QC was **rule-based**: *"each image-annotation pair underwent stringent quality control. Images with less than five cells were excluded from the dataset, and cells containing fewer than 15 pixels were also removed."* | **CONFIRMED 1-0** |
| Scale: >900,000 new cell annotations; 1,000 labelled training + 1,725 unlabelled images | **CONFIRMED 1-0** |

**Note the circularity.** The field's flagship multi-modality benchmark was
built by **models proposing and humans correcting** — structurally the same
loop as our QC review tool. If a reviewer ever attacks our proposal-triage
workflow as circular, this is the citation that shows it is standard practice
at the top of the field.

### 6.4 StarDist — nothing found

| Claim | Verdict |
|---|---|
| StarDist (Schmidt et al. MICCAI 2018; Weigert et al. WACV 2020) reports inter-annotator agreement | **UNVERIFIED (absence of evidence)** — targeted searches returned no such figure. The papers appear to use single-annotator ground truth (worm annotation credited to one person, D. Kainmüller). Primary full texts were not fetched, so this is a soft negative. |

### 6.5 Guidance on how many annotators are needed

| Claim | Verdict |
|---|---|
| **There is no prescriptive standard** for the number of annotators in biomedical image segmentation | **CONFIRMED 2-0** (two independent search syntheses; corroborated by the fact that LIVECell, NeurIPS-CellSeg and StarDist all used effectively one annotation pass with review) |
| The field's actual practice is **label fusion over multiple raters** — majority vote and **STAPLE** — rather than a fixed rater count | **CONFIRMED 2-0** |
| Reported agreement metrics in use: IoU, Dice, Cohen's κ, Fleiss' κ, common-agreement heatmaps | **CONFIRMED 2-0** |
| Inter-expert agreement in MS brain lesion segmentation with **7 experts: median Dice 0.66–0.76** against consensus | **UNVERIFIED 1-0** — from a search synthesis only; the primary paper was not fetched. **Do not cite externally without checking.** Directionally it says: on a genuinely hard segmentation task, seven experts agree at Dice ~0.7, nowhere near 0.96. |
| Relevant recent primary literature located but not read: *When Experts Disagree: Characterizing Annotator Variability for Vessel Segmentation in DSA Images* ([arXiv:2508.10797](https://www.arxiv.org/pdf/2508.10797)); *What Can We Learn from Inter-Annotator Variability in Skin Lesion Segmentation?* ([arXiv:2508.09381](https://arxiv.org/abs/2508.09381)); *Assessing Inter-Annotator Agreement for Medical Image Segmentation* (IEEE 10054393); *Automated Annotator Variability Inspection for Biomedical Image Segmentation* (IEEE 9668915) | **NOT ASSESSED** — leads for a future run if a formal annotator-count justification is ever needed |

---

## 7 — Item 6: MyoFInDer data availability — **contradiction resolved**

Weisrock, Wüst, Olenic, Lecomte-Grosbras & Thorrez, *Tissue Eng Part A* 30(19–20):652–661, 2024 ·
DOI [`10.1089/ten.tea.2024.0049`](https://doi.org/10.1089/ten.tea.2024.0049) ·
[PubMed 38832871](https://pubmed.ncbi.nlm.nih.gov/38832871/) ·
code [github.com/TissueEngineeringLab/MyoFInDer](https://github.com/TissueEngineeringLab/MyoFInDer) (GPL-3.0) ·
docs [tissueengineeringlab.github.io/MyoFInDer](https://tissueengineeringlab.github.io/MyoFInDer/) ·
[PyPI `myofinder`](https://pypi.org/project/myofinder/)

### Verdict: **MyoFInDer ships no annotated image dataset. CONFIRMED 4-0.**
### The 07-23 file's "REFUTED 0-3" was wrong and should be treated as retracted.

Four independent places were checked; none yielded data:

| Where checked | Finding |
|---|---|
| **GitHub README** | No Zenodo/DOI/figshare link, no sample or test images, no ground-truth masks. Licence GPL-3.0. README covers installation and citation only. |
| **Documentation site** | Sections are *Installation, Usage, Troubleshooting*. No data resources of any kind. |
| **Zenodo / general web search** for a MyoFInDer dataset | Nothing. |
| **Publisher page** (Liebert → SAGE) | **No Data Availability statement present.** Two supplementary PDFs only (`sj-pdf-1`, 100 KB; `sj-pdf-2`, 7 KB) — sizes far too small to be image data. Abstract says only *"As a free and open-source project, MyoFInDer can be modified or extended to meet specific needs."* |

**Residual caveat, stated honestly:** the article full text is paywalled and was
not read end-to-end. The claim is therefore *"no data availability statement or
data link exists in the README, docs, PyPI, publisher landing page, or any
indexed repository"* — which is strong, converging negative evidence, but is not
the same as having read a Methods section that says "no data were deposited".
The correct tag is **CONFIRMED 4-0 on the observable record**, not "proven".

**Where the 07-23 confusion probably came from.** MyoFInDer's segmentation
backbone is a pretrained deep model, and the paper reports validation against
manual counts — it is easy for a verifier skimming to read "AI-based, validated
against experts" as implying a released annotated set. It does not. The
07-23 row *"semantic-only: CONFIRMED 2-0"* stands and is unaffected.

---

## 8 — What was not reached

Stated plainly, per the run requirement:

1. **EMPIAR — never queried.** Three endpoint forms
   (`/empiar/api/search/…`, `/empiar/api/entry/search/…`, `/empiar/search/…`)
   all returned HTTP 404, and no working search route was found. **The EMPIAR
   leg of item 4 is unexecuted.** Risk assessment: EMPIAR archives raw
   electron-microscopy data (cryo-EM, cryo-ET, volume EM); in-vitro
   fluorescence myotube instance masks would be badly out of scope for it.
   Low residual risk, but it is a genuine gap, not a covered one.
2. **Kaggle — indirect only.** Covered by web search, not by enumerating
   Kaggle's dataset index. A private or oddly-named competition dataset could
   have been missed.
3. **SEPO-FI data availability — abstract only.** ScienceDirect full text is
   paywalled. The "no dataset released" reading is a lean, not a finding.
4. **NCL-SM download size / access conditions** — `data.ncl.ac.uk` returned
   403/timeout on every attempt. The 5.9 GB figure and "no access request"
   claim from the 07-23 file remain **UNVERIFIED**.
5. **Cellpose-SAM "error rate" metric definition** — biorxiv full text 403'd on
   every route. The numbers are confirmed from two retrievals; the metric
   *definition* is inferred.
6. **StarDist inter-annotator** — negative from search only; primary papers not
   fetched.
7. **The four annotator-variability primaries listed in §6.5** — located, not read.

Nothing above is presented as covered. Items 1, 2, 3, 5 and 6 of the original
five-gap list are fully closed; item 4 is closed for BBBC / IDR / BioImage
Archive / BioStudies and open for EMPIAR / Kaggle.

---

## 9 — Consolidated table (supersedes the 07-23 tables for these rows only)

### Datasets

| Dataset | Instance-level myotubes? | In vitro? | Public masks? | Verdict |
|---|---|---|---|---|
| **NCL-SM** `10.25405/data.ncl.24125391` | **Yes** — 50,434 myofibres / 46 sections; `Mask_All_AM`, `Mask_AM_vs_NAM`, `Mask_QA` | **No — histology cross-sections** | Yes, CC BY 4.0 | **CONFIRMED 3-0**; size/access **UNVERIFIED** |
| **arXiv 2604.14720** real set | Yes — 40 instances / 17 volumes, 3D | Yes | **No** — pipeline only (`syn_myo`) | **CONFIRMED 2-0** |
| **SEPO-FI** | **No** — nuclei detect + classify | Yes | No (leaning) | instance: **CONFIRMED 2-0**; data: **UNVERIFIED** |
| **MyoCount** | **No** — threshold + connected components; merges touching myotubes | Yes | **Raw images + CSVs only**, OSF `10.17605/OSF.IO/F5DXE`, CC0 — **no masks** | **CONFIRMED 2-0** |
| **ViaFuse** | **No** — Gaussian blur + Fiji default threshold → binary semantic mask | Yes | **No** — "on reasonable request" | **CONFIRMED 2-0** |
| **MyoFInDer** | **No** (semantic, per 07-23) | Yes | **No** — code only, GPL-3.0 | **CONFIRMED 4-0** (07-23's REFUTED 0-3 retracted) |
| **BBBC** (54 sets enumerated) | — | — | **No muscle sets at all** | **CONFIRMED 1-0** |
| **IDR** | — | — | **0 hits for `myotube`** (control query passes) | **CONFIRMED 2-0** |
| **BioImage Archive** | — | — | 2 hits, neither a segmentation set | **CONFIRMED 1-0** |
| **Kaggle** | — | — | None found | **UNVERIFIED (absence)** |
| **EMPIAR** | — | — | — | **NOT REACHED** |

### Annotator variability

| Source | Figure | Task | Verdict |
|---|---|---|---|
| **Cellpose-SAM** | **Annotator2 error 0.257** vs Annotator1; human-consensus **0.128** (extrapolated); Cellpose3 0.292, CellSAM 0.328, Cellpose-SAM 0.163 | **instance-level** cell segmentation, general microscopy | **CONFIRMED 2-0**; metric definition **INFERRED** |
| **NCL-SM** | **mean IoU 0.96** both modalities; IMC (n=53) rAoB 0.99 / rAiB 0.77; IF (n=23) rAoB 0.92 / rAiB 0.94 | **boundary** delineation, histology cross-section, n=76 | **CONFIRMED 1-0** (07-23's single pooled row corrected) |
| **MyoCount** | CV between 2 blinded investigators: **14.3 ± 2.5%** (myotube area), **17.6 ± 2.3%** (NFI) | derived scalars, in vitro, 2 raters | **CONFIRMED 2-0** |
| **Myotube Analyzer** | ICC > 0.75 on 5/6 params, 2 raters | derived scalars, in vitro | **CONFIRMED 2-0** *(carried from 07-23, not re-run)* |
| **LIVECell** | **none reported** — review hierarchy instead of duplicate annotation | — | **CONFIRMED 1-0** |
| **NeurIPS 2022 CellSeg** | **none reported** — 2 supervising biologists, model-assisted annotation, rule-based QC | — | **CONFIRMED 1-0** |
| **StarDist** | none found | — | **UNVERIFIED (absence)** |
| MS brain lesion, 7 experts | median Dice **0.66–0.76** vs consensus | hard 3D lesion segmentation | **UNVERIFIED 1-0** — do not cite externally |
| Required annotator count | **no prescriptive standard exists**; practice is label fusion (majority vote / STAPLE) | — | **CONFIRMED 2-0** |

---

## 10 — What this changes for the project

1. **The "no external baseline" conclusion is now repository-verified, not just
   inferred from tool papers.** It can be stated in a methods section with
   citations: BBBC index, IDR search (with control), BioImage Archive query,
   plus arXiv 2604.14720's own admission.
2. **Stop leading with NCL-SM's 0.96 as the human ceiling.** It is a
   boundary-overlap number on n=76 round cross-sections. **Cellpose-SAM's
   0.257 inter-annotator error rate is the better reference** — it is
   instance-level, it penalises missed and spurious objects, and it says two
   experts on ordinary cells agree at roughly 74%, not 96%. Against that, the
   operator's blind-repeat median IoU of 1.0 on complete/complete pairs and 90%
   categorical agreement look **strong, not weak**. The 07-23 conclusion holds
   and is now better supported.
3. **"No published inter-annotator agreement" is the field norm.** LIVECell,
   NeurIPS-CellSeg and (apparently) StarDist all skip it. We are not behind by
   not having one; if we compute one we are ahead.
4. **Model-proposes / human-corrects is standard practice**, per the NeurIPS
   challenge's own construction. Useful defence for our QC review loop.
5. **Two free data sources exist if raw in-vitro myotube images are ever
   wanted** (neither has masks): MyoCount validation data on OSF
   (`10.17605/OSF.IO/F5DXE`, CC0) and MyoFuse on Zenodo
   (`10.5281/zenodo.14731491`, CC-BY-4.0, per 07-23).
6. **The only realistic route to an external instance-level in-vitro baseline is
   to ask for one** — the Exler et al. group (KIT / Mannheim) holds 40 manually
   annotated 3D myotube instances across 17 volumes that they did not publish.

---

## 11 — Sources

- NCL-SM dataset — https://doi.org/10.25405/data.ncl.24125391 · https://data.ncl.ac.uk/articles/dataset/Newcastle_Skeletal_Muscle_NCL-SM_A_Fully_Annotated_Dataset_of_Images_from_Human_Skeletal_Muscle_Biopsies/24125391
- NCL-SM papers — https://arxiv.org/abs/2311.15113 · https://arxiv.org/abs/2311.11099 · IEEE Xplore 10386552
- NCL-SM code — https://github.com/atifkhanncl/NCL-SM
- Exler et al., *Data Synthesis Improves 3D Myotube Instance Segmentation* — https://arxiv.org/abs/2604.14720 · pipeline https://github.com/DavidExler/syn_myo
- SEPO-FI — https://doi.org/10.1016/j.compbiomed.2025.109706 · https://pubmed.ncbi.nlm.nih.gov/39862466/
- MyoCount — https://pmc.ncbi.nlm.nih.gov/articles/PMC6419977/ · https://pubmed.ncbi.nlm.nih.gov/30906880/ · data https://doi.org/10.17605/OSF.IO/F5DXE
- ViaFuse — https://doi.org/10.1186/s13395-021-00284-3 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8675483/ · https://github.com/tasneemsmacros/ViaFuse
- MyoFInDer — https://doi.org/10.1089/ten.tea.2024.0049 · https://pubmed.ncbi.nlm.nih.gov/38832871/ · https://github.com/TissueEngineeringLab/MyoFInDer · https://tissueengineeringlab.github.io/MyoFInDer/ · https://pypi.org/project/myofinder/
- Cellpose-SAM — https://doi.org/10.1101/2025.04.28.651001 · https://www.janelia.org/publication/cellpose-sam-superhuman-generalization-for-cellular-segmentation
- Cellpose metric definition — https://cellpose.readthedocs.io/en/latest/_modules/cellpose/metrics.html · https://doi.org/10.1101/2020.02.02.931238
- LIVECell — https://doi.org/10.1038/s41592-021-01249-6 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8440198/ · data https://doi.org/10.6084/m9.figshare.14931555
- NeurIPS 2022 Cell Segmentation Challenge — https://doi.org/10.1038/s41592-024-02233-6 · https://pmc.ncbi.nlm.nih.gov/articles/PMC11210294/ · https://neurips22-cellseg.grand-challenge.org/ · data https://zenodo.org/records/10719375 · code https://github.com/JunMa11/NeurIPS-CellSeg
- BBBC — https://bbbc.broadinstitute.org/image_sets
- IDR — https://idr.openmicroscopy.org/searchengine/api/v1/resources/all/searchvalues/?value=myotube
- BioImage Archive / BioStudies — https://www.ebi.ac.uk/biostudies/api/v1/search?query=myotube&collection=bioimages
- Annotator-variability leads (not read) — https://www.arxiv.org/pdf/2508.10797 · https://arxiv.org/abs/2508.09381 · IEEE 10054393 · IEEE 9668915
