# G-SO1 integrator validation

Evaluated: 2026-07-21 local / 2026-07-22 UTC  
Subject: round-2 single-operator blind repeat  
Official verdict: **FAIL — metrics pass, evidence provenance does not**

## Executive result

The round-2 numerical checks pass: disposition agreement is 27/30 (90%), there are no unsafe
border/complete transitions, and the median mask IoU is 1.0 over eight complete/complete pairs.
The gate itself does not pass because the round-2 decisions file contains neither reviewer identity
nor review timestamps. The required washout also cannot be verified. In addition, two first-pass
`complete` cases changed to `ambiguous` and have not yet been removed from the 377-mask bootstrap
set. T01 and T02 therefore remain blocked.

This is single-operator test-retest evidence. Inter-rater agreement was not measured.

## R02 — passed

Independent parsing and canonical schema validation confirmed:

- six wells, each with 300 first-pass decisions and 300 `reviewer_01` review-log rows;
- six valid overlap-safe `InstanceSet` exports;
- 377 complete, 31 border-truncated, 839 ambiguous, and 553 rejected decisions, totaling 1,800;
- every stored 16-character hash prefix in `six_well_snapshot.json` matches the corresponding full
  SHA-256 calculated by Codex.

Full file hashes:

| Well | Decisions SHA-256 | Instances SHA-256 | Review log SHA-256 |
|---|---|---|---|
| 32_C08_br223_igf1r | `49a68c7b7240978882f8f756e7ed8e630bb8fed063efd51218ba5d8238ebe5e6` | `e5ed8f3ba48f9fe919d0788b19d4b82c1f9e2df3b3fef1cc7d3162e7249808c0` | `d7f4d4903445fce342bc1f38d31de960ca1b73102a19a8842d5b6fb4d11bacce` |
| 19_B06_act104_trka | `d38ac9ecac30acbb5d3d8dfe7b631c77037e5078dd7c772a523eee3d1023646d` | `b9085fba642a20097a77fef752014f078c6f5f47fcb4a28ec480b84e6e0d566f` | `8431ef7d14252b5489d8d4c5eaeefd9ad5ff9d2c1a71d5e91b9d3f918d68ca02` |
| 22_B03_act104_egfrc | `88c0baee70602e37a8ac7c723b0836f8d8600a9443c405122f9ec4e2eea5e964` | `b5b0b7b3e92508d9be256e83a0834e030a63940bbfa90d2fcbd2829298166394` | `594e5e4c306e417d232029f2e0dfa973f1729f21bbf1b1811882c29c080c6825` |
| 29_C05_br223_egfrc | `b19a018218478b6fc2513880807ba3919180ef7d45ea871f703fcd46264af583` | `3d8aa6b7f1af3a161cb769756ba54c60f646085e0a1b2e8d2cc8c8180d799283` | `42f3a6f033efc0607da0bfdc730e1038406a5c898b6fead4f875a16ab3bb931c` |
| 33_C09_br223_trka | `a7c4eb463e835262c611e0208fe23cd850159d5fb5a22a58f172a3f4c4cc8b74` | `775918f98b2c15faae84febca6da5c0a45bf618458a9a1ba9f4e78218355cf15` | `65c2750167785738db3960543f333e7c66a0f8b04956e6574cd60f1df741f261` |
| 23_B02_ctrl | `2093a3d1a60b5b5bffdc35eaa78a47904fe7d37be79f27005b803fcf509aba56` | `6272e5df1c87b82b7af743750f04e84de9a81182883a50f94f7a3f8b99cd9a65` | `b60845171517c10facb2e2e0568215ccd80ecadb6145eb5fa8a46269302bbd9e` |

Snapshot SHA-256: `5171286b5bcb153ad45cfe5db7ae532c4f6158a4f6fcaebbc25ad2649dc36994`.

## R03 — passed

The rebuilt CSV contains exactly the multiset of current accept/reject decisions: 961 unique rows,
408 accepts and 553 rejects, with no stale or duplicate C08 entry.

- CSV: `a1781a4cdae5c881a0179ce57575c5a2107dc7314c1a717ff85173dae50c2fe4`
- Model: `a0129ad36f9ec941974b28c129359373398a74c2910923dd080a064624dd716d`
- Model summary: `89bb52c80d003de1910cbc95537aee6c8e6cfcb3440d711300ce82a575154736`
- Recorded training accuracy: 0.889. This is in-sample triage accuracy, not segmentation evidence.

## Round-2 metric audit

The key has exactly 30 unique cases: 10 complete, 5 border, 10 ambiguous, and 5 reject, covering
all six wells with no overlap with round 1. Every key entry matches its source first-pass decision.

| Criterion | Result | Metric verdict |
|---|---:|---|
| Disposition agreement | 27/30 = 90% | Pass |
| Unsafe border/complete transitions | 0 | Pass |
| Complete/complete pairs | 8 | Pass |
| Pair IoUs | 1, 1, 1, 1, 1, 1, 1, 0.78655 | — |
| Median mask IoU | 1.0 | Pass |

The eight IoUs compare each first-pass canonical complete mask against the round-2 unedited
proposal reconstructed from that well's `starting_labels.tif`. Seven are exact matches. The eighth,
19_B06 `myotube_0045`, has IoU 0.78655; the median remains 1.0.

Disagreements:

| Case | Source | First | Second | Required handling |
|---|---|---|---|---|
| case_02 | 19_B06 / myotube_0377 | complete | ambiguous | Exclude or relabel ambiguous before T01 |
| case_24 | 22_B03 / myotube_0321 | complete | ambiguous | Exclude or relabel ambiguous before T01 |
| case_25 | 22_B03 / myotube_0113 | ambiguous | reject | Already non-trainable |

The handoff's statement that all three disagreements were first-pass complete-to-ambiguous is not
supported by the files. Exactly two are complete-to-ambiguous; the third is ambiguous-to-reject.

## Why the official gate fails

`blind_repeat2.decisions.json` records action, note, features, and edit state, but no reviewer and no
UTC review time, either per decision or at session level. The private key supplies source lineage,
but it cannot supply who performed the second review or when. Filesystem modification time is not
accepted as biological-review provenance. Consequently the required washout interval is also not
demonstrable.

Recovery is bounded: fix the blind export to require a stable reviewer ID and UTC time, make the
two complete-to-ambiguous exclusions binding, then run the plan's targeted 10-case recheck. Do not
start T01/T02 and do not repeat the retired 100-task pilot.

## Verification commands

- Reference `blind-compare`: 90%, eight complete pairs, zero border inconsistencies.
- `python -m pytest annotation_tools/tests model_labs/tests -q`: 23 passed.
- `python -m pytest PrecisionMyotube/tests -q`: 31 passed.

Canonical result: `PrecisionMyotube/HUMAN/g_so1_result.json`.

## Remediation addendum

The provenance export defect was subsequently fixed and the expanded annotation/model-lab suite
passes 24 tests. Codex independently validated the staged 10-case recovery: correct 4/2/3/1
strata, exact source lineage, zero overlap with rounds 1 and 2, `reviewer_01`, no source IDs or
learned per-case suggestions in the blind HTML, and session/export/per-decision timestamp wiring.

The two complete-to-ambiguous cases are now bound to `training_exclude.json` (SHA-256
`b15492c167c8555dd8d306db5285792eea5ca6447cdc935268aa160d7ff847fb`), leaving 375 eligible
complete masks. The recovery page is approved but must not be served or opened before
`2026-07-29T02:23:02.904447Z` (July 28, 2026 at 7:23 PM PDT); July 29 is recommended. The gate
remains failed until that review and its evidence are validated. See
`PrecisionMyotube/HUMAN/G_SO1_RECHECK_PROTOCOL.md`.

## Project-owner development override

The project owner subsequently directed the agents not to pause implementation for this evidence
gap. G-SO1 remains incomplete as repeatability evidence, but is no longer a development gate. T01
was built immediately with 375 eligible real masks, 40 real correction pairs, and 2,290 eligible
synthetic pairs; 12 synthetic derivatives of the two excluded masks were also removed. Formal
claims must still disclose the incomplete repeatability evidence.
