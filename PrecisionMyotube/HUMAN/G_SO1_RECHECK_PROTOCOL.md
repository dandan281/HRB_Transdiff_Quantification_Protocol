# G-SO1 targeted recheck protocol

Status: approved but optional/deferred by the project owner; not a development blocker  
Purpose: remediate the round-2 provenance defect under the version-2 plan's targeted 10-case rule

## Approved design

The staged recheck is approved:

- 10 unique cases selected with seed `20260724`;
- 4 first-pass complete, 2 border, 3 ambiguous, and 1 reject;
- complete/ambiguous cases are concentrated in the 60–160 µm decision boundary;
- all source actions, border flags, and IDs match their frozen first-pass decisions;
- zero overlap with blind-repeat rounds 1 and 2;
- blind HTML contains only `case_01` through `case_10`, no real instance IDs or well names, no
  learned per-case suggestions, and neutralized priors;
- reviewer is fixed to `reviewer_01`;
- export records session start, export time, and a per-decision UTC timestamp.

Four wells rather than all six are represented. This is acceptable because this is a targeted
provenance/rule recheck, not a replacement for the all-six-well 30-case round-2 sample.

Artifacts:

- key SHA-256: `8784d3ec54f76f59df67ce5a8c911b9df43187d1eb2a26d15922e1abc939ce7f`
- HTML SHA-256: `0377ae15b3a2036f0ceaac7f46c7965eca4091e8161b1f92a8391be76a06251f`

## Washout window

The conservative anchor is the six-well snapshot freeze at
`2026-07-22T02:23:02.904447Z`. Seven complete days end at:

- earliest UTC session start: `2026-07-29T02:23:02.904447Z`;
- earliest Pacific session start: Tuesday, July 28, 2026 at 7:23:02 PM PDT;
- recommended simple operating date: Wednesday, July 29, 2026.

Do not serve or open the page before the earliest time. Opening it creates the recorded session
start and exposes the cases. Serving on July 29 is preferred to avoid a boundary-time mistake.

## Serving and operator rules

1. Serve **only** `blind_recheck.html`. Keep `blind_recheck.key.json` and
   `training_exclude.json` outside the web server's document root.
2. The operator must explicitly click a disposition for every case, including cases left
   `Ambiguous`; defaults with `decided_at: null` do not count.
3. Export once all 10 are complete. Do not edit the JSON manually.
4. Preserve the exported file unchanged for Codex validation.

## Recovery verdict rules

The recheck passes the remediation only if all are true:

- reviewer is exactly `reviewer_01`;
- session start is no earlier than the washout deadline;
- all 10 decisions have non-null ISO timestamps within the session/export interval;
- at least 9 of 10 dispositions agree with the source call;
- no unsafe border/complete transition occurs;
- every new disagreement is ambiguous or enters an explicit exclusion list;
- the binding `training_exclude.json` remains unchanged with SHA-256
  `b15492c167c8555dd8d306db5285792eea5ca6447cdc935268aa160d7ff847fb`.

The original round-2 median-IoU result (1.0 over eight complete/complete pairs) remains the mask
criterion. This 10-case recovery is not a replacement experiment and is not required to create a
new eight-pair denominator. Inter-rater agreement remains unmeasured.

The exclusion manifest is sufficient for the G-SO1 disagreement rule. T01 must subsequently
consume that manifest and reproduce 375 trainable complete masks before T01 itself can complete;
this is not a circular prerequisite for the G-SO1 recovery verdict.
