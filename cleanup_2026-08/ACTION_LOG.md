# Cleanup 2026-08 — ACTION LOG (append-only)

Protocol: `C:\Users\liqig\.claude\plans\check-this-handoff-review-giggly-snail.md` Part 5.
Every entry: `#NNN | date | plan item | files touched | command(s) | why | tests before→after | rollback handle`.
Nothing is truly deleted: files move to `cleanup_2026-08/quarantine/<original relative path>`; in-file edits roll back via the per-entry commit. Sealed artifacts (`PrecisionMyotube/runs/`, `annotation_work/bootstrap_v1`, `model_labs/classical/_runs/`, `Conversion_Efficiency/` result folders) are read-only throughout.

---

#001 | 2026-08-12 | Part5 setup | cleanup_2026-08/{baselines,quarantine}/ created | `mkdir -p`; `git count-objects -vH`, `git status --porcelain`, `git for-each-ref`, `git fsck --connectivity-only --unreachable` saved to baselines/ | audit-trail infrastructure required by user before any change | tests n/a (no code touched) | rollback: delete folder (contains only snapshots)
  - Baseline facts: 9,475 loose objects / 21.69 GiB / 0 packs; **9,283 objects unreachable** (aborted `git add` of raw data); 5 `tmp_obj_*` garbage files; HEAD 0322ebf, branch main, remote origin present.

#002 | 2026-08-12 | Tier0-2 | `.gitignore` (root) | Edit: added Cleanup-2026-08 block — Conversion_Efficiency plate result folders, Plate9_C6C2_QTFCs/, CE zips, New_Quantif dbs_cache//myotube//nuclei/ + well-prefixed overlay PNGs + Benchmark myotube*/ renders, `*.npy`, `*.npz`, `tmp/`, `output/`, `model_labs/**/_runs/`, `model_labs/_proof/`, `competitors/MyoFuse/upstream/` | ~9 GB of regenerable results and one nested git clone were un-ignored; a `git add -A` would have committed them (the 21.69 GiB .git is the fossil of exactly that accident) | tests n/a (no code touched) | rollback: checkpoint commit (entry #004)
  - Dry-run verified after edits: would-be-staged set fell from 487 files (incl. ~180 MB of overlay PNGs) to **406 files / 11.7 MB**, all code/docs/small-JSON. Code inside `New_Quantif_*` and `model_labs/` remains tracked; only data subfolders are ignored.

#003 | 2026-08-12 | Tier0-3 | `annotation_tools/.gitignore` | Edit: `qc_review/data/` + `qc_review/models/` → `**/qc_review/data/` + `**/qc_review/models/`; added `.pytest_tmp/` | slash-containing patterns were anchored to `annotation_tools/` and never matched the real paths at `annotation_tools/annotation_tools/qc_review/...` — `accept.csv`/`accept.joblib` were one `git add -A` from being committed against the file's stated intent | tests n/a | rollback: checkpoint commit (entry #004)

#004 | 2026-08-12 | Part5 Step 0 | whole tree | `git checkout -b cleanup-2026-08 && git add -A && git commit` | checkpoint commit — first snapshot of ~48 dirty paths + all untracked code since 0322ebf; every artifact SHA cited in coordination/reports previously bound to an uncommitted tree; user's retrievability requirement | tests: 496 expected baseline (run recorded in #005) | rollback: `git checkout main` (main is untouched; commit is additive on its own branch)

#005 | 2026-08-12 | Tier0-1 | `.git/objects` | baselines saved (#001), then `git gc` WITHOUT `--prune=now` | pack the 192 reachable objects; the 9,283 unreachable loose objects (21.6 GiB) age out on git's default 2-week grace period — reversible until then; immediate prune only on explicit user re-confirmation | tests n/a | rollback (≤2 weeks): objects still on disk until expiry; after-state in baselines/git-count-objects_after.txt
