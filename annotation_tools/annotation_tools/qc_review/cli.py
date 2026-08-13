"""QC review CLI: build the page, train the model from your decisions, apply them.

    build   annotation package -> review.html (with model suggestions if trained)
    train   decisions.json(s)  -> models/accept.joblib + model_summary.json
    apply   decisions.json      -> canonical InstanceSet JSON (accepted = reviewed complete)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from . import model as M
from .pipeline import build_cases
from .page import build_page


def _read_tif(path: Path):
    import tifffile
    return np.asarray(tifffile.imread(str(path)))


def _load_package(package_dir: Path):
    readme = {}
    rp = package_dir / "README.json"
    if rp.is_file():
        readme = json.loads(rp.read_text(encoding="utf-8"))
    labels = _read_tif(package_dir / "starting_labels.tif")
    fiber = _read_tif(package_dir / "fiber_raw16.tif")
    terr_path = package_dir / "semantic_territory.tif"
    territory = _read_tif(terr_path) if terr_path.is_file() else None
    dapi_path = package_dir / "dapi_raw16.tif"
    dapi = _read_tif(dapi_path) if dapi_path.is_file() else None
    stem = readme.get("image_id", package_dir.name)
    pixel_um = float(readme.get("pixel_um", 0.6493))
    return stem, labels, fiber, territory, pixel_um, dapi


def cmd_build(args):
    pkg = Path(args.package)
    stem, labels, fiber, territory, pixel_um, dapi = _load_package(pkg)
    cases = build_cases(labels, fiber, pixel_um, territory, dapi=dapi,
                        thumb_px=args.thumb_px, edit_px=args.edit_px,
                        max_cases=args.max_cases)
    total = int(np.unique(labels[labels > 0]).size)
    out = args.out or str(pkg / "review.html")
    path = build_page(stem, cases, total, out, pixel_um=pixel_um, reviewer=getattr(args, "reviewer", "") or "")
    print(json.dumps({"review_html": path, "cases": len(cases), "total": total,
                      "model": M.load_summary().get("status", "cold")}, indent=2))


def cmd_train(args):
    os.makedirs(M.DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(M.MODEL_PATH), exist_ok=True)
    # accumulate labeled rows from every supplied decisions.json into data/accept.csv
    rows = _load_existing_rows()
    for dpath in args.decisions:
        payload = json.loads(Path(dpath).read_text(encoding="utf-8"))
        stem = payload.get("stem", "")
        for cid, d in payload.get("decisions", {}).items():
            label = M.LABEL_FOR.get(d.get("action"))
            if label is None:                       # ambiguous / unknown -> not a label
                continue
            feats = d.get("features")
            if not feats:
                continue
            rows[(stem, cid)] = (M.features_vector(feats), label)
    _write_rows(rows)

    records = list(rows.values())
    pipe, info = M.fit(records)
    Path(M.SUMMARY).write_text(json.dumps(info, indent=2), encoding="utf-8")
    if pipe is not None:
        import joblib
        joblib.dump(pipe, M.MODEL_PATH)
    elif os.path.exists(M.MODEL_PATH):
        os.remove(M.MODEL_PATH)                     # under-data -> don't serve a stale model
    print(json.dumps(info, indent=2))


def _load_existing_rows():
    rows = {}
    csv_path = os.path.join(M.DATA_DIR, "accept.csv")
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                vec = [float(r[k]) for k in M.FEATURE_KEYS]
                rows[(r["stem"], r["id"])] = (vec, int(r["label"]))
    return rows


def _write_rows(rows):
    csv_path = os.path.join(M.DATA_DIR, "accept.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "id", *M.FEATURE_KEYS, "label"])
        for (stem, cid), (vec, label) in rows.items():
            w.writerow([stem, cid, *vec, label])


def _decode_rowmajor(rle: dict) -> np.ndarray:
    h, w = int(rle["h"]), int(rle["w"])
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        c = int(c)
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(h, w)


def _place_full(edit_mask: np.ndarray, geom: dict, H: int, W: int) -> np.ndarray:
    """Place one edited crop mask back into full-field coordinates."""
    from PIL import Image
    src_h, src_w = int(geom["src_h"]), int(geom["src_w"])
    r0, c0 = int(geom["origin"][0]), int(geom["origin"][1])
    small = np.asarray(
        Image.fromarray((edit_mask.astype(np.uint8) * 255)).resize((src_w, src_h), Image.NEAREST)) > 127
    full = np.zeros((H, W), dtype=bool)
    full[r0:r0 + src_h, c0:c0 + src_w] = small[:min(src_h, H - r0), :min(src_w, W - c0)]
    return full


def _edited_masks(d: dict, H: int, W: int) -> list[np.ndarray]:
    """All edited label masks for a case (a split yields more than one)."""
    geom = d["geom"]
    rles = d.get("labels_rle")
    if rles is None and d.get("mask_rle"):        # backward compat: single mask
        rles = [d["mask_rle"]]
    out = []
    for rle in (rles or []):
        m = _place_full(_decode_rowmajor(rle), geom, H, W)
        if m.any():
            out.append(m)
    return out


def _norm8(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).__mul__(255).astype(np.uint8)


def _reason_of(orig: np.ndarray, corr: np.ndarray, n_labels: int) -> str:
    """Python mirror of the page's correctionInfo() so backfilled pairs get the same label."""
    orig_n = int(orig.sum())
    added = int((corr & ~orig).sum())
    removed = int((~corr & orig).sum())
    if n_labels > 1:
        return "split"
    if added > orig_n * 0.15 and added >= removed:
        return "too_short"
    if removed > orig_n * 0.15 and removed > added:
        return "spillover"
    return "reshape"


def cmd_export_corrections(args):
    """Materialise (proposal, human-correction) pairs — Idea 2 — incl. backfill of
    pre-capture edits (reconstruct the original proposal mask from starting_labels)."""
    from PIL import Image
    pkg = Path(args.package)
    stem, labels, fiber, territory, pixel_um, dapi = _load_package(pkg)
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["decisions"]
    out_dir = Path(args.out or (pkg.parent / "corrections"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, n, backfilled = [], 0, 0
    for rid, d in decisions.items():
        if not d.get("edited") or not d.get("geom") or not d.get("labels_rle"):
            continue
        geom = d["geom"]
        eh, ew = int(geom["edit_h"]), int(geom["edit_w"])
        r0, c0 = int(geom["origin"][0]), int(geom["origin"][1])
        sh, sw = int(geom["src_h"]), int(geom["src_w"])

        corr = np.zeros((eh, ew), dtype=bool)                       # human-corrected union
        for rle in d["labels_rle"]:
            corr |= _decode_rowmajor(rle)
        n_labels = len(d["labels_rle"])

        if d.get("original_rle"):                                   # captured at edit time
            orig = _decode_rowmajor(d["original_rle"])
        else:                                                       # BACKFILL from starting_labels
            try:
                lid = int(rid.split("_")[1])
            except (IndexError, ValueError):
                continue
            crop = labels[r0:r0 + sh, c0:c0 + sw] == lid
            orig = np.asarray(Image.fromarray((crop.astype(np.uint8) * 255)).resize(
                (ew, eh), Image.NEAREST)) > 127
            backfilled += 1

        fcrop = np.asarray(Image.fromarray(_norm8(fiber[r0:r0 + sh, c0:c0 + sw])).resize(
            (ew, eh), Image.BILINEAR))
        payload = {"fiber": fcrop, "proposal": orig.astype(np.uint8), "corrected": corr.astype(np.uint8)}
        if dapi is not None:
            payload["dapi"] = np.asarray(Image.fromarray(_norm8(dapi[r0:r0 + sh, c0:c0 + sw])).resize(
                (ew, eh), Image.BILINEAR))
        np.savez_compressed(out_dir / f"{stem}__{rid}.npz", **payload)

        reason = d.get("reason") or _reason_of(orig, corr, n_labels)
        rows.append({"id": rid, "stem": stem, "reason": reason,
                     "added_px": int((corr & ~orig).sum()), "removed_px": int((~corr & orig).sum()),
                     "proposal_px": int(orig.sum()), "corrected_px": int(corr.sum()),
                     "n_labels": n_labels, "backfilled": not bool(d.get("original_rle")),
                     "action": d.get("action"), "features": d.get("features"),
                     "npz": f"{stem}__{rid}.npz"})
        n += 1

    manifest = out_dir / f"{stem}.corrections.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "manifest": str(manifest), "pairs": n,
                      "backfilled": backfilled, "captured": n - backfilled}, indent=2))


def cmd_apply(args):
    from .._schema_bridge import InstanceRecord, InstanceSet, encode_sparse_positions
    pkg = Path(args.package)
    stem, labels, fiber, territory, pixel_um, dapi = _load_package(pkg)
    H, W = labels.shape
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["decisions"]

    valid_ids = {int(v) for v in np.unique(labels[labels > 0])}
    records, acc, amb, rej, splits, bt, log_rows = [], 0, 0, 0, 0, 0, []
    letters = "abcdefghij"
    for rid, d in decisions.items():                 # iterate decisions, not the label image
        action = d.get("action", "ambiguous")
        if action in ("reject", "split"):
            rej += action == "reject"
            log_rows.append({"id": rid, "action": action, "status": "rejected",
                             "reviewer": args.reviewer})
            continue
        if d.get("edited") and d.get("geom") and (d.get("labels_rle") or d.get("mask_rle")):
            masks = _edited_masks(d, H, W)           # edited / browser-split child card
            if len(masks) > 1:
                splits += 1
        else:                                        # unedited proposal -> original mask by id
            try:
                lid = int(rid.split("_")[1])
            except (IndexError, ValueError):
                continue
            if lid not in valid_ids:
                continue
            masks = [labels == lid]
        if not masks:
            continue
        multi = len(masks) > 1
        for i, mask in enumerate(masks):
            ys, xs = np.nonzero(mask)
            if not ys.size:
                continue
            iid = f"{rid}_{letters[i]}" if multi else rid
            rle = encode_sparse_positions((H, W), ys.astype(np.int64) + xs.astype(np.int64) * H)
            if action == "accept":
                if not args.reviewer:
                    raise SystemExit("--reviewer is required to accept proposals as reviewed")
                # A border-touching fibre cannot be measured full-length: mark it
                # border_truncated (reviewed, but NOT authoritative-complete) so the
                # training exporter excludes it instead of learning a truncated length.
                tb = float((d.get("features") or {}).get("touches_border", 0)) >= 0.5
                status = "border_truncated" if tb else "complete"
                records.append(InstanceRecord(id=iid, status=status, rle=rle, source="qc_review",
                                              reviewed=True, notes=d.get("note", "")))
                bt += tb
                acc += not tb
                log_rows.append({"id": iid, "action": "accept", "status": status,
                                 "reviewer": args.reviewer, "touches_border": bool(tb)})
            else:
                records.append(InstanceRecord(id=iid, status="ambiguous", rle=rle, source="qc_review",
                                              reviewed=False, notes=d.get("note", "")))
                amb += 1
                log_rows.append({"id": iid, "action": "ambiguous", "status": "ambiguous",
                                 "reviewer": args.reviewer})

    from datetime import datetime, timezone
    prov = {"reviewer": args.reviewer, "tool": "qc_review.apply",
            "decisions_file": os.path.basename(args.decisions),
            "reviewed_at": datetime.now(timezone.utc).isoformat()}
    inst = InstanceSet((H, W), stem, records, provenance=prov)
    inst.validate()
    out = args.out or str(pkg / f"{stem}.qc.instances.json")
    inst.save(out)
    log_path = os.path.splitext(out)[0] + ".review_log.jsonl"     # reviewer-linked lineage
    with open(log_path, "w", encoding="utf-8") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"instances": out, "review_log": log_path, "reviewer": args.reviewer,
                      "accepted_complete": acc, "border_truncated": bt, "ambiguous": amb,
                      "rejected": rej, "split_proposals": splits,
                      "total_instances": len(records)}, indent=2))


_STRATA_TARGETS = {"complete": 10, "border": 5, "ambiguous": 10, "reject": 5}   # SO01 = 30 cases


def _stratum_of(d: dict) -> str | None:
    act = d.get("action")
    tb = float((d.get("features") or {}).get("touches_border", 0)) >= 0.5
    if act == "reject":
        return "reject"
    if act == "ambiguous":
        return "ambiguous"
    if act == "accept":
        return "border" if tb else "complete"
    return None


def cmd_blind_repeat(args):
    """SO01: select 30 first-pass cases (stratified, seeded) across wells and build a
    BLIND second-pass page — random order, no earlier call shown, no model hint,
    conservative default. Writes a private key mapping blind id -> real case."""
    import random
    rng = random.Random(args.seed)
    strata = {k: [] for k in _STRATA_TARGETS}
    packages = {}
    for pkgdir in args.package:
        pkg = Path(pkgdir)
        stem, labels, fiber, territory, pixel_um, dapi = _load_package(pkg)
        dpath = pkg / f"{stem}.decisions.json"
        if not dpath.is_file():
            continue
        packages[stem] = dict(labels=labels, fiber=fiber, territory=territory,
                              pixel_um=pixel_um, dapi=dapi)
        for rid, d in json.loads(dpath.read_text(encoding="utf-8"))["decisions"].items():
            s = _stratum_of(d)
            if s is None:
                continue
            try:
                lid = int(rid.split("_")[1])
            except (IndexError, ValueError):
                continue
            strata[s].append({"stem": stem, "rid": rid, "lid": lid, "stratum": s,
                              "first_action": d.get("action"),
                              "first_touches_border": (d.get("features") or {}).get("touches_border", 0),
                              "length": float((d.get("features") or {}).get("length_um", 0))})

    exclude = set()
    for kpath in (args.exclude or []):                 # don't re-show already-seen cases
        for v in json.loads(Path(kpath).read_text(encoding="utf-8"))["key"].values():
            exclude.add((v["well"], v["real_id"]))
    targets = dict(_STRATA_TARGETS)
    if args.targets:                                   # e.g. "complete=4,border=2,ambiguous=3,reject=1"
        targets = {p.split("=")[0].strip(): int(p.split("=")[1]) for p in args.targets.split(",")}
    lo = args.length_min if args.length_min is not None else 0.0
    hi = args.length_max if args.length_max is not None else 1e12
    for s in strata:
        pool = [p for p in strata[s] if (p["stem"], p["rid"]) not in exclude]
        if s in ("complete", "ambiguous"):             # focus the fuzzy classes on the length band
            pool = [p for p in pool if lo <= p["length"] <= hi]
        strata[s] = pool

    picked = []
    for s, n in targets.items():
        pool = strata[s][:]
        rng.shuffle(pool)
        if len(pool) < n:
            print(json.dumps({"warning": f"stratum {s}: only {len(pool)} available, wanted {n}"}))
        picked += pool[:n]
    rng.shuffle(picked)

    by_well = {}
    for p in picked:
        by_well.setdefault(p["stem"], set()).add(p["lid"])
    case_lookup = {}
    for stem, ids in by_well.items():
        pk = packages[stem]
        for c in build_cases(pk["labels"], pk["fiber"], pk["pixel_um"], pk["territory"],
                             dapi=pk["dapi"], thumb_px=args.thumb_px, edit_px=args.edit_px,
                             only_ids=ids):
            case_lookup[(stem, c["id"])] = c

    cases, key = [], {}
    for i, p in enumerate(picked, start=1):
        rc = case_lookup.get((p["stem"], f"myotube_{p['lid']:04d}"))
        if rc is None:
            continue
        bid = f"case_{i:02d}"
        c = dict(rc); c["id"] = bid; c["prior"] = 0.0    # neutralise; blind UI hides it anyway
        cases.append(c)
        key[bid] = {"well": p["stem"], "real_id": p["rid"], "stratum": p["stratum"],
                    "first_action": p["first_action"], "first_touches_border": p["first_touches_border"]}

    out = args.out
    build_page("blind_repeat", cases, len(cases), out, blind=True, blind_note=args.note or "",
               reviewer=args.reviewer)
    keyp = args.key or (str(Path(out).with_suffix("")) + ".key.json")
    Path(keyp).write_text(json.dumps({"seed": args.seed, "n": len(cases), "key": key}, indent=2),
                          encoding="utf-8")
    from collections import Counter
    print(json.dumps({"page": out, "key": keyp, "n_cases": len(cases),
                      "strata": dict(Counter(v["stratum"] for v in key.values()))}, indent=2))


def cmd_blind_compare(args):
    """Reference agreement report of a blind second pass vs the private key.
    Official G-SO1 (incl. mask IoU) is Codex-owned; this is a self-check."""
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))["key"]
    dec = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["decisions"]
    coarse = {"complete": "accept", "border": "accept", "reject": "reject", "ambiguous": "ambiguous"}
    n = agree = complete_pairs = 0
    disagreements, border_flags, missing = [], [], []
    for bid, meta in key.items():
        second = dec.get(bid)
        if second is None:
            missing.append(bid)
            continue
        n += 1
        d1 = meta["stratum"]
        d2 = _stratum_of(second) or "?"
        if coarse.get(d1) == coarse.get(d2):
            agree += 1
        else:
            disagreements.append({"case": bid, "first": d1, "second": d2})
        if d1 == "complete" and d2 == "complete":
            complete_pairs += 1
        if coarse.get(d1) == "accept" and coarse.get(d2) == "accept" and (d1 == "border") != (d2 == "border"):
            border_flags.append({"case": bid, "first": d1, "second": d2})
    pct = round(100 * agree / n, 1) if n else 0.0
    print(json.dumps({"n_compared": n, "missing_second_pass": missing,
                      "disposition_agreement_pct": pct, "target": ">=85%", "meets_agreement": pct >= 85,
                      "complete_complete_pairs": complete_pairs, "iou_denominator_target": ">=8",
                      "border_inconsistencies": border_flags, "disagreements": disagreements,
                      "note": "reference only; official G-SO1 incl. median IoU>=0.80 on complete pairs is Codex-owned"},
                     indent=2))


def cmd_build_retriage(args):
    """Build bounded re-triage batches over the first-pass `ambiguous` proposals."""
    import collections
    import datetime as _dt

    from .retriage import PROMOTES_TO_TARGET, batch, build_queue
    from .retriage_page import build_retriage_page

    packages = [Path(p) for p in args.package]
    print(f"scanning {len(packages)} packages for first-pass ambiguous proposals...")
    queue = build_queue(packages, thumb_px=args.thumb_px, edit_px=args.edit_px,
                        progress=lambda stem, n: print(f"  {stem:24s} {n:4d} cases"))
    if not queue:
        raise SystemExit("no ambiguous proposals found")

    batches = batch(queue, args.batch_size)
    started = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for index, chunk in enumerate(batches, start=1):
        batch_id = f"retriage_b{index:02d}"
        promotable = sum(1 for c in chunk if c["machine_category"] in PROMOTES_TO_TARGET)
        path = build_retriage_page(
            chunk, out_dir / f"{batch_id}.html", batch_id=batch_id,
            reviewer=args.reviewer, session_started_at=started,
            batch_index=index, batch_total=len(batches),
            note=args.note or "")
        written.append({"batch_id": batch_id, "html": path, "n": len(chunk),
                        "machine_promotable": promotable,
                        "median_length_um": round(float(np.median(
                            [c["features"]["length_um"] for c in chunk])), 1)})

    tally = collections.Counter(c["machine_category"] for c in queue)
    index_payload = {
        "schema": "retriage_index.v1",
        "reviewer": args.reviewer,
        "session_started_at": started,
        "n_cases": len(queue), "n_batches": len(batches),
        "batch_size": args.batch_size,
        "order": "descending expected promotion yield; stop after any batch",
        "machine_pre_classification": dict(tally),
        "machine_promotable": sum(tally[k] for k in PROMOTES_TO_TARGET),
        "batches": written,
        "note": "Machine categories are suggestions only. A card that is never "
                "explicitly confirmed exports decided_at=null and is excluded "
                "from training promotion.",
    }
    (out_dir / "retriage_index.json").write_text(
        json.dumps(index_payload, indent=2), encoding="utf-8")
    print(json.dumps(index_payload, indent=2))


def cmd_build_links(args):
    """Build the fragment-linking page from operator-confirmed fragments."""
    import datetime as _dt

    from .link_candidates import (CANDIDATE_LETTERS_MAX, find_link_candidates,
                                  load_fragments)
    from .link_page import CANDIDATE_RGB, build_link_page, render_case

    fragments_by_well = load_fragments(args.round2)
    if not fragments_by_well:
        raise SystemExit("no confirmed `fragment_too_short` cases in the round-2 bank")
    packages = {Path(p).name: Path(p) for p in args.package}

    cases = []
    skipped_no_candidate = 0
    for well in sorted(fragments_by_well):
        package = _resolve_package(packages, well, args.package)
        stem, labels, fiber, _territory, pixel_um, dapi = _load_package(package)
        ids = fragments_by_well[well]
        label_ids = [int(i.split("_")[-1]) for i in ids]
        found = find_link_candidates(labels, label_ids, pixel_um,
                                     gap_um=args.gap_um, cos_min=args.cos_min)
        for fragment_id in ids:
            candidates = found.get(fragment_id, [])[:CANDIDATE_LETTERS_MAX]
            if not candidates:
                skipped_no_candidate += 1
                continue
            fragment_label = int(fragment_id.split("_")[-1])
            candidate_labels = [int(c.candidate_id.split("_")[-1]) for c in candidates]
            img, overlay, _bbox = render_case(fiber, dapi, labels, fragment_label,
                                              candidate_labels, size=args.size)
            cases.append({
                "id": fragment_id, "well": stem,
                "uid": f"{stem}/{fragment_id}",
                "dom_id": f"{stem}__{fragment_id}".replace("/", "_").replace(".", "_"),
                "img": img, "overlay": overlay,
                "candidates": [
                    {"letter": CANDIDATE_RGB[i % len(CANDIDATE_RGB)][0],
                     "rgb": ",".join(str(v) for v in CANDIDATE_RGB[i % len(CANDIDATE_RGB)][1]),
                     "candidate_id": c.candidate_id, "gap_um": c.gap_um,
                     "cos_fragment": c.cos_fragment, "cos_candidate": c.cos_candidate}
                    for i, c in enumerate(candidates)],
            })
        print(f"  {stem:24s} {len(ids):3d} fragments -> "
              f"{sum(1 for c in cases if c['well'] == stem):3d} linkable")

    if not cases:
        raise SystemExit("no fragment had a collinear candidate")
    started = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    out = Path(args.out)
    path = build_link_page(cases, out, batch_id=args.batch_id, reviewer=args.reviewer,
                           session_started_at=started, gap_um=args.gap_um,
                           cos_min=args.cos_min, note=args.note or "")
    summary = {"page": path, "n_fragments_linkable": len(cases),
               "n_candidate_links": sum(len(c["candidates"]) for c in cases),
               "skipped_no_candidate": skipped_no_candidate,
               "gap_um": args.gap_um, "cos_min": args.cos_min, "reviewer": args.reviewer}
    Path(out).with_suffix(".index.json").write_text(json.dumps(summary, indent=2),
                                                    encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_build_active_links(args):
    """Active-learning round: widen the window, serve the least-certain pairs."""
    from .link_active import build_active_round

    packages: dict[str, Path] = {}
    for p in args.package:
        path = Path(p)
        readme = path / "README.json"
        stem = (json.loads(readme.read_text(encoding="utf-8")).get("image_id", path.name)
                if readme.is_file() else path.name)
        packages[stem] = path

    manifest = build_active_round(
        packages, Path(args.round2), Path(args.pairs), Path(args.out),
        reviewer=args.reviewer, batch_id=args.batch_id,
        wide_gap_um=args.gap_um, wide_cos_min=args.cos_min,
        train_gap_um=args.train_gap_um, train_cos_min=args.train_cos_min,
        exclude_jsonls=[Path(p) for p in (args.exclude or [])],
        max_pairs=args.max_pairs, size=args.size)
    print(json.dumps({k: manifest[k] for k in ("batch_id", "n_fragment_cards", "pool")},
                     indent=2))


def cmd_build_junctions(args):
    """Build the junction-splitting page: round 1 of the split-crossing labeling round."""
    import datetime as _dt

    import numpy as np
    import tifffile

    from classical.ridge_graph import TracerParams

    from .junction_page import BRANCH_RGB, build_junction_page, render_case
    from .junction_pairs import ROUND1_REASONS, find_junction_cases

    territory_cache = Path(args.territory_cache)
    bootstrap_dir = Path(args.bootstrap)
    reasons = None if args.reasons == "all" else ROUND1_REASONS

    cases = []
    per_well_counts: dict[str, int] = {}
    for well in args.wells:
        territory = np.load(territory_cache / f"{well}.territory.npy")
        fiber = tifffile.imread(bootstrap_dir / well / "image_fiber.tif")
        dapi_path = bootstrap_dir / well / "image_dapi.tif"
        dapi = tifffile.imread(dapi_path) if dapi_path.is_file() else None

        found, coordinates, _node_ends = find_junction_cases(well, territory, fiber, args.pixel_um,
                                                             TracerParams(), reasons=reasons)
        per_well_counts[well] = len(found)
        for jc in found:
            img, overlay, _bbox = render_case(fiber, dapi, coordinates, jc.branch_ids,
                                              jc.centroid_rc, size=args.size,
                                              radius_um=args.radius_um, pixel_um=args.pixel_um)
            case_id = jc.case_id()
            branches = [{"letter": letter, "rgb": ",".join(str(v) for v in rgb),
                        "branch_id": int(bid), "length_um": length}
                       for (letter, rgb), bid, length in
                       zip(BRANCH_RGB, jc.branch_ids, jc.branch_lengths_um)]
            letters = [b["letter"] for b in branches]
            pairs = [{"key": letters[a] + letters[b], "letters": [letters[a], letters[b]],
                      "branches": [int(jc.branch_ids[a]), int(jc.branch_ids[b])]}
                     for a, b in ((0, 1), (0, 2), (1, 2))]
            cases.append({
                "id": case_id, "well": well, "node": jc.node,
                "uid": f"{well}/{case_id}", "dom_id": f"{well}__{case_id}",
                "img": img, "overlay": overlay,
                "branches": branches, "pairs": pairs,
            })
        del territory, fiber, dapi
        print(f"  {well:22s} {len(found):4d} junctions")

    if not cases:
        raise SystemExit("no junction candidates matched the pool")
    started = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    note = (f"Round 1: {len(cases)} degree-3 junctions across {len(per_well_counts)} wells "
           f"where the classical floor's direction-only pairing is ambiguous (the adopted "
           f"decision sat near the boundary, or direction disagreed with width/intensity "
           f"continuity). See coordination/reports/claude_junction_ambiguity_measurement_"
           f"2026-07-23.md for the full measurement.")
    path = build_junction_page(cases, Path(args.out), batch_id=args.batch_id,
                               reviewer=args.reviewer, session_started_at=started, note=note)
    summary = {"page": path, "n_junctions": len(cases), "per_well": per_well_counts,
              "reviewer": args.reviewer, "reasons_pool": "round1" if reasons else "all"}
    Path(args.out).with_suffix(".index.json").write_text(json.dumps(summary, indent=2),
                                                          encoding="utf-8")
    print(json.dumps(summary, indent=2))


def cmd_train_junction_model(args):
    """Fit the junction classifier from a labeled round export; compare to the
    classical floor's fixed pairing rule."""
    from .junction_model import (classical_floor_decisions, decision_accuracy,
                                 fit_branch_point_model, fit_junction_classifier,
                                 leave_one_well_out_junction_decisions,
                                 ground_truth_decisions, recompute_examples,
                                 select_feature_set, two_stage_decisions)

    recomputed = recompute_examples(args.export, args.territory_cache, args.bootstrap,
                                    pixel_um=args.pixel_um)
    examples, junction_examples = recomputed.pairs, recomputed.junctions
    n_pos = sum(1 for e in examples if e.label == 1)
    n_bp = sum(1 for e in junction_examples if e.label == 1)
    print(f"  {len(examples)} labeled pairs ({n_pos} positive) and "
         f"{len(junction_examples)} junctions ({n_bp} branch points) from "
         f"{len(args.export)} export(s)")

    feature_name, selection = select_feature_set(examples)
    feature_keys = selection["all"][feature_name]["keys"]
    model = fit_junction_classifier(examples, feature_keys)
    print(f"  feature set: {feature_name}  LOWO AUC: {selection['chosen_auc']}  "
         f"({model.fit_info['n_positive']}/{model.fit_info['n']} positive)")
    for name, r in selection["all"].items():
        print(f"    {name:26s} auc={r['auc']}  n={r['n_scored']}  skipped={r['skipped_wells']}")

    truth = ground_truth_decisions(args.export)
    baseline = classical_floor_decisions(args.export, args.territory_cache, pixel_um=args.pixel_um)
    baseline_acc = decision_accuracy(baseline, truth)
    print(f"  classical floor (fixed STRAIGHT_DOT) junction accuracy: {baseline_acc}")

    clf_decisions_raw = leave_one_well_out_junction_decisions(examples, feature_keys)
    clf_decisions = {k: v[0] for k, v in clf_decisions_raw.items()}
    clf_acc = decision_accuracy(clf_decisions, truth)
    print(f"  single-stage argmax (LOWO)    junction accuracy: {clf_acc}")

    gate = fit_branch_point_model(junction_examples)
    two_decisions, gate_info = two_stage_decisions(examples, junction_examples, truth,
                                                   pair_keys=feature_keys)
    two_acc = decision_accuracy(two_decisions, truth)
    print(f"  TWO-STAGE + branch-point gate  junction accuracy: {two_acc}")
    print(f"    gate thresholds per fold: {gate_info['gate_thresholds_per_fold']}")

    out = {
        "exports": [str(p) for p in args.export],
        "n_examples": len(examples), "n_positive": n_pos,
        "n_junctions": len(junction_examples), "n_branch_points": n_bp,
        "feature_selection": selection, "chosen_feature_set": feature_name,
        "fit_info": model.fit_info,
        "branch_point_gate_fit_info": gate.fit_info,
        "baseline_classical_floor": baseline_acc,
        "junction_classifier_lowo_single_stage": clf_acc,
        "junction_classifier_lowo_two_stage": two_acc,
        "gate": gate_info,
        "evidence_class": "development_bootstrap_single_operator_proposal_conditioned",
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if args.model_out:
        import joblib
        Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model_out)
        print(f"  model: {args.model_out}")
    print(f"  summary: {out_path}")


def cmd_build_junction_active_round(args):
    """Active-learning round 2: widen the pool, serve the least-certain junctions."""
    from .junction_active import build_active_round

    manifest = build_active_round(
        args.territory_cache, args.bootstrap, args.prior_export, Path(args.out),
        reviewer=args.reviewer, wells=args.wells, batch_id=args.batch_id,
        pixel_um=args.pixel_um, max_junctions=args.max_junctions,
        size=args.size, radius_um=args.radius_um)
    print(json.dumps({k: manifest[k] for k in ("batch_id", "pool")}, indent=2))


def _control_only_spec(payload: dict, key: dict) -> dict:
    """Predeclare the estimator before the reviewer sees anything.

    Binding rule 6 of the release ruling requires the estimator and interval to be
    fixed in advance, so the round cannot be read whichever way the numbers fall.
    Recorded in the key, which the reviewer never sees.

    The design is stratified, not simple random: equal draws per well over unequal
    well sizes. Averaging the six well rates would silently weight a small well like
    a large one, which is the mean-of-wells mistake the linker recall reporting
    already made once. The population estimate is therefore the well-size-weighted
    mean, with per-well rates reported alongside it and never in place of it.
    """
    sampled: dict[str, int] = {}
    for meta in key.values():
        sampled[meta["well"]] = sampled.get(meta["well"], 0) + 1
    strata = []
    for well_meta in payload["wells"]:
        well = well_meta["well"]
        population = well_meta.get("n_accepted_merge_components")
        drawn = sampled.get(well, 0)
        strata.append({
            "well": well,
            "accepted_merges_in_well": population,
            "sampled": drawn,
            "inclusion_probability": (round(drawn / population, 6)
                                      if population else None),
        })
    total = sum(s["accepted_merges_in_well"] or 0 for s in strata)
    return {
        "design": "stratified by well; equal draws per well, unequal well sizes",
        "strata": strata,
        "accepted_merges_across_six_wells": total,
        "estimator": (
            "population over-merge rate = sum_w (accepted_merges_in_well_w * "
            "different_myotubes_rate_w) / sum_w accepted_merges_in_well_w, i.e. the "
            "well-size-weighted mean of the per-well rates. Report per-well rates "
            "alongside it, never instead of it."
        ),
        "interval": (
            "stratified bootstrap over wells, 10000 resamples, seed 20260731, "
            "resampling whole wells and then merges within a well; report the 2.5th "
            "and 97.5th percentiles"
        ),
        "unresolved_handling": (
            "ambiguous_2d is reported separately as an unresolved fraction and is "
            "excluded from both numerator and denominator of the rate. It is never "
            "counted as a safe merge."
        ),
        "threshold_status": (
            "LOCKED at 0.90. This round must not be used to select or tune it, and no "
            "post-hoc threshold search on these objects is authorised."
        ),
        "prespecified_at_build_time": True,
    }


def cmd_build_over_merge_page(args):
    """Blinded hand review of every candidate over-merge the linker introduces.

    Consumes `model_labs/classical/extract_over_merges.py`'s output directory and
    emits two files that must stay separate:

    * the **page** -- blinded. No well, no label id, no link probability, no
      case/control flag, random order.
    * the **key** -- everything the page withholds, so the export can be scored
      afterwards without ever having shown the reviewer the answer.

    Controls (ordinary merges at the same locked threshold that do not trip the
    over-merge rule) are interleaved so a verdict on the three real cases can be
    read against the same reviewer's behaviour on merges known not to be flagged.
    """
    import datetime as _dt

    from .over_merge_page import (assert_no_separating_field, build_over_merge_page,
                                  count_references_in_view, render_case_panels)

    src = Path(args.cases)
    payload = json.loads((src / "cases.json").read_text(encoding="utf-8"))
    if payload["threshold"] != args.expect_threshold:
        raise SystemExit(f"extracted at threshold {payload['threshold']}, expected "
                         f"{args.expect_threshold}; refusing to build a packet whose "
                         "operating point is not the locked one")

    real = [dict(c, case_kind="over_merge") for c in payload["cases"]]
    pool = [dict(c, case_kind="control") for c in payload["controls"]]
    if args.control_only:
        # The flagged cases are dropped, not hidden. Keeping them would re-enrich the
        # sample with the 3 objects the sparse-reference rule could see, which is the
        # exact bias this round exists to remove.
        if not payload.get("uniform_controls"):
            raise SystemExit(
                "--control-only requires an extraction built with --uniform-controls; "
                "this one matched controls on fragment count, so its sample is not "
                "uniform over accepted merges and cannot estimate a population rate")
        if not pool:
            raise SystemExit("no accepted merges extracted to review")
        real = []
    elif not real:
        raise SystemExit("no over-merge cases to review")

    rng = np.random.default_rng(args.seed)
    arrays_cache: dict[str, dict] = {}

    def _load(well):
        if well not in arrays_cache:
            arr = np.load(src / f"{well}.arrays.npz")
            refs_meta = json.loads((src / f"{well}.references.json").read_text(encoding="utf-8"))
            packed = np.load(src / f"{well}.references.npz")
            references = []
            for rid, meta in refs_meta.items():
                shape = tuple(int(v) for v in packed[f"{rid}__shape"])
                bits = np.unpackbits(packed[rid])[:shape[0] * shape[1]]
                references.append({"id": rid, "bbox": tuple(meta["bbox"]),
                                   "mask": bits.reshape(shape).astype(bool)})
            arrays_cache[well] = {
                "fiber": arr["fiber"], "dapi": arr["dapi"] if "dapi" in arr else None,
                "kept_only": arr["kept_only"], "merged": arr["merged"],
                "references": references}
        return arrays_cache[well]

    # --- control matching on reference density -------------------------------
    # A flagged object sits where two reference masks meet, which is a densely
    # annotated neighbourhood. Controls sampled without regard to that produced
    # 7/7/4 reference masks in view for the real cases against 0-3 for the
    # controls -- perfectly separable without judging any biology. So match.
    def _n_refs(entry):
        data = _load(entry["well"])
        return count_references_in_view(data["references"], data["fiber"].shape,
                                        entry["bbox"], args.pad_px)

    for entry in real + pool:
        entry["_n_refs"] = _n_refs(entry)

    # --- blind repeat, or a fresh round --------------------------------------
    repeat_of = None
    if args.repeat_of:
        # A second pass must show the SAME objects in a DIFFERENT order under DIFFERENT
        # uids, so intra-rater agreement is measurable. The object set is pinned from
        # the first pass's key rather than reproduced from a seed: control selection
        # below consumes the same rng, so changing the order seed would silently change
        # which controls were picked and the two passes would not be comparable.
        first = json.loads(Path(args.repeat_of).read_text(encoding="utf-8"))
        if first.get("threshold") != payload["threshold"]:
            raise SystemExit(f"first pass ran at threshold {first.get('threshold')}, "
                             f"extraction is {payload['threshold']}; not comparable")
        if args.batch_id == first["batch_id"]:
            raise SystemExit("a repeat needs its own --batch-id, or its uids collide "
                             "with the first pass and the reviewer is not blinded")
        by_object = {(e["well"], e["merged_label"]): e for e in real + pool}
        entries, missing = [], []
        for uid, meta in first["key"].items():
            entry = by_object.get((meta["well"], meta["merged_label"]))
            if entry is None:
                missing.append(uid)
                continue
            entries.append(dict(entry, first_pass_uid=uid))
        if missing:
            raise SystemExit(f"{len(missing)} object(s) from {first['batch_id']} are not in "
                             f"this extraction ({missing[:3]}); re-extract with the same "
                             "parameters before repeating")
        repeat_of = first["batch_id"]
        print(f"  blind repeat of {repeat_of}: {len(entries)} objects pinned from its key "
              f"({sum(1 for e in entries if e['case_kind'] == 'over_merge')} flagged)")
    elif args.control_only:
        # No selection happens here. The extraction already drew the sample with equal
        # probability inside each well, and its per-well denominators are what the
        # predeclared estimator weights by. Re-sampling at build time would silently
        # change those inclusion probabilities.
        entries = pool
        by_well: dict[str, int] = {}
        for entry in entries:
            by_well[entry["well"]] = by_well.get(entry["well"], 0) + 1
        print(f"  control-only round: {len(entries)} accepted merges, no flagged cases, "
              f"no density matching")
        for well in sorted(by_well):
            total = next((w.get("n_accepted_merge_components")
                          for w in payload["wells"] if w["well"] == well), None)
            share = f"{by_well[well]}/{total}" if total else f"{by_well[well]}/?"
            print(f"    {well:24s} {share} accepted merges sampled")
    else:
        per_case = max(1, args.controls_per_case)
        chosen: list[dict] = []
        taken: set[tuple] = set()
        for case in real:
            same_well = [c for c in pool
                         if c["well"] == case["well"]
                         and (c["well"], c["merged_label"]) not in taken]
            same_well.sort(key=lambda c: (abs(c["_n_refs"] - case["_n_refs"]), rng.random()))
            for control in same_well[:per_case]:
                taken.add((control["well"], control["merged_label"]))
                chosen.append(control)
        if len(chosen) < per_case * len(real):
            print(f"  NOTE: control pool exhausted -- {len(chosen)} controls for "
                  f"{len(real)} cases, wanted {per_case * len(real)}. Re-extract with a "
                  f"larger --controls to improve matching.")
        entries = real + chosen
        print(f"  matched {len(chosen)} controls to {len(real)} flagged cases on "
              f"reference density (flagged {[c['_n_refs'] for c in real]}, "
              f"controls {sorted(c['_n_refs'] for c in chosen)})")

    order = np.random.default_rng(args.order_seed if args.order_seed is not None
                                  else args.seed).permutation(len(entries))
    cases, kinds, key = [], [], {}
    for n, position in enumerate(order, start=1):
        entry = entries[int(position)]
        data = _load(entry["well"])
        links = [(tuple(p["fragment_endpoint"]), tuple(p["candidate_endpoint"]))
                 for p in entry["accepted_pairs"] if "fragment_endpoint" in p]
        if len(links) != len(entry["accepted_pairs"]):
            raise SystemExit(
                f"{entry['well']}/{entry['merged_label']}: {len(entry['accepted_pairs'])} "
                f"accepted pairs but only {len(links)} carry endpoints. The link panel "
                "would render without a bridge for this object and with one for the "
                "others, which unblinds the packet. Re-extract.")
        panels, crop, n_refs = render_case_panels(
            data["fiber"], data["dapi"], data["kept_only"], data["merged"],
            data["references"], fragment_ids=entry["fragment_ids"],
            merged_label=entry["merged_label"], links=links, bbox=entry["bbox"],
            pad_px=args.pad_px, size=args.size)
        uid = f"{args.batch_id}_{n:03d}"
        cases.append({
            "uid": uid, "n_fragments": entry["n_fragments"],
            "gaps_um": [p["gap_um"] for p in entry["accepted_pairs"] if "gap_um" in p],
            "panels": panels,
        })
        kinds.append(entry["case_kind"])
        key[uid] = {
            "well": entry["well"], "merged_label": entry["merged_label"],
            "case_kind": entry["case_kind"], "fragment_ids": entry["fragment_ids"],
            "accepted_pairs": entry["accepted_pairs"],
            "overlapping_references": entry["overlapping_references"],
            "crop_bbox": list(crop), "n_reference_masks_in_view": n_refs,
            "matched_on_reference_density": entry["_n_refs"],
            **({"first_pass_uid": entry["first_pass_uid"]} if "first_pass_uid" in entry
               else {}),
        }
        print(f"  {uid}  {entry['case_kind']:10s} {entry['n_fragments']} frags, "
              f"{n_refs} reference mask(s) in view")

    ranges = assert_no_separating_field(cases, kinds)
    print("  blinding check passed; displayed-field ranges (flagged vs control):")
    for field, spans in ranges.items():
        print(f"    {field}: {spans['flagged_range']} vs {spans['control_range']}")

    started = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    n_real = sum(1 for v in key.values() if v["case_kind"] == "over_merge")
    if args.control_only:
        # Round 2 section 3: the reviewer never once used `ambiguous_2d`, including on
        # the cases they described as three-dimensional -- they were asserting the
        # occlusion was readable. The vocabulary assumed z-overlap would land in
        # `ambiguous_2d` and it does not. Say so here rather than leave them to choose.
        note = (f"{len(cases)} ordinary accepted merges at the locked threshold "
                f"{payload['threshold']}, sampled with equal probability inside each of "
                f"the six wells. None of them were flagged by any rule; there is no "
                f"hidden group to find. For each one the question is only: is this one "
                f"myotube, or two? If you can see that two fibres overlap in z -- one "
                f"lying above the other -- that is 'distinct myotubes', not 'cannot "
                f"resolve'. Reserve 'cannot resolve in 2D' for the cases you genuinely "
                f"cannot call either way; it is recorded as unresolved and is never "
                f"counted as evidence that a merge was safe. A written reason is "
                f"required on every 'distinct myotubes' call.")
    else:
        note = (f"{len(cases)} merged objects at the locked threshold "
                f"{payload['threshold']}. Blinded: you are not told which are the "
                f"{n_real} flagged objects. An ambiguous verdict is recorded as "
                f"unresolved and is never counted as evidence that a merge was safe.")
    out = build_over_merge_page(cases, args.out, batch_id=args.batch_id,
                                reviewer=args.reviewer, session_started_at=started,
                                threshold=payload["threshold"], note=note)
    key_path = Path(args.key)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps({
        "batch_id": args.batch_id, "built_at": started, "reviewer": args.reviewer,
        "threshold": payload["threshold"],
        "threshold_status": payload["threshold_status"],
        "over_merge_definition": payload["over_merge_definition"],
        "source": str(src), "shuffle_seed": args.seed,
        "order_seed": args.order_seed if args.order_seed is not None else args.seed,
        "repeat_of_batch": repeat_of,
        "n_over_merge_cases": n_real, "n_controls": len(cases) - n_real,
        "note": "DO NOT show this file to the reviewer before they export decisions.",
        **({"control_only_round": _control_only_spec(payload, key)}
           if args.control_only else {}),
        "key": key}, indent=2), encoding="utf-8")
    print(f"\npage: {out}\nkey : {key_path}  ({n_real} flagged, {len(cases)-n_real} controls)")


def _telemetry_summary(rows, decisions):
    """What the reviewer actually looked at, when the instrument recorded it.

    Round 1 shipped no telemetry, so `available: false` there is the honest answer
    rather than a fabricated zero.
    """
    have = [decisions[r["uid"]] for r in rows
            if decisions[r["uid"]].get("ms_on_case") is not None]
    if not have:
        return {"available": False,
                "note": "instrument v1 recorded no dwell data; pace can only be "
                        "inferred from decided_at gaps, which include reading time"}
    secs = sorted(d["ms_on_case_at_decision"] / 1000 for d in have
                  if d.get("ms_on_case_at_decision") is not None)
    ref_seen = [d.get("reference_panel_seen_before_decision") for d in have]
    panels = {}
    for d in have:
        for name, ms in (d.get("panel_dwell_ms") or {}).items():
            panels[name] = panels.get(name, 0) + ms
    return {
        "available": True, "n": len(have),
        "seconds_to_decision": {
            "median": round(float(np.median(secs)), 1) if secs else None,
            "min": round(secs[0], 1) if secs else None,
            "max": round(secs[-1], 1) if secs else None,
            "n_under_5s": sum(1 for s in secs if s < 5)},
        "decided_without_viewing_reference_masks": sum(1 for v in ref_seen if v is False),
        "total_dwell_seconds_by_panel": {k: round(v / 1000, 1) for k, v in
                                         sorted(panels.items(), key=lambda kv: -kv[1])},
    }


def _intra_rater(second_key, second_decisions, first_key, first_export):
    """Same objects, same reviewer, two passes: does the reviewer agree with themselves?

    This is the test that separates "the reviewer was not discriminating" from "the
    over-merge rule is blind to most merges" -- the two readings round 1 could not
    tell apart. Objects are paired through `first_pass_uid` (written by
    `--repeat-of`), never by presentation position.
    """
    first_decisions = first_export["decisions"]
    pairs, unpaired = [], []
    for uid, meta in second_key.items():
        first_uid = meta.get("first_pass_uid")
        if first_uid is None or first_uid not in first_decisions:
            unpaired.append(uid)
            continue
        pairs.append({"second_uid": uid, "first_uid": first_uid,
                      "kind": meta["case_kind"],
                      "well": meta["well"], "merged_label": meta["merged_label"],
                      "first": first_decisions[first_uid]["decision"],
                      "second": second_decisions.get(uid, {}).get("decision")})
    both = [p for p in pairs if p["first"] and p["second"]]
    agree = [p for p in both if p["first"] == p["second"]]
    flips = [p for p in both if p["first"] != p["second"]]

    # chance-corrected: the vocabulary is 3-way but usage is skewed, so report kappa
    # against the observed marginals rather than assuming uniform.
    labels = sorted({p["first"] for p in both} | {p["second"] for p in both})
    n = len(both)
    expected = 0.0
    for label in labels:
        p1 = sum(1 for p in both if p["first"] == label) / n if n else 0
        p2 = sum(1 for p in both if p["second"] == label) / n if n else 0
        expected += p1 * p2
    observed = (len(agree) / n) if n else None
    kappa = ((observed - expected) / (1 - expected)
             if n and expected < 1 else None)
    return {
        "first_pass_batch": first_key["batch_id"], "n_paired": len(pairs),
        "n_scored_both_passes": n, "unpaired": unpaired,
        "agreement": None if observed is None else round(observed, 4),
        "expected_by_chance": round(expected, 4) if n else None,
        "cohens_kappa": None if kappa is None else round(kappa, 4),
        "n_flips": len(flips),
        "flips": [{k: p[k] for k in ("well", "merged_label", "kind", "first", "second")}
                  for p in flips],
        "reading": ("agreement at or near chance means the round-1 verdicts describe the "
                    "reviewer, not the merges, and neither pass can be used as evidence "
                    "about the flagged cases"),
    }


def _confidence_calibration(entries, decisions):
    """Does the linker's own confidence track the human verdict, and which way?

    Reported because round 2 found it runs *backwards*: every object merged at
    P = 1.0000 was called `different_myotubes`. A model whose most confident merges
    are its wrong ones cannot be deployed by raising the threshold.
    """
    rows = []
    for uid, meta in entries.items():
        pairs = meta.get("accepted_pairs") or []
        got = decisions.get(uid, {}).get("decision")
        if not pairs or got not in ("same_myotube", "different_myotubes"):
            continue
        rows.append((max(p["probability"] for p in pairs), got == "same_myotube"))
    same = [p for p, is_same in rows if is_same]
    diff = [p for p, is_same in rows if not is_same]
    if not same or not diff:
        return {"available": False, "note": "needs both verdicts to compare"}
    wins = sum(1 for a in same for b in diff if a > b)
    ties = sum(1 for a in same for b in diff if a == b)
    auc = (wins + 0.5 * ties) / (len(same) * len(diff))
    at_ceiling = [(p, is_same) for p, is_same in rows if p >= 0.99995]
    return {
        "available": True, "n": len(rows),
        "median_probability_called_same": round(float(np.median(same)), 4),
        "median_probability_called_different": round(float(np.median(diff)), 4),
        "auc_probability_predicts_same": round(auc, 3),
        "n_at_probability_1": len(at_ceiling),
        "n_at_probability_1_called_different": sum(1 for _p, s in at_ceiling if not s),
        "reading": ("0.5 = confidence carries no information about correctness; "
                    "below 0.5 = the more confident the merge, the more likely the "
                    "human calls it wrong, so a higher threshold would not help"),
    }


def _note_discipline(rows, decisions):
    """Whether the calls that need a reason carry one."""
    required = [r for r in rows if decisions[r["uid"]].get("note_required")]
    if not required:
        legacy = [r for r in rows if r["decision"] == "different_myotubes"]
        return {"enforced": False, "n_calls_needing_a_reason": len(legacy),
                "n_missing": sum(1 for r in legacy if not r["note"].strip()),
                "note": "instrument v1 did not require a reason for a "
                        "different_myotubes call"}
    missing = [r["uid"] for r in required if decisions[r["uid"]].get("note_missing")]
    return {"enforced": True, "n_calls_needing_a_reason": len(required),
            "n_missing": len(missing), "missing": missing}


def cmd_score_control_only_review(args):
    """Score a control-only safety round under the estimator declared in its key.

    Separate from `score-over-merge-review` because that command's reading rules
    are built on the flagged-versus-control contrast, which this round deliberately
    does not have. Running the wrong scorer would report a calibration check on a
    packet with nothing to calibrate against.
    """
    from .control_only_score import score_control_only

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    export = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    report = score_control_only(key, export)

    counts, primary = report["counts"], report["primary"]
    interval = primary["ci95_stratified_bootstrap"]
    print(f"{counts['n_cases']} merges reviewed: "
          f"{counts['n_different_myotubes']} different, {counts['n_same_myotube']} same, "
          f"{counts['n_unresolved_ambiguous_2d']} unresolved, "
          f"{counts['n_undecided']} undecided")
    print(f"population over-merge rate {primary['population_over_merge_rate']} "
          f"(95% CI {interval['lower']}-{interval['upper']}), implying "
          f"{primary['implied_over_merges']} of "
          f"{primary['accepted_merges_across_six_wells']} accepted merges")
    sens = report["sensitivity_to_excluded_cases"]
    print(f"sensitivity to the {sens['n_excluded']} excluded case(s): "
          f"{sens['rate_if_all_excluded_were_same_myotube']} to "
          f"{sens['rate_if_all_excluded_were_different']}")
    for well in report["per_well"]:
        print(f"  {well['well']:24s} rate={well['rate']} "
              f"({well['different_myotubes']}/{well['n_resolved']} resolved) "
              f"pop={well['population']}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport: {args.out}")
    return 0


def cmd_score_over_merge_review(args):
    """Join a blinded over-merge review export against its key and report.

    The reading rules were fixed in advance (see
    `coordination/reports/claude_over_merge_review_packet_2026-07-29.md` sec.4) and
    this command implements them rather than inventing a summary after the fact:

    * `ambiguous_2d` is **unresolved** and is never pooled with `same_myotube`;
    * the **controls are the calibration**. They are merges the benchmark did *not*
      flag, so calling them `different_myotubes` at a high rate means the reviewer's
      verdicts are not tracking the flag. In that case the verdicts on the flagged
      cases carry little weight and the question stays open;
    * with 3 flagged cases there is no statistical power either way -- this is case
      evidence, not a rate.

    Decision latency is reported because it bears on how much weight the verdicts
    carry, and it is cheap to compute from the timestamps already in the export.
    """
    import datetime as _dt

    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    export = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    entries, decisions = key["key"], export["decisions"]

    if export.get("threshold") != key.get("threshold"):
        raise SystemExit(f"threshold mismatch: export {export.get('threshold')} vs key "
                         f"{key.get('threshold')}; these are not the same packet")
    if export.get("batch_id") != key.get("batch_id"):
        raise SystemExit("batch_id mismatch between export and key")

    def _t(stamp):
        return _dt.datetime.fromisoformat(stamp.replace("Z", "+00:00"))

    rows, missing = [], []
    for uid, meta in entries.items():
        got = decisions.get(uid)
        if got is None:
            missing.append(uid)
            continue
        rows.append({"uid": uid, "kind": meta["case_kind"], "decision": got["decision"],
                     "decided_at": got["decided_at"], "note": got.get("note", ""),
                     "well": meta["well"], "merged_label": meta["merged_label"],
                     "n_fragments": len(meta["fragment_ids"])})
    rows.sort(key=lambda r: r["decided_at"] or "")

    gaps = []
    previous = _t(export["session_started_at"])
    for row in rows:
        if row["decided_at"]:
            now = _t(row["decided_at"])
            gaps.append(round((now - previous).total_seconds(), 1))
            previous = now

    def tally(kind):
        subset = [r for r in rows if r["kind"] == kind]
        out = {v: sum(1 for r in subset if r["decision"] == v)
               for v in export["decision_vocabulary"]}
        out["undecided"] = sum(1 for r in subset if not r["decision"])
        out["n"] = len(subset)
        return out

    flagged, control = tally("over_merge"), tally("control")
    confirmed = [r for r in rows if r["kind"] == "over_merge"
                 and r["decision"] == "different_myotubes"]
    unresolved = [r for r in rows if r["kind"] == "over_merge"
                  and r["decision"] in (None, "ambiguous_2d")]
    control_diff_rate = (control["different_myotubes"] / control["n"]) if control["n"] else None
    flagged_diff_rate = (flagged["different_myotubes"] / flagged["n"]) if flagged["n"] else None

    high_control_rate = control_diff_rate is not None and control_diff_rate >= 0.25
    no_discrimination = (flagged_diff_rate is not None and control_diff_rate is not None
                         and flagged_diff_rate <= control_diff_rate)

    intra = None
    if args.first_pass_key and args.first_pass_decisions:
        intra = _intra_rater(
            entries, decisions,
            json.loads(Path(args.first_pass_key).read_text(encoding="utf-8")),
            json.loads(Path(args.first_pass_decisions).read_text(encoding="utf-8")))

    # The pre-registered rule read a high control rate as "the reviewer is not
    # discriminating". That was a PROXY for reviewer noise, adopted when nothing
    # measured it directly. A blind repeat measures it directly, so when kappa is
    # available it supersedes the proxy -- reasoning that holds whichever way kappa
    # comes out. High control rate + a self-consistent reviewer does not mean the
    # verdicts are noise; it means the benchmark's flag is missing most over-merges.
    kappa = (intra or {}).get("cohens_kappa")
    reviewer_consistent = kappa is not None and kappa >= 0.8
    if high_control_rate and reviewer_consistent:
        verdict = ("controls are called different at a high rate BY A SELF-CONSISTENT "
                   "reviewer (kappa %.2f) -- the flag is under-detecting, not the "
                   "reviewer guessing; the over-merge cost is larger than the flag "
                   "count and remains unquantified" % kappa)
        calibration_failed = False
    elif high_control_rate and kappa is not None:
        verdict = ("UNRESOLVED -- controls fail calibration and the reviewer is not "
                   "self-consistent (kappa %.2f); neither pass is usable" % kappa)
        calibration_failed = True
    elif high_control_rate:
        verdict = ("UNRESOLVED -- controls fail calibration, so the flagged-case "
                   "verdicts carry little weight. Run a blind repeat: if the reviewer "
                   "is self-consistent this reads as the flag under-detecting instead")
        calibration_failed = True
    else:
        verdict = "flagged-case verdicts are usable; still case evidence, not a rate"
        calibration_failed = False

    report = {
        "batch_id": key["batch_id"], "reviewer": export["reviewer"],
        "threshold": key["threshold"], "threshold_status": key["threshold_status"],
        "session_started_at": export["session_started_at"],
        "exported_at": export.get("exported_at"),
        "n_scored": len(rows), "missing_from_export": missing,
        "flagged": flagged, "controls": control,
        "flagged_called_different_rate": (None if flagged_diff_rate is None
                                          else round(flagged_diff_rate, 4)),
        "control_called_different_rate": (None if control_diff_rate is None
                                          else round(control_diff_rate, 4)),
        "confirmed_over_merges": [
            {k: r[k] for k in ("uid", "well", "merged_label", "n_fragments")}
            for r in confirmed],
        "unresolved_flagged_cases": [
            {k: r[k] for k in ("uid", "well", "merged_label", "decision")}
            for r in unresolved],
        "decision_latency_seconds": {
            "median": round(float(np.median(gaps)), 1) if gaps else None,
            "min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None,
            "n_under_10s": sum(1 for g in gaps if g < 10), "n": len(gaps)},
        "notes_written": sum(1 for r in rows if r["note"].strip()),
        "instrument": export.get("instrument", "v1-no-telemetry"),
        "telemetry": _telemetry_summary(rows, decisions),
        "note_discipline": _note_discipline(rows, decisions),
        "confidence_calibration": _confidence_calibration(entries, decisions),
        "calibration_failed": calibration_failed,
        "verdicts_track_the_flag": not no_discrimination,
        "verdict": verdict,
        "pre_registered_rules": [
            "ambiguous_2d is unresolved and never pooled with same_myotube",
            "a high control different_myotubes rate voids the case verdicts",
            "3 flagged cases carry no statistical power either way",
            "even all-same_myotube does not promote the linker",
        ],
        "rule_revision_2026_07_30": (
            "the 'high control rate => reviewer not discriminating' rule used a proxy "
            "for reviewer noise; when a blind repeat measures kappa directly, the "
            "direct measurement supersedes it. Revised before seeing which way it "
            "pointed would have given the same rule."),
    }
    if intra is not None:
        report["intra_rater"] = intra

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.out}")


def _resolve_package(by_name, well, package_args):
    """Map a well stem to its package dir (the C08 package name differs from its stem)."""
    for p in package_args:
        path = Path(p)
        readme = path / "README.json"
        if readme.is_file():
            if json.loads(readme.read_text(encoding="utf-8")).get("image_id") == well:
                return path
        if path.name == well:
            return path
    raise SystemExit(f"no package supplied for well {well}")


def build_parser():
    p = argparse.ArgumentParser(prog="qc-review", description="Proposal QC review + learning loop")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build review.html from an annotation package")
    b.add_argument("--package", required=True)
    b.add_argument("--out")
    b.add_argument("--max-cases", type=int, default=250)
    b.add_argument("--thumb-px", type=int, default=200)
    b.add_argument("--edit-px", type=int, default=384)
    b.add_argument("--reviewer", default="", help="reviewer identity, written into export provenance")
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="train the accept/reject model from decisions.json files")
    t.add_argument("decisions", nargs="+")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("apply", help="apply decisions.json to a canonical InstanceSet")
    a.add_argument("--package", required=True)
    a.add_argument("--decisions", required=True)
    a.add_argument("--reviewer", required=True, help="reviewer identity, persisted to provenance + review log")
    a.add_argument("--out")
    a.set_defaults(func=cmd_apply)

    e = sub.add_parser("export-corrections",
                       help="materialise (proposal, human-correction, reason) pairs; backfills pre-capture edits")
    e.add_argument("--package", required=True)
    e.add_argument("--decisions", required=True)
    e.add_argument("--out", help="shared output dir (default: <annotation_work>/corrections)")
    e.set_defaults(func=cmd_export_corrections)

    br = sub.add_parser("blind-repeat",
                        help="SO01: build a 30-case blind second-pass page + private key (G-SO1 reliability)")
    br.add_argument("--package", nargs="+", required=True, help="the six well package dirs")
    br.add_argument("--out", required=True)
    br.add_argument("--key", help="private blind->real key json (default: <out>.key.json)")
    br.add_argument("--seed", type=int, default=20260722)
    br.add_argument("--thumb-px", type=int, default=176)
    br.add_argument("--edit-px", type=int, default=340)
    br.add_argument("--length-min", type=float, help="focus complete/ambiguous strata: min first-pass length_um")
    br.add_argument("--length-max", type=float, help="focus complete/ambiguous strata: max first-pass length_um")
    br.add_argument("--exclude", nargs="+", help="prior key.json file(s) whose (well, real_id) cases are excluded")
    br.add_argument("--targets", help="override strata counts, e.g. 'complete=4,border=2,ambiguous=3,reject=1'")
    br.add_argument("--note", help="short rule shown in this round's blind instructions box")
    br.add_argument("--reviewer", required=True, help="reviewer identity, written into export provenance")
    br.set_defaults(func=cmd_blind_repeat)

    rt = sub.add_parser("build-retriage",
                        help="re-triage first-pass `ambiguous` proposals into six actionable categories")
    rt.add_argument("--package", nargs="+", required=True, help="the six well package dirs")
    rt.add_argument("--out", required=True, help="output dir for the batch pages + index")
    rt.add_argument("--reviewer", required=True,
                    help="reviewer identity, written into export provenance")
    rt.add_argument("--batch-size", type=int, default=120,
                    help="cases per page; the queue is ordered so you may stop after any batch")
    rt.add_argument("--thumb-px", type=int, default=190)
    rt.add_argument("--edit-px", type=int, default=460)
    rt.add_argument("--note", help="short instruction shown at the top of every batch")
    rt.set_defaults(func=cmd_build_retriage)

    bl = sub.add_parser("build-links",
                        help="fragment linking: 'does this fragment join that one?'")
    bl.add_argument("--round2", required=True, help="retriage_round2 dir with confirmed fragments")
    bl.add_argument("--package", nargs="+", required=True, help="the well package dirs")
    bl.add_argument("--out", required=True, help="output html path")
    bl.add_argument("--reviewer", required=True,
                    help="reviewer identity, written into export provenance")
    bl.add_argument("--batch-id", default="links_b01")
    bl.add_argument("--gap-um", type=float, default=40.0, help="max gap to bridge")
    bl.add_argument("--cos-min", type=float, default=0.80, help="min collinearity, both ends")
    bl.add_argument("--size", type=int, default=460)
    bl.add_argument("--note", help="short instruction shown at the top")
    bl.set_defaults(func=cmd_build_links)

    bc = sub.add_parser("blind-compare",
                        help="reference agreement report: blind second pass vs the key (official gate = Codex)")
    bc.add_argument("--key", required=True)
    bc.add_argument("--decisions", required=True)
    bc.set_defaults(func=cmd_blind_compare)

    al = sub.add_parser("build-active-links",
                        help="active-learning round 2: widen the window, serve the "
                             "pairs the linker is least sure about (most-uncertain first)")
    al.add_argument("--round2", required=True, help="retriage_round2 dir with confirmed fragments")
    al.add_argument("--package", nargs="+", required=True, help="the well package dirs")
    al.add_argument("--pairs", required=True, help="round-1 link_pairs.jsonl (the linker's training labels)")
    al.add_argument("--out", required=True, help="output html path")
    al.add_argument("--reviewer", required=True, help="reviewer identity, written into export provenance")
    al.add_argument("--batch-id", default="links_active_b02")
    al.add_argument("--gap-um", type=float, default=80.0, help="widened candidate max gap")
    al.add_argument("--cos-min", type=float, default=0.70, help="widened candidate min collinearity")
    al.add_argument("--train-gap-um", type=float, default=40.0,
                    help="window to re-find training pairs (must cover all of --pairs)")
    al.add_argument("--train-cos-min", type=float, default=0.80)
    al.add_argument("--exclude", nargs="+",
                    help="link_pairs.jsonl file(s) whose pairs were already offered "
                         "and must not be re-served")
    al.add_argument("--max-pairs", type=int, default=160)
    al.add_argument("--size", type=int, default=460)
    al.set_defaults(func=cmd_build_active_links)

    jc = sub.add_parser("build-junction-page",
                        help="round 1: build the junction-splitting labeling page "
                             "(degree-3 junctions where the classical floor's "
                             "direction-only pairing is ambiguous)")
    jc.add_argument("--territory-cache", default="model_labs/classical/_runs/v1/_territory_cache",
                    help="cached stage-A territory .npy per well")
    jc.add_argument("--bootstrap", default="PrecisionMyotube/annotation_work/bootstrap_v1",
                    help="bootstrap_v1 dir with per-well image_fiber.tif / image_dapi.tif")
    jc.add_argument("--wells", nargs="*", default=[
        "19_B06_act104_trka", "22_B03_act104_egfrc", "23_B02_ctrl",
        "29_C05_br223_egfrc", "32_C08_br223_igf1r", "33_C09_br223_trka"])
    jc.add_argument("--pixel-um", type=float, default=0.6493)
    jc.add_argument("--out", required=True, help="output html path")
    jc.add_argument("--reviewer", required=True, help="reviewer identity, written into export provenance")
    jc.add_argument("--batch-id", default="junctions_round1")
    jc.add_argument("--reasons", choices=["round1", "all"], default="round1",
                    help="round1 = near_threshold_winner|width_or_intensity_conflict "
                         "(245 junctions); all = every near_threshold junction (615)")
    jc.add_argument("--size", type=int, default=460)
    jc.add_argument("--radius-um", type=float, default=60.0)
    jc.set_defaults(func=cmd_build_junctions)

    tj = sub.add_parser("train-junction-model",
                        help="fit the junction classifier from a labeled round export; "
                             "compare junction-level accuracy to the classical floor's "
                             "fixed STRAIGHT_DOT pairing")
    tj.add_argument("--export", nargs="+", required=True,
                    help="labeled *.junctions.json export(s); pass every round to train "
                         "on the combined set")
    tj.add_argument("--territory-cache", default="model_labs/classical/_runs/v1/_territory_cache")
    tj.add_argument("--bootstrap", default="PrecisionMyotube/annotation_work/bootstrap_v1")
    tj.add_argument("--pixel-um", type=float, default=0.6493)
    tj.add_argument("--out", required=True, help="output summary json path")
    tj.add_argument("--model-out", help="optional joblib path to save the fitted model")
    tj.set_defaults(func=cmd_train_junction_model)

    ja = sub.add_parser("build-junction-active-round",
                        help="active-learning round 2: widen the pool, serve the "
                             "junctions the classifier is least sure about")
    ja.add_argument("--territory-cache", default="model_labs/classical/_runs/v1/_territory_cache")
    ja.add_argument("--bootstrap", default="PrecisionMyotube/annotation_work/bootstrap_v1")
    ja.add_argument("--prior-export", nargs="+", required=True,
                    help="every prior round's labeled *.junctions.json export "
                         "(trains the model AND excludes already-served junctions)")
    ja.add_argument("--wells", nargs="*", default=[
        "19_B06_act104_trka", "22_B03_act104_egfrc", "23_B02_ctrl",
        "29_C05_br223_egfrc", "32_C08_br223_igf1r", "33_C09_br223_trka"])
    ja.add_argument("--pixel-um", type=float, default=0.6493)
    ja.add_argument("--out", required=True, help="output html path")
    ja.add_argument("--reviewer", required=True, help="reviewer identity, written into export provenance")
    ja.add_argument("--batch-id", default="junctions_active_r2")
    ja.add_argument("--max-junctions", type=int, default=150)
    ja.add_argument("--size", type=int, default=460)
    ja.add_argument("--radius-um", type=float, default=60.0)
    ja.set_defaults(func=cmd_build_junction_active_round)

    om = sub.add_parser("build-over-merge-page",
                        help="blinded hand review of the candidate over-merges the fragment "
                             "linker introduces at the locked threshold; writes a blinded "
                             "page plus a separate answer key")
    om.add_argument("--cases", default="model_labs/classical/_runs/over_merges_v1",
                    help="output dir of model_labs/classical/extract_over_merges.py")
    om.add_argument("--out", required=True, help="output html path (blinded)")
    om.add_argument("--key", required=True,
                    help="output json path for the answer key -- keep this away from the "
                         "reviewer until they have exported their decisions")
    om.add_argument("--reviewer", required=True,
                    help="reviewer identity, written into export provenance")
    om.add_argument("--batch-id", default="over_merge_r1")
    om.add_argument("--expect-threshold", type=float, default=0.90,
                    help="asserted against the extraction; the operating point is locked "
                         "and this review must not be used to tune it")
    om.add_argument("--seed", type=int, default=20260729,
                    help="seed for control SELECTION (keep this fixed across a repeat, "
                         "or a different set of controls is chosen)")
    om.add_argument("--order-seed", type=int, default=None,
                    help="seed for the presentation order only; defaults to --seed. "
                         "Change this (not --seed) for a blind repeat")
    om.add_argument("--repeat-of", default=None,
                    help="a prior packet's key json: rebuild exactly those objects in a "
                         "new order under new uids, for an intra-rater blind repeat")
    om.add_argument("--control-only", action="store_true",
                    help="control-only safety round: review every extracted accepted "
                         "merge, drop the flagged cases entirely, and do no density "
                         "matching. Requires an extraction built with --uniform-controls. "
                         "This is the round that estimates a population over-merge rate; "
                         "the flag-vs-control contrast is deliberately not measurable.")
    om.add_argument("--controls-per-case", type=int, default=4,
                    help="controls kept per flagged case, chosen from the extracted pool "
                         "to match it on reference density (see cmd_build_over_merge_page)")
    om.add_argument("--pad-px", type=int, default=90)
    om.add_argument("--size", type=int, default=900)
    om.set_defaults(func=cmd_build_over_merge_page)

    os_ = sub.add_parser("score-over-merge-review",
                         help="join a blinded over-merge review export against its key "
                              "and apply the pre-registered reading rules")
    os_.add_argument("--decisions", required=True, help="the reviewer's exported json")
    os_.add_argument("--key", required=True, help="the packet's key json")
    os_.add_argument("--out", default=None, help="optional path to write the report")
    os_.add_argument("--first-pass-key", default=None,
                     help="for a blind repeat: the FIRST pass's key json; adds "
                          "intra-rater agreement to the report")
    os_.add_argument("--first-pass-decisions", default=None,
                     help="for a blind repeat: the FIRST pass's export json")
    os_.set_defaults(func=cmd_score_over_merge_review)

    co = sub.add_parser("score-control-only-review",
                        help="score a control-only accepted-merge safety round under the "
                             "estimator predeclared in its key")
    co.add_argument("--decisions", required=True, help="the reviewer's exported json")
    co.add_argument("--key", required=True, help="the packet's key json")
    co.add_argument("--out", default=None, help="optional path to write the report")
    co.set_defaults(func=cmd_score_control_only_review)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
