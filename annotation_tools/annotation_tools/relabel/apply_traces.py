"""Fold committed traces into a NEW corpus version.

`bootstrap_v1` is sealed -- every ruling, hash and T03 artifact in the project
references it -- so this never writes there. It emits `bootstrap_v<N>` beside it,
carrying the same layout so `omnipose_lab.data` and the benchmark can consume it
unchanged, plus a manifest recording exactly what changed and why.

The output is a SUPERSET by construction: existing certified masks keep their
pixels and are only dropped when the operator explicitly rejected them. A
relabelling pass can therefore add supervision but cannot quietly erase the
375 masks the corpus was built on.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .raster import compose_labels, unlabelled_fibre_ignore
from .store import TraceStore

TERRITORY_CACHE = "model_labs/classical/_runs/v1/_territory_cache"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_territory(well: str, cache: Path, shape) -> np.ndarray | None:
    """The classical semantic territory: everything the ridge detector calls
    fibre-like, labelled or not. Cached by the sealed classical run."""
    p = Path(cache) / f"{well}.territory.npy"
    if not p.exists():
        return None
    terr = np.load(p)
    if tuple(terr.shape) != tuple(shape):
        raise ValueError(f"{p} shape {terr.shape} != image {shape}")
    return terr.astype(bool)


def apply_well(src_well: Path, dst_well: Path, traces_root: Path, well: str,
               *, reviewer: str, territory_cache: Path,
               exhaustive: bool = False, halo_px: float = 6.0) -> dict:
    import tifffile

    dst_well.mkdir(parents=True, exist_ok=True)
    image = tifffile.imread(src_well / "image_fiber.tif")
    base = tifffile.imread(src_well / "labels.tif").astype(np.int32)

    state = TraceStore(traces_root, well).current()
    traces = list(state["traces"].values())
    rejected = set(int(k) for k in state["rejected_existing"])

    labels, provenance = compose_labels(base, traces, image, rejected=rejected)

    # ---- the ignore mask, which is what makes partial labelling safe --------
    overlap = (tifffile.imread(src_well / "ignore.tif").astype(bool)
               if (src_well / "ignore.tif").exists()
               else np.zeros(image.shape, dtype=bool))
    if exhaustive:
        ignore, ig_stats = overlap.copy(), {
            "mode": "exhaustive",
            "note": ("operator declared this field completely labelled, so "
                     "unlabelled fibre-like pixels are asserted as background"),
        }
    else:
        terr = load_territory(well, territory_cache, image.shape)
        if terr is None:
            raise SystemExit(
                f"no territory mask for {well} in {territory_cache}.\n"
                "  Without it, unlabelled fibres would be taught as background "
                "-- which is worse than not relabelling at all.\n"
                "  Either point --territory-cache at the classical run's cache, "
                "or pass --exhaustive if every fibre in every field really is "
                "labelled.")
        unl, ig_stats = unlabelled_fibre_ignore(terr, labels, halo_px=halo_px)
        ig_stats["mode"] = "territory"
        ignore = overlap | unl
    ignore &= labels == 0                      # a target is never ignored
    tifffile.imwrite(dst_well / "ignore.tif", ignore.astype(np.uint8))

    tifffile.imwrite(dst_well / "labels.tif", labels.astype(np.uint16))
    # Images are copied byte-for-byte: this step changes annotation, never pixels.
    for name in ("image_fiber.tif", "image_dapi.tif"):
        if (src_well / name).exists():
            shutil.copy2(src_well / name, dst_well / name)

    kept = sum(1 for p in provenance if p["origin"] == "bootstrap_v1")
    added = sum(1 for p in provenance
                if p["origin"] == "relabel" and p.get("label"))
    skipped = sum(1 for p in provenance if p.get("skipped"))
    rec = {
        "well": well,
        "n_base": int(base.max()),
        "n_rejected": len(rejected),
        "n_kept_from_bootstrap": kept,
        "n_added_from_traces": added,
        "n_traces_skipped": skipped,
        "n_total": int(labels.max()),
        "labels_sha256": sha256_file(dst_well / "labels.tif"),
        "ignore_sha256": sha256_file(dst_well / "ignore.tif"),
        "ignore": ig_stats,
        "target_fraction": round(float((labels > 0).mean()), 5),
        "ignore_fraction": round(float(ignore.mean()), 5),
        "reviewer": reviewer,
        "provenance": provenance,
    }
    (dst_well / "relabel_provenance.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bootstrap", default="PrecisionMyotube/annotation_work/bootstrap_v1")
    ap.add_argument("--traces", default="PrecisionMyotube/annotation_work/relabel")
    ap.add_argument("--out", default=None,
                    help="default: <bootstrap parent>/bootstrap_v2")
    ap.add_argument("--reviewer", required=True,
                    help="who annotated; recorded per well. Authority stays human.")
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output version if it already exists")
    ap.add_argument("--territory-cache", default=TERRITORY_CACHE,
                    help="classical semantic-territory masks; fibre-like pixels "
                         "carrying no label are ignored rather than taught as "
                         "background")
    ap.add_argument("--halo-px", type=float, default=6.0,
                    help="margin around each label kept as background so the "
                         "fibre/background edge is still learned")
    ap.add_argument("--exhaustive", action="store_true",
                    help="DECLARE every fibre in every field labelled. Skips the "
                         "unlabelled-fibre ignore mask and asserts background "
                         "everywhere else. Only correct if that is literally true.")
    args = ap.parse_args(argv)

    src = Path(args.bootstrap)
    if not src.is_dir():
        raise SystemExit(f"no corpus at {src}")
    traces_root = Path(args.traces)
    dst = Path(args.out) if args.out else src.parent / "bootstrap_v2"

    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} exists; pass --force to replace it")
    if dst.resolve() == src.resolve():
        raise SystemExit("refusing to write into the sealed corpus")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    wells = args.wells or sorted(p.name for p in src.iterdir() if p.is_dir())
    print(f"source : {src}  (sealed, read-only)")
    print(f"traces : {traces_root}")
    print(f"output : {dst}\n")

    if args.exhaustive:
        print("!! --exhaustive: unlabelled fibre-like pixels will be asserted "
              "as BACKGROUND.\n"
              "   Correct only if every fibre in every field is labelled.\n")

    hdr = (f"{'well':<24}{'base':>6}{'rej':>5}{'added':>7}{'total':>7}"
           f"{'target%':>9}{'ignore%':>9}")
    print(hdr); print("-" * len(hdr))
    per_well, tot_base, tot_add, tot_tot = {}, 0, 0, 0
    for w in wells:
        rec = apply_well(src / w, dst / w, traces_root, w,
                         reviewer=args.reviewer,
                         territory_cache=Path(args.territory_cache),
                         exhaustive=args.exhaustive, halo_px=args.halo_px)
        per_well[w] = rec
        tot_base += rec["n_base"]; tot_add += rec["n_added_from_traces"]
        tot_tot += rec["n_total"]
        print(f"{w:<24}{rec['n_base']:>6}{rec['n_rejected']:>5}"
              f"{rec['n_added_from_traces']:>7}{rec['n_total']:>7}"
              f"{100*rec['target_fraction']:>8.3f}%"
              f"{100*rec['ignore_fraction']:>8.3f}%")
    print("-" * len(hdr))
    print(f"{'TOTAL':<24}{tot_base:>6}{'':>5}{tot_add:>7}{tot_tot:>7}")

    for name in ("corrections.jsonl", "synthetic.jsonl"):
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)

    manifest = {
        "corpus_version": dst.name,
        "derived_from": str(src),
        "derived_from_manifest_sha256":
            sha256_file(src / "bootstrap_manifest.json")
            if (src / "bootstrap_manifest.json").exists() else None,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reviewer": args.reviewer,
        "method": ("dense relabelling: operator-traced centrelines rasterised "
                   "to ribbons and snapped to local signal; existing certified "
                   "masks preserved unless explicitly rejected"),
        "ignore_policy": ("exhaustive: unlabelled asserted as background"
                          if args.exhaustive else
                          "fibre-like territory carrying no label is IGNORED, "
                          "not background, so partial coverage is safe; a "
                          f"{args.halo_px:g} px halo around each label stays "
                          "background to preserve the edge signal"),
        "territory_cache": None if args.exhaustive else str(args.territory_cache),
        "evidence_class": "development_bootstrap_single_operator_dense_relabel",
        "limitations": [
            "single operator; not consensus and not inter-rater agreement",
            "traces are centreline+width, so mask edges come from the snap "
            "threshold rather than from a hand-drawn boundary",
            "NOT proposal-conditioned for added instances -- unlike the v1 "
            "corpus, these were drawn directly and were never proposed by the "
            "classical detector; that is the point, and it means v1 and v2 "
            "instances are not exchangeable for provenance purposes",
        ],
        "totals": {"base": tot_base, "added": tot_add, "total": tot_tot},
        "per_well": {w: {k: v for k, v in r.items() if k != "provenance"}
                     for w, r in per_well.items()},
    }
    (dst / "bootstrap_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n-> {dst}/bootstrap_manifest.json")
    print(f"   {tot_base} -> {tot_tot} instances "
          f"({100*tot_tot/max(tot_base,1)-100:+.0f}%)")
    print("\nsealed corpus untouched. To train on the new one, point "
          "omnipose_lab.data.BOOTSTRAP at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
