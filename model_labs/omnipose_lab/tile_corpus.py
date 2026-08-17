"""Cut training tiles from a traced corpus.

Two strategies, and on a dense corpus only one of them is usable:

``window`` (default)
    Overlapping square windows across the field. Each keeps every instance that
    is WHOLE inside it; instances the window edge cuts are ignored. With ~520
    fibres per field a large window holds ~50 whole ones and clips comparatively
    few, and it looks like the crowded field the model meets at inference.

``instance``
    One tile per instance, sized to that instance. Correct for the SPARSE
    bootstrap corpus (35 fibres/field, so a small tile rarely clipped a
    neighbour). On a dense corpus it is actively harmful: measured on
    plate32_dense_v1 it destroys **2.65 fibres for every 1 it keeps**, because
    every ~450 px tile clips a dozen neighbours and the ignore policy paints
    them out. The model would learn that fields are sparse when they are not.
    Kept for reproducing the old runs, not for new ones.

Why cut instances are painted rather than merely unlabelled: Omnipose has no
per-pixel loss mask, so "ignore" has to be expressed in the image. Painting a
cut fibre out to real local background makes the background label TRUE for those
pixels; leaving it visible with label 0 would assert that a myotube is not a
myotube. Painting is only safe when it is rare, which is exactly what the window
strategy achieves and the instance strategy does not.

    python model_labs/omnipose_lab/tile_corpus.py \\
        --corpus PrecisionMyotube/annotation_work/plate32_dense_v1 \\
        --held-out B02 --window-px 1280 --limit 24
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_traced_well(corpus: Path, well: str):
    """Raw arrays for a traced-corpus well. No policy applied yet."""
    import tifffile

    wd = corpus / well
    image = tifffile.imread(wd / "image_fiber.tif")
    labels = tifffile.imread(wd / "labels.tif").astype(np.int32)
    ig = wd / "ignore.tif"
    ignore = (tifffile.imread(ig).astype(bool) if ig.exists()
              else np.zeros(image.shape, dtype=bool))
    return image, labels, ignore


def window_tiles(image, labels, ignore, *, window_px: int, overlap: float,
                 seed: int = 0, min_instances: int = 1):
    """Overlapping windows; instances cut by an edge are ignored, not taught."""
    from omnipose_lab.data import normalize_field
    from omnipose_lab.ignore_policy import DEFAULT_HALO_PX, paint_out

    h, w = labels.shape
    win = min(window_px, h, w)
    stride = max(int(round(win * (1.0 - overlap))), 1)

    # Full-field area per id: a tile's count below this means the edge cut it.
    full_area = np.bincount(labels.ravel())

    starts_r = list(range(0, max(h - win, 0) + 1, stride))
    starts_c = list(range(0, max(w - win, 0) + 1, stride))
    if starts_r[-1] != h - win:
        starts_r.append(h - win)
    if starts_c[-1] != w - win:
        starts_c.append(w - win)

    out = []
    for r0 in starts_r:
        for c0 in starts_c:
            r1, c1 = r0 + win, c0 + win
            lab = labels[r0:r1, c0:c1].copy()
            img = image[r0:r1, c0:c1].copy()
            ign = ignore[r0:r1, c0:c1].copy()

            counts = np.bincount(lab.ravel(), minlength=full_area.size)
            present = [int(v) for v in np.nonzero(counts[:full_area.size])[0]
                       if v != 0]
            cut = [v for v in present if counts[v] < full_area[v]]
            whole = [v for v in present if v not in cut]
            if len(whole) < min_instances:
                continue

            if cut:
                ign |= np.isin(lab, cut)
                lab[np.isin(lab, cut)] = 0

            # Express ignore in the image, since the loss has no mask for it.
            res = paint_out(img, ign, lab, halo_px=DEFAULT_HALO_PX,
                            seed=seed + r0 + c0)
            norm, _ = normalize_field(res.image, reference=image)

            final, links, n_frag = _split_and_link(lab, whole)
            out.append({"row": r0, "col": c0, "size": (win, win),
                        "image": norm, "labels": final, "links": links,
                        "n_whole": len(whole), "n_cut": len(cut),
                        "n_fragmented": n_frag, "n_pieces": int(final.max()),
                        "painted_fraction": res.stats.get("painted_fraction")})
    return out


def _split_and_link(lab: np.ndarray, whole: list[int]):
    """Give every connected PIECE its own label, and link pieces of one fibre.

    A fibre crossed by another is physically broken in a flat raster: the
    crossing pixels belong to neither, because one pixel cannot carry two
    identities. Leaving the pieces sharing a single label does not help --
    Omnipose builds a distance field per region, so two components produce two
    maxima, two attractors, and two instances at inference. The fibre is
    reported as two short myotubes, which is exactly the false split that
    corrupts length.

    `links` is Omnipose's mechanism for precisely this: it declares that
    separate labels are one object, so flows are computed as though the pieces
    were joined and the model learns to carry identity THROUGH a crossing.
    Measured on this corpus, ~40% of instances are affected, so this is not an
    edge case.
    """
    from scipy import ndimage as ndi

    c8 = np.ones((3, 3), dtype=bool)
    final = np.zeros_like(lab, dtype=np.int32)
    links: set[tuple[int, int]] = set()
    next_id, n_frag = 1, 0
    for old in sorted(whole):
        cc, n = ndi.label(lab == old, structure=c8)
        if n == 0:
            continue
        ids = []
        for k in range(1, n + 1):
            final[cc == k] = next_id
            ids.append(next_id)
            next_id += 1
        # Chain the pieces: (a,b), (b,c), ... is enough to make them one object.
        for a, b in zip(ids[:-1], ids[1:]):
            links.add((a, b))
        n_frag += n > 1
    return final, links, n_frag


def main(argv=None) -> int:
    import tifffile
    from PIL import Image
    from skimage.color import label2rgb

    from omnipose_lab.data import TILE_PX

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--held-out", default=None)
    ap.add_argument("--wells", nargs="+", default=None)
    ap.add_argument("--strategy", default="window", choices=["window", "instance"])
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=24,
                    help="tiles to WRITE (0 = all); all are still counted")
    ap.add_argument("--out", default="model_labs/omnipose_lab/_runs/tiles")
    args = ap.parse_args(argv)

    corpus = Path(args.corpus)
    wells = args.wells or sorted(p.name for p in corpus.iterdir() if p.is_dir())
    train = [w for w in wells if w != args.held_out]
    print(f"corpus   : {corpus}")
    print(f"held out : {args.held_out or '(none)'}")
    print(f"strategy : {args.strategy}"
          + (f"  window={args.window_px}px overlap={args.overlap:.0%}"
             if args.strategy == "window" else f"  ceiling={TILE_PX}px"))
    print(f"training : {len(train)} wells\n")

    out = Path(args.out) / (corpus.name + f"__{args.strategy}"
                            + (f"__heldout_{args.held_out}" if args.held_out else ""))
    out.mkdir(parents=True, exist_ok=True)

    hdr = (f"{'well':<6}{'inst':>6}{'tiles':>7}{'whole':>8}{'cut':>7}"
           f"{'ratio':>8}{'frag':>7}{'links':>7}{'fg%':>7}")
    print(hdr); print("-" * len(hdr))
    written, index = 0, []
    tot = {"tiles": 0, "inst": 0, "whole": 0, "cut": 0, "frag": 0, "links": 0}
    for well in train:
        image, labels, ignore = load_traced_well(corpus, well)
        tiles = window_tiles(image, labels, ignore, window_px=args.window_px,
                             overlap=args.overlap)
        nw = sum(t["n_whole"] for t in tiles)
        nc = sum(t["n_cut"] for t in tiles)
        nf = sum(t["n_fragmented"] for t in tiles)
        nl = sum(len(t["links"]) for t in tiles)
        fg = np.array([(t["labels"] > 0).mean() for t in tiles]) if tiles \
            else np.array([0.0])
        tot["tiles"] += len(tiles); tot["inst"] += int(labels.max())
        tot["whole"] += nw; tot["cut"] += nc
        tot["frag"] += nf; tot["links"] += nl
        print(f"{well:<6}{int(labels.max()):>6}{len(tiles):>7}{nw:>8}{nc:>7}"
              f"{nc/max(nw,1):>8.2f}{nf:>7}{nl:>7}"
              f"{100*np.median(fg):>6.2f}%", flush=True)

        for t in tiles:
            if args.limit and written >= args.limit:
                break
            tag = f"{written:04d}__{well}__r{t['row']}c{t['col']}__{t['size'][0]}"
            tifffile.imwrite(out / f"{tag}__image.tif",
                             t["image"].astype(np.float32))
            tifffile.imwrite(out / f"{tag}__labels.tif",
                             t["labels"].astype(np.uint16))
            over = label2rgb(t["labels"].astype(np.int32), image=t["image"],
                             bg_label=0, alpha=0.45, image_alpha=1,
                             bg_color=(0, 0, 0))
            im = Image.fromarray((np.clip(over, 0, 1) * 255).astype(np.uint8))
            im.thumbnail((1400, 1400))
            im.save(out / f"{tag}__overlay.png")
            index.append({"tag": tag, "well": well, "row": t["row"],
                          "col": t["col"], "size": list(t["size"]),
                          "n_whole": t["n_whole"], "n_cut": t["n_cut"],
                          "n_fragmented": t["n_fragmented"],
                          "n_pieces": t["n_pieces"],
                          "links": sorted(tuple(int(x) for x in p)
                                          for p in t["links"]),
                          "painted_fraction": t["painted_fraction"],
                          "fg_fraction": round(float((t["labels"] > 0).mean()), 5)})
            written += 1

    print("-" * len(hdr))
    print(f"{'TOTAL':<6}{tot['inst']:>6}{tot['tiles']:>7}{tot['whole']:>8}"
          f"{tot['cut']:>7}{tot['cut']/max(tot['whole'],1):>8.2f}"
          f"{tot['frag']:>7}{tot['links']:>7}")
    print(f"\ndestroyed:kept ratio = {tot['cut']/max(tot['whole'],1):.2f}"
          f"   (per-instance tiling on this corpus measured 2.65)")
    print(f"fragmented by crossings = {tot['frag']:,} of {tot['whole']:,} "
          f"({100*tot['frag']/max(tot['whole'],1):.1f}%), rejoined by "
          f"{tot['links']:,} links")
    print(f"wrote {written} of {tot['tiles']} tiles -> {out}")

    (out / "tiles_index.json").write_text(json.dumps(
        {"corpus": str(corpus), "held_out": args.held_out,
         "strategy": args.strategy, "window_px": args.window_px,
         "overlap": args.overlap, "train_wells": train,
         "n_tiles": tot["tiles"], "n_instances": tot["inst"],
         "n_whole_in_tiles": tot["whole"], "n_cut_ignored": tot["cut"],
         "destroyed_per_kept": round(tot["cut"] / max(tot["whole"], 1), 3),
         "n_fragmented_by_crossing": tot["frag"], "n_links": tot["links"],
         "links_note": ("labels are PIECES; each links pair declares two pieces "
                        "are one myotube. Pass to CellposeModel.train as "
                        "train_links so flows carry identity through a crossing."),
         "n_written": written, "tiles": index}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
