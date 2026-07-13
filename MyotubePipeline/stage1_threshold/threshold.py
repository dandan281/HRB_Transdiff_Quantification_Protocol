"""Stage 1 -- Threshold Builder.

Reads the extracted ch*_raw16.tif, (1) resolves channel roles (primary/overlap/dapi) and
(2) selects the display brightness/contrast on the primary (fiber) channel that maximises
myotube visibility. Writes metadata.json (single source of truth for roles + scaling) and a
bc_contactsheet.png so a human can confirm/override the auto-chosen max.

Usage:
    python threshold.py --work <stage1 dir> --stem <stem> --src <nd2 path> [--force-primary N]
"""
from __future__ import annotations
import os
import sys
import json
import argparse

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.filters import sobel
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from iohelpers import write_json, load_config  # noqa: E402

BRIGHT_PCTL = 97.5
CANDIDATE_PCTLS = [95.0, 97.5, 99.0, 99.5]
SAT_LIMIT = 0.03           # >3% saturated pixels among tissue -> penalise that candidate


def load_ch(work: str, ch: int) -> np.ndarray:
    return tifffile.imread(os.path.join(work, f"ch{ch}_raw16.tif")).astype(np.float64)


def channel_scores(a: np.ndarray) -> dict:
    """Replicates analyze_channels.py scoring: count nuclei-like vs fiber-like structures."""
    nz = a[a > 0]
    if nz.size == 0:
        return dict(nuclei=0, fiber=0, p975=0.0)
    mask = remove_small_objects(a > np.percentile(nz, 97), 10)
    props = regionprops(label(mask))
    nuclei = sum(1 for p in props if p.area < 600 and p.eccentricity < 0.85)
    fiber = sum(p.area for p in props
                if p.area >= 150 and p.eccentricity > 0.9 and p.major_axis_length > 40)
    return dict(nuclei=int(nuclei), fiber=int(fiber), p975=float(np.percentile(nz, BRIGHT_PCTL)))


def resolve_roles(scores: dict, n_ch: int, force_primary: int | None) -> dict:
    dapi = max(range(n_ch), key=lambda c: scores[c]["nuclei"])
    fibers = [c for c in range(n_ch) if c != dapi]
    if force_primary is not None and force_primary in fibers:
        primary = force_primary
    else:
        # The user traces a consistent channel; ch1 was validated on Plate23. Prefer ch1 if it is
        # a fiber channel, else the lowest-index fiber channel. Recorded, never silently assumed.
        primary = 1 if 1 in fibers else min(fibers)
    overlap = [c for c in fibers if c != primary][0]
    return dict(primary=primary, overlap=overlap, dapi=dapi)


def bc_score(primary: np.ndarray, dmax: float, tissue_mask: np.ndarray) -> dict:
    """Score a candidate display max: crisp + visible fibers, penalised for saturation."""
    nz = primary[primary > 0]
    sat = float(np.mean(primary[tissue_mask] >= dmax)) if tissue_mask.any() else 1.0
    scaled = np.clip(primary, 0, dmax) / max(dmax, 1.0) * 255.0
    grad = sobel(scaled / 255.0)
    crisp = float(grad[tissue_mask].mean()) if tissue_mask.any() else 0.0
    # fraction of tissue pixels landing in a mid-bright, clearly visible band
    vis = float(np.mean((scaled[tissue_mask] >= 40) & (scaled[tissue_mask] <= 250))) if tissue_mask.any() else 0.0
    penalty = 0.5 if sat > SAT_LIMIT else 1.0
    return dict(max=int(round(dmax)), saturation=round(sat, 4), crispness=round(crisp, 5),
                visible_frac=round(vis, 4), score=round(crisp * vis * penalty, 6))


def select_bc(primary: np.ndarray) -> tuple[int, list[dict]]:
    nz = primary[primary > 0]
    if nz.size == 0:                                       # degenerate/blank primary channel
        return 1, []
    tissue_mask = primary >= np.percentile(nz, 85)        # bright/fiber-ish region
    cands = sorted({int(round(np.percentile(nz, p))) for p in CANDIDATE_PCTLS})
    scored = [bc_score(primary, m, tissue_mask) for m in cands]
    best = max(scored, key=lambda s: s["score"])
    return best["max"], scored


def write_contactsheet(primary: np.ndarray, scored: list[dict], best_max: int, path: str) -> None:
    from PIL import Image, ImageDraw, ImageFont
    if not scored:                                        # nothing to show (blank primary)
        return
    tiles = []
    th = 460
    for s in scored:
        m = s["max"]
        scaled = (np.clip(primary, 0, m) / max(m, 1) * 255.0).astype(np.uint8)
        img = Image.fromarray(scaled).resize((th, th), Image.BILINEAR).convert("RGB")
        d = ImageDraw.Draw(img)
        tag = f"max={m}  score={s['score']}  sat={s['saturation']}"
        if m == best_max:
            tag = "[CHOSEN] " + tag
            d.rectangle([0, 0, th - 1, th - 1], outline=(255, 255, 0), width=4)
        d.rectangle([0, 0, th, 18], fill=(0, 0, 0))
        d.text((3, 3), tag, fill=(255, 255, 0))
        tiles.append(img)
    sheet = Image.new("RGB", (th * len(tiles), th), (0, 0, 0))
    for i, t in enumerate(tiles):
        sheet.paste(t, (i * th, 0))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--force-primary", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config()
    n_ch = sum(os.path.exists(os.path.join(a.work, f"ch{c}_raw16.tif")) for c in range(8))
    if n_ch < 3:
        print(f"ERROR: pipeline contract requires 3 channels (primary/overlap/dapi); found "
              f"{n_ch} ch*_raw16.tif in {a.work}", file=sys.stderr)
        sys.exit(2)

    scores = {c: channel_scores(load_ch(a.work, c)) for c in range(n_ch)}
    roles = resolve_roles(scores, n_ch, a.force_primary)

    primary = load_ch(a.work, roles["primary"])
    best_max, scored = select_bc(primary)
    write_contactsheet(primary, scored, best_max, os.path.join(a.work, "bc_contactsheet.png"))

    meta = dict(
        stem=a.stem, src_nd2=a.src.replace("\\", "/"),
        pixel_um=cfg["pixel_um"], width=cfg["width"], height=cfg["height"],
        channels=roles,
        channel_scores={str(c): scores[c] for c in range(n_ch)},
        display=dict(primary_min=0, primary_max=int(best_max), method="bc_score(crisp*visible,sat<=3%)"),
        bc_candidates=scored,
        created_by="stage1_threshold",
    )
    write_json(os.path.join(a.work, "metadata.json"), meta)
    print(f"roles primary=ch{roles['primary']} overlap=ch{roles['overlap']} dapi=ch{roles['dapi']}")
    print(f"display max={best_max} (candidates: " +
          ", ".join(f"{s['max']}:{s['score']}" for s in scored) + ")")


if __name__ == "__main__":
    main()
