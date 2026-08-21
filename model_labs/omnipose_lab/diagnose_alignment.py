"""Do the training IMAGES actually predict the training LABELS?

`diagnose_target.py` proved the target is well-formed. `diagnose_empty.py` proved
the trained net emits a constant. The LR sweep then showed 0.1 and 0.01 -- a
factor of ten apart -- plateauing at the SAME loss (~3.85). Two very different
step sizes converging to one value is not a step-size problem: it is what happens
when the only thing left to learn is the marginal, i.e. when the input carries no
usable information about the target.

The one thing nothing has checked is whether the image tile and the label tile
line up. A misaligned, transposed, or flipped label field is perfectly valid on
its own -- `diagnose_target.py` would pass it -- and yet makes the task
impossible, and the mean the optimal prediction. That is exactly the observed
failure.

Three measurements per tile, all on the arrays `build_dense_fold` hands to
`model.train()`:

``AUC``        probability a labelled pixel is brighter than an unlabelled one.
               ~0.5 = the image says nothing about the labels. Fibres are bright
               desmin on dark background, so a correct pairing must be high.
``offset``     peak of the cross-correlation between image and label mask. Any
               answer other than (0, 0) is a systematic shift.
``controls``   the same AUC against a rolled mask and against ANOTHER tile's
               image. Both should sit near 0.5; they calibrate what "no signal"
               looks like on this data, so the real number is read against a
               measured floor rather than an assumed one.

CPU-only, trains nothing, writes nothing.

    python model_labs/omnipose_lab/diagnose_alignment.py \
        --corpus PrecisionMyotube/annotation_work/plate32_dense_v1 --held-out B02
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "PrecisionMyotube", ROOT / "model_labs",
           ROOT / "model_labs" / "omnipose_lab"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def auc(img: np.ndarray, mask: np.ndarray, rng, cap: int = 400_000) -> float:
    """Mann-Whitney AUC of intensity as a classifier of `mask`, subsampled."""
    from scipy.stats import rankdata

    fg = img[mask].ravel()
    bg = img[~mask].ravel()
    if fg.size == 0 or bg.size == 0:
        return float("nan")
    if fg.size > cap:
        fg = rng.choice(fg, cap, replace=False)
    if bg.size > cap:
        bg = rng.choice(bg, cap, replace=False)
    r = rankdata(np.concatenate([fg, bg]))
    n1, n2 = fg.size, bg.size
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def peak_offset(img: np.ndarray, mask: np.ndarray) -> tuple[int, int, float]:
    """Cross-correlation peak of image against label mask, in pixels."""
    a = img.astype(np.float32) - float(img.mean())
    b = mask.astype(np.float32) - float(mask.mean())
    cc = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape)
    k = np.unravel_index(int(np.argmax(cc)), cc.shape)
    h, w = a.shape
    dy = k[0] - h if k[0] > h // 2 else k[0]
    dx = k[1] - w if k[1] > w // 2 else k[1]
    # Peak height relative to the field, so a flat correlation is visible.
    z = (cc[k] - cc.mean()) / (cc.std() + 1e-9)
    return int(dy), int(dx), float(z)


def main(argv=None) -> int:
    from omnipose_lab.train_fold import build_dense_fold

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=6)
    a = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    fold = build_dense_fold(Path(a.corpus), a.held_out,
                            window_px=a.window_px, overlap=a.overlap, seed=0)
    imgs, labs = fold["images"], fold["labels"]
    n = min(a.limit, len(imgs))
    print(f"corpus {a.corpus}  held out {a.held_out}   tiles {len(imgs)}")
    print(f"inspecting {n}\n")

    hdr = (f"{'tile':>5}{'img min':>9}{'img max':>9}{'img med':>9}"
           f"{'fg %':>8}{'I fg':>8}{'I bg':>8}{'AUC':>8}"
           f"{'AUC roll':>10}{'AUC other':>11}{'offset':>12}{'z':>7}")
    print(hdr); print("-" * len(hdr))

    aucs, offs = [], []
    for i in range(n):
        img = np.asarray(imgs[i], dtype=np.float32)
        m = np.asarray(labs[i]) > 0
        other = np.asarray(imgs[(i + 1) % len(imgs)], dtype=np.float32)

        a_true = auc(img, m, rng)
        a_roll = auc(img, np.roll(m, (211, 173), axis=(0, 1)), rng)
        a_other = auc(other, m, rng) if other.shape == m.shape else float("nan")
        dy, dx, z = peak_offset(img, m)
        aucs.append(a_true); offs.append((dy, dx))

        print(f"{i:>5}{img.min():>9.3f}{img.max():>9.3f}"
              f"{float(np.median(img)):>9.3f}{100 * m.mean():>8.2f}"
              f"{float(img[m].mean()):>8.3f}{float(img[~m].mean()):>8.3f}"
              f"{a_true:>8.3f}{a_roll:>10.3f}{a_other:>11.3f}"
              f"{f'({dy},{dx})':>12}{z:>7.1f}")

    print("-" * len(hdr))
    med = float(np.median(aucs))
    aligned = all(o == (0, 0) for o in offs)
    print(f"median AUC {med:.3f}   all offsets (0,0): {aligned}\n")

    if med < 0.60:
        print("=> IMAGE AND LABELS ARE NOT PAIRED. The intensity of a pixel says")
        print("   almost nothing about whether it is labelled, so the mean IS the")
        print("   optimal prediction and no learning rate can help. Look at the")
        print("   crop arithmetic in tile_corpus.window_tiles and at the")
        print("   image/labels pair on disk in the corpus.")
    elif not aligned:
        print("=> SYSTEMATIC SHIFT. Labels carry signal but sit off the fibres by")
        print(f"   {offs[0]} px. Fix the crop origin before training again.")
    else:
        print("=> PAIRING IS SOUND. Labels sit on bright pixels at zero offset,")
        print("   so the input does carry the signal. The failure is in the")
        print("   training path itself -- run overfit_one_tile.py, which asks")
        print("   whether the net can fit a SINGLE tile it sees every step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
