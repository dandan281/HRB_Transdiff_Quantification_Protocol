"""Train the three-head tracer net on plate32 polyline targets.

Instrumentation precedes optimisation, in the order the Omnipose post-mortem
says it should have happened there:

1. ``--probe``: the loss floor. Score the ground-truth fields as if they were
   the prediction (logits assembled by inverting each head's activation);
   every term must reach ~0. Then measure each head's share of the trunk
   gradient on a real batch. A term that cannot reach zero, or a head with no
   gradient share, disqualifies the run before it costs anything.
2. ``--overfit``: one fixed batch, can the net memorise it. If four tiles
   cannot be driven to near-floor loss, no 300-epoch run will do better.
3. Training proper: every head's loss is logged every epoch, train and
   held-out, from epoch 0. A fixed evaluation batch (the first one drawn) is
   re-scored on the same pixels every time, so curves are comparable across
   time -- random-crop variance is not mistaken for learning.

Data: per well, `image_fiber.tif` (the already-extracted fibre channel of the
dense corpus) and targets built live from the operator's ROI zip by
`centreline_targets` -- ~2.5 s per well, held in RAM (~1.9 GB for nine wells).
Images are normalised per well to [0,1] between the 1st and 99.9th intensity
percentiles, recorded in the manifest.

    python model_labs/tracer_lab/train_tracer.py --probe
    python model_labs/tracer_lab/train_tracer.py --overfit
    python model_labs/tracer_lab/train_tracer.py --epochs 60
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "annotation_tools", ROOT / "model_labs"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

CORPUS = ROOT / "PrecisionMyotube/annotation_work/plate32_dense_v1"
PLATE = ROOT / "Q_PLATES/Q_Plates/PLATE_32"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_well(well: str, snapped_dir: str | Path | None = None):
    """-> (image float32 [0,1], fields dict) for one well.

    ``snapped_dir``: build targets from ridge-snapped traces persisted by
    `snap_targets.py --all` instead of the raw ROI polylines. The snap is
    validated per well (see `_runs/snapped_v1/verification.json`) and only
    changes trace geometry laterally; a missing npz is an ERROR, not a
    fallback -- silently training half the wells on unsnapped targets would
    be the kind of quiet inconsistency this project keeps paying for.
    """
    import tifffile
    from tracer_lab.centreline_targets import build_targets, \
        targets_from_roi_zip

    manifest = json.loads((CORPUS / well / "well_manifest.json").read_text())
    zips = sorted(PLATE.glob(f"*{well}*.zip"))
    if not zips:
        raise FileNotFoundError(f"no ROI zip for {well}")
    im = tifffile.imread(CORPUS / well / "image_fiber.tif").astype(np.float32)
    lo, hi = np.percentile(im, [1.0, 99.9])
    im = np.clip((im - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    shape = tuple(manifest["acquisition"]["shape"][-2:])
    meta = {"roi_zip": zips[0].name, "norm_lo": float(lo),
            "norm_hi": float(hi), "snapped": False}
    if snapped_dir is not None:
        npz_path = Path(snapped_dir) / f"{well}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(
                f"snapped traces missing for {well}: {npz_path}")
        z = np.load(npz_path)
        polys = [z[k] for k in sorted(z.files,
                                      key=lambda s: int(s.split("_")[1]))]
        fields = build_targets(shape, polys)
        meta["snapped"] = True
    else:
        fields = targets_from_roi_zip(zips[0], shape)
    return im, fields, meta


def augment_crop(crop: dict, k: int, flip: bool) -> dict:
    """Dihedral augmentation with the orientation fixups it requires.

    Flips and 90-degree rotations are exact symmetries of every target EXCEPT
    the angle-doubled orientation, whose components transform as:

    * rot90 by k: doubled angle shifts by 180 deg per quarter turn, so odd k
      negates BOTH channels and even k leaves them alone;
    * any flip: theta -> -theta (mod pi), so sin(2t) negates and cos(2t)
      stays.

    Getting this wrong would teach the net a scrambled direction field while
    every loss still converges -- the classic silent failure this project has
    paid ~0.2 AUC for twice. `--augcheck` verifies these fixups against
    targets rebuilt from transformed polylines before any training uses them.
    """
    if k == 0 and not flip:
        return crop
    out = {}
    for key, a in crop.items():
        ax = (-2, -1)
        b = np.rot90(a, k, axes=ax) if k else a
        if flip:
            b = np.flip(b, axis=-1)
        out[key] = np.ascontiguousarray(b)
    o = out["orient"].copy()
    if k % 2 == 1:
        o = -o                      # doubled angle + 180 deg
    if flip:
        o[1] = -o[1]                # theta -> -theta: sin(2t) negates
    out["orient"] = o
    if "offset" in out:
        # A TRUE vector, not an angle-doubled one: rot90 CCW on (row, col)
        # sends a displacement (vr, vc) -> (-vc, vr), applied k times; the
        # column flip sends (vr, vc) -> (vr, -vc). Using the orientation
        # fixup here would silently scramble it -- different object, different
        # transform law.
        v = out["offset"].copy()
        for _ in range(k % 4):
            v = np.stack([-v[1], v[0]])
        if flip:
            v = np.stack([v[0], -v[1]])
        out["offset"] = np.ascontiguousarray(v)
    return out


class CropSampler:
    """Random tiles from the training wells, biased toward fibre content.

    A uniform crop of a 3636^2 field is ~94% background; half the draws are
    re-rolled until the tile holds at least `min_fg` centre mass so batches
    are not dominated by empty glass, and the other half stay uniform so the
    net still sees real background. With ``augment=True`` every crop gets a
    random dihedral transform (8 variants) via :func:`augment_crop`.
    """

    def __init__(self, wells: dict, tile: int, seed: int = 0,
                 min_fg: float = 0.01, augment: bool = False):
        self.wells = wells
        self.names = sorted(wells.keys())
        self.tile = tile
        self.rng = np.random.default_rng(seed)
        self.min_fg = min_fg
        self.augment = augment

    def draw(self):
        t = self.tile
        for attempt in range(20):
            w = self.names[self.rng.integers(len(self.names))]
            im, f, _ = self.wells[w]
            H, W = im.shape
            r = int(self.rng.integers(0, H - t))
            c = int(self.rng.integers(0, W - t))
            centre = f["centre"][r:r + t, c:c + t]
            uniform_half = (attempt == 0 and self.rng.random() < 0.5)
            if uniform_half or float(centre.mean()) >= self.min_fg:
                return {
                    "image": im[r:r + t, c:c + t],
                    "centre": centre,
                    "orient": f["orient"][:, r:r + t, c:c + t],
                    "crossing": f["crossing"][r:r + t, c:c + t],
                    "orient_valid": f["orient_valid"][r:r + t, c:c + t],
                    "offset": f["offset"][:, r:r + t, c:c + t],
                    "offset_valid": f["offset_valid"][r:r + t, c:c + t],
                }
        return {
            "image": im[r:r + t, c:c + t], "centre": centre,
            "orient": f["orient"][:, r:r + t, c:c + t],
            "crossing": f["crossing"][r:r + t, c:c + t],
            "orient_valid": f["orient_valid"][r:r + t, c:c + t],
            "offset": f["offset"][:, r:r + t, c:c + t],
            "offset_valid": f["offset_valid"][r:r + t, c:c + t]}

    def batch(self, n, device):
        import torch
        crops = [self.draw() for _ in range(n)]
        if self.augment:
            crops = [augment_crop(c, int(self.rng.integers(4)),
                                  bool(self.rng.integers(2))) for c in crops]
        x = torch.from_numpy(
            np.stack([c["image"] for c in crops])[:, None]).to(device)
        tgt = {
            "centre": torch.from_numpy(
                np.stack([c["centre"] for c in crops])).to(device),
            "orient": torch.from_numpy(
                np.stack([c["orient"] for c in crops])).to(device),
            "crossing": torch.from_numpy(
                np.stack([c["crossing"] for c in crops])).to(device),
            "orient_valid": torch.from_numpy(
                np.stack([c["orient_valid"] for c in crops])).to(device),
            "offset": torch.from_numpy(
                np.stack([c["offset"] for c in crops])).to(device),
            "offset_valid": torch.from_numpy(
                np.stack([c["offset_valid"] for c in crops])).to(device),
        }
        return x, tgt


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def loss_floor_probe(tgt, device) -> dict:
    """Assemble the perfect prediction FROM the target and score it.

    centre and orient are regressions, so the perfect prediction is the target
    itself; crossing is a logit, so the perfect logit saturates the label.
    Every term must land ~0 -- a term that cannot is a floor the optimiser
    would sit on, and the number to remember when a training curve "plateaus"
    there. (This probe already caught one: centre as BCE floors at the soft
    label's entropy, 0.082 -- which is why centre is MSE.)
    """
    import torch
    from tracer_lab.net import tracer_loss

    perfect = {
        "centre": tgt["centre"].unsqueeze(1),
        "offset": tgt["offset"] / 12.0,
        "orient": tgt["orient"],
        "crossing": torch.where(tgt["crossing"], 8.0, -8.0)
                        .to(torch.float32).unsqueeze(1),
    }
    total, terms = tracer_loss(perfect, tgt)
    return {k: float(v) for k, v in
            {**terms, "total": total}.items()}


def augcheck() -> int:
    """Prove the orientation fixups: augment(targets(P)) == targets(T(P)).

    Builds fields for a synthetic pattern, then for the same polylines
    transformed by each dihedral op, and compares against `augment_crop` of
    the originals. Orientation is compared as an axial angle where both
    fields are valid; a broken fixup shows up as a ~45-90 deg median error.
    """
    from tracer_lab.centreline_targets import build_targets

    n, S = 160, 200
    t = np.linspace(0, 1, n)
    polys = [np.column_stack([20 + 150 * t, 30 + 120 * t ** 2]),
             np.column_stack([170 - 140 * t, 25 + 150 * t]),
             np.column_stack([np.full(n, 100.0), 20 + 160 * t])]

    def transform_pts(p, k, flip):
        q = p.copy()
        for _ in range(k % 4):        # rot90 CCW on (row, col) grids
            q = np.column_stack([S - 1 - q[:, 1], q[:, 0]])
        if flip:
            q = np.column_stack([q[:, 0], S - 1 - q[:, 1]])
        return q

    base = build_targets((S, S), polys)
    crop = {k: base[k] for k in
            ("centre", "orient", "crossing", "orient_valid",
             "offset", "offset_valid")}
    worst = 0.0
    for k in range(4):
        for flip in (False, True):
            aug = augment_crop(crop, k, flip)
            ref = build_targets(
                (S, S), [transform_pts(p, k, flip) for p in polys])
            both = aug["orient_valid"] & ref["orient_valid"]
            d = np.abs((aug["orient"][:, both] * ref["orient"][:, both])
                       .sum(0)).clip(0, 1)
            err = float(np.degrees(0.5 * np.median(np.arccos(d))))
            cen = float(np.abs(aug["centre"] - ref["centre"]).mean())
            # Median, not max: a transform-law error corrupts EVERY pixel,
            # while pixels equidistant from two traces legitimately pick a
            # different nearest one after the transform and differ by many px.
            # `frac_bad` reports how many such ties there are.
            bo = aug["offset_valid"] & ref["offset_valid"]
            e = np.linalg.norm(aug["offset"][:, bo] - ref["offset"][:, bo],
                               axis=0)
            off = float(np.median(e))
            frac_bad = float((e > 1.0).mean())
            worst = max(worst, off * 10.0)  # px error weighted like degrees
            worst = max(worst, err)
            print(f"k={k} flip={int(flip)}  orient median err {err:6.2f} deg"
                  f"   centre MAE {cen:.4f}   offset median err {off:.3f} px"
                  f"   ties {frac_bad:.2%}   valid px {int(both.sum())}")
    ok = worst < 3.0
    print("AUGCHECK", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import torch
    from tracer_lab.net import TracerNet, tracer_loss, grad_shares

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true",
                    help="random dihedral augmentation on training crops")
    ap.add_argument("--snap", action="store_true",
                    help="train on ridge-snapped traces from _runs/snapped_v1 "
                         "(validated by snap_targets.py); held-out targets "
                         "stay UNSNAPPED -- evaluation is against the "
                         "operator's actual annotation")
    ap.add_argument("--ridge-weight", type=float, default=10.0,
                    help="centre MSE per-pixel weight is 1 + w*target")
    ap.add_argument("--augcheck", action="store_true",
                    help="verify orientation fixups against targets rebuilt "
                         "from transformed polylines, then exit")
    ap.add_argument("--probe", action="store_true",
                    help="loss floor + gradient shares, then exit")
    ap.add_argument("--overfit", action="store_true",
                    help="memorise one fixed batch, then exit")
    ap.add_argument("--out", default="model_labs/tracer_lab/_runs/net_v1")
    a = ap.parse_args(argv)

    if a.augcheck:
        return augcheck()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_wells = sorted(p.name for p in CORPUS.iterdir() if p.is_dir())
    train_names = [w for w in all_wells if w != a.held_out]
    print(f"device {device} | train {train_names} | held out {a.held_out}")

    t0 = time.time()
    snap_dir = (ROOT / "model_labs/tracer_lab/_runs/snapped_v1"
                if a.snap else None)
    wells = {w: load_well(w, snapped_dir=snap_dir) for w in train_names}
    # held-out targets are NEVER snapped: the yardstick is the operator's
    # annotation as drawn, and evaluation must not inherit the training fix
    held = {a.held_out: load_well(a.held_out)}
    print(f"targets built for {len(wells) + 1} wells "
          f"in {time.time() - t0:.1f} s")

    sampler = CropSampler(wells, a.tile, seed=a.seed, augment=a.augment)
    held_sampler = CropSampler(held, a.tile, seed=a.seed + 1)

    def loss(pred, tgt):
        return tracer_loss(pred, tgt, centre_ridge_weight=a.ridge_weight)

    model = TracerNet(base=a.base).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"TracerNet base={a.base}: {n_par / 1e6:.2f} M params")

    # ---- probe: floor + gradient shares --------------------------------
    x, tgt = sampler.batch(a.batch, device)
    floor = loss_floor_probe(tgt, device)
    print("\nLOSS FLOOR (ground truth scored as the prediction)")
    for k, v in floor.items():
        print(f"  {k:<10}{v:12.6f}")
    total, terms = loss(model(x), tgt)
    shares = grad_shares(model, terms)
    print("TRUNK GRADIENT SHARE (untrained net, real batch)")
    for k, v in shares.items():
        print(f"  {k:<10}{v * 100:10.1f}%")
    bad = [k for k, v in floor.items() if k != "total" and v > 0.05]
    dead = [k for k, v in shares.items() if v < 0.01]
    if bad or dead:
        print(f"!! PROBE FAIL: floor>{0.05} for {bad}, dead heads {dead}")
        return 1
    print("probe OK: every term reaches ~0 and every head steers the trunk\n")
    if a.probe:
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    # Cosine decay over the whole run. Added for net_v6 on a measurement, not
    # a habit: an 8-tile memorisation test drove the centre term to 0.004
    # while the 80-epoch run sat at 0.71 on its OWN training batch, and every
    # run so far scored better on held-out than on train -- the signature of
    # underfitting, not overfitting. The fix is more steps at a decaying rate,
    # not more regularisation.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(a.epochs, 1), eta_min=a.lr * 0.02)

    # ---- overfit one fixed batch ---------------------------------------
    if a.overfit:
        for step in range(601):
            model.train()
            opt.zero_grad()
            total, terms = loss(model(x), tgt)
            total.backward()
            opt.step()
            if step % 100 == 0:
                row = "  ".join(f"{k} {float(v):.4f}"
                                for k, v in terms.items())
                print(f"step {step:>4}  total {float(total):.4f}  {row}",
                      flush=True)
        ok = float(total) < 0.15
        print("OVERFIT", "PASS" if ok else
              "FAIL -- the net cannot memorise 8 tiles; fix before training")
        return 0 if ok else 1

    # ---- training proper -----------------------------------------------
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    fixed_train = (x, tgt)                       # memorisation reference
    fixed_held = held_sampler.batch(a.batch, device)   # generalisation, fixed
    log_path = out / "log.jsonl"
    log_f = open(log_path, "w")

    def eval_fixed(pair):
        model.eval()
        with torch.no_grad():
            total, terms = loss(model(pair[0]), pair[1])
        return float(total), {k: float(v) for k, v in terms.items()}

    print(f"{'ep':>4} {'train_total':>11} {'held_total':>10}  "
          f"per-head train | held (centre, orient, crossing)")
    best = float("inf")
    for ep in range(a.epochs):
        model.train()
        for _ in range(a.steps_per_epoch):
            xb, tb = sampler.batch(a.batch, device)
            opt.zero_grad()
            total, _ = loss(model(xb), tb)
            total.backward()
            opt.step()
        sched.step()
        tr_tot, tr = eval_fixed(fixed_train)
        he_tot, he = eval_fixed(fixed_held)
        rec = {"epoch": ep, "train": {**tr, "total": tr_tot},
               "held": {**he, "total": he_tot}}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print(f"{ep:>4} {tr_tot:>11.4f} {he_tot:>10.4f}  "
              f"{tr['centre']:.4f},{tr['orient']:.4f},{tr['crossing']:.4f},"
              f"{tr.get('offset', float('nan')):.4f} | "
              f"{he['centre']:.4f},{he['orient']:.4f},{he['crossing']:.4f},"
              f"{he.get('offset', float('nan')):.4f}",
              flush=True)
        if he_tot < best:
            best = he_tot
            torch.save({"model": model.state_dict(), "base": a.base,
                        "epoch": ep, "held_total": he_tot},
                       out / "best.pt")
    torch.save({"model": model.state_dict(), "base": a.base,
                "epoch": a.epochs - 1}, out / "last.pt")

    manifest = {
        "held_out": a.held_out, "train_wells": train_names,
        "snapped_targets": bool(a.snap),
        "tile": a.tile, "batch": a.batch, "epochs": a.epochs,
        "steps_per_epoch": a.steps_per_epoch, "lr": a.lr, "base": a.base,
        "seed": a.seed, "params_m": n_par / 1e6,
        "loss_floor": floor, "grad_shares_init": shares,
        "norms": {w: wells[w][2] for w in wells},
        "best_held_total": best,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nbest held-out total {best:.4f}; checkpoints + manifest in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
