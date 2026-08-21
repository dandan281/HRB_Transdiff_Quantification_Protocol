"""Is the TRAINING TARGET degenerate, or did the optimisation fail?

`diagnose_empty.py` showed the trained model emitting a near-constant negative
distance field (-3.804 to -3.672 across a whole tile) on data it was TRAINED on.
A randomly-initialised net emits noise, not a flat value, so it did train — it
converged to the trivial solution "no cells anywhere". Two causes:

  (a) the target Omnipose derives from (labels, links) is all background, so the
      model learned "no cells" CORRECTLY, and the bug is upstream in the data;
  (b) the target is fine and the optimisation failed.

This recomputes the SAME target `model.train()` computes, from the SAME fold, and
reports whether any foreground exists. It trains nothing and writes nothing.

Note this reads the DENSE corpus via `build_dense_fold` — the fold that was
actually trained — not the sparse bootstrap.

    python model_labs/omnipose_lab/diagnose_target.py \\
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus",
                    default="PrecisionMyotube/annotation_work/plate32_dense_v1")
    ap.add_argument("--held-out", default="B02")
    ap.add_argument("--window-px", type=int, default=1280)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--no-links", action="store_true",
                    help="recompute with links=None, to isolate links as the cause")
    a = ap.parse_args(argv)

    from omnipose import core as ocore

    from omnipose_lab.train_fold import build_dense_fold

    fold = build_dense_fold(Path(a.corpus), a.held_out,
                            window_px=a.window_px, overlap=a.overlap, seed=0)
    imgs, labs, links = fold["images"], fold["labels"], fold["links"]
    n = min(a.limit, len(imgs))
    print(f"corpus {a.corpus}  held out {a.held_out}")
    print(f"tiles {len(imgs)}  whole {fold['n_instances']}  "
          f"pieces {fold['n_pieces']}  links {fold['n_links']}"
          + ("   [--no-links: links DISABLED for this check]" if a.no_links else ""))
    print(f"inspecting {n} tiles\n")

    # ---- 1. the labels as handed to model.train() --------------------------
    print("LABELS as handed to model.train()")
    print(f"{'tile':>5}{'n_labels':>10}{'fg %':>9}{'links':>8}{'ids ok':>9}")
    for i in range(n):
        lab = labs[i].astype(np.int32)
        ids = set(int(v) for v in np.unique(lab)) - {0}
        lk = links[i]
        ok = "-" if not lk else all(x in ids for pair in lk for x in pair)
        print(f"{i:>5}{len(ids):>10}{100 * (lab > 0).mean():>9.3f}"
              f"{(len(lk) if lk else 0):>8}{str(ok):>9}")

    # ---- 2. the target Omnipose actually regresses -------------------------
    print("\nTARGET from omnipose.core.labels_to_flows()")
    bad = 0
    for i in range(n):
        lk = None if a.no_links else links[i]
        try:
            flows = ocore.labels_to_flows(
                [labs[i].astype(np.int32)], links=[lk], files=None,
                use_gpu=False, device=None, dim=2, omni=True)
        except TypeError:
            # older/newer signatures differ; fall back to the minimal call
            flows = ocore.labels_to_flows([labs[i].astype(np.int32)],
                                          links=[lk], dim=2, omni=True)
        t = np.asarray(flows[0], dtype=np.float32)
        print(f"  tile {i}: target shape {t.shape}")
        for c in range(t.shape[0]):
            ch = t[c]
            print(f"     ch{c}: min={ch.min():+9.3f} max={ch.max():+9.3f} "
                  f"mean={ch.mean():+9.3f}  nonzero={100*(ch != 0).mean():6.2f}%")
        # Foreground: any channel beyond the label channel carrying signal.
        signal = max(float(np.abs(t[c]).max()) for c in range(1, t.shape[0]))
        verdict = "OK" if signal > 1e-6 else "DEGENERATE"
        bad += verdict != "OK"
        print(f"     -> max |signal| outside label channel = {signal:.4f}  "
              f"{verdict}\n")

    print(f"{n - bad}/{n} tiles carry a non-empty target.")
    if bad == n:
        print("\n=> (a) TARGET IS DEGENERATE. The model learned 'no cells'")
        print("   correctly; the bug is upstream in labels/links.")
        print("   Re-run with --no-links to test whether the link pairs are")
        print("   collapsing it.")
    elif bad == 0:
        print("\n=> (b) TARGET IS FINE. The data is not the problem; the")
        print("   optimisation is. train_fold.py records no per-epoch loss, so")
        print("   add one, and check lr=0.1 with SGD is not diverging at this")
        print("   target scale.")
    else:
        print("\n=> MIXED. Some tiles carry no foreground; check the sampler.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
