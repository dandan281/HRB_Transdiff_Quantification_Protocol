"""CLI for the fragment-linker.

  python -m linker build       build data/pairs.csv from the training wells
  python -m linker eval        build + leave-one-well-out & plate-held-out evaluation
  python -m linker train       build + fit final model -> models/link.joblib
"""
from __future__ import annotations

import sys

from . import train as T


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cmd = argv[0] if argv else "eval"
    if cmd == "build":
        T.build(write=True)
    elif cmd == "eval":
        T.evaluate()
    elif cmd == "train":
        T.train_final()
    elif cmd == "chain":
        from . import chain as CH
        thr = float(argv[1]) if len(argv) > 1 else 0.5
        CH.evaluate(threshold=thr)
    elif cmd == "filter-eval":
        from . import tracefilter as TF
        TF.evaluate_cv()
    elif cmd == "filter-train":
        from . import tracefilter as TF
        TF.train_final()
    elif cmd == "pipeline":     # end-to-end: chain + filter on held-out
        from . import tracefilter as TF
        lk = float(argv[1]) if len(argv) > 1 else 0.5
        kp = float(argv[2]) if len(argv) > 2 else 0.5
        TF.evaluate_end_to_end(link_thr=lk, keep_thr=kp)
    elif cmd == "full":         # end-to-end: chain + filter + extend on held-out
        from . import extend as EX
        lk = float(argv[1]) if len(argv) > 1 else 0.5
        kp = float(argv[2]) if len(argv) > 2 else 0.5
        et = float(argv[3]) if len(argv) > 3 else 50.0
        EX.evaluate_full(link_thr=lk, keep_thr=kp, extend_thr=et)
    elif cmd == "overlay":      # render my traces vs manual ROIs on the image
        from . import overlay as OV
        OV.main(argv[1:] or None)
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
