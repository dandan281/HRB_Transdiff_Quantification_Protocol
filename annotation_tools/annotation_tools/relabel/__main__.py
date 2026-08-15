"""CLI: `python -m annotation_tools.relabel {serve,apply,stats}`."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

DEFAULT_BOOTSTRAP = "PrecisionMyotube/annotation_work/bootstrap_v1"
DEFAULT_TRACES = "PrecisionMyotube/annotation_work/relabel"
DEFAULT_CACHE = "PrecisionMyotube/annotation_work/relabel/_cache"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="annotation_tools.relabel",
                                 description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="open the relabelling web UI")
    s.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    s.add_argument("--traces", default=DEFAULT_TRACES)
    s.add_argument("--cache", default=DEFAULT_CACHE)
    s.add_argument("--port", type=int, default=8777)
    s.add_argument("--no-open", action="store_true")

    t = sub.add_parser("stats", help="what has been annotated so far")
    t.add_argument("--bootstrap", default=DEFAULT_BOOTSTRAP)
    t.add_argument("--traces", default=DEFAULT_TRACES)

    sub.add_parser("apply", add_help=False,
                   help="fold traces into a new corpus version "
                        "(see `apply --help`)")

    args, rest = ap.parse_known_args(argv)

    if args.cmd == "serve":
        from .server import serve
        return serve(Path(args.bootstrap), Path(args.traces), Path(args.cache),
                     port=args.port, open_browser=not args.no_open)

    if args.cmd == "stats":
        from .store import all_stats
        wells = sorted(p.name for p in Path(args.bootstrap).iterdir()
                       if p.is_dir())
        st = all_stats(Path(args.traces), wells)
        hdr = f"{'well':<24}{'added':>7}{'rejected':>10}{'records':>9}"
        print(hdr); print("-" * len(hdr))
        for w in wells:
            print(f"{w:<24}{st[w]['n_traces']:>7}"
                  f"{st[w]['n_rejected_existing']:>10}{st[w]['n_records']:>9}")
        print("-" * len(hdr))
        print(f"{'TOTAL':<24}{sum(v['n_traces'] for v in st.values()):>7}"
              f"{sum(v['n_rejected_existing'] for v in st.values()):>10}")
        return 0

    if args.cmd == "apply":
        from .apply_traces import main as apply_main
        return apply_main(rest)

    return 1


if __name__ == "__main__":
    sys.exit(main())
