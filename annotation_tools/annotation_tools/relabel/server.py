"""Localhost annotation server. Python stdlib only -- no Flask, no new deps.

Binds 127.0.0.1 by design: this serves unpublished research images and writes to
the corpus, and neither belongs on a network interface.

Rendered fields are cached to disk on first request. A 3636x3636 PNG takes a
second or two to encode and the operator would otherwise pay that on every
reload.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import page as page_mod
from .store import TraceStore, all_stats

PIXEL_UM = 0.650017


class Corpus:
    """Read-only view of the sealed bootstrap, plus the relabel trace store."""

    def __init__(self, bootstrap: Path, traces_root: Path, cache: Path):
        self.bootstrap = Path(bootstrap)
        self.traces_root = Path(traces_root)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.wells = sorted(p.name for p in self.bootstrap.iterdir()
                            if p.is_dir())
        self._lock = threading.Lock()
        self._img: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ image data
    def field(self, well: str) -> np.ndarray:
        with self._lock:
            if well not in self._img:
                import tifffile
                self._img[well] = tifffile.imread(
                    self.bootstrap / well / "image_fiber.tif")
            return self._img[well]

    def labels(self, well: str) -> np.ndarray:
        import tifffile
        return tifffile.imread(self.bootstrap / well / "labels.tif").astype(np.int32)

    def field_png(self, well: str) -> bytes:
        p = self.cache / f"{well}__field.png"
        if not p.exists():
            from PIL import Image
            a = self.field(well).astype(np.float32)
            lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
            g = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
            Image.fromarray((g * 255).astype(np.uint8)).save(p, optimize=False)
        return p.read_bytes()

    def labels_png(self, well: str) -> bytes:
        p = self.cache / f"{well}__labels.png"
        if not p.exists():
            from PIL import Image
            lab = self.labels(well)
            n = int(lab.max())
            rng = np.random.default_rng(7)
            lut = np.zeros((n + 1, 4), dtype=np.uint8)
            lut[1:, :3] = rng.integers(70, 210, size=(n, 3))
            lut[1:, 3] = 255
            Image.fromarray(lut[lab]).save(p, optimize=False)
        return p.read_bytes()

    def existing(self, well: str) -> list[dict]:
        """Per-instance bbox + centroid, for click-selection in the browser."""
        from scipy import ndimage as ndi
        lab = self.labels(well)
        out = []
        for lid, box in enumerate(ndi.find_objects(lab), start=1):
            if box is None:
                continue
            ys, xs = box
            out.append({"label": lid, "r0": int(ys.start), "r1": int(ys.stop),
                        "c0": int(xs.start), "c1": int(xs.stop),
                        "cy": int((ys.start + ys.stop) / 2),
                        "cx": int((xs.start + xs.stop) / 2)})
        return out


def make_handler(corpus: Corpus):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # quiet; the UI is the feedback
            pass

        # ------------------------------------------------------------ helpers
        def _send(self, code, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        # ---------------------------------------------------------------- GET
        def do_GET(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                if not parts:
                    q = parse_qs(u.query)
                    well = (q.get("well", [None])[0] or corpus.wells[0])
                    if well not in corpus.wells:
                        return self._send(404, b"unknown well", "text/plain")
                    html = page_mod.render(well, corpus.wells, PIXEL_UM)
                    return self._send(200, html.encode(), "text/html; charset=utf-8")

                if parts[0] == "summary":
                    stats = all_stats(corpus.traces_root, corpus.wells)
                    rows = ""
                    for w in corpus.wells:
                        s = stats[w]
                        kept = len(corpus.existing(w)) - s["n_rejected_existing"]
                        rows += (f"<tr><td>{w}</td><td><b>{kept}</b></td>"
                                 f"<td class='n'><b>{s['n_traces']}</b></td>"
                                 f"<td class='r'><b>{s['n_rejected_existing']}</b></td>"
                                 f"<td>{s['n_records']}</td></tr>")
                    cmd = ("$env:PYTHONPATH = \"PrecisionMyotube;annotation_tools\"\n"
                           "python -m annotation_tools.relabel apply "
                           "--reviewer &lt;your-name&gt;")
                    html = (page_mod.SUMMARY.replace("__ROWS__", rows)
                            .replace("__APPLYCMD__", cmd))
                    return self._send(200, html.encode(), "text/html; charset=utf-8")

                if parts[0] == "img" and len(parts) == 3:
                    well, what = parts[1], parts[2].split("?")[0]
                    if well not in corpus.wells:
                        return self._send(404, b"", "text/plain")
                    if what == "field.png":
                        return self._send(200, corpus.field_png(well), "image/png")
                    if what == "labels.png":
                        return self._send(200, corpus.labels_png(well), "image/png")
                    return self._send(404, b"", "text/plain")

                if parts[0] == "api" and parts[1] == "state" and len(parts) == 3:
                    well = parts[2]
                    if well not in corpus.wells:
                        return self._json({"error": "unknown well"}, 404)
                    img = corpus.field(well)
                    st = TraceStore(corpus.traces_root, well).current()
                    return self._json({
                        "well": well,
                        "height": int(img.shape[0]), "width": int(img.shape[1]),
                        "existing": corpus.existing(well),
                        "traces": st["traces"],
                        "rejected": {str(k): v for k, v
                                     in st["rejected_existing"].items()},
                    })

                return self._send(404, b"not found", "text/plain")
            except Exception as exc:              # never take the server down
                import traceback
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        # --------------------------------------------------------------- POST
        def do_POST(self):
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                if parts[:2] == ["api", "trace"] and len(parts) == 3:
                    well = parts[2]
                    if well not in corpus.wells:
                        return self._json({"error": "unknown well"}, 404)
                    n = int(self.headers.get("Content-Length", 0))
                    rec = json.loads(self.rfile.read(n) or b"{}")
                    saved = TraceStore(corpus.traces_root, well).append(rec)
                    return self._json(saved)
                return self._json({"error": "not found"}, 404)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


def serve(bootstrap: Path, traces_root: Path, cache: Path, *,
          port: int = 8777, open_browser: bool = True) -> int:
    corpus = Corpus(bootstrap, traces_root, cache)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(corpus))
    url = f"http://127.0.0.1:{port}/"
    print(f"corpus : {corpus.bootstrap}   ({len(corpus.wells)} wells)")
    print(f"traces : {corpus.traces_root}")
    print(f"serving: {url}   (Ctrl-C to stop)\n")
    for w in corpus.wells:
        s = TraceStore(traces_root, w).stats()
        print(f"  {w:<24} {s['n_traces']:>4} added   "
              f"{s['n_rejected_existing']:>3} rejected")
    print("\nfirst load renders the field PNG and takes a few seconds; "
          "it is cached after that.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. every committed trace is already on disk.")
    return 0
