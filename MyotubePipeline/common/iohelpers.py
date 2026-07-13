"""Shared IO helpers: run-dir layout, traces.txt read/write, logging, config, JSON.

Every stage imports from here so the on-disk contract stays in one place
(see ../conventions.md). Pure stdlib + numpy; no Fiji, no image libs.
"""
from __future__ import annotations
import json
import os
import hashlib
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # MyotubePipeline/
RUNS = os.path.join(ROOT, "runs")

STAGE_DIRS = {
    1: "stage1_threshold",
    2: "stage2_bright",
    3: "stage3_dim",
    4: "stage4_qc",
    5: "stage5_measure",
}


def load_config() -> dict:
    with open(os.path.join(HERE, "config.json"), encoding="utf-8") as fh:
        return json.load(fh)


def run_dir(stem: str) -> str:
    return os.path.join(RUNS, stem)


def stage_dir(stem: str, stage: int) -> str:
    d = os.path.join(run_dir(stem), STAGE_DIRS[stage])
    os.makedirs(d, exist_ok=True)
    return d


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)


# ---- traces.txt (the canonical polyline format, see conventions.md) ----

def read_traces(path: str) -> list[list[tuple[float, float]]]:
    """Read traces.txt / segments.txt -> list of polylines [(x,y), ...]."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            v = ln.strip().split(",")
            if len(v) < 4:
                continue
            try:
                pts = [(float(v[i]), float(v[i + 1])) for i in range(0, len(v) - 1, 2)]
            except ValueError:
                continue
            if len(pts) >= 2:
                out.append(pts)
    return out


def write_traces(path: str, traces) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for pts in traces:
            if len(pts) < 2:
                continue
            fh.write(",".join(f"{x:.2f},{y:.2f}" for x, y in pts) + "\n")
            n += 1
    return n


# ---- run.log ----

def log(stem: str, stage: str, message: str, **fields) -> None:
    """Append a structured line to runs/<stem>/run.log."""
    rd = run_dir(stem)
    os.makedirs(rd, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    extra = " ".join(f"{k}={v}" for k, v in fields.items())
    line = f"[{ts}] {stage}: {message}"
    if extra:
        line += "  " + extra
    with open(os.path.join(rd, "run.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line)


def file_sha1(path: str, nbytes: int = 1 << 20) -> str:
    """Cheap content fingerprint (first nbytes) for reproducibility logging."""
    if not os.path.exists(path):
        return "MISSING"
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        h.update(fh.read(nbytes))
    return h.hexdigest()[:12]
