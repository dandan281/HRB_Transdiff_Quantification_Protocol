"""Orchestrator -- the deterministic control layer for the staged myotube pipeline.

Owns ALL path wiring (every stage is called with explicit inputs/outputs -> no stage globs the run
tree or reads another stage's internals), logs every step to runs/<stem>/run.log, and enforces the
human-review gate between detection and final measurement.

Usage:
  python orchestrator.py --nd2 "<path.nd2>" [--force-primary N] [--force-max V]
                         [--no-dim-boost] [--auto] [--figures final,bright,dim]
  python orchestrator.py --resume <stem> [--figures ...]      # after editing/saving decisions.json

Without --auto the run stops at Stage 4 once review.html is written (if any case needs attention);
re-run with --resume <stem> after curating decisions.json. With --auto it applies the auto/default
decisions and runs straight through.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from iohelpers import load_config, run_dir, stage_dir, log, read_json, write_json, file_sha1  # noqa
from fiji import run_macro, common_macro  # noqa

CFG = load_config()
PY = CFG["python_exe"]


def D(path: str) -> str:
    """Directory arg for macros: forward slashes + trailing slash."""
    p = os.path.abspath(path).replace("\\", "/")
    return p if p.endswith("/") else p + "/"


def P(path: str) -> str:
    return os.path.abspath(path).replace("\\", "/")


def run_py(stem, label, script, *args):
    cmd = [PY, os.path.join(HERE, script), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    log(stem, label, f"{os.path.basename(script)} rc={proc.returncode}")
    if out:
        for ln in out.splitlines():
            print("   ", ln)
    if proc.returncode != 0:
        print(err, file=sys.stderr)
        raise RuntimeError(f"{script} failed (rc={proc.returncode})")
    return out


def run_fiji(stem_, tag_, macro_, timeout_=1800, **kv):
    # Underscore-suffixed positional names so NO macro arg in **kv (stem=, label=, max=, ...) can
    # ever collide with this function's own parameters.
    rc, out = run_macro(macro_, timeout=timeout_, **kv)
    log(stem_, tag_, f"{os.path.basename(macro_)} rc={rc}")
    if rc != 0:
        print(out, file=sys.stderr)
        raise RuntimeError(f"{macro_} failed (rc={rc})")
    return out


# ---------------- stages ----------------

def stage1(stem, nd2, force_primary, force_max):
    s1 = stage_dir(stem, 1)
    log(stem, "stage1", "extract + threshold + adjust", src=os.path.basename(nd2))
    run_fiji(stem, "stage1", common_or_local("stage1_threshold", "extract.ijm"),
             src=P(nd2), work=D(s1))
    args = ["--work", s1, "--stem", stem, "--src", nd2]
    if force_primary is not None:
        args += ["--force-primary", force_primary]
    run_py(stem, "stage1", "stage1_threshold/threshold.py", *args)

    meta = read_json(os.path.join(s1, "metadata.json"))
    if force_max is not None:
        meta["display"]["primary_max"] = int(force_max)
        meta["display"]["method"] += f"+forced({force_max})"
        write_json(os.path.join(s1, "metadata.json"), meta)
    ch = meta["channels"]; dmax = meta["display"]["primary_max"]
    run_fiji(stem, "stage1", common_or_local("stage1_threshold", "adjust_primary.ijm"),
             work=D(s1), primary=ch["primary"], min=meta["display"]["primary_min"], max=dmax)
    log(stem, "stage1", "done",
        primary=ch["primary"], overlap=ch["overlap"], dapi=ch["dapi"], max=dmax,
        signal=file_sha1(os.path.join(s1, "signal.png")))
    return meta


def stage2(stem, meta):
    s1, s2 = stage_dir(stem, 1), stage_dir(stem, 2)
    ch, dmax = meta["channels"], meta["display"]["primary_max"]
    seg = os.path.join(s2, "bright_segments.txt")
    run_fiji(stem, "stage2", common_macro("detect.ijm"),
             work=D(s1), primary=ch["primary"], out=P(seg), scalemode="reset")
    run_py(stem, "stage2", "stage2_bright/select_bright.py",
           "--stage1", s1, "--segments", seg, "--out", s2)
    render(stem, "stage2", s1, os.path.join(s2, "bright_traces.txt"),
           os.path.join(s2, "bright"), meta)


def stage3(stem, meta, dim_boost=False):
    s1, s2, s3 = stage_dir(stem, 1), stage_dir(stem, 2), stage_dir(stem, 3)
    ch, dmax = meta["channels"], meta["display"]["primary_max"]
    dim_seg = os.path.join(s3, "dim_segments.txt")
    py_args = ["--stage1", s1, "--stage2", s2, "--out", s3]
    if dim_boost:
        # OFF by default: brightened ('fixed') scaling makes Ridge Detection stall on dense wells
        # (your gen_detect note). Short timeout so a stall fails fast instead of hanging 30 min.
        try:
            run_fiji(stem, "stage3", common_macro("detect.ijm"), timeout_=300,
                     work=D(s1), primary=ch["primary"], out=P(dim_seg),
                     scalemode="fixed", max=max(1, dmax // 2),
                     lower=0.12, upper=0.4, low_contrast=20, high_contrast=120)
            py_args += ["--dim-segments", dim_seg]
        except Exception as e:  # dim-boost is best-effort (AWT/stall safety)
            log(stem, "stage3", f"dim-boost detection skipped: {e}")
    run_py(stem, "stage3", "stage3_dim/select_dim.py", *py_args)
    render(stem, "stage3", s1, os.path.join(s3, "dim_traces.txt"),
           os.path.join(s3, "dim"), meta)


def stage4_detect_flag(stem, meta, auto):
    s1, s2, s3, s4 = (stage_dir(stem, i) for i in (1, 2, 3, 4))
    ch, dmax = meta["channels"], meta["display"]["primary_max"]
    run_fiji(stem, "stage4", os.path.join(HERE, "stage4_qc", "composite.ijm"),
             stage1=D(s1), out=D(s4), stem=stem,
             primary=ch["primary"], overlap=ch["overlap"], dapi=ch["dapi"], max=dmax)
    flag_args = ["--stage1", s1, "--stage2", s2, "--stage3", s3, "--out", s4, "--stem", stem]
    if auto:
        flag_args.append("--auto-split")
    run_py(stem, "stage4", "stage4_qc/flag.py", *flag_args)
    run_py(stem, "stage4", "stage4_qc/build_review_html.py", "--out", s4)
    flags = read_json(os.path.join(s4, "flags.json"))
    return flags["n_review"], os.path.join(s4, "review.html")


def stage4_reconcile(stem, meta):
    s1, s4 = stage_dir(stem, 1), stage_dir(stem, 4)
    run_py(stem, "stage4", "stage4_qc/reconcile.py", "--out", s4)
    render(stem, "stage4", s1, os.path.join(s4, "final_traces.txt"),
           os.path.join(s4, "final"), meta)


def stage5(stem, meta, figures):
    s1, s2, s3, s4, s5 = (stage_dir(stem, i) for i in (1, 2, 3, 4, 5))
    src = {"final": os.path.join(s4, "final_traces.txt"),
           "bright": os.path.join(s2, "bright_traces.txt"),
           "dim": os.path.join(s3, "dim_traces.txt")}
    for fig in figures:
        traces = src.get(fig)
        if not traces or not os.path.exists(traces):
            log(stem, "stage5", f"skip figure '{fig}' (no traces)")
            continue
        render(stem, "stage5", s1, traces, os.path.join(s5, fig), meta)
    run_py(stem, "stage5", "stage5_measure/store.py", "--out", s5, "--figures", ",".join(figures))


def learn_step(stem, no_learn):
    """After a reviewed run: log (features->your decisions) and retrain the per-type models."""
    if no_learn:
        return
    s4 = stage_dir(stem, 4)
    try:
        run_py(stem, "learn", "learning/log_feedback.py", "--out", s4, "--stem", stem)
        run_py(stem, "learn", "learning/train.py")
    except Exception as e:  # learning is best-effort; never blocks deliverables
        log(stem, "learn", f"learning step skipped: {e}")


def render(stem, label, stage1_dir, traces_file, out_prefix, meta):
    """Canonical render+measure of a trace set (ROI.zip + results.csv + overlays)."""
    ch, dmax = meta["channels"], meta["display"]["primary_max"]
    run_fiji(stem, label, common_macro("trace_render_measure.ijm"),
             work=D(stage1_dir), traces=P(traces_file), out=P(out_prefix),
             primary=ch["primary"], overlap=ch["overlap"], dapi=ch["dapi"], max=dmax, label=1)


def common_or_local(stage_folder, macro):
    return os.path.join(HERE, stage_folder, macro)


# ---------------- driver ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nd2")
    ap.add_argument("--resume", metavar="STEM")
    ap.add_argument("--force-primary", type=int, default=None)
    ap.add_argument("--force-max", type=int, default=None)
    ap.add_argument("--dim-boost", action="store_true",
                    help="opt-in second dim detection pass (can stall Ridge Detection; off by default)")
    ap.add_argument("--auto", action="store_true",
                    help="apply auto/default decisions and run straight through (no review pause)")
    ap.add_argument("--no-learn", action="store_true",
                    help="skip the learning step (don't log review feedback / retrain models)")
    ap.add_argument("--figures", default="final,bright,dim")
    a = ap.parse_args()
    figures = [f.strip() for f in a.figures.split(",") if f.strip()]

    if a.resume:
        stem = a.resume
        meta = read_json(os.path.join(stage_dir(stem, 1), "metadata.json"))
        log(stem, "resume", "reconcile + final render + measure")
        stage4_reconcile(stem, meta)
        stage5(stem, meta, figures)
        learn_step(stem, a.no_learn)
        print(f"\nDONE (resumed). Deliverables in {os.path.join(run_dir(stem), 'stage5_measure')}")
        return

    if not a.nd2:
        ap.error("provide --nd2 <path> (fresh run) or --resume <stem>")
    nd2 = a.nd2
    stem = os.path.splitext(os.path.basename(nd2))[0]
    log(stem, "start", "fresh run", nd2=P(nd2))

    meta = stage1(stem, nd2, a.force_primary, a.force_max)
    stage2(stem, meta)
    stage3(stem, meta, dim_boost=a.dim_boost)
    n_review, review_html = stage4_detect_flag(stem, meta, a.auto)

    if n_review > 0 and not a.auto:
        print("\n" + "=" * 70)
        print(f"REVIEW GATE: {n_review} case(s) need your decision.")
        print(f"  1) open: {review_html}")
        print(f"  2) curate, click 'Download decisions.json', save it into:")
        print(f"     {stage_dir(stem, 4)}")
        print(f"  3) run: python orchestrator.py --resume {stem} --figures {a.figures}")
        print("=" * 70)
        return

    stage4_reconcile(stem, meta)
    stage5(stem, meta, figures)
    learn_step(stem, a.no_learn)
    print(f"\nDONE. Deliverables in {os.path.join(run_dir(stem), 'stage5_measure')}")


if __name__ == "__main__":
    main()
