"""Invoke Fiji macros with `-batch` (NEVER --headless: Ridge Detection needs AWT).

Builds the `key=value;key=value` argument string the shared macro `arg()` helper
parses, runs the macro, and returns (returncode, stdout+stderr).
"""
from __future__ import annotations
import os
import subprocess

from iohelpers import load_config, HERE


def macro_args(**kv) -> str:
    """Build the 'k=v;k=v;' string; forward-slash all paths for the macro parser."""
    parts = []
    for k, v in kv.items():
        s = str(v)
        if os.sep in s or (":" in s and "\\" in s):
            s = s.replace("\\", "/")
        parts.append(f"{k}={s}")
    return ";".join(parts) + ";"


def run_macro(macro_path: str, timeout: int = 1800, **kv):
    """Run `fiji -batch <macro> "<args>"`. Returns (returncode, combined_output)."""
    cfg = load_config()
    fiji = cfg["fiji_exe"]
    if not os.path.exists(fiji):
        raise FileNotFoundError(f"Fiji not found at {fiji} (edit common/config.json)")
    if not os.path.exists(macro_path):
        raise FileNotFoundError(f"macro not found: {macro_path}")
    args = macro_args(**kv)
    cmd = [fiji, "-batch", macro_path, args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def common_macro(name: str) -> str:
    """Absolute path to a macro that lives in common/."""
    return os.path.join(HERE, name)
