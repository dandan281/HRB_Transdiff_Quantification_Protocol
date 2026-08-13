"""Frozen channel/normalization configurations for the model bake-off (CL03.3).

Both laboratories must use identical, explicitly recorded inputs so the bake-off
compares architecture, not preprocessing. Two configurations are supported:

* ``desmin_only``  -- fiber channel alone (the myotube marker).
* ``desmin_dapi``  -- fiber + DAPI, to test whether nuclear context aids
                      instance separation without changing ground truth.

Normalization is percentile-based and fixed here so it is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChannelConfig:
    name: str
    channels: tuple[str, ...]
    norm_low_pct: float = 1.0
    norm_high_pct: float = 99.5
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


DESMIN_ONLY = ChannelConfig(
    "desmin_only", ("fiber",), note="Single myotube-marker channel.")
DESMIN_DAPI = ChannelConfig(
    "desmin_dapi", ("fiber", "dapi"),
    note="Fiber + nuclear context; does not redefine ground truth.")

CONFIGS = {c.name: c for c in (DESMIN_ONLY, DESMIN_DAPI)}


def normalize(channel: np.ndarray, cfg: ChannelConfig) -> np.ndarray:
    """Percentile-normalize one channel to [0,1] using the frozen percentiles."""
    channel = np.asarray(channel, dtype=np.float32)
    lo = np.percentile(channel, cfg.norm_low_pct)
    hi = np.percentile(channel, cfg.norm_high_pct)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((channel - lo) / (hi - lo), 0.0, 1.0)


def build_stack(channels: dict[str, np.ndarray], cfg: ChannelConfig) -> np.ndarray:
    """Assemble a normalized (C, H, W) stack in the config's channel order."""
    missing = [c for c in cfg.channels if c not in channels]
    if missing:
        raise KeyError(f"channel config {cfg.name!r} needs missing channels {missing}")
    return np.stack([normalize(channels[c], cfg) for c in cfg.channels], axis=0)


def write_config(path: str | Path, cfg: ChannelConfig) -> None:
    Path(path).write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
