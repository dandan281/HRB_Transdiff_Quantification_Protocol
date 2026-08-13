"""Synthetic elongated-object fixtures for laboratory smoke tests.

These let a laboratory verify its data layout, mask handling, checkpoint writing,
and prediction-export path *before* real ground truth exists (CL03.2), without
depending on any GPU framework.
"""
from __future__ import annotations

import numpy as np


def synthetic_field(shape: tuple[int, int] = (128, 128), n: int = 5,
                    thickness: int = 4, seed: int = 0
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(fiber, dapi, label_image)`` with ``n`` elongated fibers.

    ``fiber`` and ``dapi`` are uint16 intensity images; ``label_image`` is a
    mutually exclusive int32 label map of the drawn fibers.
    """
    rng = np.random.default_rng(seed)
    h, w = shape
    labels = np.zeros(shape, dtype=np.int32)
    fiber = rng.integers(0, 50, size=shape, dtype=np.uint16)
    for k in range(1, n + 1):
        r0 = rng.integers(0, h - 1)
        c0 = rng.integers(0, w - 1)
        angle = rng.uniform(0, np.pi)
        length = rng.integers(min(h, w) // 3, min(h, w) - 1)
        for t in range(length):
            r = int(r0 + t * np.sin(angle))
            c = int(c0 + t * np.cos(angle))
            if not (0 <= r < h and 0 <= c < w):
                break
            rr0, rr1 = max(0, r - thickness // 2), min(h, r + thickness // 2 + 1)
            cc0, cc1 = max(0, c - thickness // 2), min(w, c + thickness // 2 + 1)
            labels[rr0:rr1, cc0:cc1] = k
    fiber[labels > 0] = rng.integers(2000, 4000)
    # Sparse nuclei blobs for the DAPI channel.
    dapi = rng.integers(0, 50, size=shape, dtype=np.uint16)
    for _ in range(n * 3):
        r = rng.integers(2, h - 2)
        c = rng.integers(2, w - 2)
        dapi[r - 2:r + 3, c - 2:c + 3] = rng.integers(2000, 4000)
    return fiber, dapi, labels
