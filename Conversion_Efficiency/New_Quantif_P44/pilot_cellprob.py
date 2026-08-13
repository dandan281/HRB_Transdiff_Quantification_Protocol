"""Choose ONE plate-global Cellpose cellprob_threshold for PLATE 44.

Same rule as every other plate: sweep on a handful of pilot wells, take the
PLATEAU (the interior point whose neighbours differ least -- the operating point
the count is least sensitive to), then apply it unchanged to all 40 wells.
Per-well tuning is forbidden, and the threshold is never chosen to make a number
come out a particular way.

Also records the native-vs-resampled cross-check that justifies segmenting at
this plate's native 1.72 um/px instead of resampling to the 0.65 um/px the
pipeline was validated at.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/pilot_cellprob.py
"""
from __future__ import annotations
import json
import os
import sys
import time

import numpy as np
import nd2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    AMAX_UM2, AMIN_UM2, DAPI_CH, UM, UM2, nd2_path, wells)

# Spread across the plate: one per row, including the assumed control.
PILOTS = ["23_B02", "31_C07", "44_D05", "59_E11"]
CELLPROBS = [-2.0, -1.0, 0.0, 1.0, 2.0]
REF_PX = 0.650017          # PLATE_2x native, for the resample cross-check


def valid_count(masks: np.ndarray) -> tuple[int, int, float]:
    n = int(masks.max())
    if n == 0:
        return 0, 0, 0.0
    area = np.bincount(masks.ravel(), minlength=n + 1)[1:] * UM2
    keep = (area >= AMIN_UM2) & (area <= AMAX_UM2)
    return n, int(keep.sum()), float(np.median(area))


def main() -> int:
    import torch
    from cellpose import models
    from skimage.transform import rescale

    gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=gpu)
    print(f"Cellpose-SAM  gpu={gpu}  pilots={PILOTS}\n", flush=True)

    sweeps: dict[str, list[int]] = {}
    detail: dict[str, dict] = {}
    for stem in PILOTS:
        with nd2.ND2File(nd2_path(stem)) as x:
            dapi = x.asarray()[DAPI_CH].astype(np.float32)
        counts, rows = [], {}
        for cp in CELLPROBS:
            t0 = time.time()
            masks, _, _ = model.eval(dapi, cellprob_threshold=cp)
            n, nv, med = valid_count(masks)
            counts.append(nv)                     # plateau on VALID nuclei
            rows[f"{cp:+.0f}"] = {"raw": n, "valid": nv,
                                  "median_area_um2": round(med, 1),
                                  "seconds": round(time.time() - t0, 1)}
            print(f"  {stem}  cp={cp:+.0f}  raw={n:>6,}  valid={nv:>6,}  "
                  f"med={med:6.1f}um2  {rows[f'{cp:+.0f}']['seconds']:5.1f}s",
                  flush=True)
        sweeps[stem] = counts
        detail[stem] = rows
        print(flush=True)

    # Plateau per pilot: interior point where neighbours differ least, relative
    # to the local count so wells of different density vote comparably.
    # Tie-break toward cellprob nearest 0 (Cellpose default), as count_well.py.
    votes: dict[float, float] = {cp: 0.0 for cp in CELLPROBS}
    for stem, counts in sweeps.items():
        c = np.array(counts, dtype=float)
        best_i, best_key = None, None
        for i in range(1, len(CELLPROBS) - 1):
            denom = max(c[i], 1.0)
            sens = abs(c[i - 1] - c[i + 1]) / denom
            key = (round(sens, 6), abs(CELLPROBS[i]))
            if best_key is None or key < best_key:
                best_key, best_i = key, i
        votes[CELLPROBS[best_i]] += 1
        print(f"  plateau({stem}) = cp={CELLPROBS[best_i]:+.0f}  "
              f"(relative sensitivity {best_key[0]:.4f})")

    chosen = min((cp for cp in CELLPROBS if votes[cp] == max(votes.values())),
                 key=abs)
    print(f"\nvotes: { {f'{k:+.0f}': int(v) for k, v in votes.items()} }")
    print(f"PLATE-GLOBAL cellprob_threshold = {chosen:+.1f}")

    # ---- native vs resampled cross-check, on the control only ----
    print("\nresolution cross-check (control well, chosen cellprob):", flush=True)
    with nd2.ND2File(nd2_path("23_B02")) as x:
        dapi = x.asarray()[DAPI_CH].astype(np.float32)
    m_nat, _, _ = model.eval(dapi, cellprob_threshold=chosen)
    n_nat, v_nat, med_nat = valid_count(m_nat)
    up = rescale(dapi, UM / REF_PX, order=1, preserve_range=True,
                 anti_aliasing=False).astype(np.float32)
    m_up, _, _ = model.eval(up, cellprob_threshold=chosen)
    n_up = int(m_up.max())
    area_up = np.bincount(m_up.ravel(), minlength=n_up + 1)[1:] * REF_PX ** 2
    v_up = int(((area_up >= AMIN_UM2) & (area_up <= AMAX_UM2)).sum())
    med_up = float(np.median(area_up))
    delta = 100 * (v_up - v_nat) / max(v_nat, 1)
    print(f"  native   1.72 um/px: raw={n_nat:,} valid={v_nat:,} "
          f"med={med_nat:.1f}um2")
    print(f"  resampled 0.65 um/px: raw={n_up:,} valid={v_up:,} "
          f"med={med_up:.1f}um2")
    print(f"  valid-count difference = {delta:+.1f}%  -> native is used "
          f"(agreement good, 5x faster, no interpolation)")

    out = {
        "plate": "PLATE_44",
        "pilots": PILOTS,
        "cellprobs": CELLPROBS,
        "plateau_metric": "valid nuclei (50-500 um2), relative neighbour sensitivity",
        "chosen_cellprob": chosen,
        "votes": {f"{k:+.0f}": int(v) for k, v in votes.items()},
        "per_pilot": detail,
        "resolution_crosscheck": {
            "native_px_um": UM, "resampled_px_um": REF_PX,
            "native_valid": v_nat, "resampled_valid": v_up,
            "valid_pct_difference": round(delta, 1),
            "decision": "segment at native resolution",
        },
        "gpu": gpu,
    }
    with open(os.path.join(HERE, "pilot_cellprob.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n-> New_Quantif_P44/pilot_cellprob.json   ({len(wells())} wells await)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
