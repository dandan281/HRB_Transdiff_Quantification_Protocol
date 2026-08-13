"""Background-subtracted Desmin cache for PLATE 44, in RAW camera units.

Same preprocessing as every other plate -- white top-hat on the raw Desmin
channel, no rescaling -- but the structuring element is derived from its
PHYSICAL size, not copied as a pixel count. The other plates use disk(40 px) at
0.650017 um/px, i.e. a 26 um background scale; at this plate's 1.724571 um/px
that is disk(15 px). Copying `40` across would have applied a 69 um top-hat and
changed what counts as background.

Desmin is channel 1 on both formats, so this stage is the one that would NOT
have failed loudly on a channel mix-up -- which is exactly why the index comes
from `p44_layout`.

Resumable: an existing `<well>_dbs.npy` is reused.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe New_Quantif_P44/build_dbs.py
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import nd2
from skimage.morphology import disk, white_tophat

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from p44_layout import (  # noqa: E402
    DESMIN_CH, TOPHAT_PX, TOPHAT_UM, UM, nd2_path, wells)

CACHE = os.path.join(HERE, "dbs_cache")


def main() -> int:
    os.makedirs(CACHE, exist_ok=True)
    ws = wells()
    print(f"Desmin=ch{DESMIN_CH}  top-hat {TOPHAT_UM} um = disk({TOPHAT_PX} px) "
          f"at {UM} um/px  |  {len(ws)} wells\n", flush=True)
    recs = []
    se = disk(TOPHAT_PX)
    for i, stem in enumerate(ws, 1):
        cf = os.path.join(CACHE, f"{stem}_dbs.npy")
        if os.path.exists(cf):
            print(f"[{i:>2}/{len(ws)}] cached {stem}", flush=True)
            recs.append({"well": stem, "cached": True})
            continue
        with nd2.ND2File(nd2_path(stem)) as x:
            raw = x.asarray()[DESMIN_CH].astype(np.float32)
        d = white_tophat(raw, se)                      # RAW units, no rescale
        np.save(cf, np.clip(d, 0, 65535).astype(np.uint16))
        rec = {"well": stem, "cached": False,
               "raw_median": round(float(np.median(raw)), 1),
               "dbs_median": round(float(np.median(d)), 1),
               "dbs_p99": round(float(np.percentile(d, 99)), 1)}
        recs.append(rec)
        print(f"[{i:>2}/{len(ws)}] {stem:<10} raw_med={rec['raw_median']:7.1f} "
              f"dbs_med={rec['dbs_median']:6.1f}  p99={rec['dbs_p99']:7.1f}",
              flush=True)
    with open(os.path.join(CACHE, "dbs_manifest.json"), "w") as fh:
        json.dump({"plate": "PLATE_44", "desmin_channel": DESMIN_CH,
                   "tophat_um": TOPHAT_UM, "tophat_px": TOPHAT_PX,
                   "pixel_um": UM, "units": "raw camera (12-bit, max 4095)",
                   "per_well": recs}, fh, indent=2)
    print(f"\ndbs_cache ready ({len(ws)} wells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
