"""DIAGNOSTIC: is the flat ~11% Desmin coverage across wells real, or an artifact
of the detector's per-image percentile normalisation?

The production detector rescales each image to its own 1st/99.9th percentile, runs
CLAHE, then thresholds at per-image percentiles -- three self-normalising steps.
The intensity gate at the 90th percentile pins coverage near 10% of every image
by construction, which would erase any true between-well difference.

Here we do the opposite: keep RAW camera units, subtract slow background with a
top-hat (still in raw units), and threshold every well at the SAME ABSOLUTE value
derived from pooled background noise. If the wells separate, the flat coverage was
an artifact and absolute thresholding is the fix.

Also dumps acquisition stats -- an absolute comparison is only valid if exposure /
gain were constant across wells.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe NEW_Quantif/absolute_desmin.py
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import nd2
from skimage.morphology import white_tophat, disk

HERE = os.path.dirname(os.path.abspath(__file__))
ND2_DIR = "../Q_PLATES/Q_Plates/PLATE_23"
DESMIN_CH = 1
TOPHAT_R = 40
KS = [3, 5, 8, 12]                  # threshold = k * background sigma (shared)


def robust_sigma(a):
    """MAD-based noise estimate, immune to the bright fibre tail."""
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med)) + 1e-12


def main():
    files = sorted(glob.glob(os.path.join(ND2_DIR, "*.nd2")))
    print(f"{len(files)} wells\n")

    wells, dbs_store = [], {}
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        with nd2.ND2File(f) as x:
            raw = x.asarray()[DESMIN_CH].astype(np.float32)
            try:
                ch = x.metadata.channels[DESMIN_CH]
                expo = getattr(ch.volume, "exposureTimeMs", None) or \
                    getattr(ch.channel, "exposureTimeMs", None)
            except Exception:
                expo = None
        d_bs = white_tophat(raw, disk(TOPHAT_R))      # RAW units, no rescale
        sig = robust_sigma(d_bs)
        rec = {"well": stem, "exposure_ms": expo,
               "raw_median": float(np.median(raw)),
               "raw_p99": float(np.percentile(raw, 99)),
               "raw_p999": float(np.percentile(raw, 99.9)),
               "raw_max": float(raw.max()),
               "dbs_sigma": float(sig),
               "dbs_p90": float(np.percentile(d_bs[d_bs > 0], 90)),
               "dbs_mean": float(d_bs.mean())}
        wells.append(rec)
        dbs_store[stem] = d_bs
        print(f"{stem:<24} expo={expo}  raw_med={rec['raw_median']:7.0f}  "
              f"raw_p99.9={rec['raw_p999']:8.0f}  sigma={sig:6.1f}  "
              f"per-image p90(dbs)={rec['dbs_p90']:8.1f}")

    # ---- the point: per-image p90 varies, so a p90 gate is a DIFFERENT absolute
    #      threshold in every well. Use one shared absolute threshold instead.
    sig_ref = float(np.median([w["dbs_sigma"] for w in wells]))
    print(f"\nshared background sigma (median across wells) = {sig_ref:.2f}")
    print("per-image p90 gate spans "
          f"{min(w['dbs_p90'] for w in wells):.0f} - "
          f"{max(w['dbs_p90'] for w in wells):.0f} raw units "
          "-> NOT the same threshold well-to-well\n")

    hdr = f"{'well':<24}" + "".join(f"{'k='+str(k):>9}" for k in KS)
    print("Desmin coverage (%) at SHARED absolute threshold k*sigma")
    print(hdr); print("-" * len(hdr))
    cov = {}
    for rec in wells:
        s = rec["well"]
        row = {}
        for k in KS:
            row[k] = float(100 * (dbs_store[s] > k * sig_ref).mean())
        cov[s] = row
        print(f"{s:<24}" + "".join(f"{row[k]:>8.2f}%" for k in KS))

    ctrl = [w for w in cov if "B02" in w][0]
    print("\nfold-change vs control (B02), by threshold:")
    for k in KS:
        folds = {w.split('_', 1)[1]: round(cov[w][k] / cov[ctrl][k], 2)
                 for w in cov if w != ctrl}
        print(f"  k={k:>2}: " + "  ".join(f"{n}={v}x" for n, v in folds.items()))

    with open(os.path.join(HERE, "absolute_desmin.json"), "w") as fh:
        json.dump({"tophat_radius": TOPHAT_R, "shared_sigma": sig_ref,
                   "ks": KS, "per_well": wells, "coverage_pct": cov}, fh, indent=2)


if __name__ == "__main__":
    main()
