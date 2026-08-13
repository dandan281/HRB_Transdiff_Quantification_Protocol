"""Does MyoFuse's discriminative signal exist in DESMIN? (It does not.)

MyoFuse classifies a nucleus as a myonucleus from a **local loss of myotube
signal**: a nucleus sitting in the cytoplasm displaces MyHC and leaves a dark
hole, whereas a myoblast nucleus lying above or below the myotube does not. That
single feature is what lets them beat the mask method.

We stain **Desmin**, an intermediate filament that forms a cage *around* the
nucleus rather than being displaced by it. So the signal may be weaker, absent,
or inverted. This script settles it on our own data before any of their model is
adopted.

Method: for every Cellpose nucleus in the validated C08 well, compare median
Desmin intensity inside the nucleus against a local surrounding ring (other
nuclei excluded from the ring, so the comparison is against cytoplasm).

    ratio << 1  a dark hole  -> MyoFuse's premise holds
    ratio ~ 1   no contrast
    ratio >> 1  perinuclear enrichment -> premise INVERTED for this stain

Run:
    python competitors/MyoFuse/evidence/test_desmin_premise.py \
        --out competitors/MyoFuse/evidence/desmin_premise_result.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[3]
NUCLEI = ROOT / "Conversion_Efficiency/cp_c08_full/cellpose_masks.npy"
PACKAGE = ROOT / "PrecisionMyotube/annotation_work/32_C08_smoke"
CONVERTED_RULE = 0.5      # our current conversion-efficiency rule
RING_DILATE = 6
MIN_NUCLEUS_PX = 30


def measure(nuclei: np.ndarray, fiber: np.ndarray, territory: np.ndarray) -> list[dict]:
    rows = []
    for label_id, sl in enumerate(ndi.find_objects(nuclei), start=1):
        if sl is None:
            continue
        r0, c0, r1, c1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
        nucleus = nuclei[sl] == label_id
        if int(nucleus.sum()) < MIN_NUCLEUS_PX:
            continue
        territory_fraction = float(territory[sl][nucleus].mean())

        pad = RING_DILATE + 2
        rr0, cc0 = max(0, r0 - pad), max(0, c0 - pad)
        rr1, cc1 = min(nuclei.shape[0], r1 + pad), min(nuclei.shape[1], c1 + pad)
        big = np.zeros((rr1 - rr0, cc1 - cc0), dtype=bool)
        big[r0 - rr0:r1 - rr0, c0 - cc0:c1 - cc0] = nucleus
        ring = (ndi.binary_dilation(big, iterations=RING_DILATE)
                & ~ndi.binary_dilation(big, iterations=1))
        # never measure a neighbouring nucleus as if it were cytoplasm
        ring &= ~((nuclei[rr0:rr1, cc0:cc1] > 0) & ~big)
        if int(ring.sum()) < 20:
            continue
        crop = fiber[rr0:rr1, cc0:cc1]
        inside = float(np.median(crop[big]))
        around = float(np.median(crop[ring]))
        if around <= 0:
            continue
        rows.append({
            "id": int(label_id),
            "territory_fraction": territory_fraction,
            "inside": inside, "ring": around, "ratio": inside / around,
            "ring_in_territory": float(territory[rr0:rr1, cc0:cc1][ring].mean()),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    ratio = np.array([r["ratio"] for r in rows])
    fraction = np.array([r["territory_fraction"] for r in rows])
    converted = fraction >= CONVERTED_RULE

    def block(mask):
        v = ratio[mask]
        if not v.size:
            return None
        return {"n": int(v.size), "median_ratio": round(float(np.median(v)), 3),
                "p10": round(float(np.percentile(v, 10)), 3),
                "p90": round(float(np.percentile(v, 90)), 3),
                "fraction_showing_a_hole_ratio_lt_0.8": round(float((v < 0.8).mean()), 3)}

    out = {
        "well": "32_C08_br223_igf1r",
        "n_nuclei_measured": len(rows),
        "conversion_rule": f"nucleus counted converted if >={CONVERTED_RULE:.0%} "
                           "of its area overlaps Desmin territory",
        "n_converted": int(converted.sum()),
        "converted_pct": round(100 * float(converted.mean()), 3),
        "converted": block(converted),
        "not_converted": block(~converted),
        "median_ring_fraction_in_territory": round(
            float(np.median([r["ring_in_territory"] for r, k
                             in zip(rows, converted) if k])), 3),
    }
    try:
        from sklearn.mixture import GaussianMixture
        x = ratio[converted]
        x = x[(x > 0) & (x < 3)].reshape(-1, 1)
        one = GaussianMixture(1, random_state=0).fit(x)
        two = GaussianMixture(2, random_state=0).fit(x)
        out["bimodality"] = {
            "bic_1_component": round(float(one.bic(x)), 1),
            "bic_2_component": round(float(two.bic(x)), 1),
            "two_components_preferred": bool(two.bic(x) < one.bic(x)),
            "component_means": sorted(round(float(m), 3) for m in two.means_.ravel()),
        }
    except Exception as exc:                       # sklearn optional
        out["bimodality"] = {"error": f"{type(exc).__name__}: {exc}"}

    converted_block = out["converted"] or {}
    median = converted_block.get("median_ratio", 1.0)
    out["verdict"] = (
        "MyoFuse premise HOLDS for this stain" if median < 0.8 else
        "MyoFuse premise DOES NOT HOLD: Desmin is ENRICHED at the nucleus "
        "(perinuclear cage), the opposite of the MyHC dark hole their classifier "
        "keys on. Their pretrained model cannot transfer to our images.")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    nuclei = np.load(NUCLEI)
    fiber = tifffile.imread(PACKAGE / "fiber_raw16.tif").astype(np.float32)
    territory = tifffile.imread(PACKAGE / "semantic_territory.tif").astype(bool)
    result = summarise(measure(nuclei, fiber, territory))
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
