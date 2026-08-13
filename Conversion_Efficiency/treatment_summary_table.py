"""Consolidated per-TREATMENT results table, averaged across all 4 plates
(well = replicate), mean +/- SEM. Matches the operator's reference table layout.

Columns:
  Sample_size, Total_Num (sum of hand-traced myotubes across plates),
  Length (HUMAN hand-labels): Shorter_300 (<300um), Longer_300 (>=300um),
                              Longer_600 (>=600um)  [each with SEM],
  Conv_Eff (MACHINE per-cell Desmin+ %)            [+ SEM],
  Multinucleation (MACHINE, % of nucleated myotubes): 1, 2, >=3  [each with SEM].

Naming: same treatment across plates is pooled even when the well label differs
(B2 = BMPR2 = br223; token order ignored; m2/11m2 variants merged). All 4 plates,
all wells, NO exclusions (this is the full 4-plate average).

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe treatment_summary_table.py
"""
from __future__ import annotations
import os, glob, json, csv
import numpy as np
from length_human_vs_machine import (cond_from_machine, human_lengths, canon,  # noqa
                                     QP)

PLATES = ["23", "26", "28", "32"]
MN_JSON = {"23": "New_Quantif_P23/plate23_multinucleation.json",
           "26": "New_Quantif_P26/multinucleation.json",
           "28": "New_Quantif_P28/multinucleation.json",
           "32": "New_Quantif_P32/multinucleation.json"}
CE_JSON = {p: f"New_Quantif_P{p}/visualize_final.json" for p in PLATES}

ABBREV = {"control": "CTRL", "BMPR2 + TRKA": "TAB2", "BMPR2 + HER2mb": "B2H2",
          "ACT104 + FGFR": "A10F", "ACT104 + EGFRC": "A10E", "ACT104 + TRKA": "A10TA",
          "BMPR2 + EGFRC": "B2E", "BMPR2 + IGF1R": "B2I"}
ROW_ORDER = ["CTRL", "TAB2", "B2H2", "A10F", "A10E", "A10TA", "B2E", "B2I"]


def human_raw_count(path):
    """Total measured objects (all numeric data rows, incl. length=0 artifacts)."""
    n = 0
    for ln in open(path, encoding="utf-8", errors="ignore"):
        parts = ln.replace("\t", ",").split(",")
        if len(parts) < 6:
            continue
        try:
            float(parts[-1].strip())
        except ValueError:
            continue
        n += 1
    return n


def main():
    # per-well records keyed by (plate, position)
    recs = {}
    for p in PLATES:
        mn = json.load(open(MN_JSON[p]))["per_well"]
        ce = json.load(open(CE_JSON[p]))["per_well"]
        hfiles = {os.path.basename(f).split("_")[0]: f
                  for f in glob.glob(f"{QP}/PLATE_{p}/*Results*.csv")}
        for stem, d in mn.items():
            pos = stem.split("_")[1]
            cond = cond_from_machine(stem)
            pn = d["pct_of_nucleated"]
            rec = {"treat": ABBREV[cond], "cond": cond,
                   "m1": pn["1"], "m2": pn["2"], "m3": pn["3"] + pn["4"] + pn["5plus"],
                   "conv": ce[stem]["conversion_pct"],
                   "lt300": np.nan, "ge300": np.nan, "ge600": np.nan, "nraw": 0}
            hf = hfiles.get(pos)
            if hf:
                L, _ = human_lengths(hf)
                if L.size:
                    rec["lt300"] = 100 * np.mean(L < 300)
                    rec["ge300"] = 100 * np.mean(L >= 300)
                    rec["ge600"] = 100 * np.mean(L >= 600)
                rec["nraw"] = human_raw_count(hf)
            recs[(p, pos)] = rec

    # group by treatment
    def stat(vals):
        v = np.array([x for x in vals if not np.isnan(x)], float)
        n = v.size
        return (round(float(v.mean()), 2) if n else 0.0,
                round(float(v.std(ddof=1) / np.sqrt(n)), 2) if n > 1 else 0.0)

    groups = {}
    for r in recs.values():
        groups.setdefault(r["treat"], []).append(r)

    header = ["Treatment", "Sample_size", "Total_Num",
              "Shorter_300", "S300_Error", "Longer_300", "L300_Error",
              "Longer_600", "L600_Error", "Conv_Eff", "CE_Error",
              "Multi_1", "1_Error", "Multi_2", "2_Error", "Multi_ge3", "3_Error"]
    rows = []
    for t in ROW_ORDER:
        g = groups[t]
        n = len(g)
        total = sum(r["nraw"] for r in g)
        s300 = stat([r["lt300"] for r in g])
        l300 = stat([r["ge300"] for r in g])
        l600 = stat([r["ge600"] for r in g])
        ce = stat([r["conv"] for r in g])
        m1 = stat([r["m1"] for r in g]); m2 = stat([r["m2"] for r in g])
        m3 = stat([r["m3"] for r in g])
        rows.append([t, n, total, s300[0], s300[1], l300[0], l300[1],
                     l600[0], l600[1], ce[0], ce[1], m1[0], m1[1],
                     m2[0], m2[1], m3[0], m3[1]])

    # print aligned
    w = [9, 5, 7, 9, 8, 9, 8, 9, 8, 8, 7, 7, 7, 7, 7, 8, 7]
    print("  ".join(h[:wi].ljust(wi) for h, wi in zip(header, w)))
    print("-" * (sum(w) + 2 * len(w)))
    for r in rows:
        print("  ".join(str(x).ljust(wi) for x, wi in zip(r, w)))

    out = "New_Quantif_Averaged/treatment_summary_table.csv"
    os.makedirs("New_Quantif_Averaged", exist_ok=True)
    with open(out, "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(header); wtr.writerows(rows)
    print("\n-> " + out)
    print("\nLength = HUMAN hand-labels; Conv_Eff + Multinucleation = MACHINE. "
          "All 4 plates, well = replicate, SEM across wells.")


if __name__ == "__main__":
    main()
