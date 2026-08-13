"""Averaged multinucleation across all 4 plates, in the style of the reference figure.

CRITICAL averaging rule (operator): the unit of replication is the WELL, and wells
of the SAME condition are averaged together -- e.g. the four B02 controls (one per
plate) become ONE control bar. We do NOT pool all myotubes of a plate.

Each well contributes its own (%1, %2, %>=3) distribution over its nucleated
myotubes; those per-well percentages are averaged across the replicate wells of a
condition (mean +/- SEM).

Receptor-name normalisation (operator-confirmed synonyms):
    br223 = bmpr2 = bmpr211m2 = br223m2   -> BMPR2
    her2mb = HER2mb                        -> HER2mb
    act104 = actv104                       -> ACT104   [ASSUMED typo variant]
Each treated well = an unordered pair of factors; the condition is that pair.

Bins match the reference: 1 / 2 / >=3 nuclei (>=3 = my 3 + 4 + 5+).

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe multinuc_averaged.py
"""
from __future__ import annotations
import os, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = {  # plate -> multinucleation json
    "23": "New_Quantif_P23/plate23_multinucleation.json",
    "26": "New_Quantif_P26/multinucleation.json",
    "28": "New_Quantif_P28/multinucleation.json",
    "32": "New_Quantif_P32/multinucleation.json",
}
OUTDIR = "New_Quantif_Averaged"

# One receptor-canonicalisation source. This file used to carry its own exact-match
# CANON dict while length_human_vs_machine used prefix matching; the two agreed on
# every token seen so far but could silently diverge on the next new well name,
# making the multinucleation and length figures disagree about the same treatment.
from length_human_vs_machine import canon  # noqa: E402


def condition_of(stem):
    toks = stem.split("_")[2:]                 # drop plate-number + well-position
    names = [canon(t) for t in toks]
    if names == ["CONTROL"]:
        return "control"
    return " + ".join(sorted(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="",
                    help="comma-separated plate numbers to drop, e.g. 28")
    a = ap.parse_args()
    exclude = {p.strip() for p in a.exclude.split(",") if p.strip()}
    suffix = ("_no" + "_".join(sorted(exclude))) if exclude else ""
    os.makedirs(OUTDIR, exist_ok=True)
    if exclude:
        print(f"EXCLUDING plate(s): {', '.join(sorted(exclude))}\n")
    # gather per-well (%1, %2, %>=3) grouped by condition
    groups = {}                                # condition -> list of (well, p1,p2,p3, plate)
    for plate, path in SRC.items():
        if plate in exclude:
            continue
        per = json.load(open(path))["per_well"]
        for w, d in per.items():
            pn = d["pct_of_nucleated"]
            p1, p2 = pn["1"], pn["2"]
            p3 = pn["3"] + pn["4"] + pn["5plus"]
            cond = condition_of(w)
            groups.setdefault(cond, []).append(
                {"well": f"P{plate}:{w}", "p1": p1, "p2": p2, "p3": p3})

    # ---- audit print ----
    print("Condition grouping (well = replicate):\n")
    rows = []
    for cond, members in groups.items():
        arr = np.array([[m["p1"], m["p2"], m["p3"]] for m in members], float)
        mean = arr.mean(0)
        sem = arr.std(0, ddof=1) / np.sqrt(len(members)) if len(members) > 1 else np.zeros(3)
        rows.append({"condition": cond, "n": len(members), "mean": mean, "sem": sem,
                     "members": [m["well"] for m in members]})
        print(f"[{cond}]  n={len(members)}")
        for m in members:
            print(f"    {m['well']:<26} 1={m['p1']:5.1f}  2={m['p2']:5.1f}  >=3={m['p3']:5.1f}")
        print(f"    MEAN  1={mean[0]:5.1f}+-{sem[0]:.1f}  2={mean[1]:5.1f}+-{sem[1]:.1f}"
              f"  >=3={mean[2]:5.1f}+-{sem[2]:.1f}   (%>=2={mean[1]+mean[2]:.1f})\n")

    # order: control first, then treated by ascending % multinucleated (>=2)
    ctrl = [r for r in rows if r["condition"] == "control"]
    treated = sorted([r for r in rows if r["condition"] != "control"],
                     key=lambda r: r["mean"][1] + r["mean"][2])
    ordered = ctrl + treated

    # ---- figure (reference style: stacked 1 / 2 / >=3, error bars, legend right) ----
    labels = [r["condition"].replace("control", "no mb (ctrl)") for r in ordered]
    P = np.array([r["mean"] for r in ordered])          # (n_cond, 3)
    S = np.array([r["sem"] for r in ordered])
    ns = [r["n"] for r in ordered]
    colors = ["#111111", "#c8c8c8", "#7a7a7a"]           # 1 / 2 / >=3
    names = ["1", "2", "≥3"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7, 1.15 * len(labels)), 6.2))
    bottom = np.zeros(len(labels))
    for k in range(3):
        ax.bar(x, P[:, k], bottom=bottom, color=colors[k], edgecolor="white",
               linewidth=1.1, width=0.72, label=names[k])
        top = bottom + P[:, k]
        ax.errorbar(x, top, yerr=S[:, k], fmt="none", ecolor="black",
                    elinewidth=1.1, capsize=3)
        bottom = top
    ax.set_ylabel("% of myotubes", fontsize=12)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 50, 100])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)],
                       rotation=30, ha="right", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(title="nuclei /\nmyotube", bbox_to_anchor=(1.01, 1), loc="upper left",
              frameon=False)
    excl = f"  [excl. plate {', '.join(sorted(exclude))}]" if exclude else ""
    ax.set_title("Multinucleation, averaged across replicate wells "
                 f"(individual myotubes ≥50 µm, overlap ≥50%){excl}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"multinucleation_averaged{suffix}.png"), dpi=140,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(OUTDIR, f"multinucleation_averaged{suffix}.csv"), "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["condition", "n_replicates", "pct_1_mean", "pct_1_sem",
                      "pct_2_mean", "pct_2_sem", "pct_ge3_mean", "pct_ge3_sem",
                      "pct_ge2_mean", "member_wells"])
        for r in ordered:
            m, s = r["mean"], r["sem"]
            wtr.writerow([r["condition"], r["n"], f"{m[0]:.2f}", f"{s[0]:.2f}",
                          f"{m[1]:.2f}", f"{s[1]:.2f}", f"{m[2]:.2f}", f"{s[2]:.2f}",
                          f"{m[1]+m[2]:.2f}", ";".join(r["members"])])
    print(f"-> {OUTDIR}/multinucleation_averaged{suffix}.png + .csv")


if __name__ == "__main__":
    main()
