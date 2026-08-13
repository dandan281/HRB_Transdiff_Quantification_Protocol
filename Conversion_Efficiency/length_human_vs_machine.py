"""Myotube LENGTH distribution -- HUMAN hand-labels vs MACHINE tracer -- in the
style of the reference figure G (3 bins: 0-300 / 300-600 / >600 um).

HUMAN  : Q_Plates/PLATE_*/*Results*.csv (Fiji measurements of hand-traced myotube
         ROIs). Length is the last numeric field of each data row (>0 kept).
MACHINE: individual myotubes traced through crossings from plate{N}_myotube masks
         (real_fusion.trace_fibres), length gated >= 50 um to match the curated
         human population (both are "real myotubes"; without the gate the machine
         is swamped by sub-50 um fragments a human never traces).

Averaging rule (operator): the WELL is the replicate; wells of the same condition
are averaged (mean +/- SEM). Receptor synonyms normalised:
  br223 = bmpr2 = bmpr2-11m2 = br223-m2  -> BMPR2
  her2  = her2mb                          -> HER2mb
  egfr  = egfrc                           -> EGFRC
  act104 = actv104                        -> ACT104
Each treated well = a factor pair; the condition is that pair.

Run from Conversion_Efficiency/:
    cpenv/Scripts/python.exe length_human_vs_machine.py [--exclude 28]
"""
from __future__ import annotations
import os, sys, glob, json, csv, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from real_fusion import trace_fibres  # noqa: E402

QP = "../Q_PLATES/Q_Plates"
PLATES = ["23", "26", "28", "32"]
GATE = 50.0                                  # machine real-myotube length gate (um)
EDGES = [0, 300, np.inf]                      # bins: 0-300 / >300
BINCOLS = ["#111111", "#c8c8c8"]
BINNAMES = ["0-300 µm", ">300 µm"]
OUTDIR = "New_Quantif_Averaged"
ORDER = ["control", "BMPR2 + EGFRC", "BMPR2 + IGF1R", "BMPR2 + TRKA",
         "BMPR2 + HER2mb", "ACT104 + EGFRC", "ACT104 + TRKA", "ACT104 + FGFR"]
LABELS = ["no mb\n(ctrl)", "BMPR2\n+EGFRC", "BMPR2\n+IGF1R", "BMPR2\n+TRKA",
          "BMPR2\n+HER2mb", "ACT104\n+EGFRC", "ACT104\n+TRKA", "ACT104\n+FGFR"]


def canon(tok):
    t = tok.lower().replace("-", "").replace("_", "")
    if t.startswith("ctrl"):
        return "CONTROL"
    if t.startswith("br223") or t.startswith("bmpr2"):
        return "BMPR2"
    if t.startswith("her2"):
        return "HER2mb"
    if t.startswith("act104") or t.startswith("actv104"):
        return "ACT104"
    if t.startswith("trka"):
        return "TRKA"
    if t.startswith("egfr"):
        return "EGFRC"
    if t.startswith("igf1r"):
        return "IGF1R"
    if t.startswith("fgfr"):
        return "FGFR"
    raise ValueError(f"unknown receptor token '{tok}'")


def cond_from_tokens(toks):
    c = [canon(t) for t in toks]
    if c == ["CONTROL"]:
        return "control"
    return " + ".join(sorted(c))


def cond_from_human(fname):
    stem = os.path.basename(fname).rsplit(".", 1)[0]
    toks = [t for t in stem.split("_")[1:]                # drop well position
            if t.lower() not in ("results", "n", "tbc")]
    return cond_from_tokens(toks)


def cond_from_machine(stem):
    return cond_from_tokens(stem.split("_")[2:])          # drop plate-num + position


AREA_MAX = 100000.0   # um^2: drop whole-field / big-blob artifacts (real myotubes < ~1500)
LEN_MAX = 2500.0      # um: drop implausibly long (merged) traces

def human_lengths(path):
    """Lengths of hand-traced myotubes, with artifact rows removed:
    length<=0 (whole-field measurements), whole-field/blob Area, implausible length."""
    vals, dropped = [], 0
    for ln in open(path, encoding="utf-8", errors="ignore"):
        parts = ln.replace("\t", ",").split(",")
        if len(parts) < 6:
            continue
        try:
            area = float(parts[-5].strip())
            v = float(parts[-1].strip())
        except ValueError:
            continue                                    # header / non-numeric row
        if 0 < v < LEN_MAX and area < AREA_MAX:
            vals.append(v)
        else:
            dropped += 1
    return np.array(vals), dropped


def binpct(lengths):
    if lengths.size == 0:
        return np.zeros(3)
    h, _ = np.histogram(lengths, bins=EDGES)
    return 100 * h / h.sum()


def is_dropped(wid, drop_tokens):
    """Drop a well if its id contains ALL substrings of any drop token
    (token '+'-separated), e.g. 'p23+ctrl' drops the P23 control well only."""
    wl = wid.lower()
    return any(all(s in wl for s in toks) for toks in drop_tokens)


def gather_human(exclude, drop_tokens):
    groups, tot_drop = {}, 0
    for p in PLATES:
        if p in exclude:
            continue
        for f in sorted(glob.glob(f"{QP}/PLATE_{p}/*Results*.csv")):
            wid = f"P{p}:{os.path.basename(f)}"
            if is_dropped(wid, drop_tokens):
                print(f"  DROPPED human well: {wid}")
                continue
            L, dropped = human_lengths(f)
            tot_drop += dropped
            if L.size == 0:
                continue
            groups.setdefault(cond_from_human(f), []).append(
                {"well": wid, "pct": binpct(L), "n": L.size})
    print(f"human artifact rows removed (len<=0 / whole-field area / len>{LEN_MAX:.0f}): {tot_drop}")
    return groups


def gather_machine(exclude, drop_tokens):
    cache = os.path.join(OUTDIR, "machine_length_cache.json")
    have = json.load(open(cache)) if os.path.exists(cache) else {}
    groups, changed = {}, False
    for p in PLATES:
        if p in exclude:
            continue
        for m in sorted(glob.glob(f"plate{p}_myotube/*_myotube_mask.npy")):
            stem = os.path.basename(m).replace("_myotube_mask.npy", "")
            key = f"P{p}:{stem}"
            if is_dropped(key, drop_tokens):
                continue
            if key not in have:
                myo = np.load(m)
                _, _, fibres = trace_fibres(myo)
                have[key] = [float(f[0]) for f in fibres]
                changed = True
                print(f"  traced {key}  ({len(have[key])} fibres)")
            L = np.array([x for x in have[key] if x >= GATE])
            groups.setdefault(cond_from_machine(stem), []).append(
                {"well": key, "pct": binpct(L), "n": int(L.size)})
    if changed:
        os.makedirs(OUTDIR, exist_ok=True)
        json.dump(have, open(cache, "w"))
    return groups


def summarize(groups):
    out = {}
    for cond, members in groups.items():
        arr = np.array([m["pct"] for m in members])
        n = len(members)
        out[cond] = {"n": n, "mean": arr.mean(0),
                     "sem": (arr.std(0, ddof=1) / np.sqrt(n) if n > 1
                             else np.zeros(arr.shape[1])),
                     "members": members}
    return out


def draw(ax, summ, title):
    conds = [c for c in ORDER if c in summ]
    labs = [LABELS[ORDER.index(c)] for c in conds]
    P = np.array([summ[c]["mean"] for c in conds])
    S = np.array([summ[c]["sem"] for c in conds])
    ns = [summ[c]["n"] for c in conds]
    x = np.arange(len(conds))
    bottom = np.zeros(len(conds))
    for k in range(len(BINCOLS)):
        ax.bar(x, P[:, k], bottom=bottom, color=BINCOLS[k], edgecolor="white",
               linewidth=1.0, width=0.72, label=BINNAMES[k])
        top = bottom + P[:, k]
        ax.errorbar(x, top, yerr=S[:, k], fmt="none", ecolor="black",
                    elinewidth=1.0, capsize=3)
        bottom = top
    ax.set_ylabel("% of myotubes", fontsize=12)
    ax.set_ylim(0, 118); ax.set_yticks([0, 50, 100])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labs, ns)], fontsize=8)
    ax.set_title(title, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="")
    ap.add_argument("--drop-wells", default="",
                    help="comma-sep tokens; '+'-join required substrings, e.g. 'p23+ctrl'")
    a = ap.parse_args()
    exclude = {p.strip() for p in a.exclude.split(",") if p.strip()}
    drop_tokens = [t.strip().lower().split("+") for t in a.drop_wells.split(",") if t.strip()]
    suffix = ("_no" + "_".join(sorted(exclude))) if exclude else ""
    if drop_tokens:
        suffix += "_drop-" + "-".join("".join(toks) for toks in drop_tokens)
    os.makedirs(OUTDIR, exist_ok=True)
    if exclude:
        print(f"EXCLUDING plate(s): {', '.join(sorted(exclude))}")

    print("=== MACHINE tracing (cached) ===")
    machine = summarize(gather_machine(exclude, drop_tokens))
    human = summarize(gather_human(exclude, drop_tokens))

    print(f"\n{'condition':<18}{'source':>8}{'n':>4}{'0-300':>8}{'>300':>8}")
    for cond in ORDER:
        for src, S in [("HUMAN", human), ("MACHINE", machine)]:
            if cond in S:
                m = S[cond]["mean"]
                print(f"{cond:<18}{src:>8}{S[cond]['n']:>4}{m[0]:>8.1f}{m[1]:>8.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.4), sharey=True)
    draw(axes[0], human, "HUMAN (hand-traced ROIs)")
    draw(axes[1], machine, f"MACHINE (traced fibres ≥{GATE:.0f} µm)")
    axes[1].legend(title="Myotube length", bbox_to_anchor=(1.01, 1),
                   loc="upper left", frameon=False)
    excl = f"  [excl. plate {', '.join(sorted(exclude))}]" if exclude else ""
    fig.suptitle(f"Myotube length distribution by treatment — human vs machine{excl}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUTDIR, f"length_human_vs_machine{suffix}.png"), dpi=140,
                bbox_inches="tight", facecolor="white")

    with open(os.path.join(OUTDIR, f"length_human_vs_machine{suffix}.csv"), "w",
              newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["source", "condition", "n_wells", "pct_0_300", "pct_gt300"])
        for src, S in [("human", human), ("machine", machine)]:
            for c in ORDER:
                if c in S:
                    m = S[c]["mean"]
                    wtr.writerow([src, c, S[c]["n"], f"{m[0]:.2f}", f"{m[1]:.2f}"])
    print(f"\n-> {OUTDIR}/length_human_vs_machine{suffix}.png + .csv")


if __name__ == "__main__":
    main()
