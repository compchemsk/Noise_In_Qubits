#!/usr/bin/env python3
"""
m5_compare_close_pairs.py -- effect of the 14 close-pair molecules on the key results.

Usage:  python3 m5_compare_close_pairs.py <tables_dir_with_all_141> <tables_dir_without_14> <out.tex>

The second directory is produced by running the whole pipeline with TEMPO_EXCLUDE_DEFECTS=1.
Values are read from the LaTeX tables written by the pipeline, so the comparison table always
matches the two runs.
"""
import os, re, sys

ALL, WO, OUT = sys.argv[1], sys.argv[2], sys.argv[3]


def row(path, label, col):
    """Return the entry in column `col` (0 = first number column) of the row that starts with `label`."""
    for line in open(path):
        if line.strip().startswith(label):
            cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
            return cells[1 + col]
    raise KeyError(label + " in " + path)


ITEMS = [  # (file, row label, column for 100 K, column for 300 K, printed name)
    ("tabS_MD_pairs.tex", "Mean nearest-neighbour distance", 0, 1, r"Mean nearest-neighbor distance (\AA)"),
    ("tabS_MD_pairs.tex", "Median nearest-neighbour", 0, 1, r"Median nearest-neighbor $\nu_\mathrm{dd}$ (MHz)"),
    ("tabS_dipolar.tex", None, None, None, r"$\sqrt{M_2}$, $B_0\parallel n$ (MHz)"),
    ("tabS_spin_bath.tex", "All protons, 4--8", 0, 1, r"Protons 4--8~\AA\ from the spin"),
    ("tabS_m3_network.tex", "Resolved pairs", 0, 1, r"Resolved pairs ($|J|>0.1$~cm$^{-1}$)"),
    ("tabS_m3_network.tex", "Pairs with $|J|>1$", 0, 1, r"Pairs with $|J|>1$~cm$^{-1}$"),
    ("tabS_m3_network.tex", "Resolved partners per radical", 0, 1, "Resolved partners per radical"),
    ("tabS_m3_network.tex", "Radicals without resolved partner", 0, 1, r"Radicals without resolved partner (\%)"),
    ("tabS_m3_network.tex", "Median resolved", 0, 1, r"Median resolved $\langle|J|\rangle$ (cm$^{-1}$)"),
    ("tabS_m3_network.tex", "Largest $\\langle|J|\\rangle$", 0, 1, r"Largest $\langle|J|\rangle$ (cm$^{-1}$)"),
    ("tabS_m3_network.tex", "AF share", 0, 1, r"AF share of resolved couplings (\%)"),
    ("tabS_m3_network.tex", "Mean-field $\\theta$, mean", 0, 1, r"Mean-field $\theta$, mean over radicals (K)"),
    ("tabS_m3_network.tex", "Radicals with $2|J_", 0, 1, r"Radicals with $2|J_\mathrm{max}|>h\nu_\mathrm{Q}$ (\%)"),
    ("tabS_m3_network.tex", "$\\omega_", 0, 1, r"$\nu_\mathrm{ex}/\sqrt{M_2}$, coupled radicals"),
    ("tabS_m3_network.tex", "$\\xi_J$", 0, 1, r"$\xi_J$, persistent pairs"),
]


def dip(path, T):
    for line in open(path):
        if line.strip().startswith(f"{T} &") and "parallel" in line:
            return [c.strip() for c in line.split("&")][2]
    raise KeyError("dipolar")


L = [r"\begin{tabular}{lcccc}", r"\toprule",
     r" & \multicolumn{2}{c}{all 141 molecules} & \multicolumn{2}{c}{without the 14}\\",
     r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r"Quantity & 100 K & 300 K & 100 K & 300 K\\", r"\midrule"]
for f, lab, c1, c2, name in ITEMS:
    if f == "tabS_dipolar.tex":
        vals = [dip(os.path.join(d, f), T) for d in (ALL, WO) for T in (100, 300)]
    else:
        vals = [row(os.path.join(d, f), lab, c) for d in (ALL, WO) for c in (c1, c2)]
    L.append(name + " & " + " & ".join(vals) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(OUT, "w").write("\n".join(L) + "\n")
print("written", OUT)
