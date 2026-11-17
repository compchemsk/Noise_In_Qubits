#!/usr/bin/env python3
"""m2_17_tables_classes.py -- SI tables for radical classes, spin bath, dipolar couplings."""
import os, json, numpy as np, pandas as pd
from m2_00_io import ROOT
D = os.path.join(ROOT, "data", "module2"); OUT = os.path.join(ROOT, "tables")
sel = pd.read_csv(os.path.join(D, "module2_class_model_selection.csv"))
L = [r"\begin{tabular}{crrrrr}", r"\toprule", r"$k$ & BIC & $\Delta$BIC & Silhouette & Median ARI & Lowest ARI\\", r"\midrule"]
for r in sel.itertuples():
    L.append(f"{r.k} & {r.BIC:.0f} & {r.dBIC:.0f} & {r.silhouette:.3f} & {r.ARI_restart_median:.2f} & {r.ARI_restart_min:.2f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_class_selection.tex"), "w").write("\n".join(L) + "\n")

cp = pd.read_csv(os.path.join(D, "module2_class_properties.csv"), header=[0, 1], index_col=[0, 1])
pop = pd.read_csv(os.path.join(D, "module2_class_populations_by_replica.csv"))
CN = {0: "S", 1: "M", 2: "R", 3: "F"}
cols = [("h_spin", "Height (\\AA)", 1), ("theta0", "$\\theta_0$ ($^\\circ$)", 1), ("S2", "$S^2$", 2),
        ("cosNO", "$\\langle\\cos\\theta_\\mathrm{NO}\\rangle$", 2), ("n8", "Spins $<8$~\\AA", 2),
        ("tilt", "Tilt ($^\\circ$)", 1), ("gauche", "Gauche", 2), ("contact", "O--Au contact", 2), ("edge_dist", "Edge dist.\\ (\\AA)", 1)]
L = [r"\begin{tabular}{cc" + "c" * (len(cols) + 1) + "}", r"\toprule",
     r"$T$ (K) & Class & Population (\%) & " + " & ".join(c[1] for c in cols) + r"\\", r"\midrule"]
for T in (100, 300):
    for c in range(4):
        if (T, c) not in cp.index: continue
        p = pop[pop["T"] == T][str(c)] * 100
        cells = [f"${cp.loc[(T, c), (k, 'mean')]:.{nd}f}\\pm{cp.loc[(T, c), (k, 'std')]:.{nd}f}$" for k, _, nd in cols]
        L.append(f"{T} & {CN[c]} & ${p.mean():.1f}\\pm{p.std(ddof=1):.1f}$ & " + " & ".join(cells) + r"\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(OUT, "tabS_class_properties.tex"), "w").write("\n".join(L) + "\n")

b = pd.read_csv(os.path.join(D, "module2_spin_bath_samples.csv.gz"))
r = b.groupby(["T", "rep"]).mean(numeric_only=True)
r["conc"] = (r.tot_4_8 + r.tot_8_12) / (4 / 3 * np.pi * (12**3 - 4**3)) * 1e27 / 6.02214076e23
rows = [("head_0_4", "Own head protons, $r<4$~\\AA", 1), ("head_4_8", "Own head protons, 4--8~\\AA", 2),
        ("rest_4_8", "Own linker/chain protons, 4--8~\\AA", 2), ("nb_4_8", "Neighbour protons, 4--8~\\AA", 1),
        ("tot_4_8", "All protons, 4--8~\\AA", 1), ("tot_8_12", "All protons, 8--12~\\AA", 1),
        ("conc", "Effective $^1$H concentration, 4--12~\\AA\\ (M)", 1),
        ("sum_r6_all", "$\\sum r^{-6}$, $r\\geq4$~\\AA\\ ($10^{-3}$~\\AA$^{-6}$)", 2),
        ("CH2_nb_4_8", "Neighbour CH$_2$ groups, 4--8~\\AA", 1), ("CH3_nb_2p5_7", "Neighbour CH$_3$ groups, 2.5--7~\\AA", 1),
        ("frac_below", "Fraction of 4--12~\\AA\\ protons on the Au side", 2)]
refv = {"tot_4_8": "60.2", "tot_8_12": "163.5", "conc": "53.3", "sum_r6_all": "2.10"}
L = [r"\begin{tabular}{lccc}", r"\toprule", r"Quantity & 100 K & 300 K & Homogeneous solvent$^a$\\", r"\midrule"]
for k, lab, nd in rows:
    sc = 1e3 if k == "sum_r6_all" else 1
    g = r[k].groupby(level="T")
    L.append(lab + " & " + " & ".join(f"${g.mean()[T]*sc:.{nd}f}\\pm{g.std()[T]*sc:.{nd}f}$" for T in (100, 300)) + f" & {refv.get(k, '--')}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_spin_bath.tex"), "w").write("\n".join(L) + "\n")

dp = pd.read_csv(os.path.join(D, "module2_dipolar_by_replica.csv"))
L = [r"\begin{tabular}{llccccc}", r"\toprule",
     r"$T$ (K) & Field & $\sqrt{M_2}$ (MHz) & $\sigma_d$ (MHz) & $\tau_d$ (ps) & $\xi=2\pi\sigma_d\tau_d$ & Pairs $<8$~\AA\\", r"\midrule"]
for (T, f), d in dp.groupby(["T", "field"]):
    L.append(f"{T} & " + ("$B_0\\parallel n$" if f == "normal" else "$B_0\\perp n$") +
             f" & ${d.sqrtM2_median_MHz.mean():.0f}\\pm{d.sqrtM2_median_MHz.std():.0f}$ & ${d.sigma_d_median_MHz.mean():.0f}\\pm{d.sigma_d_median_MHz.std():.0f}$"
             f" & ${d.tau_d_median_ns.mean()*1e3:.0f}\\pm{d.tau_d_median_ns.std()*1e3:.0f}$ & ${d.xi_median.mean():.3f}\\pm{d.xi_median.std():.3f}$ & ${d.n_pairs.mean():.0f}$\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_dipolar.tex"), "w").write("\n".join(L) + "\n")
print("done")
