#!/usr/bin/env python3
"""m2_10_tables.py -- LaTeX tables for Module II (SI) from the CSV outputs."""
import os, numpy as np, pandas as pd
from m2_00_io import ROOT
D = os.path.join(ROOT, "data", "module2"); OUT = os.path.join(ROOT, "tables")
s = pd.read_csv(os.path.join(D, "module2_structure_summary.csv")); s = s[s.window == "last5"]
obs = [("h_spin (A)", "Height of N--O midpoint (\\AA)", 1), ("tilt (deg)", "Chain tilt ($^\\circ$)", 1),
       ("gauche", "Gauche fraction", 3), ("theta_NO (deg)", "$\\langle\\theta_\\mathrm{NO}\\rangle$ ($^\\circ$)", 1),
       ("frac_NO_up (theta<60)", "Fraction $\\theta_\\mathrm{NO}<60^\\circ$", 3),
       ("frac_NO_down (theta>120)", "Fraction $\\theta_\\mathrm{NO}>120^\\circ$", 3),
       ("contact_frac (O-Au<3.5)", "Fraction with $d_\\mathrm{O\\cdots Au}<3.5$~\\AA", 3),
       ("molecules_in_contact", "Radicals in contact ($>50\\%$ of time)", 1),
       ("nn1 (A)", "Nearest spin--spin distance (\\AA)", 2), ("n_within_8A", "Spins within 8~\\AA", 2)]
def cell(g, o, T, nd):
    r = s[(s.group == g) & (s.observable == o) & (s["T"] == T)].iloc[0]
    return f"${r['mean']:.{nd}f}\\pm{r['sd']:.{nd}f}$"
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{2}{c}{interior} & \multicolumn{2}{c}{edge} & \multicolumn{2}{c}{all}\\",
     r"Observable & 100 K & 300 K & 100 K & 300 K & 100 K & 300 K\\", r"\midrule"]
for o, lab, nd in obs:
    L.append(lab + " & " + " & ".join(cell(g, o, T, nd) for g in ["intact_interior", "edge", "all"] for T in (100, 300)) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_MD_structure.tex"), "w").write("\n".join(L) + "\n")

dsum = pd.read_csv(os.path.join(D, "module2_dynamics_by_replica.csv"))
g = dsum.groupby(["T", "group"]).agg(["mean", "std"])
rowsD = [("S2", "$\\langle S^2\\rangle$", 3), ("theta0", "Cone semi-angle $\\theta_0$ ($^\\circ$)", 1),
         ("C2_10ps", "$C_2(10\\,\\mathrm{ps})$", 3), ("tau_e_ns", "Median $\\tau_e$ (ns)", 2),
         ("frac_S2_gt_0p8", "Fraction $S^2>0.8$", 3), ("rmsf", "RMSF of N--O midpoint (\\AA)", 2)]
L = [r"\begin{tabular}{lcccc}", r"\toprule", r" & \multicolumn{2}{c}{interior} & \multicolumn{2}{c}{edge}\\",
     r"Quantity & 100 K & 300 K & 100 K & 300 K\\", r"\midrule"]
for k, lab, nd in rowsD:
    L.append(lab + " & " + " & ".join(f"${g.loc[(T, gr), (k, 'mean')]:.{nd}f}\\pm{g.loc[(T, gr), (k, 'std')]:.{nd}f}$"
                                       for gr in ["intact_interior", "edge"] for T in (100, 300)) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_MD_dynamics.tex"), "w").write("\n".join(L) + "\n")

p = pd.read_csv(os.path.join(D, "module2_pairs_by_replica.csv")).drop(columns="rep").groupby("T").agg(["mean", "std"])
rowsP = [("nn_mean", "Mean nearest-neighbour distance (\\AA)", 2), ("pairs_lt6", "Pairs with $r<6$~\\AA", 1),
         ("pairs_lt8", "Pairs with $r<8$~\\AA", 1), ("OO_pairs_lt5", "Pairs with $d_\\mathrm{O\\cdots O}<5$~\\AA", 1),
         ("nu_dd_median_MHz", "Median nearest-neighbour $\\nu_\\mathrm{dd}$ (MHz)", 0),
         ("frac_parallel_lt45", "Fraction of pairs ($r<8$~\\AA) with N--O angle $<45^\\circ$", 3),
         ("frac_anti_gt135", "Fraction with N--O angle $>135^\\circ$", 3),
         ("nn_persist_1ns", "Same nearest neighbour after 1 ns", 3), ("nn_persist_2ns", "Same nearest neighbour after 2 ns", 3)]
L = [r"\begin{tabular}{lcc}", r"\toprule", r"Quantity (all 141 radicals) & 100 K & 300 K\\", r"\midrule"]
for k, lab, nd in rowsP:
    L.append(lab + " & " + " & ".join(f"${p.loc[T, (k, 'mean')]:.{nd}f}\\pm{p.loc[T, (k, 'std')]:.{nd}f}$" for T in (100, 300)) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_MD_pairs.tex"), "w").write("\n".join(L) + "\n")

mx = pd.read_csv(os.path.join(D, "module2_S2_mixture.csv"))
L = [r"\begin{tabular}{ccrrrrr}", r"\toprule", r"$T$ (K) & Replica & $\Delta$BIC$_{2-1}$ & $\mu_1$ ($^\circ$) & $w_1$ & $\mu_2$ ($^\circ$) & $w_2$\\", r"\midrule"]
for r in mx.itertuples():
    L.append(f"{r.T} & {r.rep} & {r.dBIC_2vs1:.1f} & {r.mu_low:.1f} & {r.w_low:.2f} & {r.mu_high:.1f} & {r.w_high:.2f}\\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_MD_mixture.tex"), "w").write("\n".join(L) + "\n")
print("tables written")

mc = pd.read_csv(os.path.join(D, "module2_mobility_correlations.csv"))
lab = {"n8": "Spins within 8~\\AA", "nn1": "Nearest spin distance", "gauche": "Chain gauche fraction",
       "tilt": "Chain tilt", "h_spin": "Spin height", "cosNO": "$\\cos\\theta_\\mathrm{NO}$"}
L = [r"\begin{tabular}{lcccccc}", r"\toprule", r" & \multicolumn{3}{c}{100 K} & \multicolumn{3}{c}{300 K}\\",
     r"Descriptor & rep 1 & rep 2 & rep 3 & rep 1 & rep 2 & rep 3\\", r"\midrule"]
for k in ["n8", "nn1", "gauche", "tilt", "h_spin", "cosNO"]:
    v = [mc[(mc.descriptor == k) & (mc["T"] == T) & (mc.rep == r)].rho.iloc[0] for T in (100, 300) for r in (1, 2, 3)]
    L.append(lab[k] + " & " + " & ".join(f"${x:+.2f}$" for x in v) + r"\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tabS_MD_corr.tex"), "w").write("\n".join(L) + "\n")
