#!/usr/bin/env python3
"""m1_07_latex_tables.py -- writes the Module I SI tables (LaTeX) from the CSV outputs."""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data","module1")
OUT=os.path.join(HERE,"..","tables"); os.makedirs(OUT,exist_ok=True)
s=pd.read_csv(os.path.join(DATA,"module1_coupling_summary.csv"))
fr=pd.read_csv(os.path.join(DATA,"module1_frequencies.csv"))
s=s.merge(fr[["molecule","mode","freq_cm1"]],on=["molecule","mode"],how="left")
def pm(v,e,scale=1,nd=3):
    v,e=v*scale,e*scale
    return f"${v:.{nd}f}\\,({e:.{nd}f})$" if e>=10**(-nd) else f"${v:.{nd}f}$"
lines=[r"\begin{tabular}{llrrrrrr}",r"\toprule",
 r"Molecule & Mode & $\tilde\nu$ (cm$^{-1}$) & $\partial g/\partial Q$ ($10^{-5}$) & $\partial\nu_g/\partial Q$ (MHz)$^a$ & $\partial A/\partial Q$ (MHz) & $\tfrac12\partial^2A/\partial Q^2$ (MHz) & $R^2_g$ / $R^2_A$\\",r"\midrule"]
for mol in ["TEMPO","PTEMPO","PTEMPO-alkyl"]:
    d=s[s.molecule==mol].sort_values("mode")
    for i,r in enumerate(d.itertuples()):
        nu="--" if np.isnan(r.freq_cm1) else f"{r.freq_cm1:.1f}"
        lines.append(f"{mol if i==0 else ''} & {r.mode} & {nu} & {pm(r.dg_dQ,r.dg_dQ_se,1e5,3)} & ${r.dnu_g_Q_MHz:.3f}$ & {pm(r.dA_dQ_MHz,r.dA_dQ_se)} & {pm(r.d2A_dQ2_half_MHz,r.d2A_se)} & {r.R2_g:.4f} / {r.R2_A:.4f}\\\\")
    lines.append(r"\midrule")
lines[-1]=r"\bottomrule"; lines.append(r"\end{tabular}")
open(os.path.join(OUT,"tabS_couplings.tex"),"w").write("\n".join(lines)+"\n")
c=pd.read_csv(os.path.join(DATA,"module1_T1_model_comparison.csv"))
names={"M_loc":"direct + one local mode","M_ram":r"direct + Raman $bT^m$","M_A":r"direct + DFT modes, $w_k\propto(\partial A/\partial Q_k)^2$","M_g":r"direct + DFT modes, $w_k\propto(\partial\nu_g/\partial Q_k)^2$","M_Ag":r"direct + DFT modes, both"}
dsn={"PTEMPO_slow":"PTEMPO (slow)","PTEMPO_fast":"PTEMPO (fast)","SAM":"TEMPO SAM"}
L=[r"\begin{tabular}{llrrrl}",r"\toprule",r"Data set & Model & $k$ & $\chi^2$ & $\Delta$AICc & Key parameter\\",r"\midrule"]
for ds in ["PTEMPO_slow","PTEMPO_fast","SAM"]:
    d=c[c.dataset==ds]
    for i,r in enumerate(d.itertuples()):
        p=np.array(r.params.strip("[]").split(),float); e=np.array(r.param_se.strip("[]").split(),float)
        key = f"$\\Delta = {p[2]:.0f}\\pm{e[2]:.0f}$ cm$^{{-1}}$" if r.model=="M_loc" else (f"$m = {p[2]:.1f}\\pm{e[2]:.1f}$" if r.model=="M_ram" else "--")
        L.append(f"{dsn[ds] if i==0 else ''} & {names[r.model]} & {r.k} & {r.chi2:.1f} & {r.dAICc:.1f} & {key}\\\\")
    L.append(r"\midrule")
L[-1]=r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(OUT,"tabS_T1_models.tex"),"w").write("\n".join(L)+"\n")
print(open(os.path.join(OUT,"tabS_T1_models.tex")).read())

# tabS_thermal.tex (first- and second-order terms) is written by m1_16_c1_c2.py


mc=pd.read_csv(os.path.join(DATA,"module1_mode_character.csv"))
mc=mc[mc.molecule!="PTEMPO-alkyl"]
alk=pd.read_csv(os.path.join(DATA,"module1_mode_composition_alkyl.csv"))      # from m1_15_alkyl_composition.py
cs=pd.read_csv(os.path.join(DATA,"module1_coupling_summary.csv")); cs=cs[cs.molecule=="PTEMPO-alkyl"]
alk=alk.merge(cs[["mode","dA_dQ_MHz","dg_dQ"]],on="mode"); alk["absA"]=alk.dA_dQ_MHz.abs(); alk["absg"]=alk.dg_dQ.abs()
f3=lambda v: "--" if pd.isna(v) else f"{v:.3f}"
L=[r"\begin{tabular}{llrrrrrrrrrrr}",r"\toprule",
   r"Molecule & Mode & $\tilde\nu$ (cm$^{-1}$) & NO & Ring & Me(H) & Me(C) & Prop./Link & Trz & Chain & Other & $|\partial A/\partial Q|$ (MHz) & $|\partial g/\partial Q|$ ($10^{-5}$)\\",r"\midrule"]
for mol in ["TEMPO","PTEMPO"]:
    d=mc[mc.molecule==mol].sort_values("mode")
    for i,r in enumerate(d.itertuples()):
        prop=0.0 if pd.isna(r.Prop) else r.Prop
        other=1-(r.NO+r.Ring+r.MetH+r.MetC+prop)
        L.append(f"{mol if i==0 else ''} & {r.mode} & {r.freq:.1f} & {r.NO:.4f} & {r.Ring:.3f} & {r.MetH:.3f} & {r.MetC:.3f} & {f3(r.Prop)} & -- & -- & {other:.3f} & {r.absA:.3f} & {r.absg*1e5:.3f}\\\\")
    L.append(r"\midrule")
for i,r in enumerate(alk.sort_values("mode").itertuples()):
    L.append(f"{'PTEMPO-alkyl' if i==0 else ''} & {r.mode} & {r.freq:.1f} & {r.NO:.4f} & {r.Ring:.3f} & {r.MetH:.3f} & {r.MetC:.3f} & {r.Link:.3f} & {r.Trz:.3f} & {r.Chain:.3f} & {r.Other:.3f} & {r.absA:.3f} & {r.absg*1e5:.3f}\\\\")
L.append(r"\bottomrule"); L.append(r"\end{tabular}")
open(os.path.join(OUT,"tabS_composition.tex"),"w").write("\n".join(L)+"\n")
