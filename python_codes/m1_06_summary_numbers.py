#!/usr/bin/env python3
"""m1_06_summary_numbers.py -- every number quoted in the Module I text, computed here."""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data","module1")
s=pd.read_csv(os.path.join(DATA,"module1_coupling_summary.csv"))
f=pd.read_csv(os.path.join(DATA,"module1_frequencies.csv"))
s=s.merge(f[["molecule","mode","freq_cm1"]],on=["molecule","mode"],how="left")
out=[]
for mol,d in s.groupby("molecule"):
    out.append(dict(molecule=mol, n_modes=len(d),
        g0=d.g0.iloc[0], A0=d.A0_MHz.median(),
        sumA2=(d.dA_dQ_MHz**2).sum(), maxA=d.dA_dQ_MHz.abs().max(),
        maxA_mode=int(d.loc[d.dA_dQ_MHz.abs().idxmax(),"mode"]),
        sum_gQ2=(d.dnu_g_Q_MHz**2).sum(), max_gQ=d.dnu_g_Q_MHz.abs().max(),
        max_dg=d.dg_dQ.abs().max(),
        n_g_gt_A_Qband=int((d.dnu_g_Q_MHz.abs()>d.dA_dQ_MHz.abs()).sum()),
        n_g_gt_A_Xband=int((d.dnu_g_X_MHz.abs()>d.dA_dQ_MHz.abs()).sum()),
        min_R2=min(d.R2_g.min(),d.R2_A.min())))
o=pd.DataFrame(out); pd.set_option("display.width",250); print(o.to_string())
o.to_csv(os.path.join(DATA,"module1_molecule_summary.csv"),index=False)
fits=pd.read_csv(os.path.join(DATA,"module1_fits_all.csv"))
print("\nmax |c3|/|c1| where |c1| meaningful (A>0.05 MHz, g>5e-6):")
m=((fits.quantity=="A_iso_MHz")&(fits.c1.abs()>0.05))|((fits.quantity=="g_iso")&(fits.c1.abs()>5e-6))
print((fits[m].c3.abs()/fits[m].c1.abs()).max())
print("cd(0.2) vs c1 max rel dev (same mask):",((fits[m].cd_0p2-fits[m].c1).abs()/fits[m].c1.abs()).max())
print("\nTEMPO/PTEMPO table:"); print(s[s.molecule!="PTEMPO-alkyl"][["molecule","mode","freq_cm1","dg_dQ","dnu_g_Q_MHz","dA_dQ_MHz","d2A_dQ2_half_MHz","d2g_dQ2_half"]].to_string())
c=pd.read_csv(os.path.join(DATA,"module1_T1_model_comparison.csv")); print(c[["dataset","model","chi2","chi2_red","params","param_se","dAICc"]].to_string())
