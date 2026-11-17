#!/usr/bin/env python3
"""
m2_06_mobility_determinants.py
Per-radical correlation (Spearman, per replica) of the cone semi-angle theta0
with local structural descriptors averaged over the same 5-ns window:
number of spins within 8 A, nearest-spin distance, chain gauche fraction,
chain tilt, spin height and N-O orientation.  Interior, intact radicals.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
dyn = pd.read_csv(os.path.join(D, "module2_dynamics_by_molecule.csv"))
W = slice(500, 1000); rows = []
for T, r in RUNS:
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    desc = pd.DataFrame(dict(mol=o["mol_ids"], n8=o["ncount_8"][W].mean(0), nn1=o["nn1"][W].mean(0),
        gauche=o["gauche"][W].mean(0), tilt=o["tilt"][W].mean(0), h_spin=o["h_spin"][W].mean(0),
        cosNO=o["cos_NO"][W].mean(0)))
    m = dyn[(dyn["T"] == T) & (dyn.rep == r)].merge(desc, on="mol")
    m = m[m.interior & m.intact]
    for c in ["n8", "nn1", "gauche", "tilt", "h_spin", "cosNO"]:
        rho, p = spearmanr(m.theta0, m[c]); rows.append(dict(T=T, rep=r, descriptor=c, rho=rho, p=p, n=len(m)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "module2_mobility_correlations.csv"), index=False)
print(R.pivot_table(index="descriptor", columns=["T", "rep"], values="rho").round(2).to_string())
print(R.groupby(["T", "descriptor"]).agg(rho_mean=("rho", "mean"), rho_sd=("rho", "std"), p_max=("p", "max")).round(3))
