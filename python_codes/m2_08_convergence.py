#!/usr/bin/env python3
"""
m2_08_convergence.py
Time series for convergence: heavy-atom RMSD of the monolayer relative to the
first unrestrained frame, and 0.5-ns block averages of spin height, chain tilt,
gauche fraction and number of radicals in contact with Au, for every run.
"""
import os, numpy as np, pandas as pd
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
lin = P["resname"] == "LIN"; heavy = lin & ~np.char.startswith(P["name"].astype(str), "H")
out = {}; rows = []
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])
    ref = X[100, heavy]
    rmsd = np.sqrt(((X[:, heavy] - ref) ** 2).sum(2).mean(1))
    out[f"rmsd_T{T}_r{r}"] = rmsd
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    for b in range(20):
        s = slice(b * 50, (b + 1) * 50)
        rows.append(dict(T=T, rep=r, t_ns=(b + 0.5) * 0.5, h_spin=o["h_spin"][s].mean(), tilt=o["tilt"][s].mean(),
                         gauche=o["gauche"][s].mean(), n_contact=(o["dmin_OAu"][s] < 3.5).mean(0).__gt__(0.5).sum()))
np.savez_compressed(os.path.join(D, "module2_rmsd.npz"), t_ns=(np.arange(1100) - 100) * 0.01, **out)
pd.DataFrame(rows).to_csv(os.path.join(D, "module2_block_timeseries.csv"), index=False)
print({k: (v[100:600].mean().round(2), v[600:].mean().round(2)) for k, v in out.items()})
