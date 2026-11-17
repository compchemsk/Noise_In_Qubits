#!/usr/bin/env python3
"""
m2_05_S2_populations.py
Is the distribution of per-radical order parameters bimodal?  Gaussian
mixtures with 1-3 components are fitted to the cone semi-angles theta0
(interior, intact radicals), separately for each replica, and compared by BIC.
"""
import os, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from m2_00_io import ROOT
D = os.path.join(ROOT, "data", "module2")
df = pd.read_csv(os.path.join(D, "module2_dynamics_by_molecule.csv"))
rows = []
for (T, r), d in df[df.interior & df.intact].groupby(["T", "rep"]):
    x = d.theta0.to_numpy()[:, None]
    bics = {}
    for k in (1, 2, 3):
        g = GaussianMixture(k, n_init=20, random_state=0).fit(x); bics[k] = g.bic(x)
        if k == 2:
            o = np.argsort(g.means_.ravel())
            mu = g.means_.ravel()[o]; w = g.weights_[o]; sd = np.sqrt(g.covariances_.ravel()[o])
    rows.append(dict(T=T, rep=r, n=len(x), BIC1=bics[1], BIC2=bics[2], BIC3=bics[3],
                     dBIC_2vs1=bics[2] - bics[1], mu_low=mu[0], sd_low=sd[0], w_low=w[0],
                     mu_high=mu[1], sd_high=sd[1], w_high=w[1]))
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "module2_S2_mixture.csv"), index=False)
pd.set_option("display.width", 250); print(R.round(2).to_string())
print(R.groupby("T")[["dBIC_2vs1","mu_low","w_low","mu_high","w_high"]].agg(["mean","std"]).round(2))
