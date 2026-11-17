#!/usr/bin/env python3
"""
m3_08_sensitivity.py
Robustness of the network statistics to (i) the resolution threshold J_RES,
(ii) interior vs all radicals (edge distance >= 5 A, Module II definition),
(iii) the distance cutoff (pairs whose closest N/O...N/O distance stays > 8 A).
Output: data/module3/exchange_sensitivity.csv
"""
import os, numpy as np, pandas as pd
from m3_ml_common import D
lab = pd.read_csv(os.path.join(D, "..", "module2", "module2_class_labels.csv"))
rows = []
for T in (300, 100):
    for r in (1, 2, 3):
        P = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv")); P = P[P.intact]
        L = lab[(lab["T"] == T) & (lab.rep == r)].set_index("mol")
        mols = L.index.values; interior = set(L.index[L.edge_dist >= 5])
        for jr in (0.05, 0.10, 0.20, 0.30):
            for scope in ("all", "interior"):
                for rc in (9.0, 8.0):
                    Q = P[(P.Jabs_mean > jr) & (P.rNO_min < rc)]
                    S = mols if scope == "all" else np.array(sorted(interior))
                    deg = pd.Series(0, index=S, dtype=float)
                    for a, b in zip(Q.mol_i, Q.mol_j):
                        if a in deg.index: deg[a] += 1
                        if b in deg.index: deg[b] += 1
                    rows.append(dict(T=T, rep=r, J_RES=jr, scope=scope, rcut=rc, n_pairs=len(Q), degree=deg.mean(),
                                     frac_isolated=(deg == 0).mean(), frac_AF=Q.pAF_w.mean(), Jmedian=Q.Jabs_mean.median()))
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "exchange_sensitivity_by_replica.csv"), index=False)
A = R.drop(columns="rep").groupby(["T", "J_RES", "scope", "rcut"]).agg(["mean", "std"]).round(3)
A.to_csv(os.path.join(D, "exchange_sensitivity.csv")); print(A.to_string())
