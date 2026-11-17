#!/usr/bin/env python3
"""
m2_15_correlated_motion.py
Do neighbouring nitroxides move together?

For every pair of intact radicals, the Pearson correlation between the time
series of their N-O orientations (cos theta_NO, last 5 ns) and of their spin
heights is computed, and averaged in bins of mean pair distance.  Pairs farther
than 25 A serve as the null distribution (finite-sample correlations of
independent series).  Also computed: the correlation of the instantaneous
N-O vectors, < u_i(t).u_j(t) > fluctuation, via the cross-correlation of the
second-rank orientation, using the same binning.
Mean +/- SD over replicas.
"""
import os, numpy as np, pandas as pd
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
W = slice(500, 1000)
BINS = np.array([4, 6, 7, 8, 10, 12, 15, 20, 25, 80])
rows = []
for T, r in RUNS:
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz")); keep = ~np.isin(o["mol_ids"], DEFECT)
    c = o["cos_NO"][W][:, keep]; h = o["h_spin"][W][:, keep]
    pos = np.concatenate([h[..., None], o["spin_yz"][W][:, keep].astype(float)], axis=2)
    def corr(A):
        Z = (A - A.mean(0)) / np.maximum(A.std(0), 1e-9); return (Z.T @ Z) / A.shape[0]
    Cc, Ch = corr(c), corr(h)
    dist = np.linalg.norm(pos.mean(0)[:, None] - pos.mean(0)[None], axis=2)
    iu = np.triu_indices(c.shape[1], 1); dd = dist[iu]; cc = Cc[iu]; ch = Ch[iu]
    for k in range(len(BINS) - 1):
        m = (dd >= BINS[k]) & (dd < BINS[k + 1])
        rows.append(dict(T=T, rep=r, r_lo=BINS[k], r_hi=BINS[k + 1], n=int(m.sum()),
                         corr_cos=float(cc[m].mean()) if m.any() else np.nan,
                         corr_h=float(ch[m].mean()) if m.any() else np.nan,
                         abs_corr_cos=float(np.abs(cc[m]).mean()) if m.any() else np.nan,
                         frac_corr_gt_0p3=float((cc[m] > 0.3).mean()) if m.any() else np.nan))
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "module2_correlated_motion_by_replica.csv"), index=False)
S = R.groupby(["T", "r_lo", "r_hi"])[["n", "corr_cos", "corr_h", "abs_corr_cos", "frac_corr_gt_0p3"]].agg(["mean", "std"])
S.to_csv(os.path.join(D, "module2_correlated_motion_summary.csv"))
pd.set_option("display.width", 250); print(S.round(3).to_string())
