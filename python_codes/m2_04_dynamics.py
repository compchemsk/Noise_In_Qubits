#!/usr/bin/env python3
"""
m2_04_dynamics.py
Reorientational and positional dynamics of the nitroxide groups (last 5 ns,
10 ps resolution).  The Au slab is restrained, so the laboratory frame is the
surface frame.

Per molecule:
  C2(t)  = < P2( u(0).u(t) ) >  for the N->O unit vector u
  S2     = (3/2) sum_ij <u_i u_j>^2 - 1/2      (generalized order parameter;
           long-time limit of C2 for motion restricted on the 5-ns window)
  tau_e  = effective correlation time from a fit of
           C2(t) = S2 + (A - S2) exp(-t/tau_e), 10 ps <= t <= 1 ns
           (A < 1 absorbs librations faster than the 10-ps sampling)
  theta0 = semi-angle of the equivalent diffusion-in-a-cone model,
           S2 = [cos t0 (1 + cos t0) / 2]^2
  RMSF   = root-mean-square positional fluctuation of the N-O midpoint (A)
Statistics: replica means +/- SD over the three replicas, for the groups
defined in m2_03.
"""
import os, numpy as np, pandas as pd
from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
W = slice(500, 1000); DT = 0.010          # ns
LAGS = np.arange(0, 201)                   # 0-2 ns

def c2_curve(u):
    nf = u.shape[0]; out = np.empty(len(LAGS))
    for k, L in enumerate(LAGS):
        dot = (u[:nf - L] * u[L:]).sum(-1)
        out[k] = (1.5 * dot**2 - 0.5).mean()
    return out

def S2_of(u):
    M = np.einsum("ti,tj->ij", u, u) / len(u)
    return 1.5 * (M**2).sum() - 0.5

def theta0(S2):
    S = np.sqrt(np.clip(S2, 0, 1))
    c = (-1 + np.sqrt(1 + 8 * S)) / 2          # solves c(1+c)/2 = S
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

rows, curves = [], {}
for T, r in RUNS:
    d = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    S = d["S_yz"].mean(0); hull = ConvexHull(S)
    dist = np.min(-(S @ hull.equations[:, :2].T + hull.equations[:, 2]), axis=1)
    intact = ~np.isin(d["mol_ids"], DEFECT)
    u = d["u_NO"][W].astype(float)
    spin = np.concatenate([d["h_spin"][W][..., None], d["spin_yz"][W].astype(float)], axis=2)
    C = np.empty((u.shape[1], len(LAGS)))
    for m in range(u.shape[1]):
        C[m] = c2_curve(u[:, m])
        s2 = S2_of(u[:, m])
        t = LAGS[1:101] * DT
        try:
            p, _ = curve_fit(lambda t, A, tau: s2 + (A - s2) * np.exp(-t / tau), t, C[m, 1:101],
                             p0=[min(0.95, C[m, 1] + 0.02), 0.1], bounds=([s2, 1e-3], [1.0, 50]))
            A, tau = p
        except Exception:
            A, tau = np.nan, np.nan
        rmsf = np.sqrt(((spin[:, m] - spin[:, m].mean(0))**2).sum(1).mean())
        rows.append(dict(T=T, rep=r, mol=int(d["mol_ids"][m]), interior=bool(dist[m] >= 5),
                         intact=bool(intact[m]), S2=s2, theta0=theta0(s2), A_fast=A, tau_e_ns=tau,
                         C2_10ps=C[m, 1], rmsf_spin=rmsf))
    curves[f"T{T}_r{r}"] = C
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "module2_dynamics_by_molecule.csv"), index=False)
np.savez_compressed(os.path.join(D, "module2_C2_curves.npz"), lag_ns=LAGS * DT, **curves)

def grp(df):
    out = {}
    out["all"] = df
    out["interior"] = df[df.interior]; out["edge"] = df[~df.interior]
    out["intact_interior"] = df[df.interior & df.intact]
    return out
rep_rows = []
for (T, r), d in df.groupby(["T", "rep"]):
    for g, s in grp(d).items():
        rep_rows.append(dict(T=T, rep=r, group=g, S2=s.S2.mean(), S2_median=s.S2.median(),
                             theta0=s.theta0.mean(), tau_e_ns=s.tau_e_ns.median(), A_fast=s.A_fast.mean(),
                             C2_10ps=s.C2_10ps.mean(), rmsf=s.rmsf_spin.mean(),
                             frac_S2_gt_0p8=(s.S2 > 0.8).mean(), frac_S2_lt_0p4=(s.S2 < 0.4).mean()))
R = pd.DataFrame(rep_rows); R.to_csv(os.path.join(D, "module2_dynamics_by_replica.csv"), index=False)
summ = R.groupby(["T", "group"]).agg(["mean", "std"]).drop(columns="rep")
summ.to_csv(os.path.join(D, "module2_dynamics_summary.csv"))
pd.set_option("display.width", 250); print(summ.round(3).to_string())
