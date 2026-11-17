#!/usr/bin/env python3
"""
m2_07_pairs.py
Geometry of neighbouring radical pairs (input for Module III) -- last 5 ns.
Spin position = N-O midpoint.  Only intact radicals are used, and the
14 defective radicals are reported separately.

Per run:
  * histogram of all pair distances r < 12 A (spin-spin) and O1...O1
  * nearest-neighbour distance distribution
  * relative orientation of the two N->O vectors for pairs with r < 8 A
  * point-dipole coupling of the nearest neighbour,
        nu_dd = 52.04 MHz nm^3 / r^3   (g = 2.0023, secular coefficient)
  * nearest-neighbour persistence: probability that the nearest neighbour of a
    radical at time t is the same molecule at time t + tau
"""
import os, numpy as np, pandas as pd
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
W = slice(500, 1000)
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res, lin = P["name"], P["resid"], P["resname"] == "LIN"
iO = np.where(lin & (names == "O1"))[0]; iN = np.where(lin & (names == "N1"))[0]
bins = np.arange(2.0, 12.01, 0.1); centers = 0.5 * (bins[1:] + bins[:-1])
H, rows, pers = {}, [], {}
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])[100:][W]
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    intact = ~np.isin(o["mol_ids"], DEFECT)
    O, N = X[:, iO].astype(float), X[:, iN].astype(float)
    spin = 0.5 * (O + N); u = O - N; u /= np.linalg.norm(u, axis=2, keepdims=True)
    nf, nm = spin.shape[:2]
    iu = np.triu_indices(nm, 1)
    both = intact[iu[0]] & intact[iu[1]]
    hs, ho = np.zeros(len(centers)), np.zeros(len(centers)); angs = []
    nn_id = np.empty((nf, nm), int); nn_d = np.empty((nf, nm))
    for f in range(nf):
        Ds = np.linalg.norm(spin[f][:, None] - spin[f][None], axis=2)
        Do = np.linalg.norm(O[f][:, None] - O[f][None], axis=2)
        ds, do = Ds[iu][both], Do[iu][both]
        hs += np.histogram(ds, bins)[0]; ho += np.histogram(do, bins)[0]
        close = ds < 8.0
        cu = (u[f][iu[0]][both][close] * u[f][iu[1]][both][close]).sum(1)
        angs.append(np.degrees(np.arccos(np.clip(cu, -1, 1))))
        Dm = Ds.copy(); np.fill_diagonal(Dm, np.inf); Dm[:, ~intact] = np.inf
        nn_id[f] = Dm.argmin(1); nn_d[f] = Dm.min(1)
    angs = np.concatenate(angs)
    H[f"T{T}_r{r}_spin"] = hs / nf; H[f"T{T}_r{r}_OO"] = ho / nf
    H[f"T{T}_r{r}_angle"] = np.histogram(angs, np.arange(0, 181, 5))[0] / len(angs)
    nnd = nn_d[:, intact]
    nu_dd = 52.04 / (nnd / 10.0) ** 3                       # MHz
    lags = np.arange(0, 201)
    same = [np.mean(nn_id[:nf - L, intact] == nn_id[L:, intact]) for L in lags]
    pers[f"T{T}_r{r}"] = np.array(same)
    rows.append(dict(T=T, rep=r,
        pairs_lt5=hs[centers < 5].sum() / nf, pairs_lt6=hs[centers < 6].sum() / nf,
        pairs_lt8=hs[centers < 8].sum() / nf,
        OO_pairs_lt4=ho[centers < 4].sum() / nf, OO_pairs_lt5=ho[centers < 5].sum() / nf,
        nn_mean=nnd.mean(), nn_p10=np.percentile(nnd, 10), nn_p90=np.percentile(nnd, 90),
        nu_dd_median_MHz=np.median(nu_dd), nu_dd_p90_MHz=np.percentile(nu_dd, 90),
        angle_mean=angs.mean(), frac_parallel_lt45=(angs < 45).mean(), frac_anti_gt135=(angs > 135).mean(),
        nn_persist_1ns=same[100], nn_persist_2ns=same[200]))
    print(rows[-1])
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "module2_pairs_by_replica.csv"), index=False)
S = R.drop(columns="rep").groupby("T").agg(["mean", "std"]); S.to_csv(os.path.join(D, "module2_pairs_summary.csv"))
np.savez_compressed(os.path.join(D, "module2_pair_histograms.npz"), r=centers, angle=np.arange(2.5, 180, 5),
                    lag_ns=np.arange(0, 201) * 0.01, **H, **{"persist_" + k: v for k, v in pers.items()})
pd.set_option("display.width", 250); print(S.round(3).T.to_string())
