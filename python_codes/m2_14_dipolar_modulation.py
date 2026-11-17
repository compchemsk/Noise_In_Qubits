#!/usr/bin/env python3
"""
m2_14_dipolar_modulation.py
Motional modulation of the electron-electron dipolar couplings.

For each pair of intact radicals the secular point-dipole coupling is
    d_ij(t) = nu_dd(r_ij) (1 - 3 cos^2 beta_ij),  nu_dd = 52.04 MHz nm^3 / r^3,
where beta_ij is the angle between r_ij and the static field B0.  Two field
orientations are considered: B0 along the surface normal (x) and B0 in the
surface plane (average of y and z).
Per radical: local dipolar second moment M2_i(t) = sum_j d_ij(t)^2 over all
intact partners within 15 A; reported as sqrt(<M2>) (MHz).
Per pair (mean distance < 8 A over the window): amplitude of the fluctuation
sigma_d = std_t[d_ij(t)], its normalized autocorrelation C_d(t), and the
integrated correlation time tau_d (integral of C_d up to its first zero or
2 ns).  The motional-narrowing parameter xi = 2 pi sigma_d tau_d indicates
whether the modulation is fast (xi << 1, averaging) or slow (xi >> 1, quasi-
static) on the scale of the coupling.
Analysis window: last 5 ns, 10-ps resolution.  Mean +/- SD over replicas.
"""
import os, numpy as np, pandas as pd
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res, lin = P["name"], P["resid"], P["resname"] == "LIN"
mols = np.unique(res[lin]); intact = ~np.isin(mols, DEFECT)
iO = np.where(lin & (names == "O1"))[0][intact]; iN = np.where(lin & (names == "N1"))[0][intact]
DIRS = {"normal": [np.array([1.0, 0, 0])], "inplane": [np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]}
rows, prow, acfs = [], [], {}
LAG = np.arange(0, 201)
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])[100:][500:1000]
    spin = 0.5 * (X[:, iO] + X[:, iN]).astype(float)               # (t, m, 3)
    nf, nm = spin.shape[:2]
    iu = np.triu_indices(nm, 1)
    rij = spin[:, iu[1]] - spin[:, iu[0]]                           # (t, pairs, 3)
    dist = np.linalg.norm(rij, axis=2); nu = 52.04 / (dist / 10) ** 3
    close = dist.mean(0) < 8.0; within = dist < 15.0
    for key, dirs in DIRS.items():
        M2s, sig, tau, xi, C_all = [], [], [], [], []
        for b0 in dirs:
            c = (rij @ b0) / dist
            d = nu * (1 - 3 * c**2)                                  # MHz, (t, pairs)
            dd = np.where(within, d, 0.0) ** 2
            M2 = np.zeros((nf, nm)); np.add.at(M2.T, iu[0], dd.T); np.add.at(M2.T, iu[1], dd.T)
            M2s.append(np.sqrt(M2.mean(0)))
            dc = d[:, close]; dc = dc - dc.mean(0); s = dc.std(0)
            C = np.array([(dc[:nf - L] * dc[L:]).mean(0) for L in LAG]) / np.maximum(s**2, 1e-12)
            C_all.append(C.mean(1))
            tk = []
            for k in range(C.shape[1]):
                z = np.where(C[:, k] <= 0)[0]; end = z[0] if len(z) else len(LAG)
                tk.append(np.trapezoid(C[:end, k], dx=0.01))          # ns
            tk = np.array(tk); sig.append(s); tau.append(tk); xi.append(2 * np.pi * s * 1e6 * tk * 1e-9)
        M2r = np.mean(M2s, axis=0); s_ = np.concatenate(sig); t_ = np.concatenate(tau); x_ = np.concatenate(xi)
        acfs[f"T{T}_r{r}_{key}"] = np.mean(C_all, axis=0)
        rows.append(dict(T=T, rep=r, field=key, sqrtM2_median_MHz=float(np.median(M2r)),
                         sqrtM2_mean_MHz=float(M2r.mean()), sigma_d_median_MHz=float(np.median(s_)),
                         tau_d_median_ns=float(np.median(t_)), xi_median=float(np.median(x_)),
                         frac_xi_lt1=float((x_ < 1).mean()), n_pairs=int(close.sum())))
        print(rows[-1])
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "module2_dipolar_by_replica.csv"), index=False)
S = R.drop(columns="rep").groupby(["T", "field"]).agg(["mean", "std"]); S.to_csv(os.path.join(D, "module2_dipolar_summary.csv"))
np.savez_compressed(os.path.join(D, "module2_dipolar_acf.npz"), lag_ns=LAG * 0.01, **acfs)
pd.set_option("display.width", 250); print(S.round(3).to_string())
