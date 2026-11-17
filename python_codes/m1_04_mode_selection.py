#!/usr/bin/env python3
"""
m1_04_mode_selection.py
Inverse test: which computed PTEMPO vibrations are compatible with the measured
T1(T)?  R(T) = a*T + sum_k C_k n_k(n_k+1), with mode energies fixed at the DFT
values and C_k >= 0 free (non-negative least squares, weighted by the
experimental uncertainties).  Also scans every single computed mode and every
pair of modes, and reports chi^2 for each.
"""
import os, itertools
import numpy as np, pandas as pd
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data", "module1")
kB = 0.6950348
T = np.array([7, 10, 20, 30, 40, 50, 60.])
sets = {"PTEMPO_slow": (np.array([87, 57, 25.7, 13.0, 4.2, 2.00, 1.00]), np.array([2, 1, .6, .4, .1, .07, .04])),
        "PTEMPO_fast": (np.array([21, 14.4, 6.2, 3.5, 1.2, .69, .37]), np.array([1, .6, .4, .3, .1, .06, .04])),
        "SAM": (np.array([np.nan, 9.2, 3.6, 1.5, .5, .3, np.nan]), np.array([np.nan, .6, .5, .2, .1, .1, np.nan]))}
fr = pd.read_csv(os.path.join(DATA, "module1_frequencies.csv"))
nu = fr[fr.molecule == "PTEMPO"].sort_values("mode").freq_cm1.to_numpy()
def nn1(D, T): x = D / (kB * T); return np.exp(x) / np.expm1(x) ** 2
rows = []
for ds, (t, e) in sets.items():
    ok = ~np.isnan(t); Tx = T[ok]; R = 1e3 / t[ok]; s = R * e[ok] / t[ok]
    def fit(idx):
        X = np.column_stack([Tx] + [nn1(nu[i], Tx) for i in idx]) / s[:, None]
        c, r = nnls(X, R / s); return c, r ** 2
    c, chi2 = fit(range(len(nu)))
    rows.append(dict(dataset=ds, combo="all 8 (NNLS)", chi2=chi2,
                     active=";".join(f"{nu[i]:.1f}" for i in range(len(nu)) if c[i + 1] > 0)))
    for i in range(len(nu)):
        c, chi2 = fit([i]); rows.append(dict(dataset=ds, combo=f"{nu[i]:.1f}", chi2=chi2, active=f"{nu[i]:.1f}"))
    for i, j in itertools.combinations(range(len(nu)), 2):
        c, chi2 = fit([i, j])
        rows.append(dict(dataset=ds, combo=f"{nu[i]:.1f}+{nu[j]:.1f}", chi2=chi2,
                         active=";".join(f"{nu[k]:.1f}" for k, ck in zip([i, j], c[1:]) if ck > 0)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(DATA, "module1_mode_selection.csv"), index=False)
for ds in sets:
    d = df[df.dataset == ds].sort_values("chi2")
    print(f"\n{ds}: best 6 of single/pair/all"); print(d.head(6).to_string(index=False))
    print("single modes:"); print(d[~d.combo.str.contains(r"\+|all")].sort_values("combo", key=lambda s: s.astype(float)).to_string(index=False))

# ---------------------------------------------------------------------------
# Monolayer data tested against the modes computed for the monolayer head
# (PTEMPO-alkyl).  NOTE: modes 2-5 and 14-23 of PTEMPO-alkyl were not computed,
# so the window between 85 and 196 cm-1 is not represented.
# ---------------------------------------------------------------------------
nu_alk = fr[fr.molecule == "PTEMPO-alkyl"].sort_values("mode").freq_cm1.to_numpy()
t, e = sets["SAM"]; ok = ~np.isnan(t); Tx = T[ok]; R = 1e3 / t[ok]; s = R * e[ok] / t[ok]
alk = []
for E in nu_alk:
    X = np.column_stack([Tx, nn1(E, Tx)]) / s[:, None]; c, r = nnls(X, R / s)
    alk.append(dict(dataset="SAM", head_mode_cm1=E, chi2=r**2, chi2_red=r**2 / 3))
alk = pd.DataFrame(alk); alk.to_csv(os.path.join(DATA, "module1_SAM_vs_alkyl_modes.csv"), index=False)
print("\nSAM data vs single PTEMPO-alkyl modes:"); print(alk.to_string(index=False))
