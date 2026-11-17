#!/usr/bin/env python3
"""
m1_03_relaxation_models.py
Tests whether the computed low-frequency vibrations of PTEMPO can account for
the temperature dependence of the experimental spin--lattice relaxation rate.

Experimental data: Table 1 of Tesi et al., Adv. Mater. 2023, 35, 2208998
(pulsed Q band, 35.0 GHz).  Numbers in parentheses are uncertainties in the
last digit(s); they are converted exactly as printed.

Models for the rate R = 1/T1 (all with a direct one-phonon term a*T, valid
because h*nu/kB = 1.7 K << 7 K at 35 GHz):
  M_loc   : a*T + C * n(D)[n(D)+1]                 single local mode, D free
            (the model used by Tesi et al.)
  M_ram   : a*T + b*T^m                            Raman power law, m free
  M_A     : a*T + C * sum_k w_k n_k(n_k+1),  w_k ~ (dA/dQ_k)^2
  M_g     : same, w_k ~ (d nu_g/dQ_k at Q band)^2
  M_Ag    : same, w_k ~ (dA/dQ_k)^2 + (d nu_g/dQ_k)^2
In M_A, M_g, M_Ag the mode energies and relative weights are fixed by the DFT
calculations; only the overall scale C (and a) are fitted.

Models are compared by weighted chi^2 and the small-sample Akaike criterion.
"""
import os
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "module1")
kB_cm = 0.6950348          # cm^-1 per K

# ---------------- experimental data (Tesi 2023, Table 1) ----------------------
T = np.array([7, 10, 20, 30, 40, 50, 60], float)
exp = {
  "PTEMPO_slow": (np.array([87, 57, 25.7, 13.0, 4.2, 2.00, 1.00]),
                  np.array([2, 1, 0.6, 0.4, 0.1, 0.07, 0.04])),
  "PTEMPO_fast": (np.array([21, 14.4, 6.2, 3.5, 1.2, 0.69, 0.37]),
                  np.array([1, 0.6, 0.4, 0.3, 0.1, 0.06, 0.04])),
  "SAM":         (np.array([np.nan, 9.2, 3.6, 1.5, 0.5, 0.3, np.nan]),
                  np.array([np.nan, 0.6, 0.5, 0.2, 0.1, 0.1, np.nan])),
}

# ---------------- computed modes (PTEMPO) -------------------------------------
summ = pd.read_csv(os.path.join(DATA, "module1_coupling_summary.csv"))
pt = summ[summ.molecule == "PTEMPO"].sort_values("mode")
freqs = pd.read_csv(os.path.join(DATA, "module1_frequencies.csv"))
nu = freqs[freqs.molecule == "PTEMPO"].sort_values("mode").freq_cm1.to_numpy()
wA = pt.dA_dQ_MHz.to_numpy() ** 2
wg = pt.dnu_g_Q_MHz.to_numpy() ** 2
weights = {"M_A": wA / wA.sum(), "M_g": wg / wg.sum(), "M_Ag": (wA + wg) / (wA + wg).sum()}

def nn1(E_cm, T):
    x = E_cm / (kB_cm * T)
    return np.exp(x) / np.expm1(x) ** 2

def model(name, p, T):
    if name == "M_loc":
        a, C, D = p;  return a * T + C * nn1(D, T)
    if name == "M_ram":
        a, b, m = p;  return a * T + b * T ** m
    a, C = p
    w = weights[name]
    return a * T + C * sum(wk * nn1(Ek, T) for wk, Ek in zip(w, nu))

p0 = {"M_loc": [1, 1e3, 50], "M_ram": [1, 1e-3, 3], "M_A": [1, 1e3], "M_g": [1, 1e3], "M_Ag": [1, 1e3]}
lb = {"M_loc": [0, 0, 5], "M_ram": [0, 0, 1], "M_A": [0, 0], "M_g": [0, 0], "M_Ag": [0, 0]}
ub = {"M_loc": [np.inf, np.inf, 400], "M_ram": [np.inf, np.inf, 12], "M_A": [np.inf, np.inf], "M_g": [np.inf, np.inf], "M_Ag": [np.inf, np.inf]}

rows, curves = [], {}
Tf = np.linspace(5, 65, 400)
for ds, (t1_ms, e_ms) in exp.items():
    ok = ~np.isnan(t1_ms)
    Tx = T[ok]; R = 1e3 / t1_ms[ok]; sR = R * e_ms[ok] / t1_ms[ok]   # s^-1
    for name in p0:
        f = lambda p: (model(name, p, Tx) - R) / sR
        best = None
        for scale in [1e-2, 1, 1e2, 1e4]:                 # multi-start
            q0 = np.array(p0[name], float); q0[1] *= scale
            r = least_squares(f, q0, bounds=(lb[name], ub[name]), x_scale="jac", max_nfev=20000)
            if best is None or r.cost < best.cost: best = r
        k = len(best.x); n = len(Tx)
        chi2 = 2 * best.cost
        aicc = chi2 + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.inf)
        J = best.jac
        try:
            cov = np.linalg.inv(J.T @ J) * max(chi2 / max(n - k, 1), 1.0)
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(k, np.nan)
        rows.append(dict(dataset=ds, model=name, n=n, k=k, chi2=chi2, chi2_red=chi2 / max(n - k, 1),
                         AICc=aicc, params=np.array2string(best.x, precision=4),
                         param_se=np.array2string(se, precision=3)))
        curves[(ds, name)] = model(name, best.x, Tf)
res = pd.DataFrame(rows)
res["dAICc"] = res.groupby("dataset").AICc.transform(lambda s: s - s.min())
res.to_csv(os.path.join(DATA, "module1_T1_model_comparison.csv"), index=False)
np.savez(os.path.join(DATA, "module1_T1_model_curves.npz"), T=Tf,
         **{f"{d}__{m}": v for (d, m), v in curves.items()})
pd.set_option("display.width", 250)
print(res.to_string())
