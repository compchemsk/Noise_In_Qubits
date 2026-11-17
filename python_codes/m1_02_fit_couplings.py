#!/usr/bin/env python3
"""
m1_02_fit_couplings.py
Mode-resolved spin--phonon coefficients from the g_iso(Q) and A_iso(Q) scans.

For each scan X(Q) (X = g_iso or A_iso) we fit, by ordinary least squares,
    X(Q) = c0 + c1 Q + c2 Q^2            (quadratic, free intercept)
and report standard errors from the residual variance.  Diagnostics:
    * cubic fit X = c0 + c1 Q + c2 Q^2 + c3 Q^3 : is the quadratic adequate?
    * central differences at |Q| = 0.2 and 1.0 : small- vs large-amplitude slope
    * fraction of the total variation captured by the linear term

To compare g- and A-modulation on the SAME energy scale, the g derivative is
converted into a frequency shift of the EPR line,
    d nu / dQ = (mu_B B / h) dg/dQ = (nu_mw / g_iso) dg/dQ ,
evaluated at X band (9.5 GHz) and at the Q-band frequency used for the pulsed
experiments of Tesi et al. (35.0 GHz).  A_iso is already in MHz.

Output: data/module1/module1_fits.csv
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "module1")
scans = pd.read_csv(os.path.join(DATA, "module1_scans.csv"))

NU_X, NU_Q = 9.5e3, 35.0e3        # MHz

def ols(Q, y, deg):
    V = np.vander(Q, deg + 1, increasing=True)          # 1, Q, Q^2 ...
    coef, *_ = np.linalg.lstsq(V, y, rcond=None)
    res = y - V @ coef
    dof = len(y) - (deg + 1)
    s2 = (res @ res) / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(V.T @ V)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res @ res) / ss_tot if ss_tot > 0 else np.nan
    return coef, np.sqrt(np.diag(cov)), r2, np.sqrt(s2)

def central_diff(Q, y, h):
    p, m = np.isclose(Q, h), np.isclose(Q, -h)
    if p.any() and m.any():
        return (y[p][0] - y[m][0]) / (2 * h)
    return np.nan

def flag_outliers(Q, y, resolution, thresh=10.0):
    """Leave-one-out test: a point is flagged if its deviation from the quadratic
    fitted to the remaining points exceeds `thresh` x that fit's RMSE
    (RMSE floored at 3x the printed precision of the data and at 1e-3 of the span)."""
    bad = np.zeros(len(Q), bool)
    if len(Q) < 7:
        return bad
    span = y.max() - y.min()
    for k in range(len(Q)):
        m = np.ones(len(Q), bool); m[k] = False
        c, _, _, rmse = ols(Q[m], y[m], 2)
        pred = c[0] + c[1]*Q[k] + c[2]*Q[k]**2
        if abs(y[k] - pred) > thresh * max(rmse, 1e-3 * span, 3 * resolution):
            bad[k] = True
    return bad

# A geometry is excluded from BOTH scans (g and A come from the same calculation)
bad_geoms = set()
for (mol, mode, qty), d in scans.groupby(["molecule", "mode", "quantity"]):
    d = d.sort_values("Q")
    Qa, ya = d.Q.to_numpy(float), d.value.to_numpy(float)
    res = 1e-7 if qty == "g_iso" else 1e-4      # printed precision of the inputs
    for q_ in Qa[flag_outliers(Qa, ya, res)]:
        bad_geoms.add((mol, mode, round(q_, 3)))

rows, excluded = [], []
for (mol, mode, qty), d in scans.groupby(["molecule", "mode", "quantity"]):
    d = d.sort_values("Q")
    Q, y = d.Q.to_numpy(float), d.value.to_numpy(float)
    bad = np.array([(mol, mode, round(q_, 3)) in bad_geoms for q_ in Q])
    for q_, v_ in zip(Q[bad], y[bad]):
        excluded.append(dict(molecule=mol, mode=mode, quantity=qty, Q=q_, value=v_))
    c_all, _, r2_all, _ = ols(Q, y, 2)
    Q, y = Q[~bad], y[~bad]
    c, se, r2, rmse = ols(Q, y, 2)
    c3, se3, r2c, _ = ols(Q, y, 3)
    lin_share = (c[1] ** 2) / (c[1] ** 2 + c[2] ** 2) if (c[1] or c[2]) else np.nan
    row = dict(molecule=mol, mode=mode, quantity=qty, n_points=len(Q),
               Qmin=Q.min(), Qmax=Q.max(),
               c0=c[0], c1=c[1], c1_se=se[1], c2=c[2], c2_se=se[2],
               R2_quad=r2, rmse_quad=rmse,
               c3=c3[3], c3_se=se3[3], R2_cubic=r2c,
               cd_0p2=central_diff(Q, y, 0.2), cd_1p0=central_diff(Q, y, 1.0),
               linear_share=lin_share,
               span=y.max() - y.min(), n_excluded=int(bad.sum()),
               c1_with_outliers=c_all[1], R2_with_outliers=r2_all)
    rows.append(row)
fits = pd.DataFrame(rows)

# ---- unit-consistent energy scale (MHz per unit Q) ---------------------------
g_tab = fits[fits.quantity == "g_iso"].set_index(["molecule", "mode"])
A_tab = fits[fits.quantity == "A_iso_MHz"].set_index(["molecule", "mode"])
summ = pd.DataFrame(index=g_tab.index)
summ["g0"] = g_tab.c0
summ["dg_dQ"] = g_tab.c1;          summ["dg_dQ_se"] = g_tab.c1_se
summ["d2g_dQ2_half"] = g_tab.c2;   summ["d2g_se"] = g_tab.c2_se
summ["A0_MHz"] = A_tab.c0
summ["dA_dQ_MHz"] = A_tab.c1;      summ["dA_dQ_se"] = A_tab.c1_se
summ["d2A_dQ2_half_MHz"] = A_tab.c2; summ["d2A_se"] = A_tab.c2_se
summ["dnu_g_X_MHz"] = NU_X / summ.g0 * summ.dg_dQ
summ["dnu_g_Q_MHz"] = NU_Q / summ.g0 * summ.dg_dQ
summ["ratio_A_over_gQ"] = np.abs(summ.dA_dQ_MHz) / np.abs(summ.dnu_g_Q_MHz)
summ["ratio_A_over_gX"] = np.abs(summ.dA_dQ_MHz) / np.abs(summ.dnu_g_X_MHz)
summ["R2_g"] = g_tab.R2_quad; summ["R2_A"] = A_tab.R2_quad
summ["linshare_g"] = g_tab.linear_share; summ["linshare_A"] = A_tab.linear_share
summ = summ.reset_index()

fits.to_csv(os.path.join(DATA, "module1_fits_all.csv"), index=False)
pd.DataFrame(excluded).to_csv(os.path.join(DATA, "module1_excluded_points.csv"), index=False)
print("Excluded points:\n", pd.DataFrame(excluded))
summ.to_csv(os.path.join(DATA, "module1_coupling_summary.csv"), index=False)

pd.set_option("display.width", 250, "display.max_columns", 40)
print(fits[["molecule","mode","quantity","n_points","c0","c1","c1_se","c2","c2_se",
            "R2_quad","c3","c3_se","cd_0p2","cd_1p0"]].to_string(float_format=lambda v: f"{v:.4g}"))
print()
print(summ[["molecule","mode","dg_dQ","dnu_g_X_MHz","dnu_g_Q_MHz","dA_dQ_MHz","d2A_dQ2_half_MHz",
            "ratio_A_over_gQ","R2_g","R2_A"]].to_string(float_format=lambda v: f"{v:.4g}"))
