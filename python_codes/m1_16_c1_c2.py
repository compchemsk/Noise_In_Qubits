#!/usr/bin/env python3
"""
m1_16_c1_c2.py -- which term of the scan fit dominates, c1 (linear) or c2 (quadratic)?

Each scan is fitted with X(Q) = c0 + c1 Q + c2 Q^2 (m1_02). Here we ask, for every mode and for
X = A_iso and X = g_iso, which term matters more.

1. Size of each term over the scanned range. At |Q| = 1 the linear term changes X by |c1| and the
   quadratic term by |c2|. The larger one is the "dominant term".
2. Size of each term at temperature T. With <Q^2> = n + 1/2 and a Gaussian distribution of Q
   (harmonic oscillator), <Q^4> = 3<Q^2>^2, so
       rms of the linear part    = |c1| sqrt(<Q^2>)
       rms of the quadratic part = |c2| sqrt(<Q^4> - <Q^2>^2) = sqrt(2) |c2| <Q^2>
   The two parts are uncorrelated (odd and even powers of Q), so their variances add:
       rms total = sqrt(c1^2 <Q^2> + 2 c2^2 <Q^2>^2).
   At T -> 0, <Q^2> = 1/2 and the ratio quad/lin = |c2|/|c1|, the same test as in point 1.
3. Quality of a straight-line fit. R^2 of X = a + b Q is compared with R^2 of the quadratic.

g is also expressed as a line shift at 35 GHz: d(nu_g) = (nu_mw / g_iso) * dg.
Outputs: data/module1/module1_c1_c2.csv, module1_thermal_c1_c2.csv,
         tables/tabS_c1c2.tex, tables/tabS_thermal.tex
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data", "module1"); TAB = os.path.join(ROOT, "tables")
KB = 0.6950348; NU_Q = 35000.0          # MHz
sc = pd.read_csv(os.path.join(DATA, "module1_scans.csv"))
fa = pd.read_csv(os.path.join(DATA, "module1_fits_all.csv"))
ex = pd.read_csv(os.path.join(DATA, "module1_excluded_points.csv"))
fr = pd.read_csv(os.path.join(DATA, "module1_frequencies.csv"))[["molecule", "mode", "freq_cm1"]]
g0 = pd.read_csv(os.path.join(DATA, "module1_coupling_summary.csv"))[["molecule", "mode", "g0"]]

rows = []
for r in fa.itertuples():
    d = sc[(sc.molecule == r.molecule) & (sc["mode"] == r.mode) & (sc.quantity == r.quantity)].copy()
    e = ex[(ex.molecule == r.molecule) & (ex["mode"] == r.mode) & (ex.quantity == r.quantity)]
    if len(e):
        d = d[~np.isclose(d.Q.values[:, None], e.Q.values[None]).any(1)]
    Q, X = d.Q.to_numpy(), d.value.to_numpy()
    p1 = np.polyfit(Q, X, 1); res1 = X - np.polyval(p1, Q)
    r2lin = 1 - (res1 ** 2).sum() / ((X - X.mean()) ** 2).sum()
    rows.append(dict(molecule=r.molecule, mode=r.mode, quantity=r.quantity, c1=r.c1, c1_se=r.c1_se,
                     c2=r.c2, c2_se=r.c2_se, R2_lin=r2lin, R2_quad=r.R2_quad))
C = pd.DataFrame(rows).merge(fr, on=["molecule", "mode"]).merge(g0, on=["molecule", "mode"])
isg = C.quantity == "g_iso"
C["scale"] = np.where(isg, NU_Q / C.g0, 1.0)                  # g -> MHz line shift at 35 GHz
C["c1_MHz"] = C.c1 * C.scale; C["c2_MHz"] = C.c2 * C.scale
C["ratio_c2_c1"] = np.abs(C.c2) / np.abs(C.c1)
C["dominant"] = np.where(np.abs(C.c2) > np.abs(C.c1), "c2", "c1")
C.to_csv(os.path.join(DATA, "module1_c1_c2.csv"), index=False)

# thermal rms with both terms
T_rows = []
for T in (10, 60, 300):
    for r in C.itertuples():
        q2 = 1 / np.expm1(r.freq_cm1 / (KB * T)) + 0.5
        lin = abs(r.c1_MHz) * np.sqrt(q2); quad = np.sqrt(2) * abs(r.c2_MHz) * q2
        T_rows.append(dict(T=T, molecule=r.molecule, mode=r.mode, quantity=r.quantity, freq=r.freq_cm1,
                           Q2=q2, rms_lin=lin, rms_quad=quad, rms_tot=np.hypot(lin, quad)))
TH = pd.DataFrame(T_rows); TH.to_csv(os.path.join(DATA, "module1_thermal_c1_c2.csv"), index=False)

# ---------------- tables
MOLS = ["TEMPO", "PTEMPO", "PTEMPO-alkyl"]
L = [r"\begin{tabular}{llrrrcrrrc}", r"\toprule",
     r" & & & \multicolumn{3}{c}{$A_\mathrm{iso}$ (MHz)} & \multicolumn{3}{c}{$g$: line shift at 35 GHz (MHz)} & \\",
     r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
     r"Molecule & Mode & $\tilde\nu$ (cm$^{-1}$) & $|c_1|$ & $|c_2|$ & dominant & $|c_1|$ & $|c_2|$ & dominant & $R^2_\mathrm{lin}$ ($A$, $g$)\\", r"\midrule"]
for mol in MOLS:
    a = C[(C.molecule == mol) & (C.quantity == "A_iso_MHz")].sort_values("mode")
    g = C[(C.molecule == mol) & (C.quantity == "g_iso")].set_index("mode")
    for i, r in enumerate(a.itertuples()):
        q = g.loc[r.mode]
        L.append(f"{mol if i == 0 else ''} & {r.mode} & {r.freq_cm1:.1f} & {abs(r.c1):.3f} & {abs(r.c2):.3f} & ${r.dominant[0]}_{r.dominant[1]}$ & "
                 f"{abs(q.c1_MHz):.3f} & {abs(q.c2_MHz):.3f} & ${q.dominant[0]}_{q.dominant[1]}$ & {r.R2_lin:.3f}, {q.R2_lin:.3f}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(TAB, "tabS_c1c2.tex"), "w").write("\n".join(L) + "\n")

S = TH.groupby(["T", "molecule", "quantity"]).apply(
    lambda d: pd.Series(dict(lin=np.sqrt((d.rms_lin ** 2).sum()), quad=np.sqrt((d.rms_quad ** 2).sum()),
                             tot=np.sqrt((d.rms_tot ** 2).sum()), n=len(d))), include_groups=False).reset_index()
S.to_csv(os.path.join(DATA, "module1_thermal_c1_c2_summary.csv"), index=False)
L = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
     r" & & \multicolumn{3}{c}{$\delta A$ (MHz)} & \multicolumn{3}{c}{$\delta\nu_g$ at 35 GHz (MHz)}\\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
     r"$T$ (K) & Molecule & from $c_1$ & from $c_2$ & total & from $c_1$ & from $c_2$ & total\\", r"\midrule"]
for T in (10, 60, 300):
    for j, mol in enumerate(MOLS):
        a = S[(S["T"] == T) & (S.molecule == mol) & (S.quantity == "A_iso_MHz")].iloc[0]
        g = S[(S["T"] == T) & (S.molecule == mol) & (S.quantity == "g_iso")].iloc[0]
        L.append(f"{T if j == 0 else ''} & {mol} & {a.lin:.2f} & {a.quad:.2f} & {a.tot:.2f} & {g.lin:.2f} & {g.quad:.2f} & {g.tot:.2f}\\\\")
    L.append(r"\midrule")
L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
open(os.path.join(TAB, "tabS_thermal.tex"), "w").write("\n".join(L) + "\n")

pd.set_option("display.width", 200)
print(C[["molecule", "mode", "quantity", "freq_cm1", "c1_MHz", "c2_MHz", "dominant", "R2_lin", "R2_quad"]].round(4).to_string(index=False))
print(S.round(3).to_string(index=False))
