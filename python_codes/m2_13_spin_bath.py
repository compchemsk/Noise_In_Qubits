#!/usr/bin/env python3
"""
m2_13_spin_bath.py
Geometry of the proton (1H) bath around each nitroxide in the monolayer.

For every intact radical (spin = N-O midpoint) and every 10th frame of the
last 5 ns, protons are counted in the shells 0-4, 4-8 and 8-12 A
(Canarie et al., JPCL 2020: protons 4-8 A dominate nuclear-spin-driven
decoherence of nitroxides; nuclei up to ~12 A matter), separated into
  own head   : H1-H17 of the same molecule (ring CH2, CH-O and methyl H)
  own rest   : H18-H42 of the same molecule (linker, triazole, chain)
  neighbours : all protons of other molecules
Also computed:
  sum_r6     : sum of r^-6 over protons with r >= 4 A (A^-6), the radial factor
               of the hyperfine second moment
  CH2 groups : methylene carbons (two H) of other molecules within 4-8 A
               (geminal proton pairs; Jahn et al., JPCL 2022)
  CH3 groups : methyl carbons of other molecules within 2.5-7 A
               (Jahn et al., JCP 2024)
  below/above: fraction of 4-12 A protons on the gold side of the spin
Reference: a homogeneous bath with the proton density of 1:1 (v/v)
dichloromethane/toluene at 298 K (the solvent of the solution experiments of
Tesi et al.), rho_H = 53.3 M = 0.0321 A^-3, filling all space beyond 4 A:
   N(4-8) = 60.2, N(8-12) = 163.4, sum_r6 = 4 pi rho / (3 * 4^3) = 2.10e-3 A^-6.
This reference ignores the solute's own excluded volume and the higher density
of the frozen glass; it is an order-of-magnitude benchmark.
"""
import os, numpy as np, pandas as pd
from scipy.spatial import cKDTree
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res, lin = P["name"], P["resid"], P["resname"] == "LIN"
mols = np.unique(res[lin]); intact = ~np.isin(mols, DEFECT)
H = np.where(lin & np.char.startswith(names.astype(str), "H"))[0]
Hmol = res[H]; Hnum = np.array([int(n[1:]) for n in names[H]])
Hhead = Hnum <= 17
iO = np.where(lin & (names == "O1"))[0]; iN = np.where(lin & (names == "N1"))[0]
# methylene carbons (bonded to exactly two H) and methyl carbons
b = P["bonds"]; isH = np.isin(np.arange(P["natom"]), H)
nH = np.zeros(P["natom"], int)
for a1, a2 in b:
    if isH[a1] and not isH[a2]: nH[a2] += 1
    if isH[a2] and not isH[a1]: nH[a1] += 1
CH2 = np.where(lin & (nH == 2))[0]; CH3 = np.where(lin & (nH == 3))[0]
rho = 53.3 * 6.02214076e23 / 1e27
REF = dict(N4_8=4/3*np.pi*(8**3-4**3)*rho, N8_12=4/3*np.pi*(12**3-8**3)*rho, sum_r6=4*np.pi*rho/(3*4**3))
print("homogeneous 1:1 DCM/toluene reference:", {k: round(v, 4) for k, v in REF.items()})
rows = []
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])[100:][500:1000:10]
    xtop = None
    for f in range(X.shape[0]):
        spin = 0.5 * (X[f, iO] + X[f, iN])
        tH = cKDTree(X[f, H]); tC2 = cKDTree(X[f, CH2]); tC3 = cKDTree(X[f, CH3])
        for m in np.where(intact)[0]:
            idx = np.array(tH.query_ball_point(spin[m], 12.0), int)
            d = np.linalg.norm(X[f, H[idx]] - spin[m], axis=1)
            own = Hmol[idx] == mols[m]; head = own & Hhead[idx]; rest = own & ~Hhead[idx]; nb = ~own
            sh = lambda mask, lo, hi: int(((d >= lo) & (d < hi) & mask).sum())
            far = d >= 4
            c2 = np.array(tC2.query_ball_point(spin[m], 8.0), int)
            d2 = np.linalg.norm(X[f, CH2[c2]] - spin[m], axis=1)
            c2o = res[CH2[c2]] != mols[m]
            c3 = np.array(tC3.query_ball_point(spin[m], 7.0), int)
            d3 = np.linalg.norm(X[f, CH3[c3]] - spin[m], axis=1)
            c3o = res[CH3[c3]] != mols[m]
            sel = (d >= 4) & (d < 12)
            below = (X[f, H[idx[sel]], 0] > spin[m, 0]).mean() if sel.any() else np.nan   # larger x = toward Au
            rows.append(dict(T=T, rep=r, frame=f, mol=int(mols[m]),
                head_0_4=sh(head, 0, 4), head_4_8=sh(head, 4, 8), rest_0_4=sh(rest, 0, 4), rest_4_8=sh(rest, 4, 8),
                rest_8_12=sh(rest, 8, 12), nb_0_4=sh(nb, 0, 4), nb_4_8=sh(nb, 4, 8), nb_8_12=sh(nb, 8, 12),
                sum_r6_all=float((d[far] ** -6.0).sum()), sum_r6_nb=float((d[far & nb] ** -6.0).sum()),
                CH2_nb_4_8=int(((d2 >= 4) & c2o).sum()), CH3_nb_2p5_7=int(((d3 >= 2.5) & c3o).sum()),
                frac_below=below))
df = pd.DataFrame(rows)
df["tot_4_8"] = df.head_4_8 + df.rest_4_8 + df.nb_4_8
df["tot_8_12"] = df.rest_8_12 + df.nb_8_12
df.to_csv(os.path.join(D, "module2_spin_bath_samples.csv.gz"), index=False)
cols = ["head_0_4", "head_4_8", "rest_0_4", "rest_4_8", "nb_0_4", "nb_4_8", "tot_4_8", "rest_8_12", "nb_8_12", "tot_8_12",
        "sum_r6_all", "sum_r6_nb", "CH2_nb_4_8", "CH3_nb_2p5_7", "frac_below"]
rep = df.groupby(["T", "rep"])[cols].mean()
S = rep.groupby(level="T").agg(["mean", "std"]); S.to_csv(os.path.join(D, "module2_spin_bath_summary.csv"))
pd.set_option("display.width", 250); print(S.T.round(4).to_string())
pd.Series(REF).to_csv(os.path.join(D, "module2_spin_bath_reference.csv"))
