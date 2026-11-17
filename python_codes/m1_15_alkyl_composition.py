#!/usr/bin/env python3
"""
m1_15_alkyl_composition.py
Fragment participation of the treated PTEMPO-alkyl modes, from the Gaussian 16 frequency job
(raw/module1/PTEMPO-alkyl-11.log; UB3LYP/def2-TZVP).

The normal modes are rebuilt at full precision from the Cartesian Hessian stored in the archive
block of the log (the printed modes carry only two decimals): mass-weighting, projection of
translations and rotations, diagonalization. Frequencies are checked against the printed ones.
For each mode, P_F = sum_{j in F} |u_kj|^2 / sum_j |u_kj|^2 with u_k the normalized Cartesian
displacement (the definition used for TEMPO and PTEMPO). Fragments (assigned from connectivity):
  NO     nitroxide N and O            Ring  the five ring carbons
  MetH   the twelve methyl H          MetC  the four methyl C
  Link   ether O and CH2 carbon       Trz   the five triazole ring atoms (2 C, 3 N)
  Chain  the eleven CH2 carbons and the terminal S of the -(CH2)11-SH chain
  Other  all remaining atoms (non-methyl hydrogens)
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
LOG = os.path.join(ROOT, "raw", "module1", "PTEMPO-alkyl-11.log")
OUT = os.path.join(ROOT, "data", "module1", "module1_mode_composition_alkyl.csv")
TREATED = [1, 6, 7, 8, 9, 10, 11, 12, 13, 24]
MASS = {"H": 1.00782504, "C": 12.0, "N": 14.0030740, "O": 15.9949146, "S": 31.9720707}
B = 0.529177210903; EH = 4.3597447222071e-18; AMU = 1.66053906660e-27; C = 2.99792458e10

L = open(LOG).read().split("\n")
n = [i for i, l in enumerate(L) if l.startswith(" 1\\1\\") and "\\Freq\\" in l][-1]
parts = "".join(l[1:] for l in L[n:]).split("\\\\")
geo = parts[3].split("\\")[1:]
el = [g.split(",")[0] for g in geo]; X = np.array([[float(v) for v in g.split(",")[1:]] for g in geo])
N = len(el); n3 = 3 * N
h = np.array([float(v) for v in parts[5].split(",")]); assert h.size == n3 * (n3 + 1) // 2
H = np.zeros((n3, n3)); H[np.tril_indices(n3)] = h; H = H + H.T - np.diag(np.diag(H))
m = np.array([MASS[e] for e in el]); m3 = np.repeat(m, 3); sm = np.sqrt(m)
R = X / B - (m[:, None] * X / B).sum(0) / m.sum()
D = []
for k in range(3):
    v = np.zeros((N, 3)); v[:, k] = 1; D.append((v * sm[:, None]).ravel())
    e = np.zeros(3); e[k] = 1; D.append((np.cross(e, R) * sm[:, None]).ravel())
Q, _ = np.linalg.qr(np.array(D).T); P = np.eye(n3) - Q @ Q.T
w, V = np.linalg.eigh(P @ (H / np.sqrt(np.outer(m3, m3))) @ P)
fr = np.sign(w) * np.sqrt(np.abs(w) * EH / ((B * 1e-10) ** 2 * AMU)) / (2 * np.pi * C)
o = np.argsort(fr); fr, V = fr[o][6:], V[:, o][:, 6:]
printed = np.array([float(x) for l in L if "Frequencies --" in l for x in l.split()[2:]])
printed = printed[-len(fr):] if printed.size >= len(fr) else printed
assert np.abs(fr[:30] - printed[:30]).max() < 0.05, "frequencies do not match the log"
U = (V / np.repeat(sm, 3)[:, None]); U /= np.linalg.norm(U, axis=0); U = U.T.reshape(-1, N, 3)

# fragments from connectivity
rc = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05}
d = np.linalg.norm(X[:, None] - X[None], axis=2)
nb = [[j for j in range(N) if j != i and d[i, j] < 1.2 * (rc[el[i]] + rc[el[j]])] for i in range(N)]
iO = [i for i in range(N) if el[i] == "O" and len(nb[i]) == 1 and el[nb[i][0]] == "N"][0]; iN = nb[iO][0]
Ca = [j for j in nb[iN] if el[j] == "C"]
nH = lambda j: sum(el[x] == "H" for x in nb[j])
MeC = [j for a in Ca for j in nb[a] if el[j] == "C" and nH(j) == 3]
C3 = [j for a in Ca for j in nb[a] if el[j] == "C" and j not in MeC]
C4 = [j for a in C3 for j in nb[a] if el[j] == "C" and j not in Ca][0]
MetH = [x for c in MeC for x in nb[c] if el[x] == "H"]
Oe = [j for j in nb[C4] if el[j] == "O"][0]; CH2 = [j for j in nb[Oe] if j != C4][0]
t4 = [j for j in nb[CH2] if el[j] == "C"][0]; t5 = [j for j in nb[t4] if el[j] == "C" and j != CH2][0]
Ns = [i for i in range(N) if el[i] == "N" and i != iN]; Trz = [t4, t5] + Ns
N1 = [x for x in Ns if any(el[j] == "C" and j not in Trz for j in nb[x])][0]
chain, prev, cur = [], N1, [j for j in nb[N1] if el[j] == "C" and j not in Trz][0]
while el[cur] != "S":
    chain.append(cur); prev, cur = cur, [j for j in nb[cur] if el[j] in "CS" and j != prev and j not in chain][0]
chain.append(cur)
F = dict(NO=[iN, iO], Ring=Ca + C3 + [C4], MetH=MetH, MetC=MeC, Link=[Oe, CH2], Trz=Trz, Chain=chain)
assert [len(v) for v in F.values()] == [2, 5, 12, 4, 2, 5, 12], [len(v) for v in F.values()]
rows = []
for k in TREATED:
    u2 = (U[k - 1] ** 2).sum(1); r = dict(molecule="PTEMPO-alkyl", mode=k, freq=fr[k - 1])
    for g, idx in F.items(): r[g] = u2[idx].sum()
    r["Other"] = 1 - sum(r[g] for g in F); rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT, index=False)
print(T.round(4).to_string(index=False))
