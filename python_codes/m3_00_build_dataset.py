#!/usr/bin/env python3
"""
m3_00_build_dataset.py
Builds the Module III training set from the original (un-reordered) pair
geometries in generated_structures/ and the ORCA results in J-coupling.dat.

* Each pair file contains two rigid copies of the same TEMPO monomer
  (29 atoms; atoms 1-29 and 30-58).
* J = ORCA "J(3)" (Yamaguchi formula), cm^-1, for H = -2 J S1.S2
  (J > 0 ferromagnetic, J < 0 antiferromagnetic).
* Key atoms identified from connectivity: N, O (nitroxide), the two alpha
  carbons bonded to N, the four methyl carbons, C4 (para carbon).
Writes data/module3/pairs_geometry.npz and a CSV with J and metadata.
"""
import os, re, glob, numpy as np, pandas as pd
# Pair library: raw/PartIII/filtered_structures/J-coupling.dat and raw/PartIII/generated_structures/*.xyz
# (run_all.sh builds this folder from simulation_inputs/ModuleIII). Override with TEMPO_PAIRS_RAW.
RAW = os.environ.get("TEMPO_PAIRS_RAW", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "raw", "PartIII"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "module3")
os.makedirs(OUT, exist_ok=True)

J = {}
for line in open(os.path.join(RAW, "filtered_structures", "J-coupling.dat")):
    sp = line.split()
    if len(sp) == 2:
        try: J[sp[0]] = float(sp[1])
        except ValueError: pass
print("J entries parsed:", len(J))

def read_xyz(p):
    L = open(p).read().split("\n"); n = int(L[0])
    el = [l.split()[0] for l in L[2:2 + n]]
    X = np.array([[float(v) for v in l.split()[1:4]] for l in L[2:2 + n]])
    return el, X, L[1]

def key_atoms(el, X):
    """indices of N, O, alpha carbons, methyl carbons, C4 in a TEMPO monomer"""
    el = np.array(el); D = np.linalg.norm(X[:, None] - X[None], axis=2)
    bond = (D < 1.65) & (D > 0.1)
    N = int(np.where(el == "N")[0][0]); O = int(np.where(el == "O")[0][0])
    ca = [i for i in np.where(bond[N])[0] if el[i] == "C"]
    me = [j for c in ca for j in np.where(bond[c])[0]
          if el[j] == "C" and (el[bond[j]] == "H").sum() == 3]
    ring = [j for c in ca for j in np.where(bond[c])[0] if el[j] == "C" and j not in me]   # C3, C5
    c4 = [k for r in ring for k in np.where(bond[r])[0] if el[k] == "C" and k not in ca]
    c4 = int(pd.Series(c4).mode()[0])
    return dict(N=N, O=O, ca=sorted(ca), me=sorted(me), c4=c4, ring=sorted(set(ring)))

rows, G = [], []
first = None
for name in sorted(J):
    el, X, comment = read_xyz(os.path.join(RAW, "generated_structures", name + ".xyz"))
    assert len(el) == 58 and el[:29] == el[29:]
    if first is None:
        first = key_atoms(el[:29], X[:29]); print("monomer key atoms:", first)
    k = first
    def pick(off):
        return np.vstack([X[off + k["N"]], X[off + k["O"]], X[off + k["ca"][0]], X[off + k["ca"][1]],
                          X[off + k["c4"]]] + [X[off + m] for m in k["me"]])
    A, B = pick(0), pick(29)
    G.append(np.stack([A, B]))
    m = re.search(r"OO=([\d.]+)", comment)
    rows.append(dict(name=name, J=J[name], OO_header=float(m.group(1)) if m else np.nan))
G = np.array(G)                                   # (n, 2, 9, 3): N, O, Ca1, Ca2, C4, Me1-4
df = pd.DataFrame(rows)
df["r_OO"] = np.linalg.norm(G[:, 0, 1] - G[:, 1, 1], axis=1)
assert np.allclose(df.r_OO, df.OO_header, atol=0.006)
np.savez_compressed(os.path.join(OUT, "pairs_keyatoms.npz"), G=G, names=df.name.to_numpy(),
                    labels=np.array(["N", "O", "Ca1", "Ca2", "C4", "Me1", "Me2", "Me3", "Me4"]))
# all-atom geometry for descriptors that need H atoms
allX = np.array([read_xyz(os.path.join(RAW, "generated_structures", n + ".xyz"))[1] for n in df.name])
np.savez_compressed(os.path.join(OUT, "pairs_allatoms.npz"), X=allX, elements=np.array(el[:29]))
df.to_csv(os.path.join(OUT, "pairs_J.csv"), index=False)
print(df.J.describe())
print(pd.crosstab(pd.cut(df.r_OO, [2, 3, 4, 5, 6, 7, 11]), np.sign(df.J)))
print(df.groupby(pd.cut(df.r_OO, [2, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 11])).J.agg(["count", "mean", "std", "min", "max", lambda s: np.median(np.abs(s))]))
