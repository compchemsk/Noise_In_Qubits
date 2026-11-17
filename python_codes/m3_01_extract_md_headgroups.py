#!/usr/bin/env python3
"""
m3_01_extract_md_headgroups.py
Extracts the nitroxide head-group key atoms of all 141 radicals from the six
production trajectories (frames 100-1099; 10 ns, 10 ps per frame), in the
same order as the Module III training geometries:
    N1, O1, C2, C3 (alpha carbons), C5 (C4 of the piperidine ring),
    C6, C7, C8, C9 (methyl carbons)
plus the 18 methyl hydrogens (for contact descriptors).
Output: data/module3/md_keyatoms_T{T}_r{r}.npz  (float32)
"""
import os, numpy as np
from m2_00_io import *
OUT = os.path.join(ROOT, "data", "module3"); os.makedirs(OUT, exist_ok=True)
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res = P["name"], P["resid"]
lin = P["resname"] == "LIN"; mol_ids = np.unique(res[lin])
def idx(n):
    a = np.where(lin & (names == n))[0]; assert len(a) == len(mol_ids); return a
KEY = ["N1", "O1", "C2", "C3", "C5", "C6", "C7", "C8", "C9"]
HME = [f"H{k}" for k in range(6, 18)]               # H6-H17 sit on C6-C9
ik = np.stack([idx(n) for n in KEY], 1); ih = np.stack([idx(n) for n in HME], 1)
from m2_00_io import DEFECT as _DEF; DEFECT = np.array(_DEF, dtype=int)
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])[100:]
    np.savez_compressed(os.path.join(OUT, f"md_keyatoms_T{T}_r{r}.npz"),
                        K=X[:, ik].astype(np.float32), H=X[:, ih].astype(np.float32),
                        mol_ids=mol_ids, intact=~np.isin(mol_ids, DEFECT), key=np.array(KEY))
    d = np.linalg.norm(X[-1, ik[:, 0]] - X[-1, ik[:, 1]], axis=1)
    print(T, r, X.shape[0], "frames; N-O range (last frame) %.3f-%.3f A" % (d.min(), d.max()))
