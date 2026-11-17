#!/usr/bin/env python3
"""
m2_12_class_persistence.py
Do radicals keep their type?  Class descriptors are recomputed for the first
5 ns (frames 0-499) and the last 5 ns (frames 500-999) of each trajectory --
windows of equal length, so S2 is not biased -- and both are classified with
the fitted model of m2_11.  Reports the 5-ns transition matrix per replica.
"""
import os, numpy as np, pandas as pd, joblib
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
M = joblib.load(os.path.join(D, "module2_class_model.joblib"))
def S2_of(u):
    Mx = np.einsum("tmi,tmj->mij", u, u) / len(u); return 1.5 * (Mx**2).sum((1, 2)) - 0.5
def theta0(S2):
    S = np.sqrt(np.clip(S2, 0, 1)); c = (-1 + np.sqrt(1 + 8 * S)) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
def classify(o, w):
    u = o["u_NO"][w].astype(float)
    F = pd.DataFrame(dict(h_spin=o["h_spin"][w].mean(0), cosNO=o["cos_NO"][w].mean(0),
                          theta0=theta0(S2_of(u)), n8=o["ncount_8"][w].mean(0), tilt=o["tilt"][w].mean(0)))
    lab = M["gmm"].predict(M["scaler"].transform(F[M["feats"]]))
    return np.array([M["remap"][l] for l in lab])
rows, mats = [], {}
for T, r in RUNS:
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz")); keep = ~np.isin(o["mol_ids"], DEFECT)
    a, b = classify(o, slice(0, 500))[keep], classify(o, slice(500, 1000))[keep]
    Mt = np.zeros((4, 4))
    for i, j in zip(a, b): Mt[i, j] += 1
    mats[(T, r)] = Mt
    rows.append(dict(T=T, rep=r, retained=float((a == b).mean())))
R = pd.DataFrame(rows); print(R.groupby("T").retained.agg(["mean", "std"]).round(3))
for T in (100, 300):
    P = np.array([mats[(T, r)] / np.maximum(mats[(T, r)].sum(1, keepdims=True), 1) for r in (1, 2, 3)])
    print(f"{T} K transition probabilities (row = class in first 5 ns), mean over replicas:\n", P.mean(0).round(2))
    np.save(os.path.join(D, f"module2_class_transition_T{T}.npy"), P)
R.to_csv(os.path.join(D, "module2_class_persistence.csv"), index=False)
