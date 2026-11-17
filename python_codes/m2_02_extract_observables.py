#!/usr/bin/env python3
"""
m2_02_extract_observables.py
Per-molecule, per-frame observables for all six trajectories (3 replicas x
100 K and 300 K).  Frames 0-99 = restrained heating (discarded); frames
100-1099 = 10 ns at the target temperature, 10 ps per frame.

Geometry: the Au slab lies in the yz plane; the layer facing the monolayer is
the one at x ~ 0 and the monolayer extends towards -x.  Surface normal
n = (-1, 0, 0); heights are measured from the instantaneous mean x of that
top Au layer.

Saved per run (data/module2/obs_T{T}_r{r}.npz), arrays (frames x molecules):
  h_O1, h_N1, h_S           heights above the top Au layer (A)
  h_spin                    height of the N-O midpoint (spin position)
  cos_NO                    cos(angle between N->O and n)
  u_NO                      N->O unit vectors (frames x mol x 3)
  tilt                      chain tilt (S1->C13 vs n, degrees)
  gauche                    fraction of gauche C-C-C-C dihedrals in C23...C13
  dmin_OAu                  O1 distance to nearest Au atom (A)
  spin_yz                   lateral position of the N-O midpoint (frames x mol x 2)
  nn1, nn2                  nearest / second-nearest spin-spin distance (A)
  ncount_6, ncount_8        number of other spins within 6 and 8 A
"""
import os, numpy as np
from scipy.spatial import cKDTree
from m2_00_io import *

OUT = os.path.join(ROOT, "data", "module2"); os.makedirs(OUT, exist_ok=True)
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res = P["name"], P["resid"]
lin = P["resname"] == "LIN"
mol_ids = np.unique(res[lin])                       # 141 molecules
def idx(n):  # atom index of name n in every molecule, ordered by mol_ids
    a = np.where(lin & (names == n))[0]
    assert len(a) == len(mol_ids)
    return a
iN, iO, iS, iC13 = idx("N1"), idx("O1"), idx("S1"), idx("C13")
chain = [idx(f"C{k}") for k in range(23, 12, -1)]  # C23 (next to S) ... C13 (next to triazole)
au = np.where(~lin)[0]
FIRST = 100                                         # first unrestrained frame
nvec = np.array([-1.0, 0.0, 0.0])

def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1); y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))

for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])[FIRST:]
    nf, nm = X.shape[0], len(mol_ids)
    top = au[X[0, au, 0] < 0.75]                    # Au layer facing the monolayer
    xtop = X[:, top, 0].mean(1)[:, None]
    h = lambda ii: xtop - X[:, ii, 0]
    N, O = X[:, iN], X[:, iO]
    spin = 0.5 * (N + O)
    u = O - N; u /= np.linalg.norm(u, axis=2, keepdims=True)
    cvec = X[:, iC13] - X[:, iS]; cvec /= np.linalg.norm(cvec, axis=2, keepdims=True)
    tilt = np.degrees(np.arccos(np.clip(cvec @ nvec, -1, 1)))
    dih = np.stack([dihedral(X[:, chain[k]], X[:, chain[k+1]], X[:, chain[k+2]], X[:, chain[k+3]])
                    for k in range(len(chain) - 3)], axis=2)
    gauche = (np.abs(dih) < 120.0).mean(2)
    tAu = cKDTree(X[0, top]) if False else None
    dmin = np.empty((nf, nm)); nn1 = np.empty((nf, nm)); nn2 = np.empty((nf, nm))
    c6 = np.empty((nf, nm), int); c8 = np.empty((nf, nm), int)
    for f in range(nf):
        dmin[f] = cKDTree(X[f, au]).query(O[f])[0]
        tr = cKDTree(spin[f]); d, _ = tr.query(spin[f], k=3)
        nn1[f], nn2[f] = d[:, 1], d[:, 2]
        D = np.linalg.norm(spin[f][:, None] - spin[f][None], axis=2)
        c6[f] = (D < 6.0).sum(1) - 1; c8[f] = (D < 8.0).sum(1) - 1
    np.savez_compressed(os.path.join(OUT, f"obs_T{T}_r{r}.npz"),
        mol_ids=mol_ids, time_ns=(np.arange(nf) * 0.01), h_O1=h(iO), h_N1=h(iN), h_S=h(iS),
        h_spin=xtop - spin[:, :, 0], cos_NO=u @ nvec, u_NO=u.astype(np.float32), tilt=tilt,
        gauche=gauche, dmin_OAu=dmin, spin_yz=spin[:, :, 1:].astype(np.float32),
        S_yz=X[:, iS, 1:].astype(np.float32), nn1=nn1, nn2=nn2, ncount_6=c6, ncount_8=c8)
    print(f"T{T} r{r}: frames {nf}; <h_O1> {h(iO).mean():.2f} A; <tilt> {tilt.mean():.1f}; "
          f"<gauche> {gauche.mean():.3f}; <nn1> {nn1.mean():.2f} A; min dOAu {dmin.min():.2f}")
