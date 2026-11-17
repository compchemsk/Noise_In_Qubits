#!/usr/bin/env python3
"""m3_02_features_training.py -- descriptors of the 1364 DFT pairs -> data/module3/train_features.csv"""
import os, numpy as np, pandas as pd
from m3_features import features
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "module3")
df = pd.read_csv(os.path.join(D, "pairs_J.csv"))
X = np.load(os.path.join(D, "pairs_allatoms.npz"))["X"]           # (n, 58, 3)
el = np.load(os.path.join(D, "pairs_allatoms.npz"))["elements"]
key = [27, 28, 3, 4, 1, 11, 15, 19, 23]                             # N O Ca Ca C4 Me x4 (m3_00)
M = X[0, :29]
Dm = np.linalg.norm(M[:, None] - M[None], axis=2)
hme = [j for m in key[5:] for j in np.where((Dm[m] < 1.2) & (el == "H"))[0]]
assert len(hme) == 12
A, B = X[:, key], X[:, [k + 29 for k in key]]
HA, HB = X[:, hme], X[:, [h + 29 for h in hme]]
f = features(A, B, HA, HB)
F = pd.DataFrame(f); F.insert(0, "name", df.name); F["J"] = df.J
F.to_csv(os.path.join(D, "train_features.csv"), index=False)
# symmetry check: swapping A and B must leave every symmetric descriptor unchanged
g = features(B, A, HB, HA)
bad = [k for k in f if not k.startswith("u3_") and not np.allclose(f[k], g[k], atol=1e-9)]
print("descriptors:", len(f), " A<->B asymmetric (should be only u3_*):", bad)
print(F[["r_OO", "dmin_O_O", "r_NO_min", "dmin_O_H", "S_somo", "J"]].describe().T)
