#!/usr/bin/env python3
"""
m4_03_radical_level.py
Radical-level test of how the environment shapes the spin-active TEMPO modes.

Frame-level mutual information between a mode and an environmental variable is dominated by the fact that
both carry a static, radical-specific part, so the effective number of independent samples is the number
of radicals, not the number of frames (see m4_02, radical-block null). Here each radical is reduced to
  static distortion   s_k(i) = <Q_k(i,t)>_t
  dynamic amplitude   a_k(i) = Var_t[Q_k(i,t)] / (k_B T / hbar omega_k)
and to the time averages of its environment; the radicals of the three replicas are pooled (381 per T).
Tests: Spearman rho with permutation p-value (10 000 permutations, Benjamini-Hochberg over all tests),
Kruskal-Wallis across classes, and KSG mutual information with a shuffle null.
"""
import os, json, numpy as np, pandas as pd
from scipy.stats import spearmanr, kruskal
from sklearn.feature_selection import mutual_info_regression

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D2 = os.path.join(ROOT, "data", "module2"); D4 = os.path.join(ROOT, "data", "module4")
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}; KB = 0.6950348; LN2 = np.log(2)
CN = {0: "S", 1: "M", 2: "R", 3: "F"}; RMSD_MAX = 0.3; MINFR = 100
V = json.load(open(os.path.join(D4, "mode_vectors.json")))["TEMPO"]
el = V["el"]; m = np.array([MASS[e] for e in el]); Vk = np.array(V["V"]); nu = np.array(V["nu"]); K = 3
labels = pd.read_csv(os.path.join(D2, "module2_class_labels.csv"))
ENV = ["h_spin", "cos_NO", "tilt", "ncount_8", "dmin_OAu", "nn1", "gauche"]
rng = np.random.default_rng(7)

def project(T):
    data = {}
    for r in (1, 2, 3):
        a = np.load(os.path.join(D4, f"aligned_TEMPO_T{T}_r{r}.npz")); Y = a["Y"].astype(float)
        Y[a["core_rmsd"] > RMSD_MAX] = np.nan; data[r] = (Y, a["mol"])
    xbar = np.nanmean(np.concatenate([d[0].reshape(-1, len(el), 3) for d in data.values()]), 0)
    out = {}
    for r, (Y, mol) in data.items():
        dx = Y - xbar; pres = ~np.isnan(dx[..., 0]); ok = pres.sum(-1) > 0; dx = np.nan_to_num(dx)
        Q = np.einsum("kja,j,fnja->fnk", Vk, m, dx) / np.einsum("kja,j,kja,fnj->fnk", Vk, m, Vk, pres.astype(float))
        Q[~ok] = np.nan; out[r] = (Q, mol)
    return out

rows = []
for T in (100, 300):
    P = project(T)
    for r, (Q, mol) in P.items():
        o = np.load(os.path.join(D2, f"obs_T{T}_r{r}.npz")); idx = np.searchsorted(o["mol_ids"], mol)
        lab = labels[(labels["T"] == T) & (labels.rep == r)].set_index("mol").loc[mol]
        nval = (~np.isnan(Q[..., 0])).sum(0)
        for n_, mo in enumerate(mol):
            if nval[n_] < MINFR: continue
            row = dict(T=T, rep=r, mol=int(mo), cls=CN[int(lab.cls.iloc[n_])], edge=bool(lab.edge_dist.iloc[n_] < 5), nvalid=int(nval[n_]))
            for k in range(K):
                q = Q[:, n_, k]; row[f"s{k+1}"] = np.nanmean(q); row[f"a{k+1}"] = np.nanvar(q) / (KB * T / nu[k])
            for e in ENV: row[e] = float(np.nanmean(o[e][-500:][:, idx[n_]]))
            rows.append(row)
R = pd.DataFrame(rows); R.to_csv(os.path.join(D4, "m4_radical_level.csv"), index=False)

def perm_p(x, y, n=10000):
    r0 = spearmanr(x, y)[0]; rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    null = np.array([np.mean(rx * ry[rng.permutation(len(ry))]) for _ in range(n)])
    return r0, (1 + np.sum(np.abs(null) >= abs(r0))) / (n + 1)

tests = []
for T in (100, 300):
    X = R[R["T"] == T]
    for q in [f"s{k+1}" for k in range(K)] + [f"a{k+1}" for k in range(K)]:
        for e in ENV:
            g = X[[q, e]].dropna(); rho, p = perm_p(g[q].to_numpy(), g[e].to_numpy())
            I = mutual_info_regression(g[[e]].to_numpy(), g[q].to_numpy(), n_neighbors=3, random_state=0)[0] / LN2
            null = [mutual_info_regression(g[[e]].to_numpy(), rng.permutation(g[q].to_numpy()), n_neighbors=3, random_state=s)[0] / LN2 for s in range(20)]
            tests.append(dict(T=T, quantity=q, var=e, n=len(g), rho=rho, p=p, MI=I, MI_null95=np.percentile(null, 95)))
        groups = [X[X.cls == c][q].dropna().to_numpy() for c in ("F", "R", "M", "S") if (X.cls == c).sum() >= 5]
        H, p = kruskal(*groups)
        tests.append(dict(T=T, quantity=q, var="class", n=len(X), rho=np.nan, p=p, MI=np.nan, MI_null95=np.nan, H=H))
Tt = pd.DataFrame(tests)
# Benjamini-Hochberg false-discovery rate over all tests
p = Tt.p.to_numpy(); o_ = np.argsort(p); bh = np.empty_like(p); n = len(p)
bh[o_] = np.minimum.accumulate((p[o_] * n / np.arange(1, n + 1))[::-1])[::-1]; Tt["q_BH"] = np.minimum(bh, 1)
Tt.to_csv(os.path.join(D4, "m4_radical_tests.csv"), index=False)
cl = R.groupby(["T", "cls"])[[f"a{k+1}" for k in range(K)] + [f"s{k+1}" for k in range(K)]].agg(["median", "count"])
cl.to_csv(os.path.join(D4, "m4_radical_by_class.csv"))
pd.set_option("display.width", 220)
print(Tt[Tt.q_BH < 0.05].round(4).to_string()); print(cl.round(3).to_string())
print("radicals per T:", R.groupby("T").size().to_dict())
