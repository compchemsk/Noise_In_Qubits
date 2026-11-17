#!/usr/bin/env python3
"""
m4_02_mode_information.py
Statistics and mutual information of the spin-active TEMPO modes inside the monolayer.

Input : data/module4/aligned_TEMPO_T*_r*.npz (from m4_01), module2 observables and class labels.
Basis : the three TEMPO modes (78.3, 117.7, 197.8 cm-1); coverage of the mapped atoms >= 0.993.

Definitions (all per temperature, means +- SD over the three replicas)
  Q_k(i,t)   = v_k^T M [x_i(t) - xbar] / (v_k^T M v_k)                 projected dimensionless coordinate
  R_k        = <Q_k^2> / (k_B T / hbar omega_k)                         amplitude relative to a classical
                                                                        harmonic mode of the DFT frequency
  static share  = Var_i( <Q_k>_t ) / Var(Q_k)                           radical-to-radical (frozen) part
  dg(i,t)    = sum_k c1_k Q_k + c2_k (Q_k^2 - <Q_k^2>)                  same for A (MHz)
  dnu_g      = (nu_mw / g_iso) dg,   nu_mw = 35 GHz
  C(tau)     = < d(t) d(t+tau) > / < d^2 >,  d = deviation from the radical's own time mean
  I(X;Y)     = Kraskov-Stoegbauer-Grassberger estimate (k = 3), in bits
  I_G(X;Y)   = -1/2 log2(1 - r^2)   (mutual information of a Gaussian with Pearson r)
  null       = MI after random shuffling of Y (5 shuffles); reported: mean + 3 SD
"""
import os, json, numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D1 = os.path.join(ROOT, "data", "module1"); D2 = os.path.join(ROOT, "data", "module2")
D3 = os.path.join(ROOT, "data", "module3"); D4 = os.path.join(ROOT, "data", "module4")
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
KB = 0.6950348; NU_MW = 35000.0; LN2 = np.log(2)
CN = {0: "S", 1: "M", 2: "R", 3: "F"}
RMSD_MAX = 0.3          # A; chair conformers close to the DFT structure (twist-boat and other conformers excluded)
SUB = 5                 # every 5th frame for MI (50 ps spacing)

V = json.load(open(os.path.join(D4, "mode_vectors.json")))["TEMPO"]
chk = json.load(open(os.path.join(D4, "mode_checks.json")))
el = V["el"]; m = np.array([MASS[e] for e in el])               # all 29 DFT atoms
Vk = np.array(V["V"]); nu = np.array(V["nu"]); K = len(nu)
summ = pd.read_csv(os.path.join(D1, "module1_coupling_summary.csv")).query("molecule=='TEMPO'").set_index("mode")
an = pd.read_csv(os.path.join(D1, "module1_anisotropic_couplings.csv")).query("molecule=='TEMPO'").set_index("mode")
c1g, c1A = summ.dg_dQ.to_numpy(), summ.dA_dQ_MHz.to_numpy(); c2g, c2A = an.c2_g_iso.to_numpy(), an.c2_A_iso.to_numpy()
G0 = 2.0061945
labels = pd.read_csv(os.path.join(D2, "module2_class_labels.csv"))
rng = np.random.default_rng(20261116)

def mi(x, y, seed=0):
    return mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=3, random_state=seed)[0] / LN2

def mi_gauss(x, y):
    r = np.corrcoef(x, y)[0, 1]; return -0.5 * np.log2(1 - r ** 2), r

def mi_null(x, y):
    v = [mi(x, rng.permutation(y), seed=s) for s in range(5)]; return np.mean(v) + 3 * np.std(v)

def mi_block_null(x, Ymat, valid, nrep=5):
    """null for radical-level variables: permute whole radicals (columns) of the environment series,
    which keeps each radical's static character and the time correlation intact"""
    v = []
    for s_ in range(nrep):
        perm = rng.permutation(Ymat.shape[1]); y = Ymat[:, perm][valid]; g = np.isfinite(y)
        v.append(mi(x[g], y[g], seed=s_))
    return np.mean(v) + 3 * np.std(v)

ENV = {"h_spin": "spin height", "cos_NO": "N-O orientation", "tilt": "chain tilt", "ncount_8": "spins within 8 A",
       "dmin_OAu": "O-Au distance", "nn1": "nearest-neighbor distance", "gauche": "gauche fraction"}

res = {"per_rep": [], "class": [], "env": [], "intra": [], "inter_nn": [], "inter_dist": [], "acf": [], "sh": []}
for T in (100, 300):
    qcl = KB * T / nu
    allQ = {}
    for r in (1, 2, 3):
        a = np.load(os.path.join(D4, f"aligned_TEMPO_T{T}_r{r}.npz")); Y = a["Y"].astype(float); mol = a["mol"]
        ok = a["core_rmsd"] <= RMSD_MAX
        allQ[r] = (Y, mol, ok)
    xbar = np.nanmean(np.concatenate([allQ[r][0][allQ[r][2]] for r in (1, 2, 3)]), 0)  # ensemble-mean head (per DFT atom)
    Qall = {}
    for r in (1, 2, 3):
        Y, mol, ok = allQ[r]
        dx = Y - xbar; pres = ~np.isnan(dx[..., 0]); dx = np.nan_to_num(dx)        # the empty C4-H slot is masked
        num = np.einsum("kja,j,fnja->fnk", Vk, m, dx)
        den = np.einsum("kja,j,kja,fnj->fnk", Vk, m, Vk, pres.astype(float))
        Q = num / den                                                                  # (f, n, K)
        Q[~ok] = np.nan; Qall[r] = (Q, mol, ok)
    Q2mean = np.nanmean(np.concatenate([Qall[r][0].reshape(-1, K) for r in (1, 2, 3)]) ** 2, 0)
    for r in (1, 2, 3):
        Q, mol, ok = Qall[r]
        o = np.load(os.path.join(D2, f"obs_T{T}_r{r}.npz")); om = o["mol_ids"]; idx = np.searchsorted(om, mol)
        env = {k: o[k][-500:][:, idx].astype(float) for k in ENV}
        lab = labels[(labels["T"] == T) & (labels.rep == r)].set_index("mol").loc[mol]
        cls = lab.cls.map(CN).to_numpy(); edge = (lab.edge_dist < 5).to_numpy()
        dg = (c1g * Q + c2g * (Q ** 2 - Q2mean)).sum(-1); dA_ = (c1A * Q + c2A * (Q ** 2 - Q2mean)).sum(-1)
        dnu = NU_MW / G0 * dg
        rm = np.load(os.path.join(D4, f"aligned_TEMPO_T{T}_r{r}.npz"))["core_rmsd"]
        for c in ("F", "R", "M", "S"):
            pass
        res["per_rep"].append(dict(T=T, rep=r, excluded_frac=float(1 - ok.mean()),
                                   frac_twist=float(((rm > 0.3) & (rm <= 0.5)).mean()), frac_other=float((rm > 0.5).mean()),
                                   frac_absQ_gt1=np.nanmean(np.abs(Q) > 1, axis=(0, 1)).tolist(),
                                   R=(np.nanmean(Q ** 2, (0, 1)) / qcl).tolist(),
                                   static_share=[float(np.nanvar(np.nanmean(Q[..., k], 0)) / np.nanvar(Q[..., k])) for k in range(K)]))
        # class-resolved amplitudes and spin-Hamiltonian spread
        for c in ("F", "R", "M", "S"):
            sel = cls == c
            if sel.sum() < 3: continue
            qq = Q[:, sel]; dn = dnu[:, sel]; da = dA_[:, sel]
            res["class"].append(dict(T=T, rep=r, cls=c, n=int(sel.sum()), R=(np.nanmean(qq ** 2, (0, 1)) / qcl).tolist(),
                                     sd_dnu=float(np.nanstd(dn)), sd_dA=float(np.nanstd(da)),
                                     sd_dnu_static=float(np.nanstd(np.nanmean(dn, 0))), sd_dA_static=float(np.nanstd(np.nanmean(da, 0)))))
        for c, sel in (("interior", ~edge), ("edge", edge), ("all", np.ones_like(edge))):
            dn = dnu[:, sel]; da = dA_[:, sel]
            res["sh"].append(dict(T=T, rep=r, group=c, sd_dnu=float(np.nanstd(dn)), sd_dA=float(np.nanstd(da)),
                                  sd_dnu_static=float(np.nanstd(np.nanmean(dn, 0))), sd_dA_static=float(np.nanstd(np.nanmean(da, 0))),
                                  sd_dnu_dyn=float(np.sqrt(np.nanmean(np.nanvar(dn, 0)))), sd_dA_dyn=float(np.sqrt(np.nanmean(np.nanvar(da, 0))))))
        # time autocorrelation of the dynamic part (deviation from each radical's own mean)
        for name, s in (("dnu_g", dnu), ("dA", dA_)) + tuple((f"Q{k+1}", Q[..., k]) for k in range(K)):
            dd = s - np.nanmean(s, 0)
            row = dict(T=T, rep=r, quantity=name)
            for lag in (1, 2, 5, 10, 20, 50, 100):
                row[f"lag{lag}"] = float(np.nanmean(dd[:-lag] * dd[lag:]) / np.nanmean(dd ** 2))
            res["acf"].append(row)
        # ---- mutual information (subsampled frames, valid snapshots only)
        fs = np.arange(0, Q.shape[0], SUB)
        Qs = Q[fs]; valid = ~np.isnan(Qs[..., 0])
        flat = lambda arr: arr[fs][valid]
        for k in range(K):
            x = Qs[..., k][valid]
            for e in ENV:
                y = flat(env[e]); good = np.isfinite(y)
                I = mi(x[good], y[good]); Ig, rr = mi_gauss(x[good], y[good])
                res["env"].append(dict(T=T, rep=r, mode=k + 1, var=e, MI=I, MI_gauss=Ig, r=rr, null=mi_null(x[good], y[good]),
                                       null_block=mi_block_null(x, env[e][fs], valid)))
            cl = np.broadcast_to(cls, Qs.shape[:2])[valid]
            Ic = mutual_info_classif(x.reshape(-1, 1), cl, n_neighbors=3, random_state=0)[0] / LN2
            clm = np.broadcast_to(cls, Qs.shape[:2])
            bn = []
            for s_ in range(5):
                pc = clm[:, rng.permutation(clm.shape[1])][valid]
                bn.append(mutual_info_classif(x.reshape(-1, 1), pc, n_neighbors=3, random_state=s_)[0] / LN2)
            res["env"].append(dict(T=T, rep=r, mode=k + 1, var="class", MI=Ic, MI_gauss=np.nan, r=np.nan,
                                   null_block=np.mean(bn) + 3 * np.std(bn),
                                   null=(lambda v: np.mean(v) + 3 * np.std(v))([mutual_info_classif(x.reshape(-1, 1), rng.permutation(cl), n_neighbors=3, random_state=s)[0] / LN2 for s in range(5)])))
            for l in range(k + 1, K):
                y = Qs[..., l][valid]
                I = mi(x, y); Ig, rr = mi_gauss(x, y)
                res["intra"].append(dict(T=T, rep=r, pair=f"{k+1}-{l+1}", MI=I, MI_gauss=Ig, r=rr, null=mi_null(x, y)))
        # spin-Hamiltonian fluctuation vs environment
        for nm_, s in (("dnu_g", dnu), ("dA", dA_)):
            x = s[fs][valid]
            for e in ENV:
                y = flat(env[e]); good = np.isfinite(y)
                I = mi(x[good], y[good]); Ig, rr = mi_gauss(x[good], y[good])
                res["env"].append(dict(T=T, rep=r, mode=nm_, var=e, MI=I, MI_gauss=Ig, r=rr, null=mi_null(x[good], y[good]),
                                       null_block=mi_block_null(x, env[e][fs], valid)))
        # ---- between radicals: nearest neighbor and distance bins (spin positions from h and in-plane yz)
        pos = np.concatenate([o["h_spin"][-500:][:, idx, None], o["spin_yz"][-500:][:, idx]], -1).astype(float)[fs]
        Dm = np.linalg.norm(pos[:, :, None] - pos[:, None], axis=-1); n = Dm.shape[1]
        Dm[:, np.arange(n), np.arange(n)] = np.inf
        nn = Dm.argmin(2)
        Qj = np.take_along_axis(Qs, nn[..., None].repeat(K, -1), 1)
        vv = valid & ~np.isnan(Qj[..., 0])
        for k in range(K):
            for l in range(K):
                x = Qs[..., k][vv]; y = Qj[..., l][vv]
                I = mi(x, y); Ig, rr = mi_gauss(x, y)
                res["inter_nn"].append(dict(T=T, rep=r, k=k + 1, l=l + 1, MI=I, MI_gauss=Ig, r=rr, null=mi_null(x, y)))
        dnus = dnu[fs]; dnuj = np.take_along_axis(dnus, nn, 1); vvv = ~np.isnan(dnus) & ~np.isnan(dnuj)
        I = mi(dnus[vvv], dnuj[vvv]); Ig, rr = mi_gauss(dnus[vvv], dnuj[vvv])
        res["inter_nn"].append(dict(T=T, rep=r, k="dnu_g", l="dnu_g", MI=I, MI_gauss=Ig, r=rr, null=mi_null(dnus[vvv], dnuj[vvv])))
        iu = np.triu_indices(n, 1)
        for lo, hi in ((4, 7), (7, 10), (10, 14), (14, 20), (20, 30)):
            fsel, isel = np.where((Dm[:, iu[0], iu[1]] >= lo) & (Dm[:, iu[0], iu[1]] < hi))
            a_i, b_i = iu[0][isel], iu[1][isel]
            pick = rng.choice(len(fsel), size=min(15000, len(fsel)), replace=False)
            for k in range(K):
                x = Qs[fsel[pick], a_i[pick], k]; y = Qs[fsel[pick], b_i[pick], k]; g = ~np.isnan(x) & ~np.isnan(y)
                res["inter_dist"].append(dict(T=T, rep=r, bin=f"{lo}-{hi}", mode=k + 1, MI=mi(x[g], y[g]),
                                              MI_gauss=mi_gauss(x[g], y[g])[0], null=mi_null(x[g], y[g]), n=int(g.sum())))
        print(f"T{T} r{r} done; excluded {1-ok.mean():.4f}", flush=True)

json.dump(res, open(os.path.join(D4, "mode_information_raw.json"), "w"))
# replica summaries
def agg(rows, keys, cols):
    X = pd.DataFrame(rows)
    if X.empty: return X
    for c in cols:
        if c in X and isinstance(X[c].iloc[0], list):
            for j in range(len(X[c].iloc[0])): X[f"{c}{j+1}"] = X[c].apply(lambda v: v[j])
            X = X.drop(columns=c)
    num = [c for c in X.columns if c not in keys + ["rep"] and pd.api.types.is_numeric_dtype(X[c])]
    g = X.groupby(keys)[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")); return out
for name, keys, cols in (("per_rep", ["T"], ["frac_absQ_gt1", "R", "static_share"]), ("class", ["T", "cls"], ["R"]),
                         ("sh", ["T", "group"], []), ("acf", ["T", "quantity"], []), ("env", ["T", "mode", "var"], []),
                         ("intra", ["T", "pair"], []), ("inter_nn", ["T", "k", "l"], []), ("inter_dist", ["T", "bin", "mode"], [])):
    rows = res[name]
    if name in ("env", "inter_nn"):
        for r_ in rows: r_["mode" if name == "env" else "k"] = str(r_["mode" if name == "env" else "k"]);
        if name == "inter_nn":
            for r_ in rows: r_["l"] = str(r_["l"])
    agg(rows, keys, cols).to_csv(os.path.join(D4, f"m4_{name}.csv"))
print("written")
