#!/usr/bin/env python3
"""
m3_12_revision_analyses.py
Additional analyses requested in review, computed from the existing trajectories
and exchange predictions (no new DFT or MD):

 (1) applicability domain (AD) of every frame-level evaluation, of resolved pairs and
     of pairs in distance bins (not only pairs closer than 6 A);
 (2) random dilution of the intact radicals (keep 100/75/50/25 %) with the
     time-averaged couplings and dipolar widths of the remaining radicals;
 (3) network statistics without surface-bound (S) radicals and for interior radicals only;
 (4) point-dipole couplings with the spin distributed over N and O (two-point model)
     against the N-O midpoint model;
 (5) triplet zero-field splitting D = 3d/2 of resolved pairs and the ratio 2|J|/D;
 (6) AF share expected from the DFT library alone, reweighted to the distance
     distribution of the resolved MD pairs (tests whether the 75 % AF share is inherited).
Output: tables/tabS_m3_revision_*.csv and printed summaries.
"""
import os, json, numpy as np, pandas as pd, joblib
from m3_ml_common import D
ROOT = os.path.join(D, "..", ".."); TAB = os.path.join(ROOT, "tables")
CM2MHZ = 29979.2458; J_RES = 0.10; NU_DD = 52.04; Q_MHZ = 35.0e3; A_N = 46.0
rng = np.random.default_rng(20261115)
M = joblib.load(os.path.join(D, "final_model.joblib")); ADT = M["ad_threshold"]
labels = pd.read_csv(os.path.join(ROOT, "data", "module2", "module2_class_labels.csv"))
CN = {0: "S", 1: "M", 2: "R", 3: "F"}

def dd_matrix(P1, P2, w1, w2):
    """secular coupling (MHz), field along x (surface normal); P*: (f, m, k, 3) point sets with weights w"""
    tot = 0.0
    for a in range(P1.shape[2]):
        for b in range(P2.shape[2]):
            dv = P1[:, :, None, a] - P2[:, None, :, b]; r = np.linalg.norm(dv, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                c = dv[..., 0] / r; t = NU_DD / (r / 10) ** 3 * (1 - 3 * c ** 2)
            tot = tot + w1[a] * w2[b] * t
    return tot

def net_stats(keep, Pi, dd2, Jcol="Jabs_mean"):
    """keep: bool over molecule index; Pi: intact pair table; dd2: (m,m) time-mean squared dipolar coupling"""
    idx = np.where(keep)[0]; sub = Pi[keep[Pi.i.values] & keep[Pi.j.values]]
    res = sub[sub[Jcol] > J_RES]
    deg = np.zeros(len(keep)); jmax = np.zeros(len(keep)); wex2 = np.zeros(len(keep))
    for a, b, j in zip(res.i, res.j, res[Jcol]):
        for s in (a, b):
            deg[s] += 1; jmax[s] = max(jmax[s], j); wex2[s] += (2 * j * CM2MHZ) ** 2
    m2 = np.sqrt(dd2[np.ix_(idx, idx)].sum(1))
    d, jm, we = deg[idx], jmax[idx], np.sqrt(wex2[idx])
    cp = d > 0
    return dict(n=len(idx), frac_isolated=np.mean(d == 0), degree=d.mean(),
                frac_dimer_Q=np.mean(2 * jm * CM2MHZ > Q_MHZ), frac_deg_ge2=np.mean(d >= 2),
                n_res_pairs=len(res), sqrtM2_median=np.median(m2),
                wex_over_M2=np.median(we[cp] / m2[cp]) if cp.any() else np.nan,
                AF_share=res.pAF_w.mean() if len(res) else np.nan)

AD_rows, dil_rows, sub_rows, dip_rows, zfs_rows = [], [], [], [], []
md_res_rNO = []
for T in (100, 300):
    for r in (1, 2, 3):
        z = np.load(os.path.join(D, f"mdJ_T{T}_r{r}.npz")); kz = np.load(os.path.join(D, f"md_keyatoms_T{T}_r{r}.npz"))
        P = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv"))
        intact, mol, ev, ad, rNO = z["intact"], z["mol_ids"], z["evaluated"], z["ad"], z["rNO"]
        Jabs = z["Jabs"].astype(float)
        okp = P.intact.values; resp = okp & (P.Jabs_mean.values > J_RES)
        # (1) applicability domain
        inside = (ad <= ADT) & ev
        row = dict(T=T, rep=r, all_eval=inside[:, okp].sum() / ev[:, okp].sum(),
                   resolved_pairs_eval=inside[:, resp].sum() / ev[:, resp].sum(),
                   frames_Jgt_res=inside[:, okp][Jabs[:, okp] > J_RES].mean())
        for lo, hi in ((0, 6), (6, 7), (7, 8), (8, 9)):
            m = ev[:, okp] & (rNO[:, okp] >= lo) & (rNO[:, okp] < hi)
            row[f"rNO_{lo}_{hi}"] = inside[:, okp][m].mean()
        AD_rows.append(row)
        md_res_rNO.append(np.nanmin(np.where(ev[:, resp], rNO[:, resp], np.nan), 0))
        # time-mean squared dipolar couplings (midpoint and two-point N/O models)
        K = kz["K"][-500:].astype(float)
        N, O = K[:, :, 0], K[:, :, 1]; mid = 0.5 * (N + O)
        d_mid = dd_matrix(mid[:, :, None], mid[:, :, None], [1.0], [1.0])
        d_two = dd_matrix(np.stack([N, O], 2), np.stack([N, O], 2), [0.5, 0.5], [0.5, 0.5])
        rr = np.linalg.norm(mid[:, :, None] - mid[:, None], axis=-1)
        for X in (d_mid, d_two):
            np.einsum("fii->fi", X)[:] = 0.0; X[rr > 15] = 0.0
        ii = np.where(intact)[0]
        dd2 = (d_mid ** 2).mean(0); dd2[~intact] = 0; dd2[:, ~intact] = 0
        # (4) dipolar models: nearest-neighbour |d| (field average not needed: same geometry) and sqrt(M2)
        rr_i = rr[:, ii][:, :, ii].copy(); np.einsum("fii->fi", rr_i)[:] = np.inf
        nn = rr_i.argmin(2)
        f_ix = np.arange(rr_i.shape[0])[:, None]; s_ix = np.arange(len(ii))[None, :]
        nuNN_mid = NU_DD / (rr_i.min(2) / 10) ** 3
        # point-dipole prefactor with spin on N and O (orientation-free magnitude, 1/r^3 weighting)
        rNN = np.linalg.norm(N[:, ii][:, :, None] - N[:, ii][:, None], axis=-1); rOO = np.linalg.norm(O[:, ii][:, :, None] - O[:, ii][:, None], axis=-1)
        rNO_ = np.linalg.norm(N[:, ii][:, :, None] - O[:, ii][:, None], axis=-1); rON = np.swapaxes(rNO_, 1, 2)
        inv3 = 0.25 * (rNN ** -3.0 + rOO ** -3.0 + rNO_ ** -3.0 + rON ** -3.0)
        nuNN_two = NU_DD * 1e3 * inv3[f_ix, s_ix, nn]
        m2_mid = np.sqrt((d_mid[:, ii][:, :, ii] ** 2).sum(2).mean(0)); m2_two = np.sqrt((d_two[:, ii][:, :, ii] ** 2).sum(2).mean(0))
        dip_rows.append(dict(T=T, rep=r, nuNN_mid_median=np.median(nuNN_mid.mean(0)), nuNN_two_median=np.median(nuNN_two.mean(0)),
                             sqrtM2_mid_median=np.median(m2_mid), sqrtM2_two_median=np.median(m2_two),
                             ratio_M2=np.median(m2_two / m2_mid)))
        # (5) triplet ZFS of resolved pairs: D = 3/2 * nu_dd(r_spin) (point dipole, midpoint)
        Rp = P[resp]
        Dz = 1.5 * NU_DD / (Rp.r_spin_mean.values / 10) ** 3
        zfs_rows.append(dict(T=T, rep=r, D_median_MHz=np.median(Dz), twoJ_over_D_median=np.median(2 * Rp.Jabs_mean.values * CM2MHZ / Dz),
                             frac_twoJ_gt_10D=np.mean(2 * Rp.Jabs_mean.values * CM2MHZ > 10 * Dz),
                             frac_twoJ_gt_aN=np.mean(2 * Rp.Jabs_mean.values * CM2MHZ > A_N)))
        # (2) dilution and (3) subsets
        Pi = P[okp]
        lab = labels[(labels["T"] == T) & (labels.rep == r)].set_index("mol")
        cls = np.array([CN[int(lab.cls[m])] if m in lab.index else "X" for m in mol])
        edge = np.array([lab.edge_dist[m] < 5 if m in lab.index else True for m in mol])
        for keepfrac in (1.0, 0.75, 0.5, 0.25):
            nd = 1 if keepfrac == 1 else 200
            acc = []
            for _ in range(nd):
                keep = intact.copy()
                if keepfrac < 1:
                    drop = rng.choice(ii, size=int(round((1 - keepfrac) * len(ii))), replace=False); keep[drop] = False
                acc.append(net_stats(keep, Pi, dd2))
            A = pd.DataFrame(acc).mean(); A["T"], A["rep"], A["keep"] = T, r, keepfrac
            dil_rows.append(A)
        for name, keep in (("all intact", intact), ("without S", intact & (cls != "S")),
                           ("interior only", intact & ~edge)):
            s = net_stats(keep, Pi, dd2); s.update(T=T, rep=r, subset=name); sub_rows.append(s)
        print(f"T{T} r{r} done", flush=True)

def agg(rows, keys):
    X = pd.DataFrame(rows); g = X.drop(columns="rep").groupby(keys)
    return g.mean().round(4).astype(str) + " +- " + g.std().round(4).astype(str)

os.makedirs(TAB, exist_ok=True)
out = {"AD": agg(AD_rows, ["T"]), "dilution": agg(dil_rows, ["T", "keep"]), "subsets": agg(sub_rows, ["T", "subset"]),
       "dipolar_models": agg(dip_rows, ["T"]), "zfs": agg(zfs_rows, ["T"])}
for k, v in out.items():
    v.to_csv(os.path.join(TAB, f"tabS_m3_revision_{k}.csv")); print("\n==", k); print(v.T.to_string())

# (6) AF share inherited from the DFT library: P(AF | r_NO_min bin, |J|>J_RES) reweighted to MD resolved pairs
lib = pd.read_csv(os.path.join(D, "train_features.csv"))[["name", "dmin_N_O", "dmin_O_O", "dmin_N_N"]]
J = pd.read_csv(os.path.join(D, "pairs_J.csv"))[["name", "J"]]
L = lib.merge(J, on="name"); L["rmin"] = L[["dmin_N_O", "dmin_O_O", "dmin_N_N"]].min(1)
L = L[L.J.abs() > J_RES]
bins = np.array([0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 9.5])
L["b"] = np.digitize(L.rmin, bins) - 1
pAF_bin = L.groupby("b").apply(lambda g: (g.J < 0).mean()); n_bin = L.groupby("b").size()
res_r = np.concatenate(md_res_rNO); hb = np.bincount(np.digitize(res_r, bins) - 1, minlength=len(bins) - 1)
w = hb / hb.sum(); exp_AF = sum(w[b] * pAF_bin.get(b, np.nan) for b in range(len(w)) if w[b] > 0)
inh = dict(library_AF_share_resolved=float((L.J < 0).mean()), expected_AF_share_reweighted=float(exp_AF),
           bins=bins.tolist(), lib_pAF_per_bin={int(k): float(v) for k, v in pAF_bin.items()},
           lib_n_per_bin={int(k): int(v) for k, v in n_bin.items()}, md_weight_per_bin=w.round(4).tolist())
json.dump(inh, open(os.path.join(TAB, "tabS_m3_revision_AFinherited.json"), "w"), indent=1); print(inh)
