#!/usr/bin/env python3
"""
m3_07_exchange_network.py
Pair- and spin-level statistics of the predicted exchange network.

Conventions: H = -2 J S1.S2 (J > 0 ferromagnetic), J in cm^-1;
1 cm^-1 = 29 979.2458 MHz = 1.438777 K.  Singlet-triplet gap = 2|J|.
J_RES = 0.10 cm^-1 (~3x the DFT resolution) separates resolved couplings
from couplings that are indistinguishable from zero at this level.

Per pair (last 5 ns):  <|J|>, <J> = <|J|(1-2 P_AF)>, |J|-weighted P_AF,
fraction of frames with |J| > J_RES, model interval (68 % conformal, applied
frame by frame and averaged, i.e. assuming fully correlated errors).
Per spin: number of resolved partners, sum |J|, mean-field Weiss constant
theta_i = 2 S(S+1) sum_j J_ij / (3 k_B) = 0.5 sum_j J_ij / k_B (resolved
couplings only, J_ij = <J>_ij), rms exchange frequency w_ex = sqrt(sum_j (2 J_ij)^2) vs the local dipolar
width sqrt(M2) (field along the surface normal).
Outputs: data/module3/exchange_pairs_T{T}_r{r}.csv, exchange_spins_*.csv,
         exchange_summary_by_replica.csv, exchange_summary.csv,
         exchange_matrices.npz (141 x 141, replica mean/sd per T)
"""
import os, numpy as np, pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from m3_ml_common import D, to_J
CM2MHZ, CM2K = 29979.2458, 1.438777
J_RES = 0.10
NU_DD = 52.04                       # MHz nm^3
ROOT = os.path.join(D, "..", "..")
labels = pd.read_csv(os.path.join(ROOT, "data", "module2", "module2_class_labels.csv"))
CN = {0: "S", 1: "M", 2: "R", 3: "F"}

def acf_time(x, dt=0.01):
    """integrated autocorrelation time (ns) up to the first zero crossing"""
    x = x - x.mean()
    if x.std() == 0: return np.nan
    n = len(x); f = np.fft.rfft(x, 2 * n); c = np.fft.irfft(f * np.conj(f))[:n]; c /= c[0]
    k = np.argmax(c < 0) if (c < 0).any() else n
    return dt * (0.5 + c[1:k].sum())

summ, mats = [], {}
for T in (300, 100):
    for r in (1, 2, 3):
        z = np.load(os.path.join(D, f"mdJ_T{T}_r{r}.npz")); kz = np.load(os.path.join(D, f"md_keyatoms_T{T}_r{r}.npz"))
        pairs, intact, mol = z["pairs"], z["intact"], z["mol_ids"]; nm = len(mol)
        Jabs, EJ, pAF, ev = z["Jabs"].astype(float), z["EJ"].astype(float), z["pAF"].astype(float), z["evaluated"]
        y, q68 = z["yhat"].astype(float), z["q68"].astype(float)
        lo = np.where(ev, to_J(np.maximum(y - q68, 0)), 0); hi = np.where(ev, to_J(y + q68), 0)
        K = kz["K"][-Jabs.shape[0]:].astype(float); spin = 0.5 * (K[:, :, 0] + K[:, :, 1])
        rs = np.linalg.norm(spin[:, pairs[:, 0]] - spin[:, pairs[:, 1]], axis=2)            # spin-spin distance (A)
        ok = intact[pairs[:, 0]] & intact[pairs[:, 1]]
        w = np.where(ev, Jabs, 0.0)
        pAFw = np.nansum(np.nan_to_num(pAF) * w, 0) / np.maximum(w.sum(0), 1e-30)
        P = pd.DataFrame(dict(i=pairs[:, 0], j=pairs[:, 1], mol_i=mol[pairs[:, 0]], mol_j=mol[pairs[:, 1]], intact=ok,
                              Jabs_mean=Jabs.mean(0), Jabs_lo=lo.mean(0), Jabs_hi=hi.mean(0), EJ_mean=EJ.mean(0),
                              EJ_sd_time=EJ.std(0), Jabs_sd_time=Jabs.std(0), Jabs_max=Jabs.max(0),
                              pAF_w=pAFw, frac_res=(Jabs > J_RES).mean(0), r_spin_mean=rs.mean(0), r_spin_min=rs.min(0),
                              rNO_min=np.nanmin(np.where(ev, z["rNO"], np.nan), 0)))
        # dynamics of persistently resolved pairs
        # modulation parameter of the singlet-triplet gap: xi = 2 pi sigma(2J) tau_J (sigma in Hz, tau in s)
        tau, cvj, xi = (np.full(len(P), np.nan) for _ in range(3))
        for k in np.where(P.frac_res.values > 0.5)[0]:
            tau[k] = acf_time(Jabs[:, k]); cvj[k] = Jabs[:, k].std() / Jabs[:, k].mean()
            xi[k] = 2 * np.pi * (2 * Jabs[:, k].std() * CM2MHZ * 1e6) * tau[k] * 1e-9
        P["tau_J_ns"], P["cv_J"], P["xi_J"] = tau, cvj, xi
        P.to_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv"), index=False)
        # ---------- matrices
        for name, col in (("EJ", "EJ_mean"), ("Jabs", "Jabs_mean"), ("pAF", "pAF_w"), ("fres", "frac_res")):
            Mx = np.zeros((nm, nm)) if name != "pAF" else np.full((nm, nm), np.nan)
            Mx[P.i, P.j] = P[col]; Mx[P.j, P.i] = P[col]
            mats.setdefault((T, name), []).append(Mx)
        # ---------- spins
        Pi = P[P.intact]
        res = Pi[Pi.Jabs_mean > J_RES]
        idx_int = np.where(intact)[0]
        deg = np.zeros(nm); sJ = np.zeros(nm); sEJ = np.zeros(nm); jmax = np.zeros(nm); wex2 = np.zeros(nm)
        nAF = np.zeros(nm); nF = np.zeros(nm)
        # sums run over resolved couplings only: below J_RES the prediction is set by the DFT floor
        for a, b, j1, e1, p1 in zip(res.i, res.j, res.Jabs_mean, res.EJ_mean, res.pAF_w):
            for s in (a, b):
                sJ[s] += j1; sEJ[s] += e1; jmax[s] = max(jmax[s], j1); wex2[s] += (2 * j1 * CM2MHZ) ** 2
                deg[s] += 1; nAF[s] += p1; nF[s] += 1 - p1
        # local dipolar width (secular, field along the surface normal x), all intact partners < 15 A, time-averaged M2
        S = spin[:, intact]; dv = S[:, :, None] - S[:, None]; rr = np.linalg.norm(dv, axis=3)
        np.einsum("fii->fi", rr)[:] = np.inf
        cth = dv[..., 0] / rr
        dd = NU_DD / (rr / 10) ** 3 * (1 - 3 * cth ** 2); dd[rr > 15] = 0
        sqM2 = np.sqrt((dd ** 2).sum(2).mean(0))
        Sdf = pd.DataFrame(dict(mol=mol[idx_int], degree=deg[idx_int], sumJ=sJ[idx_int], sumEJ=sEJ[idx_int], Jmax=jmax[idx_int],
                                nAF=nAF[idx_int], nF=nF[idx_int], w_ex_MHz=np.sqrt(wex2[idx_int]), sqrtM2_MHz=sqM2,
                                theta_K=0.5 * CM2K * sEJ[idx_int]))   # theta = 2 S(S+1) sum_j J_ij / (3 kB) = 0.5 sum_j J_ij / kB
        lab = labels[(labels["T"] == T) & (labels.rep == r)].set_index("mol").cls
        Sdf["cls"] = [CN[int(lab[m])] for m in Sdf.mol]
        Sdf.to_csv(os.path.join(D, f"exchange_spins_T{T}_r{r}.csv"), index=False)
        # ---------- network components (resolved couplings)
        g = csr_matrix((np.ones(len(res)), (res.i, res.j)), shape=(nm, nm))
        ncomp, lab_c = connected_components(g, directed=False)
        sizes = np.bincount(lab_c[intact])
        dmask = res.r_spin_mean.values
        summ.append(dict(T=T, rep=r, n_intact=int(intact.sum()), n_pairs_res=len(res),
            n_pairs_gt1=int((Pi.Jabs_mean > 1).sum()),
            degree_mean=Sdf.degree.mean(), frac_isolated=(Sdf.degree == 0).mean(), frac_deg_ge2=(Sdf.degree >= 2).mean(),
            largest_cluster_frac=sizes.max() / intact.sum(), n_clusters_ge2=int((sizes >= 2).sum()),
            Jres_median=res.Jabs_mean.median(), Jres_q25=res.Jabs_mean.quantile(0.25), Jres_q75=res.Jabs_mean.quantile(0.75),
            Jmax=Pi.Jabs_mean.max(), Jmax_frame=Pi.Jabs_max.max(),
            frac_AF_res=res.pAF_w.mean(), nAF_per_spin=Sdf.nAF.mean(), nF_per_spin=Sdf.nF.mean(),
            theta_mean_K=Sdf.theta_K.mean(), theta_min_K=Sdf.theta_K.min(), sumJ_mean=Sdf.sumJ.mean(),
            frac_ST_gt_dip=np.mean(2 * res.Jabs_mean.values * CM2MHZ > NU_DD / (dmask / 10) ** 3),
            frac_wex_gt_M2=np.mean(Sdf.w_ex_MHz > Sdf.sqrtM2_MHz), wex_over_M2_median_coupled=np.median((Sdf.w_ex_MHz / Sdf.sqrtM2_MHz)[Sdf.degree > 0]), sqrtM2_median=Sdf.sqrtM2_MHz.median(),
            wex_median_coupled=Sdf.w_ex_MHz[Sdf.degree > 0].median(),
            frac_spins_Jmax_gt_Q=np.mean(Sdf.Jmax * 2 > 35.0e3 / CM2MHZ),     # S-T gap above the Q-band quantum
            frac_res_time_mean=Pi.loc[Pi.Jabs_mean > J_RES, "frac_res"].mean(),
            tau_J_median_ns=np.nanmedian(P.tau_J_ns[P.intact]), cvJ_median=np.nanmedian(P.cv_J[P.intact]),
            n_persistent=int(np.isfinite(P.tau_J_ns[P.intact]).sum()), xi_median=np.nanmedian(P.xi_J[P.intact]),
            sdJ_MHz_median=np.nanmedian((2 * P.Jabs_sd_time * CM2MHZ)[np.isfinite(P.tau_J_ns) & P.intact])))
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[-1].items()}, flush=True)
# class-resolved spin statistics (replica mean +- sd)
C = []
for T in (100, 300):
    for r in (1, 2, 3):
        X = pd.read_csv(os.path.join(D, f"exchange_spins_T{T}_r{r}.csv"))
        for c, g in X.groupby("cls"):
            C.append(dict(T=T, rep=r, cls=c, n=len(g), degree=g.degree.mean(), frac_isolated=(g.degree == 0).mean(),
                          sumJ=g.sumJ.mean(), theta_K=g.theta_K.mean()))
C = pd.DataFrame(C); C.to_csv(os.path.join(D, "exchange_by_class_by_replica.csv"), index=False)
C.drop(columns="rep").groupby(["T", "cls"]).agg(["mean", "std"]).to_csv(os.path.join(D, "exchange_by_class.csv"))
S = pd.DataFrame(summ); S.to_csv(os.path.join(D, "exchange_summary_by_replica.csv"), index=False)
agg = S.drop(columns="rep").groupby("T").agg(["mean", "std"]); agg.to_csv(os.path.join(D, "exchange_summary.csv"))
out = {}
for (T, name), L in mats.items():
    A = np.array(L); out[f"{name}_T{T}_mean"] = np.nanmean(A, 0); out[f"{name}_T{T}_sd"] = np.nanstd(A, 0)
out["intact"] = intact; out["mol_ids"] = mol
np.savez_compressed(os.path.join(D, "exchange_matrices.npz"), **out)
print(agg.T.round(3).to_string())
