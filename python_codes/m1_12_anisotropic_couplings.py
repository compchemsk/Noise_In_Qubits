#!/usr/bin/env python3
"""
m1_12_anisotropic_couplings.py
Anisotropic spin-phonon coupling of TEMPO and PTEMPO from the full g and 14N
hyperfine tensors printed by ORCA 5.0.1 (B2PLYP/EPR-II, SCF spin density, GIAO)
at every displaced geometry (parsed by m1_11_parse_orca_tensors.py).

All tensors refer to the fixed input frame of the displaced geometries, so the
first-order coupling tensors dG/dQ_k are obtained component by component from
quadratic fits X_ab(Q) = c0 + c1 Q + c2 Q^2 over the 11-point grid (Q = -1..1).

Checks: (i) trace/3 of the fitted tensors against the isotropic couplings used
in the main text; (ii) mass-weighted displacement at |Q|=1 against hbar/omega
(dimensionless normal coordinates); (iii) residuals of the component fits.

Quantities (S = 1/2, I = 1 for 14N; powder average over 2000 field directions):
  Zeeman, H = mu_B B n^T g S.  h(n) = g^T n, u = h/|h|, g_eff = |h|.
     secular   : d nu = nu * (u . dg^T n) / g_eff                (line shift)
     nonsecular: |V_Z|^2 = (nu/g_eff)^2 |dg^T n - (u.dg^T n) u|^2 / 4   (spin-flip)
  Hyperfine, H = S^T A I.
     nonsecular: |V_A|^2 = [I(I+1)/12] [Tr(dA dA^T) - u^T dA dA^T u]  (nuclear-state averaged)
     secular   : rms over m_I of m_I (A^T u . dA^T u)/|A^T u|
  Decomposition of dX: isotropic (Tr/3), symmetric traceless (anisotropic), antisymmetric.
The squared nonsecular matrix elements are used as fixed relative weights w_k in
  R = a T + C sum_k w_k n_k (n_k + 1)
for the T1 data of Tesi et al. (same data and fitting as m1_03_relaxation_models.py).
"""
import os, json, numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); D1 = os.path.join(HERE, "..", "data", "module1")
TAB = os.path.join(HERE, "..", "tables")
P = json.load(open(os.path.join(D1, "orca_tensors", "parsed_tensors.json")))
freqs = pd.read_csv(os.path.join(D1, "module1_frequencies.csv"))
summ = pd.read_csv(os.path.join(D1, "module1_coupling_summary.csv"))
NU_Q = 35000.0                 # MHz, Q band (as in the main text)
II = 1.0
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
HB_OVER_OMEGA = 33.7168        # hbar/(2 pi c) in amu A^2 cm^-1

def fib(n=2000):
    i = np.arange(n) + 0.5; phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
NV = fib()

def qfit(Q, Y):
    X = np.c_[np.ones_like(Q), Q, Q ** 2]
    c, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ c; dof = len(Q) - 3
    cov = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.outer(np.diag(cov), (res ** 2).sum(0) / dof))
    return c, se, res

def decomp(M):
    iso = np.trace(M) / 3; S = 0.5 * (M + M.T); an = S - iso * np.eye(3); asy = 0.5 * (M - M.T)
    return iso, an, asy

rows, comp_rows, check_rows = [], [], []
for mol in ("TEMPO", "PTEMPO"):
    R = [r for r in P[mol]]
    fr = freqs[freqs.molecule == mol].set_index("mode").freq_cm1
    X0 = np.array([r for r in R if r["Q"] == 0][0]["xyz"])
    m = np.array([MASS[e] for e in R[0]["el"]])
    eq = [b for b in [r for r in R if r["Q"] == 0][0]["blocks"] if b["density"] == "SCF"][0]
    g0 = np.array(eq["gmat"]); A0 = np.array(eq["Araw"])
    H0 = NV @ g0; geff = np.linalg.norm(H0, axis=1); U = H0 / geff[:, None]
    AU = U @ A0; aeff = np.linalg.norm(AU, axis=1)
    for k in sorted(set(r["mode"] for r in R)):
        seen = {}
        for r in R:
            if r["mode"] == k: seen.setdefault(round(r["Q"], 3), r)     # Q = 0 appears twice (pos/neg) - identical
        Qs = np.array(sorted(seen)); rr = [seen[q] for q in Qs]
        B = [[b for b in r["blocks"] if b["density"] == "SCF"][0] for r in rr]
        G = np.array([np.array(b["gmat"]).ravel() for b in B]); A = np.array([np.array(b["Araw"]).ravel() for b in B])
        giso = np.array([b["g_iso"] for b in B]); aiso = np.array([b["A_iso"] for b in B])
        cg, sg, resg = qfit(Qs, G); cA, sA, resA = qfit(Qs, A)
        dG = cg[1].reshape(3, 3); dA = cA[1].reshape(3, 3)
        dGse = sg[1].reshape(3, 3); dAse = sA[1].reshape(3, 3)
        # checks
        ci, *_ = qfit(Qs, np.c_[giso, aiso])
        s = summ[(summ.molecule == mol) & (summ["mode"] == k)].iloc[0]
        q1 = [r for r in rr if abs(r["Q"]) == 1.0]
        mw = np.mean([(m[:, None] * (np.array(r["xyz"]) - X0) ** 2).sum() for r in q1])
        check_rows.append(dict(molecule=mol, mode=k, freq=fr[k], n_Q=len(Qs),
                               dg_iso_fit=ci[1][0], dg_iso_paper=s.dg_dQ, dA_iso_fit=ci[1][1], dA_iso_paper=s.dA_dQ_MHz,
                               trace3_dG=np.trace(dG) / 3, trace3_dA=np.trace(dA) / 3,
                               mw_disp=mw, hbar_over_omega=HB_OVER_OMEGA / fr[k],
                               rms_res_g=np.sqrt((resg ** 2).mean()), rms_res_A=np.sqrt((resA ** 2).mean()),
                               max_res_g=np.abs(resg).max(), max_res_A=np.abs(resA).max()))
        gi, gan, gas = decomp(dG); ai, aan, aas = decomp(dA)
        # powder-averaged secular / nonsecular matrix elements
        dh = NV @ dG                                   # (n,3) = (dg^T n) as row vectors: n^T dG
        par = (U * dh).sum(1)
        perp2 = (dh ** 2).sum(1) - par ** 2
        VZ2 = (NU_Q / geff) ** 2 * perp2 / 4           # MHz^2
        secZ = NU_Q * par / geff                       # MHz
        MM = dA @ dA.T
        VA2 = II * (II + 1) / 12 * (np.trace(MM) - np.einsum("ni,ij,nj->n", U, MM, U))
        dAU = U @ dA
        secA = np.sqrt(2 / 3) * (AU * dAU).sum(1) / aeff
        # anisotropic-only parts (isotropic part removed)
        dGa = dG - gi * np.eye(3); dAa = dA - ai * np.eye(3)
        dha = NV @ dGa; para = (U * dha).sum(1)
        VZ2a = (NU_Q / geff) ** 2 * ((dha ** 2).sum(1) - para ** 2) / 4
        MMa = dAa @ dAa.T
        VA2a = II * (II + 1) / 12 * (np.trace(MMa) - np.einsum("ni,ij,nj->n", U, MMa, U))
        # second-order (quadratic) coupling tensors c2 = (1/2) d2X/dQ2: nonsecular parts, same averaging
        q2G = cg[2].reshape(3, 3); q2A = cA[2].reshape(3, 3)
        dh2 = NV @ q2G; par2 = (U * dh2).sum(1)
        VZ2q = ((NU_Q / geff) ** 2 * ((dh2 ** 2).sum(1) - par2 ** 2) / 4).mean()
        MM2 = q2A @ q2A.T
        VA2q = (II * (II + 1) / 12 * (np.trace(MM2) - np.einsum("ni,ij,nj->n", U, MM2, U))).mean()
        rows.append(dict(molecule=mol, mode=k, freq_cm1=fr[k], VZ2_quad=VZ2q, VA2_quad=VA2q,
                         c2_g_iso=ci[2][0], c2_A_iso=ci[2][1],
                         dg_iso=gi, dg_aniso_norm=np.linalg.norm(gan), dg_antisym_norm=np.linalg.norm(gas),
                         dg_aniso_MHz=NU_Q / 2.0062 * np.linalg.norm(gan),
                         dA_iso_MHz=ai, dA_aniso_norm_MHz=np.linalg.norm(gan * 0 + aan), dA_antisym_MHz=np.linalg.norm(aas),
                         dg_comp_se_max=dGse.max(), dA_comp_se_max=dAse.max(),
                         secZ_rms_MHz=np.sqrt((secZ ** 2).mean()), secA_rms_MHz=np.sqrt((secA ** 2).mean()),
                         VZ2_MHz2=VZ2.mean(), VA2_MHz2=VA2.mean(), VZ2_aniso=VZ2a.mean(), VA2_aniso=VA2a.mean()))
        for a in range(3):
            for b in range(3):
                comp_rows.append(dict(molecule=mol, mode=k, comp=f"{'xyz'[a]}{'xyz'[b]}", dg=dG[a, b], dg_se=dGse[a, b],
                                      dA_MHz=dA[a, b], dA_se=dAse[a, b]))
T = pd.DataFrame(rows); C = pd.DataFrame(check_rows)
T["V2_tot"] = T.VZ2_MHz2 + T.VA2_MHz2
os.makedirs(TAB, exist_ok=True)
T.to_csv(os.path.join(D1, "module1_anisotropic_couplings.csv"), index=False)
C.to_csv(os.path.join(D1, "module1_anisotropic_checks.csv"), index=False)
pd.DataFrame(comp_rows).to_csv(os.path.join(D1, "module1_tensor_derivatives_components.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(C.round(7).to_string()); print(T.round(4).to_string())

# ---------------- T1 fits with nonsecular weights (PTEMPO modes) ------------------
kB = 0.6950348
Tt = np.array([7, 10, 20, 30, 40, 50, 60], float)
exp = {"PTEMPO_slow": (np.array([87, 57, 25.7, 13.0, 4.2, 2.00, 1.00]), np.array([2, 1, 0.6, 0.4, 0.1, 0.07, 0.04])),
       "PTEMPO_fast": (np.array([21, 14.4, 6.2, 3.5, 1.2, 0.69, 0.37]), np.array([1, 0.6, 0.4, 0.3, 0.1, 0.06, 0.04])),
       "SAM": (np.array([np.nan, 9.2, 3.6, 1.5, 0.5, 0.3, np.nan]), np.array([np.nan, 0.6, 0.5, 0.2, 0.1, 0.1, np.nan]))}
pt = T[T.molecule == "PTEMPO"].sort_values("mode"); nu = pt.freq_cm1.to_numpy()
W = {"nonsec_Z": pt.VZ2_MHz2.to_numpy(), "nonsec_A": pt.VA2_MHz2.to_numpy(), "nonsec_ZA": pt.V2_tot.to_numpy(),
     "nonsec_ZA_aniso": (pt.VZ2_aniso + pt.VA2_aniso).to_numpy(),
     "quad_nonsec_ZA": (pt.VZ2_quad + pt.VA2_quad).to_numpy(),
     "quad_iso_ZA": ((NU_Q / 2.0062 * pt.c2_g_iso) ** 2 + pt.c2_A_iso ** 2).to_numpy(),
     "first_order_ZA_squared": (pt.V2_tot ** 2).to_numpy()}
def nn1(E, T): x = E / (kB * T); return np.exp(x) / np.expm1(x) ** 2
fits = []
for ds, (t1, e) in exp.items():
    ok = ~np.isnan(t1); Tx = Tt[ok]; Rr = 1e3 / t1[ok]; sR = Rr * e[ok] / t1[ok]
    for name, w in W.items():
        w = w / w.sum()
        f = lambda p: (p[0] * Tx + p[1] * sum(wk * nn1(Ek, Tx) for wk, Ek in zip(w, nu)) - Rr) / sR
        best = min((least_squares(f, [1, s], bounds=([0, 0], [np.inf, np.inf]), x_scale="jac") for s in (1, 1e2, 1e4, 1e6)),
                   key=lambda r: r.cost)
        n = len(Tx); k = 2; chi2 = 2 * best.cost
        fits.append(dict(dataset=ds, weights=name, chi2=chi2, chi2_red=chi2 / (n - k),
                         AICc=chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1),
                         weight_share_116_159=w[(nu > 110) & (nu < 160)].sum(), weight_share_le87=w[nu <= 87].sum()))
F = pd.DataFrame(fits); F.to_csv(os.path.join(D1, "module1_T1_nonsecular_fits.csv"), index=False)
print(F.round(3).to_string())
