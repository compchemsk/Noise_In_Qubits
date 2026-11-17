#!/usr/bin/env python3
"""
m1_13_multiprocess_T1.py
Spin-lattice relaxation of PTEMPO (frozen solution, slow and fast components) and of the TEMPO monolayer
(Tesi et al., Adv. Mater. 2023, Table 1) described by sums of relaxation processes.

Processes (rates in s^-1, T in K, energies in cm^-1, k_B = 0.6950348 cm^-1 K^-1)
  direct      R_D = a T
  Raman       R_R = b T^m                       (power law, m free)
  local mode  R_L = C n(Delta) [n(Delta) + 1],  n(Delta) = 1 / [exp(Delta / k_B T) - 1]
Models: every sum of these terms with up to three terms (D+L, D+R, R+L, D+R+L, D+L+L, L, R, D+L with Delta
fixed at the far-infrared modes 62.5 and 101.5 cm^-1 or at the 50 cm^-1 of the original analysis).
Fit: weighted least squares on R = 1/T1 with sigma_R = R sigma_T1 / T1 (main), and, as a check of the
weighting, unweighted least squares on ln R. Many starting points (grid over Delta and m) are used and the
best solution is kept. Comparison: chi^2, reduced chi^2, AICc = chi^2 + 2k + 2k(k+1)/(N-k-1) and
BIC = chi^2 + k ln N. Global fit: slow and fast components together with shared Delta (and m), separate
amplitudes. For each model the fractional contribution of every process to R at each temperature is stored.
"""
import os, json, itertools, numpy as np, pandas as pd
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); D1 = os.path.join(HERE, "..", "data", "module1")
KB = 0.6950348
T = np.array([7, 10, 20, 30, 40, 50, 60], float)
DATA = {"PTEMPO_slow": (np.array([87, 57, 25.7, 13.0, 4.2, 2.00, 1.00]), np.array([2, 1, 0.6, 0.4, 0.1, 0.07, 0.04])),
        "PTEMPO_fast": (np.array([21, 14.4, 6.2, 3.5, 1.2, 0.69, 0.37]), np.array([1, 0.6, 0.4, 0.3, 0.1, 0.06, 0.04])),
        "SAM": (np.array([np.nan, 9.2, 3.6, 1.5, 0.5, 0.3, np.nan]), np.array([np.nan, 0.6, 0.5, 0.2, 0.1, 0.1, np.nan]))}

def nbe(D, T): return 1.0 / np.expm1(D / (KB * T))
def term(kind, p, T):
    if kind == "D": return p[0] * T
    if kind == "R": return p[0] * T ** p[1]
    if kind == "L": n = nbe(p[1], T); return p[0] * n * (n + 1)
NPAR = {"D": 1, "R": 2, "L": 2}

# model = list of (kind, fixed-dict)
MODELS = {"D+L": [("D", {}), ("L", {})], "D+R": [("D", {}), ("R", {})], "R+L": [("R", {}), ("L", {})],
          "D+R+L": [("D", {}), ("R", {}), ("L", {})], "D+L+L": [("D", {}), ("L", {}), ("L", {})],
          "L": [("L", {})], "R": [("R", {})],
          "D+L(50)": [("D", {}), ("L", {"D": 50.0})], "D+L(62.5)": [("D", {}), ("L", {"D": 62.5})],
          "D+L(101.5)": [("D", {}), ("L", {"D": 101.5})], "D+L(62.5)+L(101.5)": [("D", {}), ("L", {"D": 62.5}), ("L", {"D": 101.5})]}

def unpack(model, x):
    """free-parameter vector -> list of per-term parameter lists (amplitudes are exp-transformed)"""
    out, i = [], 0
    for kind, fx in model:
        if kind == "D": out.append([np.exp(x[i])]); i += 1
        elif kind == "R": out.append([np.exp(x[i]), x[i + 1]]); i += 2
        else:
            if "D" in fx: out.append([np.exp(x[i]), fx["D"]]); i += 1
            else: out.append([np.exp(x[i]), x[i + 1]]); i += 2
    return out

def nfree(model): return sum(1 if (k == "D" or (k == "L" and "D" in f)) else 2 for k, f in model)

def bounds(model):
    lo, hi = [], []
    for kind, fx in model:
        lo.append(-60); hi.append(60)
        if kind == "R": lo.append(0.5); hi.append(12.0)
        if kind == "L" and "D" not in fx: lo.append(5.0); hi.append(600.0)
    return np.array(lo), np.array(hi)

def starts(model, Tx, Rr):
    grids = []
    for kind, fx in model:
        if kind == "D": grids.append([[np.log(max(Rr[0] / Tx[0], 1e-6))]])
        elif kind == "R": grids.append([[np.log(1e-3), m] for m in (2, 3, 5, 7, 9)])
        else:
            if "D" in fx: grids.append([[np.log(Rr[-1])]])
            else: grids.append([[np.log(Rr[-1]), d] for d in (20, 50, 80, 120, 160, 250, 400)])
    for combo in itertools.product(*grids):
        yield np.concatenate([np.array(c, float) for c in combo])

def fit(model, Tx, Rr, sR, mode="weighted"):
    lo, hi = bounds(model)
    def res(x):
        pred = sum(term(k, p, Tx) for (k, _), p in zip(model, unpack(model, x)))
        return (pred - Rr) / sR if mode == "weighted" else np.log(np.maximum(pred, 1e-300)) - np.log(Rr)
    best = None
    for x0 in starts(model, Tx, Rr):
        for amp_shift in (0.0, -3.0, 3.0):
            x = x0.copy()
            idx = 0
            for kind, fx in model:                       # shift amplitudes to explore scales
                x[idx] += amp_shift; idx += 1 if (kind == "D" or (kind == "L" and "D" in fx)) else 2
            x = np.clip(x, lo + 1e-9, hi - 1e-9)
            try:
                r = least_squares(res, x, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
            except Exception:
                continue
            if best is None or r.cost < best.cost: best = r
    chi2 = 2 * best.cost; n = len(Tx); k = nfree(model)
    J = best.jac; cov = None
    try: cov = np.linalg.pinv(J.T @ J) * (chi2 / max(n - k, 1) if mode == "log" else 1.0)
    except Exception: pass
    pars = unpack(model, best.x)
    contrib = np.array([term(kk, p, Tx) for (kk, _), p in zip(model, pars)])
    return dict(chi2=chi2, k=k, n=n, red=chi2 / (n - k) if n > k else np.nan,
                AICc=chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan,
                BIC=chi2 + k * np.log(n), pars=[[float(v) for v in p] for p in pars],
                se=(np.sqrt(np.diag(cov)).tolist() if cov is not None else None), x=best.x.tolist(),
                frac=(contrib / contrib.sum(0)).round(4).tolist())

if __name__ == "__main__":
    rows, store = [], {}
    for ds, (t1, e) in DATA.items():
        ok = ~np.isnan(t1); Tx = T[ok]; Rr = 1e3 / t1[ok]; sR = Rr * e[ok] / t1[ok]
        for mode in ("weighted", "log"):
            for name, model in MODELS.items():
                if nfree(model) >= len(Tx): continue
                f = fit(model, Tx, Rr, sR, mode); store[(ds, mode, name)] = f
                Ds = [p[1] for (kk, _), p in zip(model, f["pars"]) if kk == "L"]
                ms = [p[1] for (kk, _), p in zip(model, f["pars"]) if kk == "R"]
                rows.append(dict(dataset=ds, fit=mode, model=name, k=f["k"], n=f["n"], chi2=f["chi2"], chi2_red=f["red"],
                                 AICc=f["AICc"], BIC=f["BIC"], Delta=";".join(f"{d:.1f}" for d in Ds), m=";".join(f"{v:.2f}" for v in ms),
                                 frac_at_T=json.dumps(dict(zip([int(v) for v in Tx], [list(c) for c in zip(*f["frac"])])))))
    df = pd.DataFrame(rows)
    for (ds, mode), g in df.groupby(["dataset", "fit"]):
        df.loc[g.index, "dAICc"] = g.AICc - g.AICc.min(); df.loc[g.index, "dBIC"] = g.BIC - g.BIC.min()
    df.to_csv(os.path.join(D1, "module1_T1_multiprocess.csv"), index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 40)
    print(df.drop(columns="frac_at_T").round(2).to_string())
    # profile of Delta for D+L (weighted) and global fit of slow+fast with shared Delta / shared (m, Delta)
    prof = {}
    for ds in ("PTEMPO_slow", "PTEMPO_fast", "SAM"):
        t1, e = DATA[ds]; ok = ~np.isnan(t1); Tx = T[ok]; Rr = 1e3 / t1[ok]; sR = Rr * e[ok] / t1[ok]
        grid = np.arange(20, 301, 2.0); c = []
        for d in grid: c.append(fit([("D", {}), ("L", {"D": float(d)})], Tx, Rr, sR)["chi2"])
        prof[ds] = dict(Delta=grid.tolist(), chi2=c)
    json.dump(prof, open(os.path.join(D1, "module1_T1_Delta_profile.json"), "w"))
    for ds, p in prof.items():
        c = np.array(p["chi2"]); g = np.array(p["Delta"]); i = c.argmin(); ok1 = g[c <= c[i] + 1]
        print(ds, "profile min Delta", g[i], "chi2", round(c[i], 2), "Delta range (dchi2<=1):", ok1.min(), "-", ok1.max())
    # global fit slow + fast
    ts, es = DATA["PTEMPO_slow"]; tf, ef = DATA["PTEMPO_fast"]
    TT = np.concatenate([T, T]); RR = np.concatenate([1e3 / ts, 1e3 / tf]); SS = np.concatenate([1e3 / ts * es / ts, 1e3 / tf * ef / tf])
    sel = np.r_[np.zeros(7, bool), np.ones(7, bool)]
    glob = {}
    for name, kinds in (("D+L shared Delta", "DL"), ("D+R shared m", "DR"), ("D+R+L shared m, Delta", "DRL")):
        def res(x):
            a1, a2 = np.exp(x[0]), np.exp(x[1]); out = a1 * TT * ~sel + a2 * TT * sel; j = 2
            if "R" in kinds:
                b1, b2, m = np.exp(x[j]), np.exp(x[j + 1]), x[j + 2]; out = out + (b1 * ~sel + b2 * sel) * TT ** m; j += 3
            if "L" in kinds:
                c1, c2, d = np.exp(x[j]), np.exp(x[j + 1]), x[j + 2]; n = nbe(d, TT); out = out + (c1 * ~sel + c2 * sel) * n * (n + 1)
            return (out - RR) / SS
        best = None
        for d0 in (30, 60, 100, 140, 200):
            for m0 in (2, 4, 7):
                x0 = [np.log(1.0), np.log(3.0)]; lo = [-60, -60]; hi = [60, 60]
                if "R" in kinds: x0 += [np.log(1e-3), np.log(3e-3), m0]; lo += [-60, -60, 0.5]; hi += [60, 60, 12]
                if "L" in kinds: x0 += [np.log(RR[6]), np.log(RR[13]), d0]; lo += [-60, -60, 5]; hi += [60, 60, 600]
                r = least_squares(res, x0, bounds=(lo, hi), x_scale="jac", max_nfev=20000)
                if best is None or r.cost < best.cost: best = r
        k = len(best.x); chi2 = 2 * best.cost; n = 14
        glob[name] = dict(chi2=chi2, k=k, AICc=chi2 + 2 * k + 2 * k * (k + 1) / (n - k - 1), BIC=chi2 + k * np.log(n), x=best.x.tolist())
        print("GLOBAL", name, "chi2 %.1f k %d AICc %.1f BIC %.1f" % (chi2, k, glob[name]["AICc"], glob[name]["BIC"]),
              "shared:", [round(v, 2) for v in best.x[-1:]] if "L" in kinds else [round(best.x[-1], 2)])
    json.dump(glob, open(os.path.join(D1, "module1_T1_global.json"), "w"), indent=1)
