#!/usr/bin/env python3
"""
m3_06_apply_md.py
Applies the final exchange model to every radical pair in the last 5 ns
(500 frames, 10 ps apart) of each of the six production trajectories.

* All 141C2 = 9870 pairs are considered.  Pairs whose closest N/O...N/O
  distance exceeds RCUT = 9 A in a frame are assigned J = 0 for that frame
  (training pairs beyond 8 A have |J| at the DFT resolution, 0.03 cm^-1).
* The 14 defective radicals are carried along but flagged; every statistic
  in the paper uses intact-intact pairs only.
* Stored per run (data/module3/mdJ_T{T}_r{r}.npz):
    pairs (P x 2, molecule indices), Jabs (frames x P), pAF, EJ, yhat,
    q68, q90 (conformal half-widths in y), ad (applicability-domain distance),
    committee_sd (std of four regressors, every 10th frame), rNO (frames x P)
  where P = pairs that come within RCUT at least once.
"""
import os, sys, joblib, numpy as np
from m3_features import features
from m3_ml_common import prepare, to_J, D
import pandas as pd
RCUT = 9.0
LAST = 500                                        # last 5 ns
M = joblib.load(os.path.join(D, "final_model.joblib"))
def predict(A, B, HA, HB, committee=False):
    f = prepare(pd.DataFrame(features(A, B, HA, HB)))
    X = f[M["cols"]].values
    y = M["et"].predict(X)
    p = M["sign"].predict_proba(np.c_[y, f.r_NO_min.values])[:, 1]
    k = np.clip(np.searchsorted(M["conformal_edges"], y, side="right") - 1, 0, 3)
    ad = M["nn"].kneighbors(M["scaler"].transform(X))[0].mean(1)
    out = dict(y=y, pAF=p, q68=M["conformal_q"][k, 0], q90=M["conformal_q"][k, 1], ad=ad, rNO=f.r_NO_min.values)
    if committee:
        out["csd"] = np.vstack([y] + [m.predict(X) for m in M["committee"].values()]).std(0)
    return out

runs = [(T, r) for T in (300, 100) for r in (1, 2, 3)]
for T, r in runs:
    d = np.load(os.path.join(D, f"md_keyatoms_T{T}_r{r}.npz"))
    K, H = d["K"][-LAST:].astype(float), d["H"][-LAST:].astype(float)
    nf, nm = K.shape[:2]
    NO = K[:, :, :2]
    Dmin = np.linalg.norm(NO[:, :, None, :, None] - NO[:, None, :, None, :], axis=-1).min((3, 4))  # (f, m, m)
    iu = np.triu_indices(nm, 1)
    close = Dmin[:, iu[0], iu[1]] < RCUT                       # (f, 9870)
    keep = close.any(0); pairs = np.c_[iu[0][keep], iu[1][keep]]; P = len(pairs)
    fi, pj = np.nonzero(close[:, keep])                        # frame, pair-index of every evaluation
    a, b = pairs[pj, 0], pairs[pj, 1]
    out = predict(K[fi, a], K[fi, b], H[fi, a], H[fi, b])
    sub = (fi % 10) == 0
    csd = predict(K[fi[sub], a[sub]], K[fi[sub], b[sub]], H[fi[sub], a[sub]], H[fi[sub], b[sub]], committee=True)["csd"]
    store = {k: np.zeros((nf, P), np.float32) for k in ("Jabs", "pAF", "EJ", "yhat", "q68", "q90", "ad")}
    store["rNO"] = np.full((nf, P), np.nan, np.float32); store["csd"] = np.full((nf, P), np.nan, np.float32)
    Jabs = to_J(out["y"])
    store["Jabs"][fi, pj] = Jabs; store["pAF"][fi, pj] = out["pAF"]; store["EJ"][fi, pj] = Jabs * (1 - 2 * out["pAF"])
    store["yhat"][fi, pj] = out["y"]; store["q68"][fi, pj] = out["q68"]; store["q90"][fi, pj] = out["q90"]
    store["ad"][fi, pj] = out["ad"]; store["rNO"][fi, pj] = out["rNO"]; store["csd"][fi[sub], pj[sub]] = csd
    store["pAF"][store["Jabs"] == 0] = np.nan                  # no sign for pairs outside the cutoff
    store["evaluated"] = close[:, keep]
    np.savez_compressed(os.path.join(D, f"mdJ_T{T}_r{r}.npz"), pairs=pairs, mol_ids=d["mol_ids"], intact=d["intact"], **store)
    inside = np.mean(out["ad"] <= M["ad_threshold"])
    print(f"T{T} r{r}: pairs within {RCUT} A at least once {P}; evaluations {len(fi)}; "
          f"AD inside {inside:.3f}; max|J| {Jabs.max():.2f} cm-1; median committee sd (y) {np.median(csd):.3f}", flush=True)
