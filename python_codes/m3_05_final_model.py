#!/usr/bin/env python3
"""
m3_05_final_model.py
Final exchange model = magnitude regressor x sign probability.

  |J|  : ExtraTrees on the 61 descriptors, target y = asinh(|J|/J0),
         hyper-parameters from 5-fold CV on the training partition.
         Uncertainty: Mondrian split-conformal 68 % and 90 % intervals in y,
         calibrated on out-of-fold residuals in four bins of the prediction.
  sign : logistic regression for P(AF) on [predicted y, r_NO_min], fitted to
         training pairs with |J| > 0.10 cm^-1 using out-of-fold magnitudes.
  E[J] = |J| (1 - 2 P_AF)    (H = -2 J S1.S2; J > 0 ferromagnetic)

Everything is evaluated once on the 15 % hold-out set, together with the
committee (RF, GPR, KRR) and two physics baselines.
Outputs: data/module3/final_model.joblib, test_predictions.csv,
         final_test_metrics.json, learning_curves.csv, perm_importance.csv
"""
import os, json, warnings, joblib, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score, brier_score_loss, r2_score
from sklearn.inspection import permutation_importance
from m3_ml_common import *

F, SETS = load_training(); cols = SETS["all"]
sp = json.load(open(os.path.join(D, "split.json"))); tr, te = np.array(sp["train"]), np.array(sp["test"])
Xtr, Xte = F.iloc[tr][cols].values, F.iloc[te][cols].values
Jtr, Jte = F.J.values[tr], F.J.values[te]
ytr, yte = to_y(np.abs(Jtr)), to_y(np.abs(Jte))
rtr, rte = F.r_NO_min.values[tr], F.r_NO_min.values[te]
cv5 = KFold(5, shuffle=True, random_state=SEED)

# ---------------- magnitude model
gs = GridSearchCV(ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=2),
                  {"min_samples_leaf": [1, 2, 3, 5], "max_features": [0.33, 0.6, 1.0]}, cv=cv5,
                  scoring="neg_mean_squared_error").fit(Xtr, ytr)
et = gs.best_estimator_; print("ET params", gs.best_params_)
oof = cross_val_predict(ExtraTreesRegressor(n_estimators=500, random_state=SEED, n_jobs=2, **gs.best_params_), Xtr, ytr, cv=cv5)
# Mondrian conformal: bins of the prediction
edges = np.quantile(oof, [0, 0.25, 0.5, 0.75, 1.0]); edges[0], edges[-1] = -np.inf, np.inf
res = np.abs(ytr - oof); qtab = []
for k in range(4):
    mk = (oof >= edges[k]) & (oof < edges[k + 1]); n = mk.sum()
    qtab.append([np.quantile(res[mk], min(1, np.ceil((n + 1) * q) / n)) for q in (0.68, 0.90)])
qtab = np.array(qtab)
def conformal(yhat):
    k = np.clip(np.searchsorted(edges, yhat, side="right") - 1, 0, 3); return qtab[k, 0], qtab[k, 1]
yhat = et.predict(Xte); q68, q90 = conformal(yhat)

committee = {
 "RandomForest": RandomForestRegressor(500, max_features=0.33, random_state=SEED, n_jobs=2),
 "GPR-Matern": make_pipeline(StandardScaler(), GaussianProcessRegressor(C(1.0) * Matern(3.0, nu=2.5) + WhiteKernel(0.1), normalize_y=True, random_state=SEED)),
 "KRR-Laplace": make_pipeline(StandardScaler(), KernelRidge(kernel="laplacian", alpha=0.1, gamma=0.003)),
}
cpred = {"ExtraTrees": yhat}
for k, mdl in committee.items():
    cpred[k] = mdl.fit(Xtr, ytr).predict(Xte)
# physics baselines (log10|J| scale, fitted on training pairs above resolution, then mapped to y)
def base_fit(cols_b):
    mk = np.abs(Jtr) > 0.05
    lr = LinearRegression().fit(F.iloc[tr][cols_b].values[mk], np.log10(np.abs(Jtr[mk])))
    return to_y(10 ** lr.predict(F.iloc[te][cols_b].values)), lr
cpred["Baseline: exp(-r_NO)"], _ = base_fit(["r_NO_min"])
cpred["Baseline: S^2 + O...H"], _ = base_fit(["r_NO_min", "logS2_somo", "dmin_O_H"])

big = np.abs(Jte) > 0.10
mag_metrics = {}
for k, p in cpred.items():
    mag_metrics[k] = dict(R2_y=r2_score(yte, p), RMSE_y=float(np.sqrt(np.mean((p - yte) ** 2))),
                          within3x=float(np.mean(np.abs(np.log10(to_J(p[big])) - np.log10(np.abs(Jte[big]))) < np.log10(3))),
                          within10x=float(np.mean(np.abs(np.log10(to_J(p[big])) - np.log10(np.abs(Jte[big]))) < 1)),
                          MAE_absJ=float(np.mean(np.abs(to_J(p) - np.abs(Jte)))))
    print(f"{k:24s} R2_y {mag_metrics[k]['R2_y']:.3f}  RMSE_y {mag_metrics[k]['RMSE_y']:.3f}  within3x {mag_metrics[k]['within3x']:.3f}  within10x {mag_metrics[k]['within10x']:.3f}")
cstack = np.vstack([cpred[k] for k in ("ExtraTrees", "RandomForest", "GPR-Matern", "KRR-Laplace")])
cov68, cov90 = np.mean(np.abs(yte - yhat) <= q68), np.mean(np.abs(yte - yhat) <= q90)
print("conformal coverage on test: 68%% -> %.3f, 90%% -> %.3f" % (cov68, cov90))

# ---------------- sign model
mtr = np.abs(Jtr) > 0.10
Ztr = np.c_[oof, rtr]; Zte = np.c_[yhat, rte]
sign = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(Ztr[mtr], (Jtr[mtr] < 0).astype(int))
pAF = sign.predict_proba(Zte)[:, 1]
base = (Jtr[mtr] < 0).mean()
yb = (Jte[big] < 0).astype(int)
sign_metrics = dict(n_test=int(big.sum()), frac_AF_test=float(yb.mean()), base_rate_train=float(base),
                    AUC=float(roc_auc_score(yb, pAF[big])), Brier=float(brier_score_loss(yb, pAF[big])),
                    Brier_base=float(brier_score_loss(yb, np.full(big.sum(), base))),
                    mean_pAF_test=float(pAF[big].mean()))
sign_metrics["BSS"] = 1 - sign_metrics["Brier"] / sign_metrics["Brier_base"]
rng = np.random.default_rng(SEED); bo = []
for _ in range(2000):
    k = rng.integers(0, big.sum(), big.sum()); bo.append(roc_auc_score(yb[k], pAF[big][k]))
sign_metrics["AUC_CI95"] = [float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))]
print("sign:", sign_metrics)

# ---------------- signed J: magnitude x sign versus direct signed regression
Jabs = to_J(yhat); EJ = Jabs * (1 - 2 * pAF)
direct = ExtraTreesRegressor(500, random_state=SEED, n_jobs=2, **gs.best_params_).fit(Xtr, to_y(Jtr))
Jdir = to_J(direct.predict(Xte))
signed = {}
for k, p in [("magnitude x P(sign)", EJ), ("direct signed ET", Jdir), ("zero", np.zeros_like(Jte))]:
    signed[k] = metrics(Jte, p); print(k, {a: round(float(b), 3) for a, b in signed[k].items()})
# ensemble-level AF fraction among strong pairs: predicted vs DFT
strong = Jabs > 0.10
signed["ensemble_AF_fraction"] = dict(pred=float(pAF[strong].mean()), dft=float((Jte[np.abs(Jte) > 0.10] < 0).mean()))

# ---------------- applicability domain: distance to the training set in standardized descriptor space
scal = StandardScaler().fit(Xtr); nn = NearestNeighbors(n_neighbors=5).fit(scal.transform(Xtr))
d_te = nn.kneighbors(scal.transform(Xte))[0].mean(1)
d_tr = NearestNeighbors(n_neighbors=6).fit(scal.transform(Xtr)).kneighbors(scal.transform(Xtr))[0][:, 1:].mean(1)
ad_thr = float(np.percentile(d_tr, 99))

pd.DataFrame(dict(name=F.name.values[te], J=Jte, r_NO_min=rte, y_true=yte, y_ET=yhat, q68=q68, q90=q90,
                  y_RF=cpred["RandomForest"], y_GPR=cpred["GPR-Matern"], y_KRR=cpred["KRR-Laplace"],
                  committee_sd=cstack.std(0), pAF=pAF, EJ=EJ, J_direct=Jdir, ad_dist=d_te)).to_csv(os.path.join(D, "test_predictions.csv"), index=False)
pd.DataFrame(dict(name=F.name.values[tr], J=Jtr, y_true=ytr, y_oof=oof, r_NO_min=rtr)).to_csv(os.path.join(D, "train_oof.csv"), index=False)
json.dump(dict(et_params=gs.best_params_, conformal_edges=edges.tolist(), conformal_q=qtab.tolist(),
               coverage_test={"68": cov68, "90": cov90}, magnitude=mag_metrics, sign=sign_metrics,
               signed=signed, ad_threshold=ad_thr, ad_test_frac_inside=float(np.mean(d_te <= ad_thr)),
               committee_sd_test_median=float(np.median(cstack.std(0))), J0=J0),
          open(os.path.join(D, "final_test_metrics.json"), "w"), indent=1, default=float)

# ---------------- production model: refit on all 1364 pairs
Xall, Jall = F[cols].values, F.J.values; yall = to_y(np.abs(Jall))
et_all = ExtraTreesRegressor(500, random_state=SEED, n_jobs=2, **gs.best_params_).fit(Xall, yall)
oof_all = cross_val_predict(ExtraTreesRegressor(500, random_state=SEED, n_jobs=2, **gs.best_params_), Xall, yall, cv=cv5)
mall = np.abs(Jall) > 0.10
sign_all = make_pipeline(StandardScaler(), LogisticRegression(C=1.0)).fit(np.c_[oof_all, F.r_NO_min.values][mall], (Jall[mall] < 0).astype(int))
committee_all = {k: m.fit(Xall, yall) for k, m in committee.items()}
scal_all = StandardScaler().fit(Xall)
dd = NearestNeighbors(n_neighbors=6).fit(scal_all.transform(Xall)).kneighbors(scal_all.transform(Xall))[0][:, 1:].mean(1)
joblib.dump(dict(cols=cols, et=et_all, committee=committee_all, sign=sign_all, conformal_edges=edges, conformal_q=qtab,
                 scaler=scal_all, nn=NearestNeighbors(n_neighbors=5).fit(scal_all.transform(Xall)),
                 ad_threshold=float(np.percentile(dd, 99)), J0=J0), os.path.join(D, "final_model.joblib"), compress=3)

# ---------------- permutation importance (test set), grouped by descriptor family
fam = {c: ("orbital overlap" if c in SETS["orb"] else "N-O geometry" if c in SETS["geo"] else
           "contacts with H" if c.endswith("_H") else "heavy-atom contacts") for c in cols}
pi = permutation_importance(et, Xte, yte, n_repeats=30, random_state=SEED, scoring="neg_mean_squared_error", n_jobs=2)
PI = pd.DataFrame(dict(feature=cols, family=[fam[c] for c in cols], dMSE=pi.importances_mean, sd=pi.importances_std)).sort_values("dMSE", ascending=False)
PI.to_csv(os.path.join(D, "perm_importance.csv"), index=False); print(PI.head(12))
# grouped importance: permute a whole family together
rows = []
for fname in sorted(set(fam.values())):
    idx = [i for i, c in enumerate(cols) if fam[c] == fname]; inc = []
    for rep in range(30):
        Xp = Xte.copy(); Xp[:, idx] = Xp[rng.permutation(len(Xp))][:, idx]
        inc.append(np.mean((et.predict(Xp) - yte) ** 2) - np.mean((yhat - yte) ** 2))
    rows.append(dict(family=fname, n=len(idx), dMSE=np.mean(inc), sd=np.std(inc)))
pd.DataFrame(rows).to_csv(os.path.join(D, "perm_importance_grouped.csv"), index=False); print(pd.DataFrame(rows))

# ---------------- learning curves
LC = []
for n in [100, 200, 400, 600, 800, len(tr)]:
    for rep in range(5 if n < len(tr) else 1):
        sub = rng.choice(len(tr), n, replace=False) if n < len(tr) else np.arange(len(tr))
        m_ = ExtraTreesRegressor(300, random_state=rep, n_jobs=2, **gs.best_params_).fit(Xtr[sub], ytr[sub])
        p_ = m_.predict(Xte)
        ms = np.abs(Jtr[sub]) > 0.10
        lg = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xtr[sub][ms], (Jtr[sub][ms] < 0).astype(int))
        LC.append(dict(n=n, rep=rep, R2_y=r2_score(yte, p_), RMSE_y=np.sqrt(np.mean((p_ - yte) ** 2)),
                       AUC_sign_geometry=roc_auc_score(yb, lg.predict_proba(Xte[big])[:, 1])))
LC = pd.DataFrame(LC); LC.to_csv(os.path.join(D, "learning_curves.csv"), index=False)
print(LC.groupby("n").agg(["mean", "std"]).round(3))
