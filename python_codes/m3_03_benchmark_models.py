#!/usr/bin/env python3
"""
m3_03_benchmark_models.py
Nested cross-validated benchmark of eleven regressors x seven descriptor sets.

Protocol
  * 15 % of the 1364 DFT pairs are held out once (stratified by O..O band and
    sign of J) and are not used for any choice made here.
  * On the remaining 85 %: nested CV, 5 outer folds x 3 inner folds.  The
    inner loop selects hyper-parameters (grid search, MSE in y); the outer
    loop measures the generalisation error of the whole procedure.
  * Targets (env M3_TARGET): "mag"    y = asinh(|J| / 0.05 cm^-1)
                             "signed" y = asinh( J  / 0.05 cm^-1)
    The transform is linear below the DFT resolution and logarithmic above.  Errors are reported in y and, after back-transformation,
    in cm^-1.
Output: data/module3/benchmark_nestedcv.csv (+ per-fold rows)
"""
import os, sys, time, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from m3_ml_common import *

def models():
    s = StandardScaler
    return {
     "Ridge":      (make_pipeline(s(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100]}),
     "Ridge-poly2":(make_pipeline(s(), PolynomialFeatures(2), s(), Ridge()), {"ridge__alpha": [1, 10, 100, 1000]}),
     "kNN":        (make_pipeline(s(), KNeighborsRegressor(weights="distance")), {"kneighborsregressor__n_neighbors": [3, 5, 10, 20]}),
     "SVR":        (make_pipeline(s(), SVR(kernel="rbf")), {"svr__C": [1, 10, 100], "svr__gamma": ["scale", 0.01], "svr__epsilon": [0.05, 0.2]}),
     "KRR-RBF":    (make_pipeline(s(), KernelRidge(kernel="rbf")), {"kernelridge__alpha": [1e-3, 1e-2, 1e-1], "kernelridge__gamma": [0.003, 0.01, 0.03, 0.1]}),
     "KRR-Laplace":(make_pipeline(s(), KernelRidge(kernel="laplacian")), {"kernelridge__alpha": [1e-3, 1e-2, 1e-1], "kernelridge__gamma": [0.003, 0.01, 0.03]}),
     "GPR-Matern": (make_pipeline(s(), GaussianProcessRegressor(C(1.0) * Matern(length_scale=3.0, nu=2.5) + WhiteKernel(0.1), normalize_y=True, n_restarts_optimizer=1, random_state=SEED)), {}),
     "RandomForest":(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=2), {"min_samples_leaf": [1, 3], "max_features": [0.33, 1.0]}),
     "ExtraTrees": (ExtraTreesRegressor(n_estimators=300, random_state=SEED, n_jobs=2), {"min_samples_leaf": [1, 3], "max_features": [0.33, 1.0]}),
     "HistGB":     (HistGradientBoostingRegressor(random_state=SEED, max_iter=600, early_stopping=False), {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [15, 31], "l2_regularization": [0.0, 1.0]}),
     "MLP":        (make_pipeline(s(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, early_stopping=True, random_state=SEED)), {"mlpregressor__alpha": [1e-3, 1e-1, 1.0], "mlpregressor__hidden_layer_sizes": [(64, 64), (128, 64, 32)]}),
    }

if __name__ == "__main__":
    F, SETS = load_training()
    tr, te, strat = split_masks(F)
    json.dump({"train": tr.tolist(), "test": te.tolist()}, open(os.path.join(D, "split.json"), "w"))
    Ftr = F.iloc[tr].reset_index(drop=True); st = strat[tr]
    TARGET = os.environ.get("M3_TARGET", "mag")          # mag: asinh(|J|/J0); signed: asinh(J/J0)
    J = Ftr.J.values
    y = to_y(np.abs(J)) if TARGET == "mag" else to_y(J)
    Jref = np.abs(J) if TARGET == "mag" else J
    which_models = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(models())
    which_sets = sys.argv[2].split(",") if len(sys.argv) > 2 else ["all"]
    outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    out_csv = os.path.join(D, f"benchmark_{TARGET}_nestedcv_folds.csv")
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["model", "set"])
    for fs in which_sets:
        X = Ftr[SETS[fs]].values
        for name in which_models:
            if ((done.model == name) & (done.set == fs)).any():
                continue
            est, grid = models()[name]
            t0 = time.time(); pred = np.empty_like(y); fold_rows = []
            for k, (a, b) in enumerate(outer.split(X, st)):
                if grid:
                    gs = GridSearchCV(est, grid, cv=KFold(3, shuffle=True, random_state=SEED),
                                      scoring="neg_mean_squared_error", n_jobs=1 if name in ("RandomForest", "ExtraTrees") else 2)
                    gs.fit(X[a], y[a]); best = gs.best_estimator_; bp = gs.best_params_
                else:
                    best = est.fit(X[a], y[a]); bp = {}
                pred[b] = best.predict(X[b])
                m = metrics(Jref[b], to_J(pred[b]), y[b], pred[b]); m.update(target=TARGET, model=name, set=fs, fold=k, params=json.dumps(bp, default=str))
                fold_rows.append(m)
            df = pd.DataFrame(fold_rows)
            df.to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            np.save(os.path.join(D, f"cvpred_{TARGET}_{name}_{fs}.npy"), pred)
            print(f"{fs:8s} {name:13s} RMSE_y {df.RMSE_y.mean():.3f}+-{df.RMSE_y.std():.3f}  MAE_J {df.MAE_J.mean():.3f}  "
                  f"RMSE_J {df.RMSE_J.mean():.2f}  rho {df.spearman.mean():.3f}  sign {df.sign_acc.mean():.3f}  ({time.time()-t0:.0f} s)", flush=True)
