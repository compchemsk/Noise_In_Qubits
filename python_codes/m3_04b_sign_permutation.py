#!/usr/bin/env python3
"""
m3_04b_sign_permutation.py
Label-permutation test (n = 1000) for the AUC of geometry-only sign
classification (L2 logistic regression and RBF-SVC on the 58 descriptors,
pairs with |J| > 0.10 cm^-1 of the training partition, 5-fold CV).
"""
import os, json, warnings, numpy as np
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from m3_ml_common import *
F, SETS = load_training(); tr = np.array(json.load(open(os.path.join(D, "split.json")))["train"])
Ftr = F.iloc[tr]; m = (Ftr.J.abs() > 0.10).values
X = Ftr[SETS["all"]].values[m]; y = (Ftr.J.values[m] < 0).astype(int)
out = {}
for name, est, nperm in [("Logistic(C=0.1)", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)), 1000),
                         ("SVC-RBF(C=3)", make_pipeline(StandardScaler(), SVC(C=3, class_weight="balanced")), 300)]:
    meth = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    score = lambda yy, seed: roc_auc_score(yy, (lambda p: p[:, 1] if p.ndim == 2 else p)(
        cross_val_predict(est, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method=meth)))
    obs = np.mean([score(y, s) for s in range(5)])
    rng = np.random.default_rng(SEED)
    null = np.array([score(rng.permutation(y), i) for i in range(nperm)])
    out[name] = dict(AUC_obs=float(obs), null_mean=float(null.mean()), null_sd=float(null.std()),
                     null_q95=float(np.percentile(null, 95)), n_perm=nperm,
                     p_value=float((1 + (null >= obs).sum()) / (1 + nperm)))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(D, "sign_permutation_test.json"), "w"), indent=1)
