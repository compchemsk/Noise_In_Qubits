#!/usr/bin/env python3
"""
m3_04_sign_classification.py
Can the sign of J (F: J > 0, AF: J < 0) be learned from the pair geometry?

Pairs with |J| > 0.10 cm^-1 (about three times the DFT resolution, 0.03 cm^-1)
of the training partition are used.  Eight classifiers x two descriptor sets
(the 58 descriptors; the 841 symmetrised inverse interatomic distances of all
29 x 29 atom pairs) are evaluated by nested CV (5 outer x 3 inner folds).
Scores: ROC-AUC, balanced accuracy, Brier skill score relative to the base
rate.  The permutation test is in m3_04b_sign_permutation.py.
"""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
from m3_ml_common import *

THR = 0.10
F, SETS = load_training()
split = json.load(open(os.path.join(D, "split.json")))
tr = np.array(split["train"])
Ftr = F.iloc[tr].reset_index(drop=True)
m = (Ftr.J.abs() > THR).values
y = (Ftr.J[m] < 0).astype(int).values                       # 1 = antiferromagnetic
Xall = np.load(os.path.join(D, "pairs_allatoms.npz"))["X"][tr][m]
Minv = 1 / np.linalg.norm(Xall[:, :29, None] - Xall[:, None, 29:], axis=3)
Xsets = {"descriptors": Ftr[SETS["all"]].values[m], "invdist841": (Minv + Minv.transpose(0, 2, 1)).reshape(m.sum(), -1),
         "|J|,r only": np.c_[np.log10(Ftr.J.abs().values[m]), Ftr.r_NO_min.values[m]]}
s = StandardScaler
CL = {
 "Logistic": (make_pipeline(s(), LogisticRegression(max_iter=5000, class_weight="balanced")), {"logisticregression__C": [0.01, 0.1, 1]}),
 "kNN": (make_pipeline(s(), KNeighborsClassifier()), {"kneighborsclassifier__n_neighbors": [5, 15, 31]}),
 "SVC-RBF": (make_pipeline(s(), SVC(probability=True, class_weight="balanced", random_state=SEED)), {"svc__C": [0.3, 3, 30]}),
 "GPC-RBF": (make_pipeline(s(), GaussianProcessClassifier(C(1.0) * RBF(10.0), random_state=SEED)), {}),
 "RandomForest": (RandomForestClassifier(500, class_weight="balanced", random_state=SEED, n_jobs=2), {"min_samples_leaf": [1, 5]}),
 "ExtraTrees": (ExtraTreesClassifier(500, class_weight="balanced", random_state=SEED, n_jobs=2), {"min_samples_leaf": [1, 5]}),
 "HistGB": (HistGradientBoostingClassifier(random_state=SEED, class_weight="balanced"), {"learning_rate": [0.03, 0.1], "max_leaf_nodes": [7, 31]}),
 "MLP": (make_pipeline(s(), MLPClassifier((64, 32), max_iter=3000, early_stopping=True, random_state=SEED)), {"mlpclassifier__alpha": [1e-2, 1.0]}),
}
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
base = y.mean()
rows = []
for xs, X in Xsets.items():
    for name, (est, grid) in CL.items():
        if xs == "|J|,r only" and name not in ("Logistic", "GPC-RBF"):
            continue
        p = np.empty(len(y))
        for a, b in outer.split(X, y):
            e = GridSearchCV(est, grid, cv=StratifiedKFold(3, shuffle=True, random_state=SEED), scoring="roc_auc").fit(X[a], y[a]).best_estimator_ if grid else est.fit(X[a], y[a])
            p[b] = e.predict_proba(X[b])[:, 1]
        auc = roc_auc_score(y, p)
        bs, bs0 = brier_score_loss(y, p), brier_score_loss(y, np.full(len(y), base))
        # bootstrap CI of the AUC
        rng = np.random.default_rng(SEED); boots = []
        for _ in range(1000):
            k = rng.integers(0, len(y), len(y))
            if y[k].min() != y[k].max(): boots.append(roc_auc_score(y[k], p[k]))
        rows.append(dict(set=xs, model=name, n=len(y), frac_AF=base, AUC=auc, AUC_lo=np.percentile(boots, 2.5), AUC_hi=np.percentile(boots, 97.5),
                         bal_acc=balanced_accuracy_score(y, (p > 0.5).astype(int)) if name != "GPC-RBF" else balanced_accuracy_score(y, (p > base).astype(int)),
                         Brier=bs, Brier_base=bs0, BSS=1 - bs / bs0))
        np.save(os.path.join(D, f"signcv_{name}_{xs.replace('|','').replace(',','_').replace(' ','')}.npy"), p)
        print(f"{xs:12s} {name:12s} AUC {auc:.3f} [{rows[-1]['AUC_lo']:.3f},{rows[-1]['AUC_hi']:.3f}]  BSS {rows[-1]['BSS']:+.3f}  balacc {rows[-1]['bal_acc']:.3f}", flush=True)
R = pd.DataFrame(rows); R.to_csv(os.path.join(D, "sign_classification_nestedcv.csv"), index=False)
