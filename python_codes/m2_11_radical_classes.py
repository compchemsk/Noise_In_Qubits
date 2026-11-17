#!/usr/bin/env python3
"""
m2_11_radical_classes.py
Data-driven classification of the radicals.

Descriptors (per radical, averaged over the last 5 ns of a trajectory):
  h_spin   height of the N-O midpoint above Au (A)
  cosNO    mean cos(theta_NO)
  theta0   cone semi-angle from the N-O order parameter (deg)
  n8       number of other spins within 8 A
  tilt     chain tilt (deg)
(The Au-contact fraction and the chain gauche fraction are reported for each
class but not used for clustering: both are zero for many radicals, which makes
full-covariance mixtures degenerate.)
All 141 radicals of all six trajectories are pooled (846 points) so that
the same class definitions apply to both temperatures.  Descriptors are
standardized; Gaussian mixtures (full covariance, regularization 1e-2) with
k = 2-8 are compared by BIC, silhouette score and stability; k is the BIC
minimum among solutions with median restart ARI >= 0.9.  Stability: (i) agreement (adjusted Rand index, ARI)
between 30 fits with different random initializations, (ii) leave-one-replica-
pair-out: refit without replica r at both temperatures and compare labels of
the remaining points.
Outputs: class labels, class centroids, populations per replica (mean +/- SD).
"""
import os, json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial import ConvexHull
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
W = slice(500, 1000)
FEATS = ["h_spin", "cosNO", "theta0", "n8", "tilt"]
REPORT = ["contact", "gauche"]
REG = 1e-2
dyn = pd.read_csv(os.path.join(D, "module2_dynamics_by_molecule.csv"))

rows = []
for T, r in RUNS:
    o = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    S = o["S_yz"].mean(0); hull = ConvexHull(S)
    dist = np.min(-(S @ hull.equations[:, :2].T + hull.equations[:, 2]), axis=1)
    df = pd.DataFrame(dict(T=T, rep=r, mol=o["mol_ids"], edge_dist=dist,
        h_spin=o["h_spin"][W].mean(0), cosNO=o["cos_NO"][W].mean(0), n8=o["ncount_8"][W].mean(0),
        contact=(o["dmin_OAu"][W] < 3.5).mean(0), tilt=o["tilt"][W].mean(0), gauche=o["gauche"][W].mean(0)))
    df = df.merge(dyn[(dyn["T"] == T) & (dyn.rep == r)][["mol", "theta0", "S2"]], on="mol")
    rows.append(df[~df.mol.isin(DEFECT)])
X = pd.concat(rows, ignore_index=True)
Z = StandardScaler().fit_transform(X[FEATS])

# ---- model selection ---------------------------------------------------------
sel = []
for k in range(2, 9):
    gs = [GaussianMixture(k, covariance_type="full", reg_covar=REG, n_init=1, random_state=s_).fit(Z) for s_ in range(30)]
    best = min(gs, key=lambda g: g.bic(Z)); lab = best.predict(Z)
    sel.append(dict(k=k, BIC=best.bic(Z), silhouette=silhouette_score(Z, lab),
                    ARI_restart_median=float(np.median([adjusted_rand_score(lab, g.predict(Z)) for g in gs])),
                    ARI_restart_min=float(np.min([adjusted_rand_score(lab, g.predict(Z)) for g in gs]))))
sel = pd.DataFrame(sel); sel["dBIC"] = sel.BIC - sel.BIC.min()
print(sel.round(3).to_string())
sel.to_csv(os.path.join(D, "module2_class_model_selection.csv"), index=False)
# a class model is kept only if every one of the 30 restarts gives nearly the same labels
stable = sel[sel.ARI_restart_min >= 0.9]
K = int(os.environ.get("NCLASS", stable.loc[stable.BIC.idxmin(), "k"]))

# ---- final model and stability ----------------------------------------------
ref = GaussianMixture(K, covariance_type="full", reg_covar=REG, n_init=30, random_state=1).fit(Z)
lab = ref.predict(Z); prob = ref.predict_proba(Z).max(1)
aris = [adjusted_rand_score(lab, GaussianMixture(K, covariance_type="full", reg_covar=REG, n_init=1, random_state=s).fit(Z).predict(Z))
        for s in range(30)]
loo = []
for r in (1, 2, 3):
    m = (X.rep != r).to_numpy()
    g = GaussianMixture(K, covariance_type="full", reg_covar=REG, n_init=10, random_state=0).fit(Z[m])
    loo.append(adjusted_rand_score(lab[m], g.predict(Z[m])))
stab = dict(K=K, ARI_restarts_median=float(np.median(aris)), ARI_restarts_min=float(np.min(aris)),
            ARI_leave_replica_out=[float(a) for a in loo], mean_max_posterior=float(prob.mean()),
            frac_posterior_gt_0p9=float((prob > 0.9).mean()))
print(stab); json.dump(stab, open(os.path.join(D, "module2_class_stability.json"), "w"), indent=1)

# ---- order classes by spin height (low -> high) and label them ----------------
cent = X.assign(c=lab).groupby("c")[FEATS + REPORT + ["edge_dist"]].mean()
order = cent.sort_values("h_spin").index.tolist()
remap = {old: new for new, old in enumerate(order)}
X["cls"] = [remap[c] for c in lab]; X["post"] = prob
X.to_csv(os.path.join(D, "module2_class_labels.csv"), index=False)
C = X.groupby("cls")[FEATS + REPORT + ["edge_dist", "S2"]].agg(["mean", "std"])
print(C.round(2).to_string())
pop = X.groupby(["T", "rep", "cls"]).size().unstack(fill_value=0)
pop = pop.div(pop.sum(axis=1), axis=0)
popS = pop.groupby(level="T").agg(["mean", "std"])
print(popS.round(3).to_string())
pop.to_csv(os.path.join(D, "module2_class_populations_by_replica.csv"))
# class properties per temperature, replica-resolved
cp = X.groupby(["T", "rep", "cls"])[FEATS + REPORT + ["edge_dist", "S2"]].mean().groupby(["T", "cls"]).agg(["mean", "std"])
cp.to_csv(os.path.join(D, "module2_class_properties.csv"))

# ---- persist the fitted model so other scripts use identical class definitions
import joblib
sc = StandardScaler().fit(X[FEATS])
joblib.dump(dict(scaler=sc, gmm=ref, remap=remap, feats=FEATS), os.path.join(D, "module2_class_model.joblib"))
