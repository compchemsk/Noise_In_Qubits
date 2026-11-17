#!/usr/bin/env python3
"""m3_ml_common.py -- shared settings for the Module III machine-learning workflow"""
import os, numpy as np, pandas as pd
from m3_features import define_sets
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "module3")
J0 = 0.05            # cm^-1; scale of the target transform y = asinh(J/J0) (~ DFT resolution)
SEED = 2026

def to_y(J):  return np.arcsinh(np.asarray(J) / J0)
def to_J(y):  return J0 * np.sinh(np.asarray(y))

def prepare(F):
    """deterministic column transforms (no fitted parameters -> no leakage)"""
    G = F.copy()
    for c in G.columns:
        if c.startswith("soft_") or c in ("S_somo", "S_NN", "S_OO", "S_NO"):
            G[c] = np.log10(G[c] + 1e-20)
    return G

def load_training():
    F = pd.read_csv(os.path.join(D, "train_features.csv"))
    sets = define_sets([c for c in F.columns if c not in ("name", "J")])
    return prepare(F), sets

def split_masks(F, frac_test=0.15, seed=SEED):
    """stratified hold-out: strata = r_OO band x sign of J"""
    from sklearn.model_selection import train_test_split
    band = np.digitize(F.r_OO, [3.0, 4.0, 5.0, 6.0])
    strat = band * 3 + (np.sign(F.J).astype(int) + 1)
    idx = np.arange(len(F))
    tr, te = train_test_split(idx, test_size=frac_test, random_state=seed, stratify=strat)
    return np.sort(tr), np.sort(te), strat

def metrics(J_true, J_pred, y_true=None, y_pred=None):
    from scipy.stats import spearmanr
    J_true, J_pred = np.asarray(J_true), np.asarray(J_pred)
    e = J_pred - J_true
    big = np.abs(J_true) > 0.10                       # > ~3x the DFT resolution
    out = dict(MAE_J=np.mean(np.abs(e)), RMSE_J=np.sqrt(np.mean(e ** 2)),
               R2_J=1 - np.sum(e ** 2) / np.sum((J_true - J_true.mean()) ** 2),
               MedAE_J=np.median(np.abs(e)),
               spearman=spearmanr(J_true, J_pred).statistic,
               sign_acc=np.mean(np.sign(J_pred[big]) == np.sign(J_true[big])) if big.any() else np.nan,
               MAE_J_small=np.mean(np.abs(e[np.abs(J_true) <= 1.0])),
               within3x=np.mean(np.abs(np.log10(np.abs(J_pred[big]) + 1e-9) - np.log10(np.abs(J_true[big]))) < np.log10(3)) if big.any() else np.nan)
    if y_true is not None:
        ey = np.asarray(y_pred) - np.asarray(y_true)
        out.update(RMSE_y=np.sqrt(np.mean(ey ** 2)),
                   R2_y=1 - np.sum(ey ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    return out
