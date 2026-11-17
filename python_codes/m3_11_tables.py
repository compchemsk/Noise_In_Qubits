#!/usr/bin/env python3
"""m3_11_tables.py -- LaTeX tables of Module III (tables/tabS_m3_*.tex)"""
import os, json, numpy as np, pandas as pd
from m3_ml_common import D, load_training, to_y
T_ = os.path.join(D, "..", "..", "tables")
F, SETS = load_training(); tr = np.array(json.load(open(os.path.join(D, "split.json")))["train"])
v = to_y(np.abs(F.J.values[tr])).var()
pm = lambda m, s, f=2: f"${m:.{f}f}\\pm{s:.{f}f}$"
def write(name, head, rows, cols):
    s = "\\begin{tabular}{" + cols + "}\n\\toprule\n" + head + "\\\\\n\\midrule\n" + "\\\\\n".join(rows) + "\\\\\n\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(T_, name), "w").write(s)

# --- models (nested CV, all descriptors) incl. signed-target rows
B = pd.read_csv(os.path.join(D, "benchmark_mag_nestedcv_folds.csv")); B["R2y"] = 1 - B.RMSE_y ** 2 / v
Bs = pd.read_csv(os.path.join(D, "benchmark_signed_nestedcv_folds.csv"))
vs = to_y(F.J.values[tr]).var(); Bs["R2y"] = 1 - Bs.RMSE_y ** 2 / vs
g = B[B.set == "all"].groupby("model")[["R2y", "RMSE_y", "within3x"]].agg(["mean", "std"]).sort_values(("R2y", "mean"), ascending=False)
gs = Bs.groupby("model")[["R2y", "RMSE_y"]].agg(["mean", "std"])
rows = []
for m in g.index:
    sgn = pm(gs.loc[m, ("R2y", "mean")], gs.loc[m, ("R2y", "std")]) if m in gs.index else "--"
    rows.append(f"{m} & {pm(g.loc[m, ('R2y','mean')], g.loc[m, ('R2y','std')])} & {pm(g.loc[m, ('RMSE_y','mean')], g.loc[m, ('RMSE_y','std')])} & "
                f"{pm(100*g.loc[m, ('within3x','mean')], 100*g.loc[m, ('within3x','std')], 0)} & {sgn}")
write("tabS_m3_models.tex", "Model & $R^2$, $|J|$ & RMSE$_y$, $|J|$ & Within $\\times3$ (\\%) & $R^2$, signed $J$", rows, "lcccc")

# --- descriptor sets
names = {"user3": "Generation coordinates (3)", "geo": "N--O geometry (10)", "orb": "SOMO overlap (6)", "ctc": "Contacts (42)",
         "geo+orb": "Geometry + overlap (16)", "geo+ctc": "Geometry + contacts (52)", "all": "All (58)"}
rows = []
for s in names:
    c = [B[(B.set == s) & (B.model == m)].R2y for m in ("ExtraTrees", "GPR-Matern", "KRR-Laplace")]
    rows.append(f"{names[s]} & " + " & ".join(pm(x.mean(), x.std()) for x in c))
write("tabS_m3_descsets.tex", "Descriptor set & ExtraTrees & GPR (Mat\\'ern) & KRR (Laplacian)", rows, "lccc")

# --- hold-out
M = json.load(open(os.path.join(D, "final_test_metrics.json")))
rows = [f"{k.replace('Baseline: ', 'Baseline, ').replace('exp(-r_NO)', '$\\exp(-r_\\mathrm{NO})$').replace('S^2 + O...H', '$S^2$ + O$\\cdots$H')} & "
        f"${d['R2_y']:.2f}$ & ${d['RMSE_y']:.2f}$ & ${100*d['within3x']:.0f}$ & ${100*d['within10x']:.0f}$" for k, d in M["magnitude"].items()]
write("tabS_m3_holdout.tex", "Model & $R^2$ & RMSE$_y$ & Within $\\times3$ (\\%) & Within $\\times10$ (\\%)", rows, "lcccc")

# --- sign
S = pd.read_csv(os.path.join(D, "sign_classification_nestedcv.csv"))
lab = {"descriptors": "58 descriptors", "invdist841": "841 inverse distances", "|J|,r only": "DFT $|J|$ and $r_\\mathrm{NO}$ (reference)"}
rows = [f"{lab[r.set]} & {r.model} & ${r.AUC:.2f}$ & $[{r.AUC_lo:.2f},{r.AUC_hi:.2f}]$ & ${r.BSS:+.2f}$" for r in S.itertuples()]
write("tabS_m3_sign.tex", "Input & Classifier & AUC & 95\\% CI & BSS", rows, "llccc")

# --- network summary
X = pd.read_csv(os.path.join(D, "exchange_summary_by_replica.csv")); a = X.groupby("T").agg(["mean", "std"])
def r_(lbl, key, f=2, scale=1):
    return f"{lbl} & " + " & ".join(pm(scale * a.loc[T, (key, 'mean')], scale * a.loc[T, (key, 'std')], f) for T in (100, 300))
rows = [r_("Resolved pairs ($|J|>0.1$~cm$^{-1}$)", "n_pairs_res", 0), r_("Pairs with $|J|>1$~cm$^{-1}$", "n_pairs_gt1", 1),
        r_("Resolved partners per radical", "degree_mean"), r_("Radicals without resolved partner (\\%)", "frac_isolated", 0, 100),
        r_("Radicals with $\\geq2$ resolved partners (\\%)", "frac_deg_ge2", 0, 100), r_("Largest coupled cluster (\\% of radicals)", "largest_cluster_frac", 1, 100),
        r_("Median resolved $\\langle|J|\\rangle$ (cm$^{-1}$)", "Jres_median"), r_("Largest $\\langle|J|\\rangle$ (cm$^{-1}$)", "Jmax", 1),
        r_("Largest instantaneous $|J|$ (cm$^{-1}$)", "Jmax_frame", 1), r_("AF share of resolved couplings (\\%)", "frac_AF_res", 0, 100),
        r_("Mean-field $\\theta$, mean over radicals (K)", "theta_mean_K", 3), r_("Mean-field $\\theta$, most negative (K)", "theta_min_K", 2),
        r_("Radicals with $2|J_\\mathrm{max}|>h\\nu_\\mathrm{Q}$ (\\%)", "frac_spins_Jmax_gt_Q", 0, 100),
        r_("$\\omega_\\mathrm{ex}/\\sqrt{M_2}$, coupled radicals (median)", "wex_over_M2_median_coupled", 0),
        r_("Time fraction resolved, resolved pairs (\\%)", "frac_res_time_mean", 0, 100),
        r_("$\\sigma_{2J}$, persistent pairs (GHz)", "sdJ_MHz_median", 1, 1e-3), r_("$\\xi_J$, persistent pairs (median)", "xi_median", 1)]
write("tabS_m3_network.tex", "Quantity & 100 K & 300 K", rows, "lcc")

# --- sensitivity
R = pd.read_csv(os.path.join(D, "exchange_sensitivity_by_replica.csv")); R = R[R.rcut == 9.0]
g = R.groupby(["T", "J_RES", "scope"])[["degree", "frac_isolated", "frac_AF"]].agg(["mean", "std"])
rows = []
for (T, jr, sc), r in g.iterrows():
    rows.append(f"{T} & {jr:.2f} & {sc} & {pm(r[('degree','mean')], r[('degree','std')])} & {pm(100*r[('frac_isolated','mean')], 100*r[('frac_isolated','std')], 0)} & "
                f"{pm(100*r[('frac_AF','mean')], 100*r[('frac_AF','std')], 0)}")
write("tabS_m3_sensitivity.tex", "$T$ (K) & $J_\\mathrm{res}$ (cm$^{-1}$) & Radicals & Partners & Uncoupled (\\%) & AF share (\\%)", rows, "lllccc")

# --- by class
C = pd.read_csv(os.path.join(D, "exchange_by_class_by_replica.csv"))
g = C.groupby(["T", "cls"])[["n", "degree", "frac_isolated", "sumJ"]].agg(["mean", "std"])
nm = {"F": "frozen", "R": "restricted", "M": "mobile", "S": "surface-bound"}
rows = []
for (T, c), r in g.iterrows():
    rows.append(f"{T} & {nm[c]} & {pm(r[('n','mean')], r[('n','std')], 0)} & {pm(r[('degree','mean')], r[('degree','std')])} & "
                f"{pm(100*r[('frac_isolated','mean')], 100*r[('frac_isolated','std')], 0)} & {pm(r[('sumJ','mean')], r[('sumJ','std')])}")
write("tabS_m3_classes.tex", "$T$ (K) & Class & Radicals & Partners & Uncoupled (\\%) & $\\sum_j\\langle|J_{ij}|\\rangle$ (cm$^{-1}$)", rows, "llcccc")
print("tables written")
