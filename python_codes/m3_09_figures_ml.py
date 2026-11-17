#!/usr/bin/env python3
"""m3_09_figures_ml.py -- Figure 4 (learning the exchange coupling) and the ML figures of the SI."""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from plot_style import apply_style, finish_panel, save_fig, COLORS
from m3_ml_common import D, to_J, to_y, load_training
apply_style()
ROOT = os.path.join(D, "..", ".."); FMS = os.path.join(ROOT, "Figures_MS"); FSI = os.path.join(ROOT, "Figures_ESI")
AF, FM, GREY = COLORS[0], COLORS[1], "#9A9A9A"
F, SETS = load_training()
sp = json.load(open(os.path.join(D, "split.json"))); tr, te = np.array(sp["train"]), np.array(sp["test"])
B = pd.read_csv(os.path.join(D, "benchmark_mag_nestedcv_folds.csv"))
TP = pd.read_csv(os.path.join(D, "test_predictions.csv")); FM_ = json.load(open(os.path.join(D, "final_test_metrics.json")))
LC = pd.read_csv(os.path.join(D, "learning_curves.csv"))
SC = pd.read_csv(os.path.join(D, "sign_classification_nestedcv.csv"))
perm = json.load(open(os.path.join(D, "sign_permutation_test.json")))
syv = to_y(np.abs(F.J.values[tr])).var()

def r2y(df):                     # fold R2 in y relative to the variance of the training partition
    return 1 - df.RMSE_y ** 2 / syv

fig, axs = plt.subplots(2, 3, figsize=(20, 14)); fig.subplots_adjust(hspace=0.60, wspace=0.42)
# (a) DFT data
ax = axs[0, 0]
for s, c, lab in [(-1, AF, "AF ($J<0$)"), (1, FM, "F ($J>0$)")]:
    m = np.sign(F.J) == s
    ax.scatter(F.r_NO_min[m], F.J[m], s=22, color=c, edgecolor="black", lw=0.3, alpha=0.8, label=f"{lab}, n = {m.sum()}")
m0 = F.J == 0; ax.scatter(F.r_NO_min[m0], F.J[m0], s=22, color="white", edgecolor="black", lw=0.5, label=f"$J=0$, n = {m0.sum()}")
ax.axhspan(-0.1, 0.1, color=GREY, alpha=0.3, lw=0, label=r"$|J|\leq0.1$ cm$^{-1}$")
ax.set_yscale("symlog", linthresh=0.1, linscale=0.6); ax.set_ylim(-100, 40)
ax.set_xlabel(r"Closest N/O$\cdots$N/O distance (Å)"); ax.set_ylabel(r"$J_\mathrm{DFT}$ (cm$^{-1}$)")
ax.legend(fontsize=15, loc="upper right", markerscale=1.6, handletextpad=0.2); finish_panel(ax, "a")
# (b) model benchmark (nested CV, all descriptors)
ax = axs[0, 1]
b = B[B.set == "all"].copy(); b["R2y"] = r2y(b)
g = b.groupby("model").R2y.agg(["mean", "std"]).sort_values("mean")
cols = [COLORS[2] if k in ("ExtraTrees",) else COLORS[5] for k in g.index]
ax.barh(range(len(g)), g["mean"], xerr=g["std"], color=cols, edgecolor="black", lw=1.5, capsize=5, error_kw=dict(lw=2))
ax.set_yticks(range(len(g))); ax.set_yticklabels(g.index, fontsize=17)
ax.set_xlabel(r"Nested-CV $R^2$"); ax.set_xlim(0.3, 0.8)
ax.axvline(g["mean"].max(), color="black", ls=":", lw=1.5); finish_panel(ax, "b")
# (c) descriptor sets
ax = axs[0, 2]
order = ["user3", "geo", "orb", "ctc", "geo+ctc", "all"]
names = {"user3": "generation (3)", "geo": "N–O geometry (10)", "orb": "SOMO overlap (6)", "ctc": "contacts (42)",
         "geo+ctc": "geom. + contacts (52)", "all": "all (58)"}
w = 0.27
for k, (mdl, c) in enumerate([("ExtraTrees", COLORS[2]), ("GPR-Matern", COLORS[0]), ("KRR-Laplace", COLORS[4])]):
    vals = [r2y(B[(B.model == mdl) & (B.set == s)]) for s in order]
    ax.bar(np.arange(6) + (k - 1) * w, [v.mean() for v in vals], w, yerr=[v.std() for v in vals], color=c, edgecolor="black",
           lw=1.3, capsize=4, label=mdl, error_kw=dict(lw=1.6))
ax.set_xticks(range(6)); ax.set_xticklabels([names[s] for s in order], fontsize=16, rotation=35, ha="right")
ax.set_ylabel(r"Nested-CV $R^2$"); ax.set_ylim(0.3, 0.98); ax.legend(fontsize=15, loc="upper left", ncol=1, handlelength=1.2); finish_panel(ax, "c")
# (d) parity on the hold-out set
ax = axs[1, 0]
x = np.abs(TP.J.values); yp = to_J(TP.y_ET.values)
lo, hi = to_J(np.maximum(TP.y_ET - TP.q68, 0)), to_J(TP.y_ET + TP.q68)
fl = 0.008
for s, c in [(-1, AF), (1, FM)]:
    m = np.sign(TP.J) == s
    ax.errorbar(np.maximum(x[m], fl), yp[m], yerr=[yp[m] - np.maximum(lo[m], fl), hi[m] - yp[m]], fmt="o", ms=6, color=c,
                mec="black", mew=0.4, ecolor=c, elinewidth=0.8, alpha=0.85, label="AF" if s < 0 else "F")
m = TP.J == 0; ax.scatter(np.full(m.sum(), fl), yp[m], s=30, color="white", edgecolor="black", label="$J=0$ (at 0.008)")
ax.plot([fl, 100], [fl, 100], "k-", lw=1.5); ax.plot([fl, 100], [3 * fl, 300], "k:", lw=1.2); ax.plot([fl, 100], [fl / 3, 100 / 3], "k:", lw=1.2)
ax.axvspan(fl, 0.1, color=GREY, alpha=0.25, lw=0)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(fl * 0.8, 100); ax.set_ylim(fl * 0.8, 100)
ax.set_xlabel(r"$|J|_\mathrm{DFT}$ (cm$^{-1}$)"); ax.set_ylabel(r"$|J|_\mathrm{ML}$ (cm$^{-1}$)")
mm = FM_["magnitude"]["ExtraTrees"]
ax.text(0.04, 0.96, f"hold-out set, n = {len(TP)}\n$R^2$ = {mm['R2_y']:.2f} (asinh scale)\n"
        f"{100*mm['within3x']:.0f}% within ×3 for $|J|>0.1$\n68% interval coverage {100*FM_['coverage_test']['68']:.0f}%",
        transform=ax.transAxes, va="top", fontsize=17, bbox=dict(fc="white", ec="none"))
ax.legend(fontsize=14, loc="lower right"); finish_panel(ax, "d")
# (e) sign: ROC
ax = axs[1, 1]
mtr = np.abs(F.J.values[tr]) > 0.10; ytr_s = (F.J.values[tr][mtr] < 0).astype(int)
for fn, lab, c in [("signcv_Logistic_descriptors.npy", "58 descriptors (CV)", COLORS[5]),
                   ("signcv_ExtraTrees_invdist841.npy", "841 distances (CV)", COLORS[3])]:
    p = np.load(os.path.join(D, fn)); fpr, tpr, _ = roc_curve(ytr_s, p)
    auc = SC[(SC.model == fn.split("_")[1]) & (SC.set == fn.split("_")[2][:-4].replace("invdist841", "invdist841"))].AUC
    ax.plot(fpr, tpr, color=c, lw=2.5, label=f"{lab}: {float(auc.iloc[0]):.2f}")
mt = np.abs(TP.J) > 0.10
fpr, tpr, _ = roc_curve((TP.J[mt] < 0).astype(int), TP.pAF[mt])
ax.plot(fpr, tpr, color="black", lw=3, label=f"final model (hold-out): {FM_['sign']['AUC']:.2f}")
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=2)
ax.set_xlabel("False AF rate"); ax.set_ylabel("True AF rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.0)
ax.legend(fontsize=12.5, loc="lower right", title="AUC", title_fontsize=12.5, handlelength=1.2, borderaxespad=0.3)
ax.text(0.04, 0.96, f"permutation null AUC\n{perm['Logistic(C=0.1)']['null_mean']:.2f} ± {perm['Logistic(C=0.1)']['null_sd']:.2f}",
        transform=ax.transAxes, va="top", fontsize=17, bbox=dict(fc="white", ec="none"))
finish_panel(ax, "e")
# (f) learning curves
ax = axs[1, 2]
g = LC.groupby("n").agg(["mean", "std"])
ax.errorbar(g.index, g[("R2_y", "mean")], yerr=g[("R2_y", "std")], fmt="o-", color=COLORS[2], mec="black", capsize=5, label=r"$|J|$: $R^2$ (hold-out)")
ax.errorbar(g.index, g[("AUC_sign_geometry", "mean")], yerr=g[("AUC_sign_geometry", "std")], fmt="s--", color=COLORS[3], mec="black", capsize=5,
            label="sign: AUC, geometry only")
ax.axhline(0.5, color=GREY, ls=":", lw=1.5)
ax.set_xlabel("Training pairs"); ax.set_ylabel("Score"); ax.set_ylim(0.4, 0.85); ax.legend(fontsize=17, loc="lower right"); finish_panel(ax, "f")
save_fig(fig, os.path.join(FMS, "Figure4_exchange_ML"))

# ================= SI: dataset
fig, axs = plt.subplots(1, 3, figsize=(19, 6.4)); fig.subplots_adjust(wspace=0.40)
ax = axs[0]
bins = np.r_[np.arange(2.0, 6.01, 0.2), np.arange(6.5, 11, 1)]
ax.hist(F.r_OO, bins=bins, color=COLORS[5], edgecolor="black", lw=1.2)
ax.set_xlabel(r"O$\cdots$O distance (Å)"); ax.set_ylabel("Pairs after the contact filter"); finish_panel(ax, "a")
ax = axs[1]
band = pd.cut(F.r_OO, [2, 3, 4, 5, 6, 11]); g = F.groupby(band, observed=True)
for k, (iv, d) in enumerate(g):
    v = np.log10(np.abs(d.J[d.J != 0]))
    ax.boxplot(v, positions=[k], widths=0.6, patch_artist=True, boxprops=dict(facecolor=COLORS[4], lw=1.5),
               medianprops=dict(color="black", lw=2), whiskerprops=dict(lw=1.5), capprops=dict(lw=1.5), flierprops=dict(ms=4))
ax.set_xticks(range(5)); ax.set_xticklabels(["2–3", "3–4", "4–5", "5–6", "6–11"])
ax.set_xlabel(r"O$\cdots$O distance (Å)"); ax.set_ylabel(r"$\log_{10}|J|$ (cm$^{-1}$)"); ax.axhline(-1, color="black", ls=":", lw=1.5); finish_panel(ax, "b")
ax = axs[2]
far = F[F.r_OO >= 8 - 1e-6]
ax.hist(far.J, bins=np.arange(-0.155, 0.1, 0.01), color=GREY, edgecolor="black", lw=1.2)
ax.set_xlabel(r"$J$ (cm$^{-1}$), O$\cdots$O $\geq$ 8 Å"); ax.set_ylabel("Pairs")
ax.text(0.04, 0.96, f"n = {len(far)}\nmean {far.J.mean():.3f}\nSD {far.J.std():.3f} cm$^{{-1}}$", transform=ax.transAxes, va="top",
        fontsize=18, bbox=dict(fc="white", ec="none")); finish_panel(ax, "c")
save_fig(fig, os.path.join(FSI, "FigS_exchange_dataset"))

# ================= SI: importance + calibration + applicability domain
PI = pd.read_csv(os.path.join(D, "perm_importance.csv")).head(15)[::-1]
PG = pd.read_csv(os.path.join(D, "perm_importance_grouped.csv")).sort_values("dMSE")
fig, axs = plt.subplots(1, 3, figsize=(21, 7.5)); fig.subplots_adjust(wspace=0.95)
ax = axs[0]
famc = {"orbital overlap": COLORS[3], "N-O geometry": COLORS[0], "contacts with H": COLORS[4], "heavy-atom contacts": COLORS[2]}
ax.barh(range(len(PI)), PI.dMSE, xerr=PI.sd, color=[famc[f] for f in PI.family], edgecolor="black", lw=1.2, capsize=3)
ax.set_yticks(range(len(PI))); ax.set_yticklabels([s.replace("_", " ") for s in PI.feature], fontsize=19)
ax.set_xlabel(r"$\Delta$MSE on permutation"); finish_panel(ax, "a")
from matplotlib.patches import Patch

ax = axs[1]
from matplotlib.patches import Patch
ax.barh(range(len(PG)), PG.dMSE, xerr=PG.sd, color=[famc[f] for f in PG.family], edgecolor="black", lw=1.2, capsize=4)
ax.set_yticks(range(len(PG))); ax.set_yticklabels([f"{f}\n({n})" for f, n in zip(PG.family, PG.n)], fontsize=20)
ax.set_xlabel(r"$\Delta$MSE, family permuted"); finish_panel(ax, "b")
ax = axs[2]
M = __import__("joblib").load(os.path.join(D, "final_model.joblib"))
ad_md = []
for T in (100, 300):
    for r in (1, 2, 3):
        z = np.load(os.path.join(D, f"mdJ_T{T}_r{r}.npz")); ok = z["intact"][z["pairs"]].all(1)
        ev = z["evaluated"][:, ok]; a = z["ad"][:, ok][ev]; rr = z["rNO"][:, ok][ev]; ad_md.append(a[rr < 6])
ad_md = np.concatenate(ad_md)
bins = np.linspace(0, 12, 49)
ax.hist(TP.ad_dist, bins=bins, density=True, histtype="step", lw=3, color="black", label="hold-out pairs")
ax.hist(ad_md, bins=bins, density=True, histtype="stepfilled", alpha=0.5, color=COLORS[2], edgecolor="black", label=r"MD pairs, < 6 Å")
ax.axvline(M["ad_threshold"], color=FM, lw=2.5, ls="--", label="99th pct., training")
ax.set_xlim(0, 9); ax.set_ylim(top=ax.get_ylim()[1] * 1.6)
ax.set_xlabel("Distance to training set"); ax.set_ylabel("Density"); ax.legend(fontsize=18, loc="upper right"); finish_panel(ax, "c")
save_fig(fig, os.path.join(FSI, "FigS_exchange_importance_AD"))

# ================= SI: sign analysis
fig, axs = plt.subplots(1, 3, figsize=(21, 7.5)); fig.subplots_adjust(wspace=0.75)
Fs = F[np.abs(F.J) > 0.1]
ax = axs[0]
bands = [(2, 4.5), (4.5, 5.5), (5.5, 12)]; jb = [0.1, 0.5, 3, 100]
for k, (a, b_) in enumerate(bands):
    d = Fs[(Fs.r_NO_min >= a) & (Fs.r_NO_min < b_)]
    fr = [((d.J.abs() > jb[i]) & (d.J.abs() <= jb[i + 1])) for i in range(3)]
    ax.bar(np.arange(3) + (k - 1) * 0.27, [(d.J[f] < 0).mean() if f.sum() else np.nan for f in fr], 0.27, color=COLORS[[0, 5, 2][k]],
           edgecolor="black", lw=1.3, label=f"{a}–{b_ if b_ < 12 else '12'} Å")
    for i, f in enumerate(fr):
        ax.text(i + (k - 1) * 0.27, 0.03, str(f.sum()), ha="center", va="bottom", rotation=90, fontsize=14, color="white" if k == 0 else "black")
ax.set_xticks(range(3)); ax.set_xticklabels(["0.1–0.5", "0.5–3", "> 3"]); ax.set_xlabel(r"$|J|_\mathrm{DFT}$ (cm$^{-1}$)")
ax.set_ylabel("Fraction AF"); ax.set_ylim(0, 1.75); ax.legend(title=r"N/O$\cdots$N/O", fontsize=17, title_fontsize=17, loc="upper left", ncol=2, columnspacing=0.6, handlelength=1.0, handletextpad=0.3); finish_panel(ax, "a")
ax = axs[1]
s2 = SC[SC.set != "|J|,r only"].copy()
lab = [f"{m} · {'58' if s == 'descriptors' else '841'}" for m, s in zip(s2.model, s2.set)]
o = np.argsort(s2.AUC.values)
ax.errorbar(s2.AUC.values[o], range(len(o)), xerr=[s2.AUC.values[o] - s2.AUC_lo.values[o], s2.AUC_hi.values[o] - s2.AUC.values[o]],
            fmt="o", color=COLORS[3], mec="black", ms=8, capsize=4)
ax.set_yticks(range(len(o))); ax.set_yticklabels(np.array(lab)[o], fontsize=18)
ax.axvline(0.5, color=GREY, ls="--", lw=2); ax.set_xlabel("Nested-CV AUC (95% CI)"); finish_panel(ax, "b")
ax = axs[2]
cal = pd.DataFrame(dict(p=TP.pAF[mt], af=(TP.J[mt] < 0).astype(int)))
tro = pd.read_csv(os.path.join(D, "train_oof.csv")); mo = tro.J.abs() > 0.1
pt = M["sign"].predict_proba(np.c_[tro.y_oof, tro.r_NO_min][mo])[:, 1]
cal2 = pd.DataFrame(dict(p=pt, af=(tro.J[mo] < 0).astype(int)))
for dfc, c, lab_, mk in [(cal2, COLORS[5], "training (out-of-fold)", "s"), (cal, "black", "hold-out", "o")]:
    q = pd.qcut(dfc.p, 4, duplicates="drop"); g = dfc.groupby(q, observed=True).agg(p=("p", "mean"), af=("af", "mean"), n=("af", "size"))
    ax.errorbar(g.p, g.af, yerr=np.sqrt(g.af * (1 - g.af) / g.n), fmt=mk + "-", color=c, mec="black", capsize=4, label=lab_)
ax.plot([0.4, 1], [0.4, 1], color=GREY, ls="--"); ax.set_xlim(0.4, 1); ax.set_ylim(0.3, 1.05)
ax.set_xlabel(r"Predicted $P_\mathrm{AF}$"); ax.set_ylabel("Observed AF fraction"); ax.legend(fontsize=17, loc="upper left"); finish_panel(ax, "c")
save_fig(fig, os.path.join(FSI, "FigS_exchange_sign"))
print("ML figures written")
