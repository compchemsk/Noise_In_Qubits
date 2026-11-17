#!/usr/bin/env python3
"""m4_04_figures.py - figures for the mode-tracking / mutual-information analysis (module 4)."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from plot_style import apply_style, finish_panel, save_fig, COLORS
apply_style()
import matplotlib as mpl
mpl.rcParams.update({'font.size': 28, 'axes.labelsize': 31, 'xtick.labelsize': 27, 'ytick.labelsize': 27, 'legend.fontsize': 22})
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D4 = os.path.join(ROOT, "data", "module4"); FMS = os.path.join(ROOT, "Figures_MS"); FSI = os.path.join(ROOT, "Figures_ESI")
CC = {"S": "#7F7F7F", "M": COLORS[1], "R": COLORS[4], "F": COLORS[0]}
MODES = ["78.3", "117.7", "197.8"]
Rr = pd.read_csv(os.path.join(D4, "m4_radical_level.csv"))
sh = pd.read_csv(os.path.join(D4, "m4_sh.csv")); acf = pd.read_csv(os.path.join(D4, "m4_acf.csv"))
raw = json.load(open(os.path.join(D4, "mode_information_raw.json")))
tests = pd.read_csv(os.path.join(D4, "m4_radical_tests.csv"))

fig, axs = plt.subplots(2, 3, figsize=(24, 15.5)); axs = axs.ravel()
# (a) dynamic amplitude by class (radical medians, interquartile range)
ax = axs[0]; w = 0.2
for j, T in enumerate((100, 300)):
    X = Rr[Rr["T"] == T]
    for c_i, c in enumerate(["F", "R", "M", "S"]):
        d = X[X.cls == c]
        if len(d) < 5: continue
        for k in range(3):
            v = d[f"a{k+1}"]; x = k + (c_i - 1.5) * w
            ax.errorbar(x + (j - 0.5) * 0.07, v.median(), yerr=[[v.median() - v.quantile(.25)], [v.quantile(.75) - v.median()]],
                        fmt="o" if T == 100 else "^", color=CC[c], mec="black", ms=15, capsize=5, lw=2)
for c in ["F", "R", "M", "S"]: ax.plot([], [], "s", color=CC[c], mec="black", ms=13, label=c)
ax.plot([], [], "o", color="white", mec="black", label="100 K"); ax.plot([], [], "^", color="white", mec="black", label="300 K")
ax.set_xticks(range(3)); ax.set_xticklabels([f"{m}" for m in MODES]); ax.set_xlabel(r"TEMPO mode (cm$^{-1}$)")
ax.set_ylabel(r"Relative amplitude $a_k$"); ax.set_ylim(0.2, 1.12)
ax.legend(ncol=3, fontsize=20, loc="upper left", handletextpad=0.3, columnspacing=0.8); finish_panel(ax, "a")
# (b) amplitude of the 78 cm-1 mode vs O-Au distance at 300 K
ax = axs[1]; X = Rr[Rr["T"] == 300]
for c in ["R", "M", "S"]:
    d = X[X.cls == c]; ax.scatter(d.dmin_OAu, d.a1, s=80, color=CC[c], edgecolor="black", lw=0.5, alpha=0.85, label=c)
rho = spearmanr(X.dmin_OAu, X.a1)[0]
ax.set_xscale("log"); ax.set_xticks([3, 5, 10, 20]); ax.set_xticklabels(["3", "5", "10", "20"]); ax.minorticks_off()
ax.set_xlabel(r"Mean O$\cdots$Au distance ($\AA$)"); ax.set_ylabel(r"$a_1$ (78.3 cm$^{-1}$ mode)")
ax.text(0.97, 0.06, rf"300 K, $\rho$ = {rho:.2f}", transform=ax.transAxes, ha="right", fontsize=27)
ax.legend(fontsize=22, loc="upper left"); finish_panel(ax, "b")
# (c) static (radical-to-radical) spread of the spin-Hamiltonian parameters
ax = axs[2]; w = 0.36
for j, (q, lab) in enumerate((("sd_dA_static", r"$\sigma_\mathrm{static}(\delta A_\mathrm{iso})$"), ("sd_dnu_static", r"$\sigma_\mathrm{static}(\delta\nu_g)$, Q band"))):
    for i, (T, g) in enumerate([(100, "interior"), (100, "edge"), (300, "interior"), (300, "edge")]):
        row = sh[(sh["T"] == T) & (sh.group == g)].iloc[0]
        ax.bar(i + (j - 0.5) * w, row[q + "_mean"], w, yerr=row[q + "_sd"], capsize=6, color=[COLORS[0], COLORS[3]][j],
               edgecolor="black", lw=1.5, label=lab if i == 0 else None, error_kw=dict(lw=2))
ax.set_xticks(range(4)); ax.set_xticklabels(["100 K\ninterior", "100 K\nedge", "300 K\ninterior", "300 K\nedge"], fontsize=24)
ax.set_ylabel("Frozen spread (MHz)"); ax.legend(fontsize=22, loc="upper left"); finish_panel(ax, "c")
# (d) autocorrelation of the dynamic part at 10 ps - 1 ns
ax = axs[3]; lags = [1, 2, 5, 10, 20, 50, 100]
for q, col, lab in (("dA", COLORS[0], r"$\delta A_\mathrm{iso}$"), ("dnu_g", COLORS[3], r"$\delta\nu_g$"), ("Q1", COLORS[2], r"$Q_1$")):
    for T, ls in ((100, "--"), (300, "-")):
        row = acf[(acf["T"] == T) & (acf.quantity == q)].iloc[0]
        ax.plot(np.array(lags) * 10, [row[f"lag{l}_mean"] for l in lags], ls, marker="o", color=col, label=f"{lab}, {T} K")
ax.axhline(0, color="black", lw=1.2); ax.set_xscale("log"); ax.set_ylim(-0.05, 1.0)
ax.set_xlabel("Lag time (ps)"); ax.set_ylabel("Autocorrelation"); ax.legend(fontsize=18, ncol=2, loc="upper right"); finish_panel(ax, "d")
# (e) frame-level mutual information against null thresholds
ax = axs[4]
def grab(name, T, filt):
    rows = [r for r in raw[name] if r["T"] == T and filt(r)]
    return np.array([r["MI"] for r in rows]), np.array([r.get("null_block", r["null"]) for r in rows])
cats = [("intra", "same\nradical", lambda r: True),
        ("inter_nn", "nearest\nneighbor", lambda r: r["k"] != "dnu_g"),
        ("inter_dist", "4-7 $\\AA$\npairs", lambda r: r["bin"] == "4-7"),
        ("env", "environ-\nment", lambda r: r["mode"] in (1, 2, 3))]
for i, (nm, lab, f) in enumerate(cats):
    for j, T in enumerate((100, 300)):
        mi, nl = grab(nm, T, f); x = i + (j - 0.5) * 0.3
        ax.scatter(np.full(len(mi), x) + np.random.default_rng(i * 10 + j).uniform(-0.07, 0.07, len(mi)), mi, s=40,
                   color=[COLORS[0], COLORS[1]][j], edgecolor="black", lw=0.4, zorder=3)
        ax.plot([x - 0.12, x + 0.12], [np.median(nl)] * 2, color="black", lw=3, zorder=4)
ax.scatter([], [], color=COLORS[0], edgecolor="black", label="100 K"); ax.scatter([], [], color=COLORS[1], edgecolor="black", label="300 K")
ax.plot([], [], color="black", lw=3, label="null (median)")
ax.set_xticks(range(4)); ax.set_xticklabels([c[1] for c in cats], fontsize=23); ax.set_ylabel("Mutual information (bits)")
ax.legend(fontsize=20, loc="upper left"); finish_panel(ax, "e")
# (f) radical-level Spearman correlations (significant after FDR control marked)
ax = axs[5]
ENV = ["h_spin", "dmin_OAu", "cos_NO", "tilt", "gauche", "ncount_8", "nn1"]
EL = ["height", r"O$\cdots$Au", "N-O orient.", "tilt", "gauche", "spins<8 $\\AA$", "n.n. dist."]
M = np.zeros((6, len(ENV))); S = np.zeros_like(M, bool); rowsl = []
for i, (T, q) in enumerate([(100, "a1"), (100, "s1"), (300, "a1"), (300, "s1"), (300, "a2"), (300, "a3")]):
    rowsl.append(f"{T} K, {'$a$' if q[0]=='a' else '$s$'}$_{q[1]}$")
    for j, e in enumerate(ENV):
        t = tests[(tests["T"] == T) & (tests.quantity == q) & (tests["var"] == e)].iloc[0]; M[i, j] = t.rho; S[i, j] = t.q_BH < 0.05
im = ax.imshow(M, cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i,j]:.2f}" + ("*" if S[i, j] else ""), ha="center", va="center", fontsize=15)
ax.set_xticks(range(len(ENV))); ax.set_xticklabels(EL, rotation=40, ha="right", fontsize=20)
ax.set_yticks(range(6)); ax.set_yticklabels(rowsl, fontsize=20); cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cb.set_label(r"Spearman $\rho$ (radical level)", fontsize=23); finish_panel(ax, "f")
fig.tight_layout(); save_fig(fig, os.path.join(FMS, "Figure6_mode_environment"))

# ---------------- SI figure: checks ----------------
chk = json.load(open(os.path.join(D4, "mode_checks.json")))
fig, axs = plt.subplots(1, 3, figsize=(24, 8.8))
ax = axs[0]; t = chk["TEMPO"]; p = chk["PTEMPO"]
ax.bar(np.arange(3) - 0.2, t["coverage"], 0.4, color=COLORS[0], edgecolor="black", label="TEMPO modes")
ax.bar(np.arange(3, 11) + 0.2, p["coverage"], 0.4, color=COLORS[1], edgecolor="black", label="PTEMPO modes")
ax.set_xticks(range(11)); ax.set_xticklabels([f"{v:.0f}" for v in t["nu"] + p["nu"]], rotation=45, fontsize=22)
ax.set_xlabel(r"Mode (cm$^{-1}$)"); ax.set_ylabel("Coverage of mapped atoms"); ax.set_ylim(0, 1.45); ax.legend(fontsize=18, loc="upper center", ncol=2, columnspacing=0.8, handlelength=1.2); finish_panel(ax, "a")
ax = axs[1]
for T, col in ((100, COLORS[0]), (300, COLORS[1])):
    v = np.concatenate([np.load(os.path.join(D4, f"aligned_TEMPO_T{T}_r{r}.npz"))["core_rmsd"].ravel() for r in (1, 2, 3)])
    ax.hist(v, bins=np.linspace(0, 0.9, 61), histtype="step", lw=3, color=col, density=True, label=f"{T} K")
ax.axvline(0.3, ls="--", color="black", lw=2); ax.text(0.31, ax.get_ylim()[1] * 0.8, "chair cutoff", fontsize=22)
ax.set_xlabel(r"RMSD to DFT TEMPO ($\AA$)"); ax.set_ylabel("Density"); ax.legend(fontsize=22); finish_panel(ax, "b")
ax = axs[2]; d = pd.read_csv(os.path.join(D4, "m4_inter_dist.csv"))
for k, col in zip(range(1, 4), COLORS[:3]):
    for T, ls in ((100, "--"), (300, "-")):
        x = d[(d["T"] == T) & (d["mode"] == k)].copy(); x["lo"] = x["bin"].str.split("-").str[0].astype(float); x = x.sort_values("lo")
        ax.plot(x.lo + 1.5, x.MI_mean, ls, marker="o", color=col, label=f"{MODES[k-1]} cm$^{{-1}}$, {T} K")
        if k == 1 and T == 300: ax.fill_between(x.lo + 1.5, 0, x.null_mean, color="grey", alpha=0.3, label="null (300 K)")
ax.set_ylim(0, 0.068); ax.set_xlabel(r"Spin–spin distance ($\AA$)"); ax.set_ylabel("Same-mode MI (bits)"); ax.legend(fontsize=17, ncol=2, loc="upper left", columnspacing=0.6, handlelength=1.6); finish_panel(ax, "c")
fig.tight_layout(); save_fig(fig, os.path.join(FSI, "FigS_mode_tracking_checks"))
print("figures written")
