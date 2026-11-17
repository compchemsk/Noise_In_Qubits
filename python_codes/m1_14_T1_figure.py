#!/usr/bin/env python3
"""m1_14_T1_figure.py - SI figure and table for the multi-process analysis of 1/T1 (m1_13)."""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plot_style import apply_style, finish_panel, save_fig, COLORS
from m1_13_multiprocess_T1 import DATA, T, MODELS, fit, term
apply_style()
import matplotlib as mpl
mpl.rcParams.update({"font.size": 30, "axes.labelsize": 32, "xtick.labelsize": 28, "ytick.labelsize": 28, "legend.fontsize": 24})
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D1 = os.path.join(ROOT, "data", "module1"); FSI = os.path.join(ROOT, "Figures_ESI"); TAB = os.path.join(ROOT, "tables")
TITLE = {"PTEMPO_slow": "PTEMPO, slow", "PTEMPO_fast": "PTEMPO, fast", "SAM": "TEMPO monolayer"}
fig, axs = plt.subplots(2, 3, figsize=(24, 16)); axs = axs.ravel(); Tf = np.linspace(5, 65, 300); fr = {}
for i, ds in enumerate(DATA):
    t1, e = DATA[ds]; ok = ~np.isnan(t1); Tx = T[ok]; R = 1e3 / t1[ok]; sR = R * e[ok] / t1[ok]
    ax = axs[i]; ax.errorbar(Tx, R, yerr=sR, fmt="o", color="black", ms=11, capsize=4, lw=2, zorder=6, label="experiment")
    f = fit(MODELS["D+L"], Tx, R, sR); d_, l_ = f["pars"]
    ax.plot(Tf, term("D", d_, Tf) + term("L", l_, Tf), color=COLORS[1], lw=3.5, label="direct + local mode")
    ax.plot(Tf, term("D", d_, Tf), color=COLORS[1], ls=":", lw=2.5, label=r"  direct ($aT$)")
    ax.plot(Tf, term("L", l_, Tf), color=COLORS[1], ls="-.", lw=2.5, label=rf"  local mode ({l_[1]:.0f} cm$^{{-1}}$)")
    g = fit(MODELS["R+L"], Tx, R, sR); r_, l2 = g["pars"]
    ax.plot(Tf, term("R", r_, Tf) + term("L", l2, Tf), color=COLORS[0], ls="--", lw=2.5, label=rf"$bT^m$ ($m$={r_[1]:.2f}) + local mode")
    h = fit(MODELS["D+L(62.5)"], Tx, R, sR)
    ax.plot(Tf, term("D", h["pars"][0], Tf) + term("L", h["pars"][1], Tf), color="grey", lw=2, label=r"direct + local mode fixed at 62.5 cm$^{-1}$")
    ax.set_yscale("log"); ax.set_xlim(5, 65); ax.set_ylim(3, 5e3); ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)")
    ax.set_title(TITLE[ds], fontsize=30, fontweight="bold")
    if i == 0: H, LB = ax.get_legend_handles_labels()
    finish_panel(ax, "abc"[i])
    fr[ds] = (Tf, term("L", l_, Tf) / (term("D", d_, Tf) + term("L", l_, Tf)))
ax = axs[3]
for ds, col in zip(DATA, (COLORS[1], COLORS[0], COLORS[2])):
    ax.plot(fr[ds][0], 100 * fr[ds][1], color=col, lw=3.5, label=TITLE[ds])
ax.axhline(50, color="black", lw=1.2, ls="--"); ax.set_xlim(5, 65); ax.set_ylim(0, 100)
ax.set_xlabel("Temperature (K)"); ax.set_ylabel("Local-mode share (%)"); ax.legend(fontsize=24, loc="upper left"); finish_panel(ax, "d")
ax = axs[4]; prof = json.load(open(os.path.join(D1, "module1_T1_Delta_profile.json")))
for ds, col in zip(DATA, (COLORS[1], COLORS[0], COLORS[2])):
    c = np.array(prof[ds]["chi2"]); ax.plot(prof[ds]["Delta"], c - c.min(), color=col, lw=3.5, label=TITLE[ds])
for x, lab in ((50, "50"), (62.5, "62.5"), (101.5, "101.5")): ax.axvline(x, color="grey", ls=":", lw=2)
ax.set_yscale("symlog", linthresh=1); ax.set_ylim(0, 2e3); ax.set_xlim(20, 300)
ax.set_xlabel(r"Fixed local-mode energy $\Delta$ (cm$^{-1}$)"); ax.set_ylabel(r"$\Delta\chi^2$ (direct + local mode)"); ax.legend(fontsize=22, loc="upper right"); finish_panel(ax, "e")
ax = axs[5]; df = pd.read_csv(os.path.join(D1, "module1_T1_multiprocess.csv"), dtype={"Delta": str, "m": str})
order = ["D+L", "R+L", "D+R", "D+R+L", "D+L+L", "D+L(101.5)", "D+L(62.5)", "D+L(50)"]
for j, (ds, col) in enumerate(zip(DATA, (COLORS[1], COLORS[0], COLORS[2]))):
    x = df[(df.dataset == ds) & (df.fit == "weighted")].set_index("model")
    vals = [x.loc[m, "dBIC"] if m in x.index else np.nan for m in order]
    ax.barh(np.arange(len(order)) + (j - 1) * 0.27, np.clip(vals, 0.3, None), 0.27, color=col, edgecolor="black", label=TITLE[ds])
ax.set_xscale("log"); ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=25); ax.invert_yaxis()
ax.axvline(6, color="black", ls="--", lw=1.5); ax.set_xlabel(r"$\Delta$BIC to the best model"); finish_panel(ax, "f")   # colours as in panel d
fig.tight_layout(rect=(0, 0, 1, 0.91))
H = [H[-1]] + H[:-1]; LB = ["  " + l.strip() if l.startswith("  ") else l for l in [LB[-1]] + LB[:-1]]
fig.legend(H, LB, loc="upper center", ncol=3, fontsize=24, bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.4)
save_fig(fig, os.path.join(FSI, "FigS_T1_multiprocess"))

# LaTeX table rows (weighted fits)
L = []
NAMES = {"D+L": "direct + local mode", "R+L": r"$bT^m$ + local mode", "D+R": r"direct + Raman $bT^m$", "D+R+L": "direct + Raman + local mode",
         "D+L+L": "direct + two local modes", "D+L(101.5)": r"direct + local mode at 101.5 cm$^{-1}$", "D+L(62.5)": r"direct + local mode at 62.5 cm$^{-1}$",
         "D+L(50)": r"direct + local mode at 50 cm$^{-1}$", "L": "local mode only", "R": r"Raman $bT^m$ only"}
for m in ["D+L", "R+L", "D+R", "D+R+L", "D+L+L", "D+L(101.5)", "D+L(62.5)", "D+L(50)", "R", "L"]:
    cells = [NAMES[m]]
    for ds in DATA:
        x = df[(df.dataset == ds) & (df.fit == "weighted") & (df.model == m)]
        if len(x) == 0: cells += ["--", "--"]; continue
        x = x.iloc[0]; par = []
        if isinstance(x.Delta, str) and x.Delta: par.append(r"$\Delta$=" + x.Delta.replace(";", ", "))
        if isinstance(x.m, str) and x.m: par.append(r"$m$=" + x.m)
        cells += [f"{x.chi2:.1f} ({x.dBIC:.1f})", "; ".join(par) if par else "--"]
    L.append(" & ".join(cells) + r"\\")
open(os.path.join(TAB, "tabS_T1_multiprocess.tex"), "w").write("\n".join(L)); print("\n".join(L))
