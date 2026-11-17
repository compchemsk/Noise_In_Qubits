#!/usr/bin/env python3
"""toc_graphic.py -- Table-of-contents graphic (3.25 x 1.75 in, the JACS TOC size).

Left: top view of the simulated TEMPO monolayer (replica 1, 300 K). Dots are radicals;
lines are resolved exchange couplings (red antiferromagnetic,
blue ferromagnetic expected sign), drawn from the machine-learned network.
Right: measured 1/T1 of PTEMPO (slow component, Tesi et al. 2023) with the fitted
direct and local-mode contributions (direct + one local mode, weighted fit).
All data are read from the outputs of the analysis pipeline.
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from plot_style import COLORS
import plot_style  # noqa: F401  (applies frameless legends)
from m1_13_multiprocess_T1 import DATA, T as TK, fit, term, MODELS

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D2 = os.path.join(ROOT, "data", "module2"); D3 = os.path.join(ROOT, "data", "module3")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5, "axes.linewidth": 1.0,
                     "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "xtick.direction": "in", "ytick.direction": "in",
                     "mathtext.default": "regular", "pdf.fonttype": 42, "savefig.dpi": 600})

T, rep = 300, 1
pos = np.load(os.path.join(D2, f"obs_T{T}_r{rep}.npz"))["spin_yz"][-500:].mean(0)
Mx = np.load(os.path.join(D3, "exchange_matrices.npz")); intact = Mx["intact"]; mol = Mx["mol_ids"]
P = pd.read_csv(os.path.join(D3, f"exchange_pairs_T{T}_r{rep}.csv")); P = P[P.intact & (P.Jabs_mean > 0.10)]

fig = plt.figure(figsize=(3.25, 1.75))
# ---------------- left: exchange clusters on gold
ax = fig.add_axes([0.0, 0.02, 0.50, 0.90])
c = pos.mean(0); p = pos - c
R = np.abs(p).max() + 5
ax.add_patch(plt.Rectangle((-R, -R), 2 * R, 2 * R, color="#FFD54A", zorder=0))
for _, q in P.iterrows():
    a, b = int(q.i), int(q.j)
    col = "#E8001C" if q.EJ_mean < 0 else "#0050FF"
    ax.plot(p[[a, b], 0], p[[a, b], 1], color=col, lw=0.8 + 1.4 * np.log10(q.Jabs_mean / 0.1), solid_capstyle="round", zorder=2)
for k in range(len(p)):
    if intact[k]:
        ax.scatter(*p[k], s=7, color="white", edgecolor="black", lw=0.4, zorder=3)
ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.axis("off")
ax.text(0.5, 1.02, "Exchange clusters on gold", transform=ax.transAxes, ha="center", va="bottom",
        fontsize=7.5, fontweight="bold")
ax.plot([], [], color="#E8001C", lw=1.6, label="AF"); ax.plot([], [], color="#0050FF", lw=1.6, label="F")
ax.legend(loc="lower left", fontsize=6.5, handlelength=1.0, borderaxespad=0.1, ncol=2, columnspacing=0.6)

# ---------------- right: what drives T1
ax = fig.add_axes([0.655, 0.24, 0.33, 0.60])
t1, e = DATA["PTEMPO_slow"]; Rr = 1e3 / t1; sR = Rr * e / t1
f = fit(MODELS["D+L"], TK, Rr, sR); Tf = np.linspace(5, 65, 300)
dpart = term("D", f["pars"][0], Tf); lpart = term("L", f["pars"][1], Tf)
ax.plot(Tf, dpart + lpart, color=COLORS[1], lw=1.6)
ax.plot(Tf, dpart, color=COLORS[0], lw=1.3, ls=":")
ax.plot(Tf, lpart, color=COLORS[3], lw=1.3, ls="-.")
ax.errorbar(TK, Rr, yerr=sR, fmt="o", ms=2.8, color="black", capsize=0, lw=0.8)
ax.set_yscale("log"); ax.set_xlim(5, 65); ax.set_ylim(5, 3e3)
ax.set_xlabel("T (K)", labelpad=1); ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)", labelpad=1)
ax.text(8, 60, "direct", color=COLORS[0], fontsize=7, fontweight="bold")
ax.text(32, 4e2 * 0.035, "local\nmode", color=COLORS[3], fontsize=7, fontweight="bold", ha="left")
ax.text(0.40, 1.04, r"Two processes set $T_1$", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5, fontweight="bold")
out = os.path.join(ROOT, "Figures_MS", "TOC_graphic")
fig.savefig(out + ".png", dpi=600); fig.savefig(out + ".pdf")   # exact 3.25 x 1.75 in
print("TOC graphic written; Delta =", round(f["pars"][1][1], 1))
