#!/usr/bin/env python3
"""
m2_16_figures_classes.py -- Figure 3 (radical classes and spin environment) and
the corresponding SI figures.
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from plot_style import apply_style, finish_panel, save_fig, COLORS
from m2_00_io import ROOT
apply_style()
D = os.path.join(ROOT, "data", "module2"); FMS = os.path.join(ROOT, "Figures_MS"); FSI = os.path.join(ROOT, "Figures_ESI")
CN = ["S", "M", "R", "F"]
CLONG = {0: "S: surface-bound", 1: "M: mobile", 2: "R: restricted", 3: "F: frozen"}
CC = {0: "#7F7F7F", 1: COLORS[1], 2: COLORS[4], 3: COLORS[0]}
X = pd.read_csv(os.path.join(D, "module2_class_labels.csv"))
pop = pd.read_csv(os.path.join(D, "module2_class_populations_by_replica.csv"))
bath = pd.read_csv(os.path.join(D, "module2_spin_bath_summary.csv"), header=[0, 1], index_col=0)
bath = bath[bath.index.astype(str).isin(["100", "300"])]; bath.index = bath.index.astype(str)
B = lambda T, k, st: float(bath.loc[T, (k, st)])
ref = pd.read_csv(os.path.join(D, "module2_spin_bath_reference.csv"), index_col=0).iloc[:, 0]
dip = pd.read_csv(os.path.join(D, "module2_dipolar_by_replica.csv"))
cm = pd.read_csv(os.path.join(D, "module2_correlated_motion_by_replica.csv"))

fig, axs = plt.subplots(3, 2, figsize=(17, 19.5)); fig.subplots_adjust(hspace=0.45, wspace=0.48)

# (a) class map in descriptor space
ax = axs[0, 0]
for c in range(4):
    for T, mk in [(100, "o"), (300, "^")]:
        d = X[(X.cls == c) & (X["T"] == T)]
        ax.scatter(d.theta0, d.h_spin, s=36, marker=mk, color=CC[c], edgecolor="black", lw=0.4, alpha=0.85,
                   label=f"{CLONG[c].split(':')[0]}" if T == 100 else None)
ax.scatter([], [], marker="o", color="white", edgecolor="black", label="100 K")
ax.scatter([], [], marker="^", color="white", edgecolor="black", label="300 K")
ax.set_xlabel(r"Cone semi-angle $\theta_0$ (°)"); ax.set_ylabel("Spin height (Å)")
ax.set_xlim(0, 65); ax.set_ylim(0, 38); ax.legend(fontsize=17, loc="upper right", ncol=3, columnspacing=0.6, handletextpad=0.2, markerscale=1.8); finish_panel(ax, "a")

# (b) class populations
ax = axs[0, 1]; w = 0.36
for k, (T, col) in enumerate([(100, COLORS[0]), (300, COLORS[1])]):
    d = pop[pop["T"] == T][[str(c) for c in range(4)]] * 100
    ax.bar(np.arange(4) + (k - 0.5) * w, d.mean(), w, yerr=d.std(ddof=1), capsize=6, color=col,
           edgecolor="black", lw=1.5, label=f"{T} K", error_kw=dict(lw=2))
ax.set_xticks(range(4)); ax.set_xticklabels([CLONG[c].split(":")[0] for c in range(4)], fontsize=22)
ax.set_xlabel("Radical class"); ax.set_ylabel("Population (%)"); ax.set_ylim(0, 75); ax.legend(loc="upper left")
finish_panel(ax, "b")

# (c) 5-ns class transition matrix at 300 K (classes S, M, R)
ax = axs[1, 0]
P = np.load(os.path.join(D, "module2_class_transition_T300.npy")).mean(0)[:3, :3]
im = ax.imshow(P, cmap="Oranges", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=24,
                color="white" if P[i, j] > 0.6 else "black", fontweight="bold")
ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(CN[:3]); ax.set_yticklabels(CN[:3])
ax.set_xlabel("Class in second 5 ns"); ax.set_ylabel("Class in first 5 ns"); ax.set_title("300 K", fontsize=22)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("Probability", fontsize=20, labelpad=12)
finish_panel(ax, "c")

# (d) proton bath vs homogeneous solvent reference
ax = axs[1, 1]; shells = ["4–8 Å", "8–12 Å"]; xs = np.arange(2); w = 0.26
for k, (lab, T, col) in enumerate([("monolayer, 100 K", "100", COLORS[0]), ("monolayer, 300 K", "300", COLORS[1])]):
    mean = [B(T, "tot_4_8", "mean"), B(T, "tot_8_12", "mean")]
    sd = [B(T, "tot_4_8", "std"), B(T, "tot_8_12", "std")]
    nb = [B(T, "nb_4_8", "mean"), B(T, "nb_8_12", "mean")]
    ax.bar(xs + (k - 1) * w, mean, w, yerr=sd, capsize=6, color=col, edgecolor="black", lw=1.5, label=lab)
    ax.bar(xs + (k - 1) * w, nb, w, color="none", edgecolor="black", lw=0, hatch="//")
ax.bar(xs + w, [ref["N4_8"], ref["N8_12"]], w, color="white", edgecolor="black", lw=1.5, label="homogeneous solvent$^{a}$")
h, l = ax.get_legend_handles_labels()
h.append(Patch(facecolor="white", edgecolor="black", hatch="//")); l.append("from neighboring molecules")
ax.set_xticks(xs); ax.set_xticklabels(shells); ax.set_xlabel("Distance from spin center")
ax.set_ylabel("Protons per radical"); ax.set_ylim(0, 330); ax.legend(h, l, fontsize=17, loc="upper left", ncol=1)
finish_panel(ax, "d")

# (e) static vs fluctuating dipolar coupling
ax = axs[2, 0]
labels, xs = [], []
for k, (T, f) in enumerate([(100, "normal"), (100, "inplane"), (300, "normal"), (300, "inplane")]):
    d = dip[(dip["T"] == T) & (dip.field == f)]
    ax.bar(k - 0.18, d.sqrtM2_median_MHz.mean(), 0.36, yerr=d.sqrtM2_median_MHz.std(ddof=1), capsize=6,
           color=COLORS[0] if T == 100 else COLORS[1], edgecolor="black", lw=1.5)
    ax.bar(k + 0.18, d.sigma_d_median_MHz.mean(), 0.36, yerr=d.sigma_d_median_MHz.std(ddof=1), capsize=6,
           color=COLORS[0] if T == 100 else COLORS[1], edgecolor="black", lw=1.5, hatch="xx", alpha=0.55)
    labels.append(f"{T} K\n" + (r"$B_0\parallel n$" if f == "normal" else r"$B_0\perp n$"))
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=19)
ax.set_ylabel("Dipolar coupling (MHz)"); ax.set_ylim(0, 470)
ax.legend([Patch(facecolor="0.55", edgecolor="black"), Patch(facecolor="0.8", edgecolor="black", hatch="xx")],
          [r"local width $\sqrt{M_2}$", r"fluctuating part $\sigma_d$"], fontsize=17, loc="upper left", ncol=1)
finish_panel(ax, "e")

# (f) correlated motion vs distance
ax = axs[2, 1]
for T, col in [(100, COLORS[0]), (300, COLORS[1])]:
    d = cm[cm["T"] == T]; g = d.groupby(["r_lo", "r_hi"])
    rc = (g.r_lo.first() + np.minimum(g.r_hi.first(), 30)) / 2
    ax.errorbar(rc, g.corr_h.mean(), yerr=g.corr_h.std(ddof=1), fmt="-o", color=col, ms=9, capsize=5, lw=2.5,
                label=f"height, {T} K")
    ax.errorbar(rc, g.corr_cos.mean(), yerr=g.corr_cos.std(ddof=1), fmt="--s", color=col, ms=8, capsize=5, lw=2,
                mfc="white", label=f"orientation, {T} K")
ax.axhline(0, color="black", lw=1.5)
ax.set_xlabel("Pair distance (Å)"); ax.set_ylabel("Correlation coefficient"); ax.set_xlim(3, 30)
ax.set_ylim(-0.1, 0.4); ax.legend(fontsize=16, loc="upper right"); finish_panel(ax, "f")
save_fig(fig, os.path.join(FMS, "Figure3_radical_classes"))

# ---------------- SI: class portraits (violin plots) ---------------------------
feats = [("h_spin", "Spin height (Å)"), ("theta0", r"$\theta_0$ (°)"), ("cosNO", r"$\langle\cos\theta_\mathrm{NO}\rangle$"),
         ("n8", "Spins within 8 Å"), ("tilt", "Chain tilt (°)"), ("gauche", "Gauche fraction"),
         ("contact", "O–Au contact fraction"), ("edge_dist", "Distance to island edge (Å)")]
fig, axs = plt.subplots(2, 4, figsize=(17.5, 9.5)); fig.subplots_adjust(hspace=0.35, wspace=0.62)
for ax, (f, lab), L in zip(axs.flat, feats, "abcdefgh"):
    data = [X[X.cls == c][f].to_numpy() for c in range(4)]
    vp = ax.violinplot(data, positions=range(4), showmedians=True, widths=0.8)
    for c, b in enumerate(vp["bodies"]):
        b.set_facecolor(CC[c]); b.set_edgecolor("black"); b.set_alpha(0.8)
    for k in ("cmedians", "cbars", "cmins", "cmaxes"): vp[k].set_color("black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CN); ax.set_ylabel(lab, fontsize=21); finish_panel(ax, L)
save_fig(fig, os.path.join(FSI, "FigS_class_portraits"))

# ---------------- SI: transition matrices at both temperatures, dipolar ACF ----
acf = np.load(os.path.join(D, "module2_dipolar_acf.npz"))
fig, axs = plt.subplots(1, 3, figsize=(18, 6.5)); fig.subplots_adjust(wspace=0.45)
for ax, T, L in [(axs[0], 100, "a"), (axs[1], 300, "b")]:
    P = np.load(os.path.join(D, f"module2_class_transition_T{T}.npy")).mean(0)
    keep = [0, 1, 2, 3] if T == 100 else [0, 1, 2]
    P = P[np.ix_(keep, keep)]
    im = ax.imshow(P, cmap="Blues" if T == 100 else "Oranges", vmin=0, vmax=1)
    for i in range(len(keep)):
        for j in range(len(keep)):
            ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=18, color="white" if P[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(keep))); ax.set_yticks(range(len(keep)))
    ax.set_xticklabels([CN[k] for k in keep]); ax.set_yticklabels([CN[k] for k in keep])
    ax.set_xlabel("Class in second 5 ns"); ax.set_ylabel("Class in first 5 ns"); ax.set_title(f"{T} K", fontsize=22)
    finish_panel(ax, L)
ax = axs[2]
for T, col in [(100, COLORS[0]), (300, COLORS[1])]:
    for f, ls in [("normal", "-"), ("inplane", "--")]:
        Y = np.array([acf[f"T{T}_r{r}_{f}"] for r in (1, 2, 3)])
        ax.plot(acf["lag_ns"], Y.mean(0), ls, color=col, lw=2.5,
                label=f"{T} K, " + (r"$B_0\parallel n$" if f == "normal" else r"$B_0\perp n$"))
        ax.fill_between(acf["lag_ns"], Y.mean(0) - Y.std(0, ddof=1), Y.mean(0) + Y.std(0, ddof=1), color=col, alpha=0.2, lw=0)
ax.axhline(0, color="black", lw=1.2); ax.set_xlim(0, 1.0)
ax.set_xlabel("Lag time (ns)"); ax.set_ylabel(r"$C_d(t)$"); ax.legend(fontsize=16); finish_panel(ax, "c")
save_fig(fig, os.path.join(FSI, "FigS_class_transitions_dipolar"))
print("class figures written")
