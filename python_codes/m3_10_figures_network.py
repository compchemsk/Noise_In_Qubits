#!/usr/bin/env python3
"""m3_10_figures_network.py -- Figure 5 (exchange network of the monolayer) and the network SI figures."""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_style import apply_style, finish_panel, save_fig, COLORS
from m3_ml_common import D
from m2_00_io import DEFECT_IDS
CLOSE = set(DEFECT_IDS)
apply_style()
ROOT = os.path.join(D, "..", ".."); FMS = os.path.join(ROOT, "Figures_MS"); FSI = os.path.join(ROOT, "Figures_ESI")
CM2K = 1.438777; J_RES = 0.10
CC = {"S": "#7F7F7F", "M": COLORS[1], "R": COLORS[4], "F": COLORS[0]}
Mx = np.load(os.path.join(D, "exchange_matrices.npz")); intact = Mx["intact"]; nm = len(intact)
cmap = plt.get_cmap("RdBu_r"); norm = mcolors.SymLogNorm(linthresh=0.03, linscale=0.4, vmin=-1, vmax=1, base=10)
S = pd.read_csv(os.path.join(D, "exchange_summary_by_replica.csv"))
lab = pd.read_csv(os.path.join(D, "..", "module2", "module2_class_labels.csv"))

def positions(T, r):
    o = np.load(os.path.join(D, "..", "module2", f"obs_T{T}_r{r}.npz")); return o["spin_yz"][-500:].mean(0)
# spectral ordering on the 100 K replica-1 positions: neighbours end up close to the diagonal
P0 = positions(100, 1); dd = np.linalg.norm(P0[:, None] - P0[None], axis=2)
W = np.exp(-(dd / 6.0) ** 2); Lp = np.diag(W.sum(1)) - W
order = np.argsort(np.linalg.eigh(Lp)[1][:, 1])

def matrix_panel(ax, up, low, title_up, title_low, cbar=True, near=None):
    """141 x 141 grid: upper triangle = `up`, lower triangle = `low`.  Resolved couplings
    (|<J>| > 0.1 cm^-1 in time-averaged |J|) are drawn as enlarged squares coloured by <J>;
    pairs that come within the 9 A cutoff but stay unresolved are small grey dots."""
    rank = np.empty(nm, int); rank[order] = np.arange(nm)
    ax.set_xlim(-1, nm); ax.set_ylim(nm, -1); ax.set_aspect("equal")
    for k in np.where(~intact)[0]:
        ax.axhspan(rank[k] - 0.5, rank[k] + 0.5, color="#DDDDDD", lw=0, zorder=0)
        ax.axvspan(rank[k] - 0.5, rank[k] + 0.5, color="#DDDDDD", lw=0, zorder=0)
    ax.plot([-1, nm], [-1, nm], color="black", lw=1.2, zorder=1)
    for M_, tri in ((up, "u"), (low, "l")):
        i, j = np.where(np.triu(np.ones((nm, nm), bool), 1))
        v = M_[i, j]; ok = intact[i] & intact[j]
        ri, rj = rank[i], rank[j]
        a, b = (np.minimum(ri, rj), np.maximum(ri, rj)) if tri == "u" else (np.maximum(ri, rj), np.minimum(ri, rj))
        dots = ok & (v != 0) & (np.abs(v) <= J_RES * 0.6)
        ax.scatter(b[dots], a[dots], s=3, color="#9A9A9A", lw=0, zorder=2)
        big = ok & (np.abs(v) > J_RES * 0.6)
        o = np.argsort(np.abs(v[big]))
        sc = ax.scatter(b[big][o], a[big][o], s=34, marker="s", c=v[big][o], cmap=cmap, norm=norm, edgecolor="black", lw=0.4, zorder=3)
    ax.text(0.97, 0.04, title_low, transform=ax.transAxes, ha="right", fontsize=20, fontweight="bold", bbox=dict(fc="white", ec="none"))
    ax.text(0.03, 0.96, title_up, transform=ax.transAxes, ha="left", va="top", fontsize=20, fontweight="bold", bbox=dict(fc="white", ec="none"))
    ax.set_xlabel("Radical (spatial order)"); ax.set_ylabel("Radical (spatial order)")
    ax.set_xticks([0, 40, 80, 120]); ax.set_yticks([0, 40, 80, 120])
    if cbar:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03, ticks=[-1, -0.3, -0.1, -0.03, 0, 0.03, 0.1, 0.3, 1])
        cb.ax.set_yticklabels(["−1", "−0.3", "−0.1", "−0.03", "0", "0.03", "0.1", "0.3", "1"], fontsize=17)
        cb.set_label(r"$\langle J\rangle$ (cm$^{-1}$)", fontsize=21)
    return sc

def network_panel(ax, T, r, title):
    pos = positions(T, r); P = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv")); P = P[P.intact & (P.Jabs_mean > J_RES)]
    L = lab[(lab["T"] == T) & (lab.rep == r)].set_index("mol").cls
    mol = Mx["mol_ids"]
    for _, q in P.sort_values("Jabs_mean").iterrows():
        a, b = int(q.i), int(q.j)
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color=cmap(norm(q.EJ_mean)), lw=2.0 + 4.0 * np.log10(q.Jabs_mean / J_RES),
                solid_capstyle="round", zorder=2)
    cn = {0: "S", 1: "M", 2: "R", 3: "F"}
    for k in range(nm):
        if intact[k]:
            ax.scatter(*pos[k], s=70, color=CC[cn[int(L[mol[k]])]], edgecolor="black", lw=0.8, zorder=3)
            if mol[k] in CLOSE:   # one of the 7 close pairs (rings interpenetrate in the start structure)
                ax.scatter(*pos[k], s=260, facecolor="none", edgecolor="black", lw=1.8, zorder=4)
        else:
            ax.scatter(*pos[k], s=50, marker="x", color="black", lw=1.5, zorder=3)
    ax.set_aspect("equal"); ax.set_xlabel("In-plane x (Å)"); ax.set_ylabel("In-plane y (Å)")
    ax.set_title(title, fontsize=22)

fig = plt.figure(figsize=(21, 14.5))
gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.62)
ax = fig.add_subplot(gs[0, 0])
matrix_panel(ax, Mx["EJ_T100_mean"], Mx["EJ_T300_mean"], "100 K", "300 K"); finish_panel(ax, "a")
ax = fig.add_subplot(gs[0, 1]); network_panel(ax, 100, 1, "100 K, replica 1"); finish_panel(ax, "b")
ax = fig.add_subplot(gs[0, 2]); network_panel(ax, 300, 1, "300 K, replica 1"); finish_panel(ax, "c")
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=10, mfc=CC[c], mec="black", label=n) for c, n in
                   [("F", "frozen"), ("R", "restricted"), ("M", "mobile"), ("S", "surface-bound")]] +
                  [Line2D([], [], marker="o", ls="", ms=16, mfc="none", mec="black", mew=1.8, label="close pair"),
                   Line2D([], [], color=cmap(norm(-0.3)), lw=4, label=r"$\langle J\rangle<0$"), Line2D([], [], color=cmap(norm(0.3)), lw=4, label=r"$\langle J\rangle>0$")],
          fontsize=16, loc="lower left", bbox_to_anchor=(1.01, 0.0), frameon=False)
# (d) distribution of resolved couplings + experimental scale
ax = fig.add_subplot(gs[1, 0])
bins = np.logspace(-1, 3, 33); cen = np.sqrt(bins[1:] * bins[:-1])
for T, c in [(100, COLORS[0]), (300, COLORS[1])]:
    H = []
    for r in (1, 2, 3):
        P = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv")); P = P[P.intact & (P.Jabs_mean > J_RES)]
        H.append(np.histogram(P.Jabs_mean, bins)[0])
    H = np.array(H, float); m, s = H.mean(0), H.std(0)
    ax.step(cen, m, where="mid", color=c, lw=3, label=f"{T} K"); ax.fill_between(cen, m - s, m + s, step="mid", color=c, alpha=0.25, lw=0)
ax.axvspan(100 / CM2K / 2, 600 / CM2K / 2, color=COLORS[2], alpha=0.25, lw=0)
ax.text(np.sqrt(100 * 600) / CM2K / 2, 0.60 * ax.get_ylim()[1], "$\\chi(T)$ model fits", ha="center", va="center", fontsize=17, rotation=90)
ax.axvline(35.0e3 / 29979.2458 / 2, color="black", ls=":", lw=2)
ax.annotate("Q band: $2|J|=h\\nu$", xy=(35.0e3 / 29979.2458 / 2, 0.86 * ax.get_ylim()[1]), xytext=(2.0, 0.86 * ax.get_ylim()[1]),
            fontsize=17, va="center", arrowprops=dict(arrowstyle="->", lw=1.5))
ax.set_xscale("log"); ax.set_xlim(0.1, 1000)
ax.set_xlabel(r"Time-averaged $|J|$ (cm$^{-1}$)"); ax.set_ylabel("Resolved pairs per replica")
sec = ax.secondary_xaxis("top", functions=(lambda x: x * CM2K, lambda x: x / CM2K)); sec.set_xlabel(r"$|J|/k_\mathrm{B}$ (K)", fontsize=21)
ax.annotate("7 close pairs\n(outside the\ntraining domain)", xy=(6.0, 6.4), xytext=(6.0, 12.0), fontsize=15,
            ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=1.5))
ax.legend(fontsize=17, loc="center right", bbox_to_anchor=(0.99, 0.24)); finish_panel(ax, "d")
# (e) connectivity of the spins and AF fraction
ax = fig.add_subplot(gs[1, 1]); w = 0.36
for k, (T, c) in enumerate([(100, COLORS[0]), (300, COLORS[1])]):
    fr = []
    for r in (1, 2, 3):
        X = pd.read_csv(os.path.join(D, f"exchange_spins_T{T}_r{r}.csv"))
        fr.append([(X.degree == 0).mean(), (X.degree == 1).mean(), (X.degree >= 2).mean()])
    fr = 100 * np.array(fr)
    ax.bar(np.arange(3) + (k - 0.5) * w, fr.mean(0), w, yerr=fr.std(0), color=c, edgecolor="black", lw=1.5, capsize=6, label=f"{T} K", error_kw=dict(lw=2))
ax.set_xticks(range(3)); ax.set_xticklabels(["0", "1", "≥ 2"]); ax.set_xlabel(r"Partners with $|J|>0.1$ cm$^{-1}$")
ax.set_ylabel("Radicals (%)"); ax.set_ylim(0, 85); ax.legend(fontsize=17, loc="upper right")
t = S.groupby("T").agg(["mean", "std"])
ax.text(0.03, 0.97, "AF share\n" + "\n".join(
    f"{T} K: {100*t.loc[T, ('frac_AF_res','mean')]:.0f} ± {100*t.loc[T, ('frac_AF_res','std')]:.0f}%" for T in (100, 300)),
    transform=ax.transAxes, va="top", ha="left", fontsize=16, bbox=dict(fc="white", ec="none")); finish_panel(ax, "e")
# (f) by class
ax = fig.add_subplot(gs[1, 2])
C = pd.read_csv(os.path.join(D, "exchange_by_class_by_replica.csv"))
cls = ["F", "R", "M", "S"]; w = 0.36
for k, (T, c) in enumerate([(100, COLORS[0]), (300, COLORS[1])]):
    g = C[C["T"] == T].groupby("cls").degree.agg(["mean", "std"]).reindex(cls)
    nrad = C[C["T"] == T].groupby("cls").n.mean().reindex(cls)          # radicals per replica in each class
    g.loc[nrad < 3] = np.nan                                            # a class with < 3 radicals is not shown
    ax.bar(np.arange(4) + (k - 0.5) * w, g["mean"], w, yerr=g["std"], color=c, edgecolor="black", lw=1.5, capsize=6, label=f"{T} K", error_kw=dict(lw=2))
ax.set_xticks(range(4)); ax.set_xticklabels(["frozen", "restricted", "mobile", "surface-bound"], fontsize=17, rotation=25, ha="right")
ax.set_ylabel("Resolved partners per radical"); ax.set_ylim(0, 1.6); ax.legend(fontsize=17, loc="upper right")
ax.text(0 + 0.5 * w, 0.04, "about 1%", rotation=90, ha="center", va="bottom", fontsize=17)
finish_panel(ax, "f")
save_fig(fig, os.path.join(FMS, "Figure5_exchange_network"))

import matplotlib as mpl
mpl.rcParams["savefig.dpi"] = 300          # large multi-panel SI figures
# ================= SI: all six replica matrices (symmetric <J>), one shared colour bar
fig, axs = plt.subplots(2, 3, figsize=(19, 13.5)); fig.subplots_adjust(hspace=0.30, wspace=0.30, right=0.86)
for k, (a, (T, r)) in enumerate(zip(axs.flat, [(100, 1), (100, 2), (100, 3), (300, 1), (300, 2), (300, 3)])):
    Pp = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv"))
    M = np.zeros((nm, nm)); M[Pp.i, Pp.j] = Pp.EJ_mean; M[Pp.j, Pp.i] = Pp.EJ_mean
    sc = matrix_panel(a, M, M, f"{T} K", f"replica {r}", cbar=False)
    if k % 3: a.set_ylabel("")
    if k < 3: a.set_xlabel("")
cax = fig.add_axes([0.88, 0.25, 0.018, 0.5])
cb = fig.colorbar(sc, cax=cax, ticks=[-1, -0.3, -0.1, -0.03, 0, 0.03, 0.1, 0.3, 1])
cb.ax.set_yticklabels(["−1", "−0.3", "−0.1", "−0.03", "0", "0.03", "0.1", "0.3", "1"]); cb.set_label(r"$\langle J\rangle$ (cm$^{-1}$)")
save_fig(fig, os.path.join(FSI, "FigS_exchange_matrices_replicas"))

# ================= SI: replica-mean |J| (upper triangle) and replica SD (lower triangle)
fig, axs = plt.subplots(1, 2, figsize=(17, 8)); fig.subplots_adjust(wspace=0.25, right=0.86)
rank = np.empty(nm, int); rank[order] = np.arange(nm)
vnorm = mcolors.LogNorm(0.03, 5)
for a, T in zip(axs, (100, 300)):
    m, sd = Mx[f"Jabs_T{T}_mean"], Mx[f"Jabs_T{T}_sd"]
    i, j = np.where(np.triu(np.ones((nm, nm), bool), 1))
    ok = intact[i] & intact[j] & (m[i, j] > J_RES)
    ri, rj = rank[i[ok]], rank[j[ok]]
    a.set_xlim(-1, nm); a.set_ylim(nm, -1); a.set_aspect("equal"); a.set_facecolor("#F2F2F2")
    sc = a.scatter(np.maximum(ri, rj), np.minimum(ri, rj), c=m[i[ok], j[ok]], s=40, marker="s", cmap="viridis", norm=vnorm, edgecolor="black", lw=0.3)
    a.scatter(np.minimum(ri, rj), np.maximum(ri, rj), c=np.maximum(sd[i[ok], j[ok]], 0.031), s=40, marker="s", cmap="viridis", norm=vnorm, edgecolor="black", lw=0.3)
    a.plot([-1, nm], [-1, nm], "k-", lw=1)
    a.text(0.03, 0.96, r"mean $\langle|J|\rangle$", transform=a.transAxes, va="top", fontsize=18, bbox=dict(fc="white", ec="none"))
    a.text(0.97, 0.04, "SD over replicas", transform=a.transAxes, ha="right", fontsize=18, bbox=dict(fc="white", ec="none"))
    a.set_title(f"{T} K"); a.set_xlabel("Radical (spatial order)"); a.set_xticks([0, 40, 80, 120]); a.set_yticks([0, 40, 80, 120])
    finish_panel(a, "ab"[T == 300])
axs[0].set_ylabel("Radical (spatial order)")
cax = fig.add_axes([0.88, 0.2, 0.018, 0.6]); cb = fig.colorbar(sc, cax=cax); cb.set_label(r"cm$^{-1}$")
save_fig(fig, os.path.join(FSI, "FigS_exchange_matrix_mean_sd"))

# ================= SI: network maps of all replicas
fig, axs = plt.subplots(2, 3, figsize=(20, 12.5)); fig.subplots_adjust(hspace=0.30, wspace=0.35)
for a, (T, r) in zip(axs.flat, [(100, 1), (100, 2), (100, 3), (300, 1), (300, 2), (300, 3)]):
    network_panel(a, T, r, f"{T} K, replica {r}"); finish_panel(a)
save_fig(fig, os.path.join(FSI, "FigS_exchange_network_replicas"))

# ================= SI: dynamics of exchange
fig, axs = plt.subplots(1, 3, figsize=(20, 6.8)); fig.subplots_adjust(wspace=0.40)
t = np.arange(500) * 0.01
for a, T in zip(axs[:2], (100, 300)):
    z = np.load(os.path.join(D, f"mdJ_T{T}_r1.npz")); Pp = pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r1.csv"))
    sel = Pp[Pp.intact & (Pp.frac_res > 0.5)].sort_values("Jabs_mean", ascending=False).index[[0, 3, 8, 15]]
    for k, c in zip(sel, COLORS):
        a.plot(t, z["EJ"][:, k], color=c, lw=1.2, label=f"pair {Pp.mol_i[k]}–{Pp.mol_j[k]}")
    a.set_yscale("symlog", linthresh=0.1); a.axhspan(-0.1, 0.1, color="#BBBBBB", alpha=0.4, lw=0)
    a.set_xlabel("Time in analysis window (ns)"); a.set_ylabel(r"$|J|(1-2P_\mathrm{AF})$ (cm$^{-1}$)"); a.set_title(f"{T} K, replica 1")
    finish_panel(a, "ab"[T == 300])
a = axs[2]
for T, c in [(100, COLORS[0]), (300, COLORS[1])]:
    xi = pd.concat([pd.read_csv(os.path.join(D, f"exchange_pairs_T{T}_r{r}.csv")) for r in (1, 2, 3)])
    xi = xi[xi.intact].xi_J.dropna()
    a.hist(np.log10(xi), bins=np.linspace(-2, 3, 26), histtype="step", lw=3, color=c, density=True, label=f"{T} K (n = {len(xi)})")
a.axvline(0, color="black", ls=":", lw=2)
a.set_xlabel(r"$\log_{10}\xi_J$,  $\xi_J=2\pi\,\sigma_{2J}\,\tau_J$"); a.set_ylabel("Density"); a.set_ylim(top=a.get_ylim()[1] * 1.3); a.legend(fontsize=18, loc="upper right"); finish_panel(a, "c")
save_fig(fig, os.path.join(FSI, "FigS_exchange_dynamics"))
print("network figures written")
