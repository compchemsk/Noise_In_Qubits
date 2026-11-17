#!/usr/bin/env python3
"""
m2_09_figures.py -- Module II figures (main text Figure 2 + SI figures).
Distributions are computed per replica; lines show the replica mean and
shaded bands +/- 1 SD over the three replicas.
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import ConvexHull
from plot_style import apply_style, finish_panel, save_fig, COLORS
from m2_00_io import *
apply_style()
D = os.path.join(ROOT, "data", "module2"); FMS = os.path.join(ROOT, "Figures_MS"); FSI = os.path.join(ROOT, "Figures_ESI")
from m2_00_io import DEFECT, DEFECT_IDS   # DEFECT is empty by default (all molecules analysed); DEFECT_IDS only marks the 7 close pairs
W = slice(500, 1000)
CT = {300: COLORS[1], 100: COLORS[0]}
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
names, res, lin = P["name"], P["resid"], P["resname"] == "LIN"

def groups(o):
    S = o["S_yz"].mean(0); hull = ConvexHull(S)
    dist = np.min(-(S @ hull.equations[:, :2].T + hull.equations[:, 2]), axis=1)
    intact = ~np.isin(o["mol_ids"], DEFECT)
    return dist >= 5, intact, dist

def closepair(o):
    return np.isin(o["mol_ids"], DEFECT_IDS)

def band(ax, x, Y, color, label, ls="-"):
    Y = np.array(Y); m, s = Y.mean(0), Y.std(0, ddof=1)
    ax.plot(x, m, ls, color=color, lw=3, label=label); ax.fill_between(x, m - s, m + s, color=color, alpha=0.25, lw=0)

OBS = {(T, r): np.load(os.path.join(D, f"obs_T{T}_r{r}.npz")) for T, r in RUNS}

# =============================================================================
# Figure 2
# =============================================================================
fig = plt.figure(figsize=(17, 16.5))
gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.30, height_ratios=[0.55, 1, 1])

# (a) side views of the last frame of replica 1 at both temperatures
ax = fig.add_subplot(gs[0, :])
chain_names = ["S1"] + [f"C{k}" for k in range(23, 12, -1)] + ["N4", "C11", "C10", "O2", "C1", "N1", "O1"]
for k, T in enumerate([100, 300]):
    X = load_traj(run_prefix(T, 1), P["natom"])[-1]
    yoff = (-62 if T == 100 else 62) - X[~lin, 1].mean()
    o = OBS[(T, 1)]; interior, intact, _ = groups(o)
    xtop = X[(~lin) & (X[:, 0] < 0.75)].mean(0)[0] if False else X[~lin][X[~lin][:, 0] < 0.75, 0].mean()
    au = X[~lin]; ax.scatter(au[:, 1] + yoff, xtop - au[:, 0], s=4, color="#D4A017", zorder=1)
    segs = []
    for mi, mid in enumerate(o["mol_ids"]):
        pts = np.array([X[np.where((res == mid) & (names == n))[0][0]] for n in chain_names])
        segs.append(np.c_[pts[:, 1] + yoff, xtop - pts[:, 0]])
    ax.add_collection(LineCollection(segs, colors="0.45", lw=1.0, zorder=2))
    iO_ = np.where(lin & (names == "O1"))[0]
    hO = xtop - X[iO_, 0]
    ax.scatter(X[iO_, 1] + yoff, hO, s=26, color=COLORS[3], edgecolor="black", lw=0.4, zorder=3)
    ax.text(yoff + X[~lin, 1].mean(), 30, f"{T} K", ha="center", fontsize=24, fontweight="bold")
ax.set_xlim(-118, 118); ax.set_ylim(-4, 35); ax.set_aspect("equal")
ax.set_xlabel("Lateral position (Å)"); ax.set_ylabel("Height (Å)")
ax.set_xticks([-102, -62, -22, 22, 62, 102]); ax.set_xticklabels(["−40", "0", "40", "−40", "0", "40"])
finish_panel(ax, "a")

# (b) spin-height distributions: interior and edge
ax = fig.add_subplot(gs[1, 0]); bins = np.arange(-2, 32, 1.0); xc = 0.5 * (bins[1:] + bins[:-1])
for T in (100, 300):
    for grp, ls, lab in [(True, "-", "interior"), (False, "--", "edge")]:
        Y = []
        for r in (1, 2, 3):
            o = OBS[(T, r)]; interior, intact, _ = groups(o); m = (interior == grp) & intact
            Y.append(np.histogram(o["h_spin"][W][:, m].ravel(), bins, density=True)[0])
        band(ax, xc, Y, CT[T], f"{T} K, {lab}", ls)
ax.set_xlabel("Height of N–O midpoint (Å)"); ax.set_ylabel("Probability density")
ax.set_xlim(-1, 30); ax.set_ylim(top=0.62); ax.legend(loc="upper right", fontsize=16, ncol=2, columnspacing=0.6, handlelength=1.6); finish_panel(ax, "b")

# (c) N-O orientation (interior, intact)
ax = fig.add_subplot(gs[1, 1]); bins = np.linspace(-1, 1, 21); xc = 0.5 * (bins[1:] + bins[:-1])
for T in (100, 300):
    Y = []
    for r in (1, 2, 3):
        o = OBS[(T, r)]; interior, intact, _ = groups(o); m = interior & intact
        Y.append(np.histogram(o["cos_NO"][W][:, m].ravel(), bins, density=True)[0])
    band(ax, xc, Y, CT[T], f"{T} K")
ax.axhline(0.5, color="black", ls=":", lw=2, label="isotropic")
ax.set_xlabel(r"$\cos\theta_\mathrm{NO}$ (N→O vs surface normal)"); ax.set_ylabel("Probability density")
ax.legend(loc="upper left", fontsize=18); finish_panel(ax, "c")

# (d) cone semi-angle distribution
dyn = pd.read_csv(os.path.join(D, "module2_dynamics_by_molecule.csv"))
ax = fig.add_subplot(gs[2, 0]); bins = np.arange(0, 72, 3); xc = 0.5 * (bins[1:] + bins[:-1])
for T in (100, 300):
    Y = [np.histogram(dyn[(dyn["T"] == T) & (dyn.rep == r) & dyn.interior & dyn.intact].theta0, bins, density=True)[0]
         for r in (1, 2, 3)]
    band(ax, xc, Y, CT[T], f"{T} K")
ax.set_xlabel(r"Cone semi-angle $\theta_0$ (°)"); ax.set_ylabel("Probability density")
ax.set_xlim(0, 70); ax.legend(loc="upper right"); finish_panel(ax, "d")

# (e) pair-distance distribution and point-dipole coupling scale
Hs = np.load(os.path.join(D, "module2_pair_histograms.npz"))
ax = fig.add_subplot(gs[2, 1]); rr = Hs["r"]
for T in (100, 300):
    band(ax, rr, [Hs[f"T{T}_r{r}_spin"] / 0.1 / (141 - len(DEFECT)) for r in (1, 2, 3)], CT[T], f"{T} K")
ax.set_xlim(2.5, 12); ax.set_xlabel("Spin–spin distance $r$ (Å)")
ax.set_ylabel(r"Pairs per radical per Å")
sec = ax.secondary_xaxis("top", functions=(lambda r: 52.04 / (np.maximum(r, 0.1) / 10) ** 3,
                                           lambda nu: 10 * (52.04 / np.maximum(nu, 1e-3)) ** (1 / 3)))
sec.set_xticks([500, 200, 100, 50, 30]); sec.set_xlabel(r"Point-dipole coupling $\nu_\mathrm{dd}$ (MHz)", fontsize=20)
sec.tick_params(labelsize=18)
ax.legend(loc="upper left"); finish_panel(ax, "e")
save_fig(fig, os.path.join(FMS, "Figure2_monolayer_MD"))

# =============================================================================
# SI figures
# =============================================================================
# S-a: RMSD + block time series
rm = np.load(os.path.join(D, "module2_rmsd.npz")); bt = pd.read_csv(os.path.join(D, "module2_block_timeseries.csv"))
fig, axs = plt.subplots(2, 2, figsize=(14, 11)); fig.subplots_adjust(hspace=0.42, wspace=0.38)
ls = {1: "-", 2: "--", 3: ":"}
ax = axs[0, 0]
for T, r in RUNS:
    ax.plot(rm["t_ns"][100:], rm[f"rmsd_T{T}_r{r}"][100:], ls[r], color=CT[T], lw=2, label=f"{T} K, rep {r}")
ax.axvspan(5, 10, color="0.85", zorder=0); ax.set_xlabel("Time (ns)"); ax.set_ylabel("Heavy-atom RMSD (Å)")
ax.legend(fontsize=15, ncol=2, loc="lower right"); finish_panel(ax, "a")
for ax, col, lab, L in [(axs[0, 1], "h_spin", "Mean spin height (Å)", "b"), (axs[1, 0], "tilt", "Mean chain tilt (°)", "c"),
                        (axs[1, 1], "n_contact", "Au-contacting radicals", "d")]:
    for T, r in RUNS:
        d = bt[(bt["T"] == T) & (bt.rep == r)]; ax.plot(d.t_ns, d[col], ls[r], marker="o", ms=5, color=CT[T], lw=2)
    ax.axvspan(5, 10, color="0.85", zorder=0); ax.set_xlabel("Time (ns)"); ax.set_ylabel(lab); finish_panel(ax, L)
save_fig(fig, os.path.join(FSI, "FigS_MD_convergence"))

# S-b: top view of the island, edge/interior, defects, Au contacts (replica 1, 300 K)
fig, axs = plt.subplots(1, 2, figsize=(14, 7.2)); fig.subplots_adjust(wspace=0.30)
for ax, T, L in zip(axs, (100, 300), "ab"):
    o = OBS[(T, 1)]; interior, intact, dist = groups(o); S = o["S_yz"].mean(0)
    cont = (o["dmin_OAu"][W] < 3.5).mean(0) > 0.5
    ax.scatter(S[interior, 0], S[interior, 1], s=90, color=COLORS[0], edgecolor="black", label="interior")
    ax.scatter(S[~interior, 0], S[~interior, 1], s=90, color="white", edgecolor="black", label="edge")
    ax.scatter(S[cont, 0], S[cont, 1], s=30, color=COLORS[1], label="O–Au contact")
    cp = closepair(o); ax.scatter(S[cp, 0], S[cp, 1], s=220, marker="x", color="black", lw=2.5, label="close pair (rings interpenetrate)")
    ax.set_aspect("equal"); ax.set_xlabel("In-plane x (Å)"); ax.set_ylabel("In-plane y (Å)"); ax.set_title(f"{T} K, replica 1", fontsize=22)
    finish_panel(ax, L)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=18, bbox_to_anchor=(0.5, -0.04), markerscale=1.3)
save_fig(fig, os.path.join(FSI, "FigS_MD_island_map"))

# S-c: tilt and gauche distributions (interior intact)
fig, axs = plt.subplots(1, 2, figsize=(13, 5.8)); fig.subplots_adjust(wspace=0.35)
for ax, key, bins, lab, L in [(axs[0], "tilt", np.arange(0, 92, 4), "Chain tilt (°)", "a"),
                              (axs[1], "gauche", np.linspace(0, 1, 10), "Gauche fraction per chain", "b")]:
    xc = 0.5 * (bins[1:] + bins[:-1])
    for T in (100, 300):
        Y = []
        for r in (1, 2, 3):
            o = OBS[(T, r)]; interior, intact, _ = groups(o)
            Y.append(np.histogram(o[key][W][:, interior & intact].ravel(), bins, density=True)[0])
        band(ax, xc, Y, CT[T], f"{T} K")
    ax.set_xlabel(lab); ax.set_ylabel("Probability density"); ax.legend(); finish_panel(ax, L)
save_fig(fig, os.path.join(FSI, "FigS_MD_tilt_gauche"))

# S-d: reorientation C2(t), N-O relative orientation, nearest-neighbour persistence
C = np.load(os.path.join(D, "module2_C2_curves.npz"))
fig, axs = plt.subplots(1, 3, figsize=(17.5, 6.3)); fig.subplots_adjust(wspace=0.45)
ax = axs[0]
for T in (100, 300):
    Y = []
    for r in (1, 2, 3):
        d = dyn[(dyn["T"] == T) & (dyn.rep == r)].reset_index(drop=True)
        m = (d.interior & d.intact).to_numpy(); Y.append(C[f"T{T}_r{r}"][m].mean(0))
    band(ax, C["lag_ns"], Y, CT[T], f"{T} K")
ax.set_xlabel("Lag time (ns)"); ax.set_ylabel(r"$C_2(t)$"); ax.set_ylim(0.6, 1.01); ax.legend(); finish_panel(ax, "a")
ax = axs[1]; ang = Hs["angle"]
for T in (100, 300):
    band(ax, ang, [Hs[f"T{T}_r{r}_angle"] / 5 for r in (1, 2, 3)], CT[T], f"{T} K")
ax.plot(ang, np.sin(np.radians(ang)) / 2 * np.pi / 180, "k:", lw=2, label="isotropic")
ax.set_xlabel("Angle between N–O vectors (°)"); ax.set_ylabel("Density (deg$^{-1}$)")
ax.legend(); finish_panel(ax, "b")
ax = axs[2]
for T in (100, 300):
    band(ax, Hs["lag_ns"], [Hs[f"persist_T{T}_r{r}"] for r in (1, 2, 3)], CT[T], f"{T} K")
ax.set_xlabel("Lag time (ns)"); ax.set_ylabel("Same nearest neighbor"); ax.set_ylim(0.4, 1.02)
ax.legend(); finish_panel(ax, "c")
save_fig(fig, os.path.join(FSI, "FigS_MD_dynamics_pairs"))
print("Module II figures written")
