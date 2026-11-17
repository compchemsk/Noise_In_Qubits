#!/usr/bin/env python3
"""
m1_05_figures.py  --  all Module I figures (main text + Supporting Information)
Requires the CSV/NPZ outputs of m1_01 ... m1_04.
"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from plot_style import apply_style, finish_panel, save_fig, COLORS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "module1")
FMS = os.path.join(HERE, "..", "Figures_MS"); FSI = os.path.join(HERE, "..", "Figures_ESI")
apply_style()

scans = pd.read_csv(os.path.join(DATA, "module1_scans.csv"))
fits = pd.read_csv(os.path.join(DATA, "module1_fits_all.csv"))
summ = pd.read_csv(os.path.join(DATA, "module1_coupling_summary.csv"))
freqs = pd.read_csv(os.path.join(DATA, "module1_frequencies.csv"))
excl = pd.read_csv(os.path.join(DATA, "module1_excluded_points.csv"))
cmp_ = pd.read_csv(os.path.join(DATA, "module1_T1_model_comparison.csv"))
curv = np.load(os.path.join(DATA, "module1_T1_model_curves.npz"))
sel = pd.read_csv(os.path.join(DATA, "module1_mode_selection.csv"))
summ = summ.merge(freqs[["molecule", "mode", "freq_cm1"]], on=["molecule", "mode"], how="left")

MOLS = ["TEMPO", "PTEMPO", "PTEMPO-alkyl"]
LBL = {"TEMPO": "TEMPO", "PTEMPO": "PTEMPO", "PTEMPO-alkyl": "PTEMPO-alkyl"}
COL = {"TEMPO": COLORS[0], "PTEMPO": COLORS[1], "PTEMPO-alkyl": COLORS[2]}
MRK = {"TEMPO": "o", "PTEMPO": "s", "PTEMPO-alkyl": "D"}

# =============================================================================
# MAIN-TEXT FIGURE (Figure 1): four panels
# =============================================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.subplots_adjust(hspace=0.38, wspace=0.32)

# (a,b) size of the linear (c1, filled) and quadratic (c2, open) terms of every scan
cc = pd.read_csv(os.path.join(DATA, "module1_c1_c2.csv"))
offs = {"TEMPO": -1.8, "PTEMPO": 0.0, "PTEMPO-alkyl": 1.8}
for ax, qty, ylab, lab in [(axs[0, 0], "A_iso_MHz", r"Hyperfine term, $|c_1|$ or $|c_2|$ (MHz)", "a"),
                           (axs[0, 1], "g_iso", r"$g$ term, line shift (MHz)", "b")]:
    for mol in MOLS:
        d = cc[(cc.molecule == mol) & (cc.quantity == qty)].sort_values("freq_cm1")
        x = d.freq_cm1 + offs[mol]
        c1, c2 = d.c1_MHz.abs().clip(lower=1e-3), d.c2_MHz.abs().clip(lower=1e-3)
        ax.vlines(x, np.minimum(c1, c2), np.maximum(c1, c2), color=COL[mol], lw=1.5, alpha=0.7)
        ax.plot(x, c1, MRK[mol], color=COL[mol], ms=11, mec="black", mew=1.2, label=f"{LBL[mol]}, $c_1$")
        ax.plot(x, c2, MRK[mol], mfc="white", mec=COL[mol], mew=2.4, ms=11, ls="none", label=f"{LBL[mol]}, $c_2$")
    ax.set_yscale("log"); ax.set_xlim(0, 215); ax.set_ylim(1e-3, 300)
    ax.set_xlabel(r"Mode frequency (cm$^{-1}$)"); ax.set_ylabel(ylab)
    if lab == "a": ax.legend(loc="upper left", fontsize=13.5, ncol=3, columnspacing=0.5, handletextpad=0.2)
    finish_panel(ax, lab)

# (c) experimental relaxation rate of PTEMPO (slow component): direct + one local mode
from m1_13_multiprocess_T1 import fit as _fit, term as _term, MODELS as _MOD, DATA as _DATA, T as _TT
T = _TT.copy()   # used again by the SI figures below
ax = axs[1, 0]
t1, e = _DATA["PTEMPO_slow"]; R = 1e3 / t1; sR = R * e / t1
_f = _fit(_MOD["D+L"], _TT, R, sR); Tf = np.linspace(5, 65, 400)
dpart, lpart = _term("D", _f["pars"][0], Tf), _term("L", _f["pars"][1], Tf)
ax.errorbar(_TT, R, yerr=sR, fmt="o", color="black", ms=10, capsize=4, lw=2, label=r"PTEMPO, experiment$^{\,a}$", zorder=5)
ax.plot(Tf, dpart + lpart, color=COLORS[1], lw=3.2, label="fit: direct + local mode")
ax.plot(Tf, dpart, color=COLORS[0], lw=2.6, ls=":", label=r"direct part, $aT$")
ax.plot(Tf, lpart, color=COLORS[3], lw=2.6, ls="-.", label=rf"local-mode part, $\Delta$ = {_f['pars'][1][1]:.0f} cm$^{{-1}}$")
ax.set_yscale("log"); ax.set_xlim(5, 65); ax.set_ylim(5, 2e4)
ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)")
ax.legend(loc="upper left", fontsize=14.5)
finish_panel(ax, "c")

# (d) share of the local mode in 1/T1 for the three data sets
ax = axs[1, 1]
for ds, c, m, lab in [("PTEMPO_slow", COLORS[1], "s", "PTEMPO, slow"), ("PTEMPO_fast", COLORS[4], "^", "PTEMPO, fast"),
                      ("SAM", COLORS[2], "D", "TEMPO monolayer")]:
    t1, e = _DATA[ds]; ok = ~np.isnan(t1); Tx = _TT[ok]; R = 1e3 / t1[ok]; sR = R * e[ok] / t1[ok]
    f = _fit(_MOD["D+L"], Tx, R, sR); Tg = np.linspace(5, 65, 400)
    d_, l_ = _term("D", f["pars"][0], Tg), _term("L", f["pars"][1], Tg)
    ax.plot(Tg, 100 * l_ / (d_ + l_), color=c, lw=3, label=lab)
    dx, lx = _term("D", f["pars"][0], Tx), _term("L", f["pars"][1], Tx)
    ax.plot(Tx, 100 * lx / (dx + lx), m, color=c, ms=10, mec="black", mew=1.2)
ax.axhline(50, color="black", ls=":", lw=2)
ax.set_xlim(5, 65); ax.set_ylim(0, 100)
ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"Local-mode share (%)")
ax.legend(loc="lower right", fontsize=16)
finish_panel(ax, "d")
save_fig(fig, os.path.join(FMS, "Figure1_spin_phonon"))

# =============================================================================
# SI: scan grids, one figure per molecule and quantity
# =============================================================================
for mol in MOLS:
    for qty, ylab, scale, unit in [("g_iso", r"$\Delta g_\mathrm{iso}\ (\times10^{-4})$", 1e4, ""),
                                   ("A_iso_MHz", r"$\Delta A_\mathrm{iso}$ (MHz)", 1, "MHz")]:
        modes = sorted(scans[(scans.molecule == mol) & (scans.quantity == qty)]["mode"].unique())
        nc = 4 if len(modes) > 4 else len(modes)
        nr = int(np.ceil(len(modes) / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(5.0 * nc, 4.8 * nr), squeeze=False)
        fig.subplots_adjust(hspace=1.0, wspace=0.75)
        for k, mode in enumerate(modes):
            ax = axs[k // nc, k % nc]
            d = scans[(scans.molecule == mol) & (scans.quantity == qty) & (scans["mode"] == mode)]
            f = fits[(fits.molecule == mol) & (fits.quantity == qty) & (fits["mode"] == mode)].iloc[0]
            ex = excl[(excl.molecule == mol) & (excl["mode"] == mode) & (excl.quantity == qty)]
            keep = ~d.Q.round(3).isin(ex.Q.round(3))
            yk = (d.value[keep] - f.c0) * scale
            ax.plot(d.Q[keep], yk, "o", color="black", ms=8, zorder=3)
            qf = np.linspace(-1, 1, 200)
            ax.plot(qf, (f.c1 * qf + f.c2 * qf ** 2) * scale, "-", color=COL[mol], lw=2.5)
            lo, hi = yk.min(), yk.max(); pad = 0.12 * (hi - lo if hi > lo else 1)
            ax.set_ylim(lo - pad, hi + pad)
            for _, r in ex.iterrows():                      # excluded geometry
                ye = (r.value - f.c0) * scale
                if lo - pad <= ye <= hi + pad:
                    ax.plot(r.Q, ye, "x", color="red", ms=13, mew=3, zorder=4)
                else:
                    ax.annotate(f"excluded\n({ye:+.2f})", xy=(r.Q, hi + pad), xytext=(r.Q + 0.15, hi - 2.5 * pad),
                                color="red", fontsize=15, ha="left",
                                arrowprops=dict(arrowstyle="->", color="red", lw=2))
            fr = freqs[(freqs.molecule == mol) & (freqs["mode"] == mode)]
            ttl = f"Mode {mode}" + (f" ({fr.freq_cm1.iloc[0]:.1f} cm$^{{-1}}$)" if len(fr) else "")
            c1s, c2s = f"{f.c1*scale:.3g}", f"{f.c2*scale:.3g}"
            dom = "$c_2$" if abs(f.c2) > abs(f.c1) else "$c_1$"
            ax.set_title(f"{ttl}\n$c_1$ = {c1s}, $c_2$ = {c2s}\nlarger: {dom}", fontsize=16)
            ax.set_xlabel("Q", fontsize=23); ax.set_ylabel(ylab, fontsize=21)
            ax.tick_params(labelsize=19)
            finish_panel(ax)
        for k in range(len(modes), nr * nc):
            axs[k // nc, k % nc].axis("off")
        tag = "g" if qty == "g_iso" else "A"
        save_fig(fig, os.path.join(FSI, f"FigS_scans_{mol.replace('-', '_')}_{tag}"))

# =============================================================================
# SI: functional benchmark (values from the original notebook)
# =============================================================================
fig, axs = plt.subplots(1, 2, figsize=(13, 6.2)); fig.subplots_adjust(wspace=0.40)
labs = ["B3LYP", "B2PLYP", "DSD-PBEP86", "RI-MP2"]
g = [2.0057903, 2.0061612, 2.0063078, 2.0068677]
dA = [10.83, -3.12, -8.55, -18.82]
cols = [COLORS[7], COLORS[1], COLORS[2], COLORS[4]]
ax = axs[0]; ax.bar(labs, g, color=cols, edgecolor="black", lw=1.5)
ax.axhline(2.0063, color=COLORS[0], ls="--", lw=2.5, label="reference")
ax.set_ylim(2.0050, 2.0072); ax.set_ylabel(r"$g_\mathrm{iso}$"); plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.ticklabel_format(axis="y", useOffset=False); ax.legend(loc="upper left"); finish_panel(ax, "a")
ax = axs[1]; ax.bar(labs, dA, color=cols, edgecolor="black", lw=1.5)
ax.axhline(0, color="black", lw=2); ax.set_ylabel(r"$\Delta A_\mathrm{iso}$ (MHz)")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); finish_panel(ax, "b")
save_fig(fig, os.path.join(FSI, "FigS_functional_benchmark"))

# =============================================================================
# SI: relaxation-model fits for all three data sets
# =============================================================================
datasets = [("PTEMPO_slow", "PTEMPO, slow component", np.array([87, 57, 25.7, 13.0, 4.2, 2.00, 1.00]), np.array([2, 1, .6, .4, .1, .07, .04])),
            ("PTEMPO_fast", "PTEMPO, fast component", np.array([21, 14.4, 6.2, 3.5, 1.2, .69, .37]), np.array([1, .6, .4, .3, .1, .06, .04])),
            ("SAM", "TEMPO SAM", np.array([np.nan, 9.2, 3.6, 1.5, .5, .3, np.nan]), np.array([np.nan, .6, .5, .2, .1, .1, np.nan]))]
fig, axs = plt.subplots(1, 3, figsize=(19, 6.8)); fig.subplots_adjust(wspace=0.36)
for ax, (key, title, t1, e), lab in zip(axs, datasets, "abc"):
    ok = ~np.isnan(t1); R = 1e3 / t1[ok]; sR = R * e[ok] / t1[ok]
    ax.errorbar(T[ok], R, yerr=sR, fmt="o", color="black", ms=9, capsize=4, lw=2, label="experiment", zorder=5)
    for mdl, c, ls, nm in [("M_loc", COLORS[1], "-", "one local mode"), ("M_ram", COLORS[3], ":", r"Raman $T^m$"),
                           ("M_A", COLORS[0], "--", r"DFT, $w\propto(\partial A)^2$"),
                           ("M_g", COLORS[2], "-.", r"DFT, $w\propto(\partial\nu_g)^2$")]:
        ax.plot(Tf, curv[f"{key}__{mdl}"], color=c, ls=ls, lw=2.6, label=nm)
    ax.set_yscale("log"); ax.set_xlim(5, 65)
    ax.set_xlabel("Temperature (K)"); ax.set_ylabel(r"$1/T_1$ (s$^{-1}$)"); ax.set_title(title, fontsize=21)
    if lab == "a": ax.legend(fontsize=17, loc="lower right")
    finish_panel(ax, lab)
save_fig(fig, os.path.join(FSI, "FigS_T1_model_fits"))
print("Figures written.")
