#!/usr/bin/env python3
"""fig_structures.py -- skeletal formulas of TEMPO, PTEMPO and PTEMPO-alkyl, side by side.

Drawn from explicit 2D coordinates (no external chemistry toolkit). Line ends and vertices are
carbon atoms with implicit hydrogens; N and O are labelled; O carries the unpaired electron.
  TEMPO        2,2,6,6-tetramethylpiperidin-1-oxyl                         C9H18NO
  PTEMPO       4-(prop-2-yn-1-yloxy)-TEMPO                                 C12H20NO2
  PTEMPO-alkyl 4-{[1-(11-sulfanylundecyl)-1H-1,2,3-triazol-4-yl]methoxy}-TEMPO   C23H43N4O2S
               (click product with the azidoundecanethiol of the monolayer; SH in place of the Au-S bond;
                structure as in raw/module1/PTEMPO-alkyl-11.log)
"""
import os, numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CN, CO, CC, CS = "#0050FF", "#E8001C", "black", "#C99700"
LW = 3.0

def P(a):            # unit vector at angle a (degrees)
    return np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])

def bond(ax, a, b, order=1, side=None, shrink_a=0.0, shrink_b=0.0, off=0.16, t=0.15):
    a, b = np.asarray(a, float), np.asarray(b, float); d = (b - a) / np.linalg.norm(b - a)
    a2, b2 = a + d * shrink_a, b - d * shrink_b
    n = np.array([-d[1], d[0]])
    if order == 1:
        ax.plot(*zip(a2, b2), color=CC, lw=LW, solid_capstyle="round")
    elif order == 2:                       # second line drawn on the given side (inside a ring)
        ax.plot(*zip(a2, b2), color=CC, lw=LW, solid_capstyle="round")
        s = 1 if side is None else side
        ax.plot(*zip(a2 + s * off * n + d * t, b2 + s * off * n - d * t), color=CC, lw=LW, solid_capstyle="round")
    elif order == 3:
        for k in (-1, 0, 1):
            ax.plot(*zip(a2 + k * off * n, b2 + k * off * n), color=CC, lw=LW, solid_capstyle="round")

def atom(ax, p, text, color, fs=26):
    ax.text(*p, text, color=color, fontsize=fs, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.12", fc="white", ec="none"), zorder=5)

def tempo_core(ax):
    """Piperidine ring drawn as a hexagon: N1 at bottom, C4 at top. Returns position of C4."""
    ang = {"C4": 90, "C3": 30, "C2": -30, "N1": -90, "C6": -150, "C5": 150}
    p = {k: P(a) for k, a in ang.items()}
    ring = ["C4", "C3", "C2", "N1", "C6", "C5", "C4"]
    for i in range(6):
        a, b = ring[i], ring[i + 1]
        bond(ax, p[a], p[b], shrink_a=0.28 if a == "N1" else 0, shrink_b=0.28 if b == "N1" else 0)
    # gem-dimethyl groups on C2 and C6 (outward direction +/- 35 deg)
    for c, out in (("C2", -30), ("C6", -150)):
        for da in (+38, -38):
            bond(ax, p[c], p[c] + P(out + da))
    # N-O radical
    pO = p["N1"] + P(-90) * 1.05
    bond(ax, p["N1"], pO, shrink_a=0.28, shrink_b=0.28)
    atom(ax, p["N1"], "N", CN); atom(ax, pO, "O", CO)
    ax.text(pO[0] + 0.36, pO[1] + 0.02, r"$\bullet$", color=CO, fontsize=30, ha="center", va="center", zorder=6)
    return p["C4"]

XL = [(-2.2, 2.2), (-2.2, 2.6), (-2.2, 13.4)]; YL = (-3.8, 6.9)
fig, axs = plt.subplots(1, 3, figsize=(23.6, 10.8), gridspec_kw=dict(width_ratios=[b - a for a, b in XL], wspace=0.02))

# ---------------- TEMPO
ax = axs[0]; c4 = tempo_core(ax)
ax.text(0, 6.4, "TEMPO", fontsize=30, fontweight="bold", ha="center")
ax.text(0, -3.35, r"C$_9$H$_{18}$NO$^\bullet$", fontsize=24, ha="center")


# ---------------- PTEMPO
ax = axs[1]; c4 = tempo_core(ax)
pO = c4 + P(60); pC1 = pO + P(120); pC2 = pC1 + P(60); pC3 = pC2 + P(60)
bond(ax, c4, pO, shrink_b=0.28); bond(ax, pO, pC1, shrink_a=0.28)
bond(ax, pC1, pC2); bond(ax, pC2, pC3, order=3)
atom(ax, pO, "O", CO)
ax.text(*(pC3 + P(60) * 0.42), "H", fontsize=24, ha="center", va="center", fontweight="bold")
ax.text(0, 6.4, "PTEMPO", fontsize=30, fontweight="bold", ha="center")
ax.text(0, -3.35, r"C$_{12}$H$_{20}$NO$_2^\bullet$", fontsize=24, ha="center")


# ---------------- PTEMPO-alkyl (click product with an undecyl azide)
ax = axs[2]; c4 = tempo_core(ax)
pO = c4 + P(60); pC1 = pO + P(120)
bond(ax, c4, pO, shrink_b=0.28); bond(ax, pO, pC1, shrink_a=0.28); atom(ax, pO, "O", CO)
# 1,2,3-triazole: C4' attached to CH2; ring order C4', C5', N1', N2', N3'
u = P(60); t4 = pC1 + u; R = 1 / (2 * np.sin(np.radians(36))); cen = t4 + u * R
phi = np.degrees(np.arctan2(*(t4 - cen)[::-1]))
V = [cen + R * P(phi + 72 * k) for k in range(5)]          # C4', C5', N1', N2', N3'
names = ["C", "C", "N", "N", "N"]
bond(ax, pC1, t4)
isN = [False, False, True, True, True]
ring_bonds = [(0, 1, 2), (1, 2, 1), (2, 3, 1), (3, 4, 2), (4, 0, 1)]   # C4'=C5', C5'-N1', N1'-N2', N2'=N3', N3'-C4'
for i, j, o in ring_bonds:
    mid = (V[i] + V[j]) / 2; d = V[j] - V[i]; n = np.array([-d[1], d[0]])
    side = 1 if np.dot(n, cen - mid) > 0 else -1             # second line inside the ring
    bond(ax, V[i], V[j], order=o, side=side, shrink_a=0.25 if isN[i] else 0, shrink_b=0.25 if isN[j] else 0, t=0.0 if (isN[i] and isN[j]) else 0.15)
for k in (2, 3, 4):
    atom(ax, V[k], "N", CN)
# undecyl chain on N1' (11 carbons, zigzag)
out = np.degrees(np.arctan2(*(V[2] - cen)[::-1]))
pts = [V[2]]; dirs = [out] + [(-30 if k % 2 == 0 else 30) for k in range(10)]
for k, dd in enumerate(dirs):
    pts.append(pts[-1] + P(dd))
for k in range(11):
    bond(ax, pts[k], pts[k + 1], shrink_a=0.28 if k == 0 else 0)
pS = pts[-1] + P(30 if dirs[-1] < 0 else -30)            # terminal S-H
bond(ax, pts[-1], pS, shrink_b=0.3); atom(ax, pS, "S", CS)
ax.text(*(pS + np.array([0.52, 0])), "H", fontsize=24, ha="center", va="center", fontweight="bold")
yb = min(q[1] for q in pts[1:]) - 0.35
ax.plot([pts[1][0], pts[-1][0]], [yb, yb], color="#555555", lw=1.5)
for xx in (pts[1][0], pts[-1][0]): ax.plot([xx, xx], [yb, yb + 0.15], color="#555555", lw=1.5)
ax.text((pts[1][0] + pts[-1][0]) / 2, yb - 0.4, r"(CH$_2$)$_{11}$, 11 carbons", fontsize=21, ha="center", color="#333333")
ax.text(*(V[1] + P(phi + 72) * 0.42), "H", fontsize=22, ha="center", va="center", fontweight="bold")
ax.text(0, 6.4, "PTEMPO-alkyl", fontsize=30, fontweight="bold", ha="center")
ax.text(0, -3.35, r"C$_{23}$H$_{43}$N$_4$O$_2$S$^\bullet$", fontsize=24, ha="center")
ax.text(5.0, -1.7, "1,2,3-triazole formed by click\nreaction with the azido-undecanethiol\nof the monolayer; SH replaces\nthe Au–S bond",
        fontsize=19, ha="left", va="center", color="#333333")


for ax, xl in zip(axs, XL):
    ax.set_xlim(*xl); ax.set_ylim(*YL); ax.set_aspect("equal"); ax.axis("off")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "Figures_ESI", f"FigS_structures.{ext}"), dpi=300, bbox_inches="tight")
print('written; chain bonds =', len(pts) - 1)
