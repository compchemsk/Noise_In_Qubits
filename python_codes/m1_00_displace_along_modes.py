#!/usr/bin/env python3
"""
m1_00_displace_along_modes.py
Generates geometries displaced along selected normal modes in the
DIMENSIONLESS normal coordinate Q used throughout the spin-phonon analysis.

Definition.  For a mode of angular frequency w with mass-weighted, unit-norm
eigenvector l (sum_i l_i^2 = 1, i runs over 3N Cartesian components),

        H = (hbar w / 2) (P^2 + Q^2),    x_i = x_i^0 + Q sqrt(hbar / w) l_i / sqrt(m_i)

so that <Q^2> = n(w) + 1/2 in thermal equilibrium.  The relation is
independent of how the program normalizes its printed displacement vectors:
whatever vector d is printed, l = sqrt(m) d / |sqrt(m) d|.

Input : a Gaussian 16 frequency output (standard "Freq" output; the last
        "Standard orientation" block and the normal-mode tables are read).
Output: one .xyz file per mode and displacement, e.g. mode07_Q+0.40.xyz,
        ready to be inserted in an ORCA EPR input (see templates/).

Usage:
    python m1_00_displace_along_modes.py freq.log --modes 7 8 9 --qmax 1.0 --step 0.2 --out scans/
Self-check printed for every mode: sum_i m_i dx_i^2 = Q^2 hbar / w (relative error).
"""
import argparse, os, re, numpy as np

HBAR = 1.054571817e-34; AMU = 1.66053906660e-27; C_CM = 2.99792458e10; ANG = 1e-10
MASS = {1: 1.00782503, 6: 12.0, 7: 14.00307401, 8: 15.99491462, 16: 31.97207117}
SYMB = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}

def parse_gaussian(path):
    txt = open(path).read()
    blocks = txt.split("Standard orientation:")
    if len(blocks) < 2:
        raise ValueError("no 'Standard orientation' block found")
    lines = blocks[-1].splitlines()[5:]
    Z, X = [], []
    for L in lines:
        if L.strip().startswith("-----"): break
        sp = L.split(); Z.append(int(sp[1])); X.append([float(v) for v in sp[3:6]])
    Z, X = np.array(Z), np.array(X)
    nat = len(Z)
    freqs, modes = [], []
    sec = txt.split("Harmonic frequencies (cm**-1)")[-1].splitlines()
    k = 0
    while k < len(sec):
        L = sec[k]
        if L.strip().startswith("Frequencies --"):
            f = [float(v) for v in L.split("--")[1].split()]
            while not sec[k].strip().startswith("Atom"): k += 1
            disp = np.zeros((len(f), nat, 3))
            for a in range(nat):
                sp = sec[k + 1 + a].split()
                vals = np.array([float(v) for v in sp[2:2 + 3 * len(f)]])
                for j in range(len(f)):
                    disp[j, a] = vals[3 * j:3 * j + 3]
            freqs += f; modes += list(disp); k += nat
        k += 1
    return Z, X, np.array(freqs), np.array(modes)

def displacement(Z, mode, freq_cm, Q):
    m = np.array([MASS[z] for z in Z])[:, None] * AMU            # kg
    l = np.sqrt(m) * mode; l /= np.linalg.norm(l)                  # mass-weighted unit vector
    w = 2 * np.pi * C_CM * freq_cm                                  # rad/s
    dx = Q * np.sqrt(HBAR / w) * l / np.sqrt(m)                     # metres
    check = (m * dx ** 2).sum() / (Q ** 2 * HBAR / w) - 1 if Q else 0.0
    return dx / ANG, check

def write_xyz(path, Z, X, comment):
    with open(path, "w") as f:
        f.write(f"{len(Z)}\n{comment}\n")
        for z, x in zip(Z, X):
            f.write(f"{SYMB[z]:2s} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile"); ap.add_argument("--modes", type=int, nargs="+", required=True, help="1-based mode numbers")
    ap.add_argument("--qmax", type=float, default=1.0); ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--out", default="scans")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    Z, X, freqs, modes = parse_gaussian(a.logfile)
    grid = np.round(np.arange(-a.qmax, a.qmax + 1e-9, a.step), 6)
    for k in a.modes:
        f = freqs[k - 1]
        for Q in grid:
            dx, chk = displacement(Z, modes[k - 1], f, Q)
            write_xyz(os.path.join(a.out, f"mode{k:02d}_Q{Q:+.2f}.xyz"), Z, X + dx,
                      f"mode {k} ({f:.1f} cm-1), dimensionless Q = {Q:+.2f}")
        _, chk = displacement(Z, modes[k - 1], f, 1.0)
        print(f"mode {k:3d}  {f:8.1f} cm-1  max |dx| at Q=1: {np.abs(displacement(Z, modes[k-1], f, 1.0)[0]).max():.4f} A   self-check {chk:+.1e}")
