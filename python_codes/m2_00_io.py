#!/usr/bin/env python3
"""
m2_00_io.py -- shared I/O for Module II (no MDAnalysis needed).
  * read_parm7(path)   : minimal Amber parm7 parser (names, types, residues,
                         charges, masses, bonds with equilibrium lengths)
  * load_traj(prefix)  : float32 coordinates (Angstrom) written by xtc2bin
"""
import os, re, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Raw MD files (unpacked ModuleII.zip: parm7 + XTC trajectories) and decoded trajectories.
# Override with the environment variables TEMPO_MD_RAW and TEMPO_MD_BIN.
RAW = os.environ.get("TEMPO_MD_RAW", os.path.join(ROOT, "raw", "ModuleII"))
BIN = os.environ.get("TEMPO_MD_BIN", os.path.join(ROOT, "raw", "decoded"))
RUNS = [(T, r) for T in (300, 100) for r in (1, 2, 3)]

def _fmt_width(fmt):
    m = re.match(r"\((\d+)([aAiIeEfF])(\d+)", fmt)
    return int(m.group(1)), m.group(2).lower(), int(m.group(3))

def read_parm7(path):
    sections, cur, fmt, buf = {}, None, None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith("%FLAG"):
                if cur: sections[cur] = (fmt, buf)
                cur, buf = line.split()[1], []
            elif line.startswith("%FORMAT"):
                fmt = line.strip()[7:]
            elif line.startswith("%"):
                continue
            elif cur:
                buf.append(line.rstrip("\n"))
    if cur: sections[cur] = (fmt, buf)
    def get(flag):
        fmt, lines = sections[flag]
        n, typ, w = _fmt_width(fmt)
        vals = []
        for L in lines:
            for k in range(0, len(L), w):
                tok = L[k:k + w]
                if tok.strip() == "" and typ != "a": continue
                vals.append(tok.strip() if typ == "a" else (int(tok) if typ == "i" else float(tok)))
        return vals
    P = {}
    P["name"] = np.array(get("ATOM_NAME"))
    P["type"] = np.array(get("AMBER_ATOM_TYPE"))
    P["charge"] = np.array(get("CHARGE")) / 18.2223            # -> e
    P["mass"] = np.array(get("MASS"))
    P["natom"] = len(P["name"])
    rl, rp = get("RESIDUE_LABEL"), np.array(get("RESIDUE_POINTER")) - 1
    resid = np.zeros(P["natom"], int); resname = np.empty(P["natom"], object)
    for k, start in enumerate(rp):
        end = rp[k + 1] if k + 1 < len(rp) else P["natom"]
        resid[start:end] = k + 1; resname[start:end] = rl[k]
    P["resid"], P["resname"] = resid, resname.astype(str)
    req = np.array(get("BOND_EQUIL_VALUE"))
    bonds = []
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        v = np.array(get(flag), int).reshape(-1, 3)
        for a, b, t in v:
            bonds.append((a // 3, b // 3, req[t - 1]))
    P["bonds"] = np.array([(a, b) for a, b, _ in bonds], int)
    P["bond_req"] = np.array([r for _, _, r in bonds])
    return P

def frames_info(prefix):
    return np.loadtxt(prefix + ".frames.txt")

def load_traj(prefix, natom):
    X = np.fromfile(prefix + ".coords.f32", dtype=np.float32)
    return X.reshape(-1, natom, 3)

def run_prefix(T, r):
    return os.path.join(BIN, f"T{T}_r{r}")


# Seven pairs of molecules (14 molecules) whose TEMPO rings interpenetrate in the common starting
# structure (S atoms 3.1-3.3 A apart). They are INCLUDED in all analyses by default.
# Set TEMPO_EXCLUDE_DEFECTS=1 to reproduce the earlier analysis without them.
DEFECT_IDS = [2428, 2440, 2441, 2455, 2485, 2486, 2495, 2508, 2512, 2527, 2528, 2529, 2533, 2537]
EXCLUDE_DEFECTS = os.environ.get("TEMPO_EXCLUDE_DEFECTS", "0") == "1"
DEFECT = list(DEFECT_IDS) if EXCLUDE_DEFECTS else []
