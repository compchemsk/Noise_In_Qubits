#!/usr/bin/env python3
"""
m3_features.py -- descriptors for the exchange coupling of two nitroxides.

Input for one pair (or a batch, leading axes broadcast):
    A, B : (..., 9, 3) key atoms  N, O, Ca1, Ca2, C4, Me1..Me4
    HA,HB: (..., 12, 3) methyl hydrogens
All descriptors are invariant to rotation/translation, to the exchange of the
two radicals (A <-> B) and to the permutation of equivalent atoms (the two
alpha carbons, the four methyl groups).  Three families are built:

  geo  : 10 internal coordinates of the two N-O groups
         r(O..O), r(N..N), the two cross N..O distances (sorted),
         the two N-O..O angles (sorted), cos of the N-O..O-N dihedral,
         |cos| of the angle between the two pi axes, and the two angles
         between each pi axis and the O..O vector (sorted, |cos|)
  ctc  : contacts between atom types (N, O, Ca, C4, Me, H): minimum
         distance and a soft count sum exp(-(r-3)/0.5) for every unordered
         type pair (A-B and B-A combined) -> 2 x 21 = 42
  orb  : overlap of the two nitroxide pi* SOMOs built from STO-3G 2p
         functions on N and O, phi = cN p_N - cO p_O along the pi axis
         (normal to the Ca-N-Ca plane), cN^2 = 0.45, cO^2 = 0.55;
         descriptors S, S^2, log10(S^2 + 1e-12) and the N/O-resolved
         components.  In the Kahn/Hay-Thibeault-Hoffmann picture the
         antiferromagnetic part of J scales with S^2.
  user3: the three generation coordinates (O..O, O1-O2-N2 angle,
         N1-O1-O2-N2 dihedral) -- not A<->B symmetric, kept as a baseline.
"""
import numpy as np

BOHR = 0.529177210903
# STO-3G 2p contraction (exponents for zeta = 1; scaled by zeta^2)
_STO3G_A = np.array([0.9942030, 0.2310310, 0.0751386])
_STO3G_C = np.array([0.15591627, 0.60768372, 0.39195739])
ZETA = {"N": 1.95, "O": 2.25}
CN, CO = np.sqrt(0.45), -np.sqrt(0.55)

def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def pi_axis(M):
    """normal to the Ca-N-Ca plane (sign fixed so that it is arbitrary but defined)"""
    return _unit(np.cross(M[..., 2, :] - M[..., 0, :], M[..., 3, :] - M[..., 0, :]))

def _pp_overlap(RA, RB, u, v, zA, zB):
    """overlap <p_u(A)|p_v(B)> for STO-3G 2p functions (positions in Angstrom)"""
    R = (RA - RB) / BOHR                                     # bohr
    R2 = (R * R).sum(-1); uR = (u * R).sum(-1); vR = (v * R).sum(-1); uv = (u * v).sum(-1)
    a = _STO3G_A * zA ** 2; b = _STO3G_A * zB ** 2
    na = (2 * a / np.pi) ** 0.75 * 2 * np.sqrt(a); nb = (2 * b / np.pi) ** 0.75 * 2 * np.sqrt(b)
    S = 0.0
    for i in range(3):
        for j in range(3):
            p = a[i] + b[j]; mu = a[i] * b[j] / p
            S = S + _STO3G_C[i] * _STO3G_C[j] * na[i] * nb[j] * (np.pi / p) ** 1.5 * np.exp(-mu * R2) * \
                (uv / (2 * p) - (a[i] * b[j] / p ** 2) * uR * vR)
    return S

def somo_overlap(A, B):
    nA, nB = pi_axis(A), pi_axis(B)
    pos = {"N": 0, "O": 1}; c = {"N": CN, "O": CO}
    comp = {}
    for x in "NO":
        for y in "NO":
            comp[x + y] = c[x] * c[y] * _pp_overlap(A[..., pos[x], :], B[..., pos[y], :], nA, nB, ZETA[x], ZETA[y])
    S = sum(comp.values())
    return S, comp

def _angle(a, b, c):
    v1, v2 = _unit(a - b), _unit(c - b)
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1), -1, 1)))

def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = _unit(b1)
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1); y = (np.cross(b1n, v) * w).sum(-1)
    return np.degrees(np.arctan2(y, x))

TYPES = {"N": [0], "O": [1], "Ca": [2, 3], "C4": [4], "Me": [5, 6, 7, 8]}
TNAMES = list(TYPES) + ["H"]

def features(A, B, HA, HB):
    A = np.asarray(A, float); B = np.asarray(B, float); HA = np.asarray(HA, float); HB = np.asarray(HB, float)
    NA, OA, NB, OB = A[..., 0, :], A[..., 1, :], B[..., 0, :], B[..., 1, :]
    d = lambda x, y: np.linalg.norm(x - y, axis=-1)
    f = {}
    # ---------- geo
    f["r_OO"] = d(OA, OB); f["r_NN"] = d(NA, NB)
    x1, x2 = d(NA, OB), d(OA, NB)
    f["r_NO_min"], f["r_NO_max"] = np.minimum(x1, x2), np.maximum(x1, x2)
    t1, t2 = _angle(NA, OA, OB), _angle(NB, OB, OA)
    f["ang_NOO_min"], f["ang_NOO_max"] = np.minimum(t1, t2), np.maximum(t1, t2)
    f["cos_tau"] = np.cos(np.radians(_dihedral(NA, OA, OB, NB)))
    nA, nB = pi_axis(A), pi_axis(B); e = _unit(OB - OA)
    f["pipi_abs_cos"] = np.abs((nA * nB).sum(-1))
    c1, c2 = np.abs((nA * e).sum(-1)), np.abs((nB * e).sum(-1))
    f["pi_OO_cos_min"], f["pi_OO_cos_max"] = np.minimum(c1, c2), np.maximum(c1, c2)
    # ---------- contacts
    FA = {**{k: A[..., v, :] for k, v in TYPES.items()}, "H": HA}
    FB = {**{k: B[..., v, :] for k, v in TYPES.items()}, "H": HB}
    for i, ti in enumerate(TNAMES):
        for tj in TNAMES[i:]:
            D1 = np.linalg.norm(FA[ti][..., :, None, :] - FB[tj][..., None, :, :], axis=-1)
            D2 = np.linalg.norm(FB[ti][..., :, None, :] - FA[tj][..., None, :, :], axis=-1)
            D1 = D1.reshape(D1.shape[:-2] + (-1,)); D2 = D2.reshape(D2.shape[:-2] + (-1,))
            if ti == tj:
                DD = D1
            else:
                DD = np.concatenate([D1, D2], -1)
            f[f"dmin_{ti}_{tj}"] = DD.min(-1)
            f[f"soft_{ti}_{tj}"] = np.exp(-(DD - 3.0) / 0.5).sum(-1)
    # ---------- orbital overlap
    S, comp = somo_overlap(A, B)
    f["S_somo"] = np.abs(S); f["S2_somo"] = S ** 2; f["logS2_somo"] = np.log10(S ** 2 + 1e-40)
    # N/O-resolved components; their sign depends on the arbitrary pi-axis sign -> magnitudes
    f["S_NN"], f["S_OO"] = np.abs(comp["NN"]), np.abs(comp["OO"])
    f["S_NO"] = np.abs(comp["NO"] + comp["ON"])
    # ---------- user3 baseline (A as molecule 1)
    f["u3_OO"] = f["r_OO"]; f["u3_angle"] = _angle(OA, OB, NB)
    f["u3_dihedral"] = np.mod(_dihedral(NA, OA, OB, NB), 360.0)
    return f

SETS = {}
def define_sets(names):
    geo = ["r_OO", "r_NN", "r_NO_min", "r_NO_max", "ang_NOO_min", "ang_NOO_max", "cos_tau",
           "pipi_abs_cos", "pi_OO_cos_min", "pi_OO_cos_max"]
    ctc = [n for n in names if n.startswith("dmin_") or n.startswith("soft_")]
    orb = ["S_somo", "S2_somo", "logS2_somo", "S_NN", "S_OO", "S_NO"]
    SETS.update({"user3": ["u3_OO", "u3_angle", "u3_dihedral"], "geo": geo, "ctc": ctc, "orb": orb,
                 "geo+orb": geo + orb, "geo+ctc": geo + ctc, "all": geo + ctc + orb})
    return SETS
