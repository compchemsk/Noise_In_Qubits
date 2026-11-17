#!/usr/bin/env python3
"""
m4_01_mode_projection.py
Spin-active normal modes of the isolated radicals followed inside the monolayer.

What it does
  1. Rebuilds the DFT normal-mode vectors of TEMPO and PTEMPO from the displaced
     geometries of the ORCA scans: v_k = [x(Q=+1) - x(Q=-1)] / 2  (Angstrom per unit Q).
  2. Checks them: mass-weighted norms equal hbar/omega_k, and different modes are
     orthogonal in the mass-weighted metric.
  3. Maps the DFT atoms onto the radical head of every monolayer molecule (atom order
     verified through the bond graph), aligns each head in every frame onto the DFT
     equilibrium structure (mass-weighted Kabsch fit of the ring heavy atoms), and
     relabels the three hydrogens of each methyl group to remove methyl rotation.
  4. Projects the displacement from the ensemble-mean head structure onto each mode:
        Q_k = (v_k^T M dx) / (v_k^T M v_k)      (sums over the mapped atoms)
     which returns Q exactly when dx = Q v_k.
  5. Converts the projected coordinates into spin-Hamiltonian fluctuations with the DFT
     coupling constants of the isolated radical:
        dX = sum_k [c1_k Q_k + c2_k (Q_k^2 - <Q_k^2>)],  X = g_iso or A_iso,
     and the Q-band line shift d nu_g = (nu_mw / g_iso) d g_iso.
Output: data/module4/proj_<basis>_T<T>_r<r>.npz and data/module4/mode_vectors.json
"""
import os, json, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_00_io import read_parm7, load_traj, run_prefix

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
D1 = os.path.join(ROOT, "data", "module1"); D3 = os.path.join(ROOT, "data", "module3")
OUT = os.path.join(ROOT, "data", "module4"); os.makedirs(OUT, exist_ok=True)
from m2_00_io import RAW as _RAW
PARM = os.path.join(_RAW, "TEMPO-SAM-11.parm7")
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}
HB = 33.7168                     # hbar/(2 pi c) in amu A^2 cm^-1, so hbar/omega = HB / nu
NU_MW = 35000.0                  # MHz
NFR = 500                        # analysis window: last 5 ns

# ---------------- DFT modes ------------------------------------------------------
P = json.load(open(os.path.join(D1, "orca_tensors", "parsed_tensors.json")))
freq = pd.read_csv(os.path.join(D1, "module1_frequencies.csv"))
summ = pd.read_csv(os.path.join(D1, "module1_coupling_summary.csv"))
an = pd.read_csv(os.path.join(D1, "module1_anisotropic_couplings.csv"))

def dft_modes(mol):
    R = P[mol]; eq = [r for r in R if r["Q"] == 0][0]
    X0 = np.array(eq["xyz"]); el = eq["el"]; m = np.array([MASS[e] for e in el])
    modes = sorted(set(r["mode"] for r in R)); V = []
    for k in modes:
        xp = np.array([r for r in R if r["mode"] == k and abs(r["Q"] - 1) < 1e-6][0]["xyz"])
        xm = np.array([r for r in R if r["mode"] == k and abs(r["Q"] + 1) < 1e-6][0]["xyz"])
        V.append((xp - xm) / 2)
    V = np.array(V)
    nu = freq[freq.molecule == mol].set_index("mode").freq_cm1.loc[modes].to_numpy()
    s = summ[summ.molecule == mol].set_index("mode").loc[modes]
    a = an[an.molecule == mol].set_index("mode").loc[modes]
    return dict(X0=X0, el=el, m=m, V=V, nu=nu, modes=modes, c1g=s.dg_dQ.to_numpy(), c1A=s.dA_dQ_MHz.to_numpy(),
                c2g=a.c2_g_iso.to_numpy(), c2A=a.c2_A_iso.to_numpy(), g0=float(eq["blocks"][0]["g_iso"]))

def bonds_from_xyz(X, el):
    rc = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}
    B = set()
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if np.linalg.norm(X[i] - X[j]) < 1.2 * (rc[el[i]] + rc[el[j]]): B.add((i, j))
    return B

def kabsch(A, B, w):
    """rotation R and translation so that (A - ca) R + cb best fits B (weights w)"""
    ca = (w[:, None] * A).sum(0) / w.sum(); cb = (w[:, None] * B).sum(0) / w.sum()
    H = ((A - ca) * w[:, None]).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(U @ Vt))
    Dm = np.diag([1, 1, d]); return U @ Dm @ Vt, ca, cb


def neighbors(bonds, i):
    return sorted({b for a, b in bonds if a == i} | {a for a, b in bonds if b == i})

def build_map(M, B, sam_el, sam_bonds, top):
    """DFT -> SAM atom map: ring atoms by index (verified by elements), substituents by the
    bond graph, and the two methyl groups on each C-alpha, the CH2 hydrogens and the C4
    hydrogen chosen by the smallest RMSD to the DFT structure (MD reference: mean head of
    all intact radicals in the last frame of T100_r1, aligned on the ring atoms)."""
    import itertools
    el = M["el"]; bd = bonds_from_xyz(M["X0"], el)
    ring = B["ring"]
    for i in ring: assert el[i] == sam_el[i]
    # reference MD head structure
    z = np.load(os.path.join(D3, "mdJ_T100_r1.npz")); ii = np.where(z["intact"])[0]
    X = load_traj(run_prefix(100, 1), top["natom"])[-1]
    st = 2400 + 72 * (z["mol_ids"][ii] - z["mol_ids"].min())
    Hs = X[st[:, None] + np.arange(36)[None, :]].astype(float)
    w = np.array([MASS[el[i]] for i in ring])
    acc = np.zeros((36, 3))
    for A in Hs:
        Rm, ca, cb = kabsch(A[ring], M["X0"][ring], w); acc += (A - ca) @ Rm + cb
    ref = acc / len(Hs)
    # candidate groups
    heavy = lambda bonds, i, elist: [j for j in neighbors(bonds, i) if elist[j] != "H"]
    hyd = lambda bonds, i, elist: [j for j in neighbors(bonds, i) if elist[j] == "H"]
    sbd = sam_bonds
    base = {i: i for i in ring}
    options = []
    for ca_ in (1, 2):                                   # C-alpha atoms
        dm = [j for j in heavy(bd, ca_, el) if j not in ring and el[j] == "C"]
        sm = [j for j in heavy(sbd, ca_, sam_el) if j not in ring and sam_el[j] == "C"]
        assert len(dm) == 2 and len(sm) == 2
        options.append([list(zip(dm, sm)), list(zip(dm, sm[::-1]))])
    for c in (0, 3):                                     # ring CH2 groups
        dh = hyd(bd, c, el); sh = hyd(sbd, c, sam_el); assert len(dh) == 2 and len(sh) == 2
        options.append([list(zip(dh, sh)), list(zip(dh, sh[::-1]))])
    dh4 = hyd(bd, 4, el); sh4 = hyd(sbd, 4, sam_el)      # C4 hydrogens
    if len(dh4) == 1: options.append([[(dh4[0], sh4[0])]])
    else: options.append([[(d, sh4[0])] for d in dh4])   # TEMPO: two H on C4, SAM: one
    # PTEMPO: the ether CH2 (atoms 29-31) and the alkyne are not mapped, because the
    # O-CH2 torsion of the propargyl ether differs from that of the triazole linker.
    best = None
    for combo in itertools.product(*options):
        mp = dict(base)
        for grp in combo: mp.update(dict(grp))
        # methyl hydrogens follow their carbon
        for d, s_ in list(mp.items()):
            if el[d] == "C" and len(hyd(bd, d, el)) == 3 and d not in ring:
                for a, b in zip(hyd(bd, d, el), hyd(sbd, s_, sam_el)): mp[a] = b
        keys = sorted(mp); dev = np.sqrt(np.mean([((ref[mp[k]] - M["X0"][k]) ** 2).sum() for k in keys]))
        if best is None or dev < best[0]: best = (dev, mp)
    mp = best[1]
    assert all(el[k] == sam_el[v] for k, v in mp.items()) and len(set(mp.values())) == len(mp)
    print("map RMSD to DFT (A):", round(best[0], 3), "atoms:", len(mp))
    return mp

if __name__ == "__main__":
    top = read_parm7(PARM); names = top["name"]; resid = top["resid"]
    first = 2400; res_atoms = np.arange(first, first + 72)
    sam_el = [str(names[i])[0] for i in res_atoms]
    sam_bonds = {(a - first, b - first) if a < b else (b - first, a - first) for a, b in top["bonds"]
                 if first <= a < first + 72 and first <= b < first + 72}
    report = {}
    BASES = {
        # basis: DFT atom index -> SAM residue atom index; core heavy atoms used for the alignment
        "PTEMPO": dict(n=32, ring=[0, 1, 2, 3, 4, 10, 11, 28]),
        "TEMPO": dict(n=29, ring=[0, 1, 2, 3, 4, 10, 11]),
    }
    METHYLS = None
    vec_out = {}
    for basis, B in BASES.items():
        M = dft_modes(basis); mp = build_map(M, B, sam_el, sam_bonds, top)
        report.setdefault("maps", {})[basis] = {int(k): int(v) for k, v in mp.items()}
        B["core"] = [i for i in sorted(mp) if M["el"][i] != "H"]
        dA = np.array(sorted(mp)); sA = np.array([mp[i] for i in dA])
        # check 1: element identity and bond graph of the mapped atoms
        assert all(M["el"][i] == sam_el[mp[i]] for i in dA), "element mismatch"
        bd = bonds_from_xyz(M["X0"], M["el"])
        bd_m = {(a, b) for a, b in bd if a in mp and b in mp}
        bs_m = {(a, b) for a, b in sam_bonds if a in sA and b in sA}
        inv = {v: k for k, v in mp.items()}
        bs_m = {tuple(sorted((inv[a], inv[b]))) for a, b in bs_m}
        assert bd_m == bs_m, f"bond graph mismatch {bd_m ^ bs_m}"
        # check 2: mode vectors (full molecule) - norms and orthogonality
        m = M["m"]; V = M["V"]; G = np.einsum("kja,j,lja->kl", V, m, V)
        norm_ratio = np.diag(G) / (HB / M["nu"])
        ortho = G / np.sqrt(np.outer(np.diag(G), np.diag(G))); np.fill_diagonal(ortho, 0)
        # coverage of the mapped atoms and overlap of the restricted vectors
        Vr = V[:, dA]; mr = m[dA]
        Gr = np.einsum("kja,j,lja->kl", Vr, mr, Vr)
        cover = np.diag(Gr) / np.diag(G)
        ovr = Gr / np.sqrt(np.outer(np.diag(Gr), np.diag(Gr))); np.fill_diagonal(ovr, 0)
        report[basis] = dict(nu=M["nu"].tolist(), norm_ratio=norm_ratio.round(4).tolist(),
                             max_full_overlap=float(np.abs(ortho).max()), coverage=cover.round(3).tolist(),
                             max_restricted_overlap=float(np.abs(ovr).max()),
                             restricted_overlap=ovr.round(3).tolist())
        vec_out[basis] = dict(modes=M["modes"], nu=M["nu"].tolist(), V=V.tolist(), X0=M["X0"].tolist(), el=M["el"])
        print(basis, json.dumps({k: v for k, v in report[basis].items() if k != "restricted_overlap"}))
        bdd = bonds_from_xyz(M["X0"], M["el"])
        METHYLS = [(c, [j for j in neighbors(bdd, c) if M["el"][j] == "H"]) for c in (12, 16, 20, 24)]
        core_d = np.array(B["core"]); wcore = m[core_d]
        X0r = M["X0"][dA]
        if basis != "TEMPO":
            continue        # PTEMPO basis: checks only (coverage too low for a reliable projection)
        # ---- symmetry-equivalent relabelling for TEMPO. The unsubstituted TEMPO chair is mapped onto itself by
        # ring inversion combined with exchanging (i) the two methyl groups on each C-alpha, (ii) the two H of each
        # ring CH2 and (iii) the two H on C4. For every snapshot the labelling with the smallest deviation from the
        # DFT structure is chosen, as for methyl rotation. Output arrays are indexed by all 29 DFT atoms; the C4-H
        # slot not occupied by the single C4-H of the monolayer head is NaN.
        import itertools
        el = M["el"]; nat = len(el); bdd_ = bonds_from_xyz(M["X0"], el)
        hy = lambda i: [j for j in neighbors(bdd_, i) if el[j] == "H"]
        ringd = [0, 1, 2, 3, 4, 10, 11]
        inv = {v: k for k, v in mp.items()}
        sam_of = {k: mp[k] for k in mp}                                        # reference labelling
        # groups that may be exchanged (DFT indices)
        swaps = []
        for ca_ in (1, 2):
            mc = [j for j in neighbors(bdd_, ca_) if el[j] == "C" and j not in ringd]
            swaps.append([(mc[0], mc[1])] + list(zip(hy(mc[0]), hy(mc[1]))))
        for c in (0, 3):
            h = hy(c); swaps.append([(h[0], h[1])])
        # (iv) exchange of the two sides of the ring (mirror plane through N, O and C4); combined with a proper
        # rotation this maps the inverted chair onto the reference chair
        side = [(1, 2), (0, 3)] + list(zip(hy(0), hy(3)))
        m1 = [j for j in neighbors(bdd_, 1) if el[j] == "C" and j not in ringd]
        m2 = [j for j in neighbors(bdd_, 2) if el[j] == "C" and j not in ringd]
        for a_, b_ in zip(m1, m2): side += [(a_, b_)] + list(zip(hy(a_), hy(b_)))
        swaps.append(side)
        c4h = hy(4)                                                              # two DFT C4-H
        sam_c4h = [mp[i] for i in c4h if i in mp][0]
        wall = np.array([MASS[e] for e in el])
        for T in (100, 300):
            for r in (1, 2, 3):
                z = np.load(os.path.join(D3, f"mdJ_T{T}_r{r}.npz")); intact = z["intact"]; mol = z["mol_ids"]
                X = load_traj(run_prefix(T, r), top["natom"])[-NFR:]
                ii = np.where(intact)[0]; start = first + 72 * (mol[ii] - mol.min())
                assert int(resid[start[0]]) == int(resid[first]) + int(mol[ii[0]] - mol.min())
                nf, nm = NFR, len(ii)
                # SAM coordinates in the reference labelling, indexed by DFT atom (C4-H slots handled below)
                base_d = [k for k in range(nat) if k in mp and k not in c4h]
                Hb = X[:, start[:, None] + np.array([mp[k] for k in base_d])[None, :], :].astype(np.float64)
                H5 = X[:, start + sam_c4h, :].astype(np.float64)
                full = np.full((nf, nm, nat, 3), np.nan); full[:, :, base_d] = Hb
                # step 1: align on the ring atoms
                def align(Afull, atoms):
                    w = wall[atoms] / wall[atoms].sum(); A = Afull[:, :, atoms]; B0 = M["X0"][atoms]
                    ca = np.einsum("fnai,a->fni", A, w); cb = (w[:, None] * B0).sum(0)
                    Hm = np.einsum("fnai,a,aj->fnij", A - ca[:, :, None], wall[atoms], B0 - cb)
                    U, S_, Vt = np.linalg.svd(Hm); d = np.sign(np.linalg.det(U @ Vt))
                    Dm = np.ones(d.shape + (3,)); Dm[..., 2] = d; Rm = U @ (Dm[..., :, None] * Vt)
                    return ca, cb, Rm
                ca, cb, Rm = align(full, ringd)
                rot = lambda Y: np.einsum("fnai,fnij->fnaj", Y - ca[:, :, None], Rm) + cb
                full = rot(full); H5r = rot(H5[:, :, None])[:, :, 0]
                # step 2: choose the labelling (16 swap combinations x 2 C4-H slots)
                best_d = np.full((nf, nm), np.inf); best = full.copy()
                for combo in itertools.product([0, 1], repeat=len(swaps)):
                    Yc = full.copy()
                    for on, grp in zip(combo, swaps):
                        if on:
                            for a_, b_ in grp: Yc[:, :, [a_, b_]] = Yc[:, :, [b_, a_]]
                    if combo[-1]:                                   # side exchange: realign on the relabelled ring
                        c_, d_, R_ = align(Yc, ringd)
                        Yc = np.einsum("fnai,fnij->fnaj", Yc - c_[:, :, None], R_) + d_
                        H5c = np.einsum("fni,fnij->fnj", H5r - c_, R_) + d_
                    else:
                        H5c = H5r
                    for slot in c4h:
                        Yt = Yc.copy(); Yt[:, :, slot] = H5c
                        # methyl rotation relabelling inside the trial labelling
                        PERMS = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0]])
                        for c in (12, 16, 20, 24):
                            p = np.array(hy(c)); cand = Yt[:, :, p][:, :, PERMS]
                            dist = ((cand - M["X0"][p]) ** 2).sum((-1, -2)); bi = dist.argmin(-1)
                            Yt[:, :, p] = np.take_along_axis(cand, bi[:, :, None, None, None], 2)[:, :, 0]
                        used = [k for k in range(nat) if k not in c4h or k == slot]
                        dev = (wall[used][None, None, :, None] * (Yt[:, :, used] - M["X0"][used]) ** 2).sum((-1, -2)) / wall[used].sum()
                        upd = dev < best_d; best_d[upd] = dev[upd]; best[upd] = Yt[upd]
                # step 3: final mass-weighted fit on all heavy atoms of the chosen labelling
                heavy = [k for k in range(nat) if el[k] != "H"]
                ca2, cb2, Rm2 = align(best, heavy)
                out = np.einsum("fnai,fnij->fnaj", best - ca2[:, :, None], Rm2) + cb2
                hv = np.array(heavy)
                rms = np.sqrt((wall[hv][None, None, :, None] * (out[:, :, hv] - M["X0"][hv]) ** 2).sum((-1, -2)) / wall[hv].sum())
                np.savez_compressed(os.path.join(OUT, f"aligned_{basis}_T{T}_r{r}.npz"), Y=out.astype(np.float32),
                                    mol=mol[ii], core_rmsd=rms.astype(np.float32))
                print(basis, T, r, "aligned", out.shape, "median heavy-atom RMSD %.3f A, >0.5 A: %.4f" % (np.median(rms), (rms > 0.5).mean()),
                      "hist", np.histogram(rms, [0, .1, .2, .3, .4, .5, .7, 1, 3])[0].tolist(), flush=True)
    json.dump(report, open(os.path.join(OUT, "mode_checks.json"), "w"), indent=1)
    json.dump(vec_out, open(os.path.join(OUT, "mode_vectors.json"), "w"))
