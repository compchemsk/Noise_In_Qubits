"""Parse ORCA 5.0.1 EPR/NMR outputs (B2PLYP/EPR-II) of the normal-mode displacement scans.
For every file: geometry, energies, <S^2>, and for each EPRNMR block (SCF density, MP2 relaxed density):
full g matrix, principal g, iso g; raw 14N HFC matrix, A(FC), A(SD), A(ORB+DIA), principal A, A(iso)."""
import re, glob, os, sys, json, numpy as np

def mat3(lines, i):
    return np.array([[float(x) for x in lines[i + k].split()[:3]] for k in range(3)])

def parse(fn):
    L = open(fn).read().split("\n")
    m = re.search(r"mode(\d+)-(pos|neg)-([\d.]+)\.log", fn)
    mode, sgn, q = int(m.group(1)), m.group(2), float(m.group(3))
    Q = q if sgn == "pos" else -q
    out = dict(file=os.path.basename(fn), mode=mode, Q=Q, normal=any("ORCA TERMINATED NORMALLY" in l for l in L))
    # geometry
    i = next(k for k, l in enumerate(L) if l.strip() == "CARTESIAN COORDINATES (ANGSTROEM)")
    el, xyz = [], []
    for l in L[i + 2:]:
        s = l.split()
        if len(s) != 4: break
        el.append(s[0]); xyz.append([float(x) for x in s[1:]])
    out["el"], out["xyz"] = el, xyz
    for l in L:
        if l.startswith("FINAL SINGLE POINT ENERGY"): out["E_final"] = float(l.split()[-1])
        if "Expectation value of <S**2>" in l: out["S2"] = float(l.split()[-1])
        if l.startswith("Total Energy       :"): out["E_scf"] = float(l.split()[3])
        if "SCF CONVERGED AFTER" in l: out["scf_conv"] = True
    blocks = [k for k, l in enumerate(L) if "EPRNMR WITH" in l]
    out["blocks"] = []
    for b, k0 in enumerate(blocks):
        k1 = blocks[b + 1] if b + 1 < len(blocks) else len(L)
        tag = "SCF" if "SCF DENSITY" in L[k0] else ("MP2rel" if "MP2 RELAXED" in L[k0] else L[k0].strip())
        B = dict(density=tag)
        for k in range(k0, k1):
            l = L[k]
            if "Treatment of gauge" in l: B["gauge"] = l.split("...")[-1].strip()
            if l.strip() == "The g-matrix:": B["gmat"] = mat3(L, k + 1).tolist()
            if l.startswith(" g(tot)"):
                v = l.split(); B["g_pr"] = [float(x) for x in v[1:4]]; B["g_iso"] = float(v[-1])
                B["g_orient"] = mat3([x[3:] for x in L[k + 3:k + 6]], 0).tolist()
            if "Raw HFC matrix" in l: B["Araw"] = mat3(L, k + 2).tolist()
            if l.startswith(" A(FC)"): B["A_FC"] = float(l.split()[1])
            if l.startswith(" A(SD)"): B["A_SD"] = [float(x) for x in l.split()[1:4]]
            if l.startswith(" A(ORB+DIA)"): B["A_ORB"] = [float(x) for x in l.split()[1:4]]
            if l.startswith(" A(Tot)"):
                v = l.split(); B["A_pr"] = [float(x) for x in v[1:4]]; B["A_iso"] = float(v[-1])
        out["blocks"].append(B)
    return out

if __name__ == "__main__":
    res = {}
    for mol, d in (("TEMPO", "T/calc"), ("PTEMPO", "P/calc")):
        res[mol] = [parse(f) for f in sorted(glob.glob(os.path.join(d, "*.log")))]
    json.dump(res, open("parsed.json", "w"))
    for mol, R in res.items():
        nb = [len(r["blocks"]) for r in R]
        print(mol, len(R), "files; blocks per file:", {k: nb.count(k) for k in set(nb)},
              "abnormal:", [r["file"] for r in R if not r["normal"]])
        print("  missing keys:", [(r["file"], b["density"]) for r in R for b in r["blocks"]
                                  if not all(k in b for k in ("gmat", "g_iso", "Araw", "A_iso", "A_SD", "gauge"))])
