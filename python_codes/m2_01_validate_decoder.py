#!/usr/bin/env python3
"""
m2_01_validate_decoder.py
Validates the stand-alone XTC decoder: for every frame of every trajectory,
all 10,293 bonded distances are compared with the force-field equilibrium
lengths.  A decoding error would produce gross deviations.
Also identifies the heating segment (frames with restrained heavy atoms).
"""
import numpy as np, os, json
from m2_00_io import *
P = read_parm7(f"{RAW}/300K/rep1/TEMPO-SAM-11.parm7")
b, req = P["bonds"], P["bond_req"]
isH = np.array([n.startswith("H") for n in P["name"]])
heavyb = ~(isH[b[:,0]] | isH[b[:,1]])
report = {}
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])
    d = np.linalg.norm(X[:, b[:,0]] - X[:, b[:,1]], axis=2)
    dev = d - req
    report[f"T{T}_r{r}"] = dict(frames=int(X.shape[0]),
        max_abs_dev_all=float(np.abs(dev).max()),
        rms_dev_heavy=float(np.sqrt((dev[:, heavyb]**2).mean())),
        max_abs_dev_Hbonds=float(np.abs(dev[:, ~heavyb]).max()))
    print(T, r, report[f"T{T}_r{r}"])
json.dump(report, open(os.path.join(ROOT, "data", "module2", "decoder_validation.json"), "w"), indent=1)

# ---- same statistic restricted to bonds of intact molecules (defects excluded)
from m2_00_io import DEFECT_IDS as DEFECT   # the 14 molecules of the 7 interpenetrating pairs (marked/characterized only)
okb = ~np.isin(P["resid"][b[:, 0]], DEFECT)
rep2 = {}
for T, r in RUNS:
    X = load_traj(run_prefix(T, r), P["natom"])
    dev = np.linalg.norm(X[:, b[okb, 0]] - X[:, b[okb, 1]], axis=2) - req[okb]
    hb = heavyb[okb]
    rep2[f"T{T}_r{r}"] = dict(rms_dev_heavy_intact=float(np.sqrt((dev[:, hb]**2).mean())),
                             max_abs_dev_heavy_intact=float(np.abs(dev[:, hb]).max()),
                             max_abs_dev_H_intact=float(np.abs(dev[:, ~hb]).max()))
    print(T, r, rep2[f"T{T}_r{r}"])
json.dump(rep2, open(os.path.join(ROOT, "data", "module2", "decoder_validation_intact.json"), "w"), indent=1)
