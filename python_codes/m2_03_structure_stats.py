#!/usr/bin/env python3
"""
m2_03_structure_stats.py
Replica-resolved structural statistics of the TEMPO monolayer.

Analysis window: last 5 ns of each trajectory (frames 500-999 of the
unrestrained segment).  For every observable the value is first averaged
over molecules and frames within each replica; the reported uncertainty is
the standard deviation over the three independent replicas (n = 3).
Molecular subsets:
  all       141 molecules
  interior  S atom >= 5 A inside the convex hull of the S atoms
  edge      S atom <  5 A from the hull boundary (outermost row)
  intact    excludes the 14 molecules with a distorted TEMPO ring (see m2_01)
Also writes the first-5-ns averages to test stationarity.
"""
import os, json, numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from m2_00_io import *
D = os.path.join(ROOT, "data", "module2")
from m2_00_io import DEFECT   # empty by default: all 141 molecules are analysed
WIN = {"last5": slice(500, 1000), "first5": slice(0, 500)}
CONTACT = 3.5          # O1-Au distance (A) regarded as direct contact

def groups(d):
    S = d["S_yz"].mean(0); hull = ConvexHull(S)
    dist = np.min(-(S @ hull.equations[:, :2].T + hull.equations[:, 2]), axis=1)
    intact = ~np.isin(d["mol_ids"], DEFECT)
    return {"all": np.ones(len(S), bool), "interior": dist >= 5, "edge": dist < 5,
            "intact": intact, "intact_interior": intact & (dist >= 5)}, dist

def observables(d, w, m):
    ang = np.degrees(np.arccos(np.clip(d["cos_NO"][w][:, m], -1, 1)))
    c = d["cos_NO"][w][:, m]
    contact = d["dmin_OAu"][w][:, m] < CONTACT
    return {
        "h_S (A)": d["h_S"][w][:, m].mean(),
        "h_spin (A)": d["h_spin"][w][:, m].mean(),
        "sd_h_spin (A)": d["h_spin"][w][:, m].std(),
        "theta_NO (deg)": ang.mean(),
        "P2_NO": (1.5 * c**2 - 0.5).mean(),
        "frac_NO_up (theta<60)": (ang < 60).mean(),
        "frac_NO_down (theta>120)": (ang > 120).mean(),
        "tilt (deg)": d["tilt"][w][:, m].mean(),
        "gauche": d["gauche"][w][:, m].mean(),
        "contact_frac (O-Au<3.5)": contact.mean(),
        "molecules_in_contact": float((contact.mean(0) > 0.5).sum()),
        "nn1 (A)": d["nn1"][w][:, m].mean(),
        "nn2 (A)": d["nn2"][w][:, m].mean(),
        "n_within_6A": d["ncount_6"][w][:, m].mean(),
        "n_within_8A": d["ncount_8"][w][:, m].mean(),
        "frac_nn1<5A": (d["nn1"][w][:, m] < 5).mean(),
    }

rows = []
meta = {}
for T, r in RUNS:
    d = np.load(os.path.join(D, f"obs_T{T}_r{r}.npz"))
    G, dist = groups(d)
    meta[f"T{T}_r{r}"] = {k: int(v.sum()) for k, v in G.items()}
    for wn, w in WIN.items():
        for g, m in G.items():
            for k, v in observables(d, w, m).items():
                rows.append(dict(T=T, rep=r, window=wn, group=g, observable=k, value=float(v)))
df = pd.DataFrame(rows); df.to_csv(os.path.join(D, "module2_structure_by_replica.csv"), index=False)
summ = (df.groupby(["T", "window", "group", "observable"]).value
          .agg(mean="mean", sd="std", n="count").reset_index())
summ.to_csv(os.path.join(D, "module2_structure_summary.csv"), index=False)
json.dump(meta, open(os.path.join(D, "module2_group_sizes.json"), "w"), indent=1)
pd.set_option("display.width", 220, "display.max_rows", 400)
piv = summ[summ.window == "last5"].pivot_table(index=["group", "observable"], columns="T", values=["mean", "sd"])
print(meta["T300_r1"])
print(piv.round(3).to_string())
# stationarity: last5 minus first5, relative to replica SD
st = summ[summ.group == "all"].pivot_table(index="observable", columns=["T", "window"], values="mean")
print("\nfirst 5 ns vs last 5 ns (all molecules):"); print(st.round(3).to_string())
