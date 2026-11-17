#!/usr/bin/env python3
"""
m1_08_thermal_modulation.py
With Q a dimensionless normal coordinate (H_k = hbar w_k (P^2+Q^2)/2), the
thermal mean-square displacement is <Q_k^2> = n_k + 1/2.  The root-mean-square
modulation of X (= A_iso or the Q-band line position nu_g) produced by mode k is
    dX_k,rms(T) = |dX/dQ_k| sqrt(n_k(T) + 1/2),
and the total over a set of modes is the quadrature sum (independent modes).
Also reports mode-matched comparisons in common frequency windows.
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data","module1")
kB=0.6950348
s=pd.read_csv(os.path.join(DATA,"module1_coupling_summary.csv"))
f=pd.read_csv(os.path.join(DATA,"module1_frequencies.csv"))
s=s.merge(f[["molecule","mode","freq_cm1"]],on=["molecule","mode"])
def nT(E,T): return 1/np.expm1(E/(kB*T))
rows=[]
for T in [10,60,300]:
    for r in s.itertuples():
        occ=nT(r.freq_cm1,T)+0.5
        rows.append(dict(T=T,molecule=r.molecule,mode=r.mode,freq=r.freq_cm1,occ=occ,
            dA_rms=abs(r.dA_dQ_MHz)*np.sqrt(occ), dg_rms_Q=abs(r.dnu_g_Q_MHz)*np.sqrt(occ)))
m=pd.DataFrame(rows); m.to_csv(os.path.join(DATA,"module1_thermal_modulation_by_mode.csv"),index=False)
windows={"all computed":(0,1e9),"< 90 cm-1":(0,90),"190-210 cm-1":(190,210)}
out=[]
for T in [10,60,300]:
    for mol in ["TEMPO","PTEMPO","PTEMPO-alkyl"]:
        for w,(lo,hi) in windows.items():
            d=m[(m["T"]==T)&(m.molecule==mol)&(m.freq>=lo)&(m.freq<hi)]
            out.append(dict(T=T,molecule=mol,window=w,n_modes=len(d),
                 A_rms_MHz=np.sqrt((d.dA_rms**2).sum()),gQ_rms_MHz=np.sqrt((d.dg_rms_Q**2).sum())))
o=pd.DataFrame(out); o.to_csv(os.path.join(DATA,"module1_thermal_modulation_summary.csv"),index=False)
pd.set_option("display.width",200); print(o.to_string(float_format=lambda v:f"{v:.3f}"))
print(m[(m["T"]==300)].sort_values(["molecule","freq"]).to_string(float_format=lambda v:f"{v:.3f}"))
