#!/usr/bin/env python3
"""
m1_09_mode_character.py
Relates the composition of each low-frequency mode (fractional participation of
NO, ring, methyl H, methyl C and -- for PTEMPO -- propargyl atoms, from
vibcontrib*.dat) to its first-order spin--phonon coefficients.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data","module1")
def read_vc(path, mol):
    rows=[l.split() for l in open(path).read().strip().split("\n")[1:]]
    cols=["mode","freq","forceC","IR","zp_len_cm","NO","Ring","MetH","MetC"]+(["Prop"] if len(rows[0])>9 else [])
    df=pd.DataFrame([[float(x) for x in r] for r in rows],columns=cols); df["mode"]=df["mode"].astype(int); df["molecule"]=mol
    return df
vc=pd.concat([read_vc(os.path.join(D,"raw","vibcontrib_TEMPO.dat"),"TEMPO"),
              read_vc(os.path.join(D,"raw","vibcontrib_PTEMPO.dat"),"PTEMPO")],ignore_index=True)
vc.to_csv(os.path.join(D,"module1_mode_composition_all.csv"),index=False)
s=pd.read_csv(os.path.join(D,"module1_coupling_summary.csv"))
m=s.merge(vc,on=["molecule","mode"],how="left")
m["absA"]=m.dA_dQ_MHz.abs(); m["absg"]=m.dg_dQ.abs()
cols=["molecule","mode","freq","NO","Ring","MetH","MetC","Prop","absA","absg"]
pd.set_option("display.width",200); print(m[m.molecule!="PTEMPO-alkyl"][cols].to_string(float_format=lambda v:f"{v:.4g}"))
sub=m[m.molecule!="PTEMPO-alkyl"]
res=[]
for c in ["NO","Ring","MetH","MetC"]:
    for y in ["absA","absg"]:
        r,p=spearmanr(sub[c],sub[y]); res.append(dict(component=c,target=y,spearman=r,p=p,n=len(sub)))
r=pd.DataFrame(res); print(r.to_string(float_format=lambda v:f"{v:.3f}")); r.to_csv(os.path.join(D,"module1_composition_correlation.csv"),index=False)
m[cols].to_csv(os.path.join(D,"module1_mode_character.csv"),index=False)
# modes below 200 cm-1 not treated
for mol in ["TEMPO","PTEMPO"]:
    t=set(s[s.molecule==mol]["mode"]); v=vc[(vc.molecule==mol)&(vc.freq<=210)]
    print(mol,"untreated modes <=210 cm-1:",v[~v["mode"].isin(t)][["mode","freq","NO","MetH"]].to_dict("records"))
