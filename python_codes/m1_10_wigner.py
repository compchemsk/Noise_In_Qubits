#!/usr/bin/env python3
"""
m1_10_wigner.py
Statistics and figure for the Wigner-sampled geometries of TEMPO and PTEMPO
(10 000 samples each): N-O bond, the two O-N-C angles and N pyramidalization.
The dihedral column is NOT compared because its atom definition (C13) refers to
different atoms in the two molecules.
"""
import os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene, ks_2samp, gaussian_kde
from plot_style import apply_style, finish_panel, save_fig, COLORS
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data","module1")
apply_style()
W={m:pd.read_csv(os.path.join(D,"raw",f"Wigner_{m}_intcoord.dat"),sep=r"\s+") for m in ["TEMPO","PTEMPO"]}
for m,d in W.items():
    d["ONC_mean"]=0.5*(d["O-N-C2"]+d["O-N-C3"]); d["ONC_diff"]=d["O-N-C2"]-d["O-N-C3"]
cols={"N-O":("N–O bond (Å)",3),"ONC_mean":("mean O–N–C angle (°)",2),"ONC_diff":("O–N–C2 − O–N–C3 (°)",2),"N_pyramidalization":("N pyramidalization (Å)",3)}
rows=[]
for c,(lab,nd) in cols.items():
    a,b=W["TEMPO"][c],W["PTEMPO"][c]
    rows.append(dict(coordinate=c,TEMPO_mean=a.mean(),TEMPO_sd=a.std(),PTEMPO_mean=b.mean(),PTEMPO_sd=b.std(),
                     sd_ratio=b.std()/a.std(),levene_p=levene(a,b).pvalue,ks_D=ks_2samp(a,b).statistic))
st=pd.DataFrame(rows); st.to_csv(os.path.join(D,"module1_wigner_stats.csv"),index=False)
pd.set_option("display.width",200); print(st.to_string(float_format=lambda v:f"{v:.4g}"))
r=[(m,np.corrcoef(d["N-O"],d["N_pyramidalization"])[0,1]) for m,d in W.items()]; print("corr(N-O, pyr):",r)
fig,axs=plt.subplots(2,2,figsize=(14,11)); fig.subplots_adjust(hspace=0.4,wspace=0.32)
for ax,(c,(lab,nd)),L in zip(axs.flat,cols.items(),"abcd"):
    lo=min(W["TEMPO"][c].quantile(.001),W["PTEMPO"][c].quantile(.001)); hi=max(W["TEMPO"][c].quantile(.999),W["PTEMPO"][c].quantile(.999))
    x=np.linspace(lo,hi,400)
    for m,col in [("TEMPO",COLORS[0]),("PTEMPO",COLORS[1])]:
        k=gaussian_kde(W[m][c]); y=k(x)
        ax.plot(x,y,color=col,lw=3,label=m); ax.fill_between(x,0,y,color=col,alpha=0.22)
    ax.set_xlabel(lab); ax.set_ylabel("Probability density"); ax.set_ylim(0, ax.get_ylim()[1]*1.35); ax.legend(loc="upper left", fontsize=16, ncol=2, columnspacing=0.8, handlelength=1.4); finish_panel(ax,L)
save_fig(fig,os.path.join(HERE,"..","Figures_ESI","FigS_wigner_TEMPO_PTEMPO"))
