#!/usr/bin/env python3
"""
m3_13_extra_tables.py -- LaTeX tables for the further network analyses (m3_12) and the
X/Q/W-band dimer fractions. All 141 radicals are used.

Writes tables/tabS_rev_AD.tex, tabS_rev_dilution.tex, tabS_rev_subsets.tex,
       tabS_rev_dipolar.tex, tabS_rev_zfs.tex
"""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
TAB = os.path.join(ROOT, "tables"); D3 = os.path.join(ROOT, "data", "module3")
GHZ = 29.9792458                                    # GHz per cm^-1


def ms(x, d=2, scale=1.0):
    """'0.27 +- 0.02' (as written by m3_12) -> '$0.27\\pm0.02$' with d decimals."""
    m, s = [float(v) * scale for v in str(x).split("+-")]
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def wrap(cols, head, rows):
    return "\n".join([rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head + r"\\", r"\midrule"] + rows +
                     [r"\bottomrule", r"\end{tabular}"]) + "\n"


# applicability domain
A = pd.read_csv(os.path.join(TAB, "tabS_m3_revision_AD.csv")).set_index("T")
lab = [("all_eval", r"All evaluations ($r_\mathrm{N/O}\leq9$~\AA)"), ("resolved_pairs_eval", "Evaluations of resolved pairs"),
       ("rNO_0_6", r"$r_\mathrm{N/O}<6$~\AA"), ("rNO_6_7", r"$6\leq r_\mathrm{N/O}<7$~\AA"),
       ("rNO_7_8", r"$7\leq r_\mathrm{N/O}<8$~\AA"), ("rNO_8_9", r"$8\leq r_\mathrm{N/O}\leq9$~\AA")]
rows = [f"{t} & {ms(A.loc[100, k])} & {ms(A.loc[300, k])}\\\\" for k, t in lab]
open(os.path.join(TAB, "tabS_rev_AD.tex"), "w").write(wrap("lcc", r"Predictions & 100 K & 300 K", rows))

# dilution
Dl = pd.read_csv(os.path.join(TAB, "tabS_m3_revision_dilution.csv")).sort_values(["T", "keep"], ascending=[True, False])
rows = []
for r in Dl.itertuples():
    T = int(float(r.T)); first = (r.keep == 1.0)
    rows.append(f"{str(T) + ' K' if first else ''} & {100 * r.keep:.0f}\\% & {float(str(r.n).split('+-')[0]):.0f} & "
                f"{ms(r.frac_isolated)} & {ms(r.degree)} & {ms(r.frac_dimer_Q)} & {ms(r.sqrtM2_median, 0)} & {ms(r.AF_share)}\\\\")
open(os.path.join(TAB, "tabS_rev_dilution.tex"), "w").write(
    wrap("llcccccc", r"$T$ & Occupancy & Radicals & Uncoupled & Partners per radical & Dimers (Q band) & $\sqrt{M_2}$ (MHz) & AF share", rows))

# subsets
Sb = pd.read_csv(os.path.join(TAB, "tabS_m3_revision_subsets.csv"))
name = {"all intact": "all 141", "without S": "without S", "interior only": "interior only"}
rows = []
for T in (100, 300):
    for k, sub in enumerate(["all intact", "without S", "interior only"]):
        r = Sb[(Sb["T"] == T) & (Sb.subset == sub)].iloc[0]
        rows.append(f"{str(T) + ' K' if k == 0 else ''} & {name[sub]} & {ms(r.n, 0)} & {ms(r.frac_isolated)} & {ms(r.degree)} & {ms(r.frac_dimer_Q)}\\\\")
open(os.path.join(TAB, "tabS_rev_subsets.tex"), "w").write(
    wrap("llcccc", r"$T$ & Subset & Radicals & Uncoupled & Partners per radical & Dimers (Q band)", rows))

# point-dipole models
Dp = pd.read_csv(os.path.join(TAB, "tabS_m3_revision_dipolar_models.csv")).set_index("T")
lab = [("nuNN_mid_median", r"$\nu_\mathrm{dd}$ to nearest neighbor, midpoint (MHz)"),
       ("nuNN_two_median", r"$\nu_\mathrm{dd}$ to nearest neighbor, N/O 50:50 (MHz)"),
       ("sqrtM2_mid_median", r"$\sqrt{M_2}$, midpoint (MHz)"), ("sqrtM2_two_median", r"$\sqrt{M_2}$, N/O 50:50 (MHz)")]
rows = [f"{t} & {ms(Dp.loc[100, k], 0)} & {ms(Dp.loc[300, k], 0)}\\\\" for k, t in lab]
open(os.path.join(TAB, "tabS_rev_dipolar.tex"), "w").write(wrap("lcc", r"Quantity & 100 K & 300 K", rows))

# zero-field splitting and dimer fractions at three microwave frequencies
Z = pd.read_csv(os.path.join(TAB, "tabS_m3_revision_zfs.csv")).set_index("T")
band = {}
for T in (100, 300):
    for nu in (9.5, 35.0, 94.0):
        f = [(2 * pd.read_csv(os.path.join(D3, f"exchange_spins_T{T}_r{r}.csv")).Jmax * GHZ > nu).mean() for r in (1, 2, 3)]
        band[(T, nu)] = f"${np.mean(f):.2f}\\pm{np.std(f, ddof=1):.2f}$"
rows = [f"Median $|D|=\\tfrac32\\nu_\\mathrm{{dd}}$ (MHz) & {ms(Z.loc[100, 'D_median_MHz'], 0)} & {ms(Z.loc[300, 'D_median_MHz'], 0)}\\\\",
        f"Median $2|J|/|D|$ & {ms(Z.loc[100, 'twoJ_over_D_median'], 0)} & {ms(Z.loc[300, 'twoJ_over_D_median'], 0)}\\\\",
        f"Resolved pairs with $2|J|>10|D|$ & {ms(Z.loc[100, 'frac_twoJ_gt_10D'])} & {ms(Z.loc[300, 'frac_twoJ_gt_10D'])}\\\\",
        f"Resolved pairs with $2|J|>a_\\mathrm{{N}}$ (46 MHz) & {ms(Z.loc[100, 'frac_twoJ_gt_aN'])} & {ms(Z.loc[300, 'frac_twoJ_gt_aN'])}\\\\"]
for nu, nm in [(9.5, "X band (9.5 GHz)"), (35.0, "Q band (35 GHz)"), (94.0, "W band (94 GHz)")]:
    rows.append(f"Radicals with $2|J_\\mathrm{{max}}|>h\\nu$, {nm} & {band[(100, nu)]} & {band[(300, nu)]}\\\\")
open(os.path.join(TAB, "tabS_rev_zfs.tex"), "w").write(wrap("lcc", r"Quantity & 100 K & 300 K", rows))
print("extra tables written")
