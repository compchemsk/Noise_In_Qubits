# 1. Module I — intramolecular spin–phonon coupling

**Goal.** Quantify how strongly each low-frequency vibration of the radical modulates its spin Hamiltonian (isotropic g value and ¹⁴N hyperfine coupling), how functionalization changes this, and which vibrations are compatible with the measured temperature dependence of T₁.

**Molecules.**

| Label | Structure | Role |
|---|---|---|
| TEMPO | 2,2,6,6-tetramethylpiperidin-1-oxyl | reference nitroxide |
| PTEMPO | 4-propargyloxy-TEMPO | the radical clicked onto the monolayer |
| PTEMPO-alkyl | PTEMPO after the click reaction: 1,2,3-triazole bearing an undecyl chain capped by CH₃ | model of the radical head in the SAM |

---

## 1.1 Electronic structure

| Step | Program | Level | Notes |
|---|---|---|---|
| Geometry optimization + harmonic frequencies | Gaussian 16 | B3LYP-D3/def2-TZVP | no imaginary frequencies; frequencies unscaled |
| g tensor, ¹⁴N hyperfine tensor | ORCA 5.0.1 | B2PLYP/EPR-II | default spin–orbit mean-field (SOMF) treatment for g |

The magnetic level was chosen by comparing PTEMPO values from four methods with the values derived from experiment:

| Method | g_iso | ΔA_iso (MHz) |
|---|---|---|
| B3LYP | 2.00579 | +10.8 |
| **B2PLYP** | **2.00616** | **−3.1** |
| DSD-PBEP86 | 2.00631 | −8.6 |
| RI-MP2 | 2.00687 | −18.8 |
| experiment (reference) | 2.0063 | 0 |

B2PLYP gives the smallest hyperfine error together with a good g value, so it was used for all spin-Hamiltonian calculations.

Input templates are in `simulation_inputs/ModuleI/templates/`:
- `gaussian_opt_freq.gjf`: optimization and frequencies.
- `orca_epr_g_A.inp`: g and A at one displaced geometry.

## 1.2 Dimensionless normal coordinates and displaced geometries

For a mode of angular frequency ω with mass-weighted unit eigenvector **l**, the dimensionless coordinate Q is defined by

$$\hat H_k=\tfrac12\hbar\omega_k\,(\hat P_k^2+\hat Q_k^2),\qquad x_i=x_i^0+Q\,\sqrt{\hbar/\omega_k}\;l_{k,i}/\sqrt{m_i},$$

so that in thermal equilibrium

$$\langle Q_k^2\rangle=n_k+\tfrac12,\qquad n_k=\left[e^{\hbar\omega_k/k_BT}-1\right]^{-1}.$$

`python_codes/m1_00_displace_along_modes.py` reads a Gaussian frequency output and writes one geometry per mode and displacement:

```bash
python m1_00_displace_along_modes.py PTEMPO_opt_freq.log --modes 1 2 3 4 5 6 7 8 --qmax 1.0 --step 0.2 --out scans_PTEMPO
```

- **Normalization:** it does not matter how the program normalizes the printed displacement vectors. The script converts them to mass-weighted unit vectors.
- **Self-check:** for every mode it prints the relative deviation of Σᵢ mᵢΔxᵢ² from Q²ħ/ω, which should be ~10⁻¹⁶.
- **Testing:** the displacement formula was verified with a synthetic test. Check the parser on your own Gaussian output before production use.

**Modes treated.**

| Molecule | Modes | Frequencies (cm⁻¹) | Selection |
|---|---|---|---|
| TEMPO | 1–3 | 78.3, 117.7, 197.8 | all modes below 200 cm⁻¹ (the next, at 209.8 cm⁻¹, is an almost pure methyl motion) |
| PTEMPO | 1–8 | 29.8 … 207.4 | all modes below 210 cm⁻¹ |
| PTEMPO-alkyl | 1, 6–13, 24 | 5.5 … 196.3 | modes with correlated motion of the NO group and the triazole–alkyl fragment, or a change of spin density |

**Grid.** Each mode was scanned at Q = −1.0, −0.8, …, +1.0 (11 points), and g and A were recomputed with ORCA at every point.

All computed values are stored in `simulation_inputs/ModuleI/spin_phonon_scans_notebook.py`, the exported analysis notebook. `m1_01_extract_scans.py` reads the numbers from that file programmatically, without retyping any of them, and writes `data/module1/module1_scans.csv`.

## 1.3 Spin–phonon coefficients (`m1_02_fit_couplings.py`)

Each scan X(Q), with X = g_iso or A_iso, is fitted by ordinary least squares:

$$X(Q)=c_0+c_1Q+c_2Q^2,\qquad c_1=\left(\frac{\partial X}{\partial Q}\right)_0,\quad c_2=\tfrac12\left(\frac{\partial^2X}{\partial Q^2}\right)_0 .$$

- **Standard errors** come from the residual variance.
- **Adequacy of the quadratic form** is checked three ways:
  - a cubic fit;
  - central differences at |Q| = 0.2 and 1.0,
    $$\left(\partial X/\partial Q\right)_0\approx\frac{X(+\Delta Q)-X(-\Delta Q)}{2\Delta Q};$$
  - R² of the quadratic, which is ≥ 0.999 for all scans except one, at 0.98.
- **Outlier rule:** a point is flagged when it deviates from the quadratic fitted to the remaining points by more than 10 × that fit's rms residual. The rms residual is floored at 3 × the printed precision (10⁻⁷ for g, 10⁻⁴ MHz for A).
  - Because g and A come from the same calculation, a flagged geometry is removed from both scans.
  - Exactly one geometry was flagged: Q = −0.2 of mode 10 of PTEMPO-alkyl. It is listed in `data/module1/module1_excluded_points.csv`.

**Zeeman and hyperfine on one energy scale.** The g derivative is converted into the shift of the resonance line at fixed microwave frequency:

$$\frac{\partial\nu_g}{\partial Q}=\frac{\nu_\mathrm{mw}}{g_\mathrm{iso}}\frac{\partial g}{\partial Q},\qquad \nu_\mathrm{mw}=9.5\ \text{GHz (X band)},\ 35.0\ \text{GHz (Q band)}.$$

**Thermal modulation (`m1_08_thermal_modulation.py`).** The rms modulation produced by mode k, and the quadrature sum over independent modes, are

$$\delta X_k(T)=\left|\frac{\partial X}{\partial Q_k}\right|\sqrt{n_k(T)+\tfrac12},\qquad \delta X(T)=\Big[\sum_k \delta X_k^2\Big]^{1/2}.$$

## 1.4 Mode composition and local geometry

**Mode composition (`m1_09_mode_character.py`).** `vibcontrib_*.dat` gives, for every mode, the fraction of the squared normal-mode displacement carried by each fragment:

$$p_F=\sum_{j\in F}|\mathbf u_{kj}|^2\Big/\sum_j|\mathbf u_{kj}|^2 .$$

- **Fragments:** the NO group, the ring carbons, the methyl hydrogens, the methyl carbons and, in PTEMPO, the propargyl ether.
- **Analysis:** Spearman correlations between p_F and |∂X/∂Q| over the 11 TEMPO and PTEMPO modes. For NO participation and |∂A/∂Q|, ρ = 0.72 (p = 0.012).

**Wigner sampling (`m1_10_wigner.py`).** 10 000 geometries of TEMPO and of PTEMPO were sampled from the harmonic Wigner distribution at 300 K (`Wigner_*_intcoord.dat`). The script compares four quantities between the two radicals:
- the N–O bond;
- the mean of the two O–N–C angles;
- their difference;
- the pyramidalization of N.

The dihedral column is not compared, because its atom definition differs between the two molecules.

## 1.5 Spin–lattice relaxation models (`m1_03_relaxation_models.py`, `m1_04_mode_selection.py`)

**Data.** Experimental T₁ values come from Table 1 of Tesi et al. (Q band, 35 GHz): the slow and fast components of PTEMPO in frozen solution, and the SAM, at T = 7–60 K. The uncertainties are converted exactly as printed.

**Models.** All models fit the rate R = 1/T₁ by weighted least squares and include a direct term aT, which is justified because hν/k_B = 1.7 K < 7 K at 35 GHz:

| Model | R_vib(T) | Free parameters |
|---|---|---|
| one local mode | C n(Δ)[n(Δ)+1] | a, C, Δ |
| Raman power law | bTᵐ | a, b, m |
| DFT modes, A-weighted | C Σₖ wₖ nₖ(nₖ+1), wₖ ∝ (∂A/∂Qₖ)² | a, C |
| DFT modes, g-weighted | wₖ ∝ (∂ν_g/∂Qₖ)² at 35 GHz | a, C |
| DFT modes, both | wₖ ∝ (∂A/∂Qₖ)² + (∂ν_g/∂Qₖ)² | a, C |
| each DFT mode alone | C nₖ(nₖ+1) | a, C |
| all DFT modes, free weights | Σₖ Cₖ nₖ(nₖ+1), Cₖ ≥ 0 (non-negative least squares) | a, {Cₖ} |

**Model comparison.** Models are compared by χ² and by the small-sample Akaike criterion,

$$\mathrm{AICc}=\chi^2+2k+\frac{2k(k+1)}{n-k-1}.$$

**Results.**
- One local mode is preferred for both solution components (Δ = 139 ± 10 and 124 ± 14 cm⁻¹).
- Weighting the DFT modes by their isotropic sensitivities fails (χ² larger by a factor of 9–25).
- The best single computed modes are 116.4 and 158.8 cm⁻¹.

## 1.6 Outputs

| File | Content |
|---|---|
| `data/module1/module1_scans.csv` | every computed point (molecule, mode, quantity, Q, value) |
| `data/module1/module1_fits_all.csv` | c₀, c₁, c₂ (± SE), cubic term, central differences, R², outlier information |
| `data/module1/module1_coupling_summary.csv`, `module1_molecule_summary.csv` | per-mode and per-molecule summaries |
| `data/module1/module1_thermal_modulation_*.csv` | δA and δν_g at 10, 60, 300 K |
| `data/module1/module1_T1_model_comparison.csv`, `module1_T1_model_curves.npz` | relaxation fits |
| `data/module1/module1_mode_selection.csv` | single-mode and NNLS tests |
| `data/module1/module1_mode_composition_all.csv`, `module1_composition_correlation.csv` | mode character |
| `data/module1/module1_wigner_stats.csv` | Wigner statistics |
| `Figures_MS/Figure1_spin_phonon.*`, `Figures_ESI/FigS_scans_*`, `FigS_T1_model_fits`, `FigS_functional_benchmark`, `FigS_wigner_TEMPO_PTEMPO` | figures (`m1_05`, `m1_10`) |

Run order: `m1_01 → m1_02 → m1_03 → m1_04 → m1_08 → m1_09 → m1_10 → m1_05 → m1_06 → m1_07`. All of them finish within a minute.

**Caveat.** The quantities analyzed here are isotropic. They measure how sensitive the spin Hamiltonian is to each vibration, but spin–lattice relaxation is driven by the anisotropic, nonsecular parts of g and A. A full ab initio T₁ requires the tensor derivatives.
