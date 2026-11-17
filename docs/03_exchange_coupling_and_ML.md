# 3. Module III — exchange coupling, machine learning and the exchange network

**Goal.** Obtain the isotropic exchange coupling J for every pair of radicals in every MD frame, together with its uncertainty and its ferro- or antiferromagnetic character, and from these build time-resolved exchange networks of the monolayer.

```
pair library ──► broken-symmetry DFT J ──► descriptors ──► ML benchmark ──► final model
                                                                              │
MD head-group coordinates (Module II) ───────────────────────────────────────►├──► J(t) for all 9870 pairs
                                                                              └──► network statistics
```

---

## 3.1 Pair library (`simulation_inputs/ModuleIII/`)

| Step | Script | Details |
|---|---|---|
| Generation | `struct_gen_ml_new.py` | Two copies of the ωB97X-D-optimized TEMPO monomer (`TEMPO-wB97XD.xyz`, 29 atoms). O₂ is placed at O₁ + d·p, with d = 2.0–5.8 Å in 0.2 Å steps (500 directions p on a Fibonacci sphere) and 6–10 Å in 1 Å steps (50 directions). The second molecule is then rotated about O₂ by a uniformly distributed random rotation (Shoemake). Random seed 12345; 10 250 structures. |
| Contact filter | `filter_str_ml.py` | Rejects any intermolecular atom pair closer than 0.8 × (sum of Bondi radii: H 1.30, C 1.70, N 1.55, O 1.50 Å). **1364 pairs retained.** |
| Atom ordering | `process_xyz.py` | Moves the two O atoms to the top of each file before the DFT run. Coordinates are unchanged. |
| Summary | `distribution_ml.py` | O···O, O₁–O₂–N₂ angle and N₁–O₁–O₂–N₂ dihedral of each structure |

The 1364 geometries, in their original atom order, are in `pair_geometries_1364.tar.gz`. `m3_00_build_dataset.py` unpacks the archive automatically.

## 3.2 Broken-symmetry DFT (`make_inp.sh`, `orca-job-all.sh`)

ORCA 5.0.1 input, identical for all pairs (example: `struct_009867.inp`):

```
! UKS B2PLYP D3BJ EPR-II AUTOAUX TIGHTSCF
%maxcore 4000
%pal nprocs 40 end
%scf
   BrokenSym 1,1
end
* xyzfile 0 3 struct_009867.xyz
```

The calculation starts from the triplet (high-spin) configuration, and ORCA then flips one unpaired electron to obtain the broken-symmetry solution. The job script extracts the line `| J(3) =` from each output, which is the Yamaguchi estimate in cm⁻¹:

$$J=-\frac{E_\mathrm{HS}-E_\mathrm{BS}}{\langle S^2\rangle_\mathrm{HS}-\langle S^2\rangle_\mathrm{BS}},\qquad \hat H=-2J\,\hat{\mathbf S}_1\cdot\hat{\mathbf S}_2 .$$

J > 0 is ferromagnetic (F) and J < 0 antiferromagnetic (AF). All values are collected in `J-coupling.dat` (pair name, J).

**Dataset facts:**
- J ranges from −70.9 to +22.1 cm⁻¹: 994 AF, 306 F, and 64 that round to zero.
- **Numerical resolution:** the 127 pairs with O···O ≥ 8 Å, where exchange should be negligible, give J = −0.03 ± 0.03 cm⁻¹. Couplings below **J_res = 0.10 cm⁻¹** (≈ 3σ) are therefore treated as unresolved.

**Conversion to the convention of Tesi et al.** They write Σ J_ij Ŝᵢ·Ŝⱼ, so their |J| is at least twice ours.

## 3.3 Descriptors (`m3_features.py`, `m3_02_features_training.py`)

Each radical is represented by nine key atoms and its twelve methyl hydrogens:
- the nine key atoms are N, O, C_α1, C_α2, C4 and the four methyl carbons;
- the same atoms are identified in the MD head groups (N1, O1, C2, C3, C5, C6–C9; H6–H17).

All 61 descriptors are invariant to rotation, translation, exchange of the two radicals and permutation of equivalent atoms. They fall into four families:

| Family | Number | Definition |
|---|---|---|
| N–O geometry | 10 | r(O···O), r(N···N), the two cross N···O distances (sorted), the two N–O···O angles (sorted), cos of the N–O···O–N dihedral, \|cos\| of the angle between the two π axes, \|cos\| of the angles between each π axis and the O···O vector (sorted) |
| Contacts | 42 | for each of the 21 unordered pairs of atom types (N, O, C_α, C4, C_Me, H_Me): minimum distance and soft count Σ exp[−(r − 3 Å)/0.5 Å] |
| π\* (SOMO) overlap | 6 | overlap S of the two N–O π\* orbitals, S², log₁₀S², and the N–N, O–O and N–O components |
| (baseline) generation coordinates | 3 | O···O, O₁–O₂–N₂ angle, N₁–O₁–O₂–N₂ dihedral (not exchange-symmetric) |

**The π\* overlap.** The SOMO of each radical is modeled as φ = c_N p_N − c_O p_O, with c_N² = 0.45 and c_O² = 0.55. The p orbitals are STO-3G 2p functions (ζ_N = 1.95, ζ_O = 2.25) oriented along the normal to the C_α–N–C_α plane. For two contracted p functions with directions **u** and **v** centered at **A** and **B** (R = A − B):

$$\langle p_u^A|p_v^B\rangle=\sum_{ij}c_ic_jN_iN_j\left(\frac{\pi}{p}\right)^{3/2}e^{-\mu R^2}\left[\frac{\mathbf u\cdot\mathbf v}{2p}-\frac{a_ib_j}{p^2}(\mathbf u\cdot\mathbf R)(\mathbf v\cdot\mathbf R)\right],$$

with p = aᵢ + bⱼ and μ = aᵢbⱼ/p. In orbital models of exchange (Hay–Thibeault–Hoffmann, Kahn), the antiferromagnetic contribution scales with S².

**Transforms.** Before modeling, the soft counts and overlaps are log-transformed with log₁₀(x + 10⁻²⁰), as in `m3_ml_common.prepare()`. This is deterministic, so no information leaks between training and test data.

## 3.4 Machine learning

**Target.** The magnitude model is trained on y = asinh(|J|/J₀) with J₀ = 0.05 cm⁻¹. This transform is linear below the resolution and logarithmic above it; the inverse is |J| = J₀ sinh y.

**Split.** A stratified 15 % hold-out set (205 pairs; strata are the O···O band and the sign of J) is set aside once (`data/module3/split.json`) and is not used for any modeling choice.

**Benchmark (`m3_03_benchmark_models.py`).** Nested cross-validation on the remaining 85 %: five outer folds, three inner folds, and a grid search on the MSE of y. Eleven regressors were compared:

| Model | Nested-CV R² (mean ± SD) |
|---|---|
| Extremely randomized trees | 0.70 ± 0.03 |
| Random forest | 0.70 ± 0.04 |
| Gaussian process (Matérn ν = 5/2 + white noise) | 0.68 ± 0.06 |
| Histogram gradient boosting | 0.67 ± 0.04 |
| SVR (RBF) | 0.66 ± 0.05 |
| Kernel ridge (Laplacian) | 0.66 ± 0.05 |
| MLP (64-64 or 128-64-32) | 0.66 ± 0.07 |
| Kernel ridge (RBF) | 0.66 ± 0.07 |
| k-nearest neighbors | 0.65 ± 0.04 |
| Ridge, quadratic features | 0.60 ± 0.05 |
| Ridge | 0.43 ± 0.08 |

The same protocol run on the descriptor families shows which ones carry the information:
- generation coordinates: ≈ 0.47;
- N–O geometry: ≈ 0.61;
- SOMO overlap: ≈ 0.57;
- contacts: 0.67–0.70;
- all 61: 0.66–0.70.

Predicting signed J directly (environment variable `M3_TARGET=signed`) gives R² ≤ 0.

**Sign (`m3_04_sign_classification.py`, `m3_04b_sign_permutation.py`).**
- Eight classifiers on two inputs (the 61 descriptors, or the 841 symmetrized inverse interatomic distances) were evaluated by nested CV on the training pairs with |J| > J_res.
- They reach AUC = 0.50–0.63, and a label-permutation test gives AUC = 0.50 ± 0.04 (p < 0.001). The signal is real but weak, with a Brier skill score of at most 0.05.
- Where the sign information lies: F couplings occur almost only at direct N/O···N/O contacts, while through-contact couplings beyond 5.5 Å are 86–100 % AF.

**Final model (`m3_05_final_model.py`).**

- **Magnitude:** extremely randomized trees (500 trees, max_features = 0.33, min_samples_leaf = 3) on all 61 descriptors.
- **Uncertainty:** Mondrian split-conformal intervals (68 % and 90 %), calibrated on out-of-fold residuals in four bins of the prediction.
- **Sign:** logistic regression for P_AF on [predicted y, closest N/O···N/O distance], fitted to pairs with |J| > J_res using out-of-fold magnitudes.
- **Expected coupling:** ⟨J⟩ = |J| (1 − 2P_AF).
- **Applicability domain:** the mean standardized distance to the five nearest training pairs, with the 99th percentile of the training set as threshold.
- **Committee:** RF, GPR and KRR predictions provide a model-spread estimate.
- After the hold-out evaluation, all models are refitted to all 1364 pairs and saved to `data/module3/final_model.joblib`. The file is not stored in the repository (31 MB); it is regenerated in about 3 minutes.

**Hold-out results (205 pairs):**

| Model | R² (asinh) | Within ×3 (\|J\| > 0.1) | Within ×10 |
|---|---|---|---|
| Extremely randomized trees | 0.77 | 70 % | 100 % |
| Random forest | 0.77 | 64 % | 100 % |
| GPR | 0.75 | 66 % | 99 % |
| KRR (Laplacian) | 0.73 | 64 % | 99 % |
| exp(−r_NO) baseline | 0.31 | 61 % | 93 % |
| S² + O···H baseline | 0.32 | 64 % | 94 % |

- **Interval coverage:** 73 % at the 68 % level and 88 % at the 90 % level.
- **Sign model:** AUC = 0.74 (95 % CI 0.61–0.85) on the hold-out set, 0.62 in cross-validation.
- **Learning curves:** the magnitude levels off above about 600 training pairs, whereas the sign AUC is still rising, so a larger DFT library would mainly improve the sign.

## 3.5 Application to the trajectories (`m3_01`, `m3_06`)

1. **`m3_01_extract_md_headgroups.py`** extracts the nine key atoms and twelve methyl hydrogens of all 141 radicals from the six production trajectories. This step needs the decoded trajectories (Module II).
2. **`m3_06_apply_md.py`** loops over all 141C2 = 9870 pairs in each of the 500 frames of the last 5 ns:
   - pairs whose closest N/O···N/O distance exceeds 9 Å are set to J = 0;
   - all other pairs get |J|, P_AF, ⟨J⟩, conformal half-widths, the applicability-domain distance and, every 10th frame, the committee spread;
   - results go to `data/module3/mdJ_T{T}_r{r}.npz`.
3. **Domain check:** 99.3–99.7 % of MD pairs closer than 6 Å lie inside the applicability domain, and the committee spread is 0.11–0.12 in y (about 12 % in |J|).

## 3.6 Network analysis (`m3_07_exchange_network.py`, `m3_08_sensitivity.py`)

**Per pair (intact radicals only):**
- ⟨|J|⟩;
- ⟨J⟩;
- |J|-weighted P_AF;
- the fraction of frames with |J| > J_res;
- the model interval, applied frame by frame and averaged (errors assumed fully correlated);
- for persistently resolved pairs, the integrated autocorrelation time τ_J of |J|(t), the SD σ_2J of the singlet–triplet gap, and ξ_J = 2πσ_2Jτ_J.

**Per radical (resolved couplings, ⟨|J|⟩ > J_res, only):**
- number of partners;
- Σⱼ⟨|J_ij|⟩;
- mean-field constant θᵢ = 2S(S+1)ΣⱼJ_ij / 3k_B = 0.5 ΣⱼJ_ij / k_B;
- rms exchange frequency ω_ex = [Σⱼ(2J_ij/h)²]^{1/2}, compared with the local dipolar width √M₂;
- whether the strongest singlet–triplet gap exceeds the Q-band quantum (2|J| > hν, ν = 33.9 GHz);
- connected clusters of the graph of resolved couplings.

**Sensitivity (`m3_08`).** The statistics are repeated for J_res = 0.05, 0.10, 0.20 and 0.30 cm⁻¹, for interior radicals only, and with a cutoff of 8 instead of 9 Å. The cutoff has no effect on any statistic.

**Main numbers** (mean ± SD over replicas):

| | 100 K | 300 K |
|---|---|---|
| Resolved pairs | 63 ± 2 | 51 ± 7 |
| Resolved partners per radical | 0.99 ± 0.03 | 0.81 ± 0.11 |
| Radicals without resolved partner | 32 ± 1 % | 46 ± 6 % |
| Largest connected cluster | 9.7 ± 0.9 % | 7.1 ± 2.8 % |
| Median resolved ⟨\|J\|⟩ | 0.21 ± 0.01 cm⁻¹ | 0.17 ± 0.01 cm⁻¹ |
| AF share of resolved couplings | 76 ± 1 % | 75 ± 1 % |
| ω_ex/√M₂ (coupled radicals, median) | 60 ± 4 | 52 ± 3 |
| Time fraction resolved (resolved pairs) | 84 ± 2 % | 59 ± 2 % |
| ξ_J (persistent pairs, median) | ≤ 0.5 ± 0.2 | 4 ± 3 |

## 3.7 Outputs

| File | Content |
|---|---|
| `data/module3/pairs_J.csv` | name, J, O···O of the 1364 pairs |
| `data/module3/pairs_keyatoms.npz` | key-atom coordinates of the pairs |
| `data/module3/train_features.csv` | 61 descriptors + baseline coordinates + J |
| `data/module3/split.json` | hold-out partition |
| `data/module3/benchmark_*_nestedcv_folds.csv`, `cvpred_*.npy` | benchmark per fold |
| `data/module3/sign_classification_nestedcv.csv`, `sign_permutation_test.json`, `signcv_*.npy` | sign analysis |
| `data/module3/final_test_metrics.json`, `test_predictions.csv`, `train_oof.csv` | hold-out evaluation |
| `data/module3/perm_importance*.csv`, `learning_curves.csv` | interpretation |
| `data/module3/mdJ_T{T}_r{r}.npz` | J(t) for every pair that comes within 9 Å (see the data dictionary) |
| `data/module3/md_spins_T{T}_r{r}.npz` | spin positions (N–O midpoints), last 5 ns |
| `data/module3/exchange_pairs_*.csv`, `exchange_spins_*.csv` | pair- and radical-level results |
| `data/module3/exchange_summary*.csv`, `exchange_by_class*.csv`, `exchange_sensitivity*.csv`, `exchange_matrices.npz` | summaries and 141 × 141 matrices |
| `Figures_MS/Figure4_exchange_ML.*`, `Figure5_exchange_network.*`, `Figures_ESI/FigS_exchange_*` | from `m3_09`, `m3_10` |
| `tables/tabS_m3_*.tex` | from `m3_11` |
