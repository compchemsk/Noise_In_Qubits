# 4. Running the analysis

All scripts live in `python_codes/` and are run from that folder. Each one documents its inputs, method and outputs in its header docstring, and every random step uses a fixed seed. Re-running a script therefore reproduces the stored data. For example, the radical-class labels and the structural summaries are reproduced exactly, and the machine-learning metrics agree to the last printed digit.

## 4.1 Three levels of reproduction

| Level | What you need | What you get | Command |
|---|---|---|---|
| **A. From processed data** | this repository | every figure, table and statistic (all three modules) | `bash run_all.sh` |
| **B. From the DFT results** | this repository | the whole ML part from the 1364 DFT couplings, and the spin–phonon analysis from the computed scans | included in level A |
| **C. From raw trajectories** | the six XTC files (~23 GB, available on request) | per-frame observables, pair geometries, bath, dipolar couplings, J(t) | see 4.3 |

Redoing the electronic-structure calculations (Gaussian, ORCA) or the MD (AMBER) is described in docs 01–03 together with the inputs in `simulation_inputs/`.

## 4.2 Level A/B: `run_all.sh`

```bash
bash run_all.sh          # everything below, in order
bash run_all.sh m1       # Module I
bash run_all.sh m2       # Module II statistics from data/module2/obs_*.npz
bash run_all.sh m3ml     # Module III dataset, descriptors, benchmark, sign analysis, final model
bash run_all.sh net      # exchange network from data/module3/mdJ_*.npz + all figures and tables
```

Approximate run times on two CPU cores:

| Stage | Time |
|---|---|
| Module I | < 2 min |
| Module II statistics | ~5 min |
| Module III ML | ~25 min (benchmark ~10 min, sign classifiers ~10 min, final model ~3 min) |
| Network, figures, tables | ~5 min |

## 4.3 Level C: from the raw trajectories

```bash
# 1. place the trajectories
#    md_raw/100K/rep1/TEMPO-SAM-11.xtc ... md_raw/300K/rep3/TEMPO-SAM-11.xtc
# 2. decode and validate
cd python_codes
bash m2_00b_decode_trajectories.sh
python m2_01_validate_decoder.py
# 3. Module II observables and trajectory-based analyses
python m2_02_extract_observables.py     # -> data/module2/obs_T*_r*.npz
python m2_07_pairs.py
python m2_08_convergence.py
python m2_13_spin_bath.py
python m2_14_dipolar_modulation.py
# 4. Module III on the trajectories
python m3_01_extract_md_headgroups.py   # -> data/module3/md_keyatoms_*.npz, md_spins_*.npz
python m3_06_apply_md.py                # -> data/module3/mdJ_T*_r*.npz   (needs final_model.joblib from m3_05)
# 5. everything downstream
cd .. && bash run_all.sh m2 && bash run_all.sh net
```

## 4.4 Script reference

### Module I — spin–phonon coupling

| Script | Input | Output |
|---|---|---|
| `m1_00_displace_along_modes.py` | Gaussian frequency output | displaced geometries in the dimensionless coordinate Q (for new calculations) |
| `m1_01_extract_scans.py` | `simulation_inputs/ModuleI/spin_phonon_scans_notebook.py` | `module1_scans.csv` |
| `m1_02_fit_couplings.py` | scans | quadratic and cubic fits, central differences, outlier test, g → line-shift conversion |
| `m1_03_relaxation_models.py` | couplings + experimental T₁ | relaxation-model fits (χ², AICc) |
| `m1_04_mode_selection.py` | couplings + T₁ | single-mode and non-negative multi-mode fits |
| `m1_05_figures.py` | all of the above | Figure 1, SI scan figures, T₁ fits, method benchmark |
| `m1_06_summary_numbers.py` | all of the above | every number quoted in the text |
| `m1_07_latex_tables.py` | all of the above | SI tables |
| `m1_08_thermal_modulation.py` | couplings, frequencies | rms modulation at 10, 60, 300 K |
| `m1_09_mode_character.py` | `vibcontrib_*.dat` | fragment participation and its correlation with the couplings |
| `m1_10_wigner.py` | `Wigner_*_intcoord.dat` | local-geometry statistics and figure |

### Module II — molecular dynamics

| Script | Needs raw MD? | Output |
|---|---|---|
| `m2_00_io.py` | — | Amber topology parser, trajectory loader |
| `m2_00b_decode_trajectories.sh` + `xtc_reader/xtc2bin.c` | yes | decoded trajectories |
| `m2_01_validate_decoder.py` | yes | bond-length validation, heating segment |
| `m2_02_extract_observables.py` | yes | `obs_T{T}_r{r}.npz` |
| `m2_03_structure_stats.py` | no | structure by group and replica |
| `m2_04_dynamics.py` | no | S², cone angles, C₂(t), τ_e, RMSF |
| `m2_05_S2_populations.py` | no | Gaussian mixtures of cone angles |
| `m2_06_mobility_determinants.py` | no | correlations of mobility with local descriptors |
| `m2_07_pairs.py` | yes | pair distances, orientations, persistence, ν_dd |
| `m2_08_convergence.py` | yes | RMSD, block averages |
| `m2_09_figures.py` | no (uses the stored last frame) | Figure 2 and MD SI figures |
| `m2_10_tables.py` | no | MD SI tables |
| `m2_11_radical_classes.py` | no | radical classes (GMM, k = 4) |
| `m2_12_class_persistence.py` | no | class transitions between the two 5-ns windows |
| `m2_13_spin_bath.py` | yes | proton bath |
| `m2_14_dipolar_modulation.py` | yes | secular dipolar couplings, √M₂, σ_d, τ_d, ξ |
| `m2_15_correlated_motion.py` | no | correlated motion of neighbors |
| `m2_16_figures_classes.py` | no | Figure 3 and class SI figures |
| `m2_17_tables_classes.py` | no | class, bath and dipolar tables |
| `reduce_module2_trajectories.py` | yes (MDAnalysis) | optional: compact NPZ copy of the trajectories |

### Module III — exchange coupling and machine learning

| Script | Output |
|---|---|
| `m3_00_build_dataset.py` | `pairs_J.csv`, `pairs_keyatoms.npz`, `pairs_allatoms.npz` |
| `m3_features.py` | descriptor library (imported) |
| `m3_ml_common.py` | target transform, split, metrics (imported) |
| `m3_01_extract_md_headgroups.py` | MD head-group coordinates (needs raw MD) |
| `m3_02_features_training.py` | `train_features.csv` (with an A↔B symmetry check) |
| `m3_03_benchmark_models.py` | nested-CV benchmark (`M3_TARGET=mag` or `signed`; arguments: models, descriptor sets) |
| `m3_04_sign_classification.py` | sign classifiers |
| `m3_04b_sign_permutation.py` | permutation test of the sign AUC |
| `m3_05_final_model.py` | final model, hold-out metrics, conformal intervals, importance, learning curves |
| `m3_06_apply_md.py` | J(t) for all pairs and frames (needs head-group coordinates) |
| `m3_07_exchange_network.py` | pair, radical and network statistics; 141 × 141 matrices |
| `m3_08_sensitivity.py` | sensitivity to J_res, the cutoff and edge radicals |
| `m3_09_figures_ml.py` | Figure 4 and ML SI figures |
| `m3_10_figures_network.py` | Figure 5 and network SI figures |
| `m3_11_tables.py` | Module III SI tables |

### Shared

| File | Purpose |
|---|---|
| `config.py` | all paths (overridable by environment variables) |
| `plot_style.py` | figure style: large fonts, bright color-blind-aware palette, 600 dpi PNG + vector PDF |
