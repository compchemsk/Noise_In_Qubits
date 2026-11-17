# 5. Data dictionary

Every file in `data/` is produced by a script in `python_codes/` (see [04_running_the_analysis.md](04_running_the_analysis.md)).
Conventions: lengths in Å, angles in degrees, J in cm⁻¹ (Ĥ = −2J Ŝ₁·Ŝ₂), couplings ν in MHz, times in ns unless stated otherwise.
`T` is the temperature (100 or 300 K) and `rep` the replica (1–3). Molecule identifiers (`mol`, `mol_ids`) are Amber residue numbers (2401–2541).

## Key array files

| File | Arrays |
|---|---|
| `module2/obs_T{T}_r{r}.npz` | `mol_ids` (141); `time_ns` (1000); per frame × radical: `h_O1`, `h_N1`, `h_S`, `h_spin` (heights above top Au layer), `cos_NO`, `u_NO` (×3), `tilt`, `gauche`, `dmin_OAu`, `spin_yz` (×2), `S_yz` (×2), `nn1`, `nn2`, `ncount_6`, `ncount_8`. Frames = 10 ns after heating, 10 ps apart; the last 500 are the analysis window. |
| `module2/last_frame_rep1.npz` | `T100`, `T300`: all-atom coordinates (12 552 × 3) of the final frame of replica 1 |
| `module3/pairs_keyatoms.npz` | `G` (1364 × 2 × 9 × 3): N, O, Cα1, Cα2, C4, Me1–Me4 of both radicals; `names`; `labels` |
| `module3/md_spins_T{T}_r{r}.npz` | `S` (500 × 141 × 3) N–O midpoints, last 5 ns; `mol_ids`; `intact` |
| `module3/mdJ_T{T}_r{r}.npz` | `pairs` (P × 2, molecule indices 0–140) for the P pairs that come within 9 Å; per frame × pair: `Jabs` (\|J\|), `pAF`, `EJ` = \|J\|(1 − 2P_AF), `yhat` = asinh(\|J\|/0.05), `q68`, `q90` (conformal half-widths in y), `ad` (applicability-domain distance), `rNO` (closest N/O···N/O), `csd` (committee SD, every 10th frame), `evaluated` (bool; False = beyond 9 Å, J = 0); `mol_ids`; `intact` |
| `module3/exchange_matrices.npz` | 141 × 141 replica mean and SD of ⟨J⟩ (`EJ_T*`), ⟨\|J\|⟩ (`Jabs_T*`), P_AF (`pAF_T*`), fraction of time resolved (`fres_T*`); `intact`; `mol_ids` |
| `module3/split.json` | indices (into `pairs_J.csv`) of the training and hold-out pairs |
| `module3/cvpred_{target}_{model}_{set}.npy`, `signcv_*.npy` | out-of-fold predictions of the benchmark (training partition order) |

## Tabular files (columns)

### `data/module1/`

| File | Columns |
|---|---|
| `module1_SAM_vs_alkyl_modes.csv` | `dataset`, `head_mode_cm1`, `chi2`, `chi2_red` |
| `module1_T1_model_comparison.csv` | `dataset`, `model`, `n`, `k`, `chi2`, `chi2_red`, `AICc`, `params`, `param_se`, `dAICc` |
| `module1_composition_correlation.csv` | `component`, `target`, `spearman`, `p`, `n` |
| `module1_coupling_summary.csv` | `molecule`, `mode`, `g0`, `dg_dQ`, `dg_dQ_se`, `d2g_dQ2_half`, `d2g_se`, `A0_MHz`, `dA_dQ_MHz`, `dA_dQ_se`, `d2A_dQ2_half_MHz`, `d2A_se`, `dnu_g_X_MHz`, `dnu_g_Q_MHz`, `ratio_A_over_gQ`, `ratio_A_over_gX`, `R2_g`, `R2_A`, `linshare_g`, `linshare_A` |
| `module1_excluded_points.csv` | `molecule`, `mode`, `quantity`, `Q`, `value` |
| `module1_fits_all.csv` | `molecule`, `mode`, `quantity`, `n_points`, `Qmin`, `Qmax`, `c0`, `c1`, `c1_se`, `c2`, `c2_se`, `R2_quad`, `rmse_quad`, `c3`, `c3_se`, `R2_cubic`, `cd_0p2`, `cd_1p0`, `linear_share`, `span`, `n_excluded`, `c1_with_outliers`, `R2_with_outliers` |
| `module1_frequencies.csv` | `molecule`, `mode`, `freq_cm1`, `source` |
| `module1_mode_character.csv` | `molecule`, `mode`, `freq`, `NO`, `Ring`, `MetH`, `MetC`, `Prop`, `absA`, `absg` |
| `module1_mode_composition_all.csv` | `mode`, `freq`, `forceC`, `IR`, `zp_len_cm`, `NO`, `Ring`, `MetH`, `MetC`, `molecule`, `Prop` |
| `module1_mode_selection.csv` | `dataset`, `combo`, `chi2`, `active` |
| `module1_molecule_summary.csv` | `molecule`, `n_modes`, `g0`, `A0`, `sumA2`, `maxA`, `maxA_mode`, `sum_gQ2`, `max_gQ`, `max_dg`, `n_g_gt_A_Qband`, `n_g_gt_A_Xband`, `min_R2` |
| `module1_scans.csv` | `molecule`, `mode`, `quantity`, `Q`, `value`, `ref_value` |
| `module1_thermal_modulation_by_mode.csv` | `T`, `molecule`, `mode`, `freq`, `occ`, `dA_rms`, `dg_rms_Q` |
| `module1_thermal_modulation_summary.csv` | `T`, `molecule`, `window`, `n_modes`, `A_rms_MHz`, `gQ_rms_MHz` |
| `module1_wigner_stats.csv` | `coordinate`, `TEMPO_mean`, `TEMPO_sd`, `PTEMPO_mean`, `PTEMPO_sd`, `sd_ratio`, `levene_p`, `ks_D` |

### `data/module2/`

| File | Columns |
|---|---|
| `module2_S2_mixture.csv` | `T`, `rep`, `n`, `BIC1`, `BIC2`, `BIC3`, `dBIC_2vs1`, `mu_low`, `sd_low`, `w_low`, `mu_high`, `sd_high`, `w_high` |
| `module2_block_timeseries.csv` | `T`, `rep`, `t_ns`, `h_spin`, `tilt`, `gauche`, `n_contact` |
| `module2_class_labels.csv` | `T`, `rep`, `mol`, `edge_dist`, `h_spin`, `cosNO`, `n8`, `contact`, `tilt`, `gauche`, `theta0`, `S2`, `cls`, `post` |
| `module2_class_model_selection.csv` | `k`, `BIC`, `silhouette`, `ARI_restart_median`, `dBIC` |
| `module2_class_persistence.csv` | `T`, `rep`, `retained` |
| `module2_class_populations_by_replica.csv` | `T`, `rep`, `0`, `1`, `2`, `3` |
| `module2_class_properties.csv` | `Unnamed: 0`, `Unnamed: 1`, `h_spin`, `h_spin.1`, `cosNO`, `cosNO.1`, `theta0`, `theta0.1`, `n8`, `n8.1`, `tilt`, `tilt.1`, `contact`, `contact.1`, `gauche`, `gauche.1`, `edge_dist`, `edge_dist.1`, `S2`, `S2.1` |
| `module2_correlated_motion_by_replica.csv` | `T`, `rep`, `r_lo`, `r_hi`, `n`, `corr_cos`, `corr_h`, `abs_corr_cos`, `frac_corr_gt_0p3` |
| `module2_correlated_motion_summary.csv` | `Unnamed: 0`, `Unnamed: 1`, `Unnamed: 2`, `n`, `n.1`, `corr_cos`, `corr_cos.1`, `corr_h`, `corr_h.1`, `abs_corr_cos`, `abs_corr_cos.1`, `frac_corr_gt_0p3`, `frac_corr_gt_0p3.1` |
| `module2_dipolar_by_replica.csv` | `T`, `rep`, `field`, `sqrtM2_median_MHz`, `sqrtM2_mean_MHz`, `sigma_d_median_MHz`, `tau_d_median_ns`, `xi_median`, `frac_xi_lt1`, `n_pairs` |
| `module2_dipolar_summary.csv` | `Unnamed: 0`, `Unnamed: 1`, `sqrtM2_median_MHz`, `sqrtM2_median_MHz.1`, `sqrtM2_mean_MHz`, `sqrtM2_mean_MHz.1`, `sigma_d_median_MHz`, `sigma_d_median_MHz.1`, `tau_d_median_ns`, `tau_d_median_ns.1`, `xi_median`, `xi_median.1`, `frac_xi_lt1`, `frac_xi_lt1.1`, `n_pairs`, `n_pairs.1` |
| `module2_dynamics_by_molecule.csv` | `T`, `rep`, `mol`, `interior`, `intact`, `S2`, `theta0`, `A_fast`, `tau_e_ns`, `C2_10ps`, `rmsf_spin` |
| `module2_dynamics_by_replica.csv` | `T`, `rep`, `group`, `S2`, `S2_median`, `theta0`, `tau_e_ns`, `A_fast`, `C2_10ps`, `rmsf`, `frac_S2_gt_0p8`, `frac_S2_lt_0p4` |
| `module2_dynamics_summary.csv` | `Unnamed: 0`, `Unnamed: 1`, `S2`, `S2.1`, `S2_median`, `S2_median.1`, `theta0`, `theta0.1`, `tau_e_ns`, `tau_e_ns.1`, `A_fast`, `A_fast.1`, `C2_10ps`, `C2_10ps.1`, `rmsf`, `rmsf.1`, `frac_S2_gt_0p8`, `frac_S2_gt_0p8.1`, `frac_S2_lt_0p4`, `frac_S2_lt_0p4.1` |
| `module2_mobility_correlations.csv` | `T`, `rep`, `descriptor`, `rho`, `p`, `n` |
| `module2_pairs_by_replica.csv` | `T`, `rep`, `pairs_lt5`, `pairs_lt6`, `pairs_lt8`, `OO_pairs_lt4`, `OO_pairs_lt5`, `nn_mean`, `nn_p10`, `nn_p90`, `nu_dd_median_MHz`, `nu_dd_p90_MHz`, `angle_mean`, `frac_parallel_lt45`, `frac_anti_gt135`, `nn_persist_1ns`, `nn_persist_2ns` |
| `module2_pairs_summary.csv` | `Unnamed: 0`, `pairs_lt5`, `pairs_lt5.1`, `pairs_lt6`, `pairs_lt6.1`, `pairs_lt8`, `pairs_lt8.1`, `OO_pairs_lt4`, `OO_pairs_lt4.1`, `OO_pairs_lt5`, `OO_pairs_lt5.1`, `nn_mean`, `nn_mean.1`, `nn_p10`, `nn_p10.1`, `nn_p90`, `nn_p90.1`, `nu_dd_median_MHz`, `nu_dd_median_MHz.1`, `nu_dd_p90_MHz`, `nu_dd_p90_MHz.1`, `angle_mean`, `angle_mean.1`, `frac_parallel_lt45`, `frac_parallel_lt45.1`, `frac_anti_gt135`, `frac_anti_gt135.1`, `nn_persist_1ns`, `nn_persist_1ns.1`, `nn_persist_2ns` … |
| `module2_spin_bath_reference.csv` | `Unnamed: 0`, `0` |
| `module2_spin_bath_summary.csv` | `Unnamed: 0`, `head_0_4`, `head_0_4.1`, `head_4_8`, `head_4_8.1`, `rest_0_4`, `rest_0_4.1`, `rest_4_8`, `rest_4_8.1`, `nb_0_4`, `nb_0_4.1`, `nb_4_8`, `nb_4_8.1`, `tot_4_8`, `tot_4_8.1`, `rest_8_12`, `rest_8_12.1`, `nb_8_12`, `nb_8_12.1`, `tot_8_12`, `tot_8_12.1`, `sum_r6_all`, `sum_r6_all.1`, `sum_r6_nb`, `sum_r6_nb.1`, `CH2_nb_4_8`, `CH2_nb_4_8.1`, `CH3_nb_2p5_7`, `CH3_nb_2p5_7.1`, `frac_below` … |
| `module2_structure_by_replica.csv` | `T`, `rep`, `window`, `group`, `observable`, `value` |
| `module2_structure_summary.csv` | `T`, `window`, `group`, `observable`, `mean`, `sd`, `n` |
| `module2_spin_bath_samples.csv.gz` | `T`, `rep`, `frame`, `mol`, `head_0_4`, `head_4_8`, `rest_0_4`, `rest_4_8`, `rest_8_12`, `nb_0_4`, `nb_4_8`, `nb_8_12`, `sum_r6_all`, `sum_r6_nb`, `CH2_nb_4_8`, `CH3_nb_2p5_7`, `frac_below`, `tot_4_8`, `tot_8_12` |

### `data/module3/`

| File | Columns |
|---|---|
| `benchmark_mag_nestedcv_folds.csv` | `MAE_J`, `RMSE_J`, `R2_J`, `MedAE_J`, `spearman`, `sign_acc`, `MAE_J_small`, `within3x`, `RMSE_y`, `R2_y`, `target`, `model`, `set`, `fold`, `params` |
| `benchmark_signed_nestedcv_folds.csv` | `MAE_J`, `RMSE_J`, `R2_J`, `MedAE_J`, `spearman`, `sign_acc`, `MAE_J_small`, `within3x`, `RMSE_y`, `R2_y`, `target`, `model`, `set`, `fold`, `params` |
| `exchange_by_class.csv` | `Unnamed: 0`, `Unnamed: 1`, `n`, `n.1`, `degree`, `degree.1`, `frac_isolated`, `frac_isolated.1`, `sumJ`, `sumJ.1`, `theta_K`, `theta_K.1` |
| `exchange_by_class_by_replica.csv` | `T`, `rep`, `cls`, `n`, `degree`, `frac_isolated`, `sumJ`, `theta_K` |
| `exchange_pairs_T100_r1.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_pairs_T100_r2.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_pairs_T100_r3.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_pairs_T300_r1.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_pairs_T300_r2.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_pairs_T300_r3.csv` | `i`, `j`, `mol_i`, `mol_j`, `intact`, `Jabs_mean`, `Jabs_lo`, `Jabs_hi`, `EJ_mean`, `EJ_sd_time`, `Jabs_sd_time`, `Jabs_max`, `pAF_w`, `frac_res`, `r_spin_mean`, `r_spin_min`, `rNO_min`, `tau_J_ns`, `cv_J`, `xi_J` |
| `exchange_sensitivity.csv` | `Unnamed: 0`, `Unnamed: 1`, `Unnamed: 2`, `Unnamed: 3`, `n_pairs`, `n_pairs.1`, `degree`, `degree.1`, `frac_isolated`, `frac_isolated.1`, `frac_AF`, `frac_AF.1`, `Jmedian`, `Jmedian.1` |
| `exchange_sensitivity_by_replica.csv` | `T`, `rep`, `J_RES`, `scope`, `rcut`, `n_pairs`, `degree`, `frac_isolated`, `frac_AF`, `Jmedian` |
| `exchange_spins_T100_r1.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_spins_T100_r2.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_spins_T100_r3.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_spins_T300_r1.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_spins_T300_r2.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_spins_T300_r3.csv` | `mol`, `degree`, `sumJ`, `sumEJ`, `Jmax`, `nAF`, `nF`, `w_ex_MHz`, `sqrtM2_MHz`, `theta_K`, `cls` |
| `exchange_summary.csv` | `Unnamed: 0`, `n_intact`, `n_intact.1`, `n_pairs_res`, `n_pairs_res.1`, `n_pairs_gt1`, `n_pairs_gt1.1`, `degree_mean`, `degree_mean.1`, `frac_isolated`, `frac_isolated.1`, `frac_deg_ge2`, `frac_deg_ge2.1`, `largest_cluster_frac`, `largest_cluster_frac.1`, `n_clusters_ge2`, `n_clusters_ge2.1`, `Jres_median`, `Jres_median.1`, `Jres_q25`, `Jres_q25.1`, `Jres_q75`, `Jres_q75.1`, `Jmax`, `Jmax.1`, `Jmax_frame`, `Jmax_frame.1`, `frac_AF_res`, `frac_AF_res.1`, `nAF_per_spin` … |
| `exchange_summary_by_replica.csv` | `T`, `rep`, `n_intact`, `n_pairs_res`, `n_pairs_gt1`, `degree_mean`, `frac_isolated`, `frac_deg_ge2`, `largest_cluster_frac`, `n_clusters_ge2`, `Jres_median`, `Jres_q25`, `Jres_q75`, `Jmax`, `Jmax_frame`, `frac_AF_res`, `nAF_per_spin`, `nF_per_spin`, `theta_mean_K`, `theta_min_K`, `sumJ_mean`, `frac_ST_gt_dip`, `frac_wex_gt_M2`, `wex_over_M2_median_coupled`, `sqrtM2_median`, `wex_median_coupled`, `frac_spins_Jmax_gt_Q`, `frac_res_time_mean`, `tau_J_median_ns`, `cvJ_median` … |
| `learning_curves.csv` | `n`, `rep`, `R2_y`, `RMSE_y`, `AUC_sign_geometry` |
| `pairs_J.csv` | `name`, `J`, `OO_header`, `r_OO` |
| `perm_importance.csv` | `feature`, `family`, `dMSE`, `sd` |
| `perm_importance_grouped.csv` | `family`, `n`, `dMSE`, `sd` |
| `sign_classification_nestedcv.csv` | `set`, `model`, `n`, `frac_AF`, `AUC`, `AUC_lo`, `AUC_hi`, `bal_acc`, `Brier`, `Brier_base`, `BSS` |
| `test_predictions.csv` | `name`, `J`, `r_NO_min`, `y_true`, `y_ET`, `q68`, `q90`, `y_RF`, `y_GPR`, `y_KRR`, `committee_sd`, `pAF`, `EJ`, `J_direct`, `ad_dist` |
| `train_features.csv` | `name`, `r_OO`, `r_NN`, `r_NO_min`, `r_NO_max`, `ang_NOO_min`, `ang_NOO_max`, `cos_tau`, `pipi_abs_cos`, `pi_OO_cos_min`, `pi_OO_cos_max`, `dmin_N_N`, `soft_N_N`, `dmin_N_O`, `soft_N_O`, `dmin_N_Ca`, `soft_N_Ca`, `dmin_N_C4`, `soft_N_C4`, `dmin_N_Me`, `soft_N_Me`, `dmin_N_H`, `soft_N_H`, `dmin_O_O`, `soft_O_O`, `dmin_O_Ca`, `soft_O_Ca`, `dmin_O_C4`, `soft_O_C4`, `dmin_O_Me` … |
| `train_oof.csv` | `name`, `J`, `y_true`, `y_oof`, `r_NO_min` |

### Raw inputs

| File | Content |
|---|---|
| `simulation_inputs/ModuleI/spin_phonon_scans_notebook.py` | computed g_iso and A_iso for every displacement of every treated mode (TEMPO, PTEMPO, PTEMPO-alkyl) |
| `simulation_inputs/ModuleI/vibcontrib_{TEMPO,PTEMPO}.dat` | per mode: frequency, force constant, IR intensity, dimensionless prefactor, fragment participations NO, Ring, MetH, MetC (and Prop for PTEMPO) |
| `simulation_inputs/ModuleI/Wigner_{TEMPO,PTEMPO}_intcoord.dat` | 10 000 Wigner samples (300 K): N–O, O–N–C2, O–N–C3, N pyramidalization, a dihedral |
| `simulation_inputs/ModuleIII/J-coupling.dat` | pair name, J (cm⁻¹, ORCA J(3)) for the 1364 pairs |
| `simulation_inputs/ModuleIII/pair_geometries_1364.tar.gz` | the 1364 pair geometries (58 atoms; atoms 1–29 and 30–58 are the two radicals; comment line: O···O, angle, dihedral) |
