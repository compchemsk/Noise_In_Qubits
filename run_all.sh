#!/bin/bash
cd /home/claude/TEMPO_SAM_project/python_codes
for s in m1_01_extract_scans m1_02_fit_couplings m1_03_relaxation_models m1_04_mode_selection m1_06_summary_numbers m1_15_alkyl_composition m1_07_latex_tables m1_16_c1_c2 m1_08_thermal_modulation m1_09_mode_character m1_10_wigner m1_11_parse_orca_tensors m1_12_anisotropic_couplings m1_13_multiprocess_T1 m1_05_figures \
 m2_01_validate_decoder m2_02_extract_observables m2_03_structure_stats m2_04_dynamics m2_05_S2_populations m2_06_mobility_determinants m2_07_pairs m2_08_convergence m2_11_radical_classes m2_12_class_persistence m2_13_spin_bath m2_14_dipolar_modulation m2_15_correlated_motion m2_10_tables m2_17_tables_classes m2_09_figures m2_16_figures_classes \
 m3_00_build_dataset m3_01_extract_md_headgroups m3_02_features_training m3_03_benchmark_models m3_04_sign_classification m3_04b_sign_permutation m3_05_final_model m3_06_apply_md m3_07_exchange_network m3_08_sensitivity m3_12_revision_analyses m3_13_extra_tables m3_11_tables m3_09_figures_ml m3_10_figures_network m4_01_mode_projection m4_02_mode_information m4_03_radical_level m4_04_figures m1_14_T1_figure toc_graphic fig_structures; do
  t0=$(date +%s); echo "=== $s" ; 
  if [ $s = m1_11_parse_orca_tensors ]; then (cd /home/claude/gA && python3 /home/claude/TEMPO_SAM_project/python_codes/m1_11_parse_orca_tensors.py && cp parsed.json /home/claude/TEMPO_SAM_project/data/module1/orca_tensors/parsed_tensors.json) > /tmp/log_$s.txt 2>&1
  else python3 $s.py > /tmp/log_$s.txt 2>&1; fi
  echo "rc=$? $(( $(date +%s)-t0 ))s"; tail -2 /tmp/log_$s.txt
done
