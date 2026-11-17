"""
config.py -- all file-system locations used by the analysis scripts.

Every path can be overridden with an environment variable, so the repository
works wherever it is cloned. Defaults assume the repository layout:

    Noise_In_Qubits/
        python_codes/          (this folder)
        data/  tables/  Figures_MS/  Figures_ESI/
        simulation_inputs/ModuleI | ModuleII | ModuleIII
"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
INPUTS = os.path.join(ROOT, "simulation_inputs")

# Module I: exported notebook with the displacement scans
M1_NOTEBOOK = os.environ.get("TSAM_M1_NOTEBOOK", os.path.join(INPUTS, "ModuleI", "spin_phonon_scans_notebook.py"))

# Module II: raw trajectories (NOT in the repository, ~23 GB):
#   $TSAM_MD_RAW/{100K,300K}/rep{1,2,3}/TEMPO-SAM-11.xtc
M2_RAW = os.environ.get("TSAM_MD_RAW", os.path.join(ROOT, "md_raw"))
# decoded trajectories written by xtc2bin (one .coords.f32 + .frames.txt per run)
M2_BIN = os.environ.get("TSAM_MD_BIN", os.path.join(ROOT, "md_bin"))
# topology of the production runs (non-periodic; identical for all six runs)
M2_PARM7 = os.environ.get("TSAM_PARM7", os.path.join(INPUTS, "ModuleII", "TEMPO-SAM-11_production.parm7"))

# Module III: DFT exchange library
M3_RAW = os.environ.get("TSAM_PAIRS", os.path.join(INPUTS, "ModuleIII"))
