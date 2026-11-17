# 0. Installation

## Python environment

The analysis needs only standard scientific Python packages:

```bash
# option A: conda
conda env create -f environment.yml
conda activate noise-in-qubits

# option B: pip (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

No MDAnalysis, MDTraj, RDKit or XGBoost is required. The MD trajectories are
read with a small C decoder included in the repository (`python_codes/xtc_reader/xtc2bin.c`).

## C compiler (only for the raw trajectories)

```bash
gcc -O2 -o python_codes/xtc_reader/xtc2bin python_codes/xtc_reader/xtc2bin.c
```

## Quantum-chemistry and MD programs (only to redo the electronic-structure / MD steps)

| Program | Version used | Steps |
|---|---|---|
| Gaussian 16 | Rev. as available on the cluster | optimization + frequencies (Module I) |
| ORCA | 5.0.1 | g and A tensors (Module I); broken-symmetry J (Module III) |
| AMBER | 18 | topology building and MD (Module II) |

## Paths

All locations are defined in `python_codes/config.py` and can be overridden with
environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TSAM_M1_NOTEBOOK` | `simulation_inputs/ModuleI/spin_phonon_scans_notebook.py` | file with the computed scans |
| `TSAM_MD_RAW` | `md_raw/` | raw trajectories, `{100K,300K}/rep{1,2,3}/TEMPO-SAM-11.xtc` |
| `TSAM_MD_BIN` | `md_bin/` | decoded trajectories |
| `TSAM_PARM7` | `simulation_inputs/ModuleII/TEMPO-SAM-11_production.parm7` | production topology |
| `TSAM_PAIRS` | `simulation_inputs/ModuleIII/` | pair library (J values and geometries) |

All scripts are run from inside `python_codes/`:

```bash
cd python_codes
python m3_05_final_model.py
```
