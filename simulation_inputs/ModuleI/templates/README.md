# Input templates

| File | Program | Purpose |
|---|---|---|
| `gaussian_opt_freq.gjf` | Gaussian 16 | Optimization and harmonic frequencies, B3LYP-D3/def2-TZVP (doublet). The `.log` file is read by `python_codes/m1_00_displace_along_modes.py`. |
| `orca_epr_g_A.inp` | ORCA 5.0.1 | g tensor and ¹⁴N hyperfine tensor at one displaced geometry, B2PLYP/EPR-II. One job per mode and displacement. |
| `orca_broken_symmetry_J_example.inp` | ORCA 5.0.1 | One of the 1364 broken-symmetry exchange calculations (UKS B2PLYP-D3(BJ)/EPR-II, BrokenSym 1,1). |

These are templates. The exact keywords of the published runs are listed in the
documentation (`docs/`); check the syntax against your program version.
