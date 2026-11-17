# 2. Module II — molecular dynamics of the radical monolayer

**Goal.** Determine how the radical heads are arranged and how they move on gold, at 100 and 300 K, and from this derive the structure, dynamics, radical classes, nuclear-spin bath and electron–electron dipolar couplings that control decoherence.

---

## 2.1 The model

| Item | Value |
|---|---|
| Molecules | 141 alkylated PTEMPO molecules (residue `LIN`, 72 atoms each), S-anchored |
| Gold | two-layer slab, 2400 Au atoms (residues 1–2400), 139 × 99 Å², in-layer lattice 2.88 × 4.08 Å, interlayer 1.47 Å (Au(110) geometry); thin, finite slab to limit cost |
| Coverage | S atoms on an irregular lattice 2.7 Å above the top Au layer, forming a circular island; convex hull 3259 Å² → 4.3 × 10¹⁴ molecules cm⁻² (experiment: 4(2) × 10¹⁴) |
| Total atoms | 12 552 (net charge 0) |
| Organic force field | GAFF (`leaprc.gaff`) + custom parameters (`TEMPO-LIN.frcmod`, `TEMPO-LIN.lib`); missing bonded terms from `parmchk2`; nitroxide atom types `n3`, `o` |
| Charges | RESP fit to the B3LYP/def2-TZVP electrostatic potential of the radical |
| Gold | 12-6 Lennard-Jones parameters of the INTERFACE force field (`leaprc.interface_v1_5`, `fcc_metals.lib`); Au–molecule interactions purely nonbonded |
| Anchoring | no covalent Au–S terms; Au and S atoms held by harmonic positional restraints |

**Building the topology** (`simulation_inputs/ModuleII/tleap.in`):

```bash
tleap -f tleap.in        # needs TEMPO-SAM-11-init-struct-vmd.pdb (initial structure), TEMPO-LIN.frcmod, TEMPO-LIN.lib
```

Two topologies are provided:
- `TEMPO-SAM-11_setup_periodic.parm7`, with box information, for minimization and heating.
- `TEMPO-SAM-11_production.parm7`, without a box, for the non-periodic production runs.

The two contain identical atoms, bonds and parameters (12 552 atoms, 10 293 bonds), and the production topology is the one read by all analysis scripts.

## 2.2 Simulation protocol (AMBER 18, `sander.MPI`)

| Stage | Input | Length | Integration | Boundary / nonbonded | Restraints (kcal mol⁻¹ Å⁻²) |
|---|---|---|---|---|---|
| Minimization | `mini1.in` | 500 SD + 500 CG steps | — | periodic, 12 Å cutoff | heavy atoms, 100 |
| Heating 0 → 100 K | `heat1.in` | 1 ns | 1 fs, Langevin γ = 5 ps⁻¹, linear T ramp | periodic, 12 Å cutoff | heavy atoms, 50 |
| Equilibration (discarded) | `{100K,300K}/equil1.in` | 5 ns | 2 fs, Langevin γ = 2 ps⁻¹, SHAKE (H bonds) | none (`ntb=0`), no cutoff (`cut=999`) | Au and S1, 200 |
| Production (analyzed) | `{100K,300K}/equil1-2.in` | 5 ns | same | same | same |

- **Replicas:** three per temperature, started from the same heated structure with different Langevin seeds (`ig=-1`).
- **Output:** frames are saved every 10 ps (`ntwx=5000`).
- **Post-processing:** `align.in` (cpptraj) images the system with the Au slab as anchor, fits on the gold, and writes `TEMPO-SAM-11.xtc`.

Each stored trajectory has 1100 frames:
- frames 0–99: restrained heating, discarded;
- frames 100–599: first 5 ns at the target temperature, treated as equilibration;
- frames 600–1099: last 5 ns, the analysis window (500 frames).

Job scripts: `{100K,300K}/TEMPO-SAM.sh` and `TEMPO-SAM-2.sh` (SLURM). Cluster-specific paths and accounts have been replaced by placeholders.

## 2.3 Reading the trajectories

The six raw trajectories (about 23 GB) are not stored in the repository. The analysis expects them at

```
md_raw/{100K,300K}/rep{1,2,3}/TEMPO-SAM-11.xtc      (or set TSAM_MD_RAW)
```

They are decoded by a stand-alone C implementation of the GROMACS XTC compression algorithm. No MDAnalysis is needed.

```bash
cd python_codes
bash m2_00b_decode_trajectories.sh        # builds xtc_reader/xtc2bin and writes md_bin/T{T}_r{r}.coords.f32 + .frames.txt
python m2_01_validate_decoder.py          # every bond in every frame vs force-field equilibrium length
```

**Validation:**
- bonds to hydrogen (SHAKE-constrained) deviate by at most 0.016 Å, against an XTC precision of 0.01 Å;
- heavy-atom bonds of intact molecules have rms deviations of 0.022 Å (100 K) and 0.033 Å (300 K), consistent with thermal vibration.

`m2_00_io.py` provides `read_parm7()`, a minimal Amber topology parser, and `load_traj()`.

## 2.4 Excluded molecules

In 14 molecules the TEMPO ring is strongly distorted throughout all six trajectories:
- ring bonds are stretched to 1.6–2.0 Å, while the N–O bond is normal;
- the 14 molecules form seven pairs whose S atoms are 3.1–3.3 Å apart, so their rings interpenetrate in the common starting structure.

These molecules are excluded from all head-group, dynamics, pair and exchange analyses: residues 2428, 2440, 2441, 2455, 2485, 2486, 2495, 2508, 2512, 2527, 2528, 2529, 2533 and 2537. That leaves **127 intact radicals** per trajectory.

## 2.5 Observables (`m2_02` … `m2_08`)

**Geometry.** The Au slab lies in the yz plane, the monolayer extends toward −x, and the surface normal is n = (−1, 0, 0). Heights are measured from the instantaneous mean x of the top Au layer, and the spin center is the N–O midpoint.

**Groups.**
- *Interior* radicals have their S atom at least 5 Å inside the convex hull of all S atoms.
- *Edge* radicals are the outermost row.
- *Intact* radicals exclude the 14 defective molecules.

**Per-radical, per-frame quantities (`m2_02_extract_observables.py` → `data/module2/obs_T{T}_r{r}.npz`):**
- spin height;
- cos θ_NO, the angle between N→O and n;
- chain tilt, from S1→C13 against n;
- gauche fraction of the C23…C13 backbone dihedrals (|φ| < 120°);
- O–Au minimum distance;
- nearest-neighbor spin distances;
- number of spins within 6 and 8 Å.

**Reorientation (`m2_04_dynamics.py`).** For the N–O unit vector **u**:

$$C_2(t)=\langle P_2[\mathbf u(0)\cdot\mathbf u(t)]\rangle,\qquad S^2=\tfrac32\sum_{ij}\langle u_iu_j\rangle^2-\tfrac12,\qquad S^2=\left[\tfrac12\cos\theta_0(1+\cos\theta_0)\right]^2 .$$

The effective correlation time comes from fitting C₂(t) = S² + (A − S²)e^{−t/τ_e} between 10 ps and 1 ns. Because the Langevin thermostat damps the dynamics, the correlation times are indicative, whereas the order parameters are equilibrium properties.

**Other analyses:**
- **`m2_05`:** Gaussian mixtures (1–3 components, BIC) of the per-radical cone angles.
- **`m2_06`:** Spearman correlations of mobility with local descriptors.
- **`m2_07`:** pair geometry, nearest-neighbor persistence and point-dipole couplings, ν_dd = 52.04 MHz nm³ / r³.
- **`m2_08`:** RMSD and 0.5-ns block averages for convergence.

**Statistics.** Every quantity is first averaged over molecules and frames within one replica. The reported value is the mean ± SD over the three replicas (n = 3).

## 2.6 Radical classes (`m2_11`, `m2_12`)

**Descriptors.** Each intact radical in each trajectory (762 cases) is described by five time-averaged descriptors:
- spin height;
- ⟨cos θ_NO⟩;
- cone semi-angle θ₀;
- number of spins within 8 Å;
- chain tilt.

**Fitting.**
- The standardized descriptors of all six trajectories are pooled, so both temperatures share one set of class definitions.
- Gaussian mixtures (full covariance, regularization 10⁻²) with k = 2–8 components are fitted.
- k is the BIC minimum among solutions whose labels agree across 30 random initializations (median adjusted Rand index ≥ 0.9).
- The result is **k = 4**: frozen (F), restricted (R), mobile (M) and surface-bound (S).

**Persistence (`m2_12`).** Classes are recomputed for the first and second 5 ns of each trajectory, two windows of equal length, and both are classified with the same model.

## 2.7 Spin environment (`m2_13`, `m2_14`, `m2_15`)

**Proton bath (`m2_13`).**
- Protons are counted in the shells 0–4, 4–8 and 8–12 Å around each spin, on every 10th frame.
- They are split into own head, own chain and neighboring molecules.
- The reference is a homogeneous bath with the proton density of 1:1 (v/v) dichloromethane/toluene (53.3 M) filling all space beyond 4 Å.
- Also computed: Σ r⁻⁶ for protons beyond 4 Å, neighboring CH₂ groups within 4–8 Å, and CH₃ groups within 2.5–7 Å.

**Dipolar couplings (`m2_14`).** For every pair of intact radicals, the secular coupling is

$$d_{ij}(t)=\nu_\mathrm{dd}(r_{ij})\left(1-3\cos^2\beta_{ij}\right),$$

with β_ij the angle between the interspin vector and the field. The field is taken along the surface normal, or averaged over the two in-plane axes.
- **Local width:** √M₂ is computed per radical, with M₂ = Σⱼ d_ij² over partners within 15 Å.
- **Fluctuations:** for pairs closer than 8 Å, σ_d is the SD of d_ij(t), τ_d is its integrated autocorrelation time, and the motional-narrowing parameter is ξ = 2πσ_dτ_d.

**Correlated motion (`m2_15`).** Pearson correlations between the time series of two radicals (spin heights, or cos θ_NO), binned by pair distance. Pairs more than 25 Å apart serve as the null reference.

## 2.8 Outputs

| File | Content |
|---|---|
| `data/module2/obs_T{T}_r{r}.npz` | per-frame, per-radical observables (last 10 ns, 1000 frames) |
| `data/module2/module2_structure_*.csv`, `module2_dynamics_*.csv`, `module2_pairs_*.csv` | statistics by replica and summaries |
| `data/module2/module2_class_*.csv/.npy/.json`, `module2_class_model.joblib` | classes, populations, transition matrices, stability |
| `data/module2/module2_spin_bath_*.csv`, `module2_dipolar_*.csv/.npz` | spin environment |
| `data/module2/last_frame_rep1.npz` | final frame of replica 1 at both temperatures (for Figure 2a without the raw trajectories) |
| `Figures_MS/Figure2_monolayer_MD.*`, `Figure3_radical_classes.*` and the `FigS_MD_*`, `FigS_class_*` figures | from `m2_09`, `m2_16` |
| `tables/tabS_MD_*.tex`, `tabS_class_*.tex`, `tabS_spin_bath.tex`, `tabS_dipolar.tex` | from `m2_10`, `m2_17` |
