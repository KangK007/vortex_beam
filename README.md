# vortex_beam_thesis

Thesis-oriented Python project for fractional vector vortex beam (FVVB)
simulation, modal analysis, and turbulence propagation studies.

The physical model is kept unchanged from the original scripts:

- Berry expansion for fractional topological charge.
- FVVB Jones-field convention.
- Same-plane SPP/fork phase modulation.
- Angular-spectrum propagation.
- Common scalar modified von Karman phase-screen approximation for turbulence.

## Project Layout

```text
vortex_beam_thesis/
├── configs/                         # YAML parameters for reproducible runs
├── src/                             # reusable model, propagation, analysis, plotting code
├── scripts/
│   ├── field_generation/            # single-beam generation and basic diagnostics
│   ├── modal_analysis/              # reserved for OAM/LG analysis workflows
│   ├── propagation_studies/         # turbulence and propagation scans
│   └── reproduction/                # rerun reviewed configurations
├── notebooks/                       # lightweight inspection notebooks
├── results/                         # generated figures, npz data, and logs
├── thesis_figures/                  # manually selected final thesis figures
├── legacy_code/PYTHON/
│   ├── core/                        # original core model and metric modules
│   ├── chapter_03_free_space_and_spp/
│   └── chapter_04_turbulence/
└── tests/
```

## Install

```bash
cd D:/Project/毕设/github_release/vortex_beam_thesis
python -m pip install -r requirements.txt
```

## Main Workflows

### 1. Single FVVB Beam Simulation

Use this when checking one set of beam, SPP, propagation, and turbulence
parameters.

```bash
python scripts/field_generation/simulate_single_fvvb_beam.py --config configs/default.yaml
```

Saved outputs:

- `results/logs/<run>/config_used.yaml`
- `results/logs/<run>/summary.json`
- `results/data/<run>/single_simulation_raw.npz`
- `results/figures/<run>/intensity_phase.png`
- `results/figures/<run>/oam_spectrum.png`

### 2. Turbulence Response Scan

Use this for quick checks of OAM spectral broadening under different `alpha`,
`Cn2`, and propagation distance settings.

```bash
python scripts/propagation_studies/scan_turbulence_response.py --config configs/propagation_scan.yaml
```

The default scan is intentionally small. Before using results as thesis-grade
data, increase `grid_n`, `lmax`, `nr_fourier`, `nphi_fourier`, and `n_mc`, then
check `Pcap`, edge energy, finite-value diagnostics, and numerical convergence.

### 3. Reproduce Reviewed Outputs

```bash
python scripts/reproduction/reproduce_default_outputs.py
```

This reruns the default single-beam workflow. Final thesis figures should be
generated from reviewed YAML files and copied into `thesis_figures/` only after
manual inspection.

## Configuration Files

- `configs/default.yaml`: fast single-beam sanity check.
- `configs/lg_beam.yaml`: higher-resolution free-space/SPP modal analysis setup.
- `configs/propagation_scan.yaml`: lightweight turbulence scan setup.

All physical and numerical parameters should be edited in YAML files rather than
hard-coded in scripts.

## Important Physical Assumptions

- Units are SI unless explicitly stated.
- FVVB uses the existing Jones convention:
  `Ex = sum_n c_n E'_n sin(n phi)` and
  `Ey = i sum_n c_n E'_n cos(n phi)`.
- Circular components use
  `sigma+ = (Ex + i Ey) / sqrt(2)` and
  `sigma- = (Ex - i Ey) / sqrt(2)`.
- The SPP/fork phase is applied as `exp(i * fork_ell * phi)` on the same plane.
- Turbulence uses a common scalar phase screen acting on both polarization
  components. It does not model polarization-dependent scattering, birefringence,
  or full depolarization.

## Verification

```bash
python -m unittest discover -s tests
python -m compileall -q src scripts tests
```

The tests check numerical conventions and repository organization so later
debugging does not drift back to figure-number-based scripts.

## Legacy Code

The legacy source code is kept for traceability, but it has been renamed by
research topic instead of figure number:

- `legacy_code/PYTHON/core/`
- `legacy_code/PYTHON/chapter_03_free_space_and_spp/`
- `legacy_code/PYTHON/chapter_04_turbulence/`

Generated legacy outputs are intentionally excluded from Git.

