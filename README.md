# vortex_beam_thesis

This project reorganizes the existing FVVB simulation code for thesis use.
The physical model is kept unchanged: Berry expansion, FVVB Jones-field
definition, same-plane SPP phase factor, angular-spectrum propagation, and the
common scalar modified von Karman phase-screen approximation are preserved.

## Structure

```text
vortex_beam_thesis/
├── configs/                 # YAML parameters for reproducible runs
├── src/                     # reusable model, propagation, analysis, plotting code
├── scripts/                 # command-line entry points
├── notebooks/               # lightweight inspection notebooks
├── results/                 # generated figures, npz data, and logs
├── thesis_figures/          # manually selected final thesis figures
├── legacy_code/             # archived copy of the old code
├── tests/                   # minimal regression tests
├── README.md
└── requirements.txt
```

## Install

```bash
cd D:/Project/毕设/vortex_beam_thesis
python -m pip install -r requirements.txt
```

## Run One Simulation

```bash
python scripts/run_single_simulation.py --config configs/default.yaml
```

Each run creates a timestamped folder under:

```text
results/figures/
results/data/
results/logs/
```

The run saves:

- `config_used.yaml`: the exact merged configuration.
- `summary.json`: OAM summary, polarization summary, energy and diagnostics.
- `single_simulation_raw.npz`: grid, fields, intensity, phase, phase screens, and OAM spectrum.
- `intensity_phase.png` and `oam_spectrum.png`.

## Run A Quick Parameter Scan

```bash
python scripts/run_parameter_scan.py --config configs/propagation_scan.yaml
```

The default scan is intentionally small. Before using results as thesis-grade
data, increase `grid_n`, `lmax`, `nr_fourier`, `nphi_fourier`, and `n_mc`, then
check `Pcap`, edge energy, and numerical convergence.

## Reproduce Default Figures

```bash
python scripts/reproduce_figures.py
```

This reruns the default single simulation. Final thesis figures should be copied
or regenerated from reviewed YAML configs and recorded run folders.

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

Run:

```bash
python -m unittest discover -s tests
```

Current tests check Berry coefficient integer limits, linear/circular basis
round trip, generated FVVB energy, and YAML config loading.

## Legacy Code

The original code is archived under `legacy_code/PYTHON/`. The source folder
`D:/Project/毕设/代码/PYTHON` is not deleted or modified by this reorganization.

