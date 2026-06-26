# Legacy FVVB Python Code

This directory keeps the original thesis scripts as traceable reference code.
Files have been renamed by research topic instead of figure number so later
debugging can follow the physics workflow.

## Core Modules

- `core/fvvb_model_core.py`: FVVB/FVB field generation and modal projection helpers.
- `core/angular_spectrum_propagation.py`: angular-spectrum and split-step propagation.
- `core/turbulence_phase_screen.py`: modified von Karman phase-screen generation.
- `core/oam_spectrum_metrics.py`: OAM spectra, entropy, width, and similarity metrics.
- `core/polarization_stokes_metrics.py`: Stokes and polarization summaries.

## Chapter 3: Free-Space FVVB And SPP Modulation

- `chapter_03_free_space_and_spp/simulate_fvvb_intensity_distribution.py`
- `chapter_03_free_space_and_spp/analyze_fvvb_total_oam_spectrum.py`
- `chapter_03_free_space_and_spp/compare_scalar_and_vector_oam_spectra.py`
- `chapter_03_free_space_and_spp/scan_oam_width_vs_fractional_charge.py`
- `chapter_03_free_space_and_spp/scan_oam_spectrum_vs_spp_charge.py`
- `chapter_03_free_space_and_spp/scan_oam_spectrum_vs_incident_charge.py`
- `chapter_03_free_space_and_spp/check_modal_decomposition_convergence.py`
- `chapter_03_free_space_and_spp/scan_oam_width_vs_alpha_and_spp.py`
- `chapter_03_free_space_and_spp/analyze_radial_lg_spectrum_difference.py`
- `chapter_03_free_space_and_spp/scan_radial_width_vs_alpha_and_spp.py`
- `chapter_03_free_space_and_spp/check_lg_basis_waist_sensitivity.py`

## Chapter 4: Turbulence Propagation

- `chapter_04_turbulence/demo_turbulence_oam_evolution.py`
- `chapter_04_turbulence/scan_turbulence_oam_degradation.py`

## Notes

The active, cleaner thesis workflow lives in the repository-level `src/` and
`scripts/` directories. Prefer those files for new work. Use this legacy folder
only when comparing with earlier results or checking how a historical script was
translated.

