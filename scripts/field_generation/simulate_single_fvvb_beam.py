from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beam_generation import (  # noqa: E402
    apply_spp_to_vector,
    fvvb_field,
    fvvb_field_waist,
    get_grid_from_params,
)
from src.phase_analysis import (  # noqa: E402
    compute_oam_summary,
    field_intensity,
    field_phase,
    polarization_summary,
    total_oam_spectrum_from_vector,
)
from src.plotting import save_intensity_phase_figure, save_oam_spectrum_figure  # noqa: E402
from src.propagation import (  # noqa: E402
    make_phase_screens,
    propagate_vector_split_step_circular,
    propagate_vector_split_step_linear,
    vector_energy,
    vector_field_diagnostics,
)
from src.utils import (  # noqa: E402
    create_run_dirs,
    load_config,
    params_from_config,
    save_json,
    save_npz,
    save_yaml,
    turbulence_params_from_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one FVVB simulation from YAML config.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def build_initial_field(cfg, params, R, PHI):
    alpha = float(cfg["beam"]["alpha"])
    initial_plane = str(cfg["beam"]["initial_plane"])
    if initial_plane == "waist":
        return fvvb_field_waist(alpha, R, PHI, params)
    if initial_plane == "analytic_z":
        return fvvb_field(alpha, R, PHI, params)
    raise ValueError(f"Unknown beam.initial_plane: {initial_plane}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    params = params_from_config(cfg)
    dirs = create_run_dirs(cfg)

    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    Ex0, Ey0 = build_initial_field(cfg, params, R, PHI)

    if bool(cfg["spp"]["enabled"]):
        Ex0, Ey0 = apply_spp_to_vector(Ex0, Ey0, PHI, float(cfg["spp"]["fork_ell"]))

    distance = float(cfg["propagation"]["distance"]) if cfg["propagation"]["enabled"] else 0.0
    phase_screens = []
    if bool(cfg["turbulence"]["enabled"]) and int(cfg["turbulence"]["n_screens"]) > 0:
        turb_params = turbulence_params_from_config(cfg, distance=distance)
        phase_screens = make_phase_screens(
            params.grid_n,
            dx,
            dy,
            params.wavelength,
            turb_params,
            int(cfg["turbulence"]["n_screens"]),
        )

    basis = str(cfg["propagation"]["basis"])
    propagate = (
        propagate_vector_split_step_circular
        if basis == "circular"
        else propagate_vector_split_step_linear
    )
    Ex, Ey = propagate(
        Ex0,
        Ey0,
        dx,
        dy,
        params.wavelength,
        distance,
        phase_screens,
        str(cfg["propagation"]["method"]),
        float(cfg["propagation"]["pad_factor"]),
        bool(cfg["propagation"]["check_finite"]),
    )

    l_list = params.l_list
    spectrum, spectrum_meta = total_oam_spectrum_from_vector(
        Ex,
        Ey,
        X,
        Y,
        l_list,
        params.nr_fourier,
        params.nphi_fourier,
    )
    oam_summary = compute_oam_summary(l_list, spectrum)
    pol_summary = polarization_summary(Ex, Ey, dx, dy)
    diagnostics = vector_field_diagnostics(Ex, Ey, dx, dy)

    intensity = field_intensity(Ex, Ey)
    phase_ex = field_phase(Ex)

    save_yaml(dirs["logs"] / "config_used.yaml", cfg)
    save_json(
        dirs["logs"] / "summary.json",
        {
            "beam_params": params.to_dict(),
            "energy_initial": vector_energy(Ex0, Ey0, dx, dy),
            "energy_output": vector_energy(Ex, Ey, dx, dy),
            "oam_summary": oam_summary,
            "oam_meta": spectrum_meta,
            "polarization_summary": pol_summary,
            "field_diagnostics": diagnostics,
            "output_dirs": dirs,
        },
    )
    save_npz(
        dirs["data"] / "single_simulation_raw.npz",
        X=X,
        Y=Y,
        R=R,
        PHI=PHI,
        Ex_initial=Ex0,
        Ey_initial=Ey0,
        Ex=Ex,
        Ey=Ey,
        intensity=intensity,
        phase_ex=phase_ex,
        phase_screens=np.asarray(phase_screens),
        l_list=l_list,
        oam_spectrum=spectrum,
    )
    save_intensity_phase_figure(
        dirs["figures"] / "intensity_phase.png",
        X,
        Y,
        intensity,
        phase_ex,
        title=f"FVVB alpha={cfg['beam']['alpha']}",
    )
    save_oam_spectrum_figure(
        dirs["figures"] / "oam_spectrum.png",
        l_list,
        spectrum,
        title="Total OAM spectrum",
    )

    print(f"Run saved to: {dirs['data'].parent.name}/{dirs['data'].name}")
    print(f"Figures: {dirs['figures']}")
    print(f"Data: {dirs['data']}")
    print(f"Logs: {dirs['logs']}")


if __name__ == "__main__":
    main()
