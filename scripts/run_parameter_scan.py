from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.beam_generation import fvvb_field_waist, get_grid_from_params  # noqa: E402
from src.phase_analysis import compute_oam_summary, total_oam_spectrum_from_vector  # noqa: E402
from src.plotting import save_scan_width_figure  # noqa: E402
from src.propagation import make_phase_screens, propagate_vector_split_step_linear  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Run a lightweight FVVB parameter scan.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "propagation_scan.yaml"),
        help="Path to YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    params = params_from_config(cfg)
    dirs = create_run_dirs(cfg)

    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    alpha_list = np.asarray(cfg["scan"]["alpha_list"], dtype=float)
    cn2_list = np.asarray(cfg["scan"]["cn2_list"], dtype=float)
    distance_list = np.asarray(cfg["scan"]["distance_list"], dtype=float)
    n_mc = int(cfg["scan"]["n_mc"])
    base_seed = int(cfg["scan"]["seed"])
    l_list = params.l_list

    spectra = np.zeros((len(alpha_list), len(cn2_list), len(distance_list), n_mc, len(l_list)))
    widths = np.zeros((len(alpha_list), len(cn2_list), len(distance_list), n_mc))

    for ia, alpha in enumerate(alpha_list):
        Ex0, Ey0 = fvvb_field_waist(float(alpha), R, PHI, params)
        for ic, cn2 in enumerate(cn2_list):
            for idist, distance in enumerate(distance_list):
                for imc in range(n_mc):
                    screens = []
                    if cn2 > 0 and int(cfg["turbulence"]["n_screens"]) > 0:
                        seed = base_seed + ia * 10000 + ic * 1000 + idist * 100 + imc
                        turb_params = turbulence_params_from_config(
                            cfg,
                            distance=float(distance),
                            seed=seed,
                            cn2=float(cn2),
                        )
                        screens = make_phase_screens(
                            params.grid_n,
                            dx,
                            dy,
                            params.wavelength,
                            turb_params,
                            int(cfg["turbulence"]["n_screens"]),
                        )
                    Ex, Ey = propagate_vector_split_step_linear(
                        Ex0,
                        Ey0,
                        dx,
                        dy,
                        params.wavelength,
                        float(distance),
                        screens,
                        str(cfg["propagation"]["method"]),
                        float(cfg["propagation"]["pad_factor"]),
                        bool(cfg["propagation"]["check_finite"]),
                    )
                    spectrum, _meta = total_oam_spectrum_from_vector(
                        Ex,
                        Ey,
                        X,
                        Y,
                        l_list,
                        params.nr_fourier,
                        params.nphi_fourier,
                    )
                    summary = compute_oam_summary(l_list, spectrum)
                    spectra[ia, ic, idist, imc] = spectrum
                    widths[ia, ic, idist, imc] = summary["width"]

    spectra_mean = np.mean(spectra, axis=3)
    width_mean = np.mean(widths, axis=3)
    width_std = np.std(widths, axis=3)

    save_yaml(dirs["logs"] / "config_used.yaml", cfg)
    save_json(
        dirs["logs"] / "summary.json",
        {
            "beam_params": params.to_dict(),
            "alpha_list": alpha_list,
            "cn2_list": cn2_list,
            "distance_list": distance_list,
            "n_mc": n_mc,
            "note": "Quick scan; increase grid_n/lmax/n_mc before using as thesis-grade data.",
        },
    )
    save_npz(
        dirs["data"] / "parameter_scan_raw.npz",
        l_list=l_list,
        alpha_list=alpha_list,
        cn2_list=cn2_list,
        distance_list=distance_list,
        spectra=spectra,
        spectra_mean=spectra_mean,
        width_mean=width_mean,
        width_std=width_std,
    )
    save_scan_width_figure(
        dirs["figures"] / "oam_width_vs_cn2.png",
        cn2_list,
        alpha_list,
        width_mean[:, :, 0],
    )

    print(f"Scan saved to: {dirs['data'].parent.name}/{dirs['data'].name}")
    print(f"Figures: {dirs['figures']}")
    print(f"Data: {dirs['data']}")
    print(f"Logs: {dirs['logs']}")


if __name__ == "__main__":
    main()

