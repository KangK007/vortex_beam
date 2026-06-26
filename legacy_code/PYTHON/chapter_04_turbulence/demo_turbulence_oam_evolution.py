"""湍流相位屏下 FVVB OAM 谱演化的轻量演示脚本。

该脚本是毕业设计主线 B 的趋势展示：固定一个入射 FVVB，比较多个 Cn2
湍流强度下单相位屏传播后的 OAM 谱变化，并对比 linear 与 circular 两种
传播基。它用于快速理解趋势，不等价于 ``run_turbulence_scan.py`` 的完整
Monte Carlo 批量统计结果。

模型假设：一个公共标量相位屏同时作用于两个偏振分量；不包含偏振相关
散射、双折射或真实退偏振。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import (
    default_params,
    ensure_output_dir,
    fvvb_field,
    get_grid_from_params,
    save_npz,
    save_params,
    spectral_width,
    total_oam_spectrum_from_vector,
)
from propagation import (
    apply_common_phase,
    apply_common_phase_to_linear,
    circular_to_linear,
    propagate_circular_components,
    propagate_linear_components,
)
from turbulence import TurbulenceParams, fried_parameter, make_phase_screen, rytov_variance


SCRIPT = "demo_turbulence_oam_evolution"


def propagate_with_screen(Ex, Ey, phase, dx, dy, wavelength, distance, basis="linear"):
    """施加一个公共湍流相位屏，并按指定偏振基传播。

    basis="linear"：直接对 Ex/Ey 施加相位屏并传播；basis="circular"：先转为
    sigma+/sigma-，施加同一相位屏后传播，再转回 Ex/Ey。
    """
    if basis == "linear":
        Ex_t, Ey_t = apply_common_phase_to_linear(Ex, Ey, phase)
        return propagate_linear_components(Ex_t, Ey_t, dx, dy, wavelength, distance)
    if basis == "circular":
        Ep_t, Em_t = propagate_circular_components(Ex, Ey, dx, dy, wavelength, 0.0)
        Ep_t, Em_t = apply_common_phase(Ep_t, Em_t, phase)
        Ep_z, Em_z = propagate_circular_components(*circular_to_linear(Ep_t, Em_t), dx, dy, wavelength, distance)
        return circular_to_linear(Ep_z, Em_z)
    raise ValueError(f"Unknown propagation basis: {basis}")


def main() -> None:
    # 快速演示参数：用于单屏趋势展示。论文定量图应改用 run_turbulence_scan.py。
    # 若临时把本脚本改成论文级单屏演示，建议：grid_n=256~400, n_min=-100~-200,
    # n_max=100~200, lmax=20~40, nr_fourier=200~320, nphi_fourier=360~720。
    params = default_params(
        grid_n=128,
        rho_max_factor=8.0,
        n_min=-40,
        n_max=40,
        lmax=10,
        nr_fourier=100,
        nphi_fourier=180,
    )
    alpha = 0.5  # [关键变量] 入射分数阶拓扑荷，无量纲。
    propagation_distance = 200.0  # [关键变量] 传播距离，单位 m；论文 baseline 建议改为 500/1000/1500 m。
    cn2_values = [1e-16, 1e-15, 1e-14]  # [关键变量] 湍流强度 C_n^2，单位 m^(-2/3)，对应弱/中/强湍流。
    basis_list = ["linear", "circular"]  # [关键变量] 对比线偏振基与圆偏振基传播。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    Ex0, Ey0 = fvvb_field(alpha, R, PHI, params)
    mu0, meta0 = total_oam_spectrum_from_vector(Ex0, Ey0, X, Y, params.l_list, params.nr_fourier, params.nphi_fourier)
    _, sigma0 = spectral_width(params.l_list, mu0)

    spectra = {"no_turbulence": mu0}
    sigmas = {"no_turbulence": sigma0}
    phase_std = {}
    rytov = {}
    fried = {}

    fig, axes = plt.subplots(len(basis_list), 1, figsize=(8, 6), constrained_layout=True, sharex=True)
    if len(basis_list) == 1:
        axes = [axes]

    for ax, basis in zip(axes, basis_list):
        ax.plot(params.l_list, mu0, "o-", lw=1.1, ms=3, label=f"no turbulence, sigma={sigma0:.3f}")
        for idx, cn2 in enumerate(cn2_values):
            # inner_scale/outer_scale 为湍流内/外尺度，单位 m；seed 保证相位屏可复现。
            turb = TurbulenceParams(cn2=cn2, dz=propagation_distance, inner_scale=2e-3, outer_scale=50.0, seed=100 + idx)
            phase = make_phase_screen(params.grid_n, dx, dy, params.wavelength, turb)
            Ex_z, Ey_z = propagate_with_screen(Ex0, Ey0, phase, dx, dy, params.wavelength, propagation_distance, basis=basis)
            mu, meta = total_oam_spectrum_from_vector(Ex_z, Ey_z, X, Y, params.l_list, params.nr_fourier, params.nphi_fourier)
            _, sigma = spectral_width(params.l_list, mu)
            key = f"{basis}_cn2_{cn2:.0e}"
            spectra[key] = mu
            sigmas[key] = sigma
            phase_std[key] = float(np.std(phase))  # 相位屏标准差，单位 rad。
            rytov[key] = rytov_variance(cn2, params.wavelength, propagation_distance)  # Rytov 方差，衡量湍流强弱。
            fried[key] = fried_parameter(cn2, params.wavelength, propagation_distance)  # Fried 相干直径 r0，单位 m。
            ax.plot(params.l_list, mu, "o-", lw=1.1, ms=3, label=rf"$C_n^2={cn2:.0e}$, $\sigma_L={sigma:.3f}$")
        ax.set_title(f"{basis} component propagation")
        ax.set_ylabel(r"$\mu_\ell$")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel(r"OAM order $\ell$")
    fig.suptitle(rf"FVVB OAM spectra after one turbulence screen ($\alpha={alpha}$, z={propagation_distance:g} m)")
    fig.savefig(out_dir / f"{SCRIPT}.png", dpi=300)

    save_npz(out_dir, SCRIPT, l_list=params.l_list, spectra=spectra, sigmas=sigmas)
    save_params(
        out_dir,
        SCRIPT,
        {
            "params": params,
            "alpha": alpha,
            "propagation_distance": propagation_distance,
            "cn2_values": cn2_values,
            "basis_list": basis_list,
            "dx": dx,
            "dy": dy,
            "no_turbulence_meta": meta0,
            "phase_std": phase_std,
            "rytov_variance": rytov,
            "fried_parameter": fried,
            "assumption": "One scalar phase screen is applied equally to both polarization components; use for trend checks before quantitative claims.",
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
