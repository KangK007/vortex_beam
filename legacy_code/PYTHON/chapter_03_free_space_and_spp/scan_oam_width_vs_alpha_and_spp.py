"""OAM width vs alpha and SPP：FVVB OAM 谱宽 sigma_L 随入射 alpha 和 SPP 调制荷 q 的变化。

左图固定若干 q 扫描 alpha；右图固定若干 alpha 扫描 q。sigma_L 由归一化
总 OAM 谱计算，表示整数 OAM 阶上的能量扩散程度。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, save_npz, save_params, spectral_width, style_axes, total_oam_spectrum_from_vector


SCRIPT = "oam_width_vs_alpha_and_spp"


def sigma_l_for(alpha: float, fork_ell: float, params, X, Y, R, PHI) -> float:
    """返回给定入射 alpha 和 SPP 调制荷 q=fork_ell 下的 OAM 谱宽 sigma_L。"""
    Ex, Ey = fvvb_field(alpha, R, PHI, params)
    if fork_ell != 0:
        Ex, Ey = apply_spp_to_vector(Ex, Ey, PHI, fork_ell)
    mu, _ = total_oam_spectrum_from_vector(Ex, Ey, X, Y, params.l_list, params.nr_fourier, params.nphi_fourier)
    return spectral_width(params.l_list, mu)[1]


def main() -> None:
    # ???????OAM ?? alpha/SPP ????????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720；alpha_scan=np.arange(0.1,3.01,0.05)，
    # fork_scan=np.arange(0.0,3.01,0.05)，fork_cases=[0,0.5,1.0]。
    params = default_params(grid_n=140, rho_max_factor=8.0, n_min=-40, n_max=40, lmax=12, nr_fourier=100, nphi_fourier=200)
    alpha_scan = np.arange(0.1, 3.0, 0.4)  # [关键变量] 快速步长 0.4；论文建议 np.arange(0.1,3.01,0.05)。
    fork_cases = [0.0, 0.5, 1.0]  # 固定 q 情形，用于 sigma_L vs alpha。
    fork_scan = np.arange(0.0, 3.01, 0.4)  # [关键变量] 快速 q 步长 0.4；论文建议 np.arange(0.0,3.01,0.05)。
    alpha_cases = [0.3, 0.5, 0.7]  # 固定 alpha 情形，用于 sigma_L vs q。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    sigma_vs_alpha = {f"fork_{q:.1f}": np.array([sigma_l_for(float(a), q, params, X, Y, R, PHI) for a in alpha_scan]) for q in fork_cases}
    sigma_vs_fork = {f"alpha_{a:.1f}": np.array([sigma_l_for(a, float(q), params, X, Y, R, PHI) for q in fork_scan]) for a in alpha_cases}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for q in fork_cases:
        axes[0].plot(alpha_scan, sigma_vs_alpha[f"fork_{q:.1f}"], marker="o", ms=3, lw=1.2, label=rf"$q={q:.1f}$")
    for a in alpha_cases:
        axes[1].plot(fork_scan, sigma_vs_fork[f"alpha_{a:.1f}"], lw=1.2, label=rf"$\alpha={a:.1f}$")
    style_axes(axes[0], r"Incident charge $\alpha$", r"$\sigma_L$", "(a)")
    style_axes(axes[1], r"SPP charge $q$", r"$\sigma_L$", "(b)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("OAM width vs alpha and SPP  FVVB OAM spectral width")
    fig.savefig(out_dir / "oam_width_vs_alpha_and_spp.png", dpi=300)
    save_npz(out_dir, "oam_width_vs_alpha_and_spp", alpha_scan=alpha_scan, fork_scan=fork_scan, sigma_vs_alpha=sigma_vs_alpha, sigma_vs_fork=sigma_vs_fork)
    save_params(out_dir, "oam_width_vs_alpha_and_spp", {"params": params, "fork_cases": fork_cases, "alpha_cases": alpha_cases, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
