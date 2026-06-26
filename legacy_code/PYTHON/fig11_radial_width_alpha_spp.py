"""Fig. 11：FVVB LG 径向谱宽 sigma_p 随入射 alpha 和 SPP 调制荷 q 的变化。

本脚本与 Fig.9 结构相同，但指标从 OAM 谱宽 sigma_L 换为径向阶 p 的谱宽
sigma_p。sigma_p 基于 sigma+ 圆偏振分量的 LG 模态投影矩阵 Mpl 计算。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, radial_lg_spectrum, radial_width_from_mpl, save_npz, save_params, style_axes


SCRIPT = "fig11_radial_width_alpha_spp"


def sigma_p_for(alpha: float, fork_ell: float, params, X, Y, R, PHI, dx, dy) -> float:
    """返回给定 alpha 和 SPP 调制荷 q=fork_ell 下的径向谱宽 sigma_p。"""
    Ex, Ey = fvvb_field(alpha, R, PHI, params)
    if fork_ell != 0:
        Ex, Ey = apply_spp_to_vector(Ex, Ey, PHI, fork_ell)
    Ep = (Ex + 1j * Ey) / np.sqrt(2)  # [关键变量] sigma+ 圆偏振分量。
    l_list = np.arange(-params.lmax, params.lmax + 1)
    Mpl, Etot = radial_lg_spectrum(Ep, X, Y, dx, dy, params.wz, l_list, params.pmax)
    _, _, sigma_p, _ = radial_width_from_mpl(Mpl, Etot, conditional=True)
    return sigma_p


def main() -> None:
    # 论文参数指示：Fig.11 最终 sigma_p 扫描建议设置为：
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40, pmax=35；
    # alpha_scan=np.arange(0.1,3.01,0.05)，fork_scan=np.arange(0.0,3.01,0.05)。
    params = default_params(grid_n=100, rho_max_factor=8.0, n_min=-30, n_max=30, lmax=8, pmax=6)
    alpha_scan = np.arange(0.1, 3.0, 0.6)  # [关键变量] 快速步长 0.6；论文建议 np.arange(0.1,3.01,0.05)。
    fork_cases = [0.0, 0.5, 1.0]  # 固定 q 情形，用于 sigma_p vs alpha。
    fork_scan = np.arange(0.0, 3.01, 0.6)  # [关键变量] 快速 q 步长 0.6；论文建议 np.arange(0.0,3.01,0.05)。
    alpha_cases = [0.3, 0.5, 0.7]  # 固定 alpha 情形，用于 sigma_p vs q。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    sigma_vs_alpha = {f"fork_{q:.1f}": np.array([sigma_p_for(float(a), q, params, X, Y, R, PHI, dx, dy) for a in alpha_scan]) for q in fork_cases}
    sigma_vs_fork = {f"alpha_{a:.1f}": np.array([sigma_p_for(a, float(q), params, X, Y, R, PHI, dx, dy) for q in fork_scan]) for a in alpha_cases}

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for q in fork_cases:
        axes[0].plot(alpha_scan, sigma_vs_alpha[f"fork_{q:.1f}"], marker="o", ms=3, lw=1.2, label=rf"$q={q:.1f}$")
    for a in alpha_cases:
        axes[1].plot(fork_scan, sigma_vs_fork[f"alpha_{a:.1f}"], lw=1.2, label=rf"$\alpha={a:.1f}$")
    style_axes(axes[0], r"Incident charge $\alpha$", r"Radial width $\sigma_p$", "(a)")
    style_axes(axes[1], r"SPP charge $q$", r"Radial width $\sigma_p$", "(b)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Fig. 11  FVVB radial spectral width")
    fig.savefig(out_dir / "fig11_radial_width_alpha_spp.png", dpi=300)
    save_npz(out_dir, "fig11_radial_width_alpha_spp", alpha_scan=alpha_scan, fork_scan=fork_scan, sigma_vs_alpha=sigma_vs_alpha, sigma_vs_fork=sigma_vs_fork)
    save_params(out_dir, "fig11_radial_width_alpha_spp", {"params": params, "fork_cases": fork_cases, "alpha_cases": alpha_cases, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
