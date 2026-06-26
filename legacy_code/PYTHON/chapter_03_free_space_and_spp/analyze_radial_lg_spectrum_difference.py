"""Radial LG spectrum difference：SPP 调制前后 FVVB 的 LG 径向模态谱 nu_p 及其差分。

本脚本针对 sigma+ 圆偏振分量计算 LG_p^ell 投影，比较分数阶 SPP 与整数
SPP 调制前后的径向模态能量分布变化。nu_p 为对所有 ell 求和后的径向阶
归一化能量权重。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, radial_lg_spectrum, radial_width_from_mpl, save_npz, save_params, style_axes


SCRIPT = "radial_lg_spectrum_difference"


def radial_case(alpha: float, fork_ell: float, params, X, Y, R, PHI, dx, dy):
    """计算给定 alpha 与 SPP 调制荷下，SPP 前后 sigma+ 的径向谱差分。"""
    Ex0, Ey0 = fvvb_field(alpha, R, PHI, params)
    Ep0 = (Ex0 + 1j * Ey0) / np.sqrt(2)  # [关键变量] SPP 前 sigma+ 圆偏振分量。
    Ex1, Ey1 = apply_spp_to_vector(Ex0, Ey0, PHI, fork_ell)
    Ep1 = (Ex1 + 1j * Ey1) / np.sqrt(2)  # [关键变量] SPP 后 sigma+ 圆偏振分量。
    l_list = np.arange(-params.lmax, params.lmax + 1)
    M0, E0 = radial_lg_spectrum(Ep0, X, Y, dx, dy, params.wz, l_list, params.pmax)
    M1, E1 = radial_lg_spectrum(Ep1, X, Y, dx, dy, params.wz, l_list, params.pmax)
    nu0 = np.sum(M0, axis=1) / max(E0, np.finfo(float).eps)
    nu1 = np.sum(M1, axis=1) / max(E1, np.finfo(float).eps)
    return nu0, nu1, nu1 - nu0


def main() -> None:
    # ????????? LG ???????????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40, pmax=35；
    # LG 分析基腰 w_an=params.wz，若论文讨论基腰敏感性需与 LG basis-waist sensitivity 对照。
    params = default_params(grid_n=120, rho_max_factor=8.0, n_min=-40, n_max=40, lmax=10, pmax=8)
    alpha = 0.5  # [关键变量] 入射分数阶拓扑荷。
    fork_fractional = 0.5  # 分数阶 SPP 调制荷 q。
    fork_integer = 1.0  # 整数阶 SPP 调制荷 q。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    nu0_f, nu1_f, dnu_f = radial_case(alpha, fork_fractional, params, X, Y, R, PHI, dx, dy)
    nu0_i, nu1_i, dnu_i = radial_case(alpha, fork_integer, params, X, Y, R, PHI, dx, dy)
    p = np.arange(params.pmax + 1)
    p_plot = p <= 8

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), constrained_layout=True)
    axes[0, 0].semilogy(p[p_plot], nu0_f[p_plot], "o-", label="before")
    axes[0, 0].semilogy(p[p_plot], nu1_f[p_plot], "s-", label=f"after q={fork_fractional}")
    axes[0, 1].bar(p[p_plot], dnu_f[p_plot], color="#4c72b0")
    axes[1, 0].semilogy(p[p_plot], nu0_i[p_plot], "o-", label="before")
    axes[1, 0].semilogy(p[p_plot], nu1_i[p_plot], "s-", label=f"after q={fork_integer}")
    axes[1, 1].bar(p[p_plot], dnu_i[p_plot], color="#dd8452")
    style_axes(axes[0, 0], "Radial order p", r"$\nu_p$", "(a) fractional SPP")
    style_axes(axes[0, 1], "Radial order p", r"$\Delta\nu_p$", "(b)")
    style_axes(axes[1, 0], "Radial order p", r"$\nu_p$", "(c) integer SPP")
    style_axes(axes[1, 1], "Radial order p", r"$\Delta\nu_p$", "(d)")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Radial LG spectrum difference  Radial modal spectra before/after SPP")
    fig.savefig(out_dir / "radial_lg_spectrum_difference.png", dpi=300)
    save_npz(out_dir, "radial_lg_spectrum_difference", p=p, nu0_fractional=nu0_f, nu1_fractional=nu1_f, dnu_fractional=dnu_f, nu0_integer=nu0_i, nu1_integer=nu1_i, dnu_integer=dnu_i)
    save_params(out_dir, "radial_lg_spectrum_difference", {"params": params, "alpha": alpha, "fork_fractional": fork_fractional, "fork_integer": fork_integer, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
