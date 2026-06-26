"""Modal-decomposition convergence：OAM 方位阶截断 Lmax 与 LG 径向阶截断 Pmax 的收敛性分析。

本脚本比较无 SPP、整数 SPP 和分数阶 SPP 三种情形下，OAM 谱宽 sigma_L、
径向谱宽 sigma_p 与捕获功率 Pcap 随截断范围变化的趋势。该图用于说明
论文图使用的模态截断是否足以支撑主要结论。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import (
    default_params,
    apply_spp_to_vector,
    ensure_output_dir,
    fvvb_field,
    get_grid_from_params,
    radial_lg_spectrum,
    radial_width_from_mpl,
    save_npz,
    save_params,
    spectral_width,
    style_axes,
    total_oam_spectrum_from_vector,
)


SCRIPT = "modal_decomposition_convergence"


def compute_case(alpha: float, fork_ell: float, params):
    """生成指定 alpha 与 SPP 调制荷下的网格和 FVVB 场。"""
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    Ex, Ey = fvvb_field(alpha, R, PHI, params)
    if fork_ell != 0:
        Ex, Ey = apply_spp_to_vector(Ex, Ey, PHI, fork_ell)
    return X, Y, R, PHI, dx, dy, Ex, Ey


def main() -> None:
    # ??????????????????????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, nr_fourier=320,
    # nphi_fourier=720；lmax_list 可设 [10,20,30,40]，pmax_list 可设 [10,20,30,35]。
    params = default_params(grid_n=130, rho_max_factor=8.0, n_min=-40, n_max=40, nr_fourier=100, nphi_fourier=200)
    alpha = 0.5  # [关键变量] 入射分数阶拓扑荷。
    cases = [("no SPP", 0.0), ("integer SPP", 1.0), ("fractional SPP", 0.5)]  # 三类 SPP 调制情形。
    lmax_list = np.array([6, 10, 14, 18])  # [关键变量] OAM 方位阶截断扫描。
    pmax_list = np.array([3, 6, 9])  # [关键变量] LG 径向阶截断扫描。
    out_dir = ensure_output_dir(SCRIPT)

    sigma_l = {name: [] for name, _ in cases}
    sigma_p = {name: [] for name, _ in cases}
    pcap = {name: [] for name, _ in cases}

    for name, fork_ell in cases:
        X, Y, R, PHI, dx, dy, Ex, Ey = compute_case(alpha, fork_ell, params)
        for lmax in lmax_list:
            l_list = np.arange(-lmax, lmax + 1)
            mu, _ = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
            sigma_l[name].append(spectral_width(l_list, mu)[1])
        Ep = (Ex + 1j * Ey) / np.sqrt(2)  # [关键变量] sigma+ 圆偏振分量，用于 LG 径向谱。
        for pmax in pmax_list:
            l_list = np.arange(-12, 13)
            Mpl, Etot = radial_lg_spectrum(Ep, X, Y, dx, dy, params.wz, l_list, int(pmax))
            nu_p, mean_p, sig_p, cap = radial_width_from_mpl(Mpl, Etot, conditional=True)
            sigma_p[name].append(sig_p)
            pcap[name].append(cap)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for name, _ in cases:
        axes[0].plot(lmax_list, sigma_l[name], "o-", label=name)
        axes[1].plot(pmax_list, sigma_p[name], "s-", label=name)
        axes[2].plot(pmax_list, pcap[name], "^-", label=name)
    style_axes(axes[0], r"Azimuthal truncation $L_{max}$", r"$\sigma_L$", "(a)")
    style_axes(axes[1], r"Radial truncation $P_{max}$", r"$\sigma_p$", "(b)")
    style_axes(axes[2], r"Radial truncation $P_{max}$", r"Captured power $P_{cap}$", "(c)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Modal-decomposition convergence  Modal-decomposition convergence")
    fig.savefig(out_dir / "modal_decomposition_convergence.png", dpi=300)
    save_npz(out_dir, "modal_decomposition_convergence", lmax_list=lmax_list, pmax_list=pmax_list, sigma_l=sigma_l, sigma_p=sigma_p, pcap=pcap)
    save_params(out_dir, "modal_decomposition_convergence", {"params": params, "alpha": alpha, "cases": cases})
    plt.close(fig)


if __name__ == "__main__":
    main()
