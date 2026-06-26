"""OAM width vs fractional charge：FVB 与 FVVB 的 OAM 谱宽 sigma_L 随分数阶拓扑荷 alpha 的变化。

本脚本扫描 alpha，分别计算标量 FVB 和矢量 FVVB 的归一化 OAM 谱，再用
spectral_width 得到 OAM 谱标准差 sigma_L。sigma_L 越大表示 OAM 能量分布
越分散。默认参数为快速趋势绘图参数。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import (
    default_params,
    ensure_output_dir,
    fvb_field,
    fvvb_field,
    get_grid_from_params,
    oam_spectrum_angular_fourier,
    save_npz,
    save_params,
    spectral_width,
    style_axes,
    total_oam_spectrum_from_vector,
)


SCRIPT = "oam_width_vs_fractional_charge"


def main() -> None:
    # ???????OAM ?????????????????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720；alpha_list 建议步长 0.05 或 0.1 做收敛对比。
    params = default_params(grid_n=160, rho_max_factor=8.0, n_min=-50, n_max=50, lmax=10, nr_fourier=120, nphi_fourier=240)
    alpha_list = np.arange(0.1, 3.0, 0.2)  # [关键变量] 快速 alpha 步长 0.2；论文建议改为 np.arange(0.1,3.01,0.05)。
    l_list = params.l_list  # [关键变量] OAM 整数阶 ell。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    sigma_fvb, sigma_fvvb = [], []  # [关键变量] FVB/FVVB 的 OAM 谱宽 sigma_L。
    for alpha in alpha_list:
        U = fvb_field(float(alpha), R, PHI, params)
        El, Etot = oam_spectrum_angular_fourier(U, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        mu_fvb = El / max(float(np.sum(El)), np.finfo(float).eps)
        sigma_fvb.append(spectral_width(l_list, mu_fvb)[1])

        Ex, Ey = fvvb_field(float(alpha), R, PHI, params)
        mu_fvvb, _ = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        sigma_fvvb.append(spectral_width(l_list, mu_fvvb)[1])

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.plot(alpha_list, sigma_fvb, "o-", label="FVB", lw=1.4, ms=3)
    ax.plot(alpha_list, sigma_fvvb, "s-", label="FVVB", lw=1.4, ms=3)
    style_axes(ax, r"Fractional topological charge $\alpha$", r"OAM spectral width $\sigma_L$", "OAM width vs fractional charge")
    ax.legend()
    fig.savefig(out_dir / "oam_width_vs_fractional_charge.png", dpi=300)
    save_npz(out_dir, "oam_width_vs_fractional_charge", alpha_list=alpha_list, sigma_fvb=np.array(sigma_fvb), sigma_fvvb=np.array(sigma_fvvb))
    save_params(out_dir, "oam_width_vs_fractional_charge", {"params": params, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
