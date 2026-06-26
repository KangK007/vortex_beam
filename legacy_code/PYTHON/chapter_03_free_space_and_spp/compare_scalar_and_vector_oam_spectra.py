"""Scalar/vector OAM comparison：标量 FVB 与矢量 FVVB 的 OAM 谱对比。

本脚本在相同 alpha 和 OAM 截断范围下分别计算标量场 U 的 OAM 谱，以及
FVVB 的总 OAM 谱。该图用于说明矢量偏振结构会改变 OAM 能量分布，不能
直接用标量分数阶涡旋结论替代 FVVB 结论。"""

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
    total_oam_spectrum_from_vector,
)


SCRIPT = "scalar_vector_oam_comparison"


def main() -> None:
    # ?????????/?? OAM ???? FVB/FVVB ?????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720，避免截断差异造成假对比。
    params = default_params(grid_n=180, rho_max_factor=8.0, n_min=-60, n_max=60, lmax=6, nr_fourier=140, nphi_fourier=256)
    alpha_list = [0.5, 1.5, 2.5]  # [关键变量] 入射分数阶拓扑荷。
    l_list = params.l_list  # [关键变量] OAM 整数阶 ell。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    fvb_spectra, fvvb_spectra = [], []
    fig, axes = plt.subplots(2, len(alpha_list), figsize=(10, 5.5), constrained_layout=True, sharex=True)
    for j, alpha in enumerate(alpha_list):
        U = fvb_field(alpha, R, PHI, params)  # 标量 FVB 复振幅。
        El, Etot = oam_spectrum_angular_fourier(U, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        mu_fvb = El / max(float(np.sum(El)), np.finfo(float).eps)
        fvb_spectra.append(mu_fvb)

        Ex, Ey = fvvb_field(alpha, R, PHI, params)  # 矢量 FVVB 线偏振分量。
        mu_fvvb, meta = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        fvvb_spectra.append(mu_fvvb)

        axes[0, j].bar(l_list, mu_fvb, color="#d95f02")
        axes[0, j].set_title(rf"FVB, $\alpha={alpha:.1f}$")
        axes[1, j].bar(l_list, mu_fvvb, color="#1b9e77")
        axes[1, j].set_title(rf"FVVB, $\alpha={alpha:.1f}$")
        for ax in axes[:, j]:
            ax.grid(True, alpha=0.2)
            ax.set_xlim(l_list[0] - 0.5, l_list[-1] + 0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mu_\ell$")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"OAM order $\ell$")
    fig.suptitle("Scalar/vector OAM comparison  FVB/FVVB OAM-spectrum comparison")
    fig.savefig(out_dir / "scalar_vector_oam_comparison.png", dpi=300)
    save_npz(out_dir, "scalar_vector_oam_comparison", alpha_list=np.array(alpha_list), l_list=l_list, fvb=np.array(fvb_spectra), fvvb=np.array(fvvb_spectra))
    save_params(out_dir, "scalar_vector_oam_comparison", {"params": params, "alpha_list": alpha_list, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
