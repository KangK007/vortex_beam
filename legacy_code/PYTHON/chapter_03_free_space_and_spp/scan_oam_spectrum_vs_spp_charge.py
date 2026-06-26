"""OAM spectrum vs SPP charge：固定入射 alpha，改变 SPP/叉形相位板调制荷 q 后的 FVVB OAM 谱。

当前 SPP 实现为在同一计算平面对 Ex/Ey 共同乘以 H(phi)=exp(i*q*phi)。若
论文文字涉及“SPP 位于中间面并继续传播”，需要另行使用分段传播模型验证。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, save_npz, save_params, total_oam_spectrum_from_vector


SCRIPT = "oam_spectrum_vs_spp_charge"


def main() -> None:
    # ??????????? alpha ? SPP ????????
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720；fork_list 可用 [0.1,0.3,0.5,0.7,1.0]
    # 或加密为 np.arange(0.0,1.01,0.1)，并确认当前模型是同一平面等效相位调制。
    params = default_params(grid_n=180, rho_max_factor=8.0, n_min=-60, n_max=60, lmax=10, nr_fourier=140, nphi_fourier=256)
    alpha = 0.5  # [关键变量] 入射分数阶拓扑荷；图注公式未被 Word 文本提取，需人工核对。
    fork_list = [0.1, 0.3, 0.5, 0.7, 1.0]  # [关键变量] SPP/fork 调制荷 q。
    l_list = params.l_list  # [关键变量] 输出 OAM 整数阶 ell。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    Ex0, Ey0 = fvvb_field(alpha, R, PHI, params)
    spectra = []
    fig, axes = plt.subplots(1, len(fork_list), figsize=(13, 3.2), constrained_layout=True, sharey=True)
    for ax, fork_ell in zip(axes, fork_list):
        Ex, Ey = apply_spp_to_vector(Ex0, Ey0, PHI, fork_ell)  # 对两个线偏振分量共同施加 SPP 相位。
        mu, meta = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        spectra.append(mu)
        ax.bar(l_list, mu, color="#4c72b0")
        ax.set_title(rf"$q={fork_ell:.1f}$")
        ax.set_xlabel(r"$\ell$")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(l_list[0] - 0.5, l_list[-1] + 0.5)
    axes[0].set_ylabel(r"$\mu_\ell$")
    fig.suptitle(rf"OAM spectrum vs SPP charge  FVVB after SPP modulation, incident $\alpha={alpha:.1f}$")
    fig.savefig(out_dir / "oam_spectrum_vs_spp_charge.png", dpi=300)
    save_npz(out_dir, "oam_spectrum_vs_spp_charge", fork_list=np.array(fork_list), l_list=l_list, spectra=np.array(spectra))
    save_params(out_dir, "oam_spectrum_vs_spp_charge", {"params": params, "alpha": alpha, "fork_list": fork_list, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
