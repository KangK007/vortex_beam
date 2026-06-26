"""Fig. 3：不同分数阶拓扑荷 alpha 下 FVVB 的总 OAM 谱。

本脚本用于展示 FVVB 总场在整数 OAM 阶 ell 上的归一化能量权重 mu_l。
OAM 谱由两个圆偏振分量合并得到；Pcap 表示当前 l_list 截断范围捕获的
能量比例。这里使用快速绘图参数，最终论文图需核对参数 JSON。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, ensure_output_dir, fvvb_field, get_grid_from_params, save_npz, save_params, style_axes, total_oam_spectrum_from_vector


SCRIPT = "fig03_fvvb_oam_spectrum"


def main() -> None:
    # 论文参数指示：Fig.3 最终 OAM 谱建议设置为：
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720；同时检查每个 alpha 的 Pcap 是否接近 1。
    params = default_params(grid_n=180, rho_max_factor=8.0, n_min=-60, n_max=60, lmax=8, nr_fourier=140, nphi_fourier=256)
    alpha_list = [0.3, 0.5, 0.7, 1.3, 1.5, 1.7]  # [关键变量] 分数阶拓扑荷扫描。
    l_list = params.l_list  # [关键变量] OAM 整数阶 ell，范围由 lmax 决定。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    spectra = []
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), constrained_layout=True)
    for ax, alpha in zip(axes.ravel(), alpha_list):
        Ex, Ey = fvvb_field(alpha, R, PHI, params)
        # mu 为归一化总 OAM 谱；meta['Pcap'] 为当前 ell 截断范围捕获功率。
        mu, meta = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        spectra.append(mu)
        ax.bar(l_list, mu, width=0.8, color="#1f77b4")
        ax.set_title(rf"$\alpha={alpha:.1f}$, $P_{{cap}}={meta['Pcap']:.3f}$")
        ax.set_xlim(l_list[0] - 0.5, l_list[-1] + 0.5)
        ax.set_ylim(0, max(mu) * 1.2 if np.max(mu) > 0 else 1)
        ax.grid(True, alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"OAM order $\ell$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mu_\ell$")
    fig.suptitle("Fig. 3  FVVB total OAM spectra")
    fig.savefig(out_dir / "fig03_fvvb_oam_spectrum.png", dpi=300)
    save_npz(out_dir, "fig03_fvvb_oam_spectrum", alpha_list=np.array(alpha_list), l_list=l_list, spectra=np.array(spectra))
    save_params(out_dir, "fig03_fvvb_oam_spectrum", {"params": params, "alpha_list": alpha_list, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
