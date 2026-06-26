"""Fig. 7：固定 SPP 调制荷 q，改变入射分数阶拓扑荷 alpha 后的 FVVB OAM 谱。

该脚本与 Fig.6 互补：Fig.6 固定入射 alpha 扫 q，本脚本固定 q 扫 alpha。
SPP 模型仍为同一平面直接相位调制，需与论文文字保持一致。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, save_npz, save_params, total_oam_spectrum_from_vector


SCRIPT = "fig07_oam_incident_varying"


def main() -> None:
    # 论文参数指示：Fig.7 最终固定 SPP 扫 alpha 建议设置为：
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40,
    # nr_fourier=320, nphi_fourier=720；fork_ell 建议与 Fig.6 的 q=0.5 对齐，
    # alpha_list 可用 [0.3,0.5,0.7,1.3] 或按论文图注加密。
    params = default_params(grid_n=180, rho_max_factor=8.0, n_min=-60, n_max=60, lmax=10, nr_fourier=140, nphi_fourier=256)
    fork_ell = 0.5  # [关键变量] 固定 SPP/fork 调制荷 q；需人工核对图注来源。
    alpha_list = [0.3, 0.5, 0.7, 1.3]  # [关键变量] 入射分数阶拓扑荷扫描。
    l_list = params.l_list  # [关键变量] 输出 OAM 整数阶 ell。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    spectra = []
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), constrained_layout=True, sharey=True)
    for ax, alpha in zip(axes.ravel(), alpha_list):
        Ex0, Ey0 = fvvb_field(alpha, R, PHI, params)
        Ex, Ey = apply_spp_to_vector(Ex0, Ey0, PHI, fork_ell)
        mu, meta = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
        spectra.append(mu)
        ax.bar(l_list, mu, color="#55a868")
        ax.set_title(rf"$\alpha={alpha:.1f}$")
        ax.set_xlabel(r"$\ell$")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(l_list[0] - 0.5, l_list[-1] + 0.5)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\mu_\ell$")
    fig.suptitle(rf"Fig. 7  FVVB under fixed SPP ($q={fork_ell:.1f}$)")
    fig.savefig(out_dir / "fig07_oam_incident_varying.png", dpi=300)
    save_npz(out_dir, "fig07_oam_incident_varying", alpha_list=np.array(alpha_list), l_list=l_list, spectra=np.array(spectra))
    save_params(out_dir, "fig07_oam_incident_varying", {"params": params, "fork_ell": fork_ell, "alpha_list": alpha_list, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
