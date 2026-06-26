"""Supplementary Fig. S2：LG 分析基腰 w_an 对归一化模态谱宽的影响。

该脚本改变 LG 分析基腰与传播后束腰 w(z) 的比值，考察 OAM 谱宽 sigma_L
和径向谱宽 sigma_p 的敏感性。这里的 sigma_L 也由 LG 投影矩阵 Mpl 汇总得到，
因此会随分析基腰变化。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, apply_spp_to_vector, ensure_output_dir, fvvb_field, get_grid_from_params, radial_lg_spectrum, radial_width_from_mpl, save_npz, save_params, spectral_width, style_axes


SCRIPT = "supp_fig_s2_lg_waist_sensitivity"


def widths_for_basis(alpha: float, fork_ell: float, basis_waist: float, params, X, Y, R, PHI, dx, dy):
    """给定 LG 分析基腰 basis_waist，计算归一化前的 sigma_L 和 sigma_p。"""
    Ex, Ey = fvvb_field(alpha, R, PHI, params)
    if fork_ell != 0:
        Ex, Ey = apply_spp_to_vector(Ex, Ey, PHI, fork_ell)
    Ep = (Ex + 1j * Ey) / np.sqrt(2)  # [关键变量] sigma+ 圆偏振分量。
    Mpl, Etot = radial_lg_spectrum(Ep, X, Y, dx, dy, basis_waist, params.l_list, params.pmax)
    # Supplement Fig. S2 讨论的是 LG-basis waist sensitivity，
    # 因此这里的 OAM 宽度也应由 LG 投影后的 M_{p,l} 得到，
    # 而不是与基腰无关的角向傅里叶 OAM 宽度。
    mu_l_lg = np.sum(Mpl, axis=0)
    mu_l_lg = mu_l_lg / max(float(np.sum(mu_l_lg)), np.finfo(float).eps)
    sigma_l = spectral_width(params.l_list, mu_l_lg)[1]
    _, _, sigma_p, _ = radial_width_from_mpl(Mpl, Etot, conditional=True)
    return sigma_l, sigma_p


def main() -> None:
    # 论文参数指示：Fig.S2 最终基腰敏感性建议设置为：
    # grid_n=400, rho_max_factor=8.0, n_min=-200, n_max=200, lmax=40, pmax=35,
    # nr_fourier=320, nphi_fourier=720；waist_ratios 可加密为 np.linspace(0.8,1.2,9)。
    params = default_params(grid_n=100, rho_max_factor=8.0, n_min=-30, n_max=30, lmax=8, pmax=6, nr_fourier=80, nphi_fourier=160)
    alpha = 0.5  # [关键变量] 入射分数阶拓扑荷。
    cases = [("no SPP", 0.0), ("integer SPP", 1.0), ("fractional SPP", 0.5)]
    waist_ratios = np.linspace(0.8, 1.2, 5)  # [关键变量] w_an / w(z) 扫描。
    X, Y, R, PHI, dx, dy = get_grid_from_params(params)
    out_dir = ensure_output_dir(SCRIPT)

    sigma_l_norm, sigma_p_norm = {}, {}
    for name, fork_ell in cases:
        sl, sp = [], []
        for ratio in waist_ratios:
            a, b = widths_for_basis(alpha, fork_ell, ratio * params.wz, params, X, Y, R, PHI, dx, dy)
            sl.append(a)
            sp.append(b)
        sl = np.array(sl)
        sp = np.array(sp)
        ref = np.argmin(np.abs(waist_ratios - 1.0))  # 以 w_an = w(z) 作为归一化参考。
        sigma_l_norm[name] = sl / max(sl[ref], np.finfo(float).eps)
        sigma_p_norm[name] = sp / max(sp[ref], np.finfo(float).eps)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, _ in cases:
        axes[0].plot(waist_ratios, sigma_l_norm[name], "o-", label=name)
        axes[1].plot(waist_ratios, sigma_p_norm[name], "s-", label=name)
    style_axes(axes[0], r"Normalized LG-basis waist $w_{an}/w(z)$", r"Normalized $\sigma_L$", "(a)")
    style_axes(axes[1], r"Normalized LG-basis waist $w_{an}/w(z)$", r"Normalized $\sigma_p$", "(b)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Fig. S2  Basis-waist sensitivity")
    fig.savefig(out_dir / "supp_fig_s2_lg_waist_sensitivity.png", dpi=300)
    save_npz(out_dir, "supp_fig_s2_lg_waist_sensitivity", waist_ratios=waist_ratios, sigma_l_norm=sigma_l_norm, sigma_p_norm=sigma_p_norm)
    save_params(out_dir, "supp_fig_s2_lg_waist_sensitivity", {"params": params, "alpha": alpha, "cases": cases, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
