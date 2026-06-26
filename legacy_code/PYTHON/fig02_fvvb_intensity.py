"""Fig. 2：不同分数阶拓扑荷 alpha 下 FVVB 的总强度分布。

本脚本用于论文自由空间 FVVB 强度图。主要流程：设置快速绘图参数 -> 生成
X/Y/R/PHI 网格 -> 对每个 alpha 构造 Ex/Ey -> 计算总强度 I=|Ex|^2+|Ey|^2
并归一化显示。输出 PNG、NPZ 和参数 JSON；重复运行会覆盖同名 outputs。"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, ensure_output_dir, fvvb_field, get_grid_from_params, save_npz, save_params


SCRIPT = "fig02_fvvb_intensity"


def main() -> None:
    # 默认使用快速验证参数，便于逐个脚本自动测试。
    # 论文参数指示：Fig.2 高分辨率强度图建议设置为：
    # grid_n=400, rho_max_factor=1.5（近场主体）或 3.0（含更大外环窗口），
    # n_min=-200, n_max=200；若要做最终收敛复核，可再试 grid_n=512。
    params = default_params(grid_n=400, rho_max_factor=1.5, n_min=-200, n_max=200)
    alpha_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.5]  # [关键变量] 入射分数阶拓扑荷扫描。

    X, Y, R, PHI, dx, dy = get_grid_from_params(params, rho_max_factor=1.5)
    out_dir = ensure_output_dir(SCRIPT)

    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.0), constrained_layout=True)
    intensities = []
    for ax, alpha in zip(axes.ravel(), alpha_list):
        Ex, Ey = fvvb_field(alpha, R, PHI, params)
        I = np.abs(Ex) ** 2 + np.abs(Ey) ** 2  # [关键变量] FVVB 总强度。
        I_norm = I / max(float(np.max(I)), np.finfo(float).eps)
        intensities.append(I_norm)
        ax.imshow(
            I_norm,
            extent=[X.min() * 1e3, X.max() * 1e3, Y.min() * 1e3, Y.max() * 1e3],  # m -> mm 用于绘图坐标。
            origin="lower",
            cmap="inferno",
            vmin=0,
            vmax=1,
        )
        ax.set_title(rf"$\alpha={alpha:.1f}$")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Fig. 2  FVVB intensity distributions")
    fig.savefig(out_dir / "fig02_fvvb_intensity.png", dpi=500)
    save_npz(out_dir, "fig02_fvvb_intensity", alpha_list=np.array(alpha_list), X=X, Y=Y, intensity=np.array(intensities))
    save_params(out_dir, "fig02_fvvb_intensity", {"params": params, "alpha_list": alpha_list, "dx": dx, "dy": dy})
    plt.close(fig)


if __name__ == "__main__":
    main()
