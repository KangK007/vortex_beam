"""主线 B：FVVB 在大气湍流中的传播与 OAM 谱演化批量扫描。

本脚本对应“毕业设计进度评估与后续研究计划”中的湍流传播主线：扫描
[关键变量] alpha（分数阶拓扑荷）、distance（传播距离）和 cn2（湍流强度），
对每组参数进行 Monte Carlo 相位屏传播，并统计总 OAM 谱、偏振分辨 OAM
谱、谱宽、谱熵、相似度、Stokes 参数和相位屏诊断量。

重要模型假设：湍流由各向同性 modified von Karman 公共标量相位屏表示，
相位屏同等作用于 Ex/Ey 或 sigma+/sigma- 两个分量；不包含偏振相关散射、
双折射或真实退偏振。本脚本适合趋势分析，写入论文定量结论前需结合实验
几何和采样条件再次校验。"""

from __future__ import annotations

import argparse
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from fvvb_core import default_params, ensure_output_dir, fvvb_field, fvvb_field_waist, get_grid_from_params, make_cartesian_grid, save_npz, save_params, style_axes
from oam_metrics import build_alpha_to_ell_matrix, compute_oam_summary, polarization_resolved_oam
from polarization_metrics import polarization_summary
from propagation import assert_finite_array, propagate_vector_split_step_circular, propagate_vector_split_step_linear, vector_energy, vector_field_diagnostics
from turbulence import (
    TurbulenceParams,
    estimate_phase_psd_radial,
    fried_parameter,
    make_phase_screens,
    phase_screen_stats,
    phase_structure_function,
    rytov_variance,
)


SCRIPT = "run_turbulence_scan"
# 湍流模型假设会写入 params.json，避免论文中误称为完整偏振退相干模型。
ASSUMPTION = (
    "Atmospheric turbulence is represented by isotropic modified von Karman "
    "common scalar phase screens applied equally to both polarization components; "
    "polarization-dependent scattering, birefringence, and true depolarization are not modeled."
)
# [关键变量] COMPONENTS：分别统计总场、线偏振分量和圆偏振分量的 OAM 谱。
COMPONENTS = ["total", "Ex", "Ey", "sigma_plus", "sigma_minus"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan FVVB OAM-spectrum evolution under turbulence.")
    parser.add_argument("--quick", action="store_true", help="Use a small grid and short Monte Carlo scan.")
    parser.add_argument("--mc", type=int, default=None, help="Monte Carlo count per parameter set.")
    parser.add_argument("--grid-n", type=int, default=None, help="Cartesian grid size.")
    parser.add_argument("--basis", choices=["linear", "circular"], default="linear", help="Propagation basis.")
    parser.add_argument("--n-screens", type=int, default=1, help="Number of split-step phase screens.")
    parser.add_argument("--seed", type=int, default=20260624, help="Base random seed.")
    parser.add_argument("--propagation", choices=["same", "padded"], default="same", help="Angular-spectrum boundary handling.")
    parser.add_argument("--initial-field", choices=["waist", "propagated"], default="waist", help="Initial FVVB model: waist plane or legacy propagated params.z plane.")
    parser.add_argument("--pad-factor", type=float, default=1.0, help="FFT zero-padding factor when --propagation padded.")
    parser.add_argument("--rho-max-mm", type=float, default=None, help="Absolute half-width of the Cartesian window in mm.")
    parser.add_argument("--rho-max-factor", type=float, default=None, help="Window half-width in units of params.wz.")
    parser.add_argument("--edge-fraction", type=float, default=0.05, help="Border fraction used by edge diagnostics.")
    parser.add_argument("--edge-warn-threshold", type=float, default=1e-3, help="Warning threshold for edge energy fraction.")
    parser.add_argument("--fail-on-edge", action="store_true", help="Abort when edge energy fraction exceeds threshold.")
    parser.add_argument("--alpha-list", type=str, default=None, help="Comma-separated alpha list, e.g. 0.5,1.5.")
    parser.add_argument("--distance-list", type=str, default=None, help="Comma-separated distance list in m.")
    parser.add_argument("--cn2-list", type=str, default=None, help="Comma-separated Cn2 list in m^(-2/3).")
    parser.add_argument("--n-min", type=int, default=None, help="Berry expansion lower order.")
    parser.add_argument("--n-max", type=int, default=None, help="Berry expansion upper order.")
    parser.add_argument("--lmax", type=int, default=None, help="Maximum displayed/projected OAM order.")
    parser.add_argument("--nr-fourier", type=int, default=None, help="Radial samples for angular Fourier OAM projection.")
    parser.add_argument("--nphi-fourier", type=int, default=None, help="Azimuthal samples for angular Fourier OAM projection.")
    return parser.parse_args()


def parse_float_list(text: str | None, name: str) -> list[float] | None:
    """解析命令行逗号分隔浮点列表。"""
    if text is None:
        return None
    try:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {text}") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def scan_config(args: argparse.Namespace) -> dict[str, Any]:
    """生成批量扫描配置。

    quick 模式用于调试流程和语法/性能检查；非 quick 模式用于较完整的趋势扫描。
    二者的 grid_n、n_min/n_max、lmax、Monte Carlo 次数等均不同，不能把 quick
    输出直接当作最终论文高精度结果。

    论文参数指示：最终湍流章节建议设置一组明确的 paper baseline，便于与
    杜乾论文比较并写入方法章节：
    - grid_n=256 或 300 起步，收敛复核可升至 400；
    - n_min=-100, n_max=100 起步，最终复核建议 n_min=-200, n_max=200；
    - lmax=20 起步，最终 OAM 谱复核建议 lmax=40；
    - nr_fourier=200, nphi_fourier=360 起步，最终复核建议 320/720；
    - alpha_list=[0.3,0.5,0.7,1.3,1.5,1.7]，如做检测数据集再扩展为 0.05 或 0.1 间隔；
    - distance_list=[500.0,1000.0,1500.0] m；
    - cn2_list=[0.0,1e-16,1e-15,1e-14] m^(-2/3)，0.0 为无湍流参考；
    - n_mc>=50 用于趋势图，n_mc>=100 更适合论文统计误差条；
    - n_screens=1 作为杜乾可比单屏 baseline，同时复核 n_screens=3 和 5；
    - inner_scale=2e-3 m, outer_scale=50.0 m。
    """
    if args.quick:
        # 快速参数：只用于检查代码流程、数组维度和绘图是否正常；不建议写入论文结论。
        # 当前 quick：grid_n=128, n_min/n_max=-40/40, lmax=10, nr/nphi=100/180,
        # alpha=[0.5,1.5,2.5], distance=[200 m], cn2=[0,1e-16,1e-15,1e-14], n_mc=5。
        cfg = {
            "grid_n": 128,
            "n_min": -40,
            "n_max": 40,
            "lmax": 10,
            "nr_fourier": 100,
            "nphi_fourier": 180,
            "alpha_list": [0.5, 1.5, 2.5],       # [关键变量] 入射分数阶拓扑荷扫描，无量纲。
            "distance_list": [200],            # [关键变量] 传播距离扫描，单位 m。
            "cn2_list": [0.0, 1e-16, 1e-15, 1e-14],  # [关键变量] C_n^2 湍流强度，单位 m^(-2/3)。
            "n_mc": 5,                           # [关键变量] 每组参数 Monte Carlo realization 数。
        }
    else:
        # 趋势扫描参数：比 quick 更完整，但仍需按最终论文要求检查网格、截断和 Monte Carlo 收敛。
        # 论文参数调整入口：建议改为 grid_n=256~400, n_min/n_max=-100/100 到 -200/200,
        # lmax=20~40, nr/nphi=200/360 到 320/720, distance=[500,1000,1500],
        # cn2=[0,1e-16,1e-15,1e-14], n_mc=50~100。
        cfg = {
            "grid_n": 180,
            "n_min": -60,
            "n_max": 60,
            "lmax": 12,
            "nr_fourier": 140,
            "nphi_fourier": 256,
            "alpha_list": [0.3, 0.5, 0.7, 1.3, 1.5, 1.7],
            "distance_list": [100.0, 200.0, 500.0],
            "cn2_list": [0.0, 1e-16, 3e-16, 1e-15, 3e-15, 1e-14],
            "n_mc": 10,
        }
    if args.grid_n is not None:
        cfg["grid_n"] = args.grid_n
    if args.mc is not None:
        cfg["n_mc"] = args.mc
    if args.n_min is not None:
        cfg["n_min"] = args.n_min
    if args.n_max is not None:
        cfg["n_max"] = args.n_max
    if args.lmax is not None:
        cfg["lmax"] = args.lmax
    if args.nr_fourier is not None:
        cfg["nr_fourier"] = args.nr_fourier
    if args.nphi_fourier is not None:
        cfg["nphi_fourier"] = args.nphi_fourier

    alpha_override = parse_float_list(args.alpha_list, "alpha-list")
    distance_override = parse_float_list(args.distance_list, "distance-list")
    cn2_override = parse_float_list(args.cn2_list, "cn2-list")
    if alpha_override is not None:
        cfg["alpha_list"] = alpha_override
    if distance_override is not None:
        cfg["distance_list"] = distance_override
    if cn2_override is not None:
        cfg["cn2_list"] = cn2_override
    if cfg["n_min"] > cfg["n_max"]:
        raise ValueError("n_min must be <= n_max")

    cfg["basis"] = args.basis
    cfg["n_screens"] = args.n_screens
    cfg["seed"] = args.seed
    cfg["propagation"] = args.propagation
    cfg["initial_field"] = args.initial_field
    cfg["pad_factor"] = args.pad_factor
    cfg["rho_max_mm"] = args.rho_max_mm
    cfg["rho_max_factor"] = 8.0 if args.rho_max_factor is None else args.rho_max_factor
    cfg["edge_fraction"] = args.edge_fraction
    cfg["edge_warn_threshold"] = args.edge_warn_threshold
    cfg["fail_on_edge"] = args.fail_on_edge
    # 论文参数指示：inner_scale=2e-3 m、outer_scale=50.0 m 与杜乾论文 baseline 一致；
    # 若后续实验相位屏或湍流箱给出不同内/外尺度，应在这里统一改动并写入参数表。
    cfg["inner_scale"] = 2e-3
    cfg["outer_scale"] = 50.0
    return cfg


def initial_fvvb_field(alpha: float, R: np.ndarray, PHI: np.ndarray, params, initial_field: str):
    """按配置生成入射 FVVB 场。

    waist 表示在束腰面 z=0 构造 FVVB 后再传播 distance；propagated 保留旧流程，
    即先用 params.z 平面的 E'_n 解析场作为入射面。
    """
    if initial_field == "waist":
        return fvvb_field_waist(alpha, R, PHI, params)
    if initial_field == "propagated":
        return fvvb_field(alpha, R, PHI, params)
    raise ValueError(f"Unknown initial_field: {initial_field}")


def initial_field_metadata(cfg: dict[str, Any], params) -> dict[str, Any]:
    """返回初始场模型元数据，写入参数 JSON。"""
    if cfg["initial_field"] == "waist":
        return {
            "initial_field": "waist",
            "initial_plane_z": 0.0,
            "initial_field_model": "LG_p0_Berry_FVVB_waist",
            "grid_reference_radius": "w0",
            "waist_radius_w0": params.w0,
        }
    return {
        "initial_field": "propagated",
        "initial_plane_z": params.z,
        "initial_field_model": "propagated_Eprime_FVVB",
        "grid_reference_radius": "wz",
        "waist_radius_w0": params.w0,
    }


def propagate_with_basis(Ex, Ey, dx, dy, wavelength, distance, phase_screens, basis, propagation="same", pad_factor=1.0):
    """按指定传播基和数值边界处理进行 split-step 湍流传播。

    [关键变量] basis="linear" 表示直接传播 Ex/Ey；basis="circular" 表示先转为
    sigma+/sigma- 后传播，再转回 Ex/Ey。两种基都使用同一组公共标量相位屏。
    propagation="padded" 时使用零填充角谱传播以降低 FFT 周期回卷风险。
    """
    if basis == "linear":
        return propagate_vector_split_step_linear(Ex, Ey, dx, dy, wavelength, distance, phase_screens, propagation, pad_factor)
    if basis == "circular":
        return propagate_vector_split_step_circular(Ex, Ey, dx, dy, wavelength, distance, phase_screens, propagation, pad_factor)
    raise ValueError(f"Unknown basis: {basis}")


def plot_oam_spectra(out_dir, l_list, cn2_list, spectra_total_mean, alpha_list, distance_list) -> None:
    # 默认展示第一个 alpha 和第一个传播距离下，不同 cn2 的总 OAM 谱。
    ia = 0
    idist = 0
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for ic, cn2 in enumerate(cn2_list):
        ax.plot(l_list, spectra_total_mean[ia, idist, ic], "o-", lw=1.1, ms=3, label=rf"$C_n^2={cn2:.0e}$")
    style_axes(ax, r"OAM order $\ell$", r"$\mu_\ell$", rf"Total OAM spectra, $\alpha={alpha_list[ia]:.1f}$, z={distance_list[idist]:g} m")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "oam_spectra_vs_cn2.png", dpi=300)
    plt.close(fig)


def plot_width_entropy_similarity(out_dir, cn2_list, alpha_list, width_mean, entropy_mean, similarity_mean) -> None:
    idist = 0
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for ia, alpha in enumerate(alpha_list):
        ax.semilogx(cn2_list[1:], width_mean[ia, idist, 1:], "o-", lw=1.1, ms=3, label=rf"$\alpha={alpha:.1f}$")
    style_axes(ax, r"$C_n^2$ (m$^{-2/3}$)", r"OAM spectral width", "OAM width vs turbulence strength")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "oam_width_vs_cn2.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ia, alpha in enumerate(alpha_list):
        axes[0].semilogx(cn2_list[1:], entropy_mean[ia, idist, 1:], "o-", lw=1.1, ms=3, label=rf"$\alpha={alpha:.1f}$")
        axes[1].semilogx(cn2_list[1:], similarity_mean[ia, idist, 1:], "o-", lw=1.1, ms=3, label=rf"$\alpha={alpha:.1f}$")
    style_axes(axes[0], r"$C_n^2$ (m$^{-2/3}$)", "Normalized entropy", "spectrum entropy")
    style_axes(axes[1], r"$C_n^2$ (m$^{-2/3}$)", "Cosine similarity", "similarity to no-turbulence reference")
    axes[0].legend(fontsize=8)
    fig.savefig(out_dir / "entropy_similarity_vs_cn2.png", dpi=300)
    plt.close(fig)


def plot_crosstalk(out_dir, l_list, alpha_list, spectra_total_mean) -> None:
    idist = 0
    ic = spectra_total_mean.shape[2] - 1  # 使用最大非零/最后一个 cn2 展示谱扩散矩阵。
    # 这里的 alpha -> ell 矩阵是谱扩散表示，不是严格通信系统串扰矩阵。
    matrix = build_alpha_to_ell_matrix(alpha_list, spectra_total_mean[:, idist, ic, :])
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", extent=[l_list[0] - 0.5, l_list[-1] + 0.5, -0.5, len(alpha_list) - 0.5])
    ax.set_yticks(np.arange(len(alpha_list)))
    ax.set_yticklabels([f"{a:.1f}" for a in alpha_list])
    ax.set_xlabel(r"Output OAM order $\ell$")
    ax.set_ylabel(r"Input fractional charge $\alpha$")
    ax.set_title(r"Spectrum-spreading matrix: input $\alpha$ $\rightarrow$ output $\ell$")
    fig.colorbar(im, ax=ax, label=r"$\mu_\ell$")
    fig.savefig(out_dir / "crosstalk_matrix_alpha_to_ell.png", dpi=300)
    plt.close(fig)


def plot_stokes(out_dir, cn2_list, alpha_list, stokes_mean) -> None:
    ia = 0
    idist = 0
    labels = [r"$S_1/S_0$", r"$S_2/S_0$", r"$S_3/S_0$"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for idx, label in enumerate(labels, start=1):
        ax.semilogx(cn2_list[1:], stokes_mean[ia, idist, 1:, idx], "o-", lw=1.1, ms=3, label=label)
    style_axes(ax, r"$C_n^2$ (m$^{-2/3}$)", "Normalized integrated Stokes", rf"Stokes summary, $\alpha={alpha_list[ia]:.1f}$")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "stokes_summary_vs_cn2.png", dpi=300)
    plt.close(fig)


def plot_phase_validation(out_dir, cn2_list, phase_rms_mean, params, cfg, dx, dy) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    # 只取第一个 alpha、第一段传播距离做相位屏 RMS 趋势诊断。
    axes[0].semilogx(cn2_list[1:], phase_rms_mean[0, 0, 1:], "o-", lw=1.1, ms=3)
    style_axes(axes[0], r"$C_n^2$ (m$^{-2/3}$)", "phase RMS (rad)", "phase-screen RMS")

    nonzero = [cn2 for cn2 in cn2_list if cn2 > 0]
    if nonzero:
        dz = cfg["distance_list"][0] / cfg["n_screens"]
        turb = TurbulenceParams(cn2=nonzero[-1], dz=dz, inner_scale=cfg["inner_scale"], outer_scale=cfg["outer_scale"], seed=cfg["seed"])
        screen = make_phase_screens(params.grid_n, dx, dy, params.wavelength, turb, 1)[0]
        k_radial, psd_radial = estimate_phase_psd_radial(screen, dx, dy)
        if k_radial.size:
            axes[1].loglog(k_radial, psd_radial, lw=1.1)
    style_axes(axes[1], r"$\kappa$ (rad/m)", "radial PSD (arb.)", "PSD trend check")
    fig.savefig(out_dir / "phase_screen_validation.png", dpi=300)
    plt.close(fig)


def plot_intensity_examples(out_dir, params, cfg, X, Y, R, PHI, dx, dy, alpha_list, cn2_list, distance_list) -> None:
    """Save representative total-intensity maps for no/weak/medium/strong turbulence."""
    alpha = float(alpha_list[0])
    distance = float(distance_list[0])
    cn2_examples = cn2_list[: min(4, len(cn2_list))]
    Ex0, Ey0 = initial_fvvb_field(alpha, R, PHI, params, cfg["initial_field"])
    intensity_maps = []
    titles = []
    for idx, cn2 in enumerate(cn2_examples):
        if cn2 > 0:
            dz = distance / max(cfg["n_screens"], 1)
            turb = TurbulenceParams(
                cn2=float(cn2),
                dz=dz,
                inner_scale=cfg["inner_scale"],
                outer_scale=cfg["outer_scale"],
                seed=int(cfg["seed"] + idx * 1000),
            )
            screens = make_phase_screens(params.grid_n, dx, dy, params.wavelength, turb, cfg["n_screens"])
        else:
            screens = []
        Ex_z, Ey_z = propagate_with_basis(Ex0, Ey0, dx, dy, params.wavelength, distance, screens, cfg["basis"], cfg["propagation"], cfg["pad_factor"])
        intensity_maps.append(np.abs(Ex_z) ** 2 + np.abs(Ey_z) ** 2)
        titles.append("no turbulence" if cn2 == 0 else rf"$C_n^2={cn2:.0e}$")

    vmax = max(float(np.max(I)) for I in intensity_maps)
    fig, axes = plt.subplots(1, len(intensity_maps), figsize=(3.2 * len(intensity_maps), 3.3), constrained_layout=True)
    if len(intensity_maps) == 1:
        axes = [axes]
    extent = [float(X.min() * 1e3), float(X.max() * 1e3), float(Y.min() * 1e3), float(Y.max() * 1e3)]
    for ax, intensity, title in zip(axes, intensity_maps, titles):
        im = ax.imshow(intensity / max(vmax, np.finfo(float).eps), origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.colorbar(im, ax=axes, label="normalized total intensity")
    fig.suptitle(rf"FVVB total intensity after turbulence, $\alpha={alpha:.1f}$, z={distance:g} m, {cfg['propagation']} pad={cfg['pad_factor']:g}")
    fig.savefig(out_dir / "intensity_examples_weak_mid_strong.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = scan_config(args)
    if cfg["n_screens"] < 0:
        raise ValueError("--n-screens must be non-negative")
    if cfg["n_mc"] < 1:
        raise ValueError("--mc must be at least 1")

    # 论文参数指示：default_params 中的这些数值由 cfg 控制；最终出图前应保存 cfg，并确认不是 --quick 配置。
    params = default_params(
        grid_n=cfg["grid_n"],
        rho_max_factor=cfg["rho_max_factor"],
        n_min=cfg["n_min"],
        n_max=cfg["n_max"],
        lmax=cfg["lmax"],
        nr_fourier=cfg["nr_fourier"],
        nphi_fourier=cfg["nphi_fourier"],
    )
    alpha_list = np.asarray(cfg["alpha_list"], dtype=float)       # [关键变量] alpha 扫描，无量纲。
    distance_list = np.asarray(cfg["distance_list"], dtype=float) # [关键变量] 传播距离扫描，单位 m。
    cn2_list = np.asarray(cfg["cn2_list"], dtype=float)           # [关键变量] C_n^2 湍流强度，单位 m^(-2/3)。
    l_list = params.l_list                                        # [关键变量] 输出 OAM 阶 ell。
    out_dir = ensure_output_dir(SCRIPT)
    if cfg["rho_max_mm"] is not None:
        X, Y, R, PHI, dx, dy = make_cartesian_grid(params.grid_n, float(cfg["rho_max_mm"]) * 1e-3)
    else:
        reference_radius = params.w0 if cfg["initial_field"] == "waist" else params.wz
        X, Y, R, PHI, dx, dy = make_cartesian_grid(params.grid_n, cfg["rho_max_factor"] * reference_radius)
    rho_max = float(np.max(np.abs(X)))

    # 主要 Monte Carlo 数组维度：alpha × distance × cn2 × mc × ell。
    shape_spec = (len(alpha_list), len(distance_list), len(cn2_list), cfg["n_mc"], len(l_list))
    spectra_mc = {name: np.zeros(shape_spec, dtype=float) for name in COMPONENTS}
    width_mc = np.zeros(shape_spec[:-1], dtype=float)
    entropy_mc = np.zeros(shape_spec[:-1], dtype=float)
    similarity_mc = np.zeros(shape_spec[:-1], dtype=float)
    l1_mc = np.zeros(shape_spec[:-1], dtype=float)
    energy_ratio_mc = np.zeros(shape_spec[:-1], dtype=float)
    stokes_mc = np.zeros(shape_spec[:-1] + (4,), dtype=float)
    stokes_delta_mc = np.zeros(shape_spec[:-1], dtype=float)
    phase_rms_mc = np.zeros(shape_spec[:-1], dtype=float)
    pcap_mc = {name: np.zeros(shape_spec[:-1], dtype=float) for name in COMPONENTS}
    edge_energy_fraction_mc = np.zeros(shape_spec[:-1], dtype=float)
    edge_mean_to_global_mean_mc = np.zeros(shape_spec[:-1], dtype=float)
    edge_mean_to_peak_mc = np.zeros(shape_spec[:-1], dtype=float)
    finite_fraction_mc = np.zeros(shape_spec[:-1], dtype=float)
    max_abs_field_mc = np.zeros(shape_spec[:-1], dtype=float)

    # cn2=0 的自由空间传播结果作为每个 alpha/distance 的参考谱与参考偏振态。
    reference_spectra = np.zeros((len(alpha_list), len(distance_list), len(l_list)), dtype=float)
    reference_stokes = np.zeros((len(alpha_list), len(distance_list), 4), dtype=float)
    reference_energy = np.zeros((len(alpha_list), len(distance_list)), dtype=float)
    rytov = np.zeros((len(distance_list), len(cn2_list)), dtype=float)
    fried = np.zeros((len(distance_list), len(cn2_list)), dtype=float)

    # 四重扫描结构：alpha -> distance -> cn2 -> Monte Carlo realization。
    for ia, alpha in enumerate(alpha_list):
        Ex0, Ey0 = initial_fvvb_field(float(alpha), R, PHI, params, cfg["initial_field"])
        assert_finite_array("initial Ex", Ex0)
        assert_finite_array("initial Ey", Ey0)
        input_energy = vector_energy(Ex0, Ey0, dx, dy)
        for idist, distance in enumerate(distance_list):
            Ex_ref, Ey_ref = propagate_with_basis(Ex0, Ey0, dx, dy, params.wavelength, float(distance), [], cfg["basis"], cfg["propagation"], cfg["pad_factor"])
            ref_oam = polarization_resolved_oam(Ex_ref, Ey_ref, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
            reference_spectra[ia, idist] = ref_oam["spectra"]["total"]
            ref_pol = polarization_summary(Ex_ref, Ey_ref, dx, dy)
            reference_stokes[ia, idist] = ref_pol["stokes"]
            reference_energy[ia, idist] = vector_energy(Ex_ref, Ey_ref, dx, dy)

            for ic, cn2 in enumerate(cn2_list):
                rytov[idist, ic] = rytov_variance(float(cn2), params.wavelength, float(distance))
                fried[idist, ic] = fried_parameter(float(cn2), params.wavelength, float(distance))
                for imc in range(cfg["n_mc"]):
                    dz = float(distance) / max(cfg["n_screens"], 1)
                    if cn2 > 0:
                        turb = TurbulenceParams(
                            cn2=float(cn2),
                            dz=dz,
                            inner_scale=cfg["inner_scale"],
                            outer_scale=cfg["outer_scale"],
                            # 不同 alpha/distance/cn2/mc 组合使用不同 seed，保证可复现且 realization 不重复。
                            seed=int(cfg["seed"] + ia * 100000 + idist * 10000 + ic * 1000 + imc * 100),
                        )
                        screens = make_phase_screens(params.grid_n, dx, dy, params.wavelength, turb, cfg["n_screens"])
                    else:
                        screens = []
                    Ex_z, Ey_z = propagate_with_basis(Ex0, Ey0, dx, dy, params.wavelength, float(distance), screens, cfg["basis"], cfg["propagation"], cfg["pad_factor"])
                    diag = vector_field_diagnostics(Ex_z, Ey_z, dx, dy, cfg["edge_fraction"])
                    edge_energy_fraction_mc[ia, idist, ic, imc] = diag["edge_energy_fraction"]
                    edge_mean_to_global_mean_mc[ia, idist, ic, imc] = diag["edge_mean_to_global_mean"]
                    edge_mean_to_peak_mc[ia, idist, ic, imc] = diag["edge_mean_to_peak"]
                    finite_fraction_mc[ia, idist, ic, imc] = diag["finite_fraction"]
                    max_abs_field_mc[ia, idist, ic, imc] = diag["max_abs_field"]
                    if cfg["fail_on_edge"] and diag["edge_energy_fraction"] > cfg["edge_warn_threshold"]:
                        raise RuntimeError(
                            "Edge energy fraction exceeds threshold: "
                            f"alpha={alpha}, distance={distance}, cn2={cn2}, mc={imc}, "
                            f"edge={diag['edge_energy_fraction']:.3e}, threshold={cfg['edge_warn_threshold']:.3e}"
                        )
                    oam = polarization_resolved_oam(Ex_z, Ey_z, X, Y, l_list, params.nr_fourier, params.nphi_fourier)
                    for name in COMPONENTS:
                        spectra_mc[name][ia, idist, ic, imc] = oam["spectra"][name]
                        pcap_mc[name][ia, idist, ic, imc] = oam["meta"][name]["Pcap"]
                    summary = compute_oam_summary(l_list, oam["spectra"]["total"], reference_spectra[ia, idist])
                    width_mc[ia, idist, ic, imc] = summary["width"]
                    entropy_mc[ia, idist, ic, imc] = summary["entropy"]
                    similarity_mc[ia, idist, ic, imc] = summary["similarity"]
                    l1_mc[ia, idist, ic, imc] = summary["l1_distance"]
                    # [关键变量] energy_ratio：传播后采样窗口内能量 / 输入窗口能量，用于数值稳定性检查。
                    energy_ratio_mc[ia, idist, ic, imc] = vector_energy(Ex_z, Ey_z, dx, dy) / max(input_energy, np.finfo(float).eps)
                    pol = polarization_summary(Ex_z, Ey_z, dx, dy, reference_stokes[ia, idist])
                    stokes_mc[ia, idist, ic, imc] = pol["normalized_stokes"]
                    stokes_delta_mc[ia, idist, ic, imc] = pol["stokes_l2_delta"]
                    if screens:
                        phase_rms_mc[ia, idist, ic, imc] = float(np.mean([phase_screen_stats(s)["rms"] for s in screens]))

    # 对 Monte Carlo 维度(axis=3)求均值/标准差，得到 alpha × distance × cn2 × ell 的统计结果。
    spectra_mean = {name: np.mean(arr, axis=3) for name, arr in spectra_mc.items()}
    spectra_std = {name: np.std(arr, axis=3) for name, arr in spectra_mc.items()}
    width_mean = np.mean(width_mc, axis=3)
    width_std = np.std(width_mc, axis=3)
    entropy_mean = np.mean(entropy_mc, axis=3)
    entropy_std = np.std(entropy_mc, axis=3)
    similarity_mean = np.mean(similarity_mc, axis=3)
    similarity_std = np.std(similarity_mc, axis=3)
    l1_mean = np.mean(l1_mc, axis=3)
    l1_std = np.std(l1_mc, axis=3)
    energy_ratio_mean = np.mean(energy_ratio_mc, axis=3)
    energy_ratio_std = np.std(energy_ratio_mc, axis=3)
    stokes_mean = np.mean(stokes_mc, axis=3)
    stokes_std = np.std(stokes_mc, axis=3)
    stokes_delta_mean = np.mean(stokes_delta_mc, axis=3)
    stokes_delta_std = np.std(stokes_delta_mc, axis=3)
    phase_rms_mean = np.mean(phase_rms_mc, axis=3)
    phase_rms_std = np.std(phase_rms_mc, axis=3)
    pcap_mean = {name: np.mean(arr, axis=3) for name, arr in pcap_mc.items()}
    pcap_std = {name: np.std(arr, axis=3) for name, arr in pcap_mc.items()}
    edge_energy_fraction_mean = np.mean(edge_energy_fraction_mc, axis=3)
    edge_energy_fraction_std = np.std(edge_energy_fraction_mc, axis=3)
    edge_mean_to_global_mean_mean = np.mean(edge_mean_to_global_mean_mc, axis=3)
    edge_mean_to_global_mean_std = np.std(edge_mean_to_global_mean_mc, axis=3)
    edge_mean_to_peak_mean = np.mean(edge_mean_to_peak_mc, axis=3)
    edge_mean_to_peak_std = np.std(edge_mean_to_peak_mc, axis=3)
    finite_fraction_mean = np.mean(finite_fraction_mc, axis=3)
    finite_fraction_std = np.std(finite_fraction_mc, axis=3)
    max_abs_field_mean = np.mean(max_abs_field_mc, axis=3)
    max_abs_field_std = np.std(max_abs_field_mc, axis=3)

    save_npz(
        out_dir,
        f"{SCRIPT}_summary",
        alpha_list=alpha_list,
        distance_list=distance_list,
        cn2_list=cn2_list,
        l_list=l_list,
        reference_spectra=reference_spectra,
        spectra_total_mean=spectra_mean["total"],
        spectra_total_std=spectra_std["total"],
        spectra_Ex_mean=spectra_mean["Ex"],
        spectra_Ex_std=spectra_std["Ex"],
        spectra_Ey_mean=spectra_mean["Ey"],
        spectra_Ey_std=spectra_std["Ey"],
        spectra_sigma_plus_mean=spectra_mean["sigma_plus"],
        spectra_sigma_plus_std=spectra_std["sigma_plus"],
        spectra_sigma_minus_mean=spectra_mean["sigma_minus"],
        spectra_sigma_minus_std=spectra_std["sigma_minus"],
        width_total_mean=width_mean,
        width_total_std=width_std,
        entropy_total_mean=entropy_mean,
        entropy_total_std=entropy_std,
        similarity_total_mean=similarity_mean,
        similarity_total_std=similarity_std,
        l1_distance_total_mean=l1_mean,
        l1_distance_total_std=l1_std,
        energy_ratio_mean=energy_ratio_mean,
        energy_ratio_std=energy_ratio_std,
        stokes_mean=stokes_mean,
        stokes_std=stokes_std,
        stokes_delta_mean=stokes_delta_mean,
        stokes_delta_std=stokes_delta_std,
        phase_rms_mean=phase_rms_mean,
        phase_rms_std=phase_rms_std,
        pcap_total_mean=pcap_mean["total"],
        pcap_total_std=pcap_std["total"],
        pcap_Ex_mean=pcap_mean["Ex"],
        pcap_Ex_std=pcap_std["Ex"],
        pcap_Ey_mean=pcap_mean["Ey"],
        pcap_Ey_std=pcap_std["Ey"],
        pcap_sigma_plus_mean=pcap_mean["sigma_plus"],
        pcap_sigma_plus_std=pcap_std["sigma_plus"],
        pcap_sigma_minus_mean=pcap_mean["sigma_minus"],
        pcap_sigma_minus_std=pcap_std["sigma_minus"],
        edge_energy_fraction_mean=edge_energy_fraction_mean,
        edge_energy_fraction_std=edge_energy_fraction_std,
        edge_mean_to_global_mean_mean=edge_mean_to_global_mean_mean,
        edge_mean_to_global_mean_std=edge_mean_to_global_mean_std,
        edge_mean_to_peak_mean=edge_mean_to_peak_mean,
        edge_mean_to_peak_std=edge_mean_to_peak_std,
        finite_fraction_mean=finite_fraction_mean,
        finite_fraction_std=finite_fraction_std,
        max_abs_field_mean=max_abs_field_mean,
        max_abs_field_std=max_abs_field_std,
        propagation_method=np.array(cfg["propagation"]),
        pad_factor=np.array(cfg["pad_factor"]),
        rho_max=np.array(rho_max),
        rytov_variance=rytov,
        fried_parameter=fried,
    )

    plot_oam_spectra(out_dir, l_list, cn2_list, spectra_mean["total"], alpha_list, distance_list)
    plot_width_entropy_similarity(out_dir, cn2_list, alpha_list, width_mean, entropy_mean, similarity_mean)
    plot_crosstalk(out_dir, l_list, alpha_list, spectra_mean["total"])
    plot_stokes(out_dir, cn2_list, alpha_list, stokes_mean)
    plot_phase_validation(out_dir, cn2_list, phase_rms_mean, params, cfg, dx, dy)
    plot_intensity_examples(out_dir, params, cfg, X, Y, R, PHI, dx, dy, alpha_list, cn2_list, distance_list)

    # cn2_list[0] = 0 表示无湍流参考；能量误差应接近 0，用于 sanity check。
    free_space_energy_error = np.abs(energy_ratio_mean[:, :, 0] - 1.0)
    initial_meta = initial_field_metadata(cfg, params)
    save_params(
        out_dir,
        SCRIPT,
        {
            "params": params,
            "config": cfg,
            **initial_meta,
            "dx": dx,
            "dy": dy,
            "rho_max": rho_max,
            "rho_max_mm": rho_max * 1e3,
            "numerical_method": {
                "propagation": cfg["propagation"],
                "pad_factor": cfg["pad_factor"],
                "padding_assumption": "Padded angular spectrum embeds the physical window in a larger zero background and crops back to the central window; it reduces FFT wrap-around but does not replace window-convergence checks.",
                "edge_fraction": cfg["edge_fraction"],
                "edge_warn_threshold": cfg["edge_warn_threshold"],
            },
            "turbulence_propagation_distance_list": distance_list,
            "assumption": ASSUMPTION,
            "stokes_convention": "S3 = -2 Im(Ex conj(Ey)), matching fvvb_core.linear_to_circular.",
            "crosstalk_definition": "Rows are input fractional alpha values; columns are output integer OAM orders; entries are Monte Carlo mean total OAM weights.",
            "oam_pcap_definition": "Pcap is the fraction of polar-grid power captured within the displayed/projected l_list; low Pcap indicates lmax truncation risk.",
            "edge_diagnostics_definition": "edge_energy_fraction is energy in the outer edge_fraction border divided by total sampled-window energy; high values indicate window/crop risk even if energy_ratio is near 1.",
            "free_space_max_energy_error": float(np.max(free_space_energy_error)),
            "max_edge_energy_fraction_mean": float(np.max(edge_energy_fraction_mean)),
            "min_pcap_total_mean": float(np.min(pcap_mean["total"])),
            "reference_energy": reference_energy,
            "rytov_variance": rytov,
            "fried_parameter": [[None if not np.isfinite(v) else float(v) for v in row] for row in fried],
        },
    )


if __name__ == "__main__":
    main()
