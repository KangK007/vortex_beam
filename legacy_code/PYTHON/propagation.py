"""FVVB 湍流传播使用的角谱法传播工具。

本文件只处理均匀采样的二维笛卡尔网格，与 ``fvvb_core.make_cartesian_grid``
生成的 X/Y 网格一致。默认单位均为 SI：dx/dy 和 distance 为 m，wavelength
为 m，相位为 rad。

当前传播模型用于毕业设计“主线 B：大气湍流传播与 OAM 谱演化”。相位屏
被作为公共标量相位同时作用于两个偏振分量；不包含偏振相关散射、双折射
或真实退偏振。本次注释优化不改变 FFT 符号、传播核或归一化约定。"""

from __future__ import annotations

from collections.abc import Sequence
import math

import numpy as np

from fvvb_core import linear_to_circular


def assert_finite_array(name: str, arr: np.ndarray) -> None:
    """检查数组是否全部为有限值。

    论文级传播中应尽早暴露 ``nan``/``inf``，否则后续 OAM 谱和 Stokes 指标
    会被污染。大窗口 FVVB 初始场若触发本检查，常见原因是
    ``fvvb_core._e_prime_single_order`` 中 Bessel ``iv`` 在大半径处数值溢出。
    """
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if np.all(finite):
        return
    nonfinite = int(arr.size - np.count_nonzero(finite))
    finite_abs = np.abs(arr[finite]) if np.any(finite) else np.array([], dtype=float)
    max_abs = float(np.max(finite_abs)) if finite_abs.size else float("nan")
    raise FloatingPointError(
        f"Non-finite values detected in {name}: "
        f"shape={arr.shape}, nonfinite={nonfinite}/{arr.size}, finite_max_abs={max_abs:.3e}. "
        "If this occurs immediately after FVVB field construction, reduce the transverse "
        "window or handle Bessel iv overflow in fvvb_core._e_prime_single_order separately."
    )


def _center_pad_2d(U: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """将二维数组居中零填充到 ``out_shape``。"""
    if U.ndim != 2:
        raise ValueError("Only 2-D arrays can be center-padded")
    ny, nx = U.shape
    out_ny, out_nx = out_shape
    if out_ny < ny or out_nx < nx:
        raise ValueError("out_shape must be no smaller than input shape")
    out = np.zeros((out_ny, out_nx), dtype=U.dtype)
    y0 = (out_ny - ny) // 2
    x0 = (out_nx - nx) // 2
    out[y0 : y0 + ny, x0 : x0 + nx] = U
    return out


def _center_crop_2d(U: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """从二维数组中心裁剪出 ``out_shape``。"""
    if U.ndim != 2:
        raise ValueError("Only 2-D arrays can be center-cropped")
    ny, nx = U.shape
    out_ny, out_nx = out_shape
    if out_ny > ny or out_nx > nx:
        raise ValueError("out_shape must be no larger than input shape")
    y0 = (ny - out_ny) // 2
    x0 = (nx - out_nx) // 2
    return U[y0 : y0 + out_ny, x0 : x0 + out_nx]


def _padded_shape(shape: tuple[int, int], pad_factor: float) -> tuple[int, int]:
    """根据 pad factor 生成不小于原尺寸的 FFT 网格尺寸。"""
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    ny, nx = shape
    out_ny = max(ny, int(math.ceil(ny * pad_factor)))
    out_nx = max(nx, int(math.ceil(nx * pad_factor)))
    return out_ny, out_nx


def _edge_mask(shape: tuple[int, int], edge_fraction: float) -> np.ndarray:
    """生成边缘带 mask，用于窗口/回卷风险诊断。"""
    if not (0 < edge_fraction < 0.5):
        raise ValueError("edge_fraction must be between 0 and 0.5")
    ny, nx = shape
    border = max(1, int(round(edge_fraction * min(ny, nx))))
    mask = np.zeros((ny, nx), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask


def field_diagnostics(U: np.ndarray, dx: float, dy: float, edge_fraction: float = 0.05) -> dict[str, float]:
    """返回单个标量场的能量、边缘能量和有限值诊断。"""
    U = np.asarray(U)
    finite = np.isfinite(U)
    nonfinite_count = int(U.size - np.count_nonzero(finite))
    U_safe = np.nan_to_num(U, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = np.abs(U_safe) ** 2
    energy = float(np.sum(intensity) * dx * dy)
    edge = _edge_mask(U.shape, edge_fraction)
    edge_energy = float(np.sum(intensity[edge]) * dx * dy)
    edge_mean = float(np.mean(intensity[edge])) if np.any(edge) else float("nan")
    global_mean = float(np.mean(intensity))
    peak = float(np.max(intensity)) if intensity.size else float("nan")
    finite_abs = np.abs(U[finite]) if np.any(finite) else np.array([], dtype=float)
    max_abs = float(np.max(finite_abs)) if finite_abs.size else float("nan")
    eps = np.finfo(float).eps
    return {
        "energy": energy,
        "edge_energy": edge_energy,
        "edge_energy_fraction": edge_energy / max(energy, eps),
        "edge_mean_intensity": edge_mean,
        "global_mean_intensity": global_mean,
        "peak_intensity": peak,
        "edge_mean_to_global_mean": edge_mean / max(global_mean, eps),
        "edge_mean_to_peak": edge_mean / max(peak, eps),
        "nonfinite_count": float(nonfinite_count),
        "finite_fraction": float(np.count_nonzero(finite) / U.size),
        "max_abs": max_abs,
    }


def vector_field_diagnostics(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    edge_fraction: float = 0.05,
) -> dict[str, float]:
    """返回二分量 Jones 场的窗口边界和有限值诊断。"""
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    if Ex.shape != Ey.shape:
        raise ValueError("Ex and Ey must have the same shape")
    finite_x = np.isfinite(Ex)
    finite_y = np.isfinite(Ey)
    finite = finite_x & finite_y
    nonfinite_count = int(Ex.size * 2 - np.count_nonzero(finite_x) - np.count_nonzero(finite_y))
    Ex_safe = np.nan_to_num(Ex, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    Ey_safe = np.nan_to_num(Ey, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = np.abs(Ex_safe) ** 2 + np.abs(Ey_safe) ** 2
    energy = float(np.sum(intensity) * dx * dy)
    edge = _edge_mask(Ex.shape, edge_fraction)
    edge_energy = float(np.sum(intensity[edge]) * dx * dy)
    edge_mean = float(np.mean(intensity[edge])) if np.any(edge) else float("nan")
    global_mean = float(np.mean(intensity))
    peak = float(np.max(intensity)) if intensity.size else float("nan")
    max_abs_ex = float(np.max(np.abs(Ex[finite_x]))) if np.any(finite_x) else float("nan")
    max_abs_ey = float(np.max(np.abs(Ey[finite_y]))) if np.any(finite_y) else float("nan")
    eps = np.finfo(float).eps
    return {
        "energy": energy,
        "edge_energy": edge_energy,
        "edge_energy_fraction": edge_energy / max(energy, eps),
        "edge_mean_intensity": edge_mean,
        "global_mean_intensity": global_mean,
        "peak_intensity": peak,
        "edge_mean_to_global_mean": edge_mean / max(global_mean, eps),
        "edge_mean_to_peak": edge_mean / max(peak, eps),
        "nonfinite_count": float(nonfinite_count),
        "finite_fraction": float((np.count_nonzero(finite_x) + np.count_nonzero(finite_y)) / (2 * Ex.size)),
        "max_abs_Ex": max_abs_ex,
        "max_abs_Ey": max_abs_ey,
        "max_abs_field": max(max_abs_ex, max_abs_ey),
    }


def angular_spectrum_propagate(U: np.ndarray, dx: float, dy: float, wavelength: float, distance: float) -> np.ndarray:
    """用角谱法传播单个标量复光场。

    Parameters
    ----------
    U:
        Complex optical field on a Cartesian grid.
    dx, dy:
        Sampling intervals in x and y, in metres.
    wavelength:
        Wavelength in metres.
    distance:
        Propagation distance in metres.  Use zero to return a copy.

    Returns
    -------
    np.ndarray
        Propagated complex field on the same grid.
    """
    if distance == 0:
        return np.array(U, copy=True)

    ny, nx = U.shape
    k = 2 * np.pi / wavelength  # [关键变量] 波数，单位 rad/m。
    # fx/fy 为空间频率，单位 cycles/m；kx/ky 为角空间频率，单位 rad/m。
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz_sq = (k**2 - kx**2 - ky**2).astype(np.complex128)
    # H 是自由空间角谱传播传递函数；复数 sqrt 保留倏逝波形式。
    H = np.exp(1j * np.sqrt(kz_sq) * distance)
    return np.fft.ifft2(np.fft.fft2(U) * H)


def angular_spectrum_propagate_padded(
    U: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    pad_factor: float = 2.0,
    check_finite: bool = True,
) -> np.ndarray:
    """零填充后角谱传播，再裁剪回原中心窗口。

    该函数只改变 FFT 数值边界处理：把中心物理窗口嵌入更大的零背景，传播后
    裁剪回原窗口，以减弱同尺寸 FFT 的周期回卷伪影。它不改变自由空间传播核。
    """
    if check_finite:
        assert_finite_array("angular_spectrum input", U)
    if distance == 0 or pad_factor == 1:
        out = angular_spectrum_propagate(U, dx, dy, wavelength, distance)
    else:
        padded = _center_pad_2d(U, _padded_shape(U.shape, pad_factor))
        propagated = angular_spectrum_propagate(padded, dx, dy, wavelength, distance)
        out = _center_crop_2d(propagated, U.shape)
    if check_finite:
        assert_finite_array("angular_spectrum output", out)
    return out


def angular_spectrum_propagate_select(
    U: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    method: str = "same",
    pad_factor: float = 1.0,
    check_finite: bool = True,
) -> np.ndarray:
    """按指定数值边界处理方式执行角谱传播。"""
    if check_finite:
        assert_finite_array("angular_spectrum input", U)
    if method == "same":
        out = angular_spectrum_propagate(U, dx, dy, wavelength, distance)
    elif method == "padded":
        out = angular_spectrum_propagate_padded(U, dx, dy, wavelength, distance, pad_factor, check_finite=False)
    else:
        raise ValueError(f"Unknown propagation method: {method}")
    if check_finite:
        assert_finite_array("angular_spectrum output", out)
    return out


def propagate_linear_components(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    propagation: str = "same",
    pad_factor: float = 1.0,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate the two linear Jones components ``Ex`` and ``Ey`` independently."""
    return (
        angular_spectrum_propagate_select(Ex, dx, dy, wavelength, distance, propagation, pad_factor, check_finite),
        angular_spectrum_propagate_select(Ey, dx, dy, wavelength, distance, propagation, pad_factor, check_finite),
    )


def propagate_circular_components(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    propagation: str = "same",
    pad_factor: float = 1.0,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``Ex/Ey`` to circular components and propagate them independently."""
    Ep, Em = linear_to_circular(Ex, Ey)
    return (
        angular_spectrum_propagate_select(Ep, dx, dy, wavelength, distance, propagation, pad_factor, check_finite),
        angular_spectrum_propagate_select(Em, dx, dy, wavelength, distance, propagation, pad_factor, check_finite),
    )


def apply_common_phase(U1: np.ndarray, U2: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对任意两个场分量施加同一个标量相位屏。

    [关键变量] phase 为湍流相位屏，单位 rad；H=exp(i*phase) 是单位模相位
    调制。该模型假设两个偏振分量经历完全相同的相位扰动。
    """
    H = np.exp(1j * phase)
    return U1 * H, U2 * H


def apply_common_phase_to_linear(Ex: np.ndarray, Ey: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same scalar phase screen to both linear polarization components."""
    return apply_common_phase(Ex, Ey, phase)


def circular_to_linear(Ep: np.ndarray, Em: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将圆偏振分量 Ep/Em 转回 Ex/Ey。

    请勿单独改变这里的符号；它必须与 ``fvvb_core.linear_to_circular`` 和
    ``polarization_metrics`` 中的 Stokes S3 约定同时保持一致。
    """
    Ex = (Ep + Em) / np.sqrt(2)
    Ey = -1j * (Ep - Em) / np.sqrt(2)
    return Ex, Ey


def field_energy(U: np.ndarray, dx: float, dy: float) -> float:
    """Return scalar-field energy ``sum(|U|^2) dx dy`` in the sampled window."""
    return float(np.sum(np.abs(U) ** 2) * dx * dy)


def vector_energy(Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: float) -> float:
    """Return two-component field energy in the sampled window."""
    return field_energy(Ex, dx, dy) + field_energy(Ey, dx, dy)


def _step_distance(distance: float, phase_screens: Sequence[np.ndarray]) -> float:
    """计算 split-step 中每个相位屏后的传播步长 dz。"""
    if len(phase_screens) == 0:
        return distance
    # [关键变量] dz = 总传播距离 / 相位屏数量。
    return distance / len(phase_screens)


def propagate_vector_split_step_linear(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    phase_screens: Sequence[np.ndarray],
    propagation: str = "same",
    pad_factor: float = 1.0,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate ``Ex/Ey`` through common scalar phase screens.

    Each phase screen is applied to both linear components, followed by angular-
    spectrum propagation over ``distance / len(phase_screens)``.  If
    ``phase_screens`` is empty, this reduces to free-space propagation over the
    full distance.
    """
    if check_finite:
        assert_finite_array("Ex input", Ex)
        assert_finite_array("Ey input", Ey)
    Ex_z = np.array(Ex, copy=True)
    Ey_z = np.array(Ey, copy=True)
    dz = _step_distance(distance, phase_screens)
    if len(phase_screens) == 0:
        return propagate_linear_components(Ex_z, Ey_z, dx, dy, wavelength, distance, propagation, pad_factor, check_finite)
    for phase in phase_screens:
        if check_finite:
            assert_finite_array("phase screen", phase)
        # 分步顺序：先通过相位屏，再自由传播 dz；两个线偏振分量使用同一 phase。
        Ex_z, Ey_z = apply_common_phase_to_linear(Ex_z, Ey_z, phase)
        Ex_z, Ey_z = propagate_linear_components(Ex_z, Ey_z, dx, dy, wavelength, dz, propagation, pad_factor, check_finite)
    return Ex_z, Ey_z


def propagate_vector_split_step_circular(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
    phase_screens: Sequence[np.ndarray],
    propagation: str = "same",
    pad_factor: float = 1.0,
    check_finite: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate in the circular basis through common scalar phase screens.

    The input ``Ex/Ey`` is converted to ``sigma+ / sigma-``.  The same scalar
    phase screen is applied to both circular components at each split step, and
    the output is converted back to ``Ex/Ey``.
    """
    if check_finite:
        assert_finite_array("Ex input", Ex)
        assert_finite_array("Ey input", Ey)
    Ep, Em = linear_to_circular(Ex, Ey)
    dz = _step_distance(distance, phase_screens)
    if len(phase_screens) == 0:
        Ep_z = angular_spectrum_propagate_select(Ep, dx, dy, wavelength, distance, propagation, pad_factor, check_finite)
        Em_z = angular_spectrum_propagate_select(Em, dx, dy, wavelength, distance, propagation, pad_factor, check_finite)
        return circular_to_linear(Ep_z, Em_z)
    for phase in phase_screens:
        if check_finite:
            assert_finite_array("phase screen", phase)
        Ep, Em = apply_common_phase(Ep, Em, phase)
        Ep = angular_spectrum_propagate_select(Ep, dx, dy, wavelength, dz, propagation, pad_factor, check_finite)
        Em = angular_spectrum_propagate_select(Em, dx, dy, wavelength, dz, propagation, pad_factor, check_finite)
    return circular_to_linear(Ep, Em)
