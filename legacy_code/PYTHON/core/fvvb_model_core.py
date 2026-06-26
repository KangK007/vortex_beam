"""FVVB 论文配图与湍流主线共用的数值核心函数。

本文件是整个 Python 仿真项目的物理核心，负责生成分数阶标量涡旋光束
(FVB)、分数阶矢量涡旋光束 (FVVB)、OAM 角向傅里叶谱、LG 径向模态谱
以及统一的输出保存工具。默认单位均为 SI：长度 m，相位 rad。

本文件公式从原 MATLAB 程序迁移而来；本次中文注释优化只解释变量含义、
单位和模型假设，不改变任何物理公式、归一化方式或默认参数。

保留的关键物理约定
------------------
- [关键变量] alpha：分数阶拓扑荷 / fractional topological charge，无量纲。
- [关键变量] c_n / coeff：Berry 展开系数，表示分数阶相位在整数 OAM 阶 n
  上的权重；整数 alpha 使用 Kronecker delta 极限避免 0/0。
- [关键变量] Ex/Ey：FVVB 在线偏振 Jones 基下的两个电场分量。
- [关键变量] E_sigma_plus / E_sigma_minus：圆偏振分量，定义为
  (Ex ± i Ey)/sqrt(2)，其符号约定需与 Stokes S3 保持一致。
- [关键变量] fork_ell 与 H(phi)：SPP/叉形相位板的等效调制荷与相位因子，
  当前实现为同一分析平面上直接乘以 exp(i * fork_ell * phi)。若论文写作
  涉及“中间面 SPP 后再传播”，需另行验证分段传播模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any, Iterable

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import eval_genlaguerre, gammaln, iv


@dataclass
class FVVBParams:
    """FVVB/FVB 的默认物理参数与数值截断参数。

    物理参数默认使用 SI 单位；数值参数控制网格大小、Berry 展开截断、OAM
    投影范围和 LG 径向投影范围。快速测试脚本会覆盖其中部分参数；论文级
    高精度结果需要记录对应的参数 JSON，避免与快速参数混用。

    论文参数指示：自由空间 OAM/LG 最终出图时，应优先做收敛性检查，而不是
    直接沿用快速脚本参数。建议论文级基准参数写清楚并保存到 *_params.json：
    - 强度图：grid_n=400, n_min=-200, n_max=200, rho_max_factor=1.5~3.0；
    - OAM 谱：grid_n=400, n_min=-200, n_max=200, lmax=40,
      nr_fourier=320, nphi_fourier=720；
    - LG 径向谱：grid_n=400, n_min=-200, n_max=200, lmax=40,
      pmax=35, w_an=params.wz；
    - 若计算资源允许，可用默认 grid_n=512 作最终收敛复核。
    """

    wavelength: float = 632.8e-9  # [关键变量] 波长 lambda，单位 m。
    w0: float = 2.0e-3            # [关键变量] 入射束腰半径，单位 m。
    z: float = 0.2                # [关键变量] 默认传播距离，单位 m；湍流脚本中 distance 会另行指定。
    n_min: int = -200             # [关键变量] Berry 展开整数阶 n 的下截断。
    n_max: int = 200              # [关键变量] Berry 展开整数阶 n 的上截断。
    grid_n: int = 512             # [关键变量] 笛卡尔网格每个方向采样点数。
    rho_max_factor: float = 8.0   # [关键变量] 计算窗口半径 = rho_max_factor * w(z)。
    lmax: int = 12                # [关键变量] OAM 投影阶数范围为 [-lmax, lmax]。
    pmax: int = 25                # [关键变量] LG 径向阶 p 的最大截断。
    nr_fourier: int = 320         # OAM 角向傅里叶投影的径向采样数。
    nphi_fourier: int = 720       # OAM 角向傅里叶投影的方位角采样数。

    @property
    def k(self) -> float:
        # [关键变量] 波数 k = 2*pi/lambda，单位 rad/m。
        return 2 * np.pi / self.wavelength

    @property
    def zR(self) -> float:
        # [关键变量] 瑞利长度 zR，单位 m。
        return rayleigh_range(self.wavelength, self.w0)

    @property
    def wz(self) -> float:
        # [关键变量] 传播到 z 平面后的高斯光束尺度 w(z)，单位 m。
        return self.w0 * np.sqrt(1 + (self.z / self.zR) ** 2)

    @property
    def n_list(self) -> np.ndarray:
        # [关键变量] Berry 展开使用的整数 OAM 阶 n 列表。
        return np.arange(self.n_min, self.n_max + 1)

    @property
    def l_list(self) -> np.ndarray:
        # [关键变量] OAM 谱显示/投影的整数阶 ell 列表。
        return np.arange(-self.lmax, self.lmax + 1)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update({"k": self.k, "zR": self.zR, "wz": self.wz})
        return data


def default_params(**overrides: Any) -> FVVBParams:
    params = FVVBParams()
    for key, value in overrides.items():
        if not hasattr(params, key):
            raise AttributeError(f"Unknown FVVB parameter: {key}")
        setattr(params, key, value)
    return params


def rayleigh_range(wavelength: float, w0: float) -> float:
    """计算瑞利长度 z_R = k w0^2 / 2。

    参数 wavelength 为波长，单位 m；w0 为束腰半径，单位 m。此处沿用原
    MATLAB 程序的高斯束约定。
    """
    k = 2 * np.pi / wavelength
    return k * w0**2 / 2


def make_cartesian_grid(n: int, rho_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """生成与 MATLAB ``meshgrid(linspace(...))`` 一致的笛卡尔网格。

    返回的 [关键变量] X/Y 为横向坐标网格，单位 m；R 为径向坐标，单位 m；
    PHI 为方位角，单位 rad；dx/dy 为采样间隔，单位 m。
    """
    x = np.linspace(-rho_max, rho_max, n)
    y = np.linspace(-rho_max, rho_max, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return X, Y, R, PHI, dx, dy


def berry_coeff(alpha: float, n_list: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """计算分数阶涡旋的 Berry 展开系数 c_n。

    [关键变量] alpha 为分数阶拓扑荷；[关键变量] n_list 为整数 OAM 展开阶。
    整数 alpha 处直接代入会出现 0/0，因此使用 Kronecker delta 极限。

    原英文说明：

    For non-integer ``alpha`` this returns the MATLAB coefficient
    ``exp(1i*pi*alpha)*sin(pi*alpha)/(pi*(alpha-n))``.  For integer alpha,
    direct evaluation is 0/0 at the matching order and 0 elsewhere; the
    physical limiting value is the Kronecker delta.
    """
    n_list = np.asarray(n_list, dtype=float)
    hit = np.abs(alpha - n_list) < tol
    if np.any(hit):
        coeff = np.zeros_like(n_list, dtype=np.complex128)
        coeff[hit] = 1.0 + 0.0j
        return coeff
    return np.exp(1j * np.pi * alpha) * np.sin(np.pi * alpha) / (np.pi * (alpha - n_list))


def _e_prime_single_order(rho_flat: np.ndarray, abs_n: int, z: float, k: float, w0: float, zR: float) -> np.ndarray:
    """计算单个 |n| 阶对应的径向传播场 E'_n(r,z)。

    该表达式直接迁移自 MATLAB ``compute_E_prime``，包含传播相位、贝塞尔
    函数项和阶数相关相因子，是 FVVB/FVB 构造中最核心的物理公式之一。
    """
    rho_flat = np.asarray(rho_flat, dtype=float)
    # 传播振幅与轴向相位因子；z、w0、zR 均为 m，k 为 rad/m。
    term_const = (
        zR**2 / (z - 1j * zR) ** 1.5
        * np.sqrt(np.pi / (4 * z * w0**2))
        * np.exp(1j * k * z)
    )
    # 贝塞尔函数自变量，与径向坐标 rho 和传播距离 z 有关。
    arg = (zR**2 * rho_flat**2 / w0**2) / (2 * z * (z - 1j * zR))
    # 径向二次相位项，决定传播后横向相位曲率。
    phase_term = np.exp(
        1j * k * rho_flat**2 / (2 * z) * (1 + (1j * zR) / (2 * (z - 1j * zR)))
    )
    # 阶数相关的复相因子，符号约定来自原 MATLAB 推导。
    pre_factor = (-1j) ** (abs_n + 1)
    return term_const * pre_factor * np.abs(rho_flat) * phase_term * (
        iv((abs_n - 1) / 2, arg) - iv((abs_n + 1) / 2, arg)
    )


def compute_e_prime(rho: np.ndarray, n_list: np.ndarray, z: float, k: float, w0: float, zR: float) -> np.ndarray:
    """Return the full E'_n matrix with shape ``rho.size x len(n_list)``.

    This mirrors MATLAB exactly but can be memory-heavy for large grids.  Figure
    scripts normally call ``fvb_field``/``fvvb_field``, which accumulate one order
    at a time and avoid storing this matrix.
    """
    rho_flat = np.asarray(rho, dtype=float).ravel()
    out = np.empty((rho_flat.size, len(n_list)), dtype=np.complex128)
    for j, n in enumerate(n_list):
        out[:, j] = _e_prime_single_order(rho_flat, int(abs(n)), z, k, w0, zR)
    return out


def fvb_field(alpha: float, rho: np.ndarray, phi: np.ndarray, params: FVVBParams, n_list: np.ndarray | None = None) -> np.ndarray:
    """构造标量分数阶涡旋光束 FVB 的复振幅 U。

    [关键变量] alpha 控制分数阶拓扑荷；n_list/coeff 是 Berry 展开阶与系数。
    返回 U 的形状与输入 rho/phi 网格一致，后续可用于标量 OAM 谱计算。
    """
    if n_list is None:
        n_list = params.n_list
    coeff = berry_coeff(alpha, n_list)
    rho_flat = rho.ravel()
    phi_flat = phi.ravel()
    U = np.zeros(rho_flat.shape, dtype=np.complex128)
    # 逐个整数阶 n 叠加：c_n * E'_n(r,z) * exp(i n phi)。
    for n, c in zip(n_list, coeff):
        if abs(c) == 0:
            continue
        ep = _e_prime_single_order(rho_flat, int(abs(n)), params.z, params.k, params.w0, params.zR)
        U += c * ep * np.exp(1j * n * phi_flat)
    return U.reshape(rho.shape)


def fvvb_field(
    alpha: float,
    rho: np.ndarray,
    phi: np.ndarray,
    params: FVVBParams,
    n_list: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构造 FVVB 在线偏振基下的 Jones 分量 Ex 和 Ey。

    这里保留 MATLAB 约定：Ex = sum c_n E'_n sin(n phi)，
    Ey = i sum c_n E'_n cos(n phi)。[关键变量] Ex/Ey 是后续湍流传播、
    圆偏振分解、Stokes 参数和偏振分辨 OAM 谱的基础。
    """
    if n_list is None:
        n_list = params.n_list
    coeff = berry_coeff(alpha, n_list)
    rho_flat = rho.ravel()
    phi_flat = phi.ravel()
    Ex = np.zeros(rho_flat.shape, dtype=np.complex128)
    Ey = np.zeros(rho_flat.shape, dtype=np.complex128)
    # 逐个整数阶 n 叠加矢量场。sin/cos 与 i 因子是 MATLAB/FVVB 约定。
    for n, c in zip(n_list, coeff):
        if abs(c) == 0:
            continue
        ep = _e_prime_single_order(rho_flat, int(abs(n)), params.z, params.k, params.w0, params.zR)
        angle = n * phi_flat
        Ex += c * ep * np.sin(angle)
        Ey += 1j * c * ep * np.cos(angle)
    return Ex.reshape(rho.shape), Ey.reshape(rho.shape)


def lg_p0_radial(rho: np.ndarray, ell_abs: int, w: float) -> np.ndarray:
    """束腰面 LG_{p=0}^{|ell|} 的径向包络。

    该函数只返回实径向因子，不包含 ``exp(i ell phi)``。它用于构造束腰面
    FVB/FVVB 初始场，避免直接令传播后解析公式 ``_e_prime_single_order`` 的
    ``z=0`` 而产生奇异。参数 ``w`` 为束腰半径，单位 m。
    """
    ell_abs = abs(int(ell_abs))
    rho = np.asarray(rho, dtype=float)
    log_c = 0.5 * (np.log(2) - np.log(np.pi) - gammaln(ell_abs + 1)) - np.log(w)
    C = np.exp(log_c)
    return C * (np.sqrt(2) * rho / w) ** ell_abs * np.exp(-(rho**2) / w**2)


def fvb_field_waist(alpha: float, rho: np.ndarray, phi: np.ndarray, params: FVVBParams, n_list: np.ndarray | None = None) -> np.ndarray:
    """构造束腰面标量 FVB 复振幅。

    与 ``fvb_field`` 不同，本函数不使用传播后 ``E'_n(r,z)`` 公式，而是在
    ``z=0`` 入射面用 LG ``p=0`` 径向包络和 Berry 角向展开构造初始场：
    ``U=sum c_n R_|n|(rho; w0) exp(i n phi)``。
    """
    if n_list is None:
        n_list = params.n_list
    coeff = berry_coeff(alpha, n_list)
    rho_flat = rho.ravel()
    phi_flat = phi.ravel()
    U = np.zeros(rho_flat.shape, dtype=np.complex128)
    for n, c in zip(n_list, coeff):
        if abs(c) == 0:
            continue
        radial = lg_p0_radial(rho_flat, int(abs(n)), params.w0)
        U += c * radial * np.exp(1j * n * phi_flat)
    return U.reshape(rho.shape)


def fvvb_field_waist(
    alpha: float,
    rho: np.ndarray,
    phi: np.ndarray,
    params: FVVBParams,
    n_list: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构造束腰面 FVVB 在线偏振基下的 Jones 分量。

    本函数用于“束腰面生成 FVVB 后传播 distance”的湍流模型。它保留现有
    FVVB 的角向约定 ``Ex=sum c_n R_|n| sin(n phi)``、``Ey=i sum c_n R_|n| cos(n phi)``，
    但将径向因子替换为束腰面 LG ``p=0`` 包络 ``R_|n|(rho; w0)``。
    """
    if n_list is None:
        n_list = params.n_list
    coeff = berry_coeff(alpha, n_list)
    rho_flat = rho.ravel()
    phi_flat = phi.ravel()
    Ex = np.zeros(rho_flat.shape, dtype=np.complex128)
    Ey = np.zeros(rho_flat.shape, dtype=np.complex128)
    for n, c in zip(n_list, coeff):
        if abs(c) == 0:
            continue
        radial = lg_p0_radial(rho_flat, int(abs(n)), params.w0)
        angle = n * phi_flat
        Ex += c * radial * np.sin(angle)
        Ey += 1j * c * radial * np.cos(angle)
    return Ex.reshape(rho.shape), Ey.reshape(rho.shape)


def linear_to_circular(Ex: np.ndarray, Ey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """由线偏振分量 Ex/Ey 转为圆偏振分量 sigma+/sigma-。

    [关键变量] Ep = E_sigma_plus = (Ex + i Ey)/sqrt(2)，
    Em = E_sigma_minus = (Ex - i Ey)/sqrt(2)。该符号约定会影响 Stokes S3。
    """
    return (Ex + 1j * Ey) / np.sqrt(2), (Ex - 1j * Ey) / np.sqrt(2)


def apply_spp(U: np.ndarray, phi: np.ndarray, fork_ell: float) -> np.ndarray:
    """对单个光场施加 SPP/叉形相位调制 H(phi)。

    [关键变量] fork_ell 是螺旋相位板或叉形光栅调制荷，常在图中记为 q。
    当前模型为同一平面直接乘 H(phi)=exp(i*fork_ell*phi)。
    """
    return U * np.exp(1j * fork_ell * phi)


def apply_spp_to_vector(Ex: np.ndarray, Ey: np.ndarray, phi: np.ndarray, fork_ell: float) -> tuple[np.ndarray, np.ndarray]:
    """对 FVVB 的 Ex/Ey 两个线偏振分量共同施加同一个 SPP 相位因子。"""
    H = np.exp(1j * fork_ell * phi)  # [关键变量] H(phi)，单位模复相位调制函数。
    return Ex * H, Ey * H


def _interp_to_polar(U: np.ndarray, X: np.ndarray, Y: np.ndarray, nr: int, nphi: int):
    """将笛卡尔网格上的场插值到极坐标网格，用于 OAM 角向傅里叶投影。"""
    xvec = X[0, :]
    yvec = Y[:, 0]
    rmax = min(float(np.max(np.abs(xvec))), float(np.max(np.abs(yvec))))
    r = np.linspace(0, rmax, nr)
    ph = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    RR, PP = np.meshgrid(r, ph, indexing="xy")
    XX = RR * np.cos(PP)
    YY = RR * np.sin(PP)
    interp = RegularGridInterpolator((yvec, xvec), U, method="linear", bounds_error=False, fill_value=0.0)
    pts = np.column_stack([YY.ravel(), XX.ravel()])
    U_polar = interp(pts).reshape(RR.shape)
    return U_polar, RR, PP, r, ph


def oam_spectrum_angular_fourier(
    U: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    l_list: Iterable[int],
    nr: int = 320,
    nphi: int = 720,
) -> tuple[np.ndarray, float]:
    """OAM spectrum using the angular Fourier projection from MATLAB.

    Returns ``El`` and ``Etot``.  ``El/Etot`` is the energy fraction captured by
    the requested OAM range; normalise by ``sum(El)`` when a conditional spectrum
    over only the displayed l-range is desired.
    """
    U_polar, RR, PP, r, ph = _interp_to_polar(U, X, Y, nr, nphi)
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
    dphi = float(ph[1] - ph[0]) if len(ph) > 1 else 2 * np.pi
    # [关键变量] Etot：极坐标窗口内总能量积分，含 Jacobian 权重 r。
    etot = float(np.sum(np.abs(U_polar) ** 2 * RR) * dr * dphi)
    el = []
    for ell in l_list:
        # [关键变量] ell：整数 OAM 阶；a_l(r) 是该阶的角向傅里叶系数。
        phase = np.exp(-1j * ell * PP)
        a_l = (1 / (2 * np.pi)) * np.sum(U_polar * phase, axis=0) * dphi
        # [关键变量] El：第 ell 阶 OAM 投影能量。
        el.append(2 * np.pi * np.sum(np.abs(a_l) ** 2 * r) * dr)
    return np.asarray(el, dtype=float), etot


def total_oam_spectrum_from_vector(
    Ex: np.ndarray,
    Ey: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    l_list: Iterable[int],
    nr: int = 320,
    nphi: int = 720,
    conditional: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Total OAM spectrum from the two circular components of a vector field."""
    Ep, Em = linear_to_circular(Ex, Ey)
    El_p, Et_p = oam_spectrum_angular_fourier(Ep, X, Y, l_list, nr, nphi)
    El_m, Et_m = oam_spectrum_angular_fourier(Em, X, Y, l_list, nr, nphi)
    # 合并两个圆偏振分量的 OAM 能量，得到矢量场总 OAM 谱。
    El_total = El_p + El_m
    # [关键变量] mu：归一化 OAM 权重。conditional=True 时只在显示的 l_list 范围内归一化。
    denom = np.sum(El_total) if conditional else (Et_p + Et_m)
    mu = El_total / max(float(denom), np.finfo(float).eps)
    # [关键变量] Pcap：当前 l_list 截断范围捕获的总能量比例。
    meta = {"Et_plus": Et_p, "Et_minus": Et_m, "Pcap": float(np.sum(El_total) / max(Et_p + Et_m, np.finfo(float).eps))}
    return mu, meta


def spectral_width(x: Iterable[float], weights: Iterable[float]) -> tuple[float, float]:
    """Return weighted mean and standard deviation."""
    x = np.asarray(list(x), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    wn = w / s
    mean = float(np.sum(x * wn))
    sigma = float(np.sqrt(np.sum((x - mean) ** 2 * wn)))
    return mean, sigma


def lg_mode_xy(R: np.ndarray, PHI: np.ndarray, p: int, ell: int, w: float) -> np.ndarray:
    """Laguerre-Gaussian mode matching the MATLAB ``LG_mode_xy`` function."""
    abs_l = abs(int(ell))
    log_c = 0.5 * (np.log(2) + gammaln(p + 1) - np.log(np.pi) - gammaln(p + abs_l + 1)) - np.log(w)
    C = np.exp(log_c)
    rho2 = 2 * R**2 / w**2
    Lpa = eval_genlaguerre(p, abs_l, rho2)
    return C * (np.sqrt(2) * R / w) ** abs_l * Lpa * np.exp(-(R**2) / w**2) * np.exp(1j * ell * PHI)


def radial_lg_spectrum(
    U: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    w_an: float,
    l_list: Iterable[int],
    pmax: int,
) -> tuple[np.ndarray, float]:
    """将光场投影到 LG_p^ell 模式，返回径向-角向能量矩阵。

    [关键变量] w_an 是 LG 分析基腰，单位 m；pmax 是径向阶截断；l_list 是
    方位 OAM 阶范围。返回 Mpl[p, j]，其中行对应径向阶 p，列对应 l_list[j]。
    """
    l_arr = np.asarray(list(l_list), dtype=int)
    Mpl = np.zeros((pmax + 1, len(l_arr)), dtype=float)
    etot = float(np.sum(np.abs(U) ** 2) * dx * dy)
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    for j, ell in enumerate(l_arr):
        for p in range(pmax + 1):
            # [关键变量] Mpl[p, j]：LG 径向阶 p、方位阶 ell 的投影能量。
            LG = lg_mode_xy(R, PHI, p, int(ell), w_an)
            c = np.sum(np.conj(LG) * U) * dx * dy
            Mpl[p, j] = float(np.abs(c) ** 2)
    return Mpl, etot


def radial_width_from_mpl(Mpl: np.ndarray, etot: float, conditional: bool = True) -> tuple[np.ndarray, float, float, float]:
    """由 LG 投影矩阵计算径向谱 nu_p、平均径向阶、sigma_p 和捕获功率。"""
    # [关键变量] nu_p：对所有 ell 求和后的径向阶 p 能量分布。
    nu_p = np.sum(Mpl, axis=1) / max(etot, np.finfo(float).eps)
    pcap = float(np.sum(nu_p))  # [关键变量] pcap/Pcap：LG 截断范围捕获的功率比例。
    weights = nu_p / max(pcap, np.finfo(float).eps) if conditional else nu_p
    p_list = np.arange(len(nu_p))
    mean, sigma = spectral_width(p_list, weights)
    return nu_p, mean, sigma, pcap


def ensure_output_dir(script_name: str) -> Path:
    """返回脚本专属输出目录 ``PYTHON/outputs/<script_name>/``。

    注意：各图脚本会使用固定文件名保存 PNG/NPZ/JSON，重复运行会覆盖已有仿真
    结果。若这些结果已作为论文图源，应先备份或改用独立输出目录。
    """
    out = Path(__file__).resolve().parent / "outputs" / script_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonify(value.to_dict())
    return value


def save_params(out_dir: Path, name: str, params: dict[str, Any]) -> None:
    """保存参数 JSON，用于论文图和仿真结果复现；同名文件会被覆盖。"""
    with (out_dir / f"{name}_params.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonify(params), f, ensure_ascii=False, indent=2)


def save_npz(out_dir: Path, name: str, **arrays: Any) -> None:
    """保存数值数组 NPZ；同名文件会被覆盖，运行前需确认不是最终论文数据。"""
    np.savez(out_dir / f"{name}.npz", **arrays)


def style_axes(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)


def get_grid_from_params(params: FVVBParams, rho_max_factor: float | None = None):
    factor = params.rho_max_factor if rho_max_factor is None else rho_max_factor
    return make_cartesian_grid(params.grid_n, factor * params.wz)
