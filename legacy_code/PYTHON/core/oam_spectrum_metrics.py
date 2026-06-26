"""FVVB 湍流传播分析中的 OAM 谱统计指标。

本文件复用 ``fvvb_core`` 中的角向傅里叶 OAM 投影，进一步给出毕业设计
主线 B 需要的谱宽、谱熵、与无湍流参考谱相似度、偏振分辨 OAM 谱以及
alpha -> ell 谱扩散矩阵。所有谱默认解释为显示 OAM 阶范围内的归一化
能量权重；用于论文时需同时报告 l_list/lmax 与 Pcap。"""

from __future__ import annotations

from typing import Any

import numpy as np

from fvvb_core import (
    linear_to_circular,
    oam_spectrum_angular_fourier,
    spectral_width,
    total_oam_spectrum_from_vector,
)


def normalize_spectrum(power: np.ndarray, eps: float | None = None) -> np.ndarray:
    """将非负谱权重归一化为概率分布。

    [关键变量] power 通常为各 OAM 阶能量 El 或其组合；返回值可记为 mu_l。
    若总和接近 0，则用 eps 避免除零。
    """
    if eps is None:
        eps = np.finfo(float).eps
    power = np.asarray(power, dtype=float)
    return power / max(float(np.sum(power)), eps)


def spectrum_entropy(mu: np.ndarray, normalized: bool = True) -> float:
    """计算 OAM 谱的 Shannon 熵。

    [关键变量] mu 是归一化 OAM 权重。normalized=True 时除以 log(N)，使显示
    OAM 范围内的均匀谱接近 1；熵越大表示谱越分散。

    原英文说明：

    ``mu`` is normalized internally.  When ``normalized`` is true, the entropy is
    divided by ``log(N)`` so that a flat spectrum over the displayed OAM range is
    close to 1.
    """
    mu = normalize_spectrum(mu)
    use = mu > 0
    if not np.any(use):
        return float("nan")
    entropy = float(-np.sum(mu[use] * np.log(mu[use])))
    if normalized and mu.size > 1:
        entropy /= float(np.log(mu.size))
    return entropy


def spectrum_similarity(mu: np.ndarray, ref_mu: np.ndarray | None) -> dict[str, float]:
    """比较当前谱与参考谱的相似度。

    cosine 越接近 1，表示与无湍流参考谱越相似；L1 距离越大，表示 OAM 谱
    能量重新分布越明显。
    """
    if ref_mu is None:
        return {"cosine": float("nan"), "l1": float("nan")}
    mu = normalize_spectrum(mu)
    ref_mu = normalize_spectrum(ref_mu)
    denom = float(np.linalg.norm(mu) * np.linalg.norm(ref_mu))
    cosine = float(np.dot(mu, ref_mu) / max(denom, np.finfo(float).eps))
    l1 = float(np.sum(np.abs(mu - ref_mu)))
    return {"cosine": cosine, "l1": l1}


def compute_oam_summary(l_list: np.ndarray, mu: np.ndarray, ref_mu: np.ndarray | None = None) -> dict[str, float]:
    """汇总单个 OAM 谱的平均阶、谱宽、熵、峰值和相似度指标。

    [关键变量] l_list 是整数 OAM 阶 ell；width 即 OAM 谱宽 sigma_L。
    """
    mu = normalize_spectrum(mu)
    l_arr = np.asarray(l_list, dtype=float)
    mean_l, width = spectral_width(l_arr, mu)
    sim = spectrum_similarity(mu, ref_mu)
    peak_idx = int(np.argmax(mu)) if mu.size else -1
    return {
        "mean_l": float(mean_l),
        "width": float(width),
        "entropy": spectrum_entropy(mu, normalized=True),
        "peak_l": float(l_arr[peak_idx]) if peak_idx >= 0 else float("nan"),
        "peak_fraction": float(mu[peak_idx]) if peak_idx >= 0 else float("nan"),
        "similarity": sim["cosine"],
        "l1_distance": sim["l1"],
    }


def _component_oam_spectrum(
    U: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    l_list: np.ndarray,
    nr: int,
    nphi: int,
) -> tuple[np.ndarray, dict[str, float]]:
    El, Etot = oam_spectrum_angular_fourier(U, X, Y, l_list, nr, nphi)
    mu = normalize_spectrum(El)
    pcap = float(np.sum(El) / max(float(Etot), np.finfo(float).eps))
    return mu, {"Etot": float(Etot), "Pcap": pcap}


def polarization_resolved_oam(
    Ex: np.ndarray,
    Ey: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    l_list: np.ndarray,
    nr: int,
    nphi: int,
) -> dict[str, Any]:
    """计算总场和偏振分辨 OAM 谱。

    [关键变量] total 是两个圆偏振分量合并后的总 OAM 谱；Ex/Ey 是线偏振
    分量谱；sigma_plus/sigma_minus 是圆偏振分量谱。各分量谱默认在显示的
    l_list 范围内自归一化，比较定量能量时需同时查看 meta 中的 Etot/Pcap。

    原英文说明：

    The component spectra are self-normalized over the displayed ``l_list``.  The
    total spectrum follows ``total_oam_spectrum_from_vector`` and combines the two
    circular components.
    """
    # [关键变量] Ep/Em 分别为 sigma+ / sigma- 圆偏振分量。
    Ep, Em = linear_to_circular(Ex, Ey)
    mu_total, meta_total = total_oam_spectrum_from_vector(Ex, Ey, X, Y, l_list, nr, nphi)
    mu_ex, meta_ex = _component_oam_spectrum(Ex, X, Y, l_list, nr, nphi)
    mu_ey, meta_ey = _component_oam_spectrum(Ey, X, Y, l_list, nr, nphi)
    mu_sp, meta_sp = _component_oam_spectrum(Ep, X, Y, l_list, nr, nphi)
    mu_sm, meta_sm = _component_oam_spectrum(Em, X, Y, l_list, nr, nphi)
    return {
        "spectra": {
            "total": mu_total,
            "Ex": mu_ex,
            "Ey": mu_ey,
            "sigma_plus": mu_sp,
            "sigma_minus": mu_sm,
        },
        "meta": {
            "total": meta_total,
            "Ex": meta_ex,
            "Ey": meta_ey,
            "sigma_plus": meta_sp,
            "sigma_minus": meta_sm,
        },
    }


def build_alpha_to_ell_matrix(alpha_list: np.ndarray, spectra_mean: np.ndarray) -> np.ndarray:
    """生成 alpha -> output ell 的 OAM 谱扩散矩阵。

    该矩阵用于描述分数阶输入 alpha 在整数 OAM 阶 ell 上的能量扩散趋势；
    它不是严格通信系统中“发送整数模式 -> 接收整数模式”的串扰矩阵。

    原英文说明：

    ``spectra_mean`` is expected to have shape ``(n_alpha, n_ell)``.  Rows are
    normalized again for robustness.  For fractional FVVBs this matrix is a
    spectrum-spreading representation, not the strict integer-mode crosstalk
    matrix used in OAM communication.
    """
    matrix = np.asarray(spectra_mean, dtype=float)
    if matrix.shape[0] != len(alpha_list):
        raise ValueError("spectra_mean first dimension must match alpha_list")
    return np.vstack([normalize_spectrum(row) for row in matrix])
