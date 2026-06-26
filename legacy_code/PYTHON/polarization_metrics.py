"""FVVB 湍流传播中的偏振与 Stokes 指标。

当前湍流模型使用公共标量相位屏，因此本文件计算的是相干 Jones 场传播后
的偏振结构变化，而不是完整的湍流诱导退偏振模型。Stokes 符号约定必须与
``fvvb_core.linear_to_circular`` 中 sigma+/sigma- 的定义保持一致。"""

from __future__ import annotations

from typing import Any

import numpy as np


def stokes_parameters(Ex: np.ndarray, Ey: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """由线偏振 Jones 分量 Ex/Ey 计算局域 Stokes 参数。

    [关键变量] S0 为总强度，S1 表示水平/垂直线偏振差，S2 表示 ±45° 线偏振
    相关量，S3 表示圆偏振相关量。这里 S3 = -2 Im(Ex conj(Ey))，与本项目
    的圆偏振定义一致，请勿单独更改符号。

    原英文说明：

    Convention used here:
    ``S0 = |Ex|^2 + |Ey|^2``;
    ``S1 = |Ex|^2 - |Ey|^2``;
    ``S2 = 2 Re(Ex conj(Ey))``;
    ``S3 = -2 Im(Ex conj(Ey))``.
    The sign of ``S3`` matches the circular-component convention in
    ``fvvb_core.linear_to_circular``.
    """
    cross = Ex * np.conj(Ey)
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(cross)
    S3 = -2 * np.imag(cross)
    return S0, S1, S2, S3


def integrated_stokes(Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """计算采样窗口内的积分 Stokes 向量，积分权重为 dx*dy。"""
    stokes = stokes_parameters(Ex, Ey)
    return np.asarray([float(np.sum(s) * dx * dy) for s in stokes], dtype=float)


def polarization_summary(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    ref: np.ndarray | None = None,
) -> dict[str, Any]:
    """返回积分 Stokes、归一化 Stokes 与相对参考偏振变化量。

    [关键变量] normalized_stokes = [1, S1/S0, S2/S0, S3/S0]；
    stokes_l2_delta 表示当前归一化偏振态与参考态的 L2 距离。

    原英文说明：

    If ``ref`` is provided, it should be a 4-element integrated Stokes vector;
    the returned ``stokes_l2_delta`` compares normalized Stokes vectors.
    """
    stokes = integrated_stokes(Ex, Ey, dx, dy)
    s0 = max(float(stokes[0]), np.finfo(float).eps)
    normalized = stokes / s0
    result: dict[str, Any] = {
        "stokes": stokes,
        "normalized_stokes": normalized,
        "S0": float(stokes[0]),
        "S1_over_S0": float(normalized[1]),
        "S2_over_S0": float(normalized[2]),
        "S3_over_S0": float(normalized[3]),
    }
    if ref is not None:
        ref = np.asarray(ref, dtype=float)
        ref_norm = ref / max(float(ref[0]), np.finfo(float).eps)
        result["stokes_l2_delta"] = float(np.linalg.norm(normalized[1:] - ref_norm[1:]))
    else:
        result["stokes_l2_delta"] = float("nan")
    return result
