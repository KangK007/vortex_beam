"""Propagation and turbulence-screen utilities for FVVB simulations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any
import math

import numpy as np

from .beam_generation import linear_to_circular


@dataclass
class TurbulenceParams:
    """Single-layer modified von Karman phase-screen parameters."""

    cn2: float = 1e-15
    dz: float = 100.0
    inner_scale: float = 2e-3
    outer_scale: float = 50.0
    seed: int = 1
    spectrum: str = "von_karman"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rytov_variance(cn2: float, wavelength: float, distance: float) -> float:
    k = 2 * np.pi / wavelength
    return float(1.23 * cn2 * k ** (7 / 6) * distance ** (11 / 6))


def fried_parameter(cn2: float, wavelength: float, distance: float) -> float:
    if cn2 <= 0 or distance <= 0:
        return float("inf")
    k = 2 * np.pi / wavelength
    return float((0.423 * k**2 * cn2 * distance) ** (-3 / 5))


def phase_psd_von_karman(
    kappa: np.ndarray,
    wavelength: float,
    cn2: float,
    dz: float,
    inner_scale: float = 2e-3,
    outer_scale: float = 50.0,
) -> np.ndarray:
    """Return the modified von Karman thin phase-screen PSD."""
    k = 2 * np.pi / wavelength
    kappa0 = 2 * np.pi / outer_scale
    kappam = 5.92 / inner_scale
    refractive_psd = 0.033 * cn2 * np.exp(-(kappa / kappam) ** 2) / (
        kappa**2 + kappa0**2
    ) ** (11 / 6)
    return 2 * np.pi * k**2 * dz * refractive_psd


def make_phase_screen(
    n: int,
    dx: float,
    dy: float,
    wavelength: float,
    params: TurbulenceParams,
) -> np.ndarray:
    """Generate one random scalar phase screen in radians."""
    if params.spectrum != "von_karman":
        raise ValueError(f"Unsupported turbulence spectrum: {params.spectrum}")

    fx = np.fft.fftfreq(n, d=dx)
    fy = np.fft.fftfreq(n, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    kappa = 2 * np.pi * np.hypot(FX, FY)
    psd = phase_psd_von_karman(
        kappa,
        wavelength,
        params.cn2,
        params.dz,
        params.inner_scale,
        params.outer_scale,
    )
    psd[0, 0] = 0.0
    dkx = 2 * np.pi / (n * dx)
    dky = 2 * np.pi / (n * dy)
    rng = np.random.default_rng(params.seed)
    noise = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    spectral_phase = noise * np.sqrt(psd) * np.sqrt(dkx * dky)
    phase = np.fft.ifft2(spectral_phase).real * n * n
    return phase - float(np.mean(phase))


def make_phase_screens(
    n: int,
    dx: float,
    dy: float,
    wavelength: float,
    params: TurbulenceParams,
    count: int,
) -> list[np.ndarray]:
    screens = []
    for idx in range(count):
        layer_params = TurbulenceParams(**{**params.to_dict(), "seed": params.seed + idx})
        screens.append(make_phase_screen(n, dx, dy, wavelength, layer_params))
    return screens


def phase_screen_stats(phase: np.ndarray) -> dict[str, float]:
    phase = np.asarray(phase, dtype=float)
    return {
        "mean": float(np.mean(phase)),
        "std": float(np.std(phase)),
        "rms": float(np.sqrt(np.mean(phase**2))),
        "min": float(np.min(phase)),
        "max": float(np.max(phase)),
    }


def assert_finite_array(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if np.all(finite):
        return
    nonfinite = int(arr.size - np.count_nonzero(finite))
    raise FloatingPointError(f"Non-finite values detected in {name}: {nonfinite}/{arr.size}")


def _center_pad_2d(U: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
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
    if pad_factor < 1:
        raise ValueError("pad_factor must be >= 1")
    ny, nx = shape
    return max(ny, int(math.ceil(ny * pad_factor))), max(
        nx, int(math.ceil(nx * pad_factor))
    )


def _edge_mask(shape: tuple[int, int], edge_fraction: float) -> np.ndarray:
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


def vector_field_diagnostics(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    edge_fraction: float = 0.05,
) -> dict[str, float]:
    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    if Ex.shape != Ey.shape:
        raise ValueError("Ex and Ey must have the same shape")
    finite_x = np.isfinite(Ex)
    finite_y = np.isfinite(Ey)
    Ex_safe = np.nan_to_num(Ex, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    Ey_safe = np.nan_to_num(Ey, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = np.abs(Ex_safe) ** 2 + np.abs(Ey_safe) ** 2
    energy = float(np.sum(intensity) * dx * dy)
    edge = _edge_mask(Ex.shape, edge_fraction)
    edge_energy = float(np.sum(intensity[edge]) * dx * dy)
    global_mean = float(np.mean(intensity))
    peak = float(np.max(intensity)) if intensity.size else float("nan")
    eps = np.finfo(float).eps
    return {
        "energy": energy,
        "edge_energy": edge_energy,
        "edge_energy_fraction": edge_energy / max(energy, eps),
        "global_mean_intensity": global_mean,
        "peak_intensity": peak,
        "edge_mean_to_peak": float(np.mean(intensity[edge])) / max(peak, eps),
        "finite_fraction": float(
            (np.count_nonzero(finite_x) + np.count_nonzero(finite_y)) / (2 * Ex.size)
        ),
    }


def angular_spectrum_propagate(
    U: np.ndarray,
    dx: float,
    dy: float,
    wavelength: float,
    distance: float,
) -> np.ndarray:
    """Propagate one scalar field by the angular spectrum method."""
    if distance == 0:
        return np.array(U, copy=True)
    ny, nx = U.shape
    k = 2 * np.pi / wavelength
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    kx = 2 * np.pi * FX
    ky = 2 * np.pi * FY
    kz_sq = (k**2 - kx**2 - ky**2).astype(np.complex128)
    H = np.exp(1j * np.sqrt(kz_sq) * distance)
    return np.fft.ifft2(np.fft.fft2(U) * H)


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
    if check_finite:
        assert_finite_array("angular_spectrum input", U)
    if method == "same":
        out = angular_spectrum_propagate(U, dx, dy, wavelength, distance)
    elif method == "padded":
        padded = _center_pad_2d(U, _padded_shape(U.shape, pad_factor))
        out = _center_crop_2d(
            angular_spectrum_propagate(padded, dx, dy, wavelength, distance),
            U.shape,
        )
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
    return (
        angular_spectrum_propagate_select(
            Ex, dx, dy, wavelength, distance, propagation, pad_factor, check_finite
        ),
        angular_spectrum_propagate_select(
            Ey, dx, dy, wavelength, distance, propagation, pad_factor, check_finite
        ),
    )


def apply_common_phase(U1: np.ndarray, U2: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = np.exp(1j * phase)
    return U1 * H, U2 * H


def apply_common_phase_to_linear(
    Ex: np.ndarray,
    Ey: np.ndarray,
    phase: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return apply_common_phase(Ex, Ey, phase)


def circular_to_linear(Ep: np.ndarray, Em: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Ex = (Ep + Em) / np.sqrt(2)
    Ey = -1j * (Ep - Em) / np.sqrt(2)
    return Ex, Ey


def field_energy(U: np.ndarray, dx: float, dy: float) -> float:
    return float(np.sum(np.abs(U) ** 2) * dx * dy)


def vector_energy(Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: float) -> float:
    return field_energy(Ex, dx, dy) + field_energy(Ey, dx, dy)


def _step_distance(distance: float, phase_screens: Sequence[np.ndarray]) -> float:
    if len(phase_screens) == 0:
        return distance
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
    """Propagate ``Ex/Ey`` through common scalar phase screens."""
    if check_finite:
        assert_finite_array("Ex input", Ex)
        assert_finite_array("Ey input", Ey)
    Ex_z = np.array(Ex, copy=True)
    Ey_z = np.array(Ey, copy=True)
    dz = _step_distance(distance, phase_screens)
    if len(phase_screens) == 0:
        return propagate_linear_components(
            Ex_z, Ey_z, dx, dy, wavelength, distance, propagation, pad_factor, check_finite
        )
    for phase in phase_screens:
        if check_finite:
            assert_finite_array("phase screen", phase)
        Ex_z, Ey_z = apply_common_phase_to_linear(Ex_z, Ey_z, phase)
        Ex_z, Ey_z = propagate_linear_components(
            Ex_z, Ey_z, dx, dy, wavelength, dz, propagation, pad_factor, check_finite
        )
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
    """Propagate in circular basis through common scalar phase screens."""
    if check_finite:
        assert_finite_array("Ex input", Ex)
        assert_finite_array("Ey input", Ey)
    Ep, Em = linear_to_circular(Ex, Ey)
    dz = _step_distance(distance, phase_screens)
    if len(phase_screens) == 0:
        Ep_z = angular_spectrum_propagate_select(
            Ep, dx, dy, wavelength, distance, propagation, pad_factor, check_finite
        )
        Em_z = angular_spectrum_propagate_select(
            Em, dx, dy, wavelength, distance, propagation, pad_factor, check_finite
        )
        return circular_to_linear(Ep_z, Em_z)
    for phase in phase_screens:
        if check_finite:
            assert_finite_array("phase screen", phase)
        Ep, Em = apply_common_phase(Ep, Em, phase)
        Ep = angular_spectrum_propagate_select(
            Ep, dx, dy, wavelength, dz, propagation, pad_factor, check_finite
        )
        Em = angular_spectrum_propagate_select(
            Em, dx, dy, wavelength, dz, propagation, pad_factor, check_finite
        )
    return circular_to_linear(Ep, Em)

