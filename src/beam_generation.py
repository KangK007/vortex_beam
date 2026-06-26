"""Beam generation utilities for fractional vector vortex beams.

The formulas in this module are migrated from the original ``fvvb_core.py``.
They keep the Berry expansion, Jones-field convention, circular-basis
conversion, and SPP phase factor unchanged.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.special import iv


@dataclass
class FVVBParams:
    """Physical and numerical parameters for FVB/FVVB calculations.

    Lengths are in metres and phases are in radians.
    """

    wavelength: float = 632.8e-9
    w0: float = 2.0e-3
    z: float = 0.2
    n_min: int = -200
    n_max: int = 200
    grid_n: int = 512
    rho_max_factor: float = 8.0
    lmax: int = 12
    pmax: int = 25
    nr_fourier: int = 320
    nphi_fourier: int = 720

    @property
    def k(self) -> float:
        return 2 * np.pi / self.wavelength

    @property
    def zR(self) -> float:
        return rayleigh_range(self.wavelength, self.w0)

    @property
    def wz(self) -> float:
        return self.w0 * np.sqrt(1 + (self.z / self.zR) ** 2)

    @property
    def n_list(self) -> np.ndarray:
        return np.arange(self.n_min, self.n_max + 1)

    @property
    def l_list(self) -> np.ndarray:
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
    """Return Rayleigh length ``z_R = k w0^2 / 2``."""
    k = 2 * np.pi / wavelength
    return k * w0**2 / 2


def make_cartesian_grid(
    n: int,
    rho_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Create a Cartesian grid matching MATLAB ``meshgrid(linspace(...))``."""
    x = np.linspace(-rho_max, rho_max, n)
    y = np.linspace(-rho_max, rho_max, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return X, Y, R, PHI, dx, dy


def get_grid_from_params(
    params: FVVBParams,
    rho_max_factor: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    factor = params.rho_max_factor if rho_max_factor is None else rho_max_factor
    return make_cartesian_grid(params.grid_n, factor * params.wz)


def berry_coeff(alpha: float, n_list: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Return Berry expansion coefficients for fractional topological charge."""
    n_list = np.asarray(n_list, dtype=float)
    hit = np.abs(alpha - n_list) < tol
    if np.any(hit):
        coeff = np.zeros_like(n_list, dtype=np.complex128)
        coeff[hit] = 1.0 + 0.0j
        return coeff
    return np.exp(1j * np.pi * alpha) * np.sin(np.pi * alpha) / (
        np.pi * (alpha - n_list)
    )


def _e_prime_single_order(
    rho_flat: np.ndarray,
    abs_n: int,
    z: float,
    k: float,
    w0: float,
    zR: float,
) -> np.ndarray:
    """Return propagated radial factor ``E'_n(r,z)`` from the original model."""
    rho_flat = np.asarray(rho_flat, dtype=float)
    term_const = (
        zR**2
        / (z - 1j * zR) ** 1.5
        * np.sqrt(np.pi / (4 * z * w0**2))
        * np.exp(1j * k * z)
    )
    arg = (zR**2 * rho_flat**2 / w0**2) / (2 * z * (z - 1j * zR))
    phase_term = np.exp(
        1j
        * k
        * rho_flat**2
        / (2 * z)
        * (1 + (1j * zR) / (2 * (z - 1j * zR)))
    )
    pre_factor = (-1j) ** (abs_n + 1)
    return term_const * pre_factor * np.abs(rho_flat) * phase_term * (
        iv((abs_n - 1) / 2, arg) - iv((abs_n + 1) / 2, arg)
    )


def fvb_field(
    alpha: float,
    rho: np.ndarray,
    phi: np.ndarray,
    params: FVVBParams,
    n_list: np.ndarray | None = None,
) -> np.ndarray:
    """Construct scalar FVB complex amplitude at the analytic ``z`` plane."""
    if n_list is None:
        n_list = params.n_list
    coeff = berry_coeff(alpha, n_list)
    rho_flat = rho.ravel()
    phi_flat = phi.ravel()
    U = np.zeros(rho_flat.shape, dtype=np.complex128)
    for n, c in zip(n_list, coeff):
        if abs(c) == 0:
            continue
        ep = _e_prime_single_order(
            rho_flat, int(abs(n)), params.z, params.k, params.w0, params.zR
        )
        U += c * ep * np.exp(1j * n * phi_flat)
    return U.reshape(rho.shape)


def fvvb_field(
    alpha: float,
    rho: np.ndarray,
    phi: np.ndarray,
    params: FVVBParams,
    n_list: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct FVVB Jones components ``Ex`` and ``Ey`` at the analytic ``z`` plane."""
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
        ep = _e_prime_single_order(
            rho_flat, int(abs(n)), params.z, params.k, params.w0, params.zR
        )
        angle = n * phi_flat
        Ex += c * ep * np.sin(angle)
        Ey += 1j * c * ep * np.cos(angle)
    return Ex.reshape(rho.shape), Ey.reshape(rho.shape)


def lg_p0_radial(rho: np.ndarray, ell_abs: int, w: float) -> np.ndarray:
    """Return the waist-plane radial envelope of ``LG_{p=0}^{|ell|}``."""
    from scipy.special import gammaln

    ell_abs = abs(int(ell_abs))
    rho = np.asarray(rho, dtype=float)
    log_c = 0.5 * (np.log(2) - np.log(np.pi) - gammaln(ell_abs + 1)) - np.log(w)
    C = np.exp(log_c)
    return C * (np.sqrt(2) * rho / w) ** ell_abs * np.exp(-(rho**2) / w**2)


def fvb_field_waist(
    alpha: float,
    rho: np.ndarray,
    phi: np.ndarray,
    params: FVVBParams,
    n_list: np.ndarray | None = None,
) -> np.ndarray:
    """Construct waist-plane scalar FVB using the original LG p=0 expansion."""
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
    """Construct waist-plane FVVB Jones components."""
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
    """Convert linear Jones components to ``sigma+`` and ``sigma-`` components."""
    return (Ex + 1j * Ey) / np.sqrt(2), (Ex - 1j * Ey) / np.sqrt(2)


def apply_spp(U: np.ndarray, phi: np.ndarray, fork_ell: float) -> np.ndarray:
    """Apply same-plane SPP/fork phase modulation ``exp(i fork_ell phi)``."""
    return U * np.exp(1j * fork_ell * phi)


def apply_spp_to_vector(
    Ex: np.ndarray,
    Ey: np.ndarray,
    phi: np.ndarray,
    fork_ell: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same SPP phase factor to both linear polarization components."""
    H = np.exp(1j * fork_ell * phi)
    return Ex * H, Ey * H

