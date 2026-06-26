"""OAM, LG radial-mode, phase, and polarization analysis utilities."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.special import eval_genlaguerre, gammaln

from .beam_generation import linear_to_circular


def field_intensity(Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
    return np.abs(Ex) ** 2 + np.abs(Ey) ** 2


def field_phase(U: np.ndarray) -> np.ndarray:
    return np.angle(U)


def _interp_to_polar(
    U: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    nr: int,
    nphi: int,
):
    xvec = X[0, :]
    yvec = Y[:, 0]
    rmax = min(float(np.max(np.abs(xvec))), float(np.max(np.abs(yvec))))
    r = np.linspace(0, rmax, nr)
    ph = np.linspace(-np.pi, np.pi, nphi, endpoint=False)
    RR, PP = np.meshgrid(r, ph, indexing="xy")
    XX = RR * np.cos(PP)
    YY = RR * np.sin(PP)
    interp = RegularGridInterpolator(
        (yvec, xvec), U, method="linear", bounds_error=False, fill_value=0.0
    )
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
    """Return angular-Fourier OAM energies and total sampled polar energy."""
    U_polar, RR, PP, r, ph = _interp_to_polar(U, X, Y, nr, nphi)
    dr = float(r[1] - r[0]) if len(r) > 1 else 1.0
    dphi = float(ph[1] - ph[0]) if len(ph) > 1 else 2 * np.pi
    etot = float(np.sum(np.abs(U_polar) ** 2 * RR) * dr * dphi)
    el = []
    for ell in l_list:
        phase = np.exp(-1j * ell * PP)
        a_l = (1 / (2 * np.pi)) * np.sum(U_polar * phase, axis=0) * dphi
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
    """Return total FVVB OAM spectrum from the two circular components."""
    Ep, Em = linear_to_circular(Ex, Ey)
    El_p, Et_p = oam_spectrum_angular_fourier(Ep, X, Y, l_list, nr, nphi)
    El_m, Et_m = oam_spectrum_angular_fourier(Em, X, Y, l_list, nr, nphi)
    El_total = El_p + El_m
    denom = np.sum(El_total) if conditional else (Et_p + Et_m)
    mu = El_total / max(float(denom), np.finfo(float).eps)
    meta = {
        "Et_plus": Et_p,
        "Et_minus": Et_m,
        "Pcap": float(np.sum(El_total) / max(Et_p + Et_m, np.finfo(float).eps)),
    }
    return mu, meta


def spectral_width(x: Iterable[float], weights: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(x), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    s = float(np.sum(w))
    if not np.isfinite(s) or s <= 0:
        return np.nan, np.nan
    wn = w / s
    mean = float(np.sum(x * wn))
    sigma = float(np.sqrt(np.sum((x - mean) ** 2 * wn)))
    return mean, sigma


def normalize_spectrum(power: np.ndarray, eps: float | None = None) -> np.ndarray:
    if eps is None:
        eps = np.finfo(float).eps
    power = np.asarray(power, dtype=float)
    return power / max(float(np.sum(power)), eps)


def spectrum_entropy(mu: np.ndarray, normalized: bool = True) -> float:
    mu = normalize_spectrum(mu)
    use = mu > 0
    if not np.any(use):
        return float("nan")
    entropy = float(-np.sum(mu[use] * np.log(mu[use])))
    if normalized and mu.size > 1:
        entropy /= float(np.log(mu.size))
    return entropy


def spectrum_similarity(mu: np.ndarray, ref_mu: np.ndarray | None) -> dict[str, float]:
    if ref_mu is None:
        return {"cosine": float("nan"), "l1": float("nan")}
    mu = normalize_spectrum(mu)
    ref_mu = normalize_spectrum(ref_mu)
    denom = float(np.linalg.norm(mu) * np.linalg.norm(ref_mu))
    cosine = float(np.dot(mu, ref_mu) / max(denom, np.finfo(float).eps))
    l1 = float(np.sum(np.abs(mu - ref_mu)))
    return {"cosine": cosine, "l1": l1}


def compute_oam_summary(
    l_list: np.ndarray,
    mu: np.ndarray,
    ref_mu: np.ndarray | None = None,
) -> dict[str, float]:
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


def lg_mode_xy(R: np.ndarray, PHI: np.ndarray, p: int, ell: int, w: float) -> np.ndarray:
    abs_l = abs(int(ell))
    log_c = (
        0.5
        * (np.log(2) + gammaln(p + 1) - np.log(np.pi) - gammaln(p + abs_l + 1))
        - np.log(w)
    )
    C = np.exp(log_c)
    rho2 = 2 * R**2 / w**2
    Lpa = eval_genlaguerre(p, abs_l, rho2)
    return (
        C
        * (np.sqrt(2) * R / w) ** abs_l
        * Lpa
        * np.exp(-(R**2) / w**2)
        * np.exp(1j * ell * PHI)
    )


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
    l_arr = np.asarray(list(l_list), dtype=int)
    Mpl = np.zeros((pmax + 1, len(l_arr)), dtype=float)
    etot = float(np.sum(np.abs(U) ** 2) * dx * dy)
    R = np.hypot(X, Y)
    PHI = np.arctan2(Y, X)
    for j, ell in enumerate(l_arr):
        for p in range(pmax + 1):
            LG = lg_mode_xy(R, PHI, p, int(ell), w_an)
            c = np.sum(np.conj(LG) * U) * dx * dy
            Mpl[p, j] = float(np.abs(c) ** 2)
    return Mpl, etot


def radial_width_from_mpl(
    Mpl: np.ndarray,
    etot: float,
    conditional: bool = True,
) -> tuple[np.ndarray, float, float, float]:
    nu_p = np.sum(Mpl, axis=1) / max(etot, np.finfo(float).eps)
    pcap = float(np.sum(nu_p))
    weights = nu_p / max(pcap, np.finfo(float).eps) if conditional else nu_p
    p_list = np.arange(len(nu_p))
    mean, sigma = spectral_width(p_list, weights)
    return nu_p, mean, sigma, pcap


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


def stokes_parameters(
    Ex: np.ndarray,
    Ey: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cross = Ex * np.conj(Ey)
    S0 = np.abs(Ex) ** 2 + np.abs(Ey) ** 2
    S1 = np.abs(Ex) ** 2 - np.abs(Ey) ** 2
    S2 = 2 * np.real(cross)
    S3 = -2 * np.imag(cross)
    return S0, S1, S2, S3


def integrated_stokes(Ex: np.ndarray, Ey: np.ndarray, dx: float, dy: float) -> np.ndarray:
    stokes = stokes_parameters(Ex, Ey)
    return np.asarray([float(np.sum(s) * dx * dy) for s in stokes], dtype=float)


def polarization_summary(
    Ex: np.ndarray,
    Ey: np.ndarray,
    dx: float,
    dy: float,
    ref: np.ndarray | None = None,
) -> dict[str, Any]:
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

