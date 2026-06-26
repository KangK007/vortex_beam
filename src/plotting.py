"""Plotting helpers for thesis FVVB simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def style_axes(ax, xlabel: str, ylabel: str, title: str | None = None) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25)


def save_intensity_phase_figure(
    path: Path,
    X: np.ndarray,
    Y: np.ndarray,
    intensity: np.ndarray,
    phase: np.ndarray,
    title: str,
) -> None:
    """Save total intensity and one representative phase map."""
    extent_mm = [
        float(X.min() * 1e3),
        float(X.max() * 1e3),
        float(Y.min() * 1e3),
        float(Y.max() * 1e3),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), constrained_layout=True)
    im0 = axes[0].imshow(intensity, extent=extent_mm, origin="lower", cmap="inferno")
    axes[0].set_title("Total intensity")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phase, extent=extent_mm, origin="lower", cmap="twilight")
    axes[1].set_title("Phase of Ex")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("y (mm)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_oam_spectrum_figure(
    path: Path,
    l_list: np.ndarray,
    spectrum: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.bar(l_list, spectrum, width=0.8, color="#4068a8", edgecolor="black", linewidth=0.4)
    style_axes(ax, "OAM order ell", "Normalized weight", title)
    ax.set_ylim(0, max(float(np.max(spectrum)) * 1.15, 0.05))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_scan_width_figure(
    path: Path,
    cn2_list: np.ndarray,
    alpha_list: np.ndarray,
    width_mean: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for idx, alpha in enumerate(alpha_list):
        ax.plot(cn2_list, width_mean[idx], marker="o", label=f"alpha={alpha:g}")
    ax.set_xscale("symlog", linthresh=1e-18)
    style_axes(ax, "Cn2 (m^(-2/3))", "OAM spectral width", "Width vs turbulence")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=300)
    plt.close(fig)

