"""Configuration, serialization, and run-directory utilities."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .beam_generation import FVVBParams
from .propagation import TurbulenceParams


DEFAULT_CONFIG: dict[str, Any] = {
    "run": {
        "name": "default",
        "output_root": "results",
    },
    "beam": {
        "wavelength": 632.8e-9,
        "w0": 2.0e-3,
        "z": 0.2,
        "alpha": 0.5,
        "initial_plane": "waist",
        "n_min": -20,
        "n_max": 20,
        "grid_n": 96,
        "rho_max_factor": 4.0,
        "lmax": 12,
        "pmax": 12,
        "nr_fourier": 120,
        "nphi_fourier": 240,
    },
    "spp": {
        "enabled": False,
        "fork_ell": 0.0,
    },
    "propagation": {
        "enabled": True,
        "distance": 0.2,
        "basis": "linear",
        "method": "same",
        "pad_factor": 1.0,
        "check_finite": True,
    },
    "turbulence": {
        "enabled": False,
        "cn2": 0.0,
        "n_screens": 0,
        "dz": None,
        "inner_scale": 2e-3,
        "outer_scale": 50.0,
        "seed": 1,
        "spectrum": "von_karman",
    },
    "scan": {
        "alpha_list": [0.5],
        "cn2_list": [0.0],
        "distance_list": [0.2],
        "n_mc": 1,
        "seed": 1,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config and merge it with project defaults."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = deep_update(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(path)
    return cfg


def params_from_config(cfg: dict[str, Any]) -> FVVBParams:
    beam = cfg["beam"]
    return FVVBParams(
        wavelength=float(beam["wavelength"]),
        w0=float(beam["w0"]),
        z=float(beam["z"]),
        n_min=int(beam["n_min"]),
        n_max=int(beam["n_max"]),
        grid_n=int(beam["grid_n"]),
        rho_max_factor=float(beam["rho_max_factor"]),
        lmax=int(beam["lmax"]),
        pmax=int(beam["pmax"]),
        nr_fourier=int(beam["nr_fourier"]),
        nphi_fourier=int(beam["nphi_fourier"]),
    )


def turbulence_params_from_config(
    cfg: dict[str, Any],
    distance: float | None = None,
    seed: int | None = None,
    cn2: float | None = None,
) -> TurbulenceParams:
    turb = cfg["turbulence"]
    n_screens = max(int(turb["n_screens"]), 1)
    dz_cfg = turb.get("dz")
    dz = float(dz_cfg) if dz_cfg is not None else float(distance or 0.0) / n_screens
    return TurbulenceParams(
        cn2=float(turb["cn2"] if cn2 is None else cn2),
        dz=dz,
        inner_scale=float(turb["inner_scale"]),
        outer_scale=float(turb["outer_scale"]),
        seed=int(turb["seed"] if seed is None else seed),
        spectrum=str(turb["spectrum"]),
    )


def create_run_dirs(cfg: dict[str, Any], run_name: str | None = None) -> dict[str, Path]:
    root = Path(cfg["run"]["output_root"])
    if not root.is_absolute():
        root = Path.cwd() / root
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or str(cfg["run"]["name"])
    run_id = f"{timestamp}_{name}"
    dirs = {
        "root": root,
        "figures": root / "figures" / run_id,
        "data": root / "data" / run_id,
        "logs": root / "logs" / run_id,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, (np.integer, np.floating)):
        item = value.item()
        return item if not isinstance(item, float) or np.isfinite(item) else None
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    if hasattr(value, "to_dict"):
        return jsonify(value.to_dict())
    return value


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonify(data), f, ensure_ascii=False, indent=2, allow_nan=False)


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(jsonify(data), f, allow_unicode=True, sort_keys=False)


def save_npz(path: str | Path, **arrays: Any) -> None:
    np.savez(Path(path), **arrays)
