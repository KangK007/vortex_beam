from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    """Reproduce the default single-run figures.

    This is intentionally conservative. Thesis-grade figure freezing should use
    reviewed YAML files and recorded run folders.
    """
    cmd = [
        sys.executable,
        str(
            PROJECT_ROOT
            / "scripts"
            / "field_generation"
            / "simulate_single_fvvb_beam.py"
        ),
        "--config",
        str(PROJECT_ROOT / "configs" / "default.yaml"),
    ]
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
