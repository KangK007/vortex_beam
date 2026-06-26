import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.beam_generation import berry_coeff, default_params, linear_to_circular
from src.propagation import circular_to_linear, vector_energy
from src.utils import load_config


class CoreModelTests(unittest.TestCase):
    def test_integer_alpha_uses_kronecker_delta_limit(self):
        n_list = np.arange(-2, 3)

        coeff = berry_coeff(1.0, n_list)

        expected = np.array([0, 0, 0, 1, 0], dtype=np.complex128)
        np.testing.assert_allclose(coeff, expected)

    def test_linear_circular_round_trip_preserves_field(self):
        Ex = np.array([[1 + 2j, 3 - 1j]])
        Ey = np.array([[0.5 - 0.2j, -1 + 0.1j]])

        Ep, Em = linear_to_circular(Ex, Ey)
        Ex_back, Ey_back = circular_to_linear(Ep, Em)

        np.testing.assert_allclose(Ex_back, Ex)
        np.testing.assert_allclose(Ey_back, Ey)

    def test_vector_energy_is_positive_for_generated_fvvb(self):
        from src.beam_generation import fvvb_field_waist, get_grid_from_params

        params = default_params(grid_n=32, n_min=-4, n_max=4, rho_max_factor=3.0)
        X, Y, R, PHI, dx, dy = get_grid_from_params(params)
        Ex, Ey = fvvb_field_waist(0.5, R, PHI, params)

        self.assertGreater(vector_energy(Ex, Ey, dx, dy), 0.0)
        self.assertEqual(Ex.shape, X.shape)
        self.assertEqual(Ey.shape, Y.shape)

    def test_load_config_merges_yaml_into_nested_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(
                "simulation:\n  alpha: 0.7\nbeam:\n  grid_n: 24\n",
                encoding="utf-8",
            )

            cfg = load_config(path)

        self.assertEqual(cfg["simulation"]["alpha"], 0.7)
        self.assertEqual(cfg["beam"]["grid_n"], 24)


if __name__ == "__main__":
    unittest.main()
