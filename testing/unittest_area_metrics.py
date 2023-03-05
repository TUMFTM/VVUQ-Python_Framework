"""
This module includes unittests for the src.variants.metrics.area_metrics module.

Contact person: Stefan Riedmaier
Creation date: 24.10.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import unittest

# -- third-party imports --
import numpy as np

# -- custom imports --
from src.variants.metrics import area_metrics


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class TestAreaMetrics(unittest.TestCase):
    """
    This class includes unittests for the area metrics module.
    """

    def test_merge_ecdf_functions(self):
        """
        This function tests the merge_ecdf_functions function.
        """

        # --  -----------------------------------------------------------------------

        # -- 5 vs. 4
        ay = np.linspace(0, 1, 5)
        by = np.linspace(0, 1, 4)
        y_unique, a_idx, b_idx = area_metrics.merge_ecdf_functions(ay, by)
        y_unique_exp = np.array([0, 0.25, 1/3, 0.5, 2/3, 0.75, 1])
        a_idx_exp = np.array([0, 1, 2, 2, 3, 3, 4])
        b_idx_exp = np.array([0, 1, 1, 2, 2, 3, 3])
        np.testing.assert_allclose(y_unique, y_unique_exp)
        np.testing.assert_array_equal(a_idx, a_idx_exp)
        np.testing.assert_array_equal(b_idx, b_idx_exp)

        # -- 4 vs. 5
        y_unique, a_idx, b_idx = area_metrics.merge_ecdf_functions(by, ay)
        np.testing.assert_allclose(y_unique, y_unique_exp)
        np.testing.assert_array_equal(a_idx, b_idx_exp)
        np.testing.assert_array_equal(b_idx, a_idx_exp)

        # -- 6 vs. 4
        ay = np.linspace(0, 1, 6)
        by = np.linspace(0, 1, 4)
        y_unique, a_idx, b_idx = area_metrics.merge_ecdf_functions(ay, by)
        y_unique_exp = np.array([0, 0.2, 1/3, 0.4, 0.6, 2/3, 0.8, 1])
        a_idx_exp = np.array([0, 1, 2, 2, 3, 4, 4, 5])
        b_idx_exp = np.array([0, 1, 1, 2, 2, 2, 3, 3])
        np.testing.assert_allclose(y_unique, y_unique_exp)
        np.testing.assert_array_equal(a_idx, a_idx_exp)
        np.testing.assert_array_equal(b_idx, b_idx_exp)

        # -- 4 vs. 6
        y_unique, a_idx, b_idx = area_metrics.merge_ecdf_functions(by, ay)
        np.testing.assert_allclose(y_unique, y_unique_exp)
        np.testing.assert_array_equal(a_idx, b_idx_exp)
        np.testing.assert_array_equal(b_idx, a_idx_exp)


if __name__ == '__main__':
    unittest.main()
