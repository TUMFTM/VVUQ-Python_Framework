"""
This module includes unittests for the src.helpers.signal_processing.digitalization module.

Contact person: Stefan Riedmaier
Creation date: 24.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import unittest

# -- third-party imports --
import numpy as np

# -- custom imports --
from src.helpers.signal_processing import digitalization


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class TestDigitalization(unittest.TestCase):
    """
    This class includes unittests for the digitalization module.
    """

    def test_pullup_glitches(self):
        """
        This function tests the pullup_glitches function.

        x: input vector
        y: output vector
        z: expected output vector
        """

        # -- 1d-array x, max_glitch_duration = 1 -----------------------------------------------------------------------

        x = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0])
        np.testing.assert_array_equal(z, y)

        x = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
             0, 0])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
             0, 0])
        np.testing.assert_array_equal(z, y)

        x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(z, y)

        x = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(z, y)

        x = np.array([1, 1, 1, 1, 1, 1, 1, 0, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(z, y)

        x = np.array([1, 0, 0, 0, 0, 0, 0, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([1, 0, 0, 0, 0, 0, 0, 1])
        np.testing.assert_array_equal(z, y)

        # -- 1d-array x, max_glitch_duration = 4 -----------------------------------------------------------------------

        x = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(z, y)

        x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(z, y)

        x = np.array([1, 1, 0, 0, 0, 1, 1, 0, 0])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(z, y)

        x = np.array([0, 1, 1, 1, 0, 0, 0])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([0, 1, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(z, y)

        x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(z, y)

        x = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
        y = digitalization.pullup_glitches(x, max_glitch_duration=4)
        z = np.array([0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(z, y)

        # -- 2d-array x ------------------------------------------------------------------------------------------------

        x = np.array([[0, 0, 1, 1, 1, 0, 1, 1, 0],
                      [1, 1, 0, 1, 0, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([[0, 0, 1, 1, 1, 1, 1, 1, 0],
                      [1, 1, 1, 1, 0, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(z, y)

        x = np.array([[0, 0, 1, 1, 1, 0, 1, 1, 0],
                      [1, 1, 0, 1, 0, 0, 0, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1, axis=0)
        z = np.array([[0, 0, 1, 1, 1, 0, 1, 1, 0],
                      [1, 1, 1, 1, 1, 0, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(z, y)

        # -- 3d-array x ------------------------------------------------------------------------------------------------

        x = np.array([[[0, 0, 1, 1, 1, 0, 1, 1, 0],
                       [1, 1, 0, 1, 0, 0, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]],
                      [[0, 0, 1, 1, 1, 0, 1, 1, 0],
                       [1, 1, 0, 1, 0, 0, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]]])
        y = digitalization.pullup_glitches(x, max_glitch_duration=1)
        z = np.array([[[0, 0, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 0, 0, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]],
                      [[0, 0, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 0, 0, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1]]])
        np.testing.assert_array_equal(z, y)

    def test_get_event_boundaries(self):
        """
        This function tests the get_event_boundaries function.
        """

        # -- 1d-array x, min_length = 1 --------------------------------------------------------------------------------

        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp, stop_indices_exp, event_length_exp = (np.array([0]),), (np.array([5]),), np.array([5])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([0, 0, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp, stop_indices_exp, event_length_exp = (np.array([]),), (np.array([]),), np.array([])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([1, 1, 1, 1], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp, stop_indices_exp, event_length_exp = (np.array([0]),), (np.array([4]),), np.array([4])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
                      1, 1, 1, 1, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp = (np.array([5, 9, 20, 30]),)
        stop_indices_exp = (np.array([8, 12, 26, 36]),)
        event_length_exp = np.array([3, 3, 6, 6])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        # -- 1d-array x, min_length > 1 --------------------------------------------------------------------------------

        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=5)
        start_indices_exp, stop_indices_exp, event_length_exp = (np.array([0]),), (np.array([5]),), np.array([5])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=6)
        start_indices_exp, stop_indices_exp, event_length_exp = (np.array([]),), (np.array([]),), np.array([])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
                      1, 1, 1, 1, 0, 0], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=4)
        start_indices_exp = (np.array([20, 30]),)
        stop_indices_exp = (np.array([26, 36]),)
        event_length_exp = np.array([6, 6])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        # -- 2d-array x ------------------------------------------------------------------------------------------------

        x = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 0, 1, 1, 0]], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp = (np.array([0, 1, 1]), np.array([0, 3, 6]))
        stop_indices_exp = (np.array([0, 1, 1]), np.array([5, 5, 8]))
        event_length_exp = np.array([5, 2, 2])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        # -- 3d-array x ------------------------------------------------------------------------------------------------

        x = np.array([[[1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 1, 1, 0, 1, 1, 0]],
                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 0, 1, 0, 1]]], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp = (np.array([0, 0, 0, 1, 1, 1, 1, 1]),
                             np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                             np.array([0, 3, 6, 0, 2, 4, 6, 8]))
        stop_indices_exp = (np.array([0, 0, 0, 1, 1, 1, 1, 1]),
                            np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                            np.array([5, 5, 8, 1, 3, 5, 7, 9]))
        event_length_exp = np.array([5, 2, 2, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        x = np.array([[[1, 1, 1, 1, 1, 0, 0, 0, 0],
                       [1, 0, 1, 1, 1, 0, 1, 1, 0]],
                      [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [1, 0, 1, 0, 1, 1, 0, 0, 1]]], dtype=int)
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp = (np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                             np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                             np.array([0, 0, 2, 6, 0, 2, 4, 8]))
        stop_indices_exp = (np.array([0, 0, 0, 0, 1, 1, 1, 1]),
                            np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                            np.array([5, 1, 5, 8, 1, 3, 6, 9]))
        event_length_exp = np.array([5, 1, 3, 2, 1, 1, 2, 1])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)

        # -- nan-values ------------------------------------------------------------------------------------------------

        x = np.array([[[1, 1, 1, 1, 1, 0, 0, np.nan, np.nan],
                       [0, 0, 0, 1, 1, 0, 1, 1, 0]],
                      [[0, 0, 0, 0, 0, 0, np.nan, np.nan, np.nan],
                       [1, 0, 1, 0, 1, 0, 1, np.nan, np.nan]]])
        start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
        start_indices_exp = (np.array([0, 0, 0, 1, 1, 1, 1]),
                             np.array([0, 1, 1, 1, 1, 1, 1]),
                             np.array([0, 3, 6, 0, 2, 4, 6]))
        stop_indices_exp = (np.array([0, 0, 0, 1, 1, 1, 1]),
                            np.array([0, 1, 1, 1, 1, 1, 1]),
                            np.array([5, 5, 8, 1, 3, 5, 7]))
        event_length_exp = np.array([5, 2, 2, 1, 1, 1, 1])
        np.testing.assert_array_equal(start_indices_exp, start_indices)
        np.testing.assert_array_equal(stop_indices_exp, stop_indices)
        np.testing.assert_array_equal(event_length_exp, event_length)


if __name__ == '__main__':
    unittest.main()
