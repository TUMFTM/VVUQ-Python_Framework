"""
This module offers helper functions for signal interpolation.

They are important for the data-driven assessment, e.g. in the UNECE-R79 use case.

It includes a resample and a interpolation function based on scipy.
They are mainly based on numpy arrays for fast vectorized operations.
See details in their respective documentations.

Contact person: Stefan Riedmaier
Creation date: 17.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def resample_signal(quantity, length):
    """
    This function resamples a signal via scipys Fourier method.

    :param np.ndarray quantity: signal to be resampled
    :param int length: number of samples in the resampled signal
    :return: resampled signal
    :rtype: np.ndarray
    """

    # apply the resample method to the signal
    quantity_resampled = signal.resample(quantity, length)

    return quantity_resampled


def interpolate_signal(quantity, length, method='nearest'):
    """
    This function interpolates a signal based on scipy interp1d method.

    :param np.ndarray quantity: signal to be interpolated
    :param int length: number of samples in the interpolated signal
    :param str method: (optional) scipy interpolation method
    :return: interpolated signal
    :rtype: np.ndarray
    """

    # create x vectors
    x = np.linspace(0, len(quantity), num=len(quantity), endpoint=True)
    x_interpolated = np.linspace(0, len(quantity), num=length, endpoint=True)

    # create the interpolation function
    func = interp1d(x, quantity, kind=method)

    # apply the interpolation function to the signal
    quantity_interpolated = func(x_interpolated)

    return quantity_interpolated
