"""
This module includes time series validation metrics.

It includes several time series metrics. See details in their own documentations.
The functions are mainly based on numpy arrays for fast vectorized operations.

Contact person: Stefan Riedmaier
Creation date: 03.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def sprague_geers_error_factors(y_true, y_pred, axis=-1):
    """
    This function calculates the magnitude, phase and comprehensive error factor by Sprague and Geers.

    Literature:
    Sprague, M.A., Geers, T.L.: A Spectral-element method for modelling cavitation in transient fluidstructure
    interaction. International Journal for Numerical Methods in Engineering 60 (2004), pp. 2467-2499

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: comprehensive, magnitude, phase error factor by Sprague and Geers
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """

    y_pred_sq_sum = np.sum(y_pred ** 2, axis=axis)
    y_true_sq_sum = np.sum(y_true ** 2, axis=axis)

    # Magnitude Error Factor by Sprague and Geers
    mef = y_pred_sq_sum / y_true_sq_sum - 1

    # Phase Error Factor by Sprague and Geers
    pef = 1 / np.pi * np.arccos(np.sum(y_pred * y_true, axis=axis) / np.sqrt(y_pred_sq_sum * y_true_sq_sum))

    # Comprehensive Error Factor by Sprague and Geers
    cef = np.sqrt(mef ** 2 + pef ** 2)

    return cef, mef, pef


def russell_error_factors(y_true, y_pred, axis=-1):
    """
    This function calculates the magnitude, phase and comprehensive error factor by Russell.

    Literature:
    Russell, D. M.: Error measures for comparing transient data: Part I: Development of a comprehensive error
    measure, Part II: Error measures case study. Proceedings of the 68th Shock and Vibration Symposium, Hunt Valley,
    Maryland, 1997

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: comprehensive, magnitude, phase error factor by Russell
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray)
    """

    y_pred_sq_sum = np.sum(y_pred ** 2, axis=axis)
    y_true_sq_sum = np.sum(y_true ** 2, axis=axis)
    magnitude = (y_pred_sq_sum - y_true_sq_sum) / np.sqrt(y_pred_sq_sum * y_true_sq_sum)

    # Magnitude Error Factor by Russell
    mef = np.sign(magnitude) * np.log10(1 + np.abs(magnitude))

    # Phase Error Factor by Russell
    pef = 1 / np.pi * np.arccos(np.sum(y_pred * y_true, axis=axis) / np.sqrt(y_pred_sq_sum * y_true_sq_sum))

    # Comprehensive Error Factor by Russell
    cef = np.sqrt(np.pi / 4 * (mef ** 2 + pef ** 2))

    return cef, mef, pef


def dtw(y_true, y_pred):
    """
    This function calculates the dynamic time warping (DTW) distance.

    https://pypi.org/project/dtw-python/
    from dtw import dtw, warpArea

    https://pypi.org/project/dtaidistance/
    https://pypi.org/project/fastdtw/
    https://github.com/pierre-rouanet/dtw

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :return: dynamic time warping distance
    :rtype: np.ndarray
    """

    raise NotImplementedError("Dynamic Time Warping not fully implemented")

    # ds = dtw(y_pred, y_true)
    # area = warpArea(ds)
    # return area
