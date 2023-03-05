"""
This module offers helper functions for signal filtering.

They are important for the data-driven assessment, e.g. in the UNECE-R79 use case.

It includes the following filter functions:
- butterworth_filter: Butterworth filter
- moving_average_filter: moving average filter
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
from scipy.ndimage.filters import uniform_filter1d

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def butterworth_filter(quantity, filter_settings='GRVA', sample_rate=100):
    """
    This function applies a butterworth filter to a measured quantity.

    The following pre-defined settings of the butterworth filter are available:
    - 'GRVA': Filter recommended in GRVA-04-09 (UNECE)
    - 'NCAP': Filter used in vehicle dynamics from NCAP
    - 'VDA': Filter used in VDA-Draft (FKT SA FAS UAG Workshop B1)

    :param np.ndarray quantity: signal to be filtered
    :param str filter_settings: (optional) pre-configured filter types
    :param float sample_rate: (optional) sample rate of the signal
    :return: filtered signal
    :rtype: np.ndarray
    """

    # evaluate the filter type
    if filter_settings == 'GRVA':
        order = 4
        frequency = 0.5
    elif filter_settings == 'NCAP':
        order = 10
        frequency = 12
    elif filter_settings == 'VDA':
        order = 6
        frequency = 12
    else:
        raise ValueError("type of butterworth filter not available")

    # create the butterworth filter
    numerator_polynomial, denominator_polynomial = signal.butter(order, frequency, btype='lowpass', fs=sample_rate)

    # apply the butterworth filter to the signal
    quantity_filtered = signal.filtfilt(numerator_polynomial, denominator_polynomial, quantity)

    return quantity_filtered


def moving_average_filter(quantity, sample_rate=100, sample_window_length=0.5):
    """
    This function applies a moving average filter to a measured quantity.

    The implementation is based on the following stack overflow article:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean/43200476#43200476

    :param np.ndarray quantity: signal to be filtered
    :param float sample_rate: (optional) sample rate of the signal
    :param float sample_window_length: (optional) length of the moving average window
    :return: filtered signal
    :rtype: np.ndarray
    """

    length = int(sample_rate * sample_window_length)

    # fast scipy solution
    quantity_filtered = uniform_filter1d(quantity, size=length, mode='nearest')

    # slower solution based on numpy convolution
    # kernel = np.ones((length,))/length
    # quantity_filtered = np.convolve(quantity, kernel, mode='valid')

    return quantity_filtered
