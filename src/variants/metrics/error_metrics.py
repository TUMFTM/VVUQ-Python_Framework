"""
This module includes error validation metrics.

It includes several error metrics. See details in their own documentations.
The functions are mainly based on numpy arrays for fast vectorized operations.

Contact person: Stefan Riedmaier
Creation date: 03.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
import xarray as xr
# from sklearn.metrics import r2_score, mean_squared_error

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def absolute_deviation(y_true, y_pred):
    """
    This function calculates the absolute deviation between two values.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :return: absolute difference
    :rtype: np.ndarray
    """

    abs_diff = y_pred - y_true
    return abs_diff


def relative_deviation(y_true, y_pred):
    """
    This function calculates the relative deviation between two values.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :return: relative difference
    :rtype: np.ndarray
    """

    rel_diff = (y_pred - y_true) / y_pred
    return rel_diff


def relative_deviation_2prediction(y_true, y_pred):
    """
    This function calculates the relative deviation between two values in relation to the predicted value.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :return: relative difference
    :rtype: np.ndarray
    """

    rel_diff = (y_pred - y_true) / y_pred

    if isinstance(rel_diff, xr.DataArray):
        rel_diff.data[np.isnan(rel_diff.data)] = 0
    elif isinstance(rel_diff, np.ndarray):
        rel_diff[np.isnan(rel_diff)] = 0
    else:
        raise ValueError("input data types currently not supported.")

    return rel_diff


def se(y_true, y_pred, axis=-1):
    """
    This function calculates the squared error (SE).

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: squared error
    :rtype: np.ndarray
    """

    squared_error = np.sum((y_pred - y_true) ** 2, axis=axis)
    return squared_error


def me(y_true, y_pred, axis=-1):
    """
    This function calculates the mean error (ME).

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: mean error
    :rtype: np.ndarray
    """

    mean_error = np.mean(y_pred - y_true, axis=axis)
    return mean_error


def mne(y_true, y_pred, axis=-1):
    """
    This function calculates the mean normalized error (MNE).

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: mean normalized error
    :rtype: np.ndarray
    """

    mean_norm_error = np.mean((y_pred - y_true) / y_true, axis=axis)
    return mean_norm_error


def mae(y_true, y_pred, axis=-1):
    """
    This function calculates the mean absolute error (MAE).

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: mean absolute error
    :rtype: np.ndarray
    """

    mean_abs_error = np.mean(np.abs(y_pred - y_true), axis=axis)
    return mean_abs_error


def mane(y_true, y_pred, axis=-1):
    """
    This function calculates the mean absolute normalized error (MANE).

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: mean absolute normalized error
    :rtype: np.ndarray
    """

    mean_abs_norm_error = np.mean(np.abs(y_pred - y_true) / y_true, axis=axis)
    return mean_abs_norm_error


def rmse(y_true, y_pred, axis=-1):
    """
    This function calculates the root mean square error (RMSE).

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: root mean square error
    :rtype: np.ndarray
    """

    root_mean_square_error = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=axis))

    # sklearn provides a function to calculate the RMSE
    # root_mean_square_error = mean_squared_error(y_true, y_pred, squared=False)

    return root_mean_square_error


def nrmse(y_true, y_pred, normalization='rmsne', axis=-1):
    """
    This function calculates the normalized root mean square error (NRMSE).

    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param normalization: normalize by 'mean', 'range' or 'rmsne' (values) of the observed data
    :param int axis: (optional) along this axis the operation is performed
    :return: normalized root mean square error
    :rtype: np.ndarray
    """

    root_mean_square_error = rmse(y_true, y_pred, axis=axis)

    if normalization == 'mean':
        norm_rmse = root_mean_square_error / np.mean(y_true, axis=axis)
    elif normalization == 'range':
        norm_rmse = root_mean_square_error / (np.max(y_true, axis=axis) - np.min(y_true, axis=axis))
    elif normalization == 'rmsne':
        norm_rmse = np.sqrt(np.mean(((y_pred - y_true) / y_true) ** 2, axis=axis))
    else:
        raise ValueError("this normalization method for the NRMSE metric does not exist.")

    return norm_rmse


def r_squared(y_true, y_pred, axis=-1):
    """
    This function calculatest the coefficient of determination R^2.

    https://en.wikipedia.org/wiki/Coefficient_of_determination

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: coefficient of determination
    :rtype: np.ndarray
    """

    y_bar = np.mean(y_true, axis=axis)

    # regression (explained) sum os squares
    # ss_reg = np.sum((y_pred - y_bar)**2, axis=axis)

    # total sum of squares
    ss_tot = np.sum((y_true - y_bar)**2, axis=axis)

    # residual (error) sum of squares
    ss_res = np.sum((y_true - y_pred)**2, axis=axis)

    # r squared (general case)
    r_sq = 1 - ss_res / ss_tot

    # in the special case ss_res + ss_reg = ss_tot, the equation can be simplified
    # r_sq = ss_reg / ss_tot

    # sklearn.metrics contains a function r2_score with some additional options
    # r_sq = r2_score(y_true, y_pred)

    return r_sq


def correlation_coefficient(y_true, y_pred, axis=-1):
    """
    This function calculatest the correlation coefficient R.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: correlation coefficient
    :rtype: np.ndarray
    """

    corr_coef = 1 / (y_true.shape[axis] - 1) * np.sum((y_pred - np.mean(y_pred, axis=axis)) * (
            y_true - np.mean(y_true, axis=axis)) / (np.std(y_pred, axis=axis) * np.std(y_true, axis=axis)), axis=axis)

    # there is a numpy function that can even handle more than two inputs
    # cov_matrix = np.corrcoef(y_true, y_pred)
    # corr_coef = cov_matrix[1, 0]

    return corr_coef


def theils_u(y_true, y_pred, axis=-1):
    """
    This function calculates Theil's Inequality Coefficient and the Bias, Variance, and Covariance Proportion.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: Theil's Inequality Coefficient and the Bias, Variance and Covariance Proportion
    :rtype: np.ndarray
    """

    # Mean Squared Error
    mse = np.mean((y_pred - y_true) ** 2, axis=axis)
    root_mean_square_error = np.sqrt(mse)

    # Standard deviations
    y_true_std = np.std(y_true, axis=axis)
    y_pred_std = np.std(y_pred, axis=axis)

    # Correlation Coefficient
    corr_coef = correlation_coefficient(y_true, y_pred, axis=axis)

    # Theil's Bias Proportion
    u_m = (np.mean(y_pred, axis=axis) - np.mean(y_true, axis=axis)) ** 2 / mse

    # Theil's Variance Proportion
    u_s = (y_pred_std - y_true_std) ** 2 / mse

    # Theil's Covariance Proportion
    u_c = 2 * (1 - corr_coef) * y_pred_std * y_true_std / mse

    # Theil's Inequality Coefficient
    u = root_mean_square_error / (np.mean(y_pred ** 2, axis=axis) + np.mean(y_true ** 2, axis=axis))

    return u, u_m, u_s, u_c


def vm_oberkampf_2002(y_true, y_pred, axis=-1):
    """
    This function calculates a validation metric according to Oberkampf and Trucano.

    The theory can be found in [1, Eq. 16].

    Literature:
    [1] W. L. Oberkampf and T. G. Trucano, Verification and validation in computational fluid dynamics,
    In: Progress in Aerospace Sciences 38 (3), 2002, S. 209–272

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: (optional) along this axis the operation is performed
    :return: validation metric according to Oberkampf and Trucano
    :rtype: np.ndarray
    """

    vm = np.mean(np.tanh((y_pred - y_true) / y_true), axis=axis)

    # in the reference, the metric quantifies agreement, whereas our metrics quantify disagreement -> commented
    # vm = 1 - vm

    return vm
