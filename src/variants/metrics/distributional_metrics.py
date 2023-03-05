"""
This module includes distributional validation metrics.

It includes several distributional metrics. See details in their own documentations.
The functions are mainly based on numpy arrays for fast vectorized operations.

Contact person: Stefan Riedmaier
Creation date: 03.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
from scipy import stats

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def compare_means_with_ci(y_true, y_pred, axis=-1):
    """
    This function implements the validation metric of comparing means including CIs according to Oberkampf.

    The theory can be found in [1, Ch. 12.4].

    Literature:
    [1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
    Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :param int axis: along this axis the operation is performed
    :return: confidence interval centered around error by comparing mean values
    :rtype: np.ndarray
    """

    prob = 0.9
    alpha = 1 - prob
    n = y_true.shape[axis]

    # calculate means and standard deviation
    y_true_mean = np.mean(y_true, axis=axis)
    y_true_std = np.std(y_true, axis=axis)
    y_pred_mean = np.mean(y_pred, axis=axis)

    # calculate the error between both means
    error = y_pred_mean - y_true_mean

    # get the inverse cdf of the t distribution evaluated at the prob. '1-alpha/2' for 'n-1' degrees of freedom
    # compare tinv(p, nu) in Matlab
    t_alpha = stats.t.ppf(1 - alpha / 2, df=n - 1)

    confidence_interval = t_alpha * y_true_std / (n ** 0.5)

    error_interval = [error - confidence_interval, error + confidence_interval]

    return error_interval


def ks_test(y_true, y_pred):
    """
    This function calculates the Kolmogrov-Smirnov (KS) test statistic on two samples.

    :param np.ndarray y_true: observed outputs
    :param np.ndarray y_pred: predicted outputs
    :return: KS statistic and p-value
    :rtype: np.ndarray
    """

    # calculate the KS statistic
    # the binary result must be determined with a significance level alpha separately
    (statistic, p_value) = stats.ks_2samp(y_true, y_pred)

    return statistic, p_value
