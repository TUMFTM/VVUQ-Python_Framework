"""
This module is responsible for training and inference of linear and polynomial regression models.

It includes linear and polynomial regression:
- It includes the two functions train_linear_regression_model and infer_linear_regression_model for training and
inference of linear regression models. It includes the two functions prepare_lr_prediction_interval and
calc_lr_prediction_interval for the corresponding calculation of external prediction intervals.
- It includes the two functions train_polynomial_regression_model and infer_polynomial_regression_model for training
and inference of polynomial regression models.
See details in their own documentations.

The theory of these non-simultaneous Bonferroni-type prediction intervals can be found, e.g., in [1, Sec. 5.5],
[2, Sec. 5.3], [3, Ch. 13.7.3], and [4, Ch. 3.2.1].

Literature:
[1] C. J. Roy and W. L. Oberkampf, „A comprehensive framework for verification, validation,
and uncertainty quantification in scientific computing,“ Computer Methods in Applied
Mechanics and Engineering, vol. 200, no. 25-28, pp. 2131–2144, 2011.
[2] C. J. Roy and M. S. Balch, „A Holistic Approach to Uncertainty Quantification with Application
to Supersonic Nozzle Thrust,“ International Journal for Uncertainty Quantification,
vol. 2, no. 4, pp. 363–381, 2012.
[3] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.
[4] R. G. Miller, Simultaneous Statistical Inference, New York, NY, Springer New York, 1981,
ISBN: 978-1-4613-8124-2.

Contact person: Stefan Riedmaier
Creation date: 07.09.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy import stats

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def train_linear_regression_model(x, y):
    """
    This function trains a multiple linear regression model.

    :param np.ndarray x: inputs
    :param np.ndarray y: outputs
    :return: trained linear regression model
    """

    # create the linear regression model
    model = LinearRegression()

    # train the linear regression model
    model.fit(x, y)

    return model


def prepare_lr_prediction_interval(x, y, model, alpha=0.95):
    """
    This function calculates the design array, mean squared error and t-value of the linear regression model.

    The three of them are required for the subsequent calculation of external prediction intervals.

    :param np.ndarray x: inputs
    :param np.ndarray y: ouputs
    :param model: trained linear regression model
    :param float alpha: (optional) level of significance (probability of making type I error)
    :return: design array, mean squared error and t-value
    :rtype: tuple(np.ndarray, float, float)
    """

    # design_array (Design Matix) Add column with 1 to represent the constant coefficiant
    design_array = np.zeros((x.shape[0], x.shape[1] + 1))
    design_array[:, 0] = 1
    design_array[:, 1:x.shape[0]] = x

    # the number of model coeficients determine the degree of freedom (dof)
    dof = design_array.shape[1]

    # determine N (number validation experiments) - d (dof)
    number_validation_experiment = y.shape[0]
    n_minus_d = number_validation_experiment - dof

    # Mean Squared Error
    y_prediction = model.predict(x)
    mse = (sum((y - y_prediction) ** 2) / n_minus_d) ** (1 / 2)

    # Stud. t-distribution (two-sided interval is used)
    t_value = stats.t.ppf(1 - (alpha / 2), n_minus_d)

    return design_array, mse, t_value


def infer_linear_regression_model(x, model):
    """
    This function predicts outputs using the linear regression model.

    :param np.ndarray x: input data
    :param model: trained linear regression model
    :return: predicted outputs
    :rtype: np.ndarray
    """

    y = model.predict(x)
    return y


def calc_prediction_interval(x, design_array, mse, t_value):
    """
    This function calculates external prediction intervals for a linear and polynomial regression.

    :param np.ndarray x: input array for prediction
    :param np.ndarray design_array: input array from training, extended to linear model coefficients
    :param float mse: mean squared error
    :param float t_value: t-value from t-distribution
    :return: prediction interval
    :rtype: np.ndarray
    """

    # extend x by a leading column of ones
    x_ext = np.ones((x.shape[0], x.shape[1] + 1))
    x_ext[:, 1:] = x

    # calculate (X^T * X)^-1
    design_array_inverse = np.linalg.inv(np.dot(design_array.T, design_array))

    # calculate the prediction interval
    prediction_interval = \
        t_value * mse * (1 + np.diag(np.linalg.multi_dot([x_ext, design_array_inverse, x_ext.T]))) ** (1 / 2)

    return prediction_interval


def train_polynomial_regression_model(x, y, degree=2):
    """
    This function trains a polynomial regression model using sklearn.

    :param np.ndarray x: input array
    :param np.ndarray y: output array
    :param int degree: (optional) degree of the polynomial
    :return: polynomial regression model
    """

    # create a polynomial transformer instance to transform the input data to include higher terms
    transformer = PolynomialFeatures(degree=degree, include_bias=False)

    # fit the transformer to the input data
    transformer.fit(x)

    # transform the input data using the fitted transformer
    x_ = transformer.transform(x)

    # as a one-liner
    # x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

    # create and train the polynomial model using the linear regression method with the transformed data
    model = LinearRegression()
    model.fit(x_, y)

    return model


def prepare_pr_prediction_interval(x, y, model, degree, alpha=0.95):
    """
    This function calculates the design array, mean squared error and t-value of the polynomial model.

    The three of them are required for the subsequent calculation of external prediction intervals.

    :param np.ndarray x: inputs
    :param np.ndarray y: ouputs
    :param model: trained polynomial regression model
    :param degree: degree of the polynomial model
    :param float alpha: (optional) level of significance (probability of making type I error)
    :return: design array, mean squared error and t-value
    :rtype: tuple(np.ndarray, float, float)
    """

    # design_array (Design Matix) Add column with 1 to represent the constant coefficiant
    design_array = np.zeros((x.shape[0], x.shape[1] + 1))
    design_array[:, 0] = 1
    design_array[:, 1:x.shape[0]] = x

    # the number of model coeficients determine the degree of freedom (dof)
    dof = design_array.shape[1]

    # determine N (number validation experiments) - d (dof)
    number_validation_experiment = y.shape[0]
    n_minus_d = number_validation_experiment - dof

    # Mean Squared Error
    y_prediction = infer_polynomial_regression_model(x, model, degree)
    mse = (sum((y - y_prediction) ** 2) / n_minus_d) ** (1 / 2)

    # Stud. t-distribution (two-sided interval is used)
    t_value = stats.t.ppf(1 - (alpha / 2), n_minus_d)

    return design_array, mse, t_value


def infer_polynomial_regression_model(x, model, degree=2):
    """
    This function predicts outputs using the polynomial regression model.

    :param np.ndarray x: input data
    :param model: trained linear regression model
    :param degree: (optional) degree of the polynomial
    :return: predicted outputs
    :rtype: np.ndarray
    """

    # transform the input data
    x_ = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(x)

    # perform teh actual model prediction
    y = model.predict(x_)

    return y
