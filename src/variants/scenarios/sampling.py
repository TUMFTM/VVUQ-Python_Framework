"""
This module includes sampling methods.

It includes several sampling techniques:
- mcs_uniform: uniform Monte Carlo sampling,
- mcs_gaussian: gaussian Monte Carlo sampling,
- lhs: Latin Hypercube sampling,
- full_factorial_doe: full factorial design of experiments (with all permutations),
- cartesian_product: low-level function to get permutations.
See details in their own documentations.

Contact person: Stefan Riedmaier
Creation date: 08.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def mcs_uniform(min_list, max_list, number_samples):
    """
    This function generates monte carlo samples (MCS) from a uniform distribution within the min-max-ranges.

    :param list[float] min_list: lower limits of all parameters
    :param list[float] max_list: upper limits of all parameters
    :param int number_samples: number of samples (same for all parameters)
    :return: monte carlo samples
    :rtype: np.ndarray
    """

    number_params = len(min_list)

    # generate monte carlo samples
    samples = np.random.uniform(size=[number_samples, number_params])
    for i in range(number_params):
        # scale the samples to a range between min and max
        samples[:, i] = min_list[i] + (max_list[i] - min_list[i]) * samples[:, i]

    return samples


def mcs_gaussian(mu_list, sigma_list, number_samples):
    """
    This function generates monte carlo samples (MCS) from a normal distribution.

    :param list[float] mu_list: expected value of all parameters
    :param list[float] sigma_list: standard deviation of all parameters
    :param int number_samples: number of samples (same for all parameters)
    :return: monte carlo samples
    :rtype: np.ndarray
    """

    # distinguish between a fixed number of samples for all parameters and a variable one for each normal distribution
    if isinstance(number_samples, int):
        # create samples for fixed number
        aleatory_samples = [np.random.normal(mu, sigma, number_samples) for (mu, sigma) in zip(mu_list, sigma_list)]
    elif isinstance(number_samples, list):
        # create samples for variable number
        aleatory_samples = [np.random.normal(mu, sigma, num_samples) for (mu, sigma, num_samples) in zip(
            mu_list, sigma_list, number_samples)]
    else:
        raise ValueError("argument number_samples must be either an int value or a list of int values.")

    # convert the list to a numpy array and transpose it
    aleatory_samples = np.asarray(aleatory_samples).T

    return aleatory_samples


def lhs(min_list, max_list, number_samples):
    """
    This function generates latin hypercube samples (LHS).

    Within each cell, the lowest sample starts at its lower cell boundary and the highest sample ends at its upper
    cell boundary. The samples in between move within their cell in an equidistant grid.
    There are also other options such as centering the samples within their cell, etc.

    :param list[float] min_list: lower limits of all parameters
    :param list[float] max_list: upper limits of all parameters
    :param int number_samples: number of samples (same for all parameters)
    :return: latin hypercube samples
    :rtype: np.ndarray
    """

    number_params = len(min_list)

    # generate distinct samples by sorting the arguments of uniform random numbers
    samples = np.random.uniform(size=[number_samples, number_params])
    for i in range(0, number_params):
        samples[:, i] = np.argsort(samples[:, i])
        # scale the samples to a range between min and max
        samples[:, i] = min_list[i] + (max_list[i] - min_list[i]) / (number_samples - 1) * samples[:, i]

    return samples


def full_factorial_doe(min_list, max_list, number_samples):
    """
    This function generates a full factorial design of experiments.

    :param list[float] min_list: lower limits of all parameters
    :param list[float] max_list: upper limits of all parameters
    :param list[int] number_samples: number of samples of all parameters
    :return: full factorial samples
    :rtype: np.ndarray
    """

    samples_1d_list = []
    # generate the equidistant samples for each parameter
    for (min_value, max_value, num) in zip(min_list, max_list, number_samples):
        samples_1d = np.linspace(min_value, max_value, num)
        samples_1d_list.append(samples_1d)

    samples = cartesian_product(*samples_1d_list)

    return samples


def cartesian_product(*arrays):
    """
    This function calculates a cartesian product.

    The code is based on the following stack overflow article:
    https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points

    :param arrays: multiple 1d arrays
    :return: 2d array with the cartesian product of all 1d arrays
    :rtype: np.ndarray
    """
    return np.stack(np.meshgrid(*arrays, indexing='ij'), axis=-1).reshape(-1, len(arrays))
