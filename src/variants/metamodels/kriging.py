"""
This module is responsible for training and inference of gaussian processes (kriging).

It includes the two functions train_gp_model and infer_gp_model for training and inference.
See details in their own documentations.

Contact person: Stefan Riedmaier
Creation date: 04.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import sklearn.gaussian_process as gp

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def train_gp_model(x, y):
    """
    This function trains a gaussian process (GP).

    It is currently based on the following link using sklearn:
    https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319

    Alternatives:
    - GPy: often mentioned
    https://sheffieldml.github.io/GPy/
    - GPflow: GPy fork using tensorflow
    https://github.com/GPflow/GPflow
    - PyMC3: GP with theano
    https://docs.pymc.io/
    - excellent tutorial including numpy, sk-learn, GPflow, PyMC3
    https://blog.dominodatalab.com/fitting-gaussian-process-models-python/

    :param np.ndarray x: inputs of the training data set
    :param np.ndarray y: outputs of the training data set
    :return: trained gaussian process
    """

    # kernel selection
    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3))
    gp.kernels.RBF(10.0, (1e-3, 1e3))

    # model creation
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

    # model fitting
    model.fit(x, y)

    return model


def infer_gp_model(x, model):
    """
    This function predicts outputs with a gaussian process (GP) model for the inputs x.

    :param np.ndarray x: inputs of the test data set
    :param model: gaussian process (GP) model
    :return: predicted outputs of the test data set
    """

    y_pred, std = model.predict(x, return_std=True)

    return y_pred, std
