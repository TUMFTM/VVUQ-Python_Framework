"""
This module is responsible for training and inference of neural networks.

It includes the two functions train_mlp_model and infer_mlp_model for training and inference of multi layer perceptrons.
In addition, it offers a custom loss function to learn an upper bound.
See details in their own documentations.

It is a new implementation of the research we have done in student thesis [1].

Thesis:
[1] J. Wang, „Maschinelles Lernen zur Metamodellierung von Fehlern in der Simulation
automatisierter Fahrzeuge,“ Master’s Thesis, Technical University of Munich, Munich,
Germany, 2020.

Contact person: Stefan Riedmaier
Creation date: 04.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def train_mlp_model(x, y):
    """
    This function trains a multi-layer perceptron (MLP).

    :param np.ndarray x: training input data
    :param np.ndarray y: training output data
    :return: trained multi-layer perceptron
    """

    # normalization
    # scale = MinMaxScaler()
    scale = StandardScaler(with_mean=True, with_std=True)
    xs = scale.fit_transform(x)

    # data splitting
    # percentage = 0.9
    # (train_x, test_x, train_y, test_y) = train_test_split(xs, y, test_size=percentage, random_state=0)

    # model creation
    hidden_layers = [128, 64, 32, 32, 0]
    model = Sequential()
    model.add(Dense(hidden_layers[0], input_dim=x.shape[1], activation="tanh", name="hidden1"))
    model.add(Dense(hidden_layers[1], activation="tanh", name="hidden2"))
    model.add(Dense(hidden_layers[2], activation="tanh", name="hidden3"))
    model.add(Dense(hidden_layers[3], activation="relu", name="hidden4"))
    # model.add(Dense(hidden_layers[4], activation="relu", name="hidden5"))
    model.add(Dense(1, activation="linear"))

    # model fitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    model.compile(loss=upper_bound_loss, optimizer='adam')
    model.fit(x, y, batch_size=32, epochs=200, verbose=0, validation_split=0.1, callbacks=[early_stopping])

    return model


def upper_bound_loss(y_true, y_pred):
    """
    This function defines a loss calculation to learn an upper bound on data.

    :param np.ndarray y_true: ground truth outputs
    :param np.ndarray y_pred: predicted outputs
    :return: loss value
    :rtype: float
    """

    # standard square-loss penalizing deviations from the GT equally in both directions
    ms_loss = K.mean(tf.square(y_pred - y_true), axis=-1)

    # tanh-loss penalizing deviations if the prediction is below the GT and favoring deviations if it is above the GT
    penalty = K.mean(tf.tanh(y_true - y_pred))

    # combine both losses with a weighting factor
    # if weight = 0: standard loss
    weight = 1
    loss = ms_loss + weight * penalty

    return loss


def infer_mlp_model(x, model):
    """
    This function predicts outputs of a multi-layer perceptron model for inputs x.

    :param np.ndarray x: test data set inputs
    :param model: multi-layer perceptron
    :return: predicted outputs on the test data set
    :rtype: np.ndarray
    """

    y = model.predict(x)

    # calculate loss and accuracy
    # score = model.evaluate(x, y.reshape(-1, 1), verbose=0)

    return y
