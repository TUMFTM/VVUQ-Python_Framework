"""
This module is responsible for the overall VV&UQ process.

It includes one class for VVUQ (Verification, Validation and Uncertainty Quantification).
See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --

# -- custom imports --
from src.domains.Verification import Verification
from src.domains.Calibration import Calibration
from src.domains.Validation import Validation
from src.domains.Application import Application


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class VVUQ:
    """
    This class is responsible for the overall VV&UQ process.

    It includes a main method called "process". See details in its own documentation.
    """

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """

        # -- INSTANTIATE VVUQ OBJECTS ----------------------------------------------------------------------------------
        self.verification = Verification(config)
        self.calibration = Calibration(config)
        self.validation = Validation(config)
        self.application = Application(config)

    def process(self):
        """
        This method runs through each step of the VV&UQ process.

        It contains the following steps:
        1) Model verification to determine the numerical error / uncertainty.
        2) Model calibration to infer model parameters and to determine parametric uncertainties.
        3) Model validation to assess the model quality and to determine the model-form error / uncertainty.
        4) Model application to perform prediction with the model in its intended use case.

        :return:
        """

        # Model Verification
        numerical_uncertainty_da = self.verification.process()

        # Model Calibration
        self.calibration.process()

        # Model Validation
        error_model_validation = self.validation.process()

        # Model Prediction
        self.application.process(error_model_validation, numerical_uncertainty_da)

        return
