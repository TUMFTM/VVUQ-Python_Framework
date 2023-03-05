"""
This module is responsible for the model calibration process.

It includes one class for the model calibration. See details in its own documentation.
However, the class is only added as a placeholder for the future and not yet filled with actual calibration methods.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""


class Calibration:
    """
    This class is responsible for the model calibration process.

    It includes a main method called "process". See details in its own documentation.
    """

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """

        self.config = config

        if self.config['calibration']['method'] != "none":
            raise NotImplementedError("Model calibration is currently not implemented.")

        return

    def process(self):
        """
        This method runs through each step of the model calibration process.
        """

        if self.config['calibration']['method'] != "none":
            raise NotImplementedError("Model calibration is currently not implemented.")

        return
