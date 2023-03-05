"""
This module is responsible for reading erg files from CarMaker.

It builds on the cmerg package from PyPI and https://github.com/danielhrisca/cmerg,
which parses an erg file efficiently using regular expressions.
This module extends the capabilities by adding further methods such as extracting only a subset of the quantities
and returning them as an array. This satisfies the conventions of the DataFileReader parent class.

This module includes a main class called ErgFileReader. See details in its documentation.
It requires two files with the extensions .erg and .erg.info. The former contains the data, the latter the meta-data.

Contact person: Stefan Riedmaier
Creation date: 23.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
from collections import OrderedDict

# -- third-party imports --
import numpy as np
import cmerg

# -- custom imports --
from src.helpers.parsers.DataFileReader import DataFileReader


# -- CLASSES -----------------------------------------------------------------------------------------------------------

class ErgFileReader(DataFileReader):
    """
    This child class is responsible for reading erg files from CarMaker.

    It is based on the cmerg package and its regular expressions.

    It includes basic functions for reading erg files, extracting quantities and returning different data structures.
    See the respective methods for more details.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        # call the generic init method of the base class
        super().__init__()

        # -- specific instantiations
        self.cmerg = None
        self.ergsignals = OrderedDict()

    def select_quantities(self, quantity_name_list=None):
        """
        This function selects and prepares the desired quantities.

        The order matches the list of desired quantities.
        In case no list is provided, all measured quantities are returned.

        :param list[str] quantity_name_list: list of quantity names
        :return: dict of selected quantity objects
        :rtype: dict[Signal]
        """

        self.quantity_dict = dict()

        if quantity_name_list is None:
            # if no quantity selection is given, iterate through all quantities from the data file
            for quantity_name in self.ergsignals:
                # process the selected quantity and store the internal data structure ("Signal") in a dict
                self.quantity_dict[quantity_name] = self.cmerg.get(quantity_name)
        else:
            # iterate through the selected quantities
            for quantity_name in quantity_name_list:
                if quantity_name in self.ergsignals:
                    self.quantity_dict[quantity_name] = self.cmerg.get(quantity_name)
                else:
                    raise ValueError("desired quantity not stored")

        return self.quantity_dict

    def get_quantity_array(self):
        """
        This function creates a 2d array of the previously selected quantities.

        The array has one row per desired quantity and one column per time step.

        :return: array of quantity values
        :rtype: np.ndarray
        """

        # convert from dict of "Signal" to list of 1D samples array to 2D array
        quantity_samples_list = [quantity_signal.samples for quantity_signal in self.quantity_dict.values()]
        quantity_array = np.array(quantity_samples_list)

        return quantity_array

    def get_units(self):
        """
        This function returns the units of the desired quantities.

        :return: units of the desired quantities
        :rtype: list[str]
        """

        unit_list = [quantity_signal.unit for quantity_signal in self.quantity_dict.values()]
        return unit_list

    def read_data_file(self, erg_file_path):
        """
        This function parses an erg file, extracts the quantity values and adds them to a dictionary.

        The function calls the function get_erg_quantity_names internally to get the quantity names.

        :param str erg_file_path: path to the erg file
        :return: ordered dictionary of all erg quantity objects with names, data types and values
        :rtype: OrderedDict
        """

        # use the cmerg package to read an erg file based on regular expression
        self.cmerg = cmerg.ERG(erg_file_path)

        # extract an ordered dict of "ERGSignals" as internal data structure of the init method of the cmerg package
        self.ergsignals = self.cmerg.signals

        return self.ergsignals
