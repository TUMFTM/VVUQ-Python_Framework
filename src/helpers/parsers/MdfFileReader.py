"""
This module is responsible for reading mdf files.

It includes a main class called MdfFileReader. See details in its own documentation.

It builds on the asammdf package from PyPI and https://github.com/danielhrisca/asammdf.
This module extends the capabilities by adding further methods such as extracting only a subset of the quantities
and returning them as an array. This satisfies the conventions of the DataFileReader parent class.

asammdf 7 offers, e.g., the following methods:
1) MDF(path) provides a MDF file object
2) MDF(path, channels=channels_list) provides a MDF file object with the selected channels
3) MDF(path).filter(channels_list) provides a MDF file object with the selected channels
4) MDF(path).get() reads all raw samples and provides a list of Signal objects. It should not be called in a loop.
5) MDF(path).select(channels_list) provides a list of selected Signal objects.

We use the first method to instantiate the file objects.

We decided not to integrate the second method when it was added to asammdf, since it would contradict how we split
the methods of our DataFileReader classes. They state that it can lead to a "big speed improvement" but do not exactly
specify compared to what. It looks similar to the third method but might be faster due to one method instead of two.
Since they mention that it PRESERVES the selected METADATA, the effect might also be small compared to 3).
https://asammdf.readthedocs.io/en/master/tips.html#selective-channel-loading

We do not use the third method, since we want the selected channels as Signal object and not as MDF object.
We assume calling "select" directly is faster than calling it after "filter", but we have not tested it.

Most importantly, we do not use "get", since it is explicitly not recommended for several channels in a loop.

Instead, we use the "select" method to obtain Signal objects before converting them to numpy arrays.


Contact person: Stefan Riedmaier
Creation date: 18.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
from asammdf import MDF

# -- custom imports --
from src.helpers.parsers.DataFileReader import DataFileReader


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class MdfFileReader(DataFileReader):
    """
    This child class is responsible for reading mdf files.

    It includes basic functions for reading mdf files, extracting quantities and returning different data structures.
    See the respective methods for more details.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        # call the generic init method of the base class
        super().__init__()

        # -- specific instantiations
        self.mdf_file = None

    def select_quantities(self, quantity_name_list=None):
        """
        This function selects and prepares the desired quantities.

        The order matches the list of desired quantities.
        In case no quantity name list is provided, all measured quantities are returned.

        The MDF file contains channels and groups with a channel name, group index, and channel index.
        Thus, only providing a channel name can be ambiguous if it occurs multiple times.
        This is often the case for the time channel. Then, each other channel forms one group with the time in index 0
        and the channel data in index 1, respectively.

        If a quantity name list is provided, we currently assume no ambiguous quantity such as time.
        If a list is provided, we select the first occurrence of each quantity.

        If a channel does not exist, asammdf raises an MdfException error.

        :param list[str] quantity_name_list: list of quantity names
        :return: dict of selected quantity objects
        :rtype: dict
        """

        self.quantity_dict = dict()

        if quantity_name_list is None:
            # if no specific channel list is provided, take "all"

            # read the channel name, group index, and channel index from channels_db and convert it to the select format
            quantity_name_list = list(self.mdf_file.channels_db)
            select_list = [(ch_name,) + gr_ch_idx[0] for ch_name, gr_ch_idx in self.mdf_file.channels_db.items()]

            # get a list of all Signal objects
            quantity_list = self.mdf_file.select(select_list)

        else:
            # get a list of selected Signal objects
            quantity_list = self.mdf_file.select(quantity_name_list)

        # convert the list to a dict
        self.quantity_dict = dict(zip(quantity_name_list, quantity_list))

        return self.quantity_dict

    def get_quantity_array(self):
        """
        This function converts the list of values of the desired quantity objects to a 2d numpy array.

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

    def read_data_file(self, filepath):
        """
        This function calls the mdf method from asammdf to parse and load an mdf file.

        :param filepath: path to the mdf file
        :return:
        """

        self.mdf_file = MDF(filepath)
