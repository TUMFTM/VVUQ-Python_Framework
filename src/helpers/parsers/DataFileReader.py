"""
This module is responsible for reading data files.

It includes one parent class for data file readers. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 22.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
from abc import ABCMeta, abstractmethod
import numpy as np
import xarray as xr

# -- custom imports --


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class DataFileReader:
    """
    This class is a parent class for data file readers.

    It includes basic functions for reading files and functions that must be overwritten (abstract methods).
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        self.quantity_dict = dict()

    def read_multiple_files(self, filepath_list, quantity_name_list, scenarios_da):
        """
        This function reads multiple erg result files and return them as a multi-dimensional xarray.

        :param list[str] filepath_list: list of absolute erg file paths for each scenario
        :param list[str] quantity_name_list: list of quantity names
        :param xr.DataArray scenarios_da: data array of scenarios
        :return: result data array
        :rtype: xr.DataArray
        """

        # read the results for all simulations
        array_list = []
        for filepath in filepath_list:
            # read the measurement file
            self.read_data_file(filepath)

            # extract the desired quantities from the read file
            self.select_quantities(quantity_name_list)

            # get the quantities as a 2d array: number_quantities times number_timesteps
            quantity_array = self.get_quantity_array()

            # append the results of each simulation in a list
            array_list.append(quantity_array)

        number_quantities = array_list[0].shape[0]

        # get the units of the desired quantities
        unit_list = self.get_units()

        # use the dimensions of the scenario array, except the 'parameters'-dimension
        idx = scenarios_da.dims.index('parameters')
        sc_shape = scenarios_da.shape[:idx] + scenarios_da.shape[idx + 1:]

        # check if all arrays in the list have the same number of time steps
        if all(array_list[i].shape[1] == array_list[0].shape[1] for i in range(len(array_list))):
            # same length arrays can be concatenated directly
            result_array_2d = np.concatenate(array_list, axis=1)

            # reshape to quantities times space (times epistemic times aleatory) times fixed number time steps
            result_array = result_array_2d.reshape((number_quantities, *sc_shape, array_list[0].shape[1]))
        else:

            # get the number of time steps of each time signal
            duration_array = np.array([array_element.shape[1] for array_element in array_list])

            # determine the number of time steps of the longest time signal
            max_duration = duration_array.max()

            # create a boolean array: True during each actual time value, False to append to the maximum duration
            mask_array = np.arange(max_duration) < duration_array[:, None]

            # extend the boolean array to all quantities (number quantities x space/epistemic/aleatory x max_timesteps)
            mask_array = np.tile(mask_array[None, :, :], (number_quantities, 1, 1))

            # create an array of nan values and insert the actual data at the True locations
            result_array_3d = np.ones(mask_array.shape) * np.nan
            result_array_3d[mask_array] = np.concatenate(array_list, axis=1).reshape((-1))

            # check whether the scenario array contains nan values
            if np.isnan(scenarios_da.data).sum():
                # -- case experimental repetitions with variable number

                # create a boolean array: True during actual repetitions, False to append to the maximum repetition
                mask_array = ~np.isnan(scenarios_da.data[:, :, 0])

                # extend the boolean array to all quantities and to all timesteps
                sc_shape_ones = (1,) * len(sc_shape)
                mask_array = np.tile(mask_array[None, ..., None], (number_quantities, *sc_shape_ones, max_duration))

                # create an array of nan values and insert the actual data at the True locations
                result_array = np.ones((number_quantities, *sc_shape, max_duration)) * np.nan
                result_array[mask_array] = result_array_3d.flatten()

            else:
                # reshape to quantities times space (times epistemic times aleatory) times max. number time steps
                result_array = result_array_3d.reshape((number_quantities, *sc_shape, max_duration))

            # create 2d version for csv handling
            result_array_2d = result_array.reshape((number_quantities, -1))

        # -- create xarray
        # use the dimensions of the scenario array, except the 'parameters'-dimension, and add 'quantities', 'timesteps'
        dims = ('quantities',) + scenarios_da.dims[:idx] + scenarios_da.dims[idx + 1:] + ('timesteps',)

        # create the data array with the dimensions and the quantitiy names as coordinates
        result_da = xr.DataArray(result_array, dims=dims, coords={'quantities': quantity_name_list})

        # store important metadata
        result_da.attrs['array2d'] = result_array_2d

        # store the units in the coordinates xarray attributes dictionary
        result_da.quantities.attrs['units'] = unit_list

        return result_da

    @abstractmethod
    def get_quantity_array(self):
        """
        Abstract method that must be overwritten by the child class.
        """
        raise NotImplementedError("method must be overwritten by child class")

    @abstractmethod
    def get_units(self):
        """
        Abstract method that must be overwritten by the child class.
        """
        raise NotImplementedError("method must be overwritten by child class")

    @abstractmethod
    def select_quantities(self, quantity_name_list=None):
        """
        Abstract method that must be overwritten by the child class.
        """
        raise NotImplementedError("method must be overwritten by child class")

    @abstractmethod
    def read_data_file(self, filepath):
        """
        Abstract method that must be overwritten by the child class.
        """
        raise NotImplementedError("method must be overwritten by child class")
