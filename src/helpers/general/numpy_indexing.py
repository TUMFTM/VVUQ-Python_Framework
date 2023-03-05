"""
This module contains general helper functions.

It includes the following methods:
- slices_to_index_array: convert multiple slices to an index array for vectorized integer indexing
- string_to_index: convert a string of indexing information to an actual indexing object

Contact person: Stefan Riedmaier
Creation date: 29.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def slices_to_index_array(start_idx, end_idx=(), length=np.empty(0), axis=-1, x=np.empty(0), values=0, repeats=True):
    """
    This function converts multiple slices to an index array for vectorized integer indexing at once.

    :param tuple start_idx: start indices
    :param tuple end_idx: (alternative to length) end indices
    :param np.ndarray length: (alternative to end_idx) length of the slices
    :param int axis: (optional) axis parameter to select one dimension from the index tuples for slicing
    :param np.ndarray x: (optional) array in which the values shall be inserted between the slice indices
    :param np.array values: (optional) values to be inserted between the slice indices
    :param bool repeats: (optional) flag whether the indices and values must be repeated times the "length"
    :return: integer index array, array filled with values
    :rtype: (tuple, np.ndarray)
    """

    if end_idx:
        # determine the length of each slice
        length = end_idx[axis] - start_idx[axis]
    elif length.shape != (0,):
        pass
    else:
        raise ValueError("either the end index or the length must be provided.")

    if repeats:
        # repeat the start indices times the slice length
        idx_slice = np.repeat(np.array(start_idx), length, axis=1)
    else:
        idx_slice = np.array(start_idx)

    # determine the indices between the slices
    idx_slice_axis = np.ones(length.sum(), dtype=np.int)
    idx_slice_axis[np.cumsum(length)[:-1]] -= length[:-1]
    idx_slice[axis, :] += np.cumsum(idx_slice_axis) - 1
    idx_slice = tuple(idx_slice)

    if x.shape != (0,):
        x = x.copy()

        # if values is just a scalar value, repeat it times the number of slices
        if isinstance(values, int):
            values = np.ones(len(length), dtype=int) * values
        elif isinstance(values, float):
            values = np.ones(len(length)) * values

        # fill the slices with the values
        if repeats:
            x[idx_slice] = np.repeat(values, length)
        else:
            x[idx_slice] = values

    return idx_slice, x


def string_to_index(s):
    """
    This function converts a string of indexing information to an actual indexing object.

    It is intended for config parsing where the user enters the indexing information in a string field.

    :param string s: indexing information
    :return: index object
    :rtype: int | slice | list
    """
    if s.lstrip('+-').isdigit():
        # case single integer index
        return int(s)

    elif len(s) > 7 and s[:6] == 'slice(' and s[-1] == ')':
        # case slice indexing

        # get the slice args
        idx_str_list = s[6:-1].split(',')

        if len(idx_str_list) > 3:
            raise ValueError("A slice cannot have more than three arguments.")

        # iterate through the slice args
        idx_list = list()
        for i in idx_str_list:
            # remove leading and trailing white spaces
            i = i.strip()

            if i == "None":
                # case slice None arg representing a default value
                idx_list.append(None)
            elif i.lstrip('+-').isdigit():
                # case slice integer arg
                idx_list.append(int(i))
            else:
                raise ValueError("A slice arg can either be None or int.")

        # create the slice object
        return slice(*idx_list)

    elif ',' in s:
        # split a comma-separated list
        idx_str_list = s.split(',')

        if all(i.strip().lstrip('+-').isdigit() for i in idx_str_list):
            # case integer indexing
            idx_list = [int(i) for i in idx_str_list]

        elif all(i.strip() in {'True', 'False'} for i in idx_str_list):
            # case boolean indexing
            idx_list = [i == "True" for i in idx_str_list]

        else:
            raise ValueError("Advanced indexing arrays can contain either only ints or only bools.")

        return idx_list

    else:
        raise ValueError("The string does not match a valid indexing method.")
