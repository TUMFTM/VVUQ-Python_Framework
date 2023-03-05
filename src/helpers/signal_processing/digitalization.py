"""
This module offers helper functions for binary signals.

They are important for the data-driven assessment, e.g., in the UNECE-R79 use case.

It includes the following functions:
- pullup_glitches: sets short glitches of zeros in a binary signal to ones,
- get_event_boundaries: determines the start and stop indices of a signal level of One,
- select_longest_events: selects the longest event per test scenario.
They are mainly based on numpy arrays for fast vectorized operations.
See details in their respective documentations.

Contact person: Stefan Riedmaier
Creation date: 24.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np

# -- custom imports --
from src.helpers.general.numpy_indexing import slices_to_index_array


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def pullup_glitches(x, max_glitch_duration, axis=-1):
    """
    This function sets short glitches of zeros in a binary signal to ones.

    Examples:
    max_glitch_duration = 4
    x = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])
    ->  np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    x = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])
    ->  np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1])

    :param np.ndarray x: binary signal with glitches
    :param int max_glitch_duration: maximum number of samples to count as a short glitch
    :param int axis: (optional) axis parameter for multi-dimensional arrays
    :return: binary signal without glitches
    :rtype: np.ndarray
    """

    x = np.copy(x)

    bool_flag = False
    if x.dtype == 'bool':
        x = np.array(x, dtype=int)
        bool_flag = True

    if axis == -1:
        axis = x.ndim - 1

    # -- DETERMINE EDGES --

    # add leading and trailing Ones to the signals to uniformly start with falling and end with rising edges
    # (will be corrected at the end, otherwise different number of rising and falling edges possible)
    shape = x.shape[:axis] + (1,) + x.shape[axis + 1:]
    ones = np.ones(shape, dtype=int)
    y = np.concatenate((ones, x, ones), axis=axis)

    # subtract shifted signals to determine rising and falling edges
    edges = np.zeros(shape=y.shape, dtype=int)
    idx_list_wo_first = [slice(None, )] * edges.ndim
    idx_list_wo_last = [slice(None, )] * edges.ndim
    idx_list_wo_first[axis] = slice(1, None)
    idx_list_wo_last[axis] = slice(None, -1)
    edges[tuple(idx_list_wo_last)] = y[tuple(idx_list_wo_last)] - y[tuple(idx_list_wo_first)]

    # determine falling and rising edges
    is_falling_edge = edges == 1
    is_rising_edge = edges == -1

    # get the indices before falling and rising edges
    idx_falling_edges = np.where(is_falling_edge)
    idx_rising_edges = np.where(is_rising_edge)

    # correct the falling edges by removing those with index zero (those we added at the beginning for uniformity)
    # correct the rising edges by removing those with index "length" (those we added at the beginning for uniformity)
    idx_falling_edges_mask = idx_falling_edges[axis] != 0
    idx_rising_edges_mask = idx_rising_edges[axis] != x.shape[axis]
    idx_falling_edges = tuple(np.array(idx_falling_edges)[:, idx_falling_edges_mask & idx_rising_edges_mask])
    idx_rising_edges = tuple(np.array(idx_rising_edges)[:, idx_falling_edges_mask & idx_rising_edges_mask])

    # -- GLITCH HANDLING --

    # determine the lengths of the zero glitches
    len_glitch = idx_rising_edges[axis] - idx_falling_edges[axis]

    # check if the lengths falls below the maximum threshold
    is_short_glitch = (0 < len_glitch) & (len_glitch <= max_glitch_duration)

    # remove the indices of short glitches
    idx_falling_edges_short = tuple(np.array(idx_falling_edges)[:, is_short_glitch])
    idx_rising_edges_short = tuple(np.array(idx_rising_edges)[:, is_short_glitch])

    # remove short glitches
    _, x = slices_to_index_array(start_idx=idx_falling_edges_short, end_idx=idx_rising_edges_short, axis=axis, x=x,
                                 values=1)

    if bool_flag:
        x = np.array(x, dtype=bool)

    return x


def get_event_boundaries(x, min_length=1, axis=-1):
    """
    This function determines the start and stop indices of a signal level of One, provided it exceeds the min_length.

    It aims at getting the boundaries of true events.

    Examples:
    x = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    start_indices, stop_indices, event_length = digitalization.get_event_boundaries(x, min_length=1)
    start_indices_exp, stop_indices_exp, event_length_exp = (np.array([0]),), (np.array([5]),), np.array([5])

    :param np.ndarray x: binary signal
    :param min_length: (optional) minimum length to be considered an event
    :param int axis: (optional) axis parameter for multi-dimensional arrays
    :return: start (at first ones) and stop (after last ones) indices and event length
    :rtype: (tuple, tuple, np.ndarray)
    """

    if x.dtype == 'bool':
        x = np.array(x, dtype=int)

    if axis == -1:
        axis = x.ndim - 1

    # replace nan values with zeros (does not affect the Ones that are relevant for the event boundaries)
    nan_mask = np.isnan(x)
    if nan_mask.sum() > 0:
        x = x.copy()
        x[nan_mask] = 0
        x = np.array(x, dtype=int)

    # add leading and trailing zeros to the signals, not to lose the start and end bit (not covered by edges otherwise)
    shape = x.shape[:axis] + (1,) + x.shape[axis + 1:]
    zeros = np.zeros(shape, dtype=int)
    y = np.concatenate((zeros, x, zeros), axis=axis)

    # subtract shifted signals to determine rising and falling edges
    edges = np.zeros(shape=y.shape, dtype=int)
    idx_list_wo_first = [slice(None, )] * edges.ndim
    idx_list_wo_last = [slice(None, )] * edges.ndim
    idx_list_wo_first[axis] = slice(1, None)
    idx_list_wo_last[axis] = slice(None, -1)
    edges[tuple(idx_list_wo_last)] = y[tuple(idx_list_wo_last)] - y[tuple(idx_list_wo_first)]

    # determine falling and rising edges
    is_falling_edge = edges == 1
    is_rising_edge = edges == -1

    # get the indices before falling and rising edges
    idx_falling_edges = np.where(is_falling_edge)
    idx_rising_edges = np.where(is_rising_edge)

    # calculate the length of the events
    event_length = idx_falling_edges[axis] - idx_rising_edges[axis]

    # check whether the events are long enough
    len_mask = event_length >= min_length

    # apply the mask to the index array
    idx_falling_edges_long = tuple(np.array(idx_falling_edges)[:, len_mask])
    idx_rising_edges_long = tuple(np.array(idx_rising_edges)[:, len_mask])

    # the indices can directly be used as start and stop indices (slicing), due to padding the zeros at the beginning
    start_indices = idx_rising_edges_long
    stop_indices = idx_falling_edges_long

    return start_indices, stop_indices, event_length[len_mask]


def select_longest_events(start_idx, stop_idx, event_length, idx_time, scenario_shape):
    """
    This function selects the longest event per test scenario.

    Sometimes the experiments are planned to contain one event per test scenario / experiment, where the scenario
    conditions are met. However, it is possible that the condition check includes gaps and results in multiple events.
    Then, one option is to select only the longest event and discard the shorter ones, because they are not independent
    repetitions of a test scenario in multiple experiments.

    :param tuple start_idx: start indices
    :param tuple stop_idx: stop indices
    :param np.ndarray event_length: length of the events
    :param int idx_time: index of the time dimension in the index tuples
    :param tuple scenario_shape: shape of the scenario array
    :return: start indices, stop indices and event length of the longest events per test scenario
    :rtype: (tuple, tuple, np.ndarray)
    """

    # in case of degenerate emtpy arrays (no event), also return empty arrays
    if start_idx[0].shape == (0,) or stop_idx[0].shape == (0,) or event_length.shape == (0,):
        return (np.array([], dtype=int),), (np.array([], dtype=int),), np.array([], dtype=int)

    # determine the unique events, since we want just one LKFT event per curve (test scenario)
    base_idx = np.array(start_idx[:idx_time] + stop_idx[idx_time + 1:])
    idx_unique, counts = np.unique(base_idx, axis=1, return_counts=True)

    # create an array with the non-time dimensions times the maximum number of events in one scenario
    max_count = np.max(counts)
    shape = scenario_shape[:idx_time] + (max_count,) + scenario_shape[idx_time + 1:]
    event_length_array = np.zeros(shape=shape, dtype=int)

    # fill the array with the event length at the correct positions
    start_idx_z = start_idx[:idx_time] + (np.zeros(len(start_idx[0]), dtype=int),) + start_idx[idx_time + 1:]
    _, event_length_array = slices_to_index_array(start_idx_z, length=counts, axis=idx_time, x=event_length_array,
                                                  values=event_length, repeats=False)

    # -- select the longest event per test scenario
    longest_events_idx = np.argmax(event_length_array, axis=idx_time)

    # extract the np.max value from the argmax result without having to calling np.max additionaly
    shape = event_length_array.shape[:idx_time] + event_length_array.shape[idx_time + 1:]
    dim_idx = list(np.ix_(*[np.arange(i) for i in shape]))
    dim_idx.append(longest_events_idx)
    longest_events = event_length_array[tuple(dim_idx)]
    # longest_events = np.max(event_length_array, axis=idx_time)

    # discard the events with zero length (from initializing the event_length_array)
    event_mask = longest_events != 0

    # determine the cumulative indices into the index arrays
    counts_cumsum = np.zeros(counts.shape, dtype=int)
    counts_cumsum[1:] = np.cumsum(counts)[:-1]
    longest_events_cum_idx = longest_events_idx[event_mask] + counts_cumsum

    # determine the new indices and length of the longest events
    start_idx_longest = tuple(np.array(start_idx)[:, longest_events_cum_idx])
    stop_idx_longest = tuple(np.array(stop_idx)[:, longest_events_cum_idx])
    event_length_longest = event_length[longest_events_cum_idx]
    # event_length_longest = longest_events[event_mask]

    return start_idx_longest, stop_idx_longest, event_length_longest
