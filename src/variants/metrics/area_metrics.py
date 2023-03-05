"""
This module includes area validation metrics.

It includes several types of area validation metrics. See details in their own documentations.
The functions are mainly based on numpy arrays for fast vectorized operations.

The general theory can be found, e.g., in [1, Ch. 12.8.2].

Literature:
[1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.

Contact person: Stefan Riedmaier
Creation date: 03.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np

# -- custom imports --


# -- FUNCTIONS ---------------------------------------------------------------------------------------------------------

def avm(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, axis=-1):
    """
    This function calculates the area validation metric (AVM).

    The theory can be found, e.g., in [1, Ch. 12.8.2].

    Literature:
    [1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
    Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.

    :param np.ndarray pbox_y_model: y vector of the model
    :param np.ndarray pbox_y_system: y vector of the system
    :param list[np.ndarray, np.ndarray] pbox_x_model_list: 2-element list with left and right x vector of the model
    :param list[np.ndarray, np.ndarray] pbox_x_system_list: 2-element list with left and right x vector of the system
    :param int axis: (optional) along this axis the operation is performed
    :return: area validation metric
    :rtpye: np.ndarray
    """

    (area_left, area_right, _, _) = calc_areas(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list,
                                               axis=axis)

    # use the sum of the left and right area as the metric / both interval boundaries
    area_total = area_left + area_right
    metric = np.stack((area_total, area_total), axis=axis)

    return metric


def mavm(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, axis=-1, f0=4.0, f1=1.25):
    """
    This function calculates the modified area validation metric (MAVM).

    Literature:
    I. T. Voyles and C. J. Roy, „Evaluation of Model Validation Techniques in the Presence of
    Aleatory and Epistemic Input Uncertainties,“ in 17th AIAA Non-Deterministic Approaches
    Conference, 2015, ISBN: 978-1-62410-347-6.

    :param np.ndarray pbox_y_model: y vector of the model
    :param np.ndarray pbox_y_system: y vector of the system
    :param list[np.ndarray, np.ndarray] pbox_x_model_list: 2-element list with left and right x vector of the model
    :param list[np.ndarray, np.ndarray] pbox_x_system_list: 2-element list with left and right x vector of the system
    :param int axis: (optional) along this axis the operation is performed
    :param float f0: (optional) f0 value of the safety factor fs
    :param float f1: (optional) f1 value of the safety factor fs
    :return: modified area validation metric
    :rtpye: np.ndarray
    """

    (area_left, area_right, _, _) = calc_areas(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list,
                                               axis=axis)

    # calculate the safety factor fs
    n = len(pbox_y_model)
    fs = f1 + 1.2 * (f0 - f1) / (n ** (1 / 3))

    # calculate the actual mavm metric
    term1 = (area_right - area_left) / 2
    term2 = fs * (area_right + area_left) / 2
    mavm_left = abs(term1 - term2)
    mavm_right = abs(term1 + term2)
    metric = np.stack((mavm_left, mavm_right), axis=axis)

    return metric


def iavm(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, axis=-1):
    """
    This function calculates the interval area validation metric (IAVM).

    d_upper_plus / area_right:      system upper limit function / left pbox edge  > model ECDF / right pbox edge
    d_upper_minus / area_left_wc:   system upper limit function / left pbox edge  < model ECDF / left pbox edge
    d_lower_plus / area_right_wc:   system lower limit function / right pbox edge > model ECDF / right pbox edge
    d_lower_minus / area_left:      system lower limit function / right pbox edge < model ECDF / left pbox edge

    Literature:
    N. Wang, W. Yao, Y. Zhao, X. Chen, X. Zhang and L. Li, „A New Interval Area Metric for
    Model Validation With Limited Experimental Data,“ Journal of Mechanical Design, vol.
    140, no. 6, 2018.

    :param np.ndarray pbox_y_model: y vector of the model
    :param np.ndarray pbox_y_system: y vector of the system
    :param list[np.ndarray, np.ndarray] pbox_x_model_list: 2-element list with left and right x vector of the model
    :param list[np.ndarray, np.ndarray] pbox_x_system_list: 2-element list with left and right x vector of the system
    :param int axis: (optional) along this axis the operation is performed
    :return: interval area validation metric
    :rtpye: np.ndarray
    """

    (area_left, area_right, area_left_wc, area_right_wc) = calc_areas(pbox_y_model, pbox_y_system, pbox_x_model_list,
                                                                      pbox_x_system_list, axis=axis)

    # init metric arrays
    metric_left = np.zeros(shape=area_left.shape)
    metric_right = np.zeros(shape=area_left.shape)

    # case 1 according to paper
    is_dum_zero = area_left_wc == 0
    metric_left[is_dum_zero] = area_right[is_dum_zero]
    metric_right[is_dum_zero] = area_right_wc[is_dum_zero]

    # case 2 according to paper
    is_dlp_zero = area_right_wc == 0
    metric_left[is_dlp_zero] = area_left[is_dlp_zero]
    metric_right[is_dlp_zero] = area_left_wc[is_dlp_zero]

    # case 3 according to paper
    is_nonzero = (area_left_wc > 0) & (area_right_wc > 0)
    metric_left[is_nonzero] = area_right[is_nonzero] + area_left[is_nonzero]
    metric_right[is_nonzero] = metric_left[is_nonzero] + np.maximum(area_left_wc[is_nonzero] - area_left[is_nonzero],
                                                                    area_right_wc[is_nonzero] - area_right[is_nonzero])
    # combine the left and right metric values to intervals
    metric = np.stack((metric_left, metric_right), axis=-1)

    return metric


def calc_areas(pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, axis=-1):
    """
    This function determines the different areas.

    It calls the calc_left_area and calc_right_area functions to determine the areas.

    :param np.ndarray pbox_y_model: y vector of the model
    :param np.ndarray pbox_y_system: y vector of the system
    :param list[np.ndarray, np.ndarray] pbox_x_model_list: 2-element list with left and right x vector of the model
    :param list[np.ndarray, np.ndarray] pbox_x_system_list: 2-element list with left and right x vector of the system
    :param int axis: (optional) along this axis the operation is performed
    :return: arrays of left, right, worst case left and worst case right areas
    :rtype: tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    """

    if isinstance(pbox_y_model, np.ndarray) and isinstance(pbox_y_system, np.ndarray):
        # -- case: equal number of cdf steps in 1D arrays pbox_y_model and pbox_y_system

        # merge the probability vectors
        y_unique, pbox_model_merged, pbox_system_merged = merge_ecdf_functions(
            pbox_y_model, pbox_y_system, pbox_x_model_list, pbox_x_system_list, axis=axis)

        # calculate the area where the right system pbox edge is on the left of the left model pbox edge
        area_left = calc_left_area(y_unique, pbox_model_merged[0], pbox_system_merged[1], axis=axis)

        # calculate the area where the left system pbox edge is on the right of the right model pbox edge
        area_right = calc_right_area(y_unique, pbox_model_merged[1], pbox_system_merged[0], axis=axis)

        # calculate the worst case area where the left system pbox edge is on the left of the left model pbox edge
        area_left_wc = calc_left_area(y_unique, pbox_model_merged[0], pbox_system_merged[0], axis=axis)

        # calc the worst case area where the right system pbox edge is on the right of the right model pbox edge
        area_right_wc = calc_right_area(y_unique, pbox_model_merged[1], pbox_system_merged[1], axis=axis)

    elif isinstance(pbox_y_model, list) and isinstance(pbox_y_system, list):
        # -- case: variable number of cdf steps in list of 1D arrays pbox_y_model and pbox_y_system

        # we assume two (equal) arrays in pbox_x_model/system_list with the dims: qois x space_samples x repetitions
        # and we assume number space_samples list elements in pbox_y_model/system
        sh = pbox_x_model_list[0].shape[:-1]

        # init arrays
        area_left, area_right, area_left_wc, area_right_wc = np.empty(sh), np.empty(sh), np.empty(sh), np.empty(sh)

        # loop through the space samples
        for sp_idx in range(len(pbox_y_model)):

            # extract one space samples from the x arrays
            sp_pbox_x_model_list = [x[:, sp_idx, :] for x in pbox_x_model_list]
            sp_pbox_x_system_list = [x[:, sp_idx, :] for x in pbox_x_system_list]

            # exclude the nan values
            sp_pbox_x_model_list = [x[~np.isnan(x)].reshape((x.shape[0], -1)) for x in sp_pbox_x_model_list]
            sp_pbox_x_system_list = [x[~np.isnan(x)].reshape((x.shape[0], -1)) for x in sp_pbox_x_system_list]

            # merge the probability vectors
            sp_y_unique, sp_pbox_model_merged, sp_pbox_system_merged = merge_ecdf_functions(
                pbox_y_model[sp_idx], pbox_y_system[sp_idx], sp_pbox_x_model_list, sp_pbox_x_system_list, axis=axis - 1)

            # calculate the area where the right system pbox edge is on the left of the left model pbox edge
            area_left[:, sp_idx] = calc_left_area(
                sp_y_unique, sp_pbox_model_merged[0], sp_pbox_system_merged[1], axis=axis - 1)

            # calculate the area where the left system pbox edge is on the right of the right model pbox edge
            area_right[:, sp_idx] = calc_right_area(
                sp_y_unique, sp_pbox_model_merged[1], sp_pbox_system_merged[0], axis=axis - 1)

            # calculate the worst case area where the left system pbox edge is on the left of the left model pbox edge
            area_left_wc[:, sp_idx] = calc_left_area(
                sp_y_unique, sp_pbox_model_merged[0], sp_pbox_system_merged[0], axis=axis - 1)

            # calc the worst case area where the right system pbox edge is on the right of the right model pbox edge
            area_right_wc[:, sp_idx] = calc_right_area(
                sp_y_unique, sp_pbox_model_merged[1], sp_pbox_system_merged[1], axis=axis - 1)

    else:
        raise ValueError("different data type of arguments pbox_y_model and pbox_y_system expected.")

    return area_left, area_right, area_left_wc, area_right_wc


def calc_left_area(y, x1, x2, axis=-1):
    """
    This function calculates the areas where x2 is on the left side of x1.

    The areas can be interpreted as rectangles.

    :param np.ndarray y: joint y vector
    :param np.ndarray x1: first x vector
    :param np.ndarray x2: second x vector
    :param int axis: (optional) along this axis the operation is performed
    :return: calculated areas
    :rtype: np.ndarray
    """

    # get the indices where x2 is on the left of x1
    idx = np.nonzero(x1 > x2)

    # calculate the area of each rectangle on the left side (one rectangle refers to one step of the ecdfs)
    area_1d = (x1[idx] - x2[idx]) * (y[idx[axis]] - y[idx[axis] - 1])

    # put the areas into the orignal shape of the x arrays
    area_nd = np.zeros(x1.shape)
    area_nd[idx] = area_1d

    # sum along the last dimension to get the area on the left between the whole ecdfs
    area = np.sum(area_nd, axis=axis)

    return area


def calc_right_area(y, x1, x2, axis=-1):
    """
    This function calculates the area where x2 is on the right side of x1.

    The areas can be interpreted as rectangles.

    :param np.ndarray y: joint y vector
    :param np.ndarray x1: first x vector
    :param np.ndarray x2: second x vector
    :param int axis: (optional) along this axis the operation is performed
    :return: calculated areas
    :rtype: np.ndarray
    """

    # get the indices where x2 is on the right of x1
    idx = np.nonzero(x1 < x2)

    # calculate the area of each rectangle on the right side (one rectangle refers to one step of the ecdfs)
    area_1d = (x2[idx] - x1[idx]) * (y[idx[axis]] - y[idx[axis] - 1])

    # put the areas into the orignal shape of the x arrays
    area_nd = np.zeros(x1.shape)
    area_nd[idx] = area_1d

    # sum along the last dimension to get the area on the right between the whole ecdfs
    area = np.sum(area_nd, axis=axis)

    return area


def merge_ecdf_functions(ay, by, ax_list=None, bx_list=None, axis=-1):
    """
    This function merges ecdf functions to get a joint y vector and the corresponding x vectors.

    This is important for the rectangular area calculations, since both ecdfs need to have the same number of steps.

    If the ax_list resp. the bx_list is provided, the resulting x vectors are returned.
    Otherwise, the index vectors are returned to index x vectors.

    :param np.ndarray ay: first y vector
    :param np.ndarray by: second y vector
    :param list[np.ndarray, np.ndarray] ax_list: (optional) list of first x vectors
    :param list[np.ndarray, np.ndarray] bx_list: (optional) list of second x vectors
    :param int axis: (optional) along this axis the operation is performed
    :return: joint y vector and the corresponding x vectors resp. the indices to index them
    :rtype: tuple(np.ndarray, list[np.ndarray], list[np.ndarray])
    """

    la = len(ay)
    lb = len(by)

    # equal length of the ecdfs means that ay and by are equal, then there is nothing to merge
    if la == lb:
        return ay, ax_list, bx_list

    # check the shapes between x and y, respectively
    if ax_list and not all(elem.shape[axis] == la for elem in ax_list):
        raise IndexError("shapes of ay and of at least one element in ax_list do not match")
    if bx_list and not all(elem.shape[axis] == lb for elem in bx_list):
        raise IndexError("shapes of by and of at least one element in bx_list do not match")

    # concatenate
    y_merged = np.concatenate((ay, by))

    # call unique to get a sorted array without duplicates and the indices of the first occurrences relating to a
    y_unique, a_idx = np.unique(y_merged, return_index=True)

    # call unique again with an inverted array to get the indices of the last occurences relating to b
    _, b_idx_inv = np.unique(y_merged[::-1], return_index=True)
    b_idx = len(y_merged) - b_idx_inv - 1

    # unfortunately, numpy does not offer to return the last indices directly
    # we could avoid calling it (and sorting) twice, but this would require internal modifications in numpy
    # one would have to add the following four code lines at the end of the return_index-if-clause in _unique1d:
    # mask_last = np.empty(aux.shape, dtype=np.bool_)
    # mask_last[:-1] = aux[1:] != aux[:-1]
    # mask_last[-1] = True
    # ret += (perm[mask_last],)

    # shift the indices in the original index space (before concatenation)
    b_idx = b_idx - la

    # replace the b-indices in a_idx (>= la) with the next right neighbor element coming from a (< la)
    for i in range(len(a_idx)):
        if a_idx[i] >= la:
            for j in range(i + 1, len(a_idx)):
                if a_idx[j] < la:
                    a_idx[i] = a_idx[j]
                    break

    # replace the a-indices in b_idx (<0) with the next right neighbor element coming from b (>0)
    for i in range(len(b_idx)):
        if b_idx[i] < 0:
            for j in range(i + 1, len(b_idx)):
                if b_idx[j] > 0:
                    b_idx[i] = b_idx[j]
                    break

    # prepare the return structure
    return_tuple = (y_unique,)

    # -- extend ax and bx to the new x values by using the determined indices

    if ax_list:
        ax_merged_list = []
        for ax in ax_list:
            idx_list = [slice(None,)] * ax.ndim
            idx_list[axis] = a_idx
            ax_merged_list.append(ax[tuple(idx_list)])

        return_tuple = (*return_tuple, ax_merged_list)
    else:
        return_tuple = (*return_tuple, a_idx)

    if bx_list:
        bx_merged_list = []
        for bx in bx_list:
            idx_list = [slice(None, )] * bx.ndim
            idx_list[axis] = b_idx
            bx_merged_list.append(bx[tuple(idx_list)])

        return_tuple = (*return_tuple, bx_merged_list)
    else:
        return_tuple = (*return_tuple, b_idx)

    # special code for axis = -1
    # ax_merged_list = [ax[..., a_idx] for ax in ax_list]
    # bx_merged_list = [bx[..., b_idx] for bx in bx_list]

    # return the joint y vector and possibly both x vectors
    return return_tuple


def polygon_area(x, y):
    """
    This function calculates the area of a simple polygon with Shoelace formula.

    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

    :param x: input vector
    :param y: output vector
    :return: area
    """

    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area
