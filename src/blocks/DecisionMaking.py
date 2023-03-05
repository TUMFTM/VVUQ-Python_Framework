"""
This module is responsible for decision making.

It includes one class for the decision making. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
import xarray as xr

# -- custom imports --


class DecisionMaking:
    """
    This class is responsible for decision making.

    It includes one main method called "check_tolerances" for the decision making in the validation domain and
    one main method called "check_regulation" for the decision making in the application domain.
    See more details in the documentation of the check_tolerances and check_regulation method.
    """

    def __init__(self, config, domain):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        :param str domain: type of VVUQ domain
        """

        # -- ASSIGN PARAMETERS TO INSTANCE ATTRIBUTES ------------------------------------------------------------------
        self.config = config
        self.domain = domain

        # -- CREATE CONFIG SUB-DICT POINTERS ---------------------------------------------------------------------------
        self.cfgdm = self.config[domain]['decision_making']

        # -- INSTANTIATE OBJECTS ---------------------------------------------------------------------------------------
        self.decision_da = xr.DataArray(None)

    def check_tolerances(self, metric_da, qois_model_da, qois_system_da):
        """
        This function compares model validation results with defined thresholds.

        It offers absolute and relative tolerances as well as tolerances based on ISO 19364 with interleaves
        (calculate_iso19364_boundaries).

        It returns a boolean data array with the same shape as the input array, as well as one global boolean value.

        :param xr.DataArray metric_da: data array from validation metric
        :param xr.DataArray qois_model_da: data array of the model from the qoi assessment
        :param xr.DataArray qois_system_da: data array of the system from the qoi assessment
        :return: boolean decision data array and overall boolean value
        :rtype: tuple(xr.DataArray, bool)
        """

        # run through the number of qois
        decision_list = []
        for i in range(len(self.cfgdm['qois_name_list'])):

            # -- determine upper and lower bounds --

            if self.cfgdm['qois_type_list'][i] == 'absolute':
                # an absolute threshold can be directly used as reference
                lower_bound = self.cfgdm['qois_lower_threshold_list'][i]
                upper_bound = self.cfgdm['qois_upper_threshold_list'][i]

            elif self.cfgdm['qois_type_list'][i] == 'relative':
                # extract the kpi data array of one qoi, respectively
                single_qoi_kpi_system_da = qois_system_da.loc[{'qois': self.cfgdm['qois_name_list'][i]}]

                # a relative threshold must be multiplied with the ground truth data itself
                lower_bound = self.cfgdm['qois_lower_threshold_list'][i] * single_qoi_kpi_system_da
                upper_bound = self.cfgdm['qois_upper_threshold_list'][i] * single_qoi_kpi_system_da

            elif self.cfgdm['qois_type_list'][i] == 'ISO19364':
                # extract the time series response of the model
                single_qoi_ts_model_da = qois_model_da.loc[{'qois': self.cfgdm['qois_name_list'][i]}]
                idx_time = single_qoi_ts_model_da.dims.index('timesteps')

                # in ISO 19364 the lateral acceleration is the x-quantity
                # x = qois_model_da.loc[{'qois': 'Car.ay'}].data

                # here, the time is used instead
                time_vector = np.arange(single_qoi_ts_model_da.shape[idx_time])
                shape = single_qoi_ts_model_da.shape[:idx_time] + (1,) + single_qoi_ts_model_da.shape[idx_time + 1:]
                x = np.tile(time_vector, shape)

                _, lower_bound, _, upper_bound = self.calculate_iso19364_boundaries(
                    x=x, y=single_qoi_ts_model_da.data, axis=idx_time, y_offset=self.cfgdm['qois_offset_list'][i],
                    y_gain=self.cfgdm['qois_gain_list'][i])

                raise NotImplementedError("Interpolation has to be implemented for ISO 19364.")

            else:
                raise ValueError("this threshold type is not available")

            # -- compare data against bounds --

            if self.cfgdm['qois_type_list'][i] in ('absolute', 'relative'):
                # extract the metric data array of one qoi, respectively
                single_qoi_metric_da = metric_da.loc[{'qois': self.cfgdm['qois_name_list'][i]}]

                # check if the data is above the lower bound
                single_qoi_lower_decision_da = single_qoi_metric_da >= lower_bound
                # check if the data is below the upper bound
                single_qoi_upper_decision_da = single_qoi_metric_da <= upper_bound

                # combine both
                single_qoi_decision_da = single_qoi_lower_decision_da & single_qoi_upper_decision_da

            elif self.cfgdm['qois_type_list'][i] == 'ISO19364':
                # extract the time series response of the system
                single_qoi_ts_system_da = qois_system_da.loc[{'qois': self.cfgdm['qois_name_list'][i]}]

                # check if the data is above the lower bound
                single_qoi_lower_decision_da = single_qoi_ts_system_da >= lower_bound
                # check if the data is below the upper bound
                single_qoi_upper_decision_da = single_qoi_ts_system_da <= upper_bound

                # combine both
                single_qoi_decision_da = single_qoi_lower_decision_da & single_qoi_upper_decision_da

            else:
                raise ValueError("this threshold type is not available")

            decision_list.append(single_qoi_decision_da)

        # concatenate the single data arrays of each qoi
        self.decision_da = xr.concat(decision_list, 'qois')  # type: xr.DataArray

        # -- get results for each space-point
        if 'interval' in self.decision_da.dims:
            decision_space_da = np.all(self.decision_da, axis=self.decision_da.dims.index('interval'))
        else:
            decision_space_da = self.decision_da

        # check if all decision are true (passed) or if at least one is false (failed)
        global_decision = self.decision_da.data.all()

        return decision_space_da, global_decision

    def check_regulation(self, qois_kpi_da):
        """
        This function compares assessment results with regulation thresholds.

        It returns a boolean data array with the same shape as the input array, as well as one global boolean value.

        :param xr.DataArray qois_kpi_da: result data array
        :return: boolean decision data array and overall boolean value
        :rtype: tuple(xr.DataArray, bool)
        """

        # run through the number of qois
        qoi_space_decision_list = []
        for (name, thresh_type, lower_tresh, upper_tresh) in zip(
                self.cfgdm['qois_name_list'], self.cfgdm['qois_type_list'], self.cfgdm['qois_lower_threshold_list'],
                self.cfgdm['qois_upper_threshold_list']):

            if thresh_type == 'absolute':
                # an absolute threshold can be directly used as reference
                lower_bound = lower_tresh
                upper_bound = upper_tresh
            else:
                raise ValueError("this threshold type is not available")

            if thresh_type == 'absolute':

                # -- extract only one element from each dimension, except keep the full space_samples dimension

                # create index dictionaries
                lower_idx_dict = {'qois': name}
                upper_idx_dict = {'qois': name}
                if 'aleatory_samples' in qois_kpi_da.dims or 'repetitions' in qois_kpi_da.dims:

                    # get the correct index of the aleatory samples or repetitions
                    if 'aleatory_samples' in qois_kpi_da.dims:
                        dim_name = 'aleatory_samples'
                    elif 'repetitions' in qois_kpi_da.dims:
                        dim_name = 'repetitions'
                    else:
                        raise ValueError("should be unreachable")

                    if 'confidence' in self.cfgdm:
                        # -- if the user selected a confidence value

                        # create a small tolerance for floating point comparisons
                        eps = 1e-6

                        if isinstance(qois_kpi_da.probs, np.ndarray):
                            # -- case equal number of cdf steps for all space samples

                            # lower bound: get the last element that is smaller than the inverse confidence (e.g. 5%)
                            # argmin gives the first False element, -1 gives the last True element
                            # e.g. prob = [0, 0.25, 0.5, 0.75, 1]
                            # c=1...0.76: idx=0 (-inf -> too less steps), c=0.75...0.51: idx=1 (first step), etc.
                            lower_idx_dict[dim_name] = \
                                np.argmin(qois_kpi_da.probs <= (1 - self.cfgdm['confidence'] + eps)) - 1

                            # upper bound: get the second element that is greater than the confidence (e.g. 95%)
                            # argmax gives the first True element, +1 gives the second True element
                            # e.g. prob = [0, 0.25, 0.5, 0.75, 1]
                            # c=1..0.76: idx=5 (out of bounds -> too less steps), c=0.75..0.51: idx=4 (first step), ..
                            upper_idx_dict[dim_name] = np.argmax(
                                qois_kpi_da.probs >= (self.cfgdm['confidence'] - eps)) + 1

                            if lower_idx_dict[dim_name] == 0 or upper_idx_dict[dim_name] == len(qois_kpi_da.probs):
                                raise ValueError("The selected confidence is too high for the number of samples.")

                        elif isinstance(qois_kpi_da.probs, list):
                            # -- case varying number of cdf steps for the space samples

                            # same as above but with list comprehensions
                            lower_idx_dict[dim_name] = [np.argmin(probs <= (1 - self.cfgdm['confidence'] + eps)) - 1
                                                        for probs in qois_kpi_da.probs]
                            upper_idx_dict[dim_name] = [np.argmax(probs >= (self.cfgdm['confidence'] - eps)) + 1
                                                        for probs in qois_kpi_da.probs]

                            if 0 in lower_idx_dict[dim_name] or any(idx == len(probs) for idx, probs in
                                                                    zip(upper_idx_dict[dim_name], qois_kpi_da.probs)):
                                raise ValueError("The selected confidence is too high for the number of samples.")

                            # this has no effect, since xarray vectorized indexing works different than for numpy
                            # http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
                            # number_space_samples = qois_kpi_da.shape[qois_kpi_da.dims.index('space_samples')]
                            # lower_idx_dict['space_samples'] = [i for i in range(number_space_samples)]
                            # upper_idx_dict['space_samples'] = [i for i in range(number_space_samples)]

                    else:
                        # -- if the user did not select a confidence value, we assume the highest possible
                        # take the second aleatory element as lowest value (first is -inf) and the last as highest value
                        lower_idx_dict[dim_name] = 1
                        upper_idx_dict[dim_name] = -1

                if 'pbox_edges' in qois_kpi_da.dims:
                    # take the left pbox edge as lowest value and the right edge as highest value
                    lower_idx_dict['pbox_edges'] = 'left'
                    upper_idx_dict['pbox_edges'] = 'right'

                if 'interval' in qois_kpi_da.dims:
                    # take the left interval as lowest value and the right interval as highest value
                    lower_idx_dict['interval'] = 'left'
                    upper_idx_dict['interval'] = 'right'

                # extract the relevant data from the arrays
                single_qoi_kpi_lower_da = qois_kpi_da.loc[lower_idx_dict]
                single_qoi_kpi_upper_da = qois_kpi_da.loc[upper_idx_dict]

                if 'probs' in qois_kpi_da.attrs and isinstance(qois_kpi_da.probs, list):
                    # save the coords
                    coords = single_qoi_kpi_lower_da.coords

                    # take the diagonal elements for variable cdf steps, since xarray vector indexing works different
                    # http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
                    single_qoi_kpi_lower_da = xr.DataArray(single_qoi_kpi_lower_da.data.diagonal(),
                                                           dims=('space_samples',), coords=coords)
                    single_qoi_kpi_upper_da = xr.DataArray(single_qoi_kpi_upper_da.data.diagonal(),
                                                           dims=('space_samples',), coords=coords)

                # check if the data is above the lower bound
                single_qoi_lower_space_decision_da = single_qoi_kpi_lower_da > lower_bound
                # check if the data is below the upper bound
                single_qoi_upper_space_decision_da = single_qoi_kpi_upper_da <= upper_bound

                # combine both
                single_qoi_space_decision_da = single_qoi_lower_space_decision_da & single_qoi_upper_space_decision_da

            else:
                raise ValueError("this threshold direction is not available")

            # store the results from each qoi iteration in a list
            qoi_space_decision_list.append(single_qoi_space_decision_da)

        # concatenate the single data arrays of each qoi
        decision_space_da = xr.concat(qoi_space_decision_list, 'qois')  # type: xr.DataArray

        # check if all decision are true (passed) or if at least one is false (failed)
        global_decision = decision_space_da.data.all()

        return decision_space_da, global_decision

    @staticmethod
    def calculate_iso19364_boundaries(x, y, axis=-1, y_offset=5.0, y_gain=0.03, x_offset=0.1, x_gain=0.06):
        """
        This function calculates the ISO 19364 time series boundaries.

        Example:
        - x: lateral acceleration with x_offset = 0.1 and x_gain = 0.06
        - y: steering wheel angle with y_offset = 5.0 and y_gain = 0.03

        :param np.ndarray x: input signal
        :param np.ndarray y: output signal
        :param axis: along this axis the operations shall be applied
        :param float y_offset: aditive tolerance for the output quantity
        :param float y_gain: multiplicative tolerance for the output quantity
        :param float x_offset: aditive tolerance for the input quantity
        :param float x_gain: multiplicative tolerance for the input quantity
        :return: x and y values of bottom and top boundaries
        :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """

        # calculate deltas: "difference between [...] the current value and preceding value"
        idx_tuple_wo_first = (slice(None),) * axis + (slice(1, None),) + (slice(None),) * (x.ndim - axis - 1)
        idx_tuple_wo_last = (slice(None),) * axis + (slice(None, -1),) + (slice(None),) * (x.ndim - axis - 1)
        delta_x = np.zeros(shape=x.shape)
        delta_y = np.zeros(shape=y.shape)
        delta_x[idx_tuple_wo_first] = x[idx_tuple_wo_first] - x[idx_tuple_wo_last]
        delta_y[idx_tuple_wo_first] = y[idx_tuple_wo_first] - y[idx_tuple_wo_last]

        # calculate epsilons by combining the data with the additive and multiplicative tolerances
        eps_x = x_offset + x_gain * np.abs(x)
        eps_y = y_offset + y_gain * np.abs(y)

        # calculate D by crossing the data and epsilons
        d = np.sqrt((delta_x * eps_y) ** 2 + (delta_y * eps_x) ** 2)

        # calculate the top boundaries
        x_t = x - delta_y * eps_x ** 2 / d
        y_t = y + delta_x * eps_y ** 2 / d

        # calculate the bottom boundaries
        x_b = x + delta_y * eps_x ** 2 / d
        y_b = y - delta_x * eps_y ** 2 / d

        return x_b, y_b, x_t, y_t
