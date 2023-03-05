"""
This module is responsible for finding events valid according to UNECE regulation 79.

It includes one class for the event finding. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 25.06.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
import xarray as xr
import json

# -- custom imports --
from src.helpers.signal_processing.digitalization import pullup_glitches, get_event_boundaries, select_longest_events
from src.helpers.general.numpy_indexing import slices_to_index_array


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class UNECEr79EventFinder:
    """
    This class is responsible for finding events valid according to UNECE regulation 79.

    It includes two main functions called "find_lkft_events" and "cluster_lkft_events". The former calls the remaining
    methods. See the respective documentations for more details.
    """

    def __init__(self):
        """
        This method initializes a new class instance.
        """

        # -- read configs for the vehicle and the unece regulation
        r79_vehicle_config_path = "configs/r79_vehicle_config.json"
        with open(r79_vehicle_config_path) as r79_vehicle_config_file:
            self.r79_vehicle_cfg = json.load(r79_vehicle_config_file)

        # -- instantiate further instance attributes
        self.ay_upper_bounds = np.ndarray(shape=(1,))
        self.ay_lower_bounds = np.ndarray(shape=(1,))
        self.start_idx_longest = None
        self.stop_idx_longest = None

    def find_lkft_events(self, quantities_ts_da, qois_ts_da, scenarios_ts_da, scenarios_kpi_da):
        """
        This function extracts events from signals that satisfy the UNECE-R79 Lane Keeping Functional Test conditions.

        :param xr.DataArray quantities_ts_da: data-array with quantity signals
        :param xr.DataArray qois_ts_da: data-array with qoi signals
        :param xr.DataArray scenarios_ts_da: data-array with scenario signals
        :param xr.DataArray scenarios_kpi_da: data-array with scenario parameter values
        :return: data-array with events of qoi resp. scenario signals
        :rtype: (xr.DataArray, xr.DataArray)
        """

        # determine the index of the time dimension (used multiple times as axis parameter)
        idx_qu = quantities_ts_da.dims.index('quantities')
        dims_wo_quantities = quantities_ts_da.dims[:idx_qu] + quantities_ts_da.dims[idx_qu + 1:]
        idx_time = dims_wo_quantities.index('timesteps')

        if np.nanmax(quantities_ts_da.loc[{'quantities': 'LatCtrl.LKAS.IsActive', 'space_samples': 0}].data) > 1:
            raise NotImplementedError("just ACSF status 0 and 1 supported.")

        # check if the ay_ref channel is missing or consists only of NaN values
        if 'Car.ay_ref' not in quantities_ts_da.quantities.values.tolist() or (
                ~np.isnan(quantities_ts_da.loc[{'quantities': 'Car.ay_ref'}].data)).sum() == 0:

            # then check whether the curvature is available so ay_ref can be calculated
            if "LatCtrl.LKAS.CurveXY_trg" in quantities_ts_da.quantities.values.tolist():
                # calculate the reference lateral acceleration based on map curvature and actual velocity)
                quantities_ts_da.loc[{'quantities': 'Car.ay_ref'}] = np.multiply(
                    quantities_ts_da.loc[{'quantities': 'Car.v'}] ** 2,
                    quantities_ts_da.loc[{'quantities': 'LatCtrl.LKAS.CurveXY_trg'}])
            else:
                raise ValueError("curvature or reference lateral acceleration required for R-79 assessment.")

        # check whether the LKFT test conditions are met
        mask_dict, ay_abs = self.check_lkft_conditions(quantities_ts_da, scenarios_kpi_da)

        # remove short glitches from LKFT conditions
        mask_dict = self.pullup_lkft_glitches(mask_dict, idx_time, ay_abs)

        # combine all validity checks
        mask = sum(mask_dict.values()) == len(mask_dict)

        # extract the events where the validity checks were passed
        sample_rate = 100
        min_event_length = 4  # s
        start_idx, stop_idx, event_length = get_event_boundaries(mask, min_length=int(min_event_length * sample_rate),
                                                                 axis=idx_time)

        # select the longest events
        self.start_idx_longest, self.stop_idx_longest, event_length_longest = select_longest_events(
            start_idx, stop_idx, event_length, idx_time, mask.shape)

        # trim the desired signals
        qois_ts_events_da, scenarios_ts_events_da = self.trim_lkft_signals(
            self.start_idx_longest, event_length_longest, idx_time, qois_ts_da, scenarios_ts_da)

        return qois_ts_events_da, scenarios_ts_events_da

    def check_lkft_conditions(self, responses_da, scenarios_kpi_da):
        """
        This function checks the validity of LKFT test scenarios according to UNECE-R79.

        It checks that:
        - the lateral acceleration is between 80 and 90 % of aysmax, specified by the manufacturer.
        - the velocity is between vmin and vmax of the LKAS, specified by the manufacturer.
        - the LKAS was turned on by the driver
        - the LKAS was active (lane recognized, etc.)

        :param xr.DataArray responses_da: array of response signals
        :param xr.DataArray scenarios_kpi_da: data-array with scenario parameter values
        :return: dictionary of mask arrays, array of absolute lateral acceleration
        :rtype: (dict(np.ndarray), np.ndarray)
        """

        # check whether the ACA (LKAS) was active
        aca_active = responses_da.loc[{'quantities': 'LatCtrl.LKAS.SwitchedOn'}].data
        aca_mask = aca_active == 1

        # check ACSF status
        acsf_status = responses_da.loc[{'quantities': 'LatCtrl.LKAS.IsActive'}].data
        acsf_status_mask = acsf_status == 1

        # check human event trigger (interesting to filter big measurement files)
        # event_trigger = responses_da.loc[{'quantities': 'Eventtrigger'}].data
        # event_trigger_mask = event_trigger == 1

        # check velocity
        vx = responses_da.loc[{'quantities': 'Car.v'}].data
        vx[np.isnan(vx)] = 0
        vx_mask = (vx >= self.r79_vehicle_cfg['vx_min']['value']) & (vx <= self.r79_vehicle_cfg['vx_max']['value'])

        # extract the absolute measured lateral acceleration
        ay = responses_da.loc[{'quantities': 'Car.ay'}].data
        ay_abs = np.abs(ay)
        ay_abs[np.isnan(ay_abs)] = 0

        # extract the absolute reference lateral acceleration
        ay_ref = responses_da.loc[{'quantities': 'Car.ay_ref'}].data
        ay_ref_abs = np.abs(ay_ref)
        ay_ref_abs[np.isnan(ay_ref_abs)] = 0

        # compare the actual lateral acceleration signals with the ay band (if one fixed configurable band)
        # ay_mask = (ay_ref_abs >= self.r79_vehicle_cfg['ay_lower_bound']['value'] *
        #            self.r79_vehicle_cfg['ay_smax']['value']) & \
        #           (ay_ref_abs <= self.r79_vehicle_cfg['ay_upper_bound']['value'] *
        #            self.r79_vehicle_cfg['ay_smax']['value'])

        if scenarios_kpi_da.dims != ():
            # calculate the ideal lateral acceleration from the scenario planning
            if '_ay_norm' in scenarios_kpi_da.parameters.values.tolist():
                ay_target = scenarios_kpi_da.loc[{'parameters': '_ay_norm'}].data * \
                            self.r79_vehicle_cfg['ay_smax']['value']
            elif '_ay' in scenarios_kpi_da.parameters.values.tolist():
                ay_target = scenarios_kpi_da.loc[{'parameters': '_ay'}].data
            else:
                raise IndexError("lateral acceleration not available in scenario array")

            # create bins for the acceleration from 0 m/s^2 to ays_max with 10 bins (0-10%, ..., 80-90%, 90-100%)
            ay_bins = np.linspace(0, self.r79_vehicle_cfg['ay_smax']['value'], 11)

            # determine the matching ay band based on the scenario planning
            ay_digitized = np.digitize(ay_target, ay_bins)

            # "digitize" assigns the value to the right index (upper bound) -> -1 for the lower bound
            self.ay_lower_bounds = ay_bins[ay_digitized - 1]

            # decrement the highest index (appears e.g. by nan (from repetitions)) by one to avoid an indexing error
            ay_digitized[ay_digitized == len(ay_bins)] = len(ay_bins) - 1
            self.ay_upper_bounds = ay_bins[ay_digitized]

            # compare the actual lateral acceleration signals with the ay band
            ay_mask = (ay_ref_abs >= self.ay_lower_bounds[..., None]) & (ay_ref_abs <= self.ay_upper_bounds[..., None])
        else:
            raise NotImplementedError("data-driven selection of the matching ay band from unece-r79 is not implemented")

        mask_dict = {'aca': aca_mask, 'acsf_status': acsf_status_mask, 'vx': vx_mask, 'ay': ay_mask}

        return mask_dict, ay_abs

    def pullup_lkft_glitches(self, mask_dict, idx_time, ay_abs):
        """
        This function pulls up short glitches in binary LKFT signals / conditions.

        According to UNECE-R79, the ay condition can be torn under certain circumstances:
        "Notwithstanding the sentence above, for time periods of not more than 2 s the lateral acceleration of the
        system may exceed the specified value aysmax by not more than 40 %, while not exceeding the maximum value
        specified in the table in paragraph 5.6.2.1.3. of this Regulation by more than 0.3 m / s2."

        :param dict mask_dict: dictionary of mask arrays
        :param idx_time: index of the time dimension in the mask arrays
        :param np.ndarray ay_abs: array of absolute lateral acceleration
        :return: dictionary of mask arrays without short glitches
        :rtype: dict(np.ndarray)
        """

        # -- pull-up glitches in ACSF status
        sample_rate = 100
        mask_dict['acsf_status'] = pullup_glitches(mask_dict['acsf_status'], max_glitch_duration=int(0.8 * sample_rate),
                                                   axis=idx_time)

        # -- pullup glitches in ay condition

        # pullup glitches in ay condition which are not longer as 2s
        ay_mask_wo_glitches = pullup_glitches(mask_dict['ay'], max_glitch_duration=int(2 * sample_rate), axis=idx_time)

        # check whether it was a valid glitch removal (not more than 40 %, not more than 0.3)
        ay_valid_glitch_mask = (ay_abs <= 1.4 * self.r79_vehicle_cfg['ay_smax']['value']) & \
                               (ay_abs <= self.r79_vehicle_cfg['ay_max']['value'] + 0.3)

        # exclude 2s glitches below the ay band (include only the glitches above the band and below the notwith.-cond.)
        # ay_valid_glitch_mask = ay_valid_glitch_mask & (ay_abs >= self.ay_lower_bounds[..., None])

        # in case of invalid glitch removals, go back to the initial glitches
        ay_glitch_mask = mask_dict['ay'] != ay_mask_wo_glitches
        ay_mask_wo_glitches[ay_glitch_mask] = ay_valid_glitch_mask[ay_glitch_mask]
        mask_dict['ay'] = ay_mask_wo_glitches

        return mask_dict

    @staticmethod
    def trim_lkft_signals(start_idx, event_length, idx_time, qois_ts_da, scenarios_ts_da):
        """
        This function trims the desired LKFT signals by using the event start indices and lengths.

        :param tuple start_idx: start indices
        :param np.ndarray event_length: length of the events
        :param int idx_time: index of the time dimension
        :param xr.DataArray qois_ts_da: data-array with qoi signals
        :param xr.DataArray scenarios_ts_da: data-array with scenario signals
        :return: data-array with events of qoi resp. scenario signals
        :rtype: (xr.DataArray, xr.DataArray)
        """

        # determine the slice indices into the full time series (source array)
        idx_slice_source, _ = slices_to_index_array(start_idx, length=event_length, axis=idx_time)

        # determine the slice indices into the target array
        number_events = len(event_length)
        # replace the start indices of the time dimension with zeros
        start_idx_target = start_idx[:idx_time] + (np.zeros(number_events, dtype=int),) + start_idx[idx_time + 1:]
        # # if just the valid events shall be used in a purely data-driven approach
        # start_idx_target = [np.arange(number_events)] * 2
        # start_idx_target[idx_time] = np.zeros(number_events, dtype=int)
        idx_slice_target, _ = slices_to_index_array(tuple(start_idx_target), length=event_length, axis=idx_time)

        # -- trim the qoi signals
        # extend the index tuple by a full slice of the qois dimension
        idx_qu = qois_ts_da.dims.index('qois')
        idx_slice_qu_source = idx_slice_source[:idx_qu] + (slice(None),) + idx_slice_source[idx_qu:]
        idx_slice_qu_target = idx_slice_target[:idx_qu] + (slice(None),) + idx_slice_target[idx_qu:]

        # create a target data array, initialized with nan values, for the extracted events
        shape_list = list(qois_ts_da.shape)
        # # if just the valid events shall be used in a purely data-driven approach
        # shape_list[qois_ts_da.dims.index('space_samples')] = number_events
        shape_list[qois_ts_da.dims.index('timesteps')] = event_length.max()
        qois_ts_events_array = np.ones(shape=tuple(shape_list)) * np.nan
        qois_ts_events_da = xr.DataArray(qois_ts_events_array, dims=qois_ts_da.dims, coords=qois_ts_da.coords,
                                         attrs=qois_ts_da.attrs)

        # fill the target data array with the extracted events
        qois_ts_events_da.data[idx_slice_qu_target] = qois_ts_da.data[idx_slice_qu_source]

        # -- trim the scenario signals
        if scenarios_ts_da.dims != ():
            # extend the index tuple by a full slice of the parameter dimension
            idx_pa = scenarios_ts_da.dims.index('parameters')
            idx_slice_pa_source = idx_slice_source[:idx_pa] + (slice(None),) + idx_slice_source[idx_pa:]
            idx_slice_pa_target = idx_slice_target[:idx_pa] + (slice(None),) + idx_slice_target[idx_pa:]

            # create a target data array, initialized with nan values, for the extracted events
            shape_list = list(scenarios_ts_da.shape)
            shape_list[scenarios_ts_da.dims.index('space_samples')] = number_events
            shape_list[scenarios_ts_da.dims.index('timesteps')] = event_length.max()
            scenarios_ts_events_array = np.ones(shape=tuple(shape_list)) * np.nan
            scenarios_ts_events_da = xr.DataArray(scenarios_ts_events_array, dims=scenarios_ts_da.dims,
                                                  coords=scenarios_ts_da.coords, attrs=scenarios_ts_da.attrs)

            # fill the target data array with the extracted events
            scenarios_ts_events_da.data[idx_slice_pa_target] = scenarios_ts_da.data[idx_slice_pa_source]
        else:
            scenarios_ts_events_da = scenarios_ts_da

        return qois_ts_events_da, scenarios_ts_events_da

    def cluster_lkft_events(self, scenarios_mean_da):
        """
        This function clusters lkft events into bins to extract repetitions.

        :param xr.DataArray scenarios_mean_da: array of scenario parameters (mean values of the time signals)
        :return: array of clustered scenarios
        :rtype: xr.DataArray
        """

        # -- cluster the data into scenario bins

        # create bins for the velocity from 0 kph to 210 kph in 10 kph steps
        vx_bins = np.arange(0, 220, 10)

        # assign the mean velocities to the velocity bins
        vx_digitized = np.digitize(scenarios_mean_da.loc[{'parameters': '$Ego_Init_Velocity'}].data, vx_bins)

        # create bins for the acceleration from 0 m/s^2 to ays_max with 10 bins (0-10%, ..., 80-90%, 90-100%)
        ay_bins = np.linspace(0, self.r79_vehicle_cfg['ay_smax']['value'], 11)

        # assign the mean lateral accelerations to the acceleration bins
        ay_digitized = np.digitize(np.abs(scenarios_mean_da.loc[{'parameters': '$Acceleration'}].data), ay_bins)

        # stack the velocity and acceleration together into one array
        scenarios_idx = np.stack((vx_digitized, ay_digitized))

        # -- determine the unique bins (space samples) with at least one scenario (repetitions)
        # get inverse indices to re-construct 'scenarios_idx', and the number of scenarios per bin (repetitions)
        _, scenarios_idx_inv, counts = np.unique(scenarios_idx, axis=1, return_inverse=True, return_counts=True)

        # -- determine the indices to reshape the data for separate 'space_samples'- and 'repetitions'-dimensions

        # use the counts to generate the repetitions-dimension indices
        start_idx = (np.zeros(scenarios_idx.shape[1], dtype=int),)
        idx_slice = slices_to_index_array(start_idx=start_idx, length=counts, repeats=False)[0][0]

        # since unique sorts, 'idx_slice' is in the wrong order
        # -> perform some operations to get it into the correct order
        # there should be a simpler and more efficient way to do this (w/o sorting two times via unique and argsort ...)
        idx_sort = np.argsort(scenarios_idx_inv)
        scenarios_idx_inv_sorted = scenarios_idx_inv[idx_sort]
        cum_diff = np.arange(scenarios_idx.shape[1]) - scenarios_idx_inv_sorted
        cum_diff_resorted = np.zeros(scenarios_idx.shape[1], dtype=int)
        cum_diff_resorted[idx_sort] = cum_diff
        scenarios_idx_inv_cum = scenarios_idx_inv + cum_diff_resorted
        idx_repetition = idx_slice[scenarios_idx_inv_cum]

        # -- create and fill data-array
        # create nan-array with distinct bins times maximum number of repetitions per bin times two parameters vx and ay
        dims = ('space_samples', 'repetitions', 'parameters')
        shape = (len(counts), counts.max(), 2)
        scenarios_array = np.ones(shape=shape) * np.nan
        scenarios_da = xr.DataArray(scenarios_array, dims=dims, coords=scenarios_mean_da.coords)

        if scenarios_mean_da.dims == ('parameters', 'space_samples'):
            scenarios_da.data[scenarios_idx_inv, idx_repetition, :] = scenarios_mean_da.data.T
        elif scenarios_mean_da.dims == ('space_samples', 'parameters'):
            scenarios_da.data[scenarios_idx_inv, idx_repetition, :] = scenarios_mean_da.data
        else:
            raise IndexError("this scenario data-array is expected to have two specific dimensions")

        return scenarios_da
