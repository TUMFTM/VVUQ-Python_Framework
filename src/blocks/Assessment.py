"""
This module is responsible for assessing results from simulations and experiments.

It includes one class for the assessment. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import warnings
import itertools

# -- third-party imports --
import numpy as np
import xarray as xr

# -- custom imports --
from src.commonalities.CsvHandler import AssessmentCsvHandler
from src.helpers.signal_processing import filtering
from src.applications.UNECE_R79.unecer79_event_finder import UNECEr79EventFinder
from src.helpers.namespaces.QuantityNamespace import R79ParameterNamespaceHandler


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Assessment:
    """
    This class is responsible for assessing results from simulations and experiments.

    It includes a main method called "run_process" that calls the other class methods.
    See more details in the documentation of the run_process method.
    """

    def __init__(self, config, domain, instance):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        :param str domain: type of VVUQ domain
        :param str instance: testing instance
        """

        # -- ASSIGN PARAMETERS TO INSTANCE ATTRIBUTES ------------------------------------------------------------------
        self.config = config
        self.domain = domain
        self.instance = instance

        # -- CREATE CONFIG SUB-DICT POINTERS ---------------------------------------------------------------------------
        self.cfgti = self.config['cross_domain'][instance]
        self.cfgas = self.config['cross_domain']['assessment']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        # instantiate csv file handler for saving and reloading assessment kpis to and from file
        self.assessment_csv_handler = AssessmentCsvHandler(config, domain, instance)

        # create the csv path
        self.csv_path = self.cfgti['result_folder'] + '/' + self.domain + '/parameter_erg_mapping.csv'

        # initialize empty data arrays
        self.scenarios_ts_da = xr.DataArray(None)
        self.quantities_ts_da = xr.DataArray(None)
        self.qois_ts_da = xr.DataArray(None)
        self.qois_ts_untrimmed_da = xr.DataArray(None)
        self.qois_kpi_da = xr.DataArray(None)
        self.qois_kpi_raw_da = xr.DataArray(None)

        if self.cfgas['method'] == "UNECE-R79":
            self.unece_event_finder = UNECEr79EventFinder()

    def run_process(self, quantities_ts_da, scenarios_kpi_da):
        """
        This function performs the assessment process of results from simulations and experiments.

        It consists of multiple steps based on separate functions:
        1) perform_filtering: filtering of noisy measurement quantities
        2) calculate_qois: calculation of Quantities of Interest (QoIs) from the measured quantities
        3) data-driven pipeline, partly implemented for the UNECE-R79 use case
        3.1) extract_scenario_timeseries: calculation of scenario time series from measured quantities
        3.2) unece_event_finder.find_lkft_events: data-driven event finding (trimming of scenario and qoi time series)
        3.3) calculation of scenario parameters from scenario time series
        3.4) cluster_lkft_events: clustering of events to determine repetitions, etc.
        4) calculate_kpis: calculation of Key Performance Indicators (KPIs) from the qoi time signals
        5) create_ecdf: creation of empirical CDFs from aleatory samples (and repetitions)
        6) get_pbox_edges: creation of p-boxes from epistemic samples

        The qoi array can have different dimensions depending on the selected case:
        - 'qois', 'space_samples': deterministic simulations with KPI assessment
        - 'qois', 'space_samples', 'timesteps': deterministic simulations without KPI assessment
        - 'qois', 'space_samples', 'aleatory_samples'/'repetitions': ECDFs based on KPI assessment
        - 'qois', 'space_samples', 'pbox_edges', 'aleatory_samples': p-boxes based on KPI assessment

        :param xr.DataArray quantities_ts_da: array of quantity time series (ts)
        :param xr.DataArray scenarios_kpi_da: array of scenario parameters (kpis)
        :return: array of processed qois and scenario parameters
        :rtype: tuple(xr.DataArray, xr.DataArray)
        """

        # if kpis shall be read from a csv file, we dont need to process any time responses (overwritten anyway)
        if self.cfgas['method'] != "read_csv":
            # filter the quantities
            self.quantities_ts_da = self.perform_filtering(quantities_ts_da)

            # calculate quantities of interest
            self.qois_ts_untrimmed_da = self.calculate_qois(self.quantities_ts_da)
            self.qois_ts_da = self.qois_ts_untrimmed_da
        else:
            self.qois_ts_untrimmed_da = quantities_ts_da
            self.qois_ts_da = self.qois_ts_untrimmed_da

        if self.cfgas['method'] == "UNECE-R79" and self.cfgas['is_event_finder']:
            # determine parameter signals
            self.scenarios_ts_da = self.extract_scenario_timeseries(self.quantities_ts_da, scenarios_kpi_da)

            # find LKFT events
            self.qois_ts_da, self.scenarios_ts_da = self.unece_event_finder.find_lkft_events(
                self.quantities_ts_da, self.qois_ts_untrimmed_da, self.scenarios_ts_da, scenarios_kpi_da)

            if self.scenarios_ts_da.dims != ():

                raise NotImplementedError("data-driven clustering of lkft events not fully implemented yet")

                # calculate the mean value of the scenario signals as KPI
                scenarios_mean = np.nanmin(self.scenarios_ts_da.data, axis=self.scenarios_ts_da.dims.index('timesteps'))
                idx_time = self.scenarios_ts_da.dims.index('timesteps')
                dims = self.scenarios_ts_da.dims[:idx_time] + self.scenarios_ts_da.dims[idx_time + 1:]
                scenarios_mean_da = xr.DataArray(scenarios_mean, dims=dims, coords=self.scenarios_ts_da.coords)

                # cluster the LKFT events into space samples and repetitions
                scenarios_kpi_da = self.unece_event_finder.cluster_lkft_events(scenarios_mean_da)

        # calculate kpis of the quantities of interest
        self.qois_kpi_raw_da = self.calculate_kpis(self.qois_ts_da)
        qois_kpi_da = self.qois_kpi_raw_da

        if 'aleatory_samples' in qois_kpi_da.dims or 'repetitions' in qois_kpi_da.dims:
            # create ECDFs (empirical cumulative distribution functions)
            qois_kpi_da = self.create_ecdf(qois_kpi_da)

        if 'epistemic_samples' in qois_kpi_da.dims:
            # create the probability-box (p-box)
            qois_kpi_da = self.get_pbox_edges(qois_kpi_da)

        self.qois_kpi_da = qois_kpi_da

        return qois_kpi_da, scenarios_kpi_da

    @staticmethod
    def perform_filtering(responses_da):
        """
        This function handles the filtering of the total data array including possible nan-values.

        Unfortunately the scipy signal.filtfilt function (e.g. for butterworth) can not handle nan-values.
        If the input signal contains nan values, each element of the filtered signal is nan.
        However, our data array representation includes nan values to fill up values, since not all physical test
        scenarios have the same duration.

        :param xr.DataArray responses_da: quantities data array
        :return: filtered quantities data array
        :rtype: xr.DataArray
        """

        # check if at least one quantity shall be filtered
        if not all(filt == "none" for filt in responses_da.quantities.filter_list):
            # copy the data array
            responses_filt_da = responses_da.copy()

            # loop through the quantities and check whether they shall be filtered
            for i in range(responses_da.shape[responses_da.dims.index('quantities')]):
                if responses_da.quantities.filter_list[i] != "none":

                    # create a dictionary to index the data array
                    idx_dict = {'quantities': i}

                    # "remove" the quantities- and timesteps-elements from the dims- and space-tuple
                    dim_red = tuple(dim for dim in responses_da.dims if dim not in {'quantities', 'timesteps'})
                    shape_red = tuple(sh for (sh, dim) in zip(responses_da.shape, responses_da.dims) if dim not in {
                        'quantities', 'timesteps'})

                    # create a multi-dimensional index for the reduced shape
                    for idx in np.ndindex(shape_red):
                        # update the index dictionary with the loop-idx
                        idx_dict.update(zip(dim_red, idx))

                        # extract the 1d array / time vector from the full data array
                        quantity_ts = responses_filt_da[idx_dict].data

                        # check which elements are not "nan"
                        mask = ~np.isnan(quantity_ts)

                        # if all values are nan (variable repetitions), keep them as they are
                        if mask.sum() == 0:
                            mask = np.ones(shape=mask.shape, dtype=bool)

                        # apply the filter to the elements being not "nan" and assign them to the correct positions
                        responses_filt_da[idx_dict].data[mask] = filtering.butterworth_filter(quantity_ts[mask])
        else:
            # if no filtering, just assign a reference to the data array
            responses_filt_da = responses_da

        return responses_filt_da

    @staticmethod
    def extract_scenario_timeseries(quantities_ts_da, scenarios_da):
        """
        This function extracts timeseries for scenario parameters from the measured quantity time signals.

        :param xr.DataArray quantities_ts_da: data-array with time series of the quantities
        :param xr.DataArray scenarios_da: data-array with scalar scenario parameters
        :return: data-array with time series of the scenario parameters
        :rtype: xr.DataArray
        """

        # initialize the namespace handler to map from scenario parameters to corresponding quantities
        namespace_handler = R79ParameterNamespaceHandler()

        # -- check whether there are scenario parameters, for which a measured time series quantity exists
        quantity_name_list = quantities_ts_da.quantities.values.tolist()
        parameter_name_list = scenarios_da.parameters.values.tolist()
        scenarios_ts_list = []
        parameter_ts_name_list = []
        # loop through the scenario parameters
        for parameter_name in parameter_name_list:
            # map the scenario parameters to corresponding quantities
            quantity_name = namespace_handler.quantity_name_mapper_none(parameter_name)
            # check if the quantity was measured
            if quantity_name in quantity_name_list:
                # append the quantity time series data and the parameter name to lists
                scenarios_ts_list.append(quantities_ts_da[{'quantities': quantity_name_list.index(quantity_name)}].data)
                parameter_ts_name_list.append(parameter_name)

        # create data-array
        if scenarios_ts_list:
            scenarios_ts_array = np.array(scenarios_ts_list)
            idx_qu = quantities_ts_da.dims.index('quantities')
            dims = quantities_ts_da.dims[:idx_qu] + ('parameters',) + quantities_ts_da.dims[idx_qu + 1:]
            coords = {'parameters': parameter_ts_name_list}
            scenarios_ts_da = xr.DataArray(scenarios_ts_array, dims=dims, coords=coords)
        else:
            scenarios_ts_da = xr.DataArray(None)

        return scenarios_ts_da

    def calculate_qois(self, responses_da):
        """
        This function calculates the Quantity of Interest (QoI) signals.

        The quantities from simulation or experiment can be selected directly as QoIs.
        It is also possible to combine mutliple response quantities to a new QoI. An example would be the calculation
        of a Time-to-Collision (TTC) signal from the distance and relative velocity signals of two vehicles.

        Available QoIs:
        - Quantities from simulation and experiment
        - Time-To-Collision (TTC)

        :param xr.DataArray responses_da: time responses of the original quantities
        :return: time responses of the quantities of interest
        :rytpe: xr.DataArray
        """

        # replace the quantities dimension of the responses array with a qoi dimension
        idx_qu = responses_da.dims.index('quantities')
        dims = responses_da.dims[:idx_qu] + ('qois',) + responses_da.dims[idx_qu + 1:]
        shape = responses_da.shape[:idx_qu] + (len(self.cfgas['qois_name_list']),) + responses_da.shape[idx_qu + 1:]

        # create an empty data-array for the qois' timeseries
        qois_ts_array = np.empty(shape)
        qois_ts_da = xr.DataArray(qois_ts_array, dims=dims, coords={'qois': self.cfgas['qois_name_list']})

        # -- determine the QoIs
        quantity_name_list = responses_da.quantities.values.tolist()
        for i in range(len(self.cfgas['qois_name_list'])):

            if self.cfgas['qois_name_list'][i] in quantity_name_list:
                # extract the qoi signals from all response quantities
                one_qoi_ts_da = responses_da[{'quantities': quantity_name_list.index(self.cfgas['qois_name_list'][i])}]

            elif self.cfgas['qois_name_list'][i] == 'TTC':
                # calculate the Time-to-Collision (TTC)
                one_qoi_ts_da = self.calculate_ttc(responses_da)

            elif self.cfgas['qois_name_list'][i] in {'D2LL', 'D2RL', 'D2L'}:

                if self.cfgas['qois_name_list'][i] in {'D2LL', 'D2L'} and 'LatCtrl.DistToLeft' in quantity_name_list:
                    # -- determine distance to left line (from right vehicle edge, out of positive R+_0 numbers)

                    # extract the distance to left line channel
                    one_qoi_ts_da = responses_da[{'quantities': quantity_name_list.index('LatCtrl.DistToLeft')}]

                    # if a CarMaker sensor outputs 0 in the first timestep, replace it with the second value
                    zero_mask = one_qoi_ts_da[{'timesteps': 0}].data == 0
                    one_qoi_ts_da[{'timesteps': 0}].data[zero_mask] = one_qoi_ts_da[{'timesteps': 1}].data[zero_mask]

                    # subtract the half vehicle width to get the distance from the left vehicle boundary to the line
                    one_qoi_ts_da = one_qoi_ts_da - 1.8800 / 2

                if self.cfgas['qois_name_list'][i] in {'D2RL', 'D2L'} and 'LatCtrl.DistToRight' in quantity_name_list:
                    raise NotImplementedError("currently only distance to left line implemented, not right line.")

                if self.cfgas['qois_name_list'][i] == 'D2L':
                    # -- determine distance to line based on distance to left line and distance to right line

                    if 'D2LL' in quantity_name_list and 'D2RL' in quantity_name_list:
                        # -- (IMU) case with vehicle edge as coordinate origin

                        # calculate the min value between left and right at each point (will be used with min-KPIs)
                        one_qoi_ts_da = responses_da.loc[{'quantities': ['D2LL', 'D2RL']}]
                        one_qoi_ts_da = one_qoi_ts_da.min(dim='quantities')

                    elif 'LatCtrl.DistToLeft' in quantity_name_list and 'LatCtrl.DistToRight' in quantity_name_list:
                        # -- (CarMaker) case with vehicle center as coordinate origin
                        raise NotImplementedError(
                            "calculation of distance to line for CarMaker currently not implemented.")

                    else:
                        raise ValueError("left and right distance to line signals required for D2L QoI.")

                # embed the less-comparison into a warning context manager
                with warnings.catch_warnings():
                    # suppress "RuntimeWarning: invalid value encountered in less" since it is intended
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    # set negative distance values to zero
                    # (if D2LL is included in the quantities and qois, but negative, we would not get here)
                    one_qoi_ts_da.data[one_qoi_ts_da.data < 0] = 0

            elif self.cfgas['qois_name_list'][i] == 'Car.Jerk':
                # we currently assume a sample rate of 100Hz or 0.01s for the measurements (default in CarMaker)
                # this value could be extracted from the measurement files in the future for more flexibility
                sample_rate = 100

                # determine the lateral jerk signal by calculating the derivative of the lateral acceleration signal
                ay_da = responses_da.loc[{'quantities': 'Car.ay'}]
                idx_time = ay_da.dims.index('timesteps')
                jerk_array = np.gradient(ay_da.data, axis=idx_time) * sample_rate

                # take the absolute value of the lateral jerk, since both directions should count for the max value
                jerk_array = np.abs(jerk_array)

                one_qoi_ts_da = xr.DataArray(jerk_array, dims=ay_da.dims)

            else:
                raise ValueError("quantity of interest does not exist")

            # fill the complete time series data array
            qois_ts_da[{'qois': i}] = one_qoi_ts_da

            # Plots.plot_timeseries(qois_ts_da, {'qois': self.cfgas['qois_name_list'][i]})

        return qois_ts_da

    def calculate_kpis(self, qois_ts_da):
        """
        This function calculates the Key Performance Indicator (KPI) values.

        It extracts characteristic values or KPIs from the time signals.

        Available KPIs:
        - none: time signals of QoIs used without extraction of characteristic values
        - min: minimum value of a OoI signal
        - max: maximum value of a QoI signal
        - mean: mean value of a QoI signal

        :param xr.DataArray qois_ts_da: time responses of the quantities of interest
        :return: kpis of the quantities of interest (or time series if "none" in self.cfgas['qois_kpi_list'])
        :rytpe: xr.DataArray
        """

        # -- case: only time series assessment -> dirctly return with some checks
        if "none" in self.cfgas['qois_kpi_list']:
            # if "none" is selected for the kpi of one QoI (full time vector), then it must be selected for all of them
            if not (all(elem == "none" for elem in self.cfgas['qois_kpi_list'])):
                raise ValueError("time vectors can just be selected for all quantities")
            if self.cfgas['method'] == "read_csv":
                raise ValueError("time vectors must be read in the simulator class")

            return qois_ts_da

        # if kpis shall be read from a csv file, we dont need to process any kpis (overwritten anyway)
        if self.cfgas['method'] == "read_csv":
            qois_kpi_da = self.assessment_csv_handler.load(qois_ts_da)
            return qois_kpi_da

        # remove the time dimension
        idx_ts = qois_ts_da.dims.index('timesteps')
        dims = qois_ts_da.dims[:idx_ts] + qois_ts_da.dims[idx_ts + 1:]
        shape = qois_ts_da.shape[:idx_ts] + qois_ts_da.shape[idx_ts + 1:]

        # check if at least one kpi is of the x_mean type
        if self.cfgas['repetition_kpi'] == "mean":
            # (it would be good to check the selected propagation method, but this info is lost in the read_csv case)

            # remove the 'repetitions'-dimension from the experimental target dims and shape (dropped after x_mean)
            if 'repetitions' in qois_ts_da.dims:
                idx_rep = dims.index('repetitions')
                dims = dims[:idx_rep] + dims[idx_rep + 1:]
                shape = shape[:idx_rep] + shape[idx_rep + 1:]

        # create an empty xarray
        qois_kpi_array = np.empty(shape)
        qois_kpi_da = xr.DataArray(qois_kpi_array, dims=dims, coords={'qois': self.cfgas['qois_name_list']})

        # -- calculate the KPIs
        for i in range(len(self.cfgas['qois_name_list'])):
            # extract the data array of a single quantity
            one_qoi_ts_da = qois_ts_da[{'qois': i}]

            # embed the nanmin/nanmax/nanmean calculations into an warning context manager
            with warnings.catch_warnings():

                # distinguish the warning handling between single and double (repetition mean) nanxxx calculations
                if self.cfgas['repetition_kpi'] == "none" or 'repetitions' not in one_qoi_ts_da.dims:

                    if 'repetitions' not in one_qoi_ts_da.dims:
                        # raise errors for All-NaN slices, since it would cause the error learning to fail later
                        warnings.filterwarnings('error')
                    else:
                        # suppress "RuntimeWarning: All-NaN slice encountered", since it will be used in variable CDFs
                        warnings.simplefilter("ignore", category=RuntimeWarning)

                    try:
                        if self.cfgas['qois_kpi_list'][i] == 'min':
                            # calucate the min value of the timesteps-dimension
                            qois_kpi_da[{'qois': i}] = np.nanmin(one_qoi_ts_da.data,
                                                                 axis=one_qoi_ts_da.dims.index('timesteps'))

                        elif self.cfgas['qois_kpi_list'][i] == 'max':
                            # calucate the max value of the timesteps-dimension
                            qois_kpi_da[{'qois': i}] = np.nanmax(one_qoi_ts_da.data,
                                                                 axis=one_qoi_ts_da.dims.index('timesteps'))

                        elif self.cfgas['qois_kpi_list'][i] == 'mean':
                            # calucate the mean value of the timesteps-dimension
                            qois_kpi_da[{'qois': i}] = np.nanmean(one_qoi_ts_da.data,
                                                                  axis=one_qoi_ts_da.dims.index('timesteps'))
                        else:
                            raise ValueError("this kpi type is not available.")
                    except Warning:
                        raise ValueError("valid output (not only nan values) required for each space scenario")

                elif self.cfgas['repetition_kpi'] == "mean":

                    # suppress "RuntimeWarning: All-NaN slice encountered" since it is intended for variable repetitions
                    warnings.simplefilter("ignore", category=RuntimeWarning)

                    if self.cfgas['qois_kpi_list'][i] == 'min':
                        # calucate the min value of the timesteps-dimension
                        qois_kpi_repetitions = np.nanmin(one_qoi_ts_da.data, axis=one_qoi_ts_da.dims.index('timesteps'))
                    elif self.cfgas['qois_kpi_list'][i] == 'max':
                        # calucate the max value of the timesteps-dimension
                        qois_kpi_repetitions = np.nanmax(one_qoi_ts_da.data, axis=one_qoi_ts_da.dims.index('timesteps'))
                    elif self.cfgas['qois_kpi_list'][i] == 'mean':
                        # calucate the mean value of the timesteps-dimension
                        qois_kpi_repetitions = np.nanmean(one_qoi_ts_da.data,
                                                          axis=one_qoi_ts_da.dims.index('timesteps'))
                    else:
                        raise ValueError("this kpi type is not available.")

                    # catch nanmean "RuntimeWarning: Mean of empty slice", since the error learning will fail then
                    warnings.filterwarnings('error')
                    try:
                        # calucate the mean value of the repetitions-dimension
                        qois_kpi_da[{'qois': i}] =\
                            np.nanmean(qois_kpi_repetitions, axis=one_qoi_ts_da.dims.index('repetitions'))
                    except Warning:
                        raise ValueError("valid output required for each space scenario")

                else:
                    raise ValueError("this repetition handling is not available.")

        # create a 2d array with the qois as columns and add it to the attrs dictionary
        if qois_kpi_da.dims.index('qois') == 0:
            qois_kpi_da.attrs['array2d'] = np.reshape(qois_kpi_da.data, (qois_kpi_da.shape[0], -1)).T
        elif qois_kpi_da.dims.index('qois') == (len(qois_kpi_da.dims) - 1):
            qois_kpi_da.attrs['array2d'] = np.reshape(qois_kpi_da.data, (-1, qois_kpi_da.shape[-1]))
        else:
            raise NotImplementedError("central positioning of the 'qois'-dimension not fully supported yet")

        # save the kpis to a csv file
        self.assessment_csv_handler.save(qois_kpi_da)

        return qois_kpi_da

    @staticmethod
    def calculate_ttc(responses):
        """
        This function calculates the Time-to-Collision (TTC) time signal.

        The TTC is the ratio between the distance and the relative velocity between the ego and a traffic vehicle.

        This function also corrects values from virtual CarMaker sensors, because CarMaker uses zero as default value
        if the traffic vehicle is out of the sensor range. However, zero can already have a meaning, such as an
        accident for TTC=0 or equal velocities for a relative velocity of zero.
        Thus, the default values are currently corrected to np.nan to avoid misleading overlaps.

        Required quantities:
        - 'AccelCtrl.ACC.Time2Collision': directly the TTC signal from the simulator, or
        - 'Sensor.Object.RadarL.relvTgt.NearPnt.ds.x': distance between ego and traffic vehicle, and
        - 'Sensor.Object.RadarL.relvTgt.NearPnt.dv.x': relative velocity between ego and traffic vehicle

        :param xr.DataArray responses: time responses from simulation or experiment
        :return: array of TTC signals
        :rtype: xr.DataArray
        """

        # initialize an xarray
        idx = responses.dims.index('quantities')
        dims = responses.dims[:idx] + responses.dims[idx + 1:]
        shape = responses.shape[:idx] + responses.shape[idx + 1:]
        ttc_array = np.empty(shape)
        ttc_da = xr.DataArray(ttc_array, dims=dims)

        # check if the TTC signal is directly available in the output quantities
        quantity_name_list = responses.quantities.values.tolist()
        if 'AccelCtrl.ACC.Time2Collision' in quantity_name_list:
            # extract the TTC from all quantities
            idx_ttc = quantity_name_list.index('AccelCtrl.ACC.Time2Collision')
            # use ellipsis indexing to be independent of the number of dimensions
            ttc_array[:] = responses[{'quantities': idx_ttc}].data

            # -- correct CarMaker TTC
            # CarMaker sets all sensor quantities to zero, if the traffic vehicle is out of range
            # Out of range also happens after an accident, because the ego vehicle drives through the traffic vehicle
            # Problem: TTC of zero can mean an accident or any out of range situation
            # Solution: replace zero with NaN and correct it in the accident case afterwards
            ttc_array[ttc_array == 0] = np.NaN

        # check if the distance and relative velocity to the traffic vehicle are available
        elif ('Sensor.Object.RadarL.relvTgt.NearPnt.ds.x' and 'Sensor.Object.RadarL.relvTgt.NearPnt.dv.x') in \
                quantity_name_list:

            # extract the relative velocity
            idx_v = quantity_name_list.index('Sensor.Object.RadarL.relvTgt.NearPnt.dv.x')
            dv_array = responses[{'quantities': idx_v}].data

            # -- correct CarMaker TTC
            # replace all positive values indicating a faster traffic vehicle with NaN
            # so that afterwards TTC = ds / |dv| can be used and also yields NaN in case of a faster traffic vehicle
            # (otherwise use TTC = ds / (-dv) and set the negative TTCs to NaN afterwards)
            dv_array[dv_array > 0] = np.NaN

            # extract the distance
            idx_s = quantity_name_list.index('Sensor.Object.RadarL.relvTgt.NearPnt.ds.x')
            ds_array = responses[{'quantities': idx_s}].data

            # CarMaker sets all sensor quantities to zero, if the traffic vehicle is out of range.
            # Out of range also happens after an accident, because the ego vehicle drives through the traffic vehicle.
            # Problem: distance of zero can mean an accident or any out of range situation
            # Problem: relative velocity of zero can mean same velocities, an accident or any out of range situation
            # Solution: division by zero in TTC calculation leads to NaN, correct it in the accident case afterwards

            # calculate ttc as ratio from distance to relative velocity
            ttc_array[:] = np.divide(ds_array, np.absolute(dv_array))
        else:
            raise

        # -- correct CarMaker TTC
        # an accident appears as the last transition from a valid value to NaN, if the preceding values decrease
        # (in case of a transition to out of sensor range they would increase)

        # get the index of the last valid number (except NaN)
        last_valid_index = (~np.isnan(ttc_array)).cumsum(-1).argmax(-1)

        # check if it is a transition from a valid value to NaN or the valid value appears at the last time step
        transition_mask = last_valid_index < (ttc_array.shape[-1] - 1)

        # determine if the TTC value decreases from the second last value to the last value
        second_last_valid_index = last_valid_index - 1
        idx_array = np.indices(last_valid_index.shape, sparse=False)
        ttc_last_values = ttc_array[(*tuple(idx_array), last_valid_index)]
        ttc_second_last_values = ttc_array[(*tuple(idx_array), second_last_valid_index)]
        decreasing_mask = ttc_last_values < ttc_second_last_values

        # correct the first accident (nan) value to zero if both masks are true
        first_invalid_index = last_valid_index[transition_mask & decreasing_mask] + 1
        idx_array = [idx_array[i][transition_mask & decreasing_mask] for i in range(idx_array.shape[0])]
        ttc_array[(*tuple(idx_array), first_invalid_index)] = 0

        return ttc_da

    @staticmethod
    def create_ecdf(kpi_da):
        """
        This function creats empirical cumulative distribution functions (ECDFs).

        It sorts the x-values and calculates the corresponding y-values / probabilities.
        The x values of the return array can be assessed via da.data.
        The y values of the return array can be assessed via da.probs.

        It keeps duplicate values in the data array that get paired with successive probabilities.
        This is realized by using only sort instead of unique.
        It can be imagined as invisible substeps within one actual step.
        It has the advantage that
        - the probs correspond to the 'linspace' function and
        - the data and probs array can use their full shape without the need for complex masking or nan-filling.
        The functions that deal with ECDFs are prepared for this ECDF representation.
        This includes the Metric, Error Integration, and Decision Making classes as well as the ECDF step plotting.
        From a strict mathematical function point of view, only the actual last step counts.

        The ECDF steps always refer to the 'repetition' or 'aleatory_samples' dimension.
        Theoretically, all 'qois', 'space_samples', or 'epistemic_samples' could have distinct repetitions and probs.
        However, in practice, only experiments with different repetitions for different space samples are of relevance.
        Each QoI and each epistemic sample has the same repetitions.
        Thus, we keep the probs array sparse and do not repeat it to the same shape as the data array.

        :param xr.DataArray kpi_da: array with the processed kpis
        :return: array with the processed kpis, converted to a ECDF representation
        :rtype: xr.DataArray
        """

        # get the correct index of the aleatory samples for the ecdf
        if 'aleatory_samples' in kpi_da.dims:
            idx = kpi_da.dims.index('aleatory_samples')
        elif 'repetitions' in kpi_da.dims:
            idx = kpi_da.dims.index('repetitions')
        else:
            raise ValueError("should be unreachable")

        # -- sort the aleatory samples to get the correct order for the ecdf (kpi_da.data -> x array)
        sorted_data_array = np.sort(kpi_da.data, axis=idx)

        # -- add a leading vector of -inf at the idx-dimension to create a step function (empirical cdf)
        # increment the shape value at the index by one (tuple syntax)
        shape = kpi_da.shape[:idx] + (kpi_da.shape[idx] + 1,) + kpi_da.shape[idx + 1:]

        # initialize a new xarray with -inf
        data = np.ones(shape) * (-np.inf)
        kpi_da = xr.DataArray(data, dims=kpi_da.dims, coords=kpi_da.coords, attrs=kpi_da.attrs)

        # insert the sorted array
        kpi_da[{kpi_da.dims[idx]: slice(1, None)}] = sorted_data_array

        # -- calculate the corresponding probabilities (kpi_da.probs -> y vector)
        if 'repetitions' in kpi_da.dims:

            # check if there are nan values indicating space samples with different number of repetitions
            nan_mask_da = np.isnan(kpi_da[{'qois': 0}])
            if nan_mask_da.data.sum():

                # get the index of the first invalid data (nan value)
                first_nan_idx = np.argmax(nan_mask_da.data, axis=nan_mask_da.dims.index('repetitions'))

                # get indices of zero indicating that all boolean values were False (no nan values in these rows at all)
                zero_mask = first_nan_idx == 0

                # check that not all values were True (full nan-slice)
                if nan_mask_da[{'space_samples': zero_mask}].sum():
                    raise ValueError("Each sample requires valid data.")

                # set the zero indices to the maximum number of repetitions
                first_nan_idx[zero_mask] = kpi_da.shape[kpi_da.dims.index('repetitions')]

                # create a list for the probability vector of each space sample
                kpi_da.attrs['probs'] = [np.linspace(0, 1, idx) for idx in first_nan_idx]

                # convert the list of 1D arrays to a 2D array with nan-filling
                # https://stackoverflow.com/questions/38619143/convert-python-sequence-to-numpy-array-filling-missing-values
                # https://stackoverflow.com/questions/43146266/convert-list-of-lists-with-different-lengths-to-a-numpy-array
                probs_array = np.array(list(itertools.zip_longest(*kpi_da.probs, fillvalue=0))).T
                kpi_da.attrs['probs_da'] = xr.DataArray(data=probs_array, dims=('space_samples', 'repetitions'))

            else:
                kpi_da.attrs['probs'] = np.linspace(0, 1, kpi_da.shape[idx])
                kpi_da.attrs['probs_da'] = xr.DataArray(data=kpi_da.probs, dims='repetitions')
        else:
            kpi_da.attrs['probs'] = np.linspace(0, 1, kpi_da.shape[idx])
            kpi_da.attrs['probs_da'] = xr.DataArray(data=kpi_da.probs, dims='aleatory_samples')

        return kpi_da

    @staticmethod
    def get_pbox_edges(kpi_da):
        """
        This function extracts both edges of the ECDFs within the input array.

        It replaces the 'epistemic_samples'-dimension with a 'pbox_edges'-dimension with a left and right label.

        :param xr.DataArray kpi_da: array with the processed kpis, converted to a ECDF representation
        :return: array with the processed kpis, converted to pbox edges representation
        """

        # epistemic_option_list = [{'qois', 'space_samples', 'epistemic_samples', 'aleatory_samples'},
        #                          {'qois', 'space_samples', 'epistemic_samples'}]
        # if set(kpi_da.dims) in epistemic_option_list:

        idx = kpi_da.dims.index('epistemic_samples')
        kpi_min_array = np.amin(kpi_da.data, axis=idx)
        kpi_max_array = np.amax(kpi_da.data, axis=idx)

        # replace the 'epistemic_samples'-dimension with one for the two pbox edges
        dims = kpi_da.dims[:idx] + ('pbox_edges',) + kpi_da.dims[idx + 1:]

        # stack the left and right pbox edges
        kpi_minmax_array = np.stack((kpi_min_array, kpi_max_array), axis=idx)

        # create the xarray
        kpi_da = xr.DataArray(kpi_minmax_array, dims=dims, coords=kpi_da.coords, attrs=kpi_da.attrs)

        # specify coordinate labels for the pbox_edges dimension
        kpi_da['pbox_edges'] = ('pbox_edges', ['left', 'right'])

        return kpi_da
