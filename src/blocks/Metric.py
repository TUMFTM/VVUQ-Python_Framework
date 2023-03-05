"""
This module is responsible for validation metrics.

It includes one class for the validation metrics. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --
import math as math

# -- third-party imports --
import numpy as np
import pandas as pd
import xarray as xr

# -- custom imports --
import src.variants.metrics.error_metrics as em
import src.variants.metrics.area_metrics as am
import src.variants.metrics.time_metrics as tm
import src.variants.metrics.distributional_metrics as dm


# -- MODULE-LEVEL VARIABLES --------------------------------------------------------------------------------------------
# -- create dictionaries with different types of validation metrics
area_metric_dict = {
    'avm': am.avm,
    'mavm': am.mavm,
    'iavm': am.iavm
}

distributional_metric_dict = {
    'mean_ci': dm.compare_means_with_ci,
    'ks_test': dm.ks_test
}

deviation_metric_dict = {
    'absolute_deviation': em.absolute_deviation,
    'relative_deviation': em.relative_deviation,
    'relative_deviation_2prediction': em.relative_deviation_2prediction,
}

transformed_deviation_metric_dict = {
    'transformed_deviation': 'dummy',
}

error_metric_dict = {
    'se': em.se,
    'me': em.me,
    'mne': em.mne,
    'mae': em.mae,
    'mane': em.mane,
    'rmse': em.rmse,
    'nrmse': em.nrmse,
    'r_squared': em.r_squared,
    'corrcoef': em.correlation_coefficient,
    'TheilsU': em.theils_u,
    'Oberkampf_2002': em.vm_oberkampf_2002
}

time_metric_dict = {
    'Russell': tm.russell_error_factors,
    'Sprague': tm.sprague_geers_error_factors
}


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Metric:
    """
    This class is responsible for validation metrics.

    It includes a main method called "calculate_metric" that calls the other methods.
    See more details in the documentation of the calculate_metric method.
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
        self.cfgme = self.config[domain]['metric']
        self.cfgdm = self.config[domain]['decision_making']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.pbox_y_model = np.ndarray(shape=(1,))
        self.pbox_y_system = np.ndarray(shape=(1,))
        self.pbox_x_model_list = []
        self.pbox_x_system_list = []

        self.metric_da = xr.DataArray(None)

    def calculate_metric(self, qois_model_da, qois_system_da):
        """
        This function calculates the selected validation metrics.

        It offers several options depending on the selected case, e.g.
        - area metrics for ECDFs and p-boxes
        - time series metrics for time signals
        - error metrics for scalars and time signals
        - etc.
        See more details in the documentation of the respective methods.

        :param xr.DataArray qois_model_da: array of responses from the simulation model
        :param xr.DataArray qois_system_da: array of responses from the system
        :return: array of validation metrics
        :rtype: xr.DataArray
        """

        ensemble_flag = False
        if ensemble_flag:
            # pool the results from all space samples to one sample with several repetitions
            qois_system_u_da = self.upooling_transform(qois_model_da, qois_system_da)
            qois_system_da = self.upooling_backtransform(qois_model_da, qois_system_u_da)

            # ToDo: add to config, check whether area metric works as global metric, exclude error model
            raise NotImplementedError("U-pooling is not fully implemented yet.")

        # -- distinguish different types of validation metrics
        if self.cfgme['metric'] in area_metric_dict.keys():
            # calculate area metric
            metric_da = self.calculate_area_metrics(qois_model_da, qois_system_da)

        elif self.cfgme['metric'] in distributional_metric_dict.keys():

            # dimension check
            if set(qois_model_da.dims) != {'qois', 'space_samples', 'repetitions'} or \
                    set(qois_model_da.dims) != {'qois', 'space_samples', 'aleatory_samples'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples', 'repetitions'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples', 'aleatory_samples'}:
                raise IndexError("selected metric can be applied only to CDFs.")

            if 'aleatory_samples' in qois_model_da.dims:
                idx_dim = qois_model_da.dims.index('aleatory_samples')
            elif 'repetitions' in qois_model_da.dims:
                idx_dim = qois_model_da.dims.index('repetitions')
            else:
                raise ValueError("should be unreachable.")

            # calculate probabilistic metric
            if self.cfgme['metric'] == "ks_test":
                raise NotImplementedError("loop through all qois and space samples and call the ks_test method.")
            else:
                metric_da = distributional_metric_dict[self.cfgme['metric']](qois_system_da, qois_model_da,
                                                                             axis=idx_dim)
        elif self.cfgme['metric'] in deviation_metric_dict.keys():

            # dimension check
            if set(qois_model_da.dims) != {'qois', 'space_samples'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples'}:
                raise IndexError("selected metric can be applied only to deterministic KPI responses.")

            # calculate deviation
            metric_da = deviation_metric_dict[self.cfgme['metric']](qois_system_da, qois_model_da)

        elif self.cfgme['metric'] in transformed_deviation_metric_dict.keys():

            # dimension check
            if set(qois_model_da.dims) != {'qois', 'space_samples'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples'}:
                raise IndexError("selected metric can be applied only to deterministic KPI responses.")

            # combine all thresholds in one data array
            thresh_array = np.concatenate((np.array(self.cfgdm['qois_lower_threshold_list'])[:, None],
                                           np.array(self.cfgdm['qois_upper_threshold_list'])[:, None]), axis=1)
            thresh_da = xr.DataArray(thresh_array, dims=('qois', 'threshold'),
                                     coords={'qois': self.cfgdm['qois_name_list'], 'threshold': ['lower', 'upper']})

            # determine the distances to the closest regulation threshold, respectively
            lower_dist_da = np.abs(qois_model_da - thresh_da.loc[{'threshold': 'lower'}])
            upper_dist_da = np.abs(qois_model_da - thresh_da.loc[{'threshold': 'upper'}])
            min_dist_da = np.minimum(lower_dist_da, upper_dist_da)

            # calculate deviation, weighted with the threshold distance in the denominator
            metric_da = (qois_model_da - qois_system_da) / min_dist_da

            # set nan values from "division by zero" to zeros
            # fine for error integration via uncertainty expansion and generally if the GT would also be zero
            metric_da.data[np.isnan(metric_da.data)] = 0

        elif self.cfgme['metric'] in error_metric_dict.keys():

            # dimension check
            if set(qois_model_da.dims) != {'qois', 'space_samples', 'timesteps'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples', 'timesteps'}:
                raise IndexError("selected metric can be applied only to time series responses.")

            # get the time dimension
            idx_time = qois_model_da.dims.index('timesteps')
            if idx_time != qois_system_da.dims.index('timesteps'):
                raise NotImplementedError("time dimension of model and system must match. reordering not implemented.")

            # calculate error metric
            if self.cfgme['metric'] == "nrmse":
                metric_da = error_metric_dict[self.cfgme['metric']](qois_system_da, qois_model_da,
                                                                    normalization="rmsne", axis=idx_time)
            else:
                metric_da = error_metric_dict[self.cfgme['metric']](qois_system_da, qois_model_da, axis=idx_time)

        elif self.cfgme['metric'] in time_metric_dict.keys():

            # dimension check
            if set(qois_model_da.dims) != {'qois', 'space_samples', 'timesteps'} or \
                    set(qois_system_da.dims) != {'qois', 'space_samples', 'timesteps'}:
                raise IndexError("selected metric can be applied only to time series responses.")

            # get the time dimension
            idx_time = qois_model_da.dims.index('timesteps')
            if idx_time != qois_system_da.dims.index('timesteps'):
                raise NotImplementedError("time dimension of model and system must match. reordering not implemented.")

            # calculate the time metric
            metric_da = time_metric_dict[self.cfgme['metric']](qois_system_da, qois_model_da, axis=idx_time)

        else:
            raise ValueError("selected metric not available.")

        return metric_da

    def calculate_area_metrics(self, qois_kpi_model_da, qois_kpi_system_da):
        """
        This function handles the non-deterministic area validation metrics.

        :param xr.DataArray qois_kpi_model_da: pbox edges of the model
        :param xr.DataArray qois_kpi_system_da: pbox edges of the system
        :return: probabilistic metric
        :rtype: xr.DataArray
        """

        if self.cfgme['metric'] == 'iavm':
            qois_kpi_system_da = self.iavm_get_limit_functions(qois_kpi_system_da)

        if 'pbox_edges' not in qois_kpi_model_da.dims:
            # create a degenerate p-box by using the ecdf as both left and right edges
            # concatenate along new dimension 'pbox_edges' with coordinate labels left and right
            # this enables a uniform p-box interface in the subsequent methods
            qois_kpi_model_da = xr.concat([qois_kpi_model_da, qois_kpi_model_da],
                                          pd.Index(['left', 'right'], name='pbox_edges'))  # type: xr.DataArray

            # reordering from first to penultimate position
            # optional if indexing is performed always with the dimension names
            dims = qois_kpi_model_da.dims[1:-1] + ('pbox_edges', qois_kpi_model_da.dims[-1])
            qois_kpi_model_da = qois_kpi_model_da.transpose(*dims)

        if 'pbox_edges' not in qois_kpi_system_da.dims:
            # same for the system as for the model
            qois_kpi_system_da = xr.concat([qois_kpi_system_da, qois_kpi_system_da],
                                           pd.Index(['left', 'right'], name='pbox_edges'))  # type: xr.DataArray
            dims = qois_kpi_system_da.dims[1:-1] + ('pbox_edges', qois_kpi_system_da.dims[-1])
            qois_kpi_system_da = qois_kpi_system_da.transpose(*dims)

        if 'aleatory_samples' in qois_kpi_model_da.dims:
            idx_dim = qois_kpi_model_da.loc[{'pbox_edges': 'left'}].dims.index('aleatory_samples')
        elif 'repetitions' in qois_kpi_model_da.dims:
            idx_dim = qois_kpi_model_da.loc[{'pbox_edges': 'left'}].dims.index('repetitions')
        else:
            raise NotImplementedError("aleatory samples of CDF or pbox yet required for area metric calculation.")

        # convert the xarray structures to the argument structure of the area metric calculations
        self.pbox_y_model = qois_kpi_model_da.probs
        self.pbox_y_system = qois_kpi_system_da.probs
        self.pbox_x_model_list = [qois_kpi_model_da.loc[{'pbox_edges': 'left'}].data,
                                  qois_kpi_model_da.loc[{'pbox_edges': 'right'}].data]
        self.pbox_x_system_list = [qois_kpi_system_da.loc[{'pbox_edges': 'left'}].data,
                                   qois_kpi_system_da.loc[{'pbox_edges': 'left'}].data]

        # check which metric is choosen
        if self.cfgme['metric'] == 'avm':
            metric = am.avm(self.pbox_y_model, self.pbox_y_system,
                            self.pbox_x_model_list, self.pbox_x_system_list, idx_dim)
        elif self.cfgme['metric'] == 'mavm':
            metric = am.mavm(self.pbox_y_model, self.pbox_y_system,
                             self.pbox_x_model_list, self.pbox_x_system_list, idx_dim,
                             self.cfgme['mavm_f0'], self.cfgme['mavm_f1'])
        elif self.cfgme['metric'] == 'iavm':
            # the area metric functions are not yet prepared for pbox edges / limit functions with distinct probs
            raise NotImplementedError("The IAVM metric is not fully implemented yet.")

            metric = am.iavm(self.pbox_y_model, self.pbox_y_system,
                             self.pbox_x_model_list, self.pbox_x_system_list, idx_dim)
        else:
            raise ValueError("selected metric not available")

        # -- create the xarray
        # add interval dim instead of pbox_edges and aleatory_samples (assuming qois before space_samples)
        dims = ('qois', 'space_samples', 'interval')
        # specify the corresponding coordinate labels
        coords = {'qois': qois_kpi_model_da.qois.values.tolist(), 'interval': ['left', 'right']}
        # create the DataArray
        self.metric_da = xr.DataArray(metric, dims=dims, coords=coords)

        return self.metric_da

    def iavm_get_limit_functions(self, qois_kpi_system_da):
        """
        This function calculates the two limit functions representing the uncertainty in the measurement cdf.

        The lower and upper limit function are required for the Interval Area Validation Metric from [1].
        They represent the uncertainty due to the finite number of experimental repetitions.

        Literature:
        [1] N. Wang, W. Yao, Y. Zhao, X. Chen, X. Zhang and L. Li, „A New Interval Area Metric for
        Model Validation With Limited Experimental Data,“ Journal of Mechanical Design, vol.
        140, no. 6, 2018.
    
        :param xr.DataArray qois_kpi_system_da: array of system results
        :return: array of left and right limit functions
        :rtype: tuple(np.ndarray, np.ndarray)
        """

        if 'pbox_edges' in qois_kpi_system_da.dims or 'repetitions' not in qois_kpi_system_da.dims:
            raise TypeError("The IAVM limit functions assume a single experimental ECDF.")

        # calc z_0 of the Dvoretzky-Kiefer-Wolfowitz inequality at a given confidence alpha
        z_0 = math.sqrt((-1 / 2) * math.log(self.cfgme['iavm_alpha'] / 2))

        # get the number of experimental observations/repetitions
        n = qois_kpi_system_da.shape[qois_kpi_system_da.dims.index('repetitions')]

        # calc z_0 / sqrt(n) as the offset to shift the CDF function
        z_0_n = z_0 / math.sqrt(n)

        if z_0_n >= 1:
            raise ValueError("The number of repetitions and the confidence of the IAVM metric must be chosen " +
                             "so that the offset z_0 / sqrt(n) is smaller than 1.")

        probs_array = qois_kpi_system_da.probs_da.data

        # -- determine the lower limit function
        lower_probs_array = np.ones(probs_array.shape) * np.nan

        # set the first ECDF steps to zero depending on the confidence
        lower_less_condition = probs_array <= z_0_n
        lower_probs_array[lower_less_condition] = 0

        # shift the ECDF downwards
        lower_greater_condition = (z_0_n < probs_array) & (probs_array < 1)
        lower_probs_array[lower_greater_condition] = probs_array[lower_greater_condition] - z_0_n

        # set the last ECDF step to 1
        lower_one_condition = probs_array == 1
        lower_probs_array[lower_one_condition] = 1

        # -- determine the upper limit function
        upper_probs_array = np.ones(probs_array.shape) * np.nan

        # set the first ECDF step to zero
        upper_zero_condition = probs_array == 0
        upper_probs_array[upper_zero_condition] = 0

        # shift the ECDF upwards
        upper_less_condition = (0 < probs_array) & ((probs_array + z_0_n) < 1)
        upper_probs_array[upper_less_condition] = probs_array[upper_less_condition] + z_0_n

        # set the last ECDF steps to zero depending on the confidence
        upper_greater_condition = (probs_array + z_0_n) >= 1
        upper_probs_array[upper_greater_condition] = 1

        # update the probs array to the probs of the upper and lower limit function
        probs_da = xr.DataArray(data=[upper_probs_array, lower_probs_array],
                                dims=('pbox_edges',) + qois_kpi_system_da.probs_da.dims,
                                coords={'pbox_edges': ['left', 'right']})

        if 'space_samples' in probs_da.dims:
            # swap 'space_samples' and 'pbox_edges'
            # optional if indexing is performed always with the dimension names
            dims = ('space_samples', 'pbox_edges', 'repetitions')
            qois_kpi_system_da = qois_kpi_system_da.transpose(*dims)

        qois_kpi_system_da.attrs['probs_da'] = probs_da

        # plot the original ECDF and the lower and upper limit function
        # import matplotlib.pyplot as plt
        # plt.step(qois_kpi_system_da[0, 0, :], probs_array, where='post', label='System')
        # plt.step(qois_kpi_system_da[0, 0, :], lower_probs_array, where='post', label='Lower Limit')
        # plt.step(qois_kpi_system_da[0, 0, :], upper_probs_array, where='post', label='Upper Limit')
        # plt.legend()

        return qois_kpi_system_da

    @staticmethod
    def upooling_transform(qois_model_da, qois_system_da):
        """
        This function performs the transformation to u-values for the u-pooling method.

        The u-pooling theory can be found in [1, Ch. 12.8.3].
        Q. He extends it to model predictions in the form of p-boxes in [2].
        During the initial transformation to u values, he vertically intersects a system value with a model p-box.
        The min u value corresponds to the right p-box edge and the max u value to the left p-box edge, since CDFs rise.
        Thus, the min/max to left/right mapping is inverted along the u/probs-dimension and
        will be dealt with during back-transformation.

        Literature:
        [1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
        Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.
        [2] Q. He, Model validation based on probability boxes under mixed uncertainties.
        Adv Mech Eng 11(5), 2019

        :param xr.DataArray qois_model_da: array of responses from the simulation model
        :param xr.DataArray qois_system_da: array of responses from the system
        :return: array of transformed u values of the system
        :rtype: xr.DataArray
        """
        if 'repetitions' in qois_system_da.dims:
            # for debugging
            qois_system_da = qois_system_da.loc[{'repetitions': 1}]

            # our main use case for u-pooling is when we do not have experimental repetitions at all
            # then we use u-pooling to aggregate data across several space samples to get repetitions at one sample
            # thus, a use case where we have a few repetitions but we want to use u-pooling to get more, is yet neither
            # relevant nor implemented
            # it is also not defined whether to calc one intersection between the system ECDF and model pbox
            # (loop-based implementation available) or probably whether to separately intersect each repetition
            # raise NotImplementedError("U-pooling is not yet implemented for multiple experimental repetitions.")

        # get the indices where the system values intersect the model pbox edges (vertical line from x-axis upwards)
        # this is basically an alternative to a (not available) nd searchsorted function with an axis argument
        idx_intersection_da = (qois_model_da < qois_system_da).sum(dim='aleatory_samples') - 1

        # get the u values by indexing the probs array (horizontal line to the y-axis)
        qois_system_u_da = qois_model_da.probs_da[idx_intersection_da]

        # quick plot to visualize the intersection
        # import matplotlib.pyplot as plt
        # plt.step(qois_model_da[0, 0, 0, :], qois_model_da.probs, where='post', label='Model CDF')
        # plt.axvline(x=qois_system_da[0, 0], linestyle='--', label='System Data')
        # plt.axhline(y=qois_system_u_da[0, 0, 0], linestyle='--', label='System U')
        # plt.legend()

        return qois_system_u_da

    @staticmethod
    def upooling_backtransform(qois_model_da, qois_system_u_da):
        """
        This function back-transforms the u values and pools them from several space samples to one.

        The u-pooling theory can be found in [1, Ch. 12.8.3].
        Q. He extends u-pooling to model predictions in the form of p-boxes in [2].
        During the initial transformation to u values, he vertically intersects a system value with a model p-box.
        The min u value corresponds to the right p-box edge and the max u value to the left p-box edge, since CDFs rise.
        During back-transformation, he horizontally intersects both u values with both p-box edges to obtain 4 values.
        Then he proceeds with the min and max of the 4 data values as the new interval boundaries.
        The min value corresponds to the intersection of the min u value with the left p-box edge and the max value to
        the intersection of the max u value to the right p-box edge, since CDFs rise and p-box edges do not cross.
        However, this is exactly the opposite compared to the initial transformation.
        To implement this, we could swap the 'left' and 'right' coordinates of the 'pbox_edges' dimension.
        This basically transforms scalar (vertical) system values to wide and overly conservative intervals.

        Instead, we keep the initial mapping from the transformation also during the back-transformation (w/o swapping):
        a) min u value from right transformation p-box edge with right back-transformation p-box edge
        b) max u value from left transformation p-box edge with left back-transformation p-box edge
        This leads to tighter intervals closer to the scalar system value.
        Whether a) or b) leads to the min/left or max/right interval boundaries depends on the course of the p-box.

        This topic could be discussed in the future.

        Literature:
        [1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
        Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.
        [2] Q. He, Model validation based on probability boxes under mixed uncertainties.
        Adv Mech Eng 11(5), 2019

        :param xr.DataArray qois_model_da: array of responses from the simulation model
        :param xr.DataArray qois_system_u_da: array of transformed u values of the system
        :return: array of back-transformed responses from the system
        :rtype: xr.DataArray
        """

        # select the backtransformation point
        qois_model_da = qois_model_da.loc[{'space_samples': 0}]

        # get the indices where the system u values intersect the model pbox edges (horizontal line from y-axis)
        # if the probs coincide so that the intersection occurs at a horizontal step, we currently take the left x value
        idx_intersection_da = (qois_model_da.probs_da < qois_system_u_da).sum(dim='aleatory_samples')

        # in the zero prob special case, we do not take the left step value (-inf) but the right one
        idx_intersection_da.data[idx_intersection_da.data == 0] = 1

        # put the aleatory_samples dimension at the first place so that the indexing works
        dim_idx = qois_model_da.dims.index('aleatory_samples')
        dims = ('aleatory_samples', ) + qois_model_da.dims[:dim_idx] + qois_model_da.dims[dim_idx + 1:]
        qois_model_da_transposed = qois_model_da.transpose(*dims)

        # get the x values by indexing the model data (vertical line downwards to x-axis)
        qois_system_da = qois_model_da_transposed[idx_intersection_da]

        # sort the values for the p-box edges since what is left/right depends on the course of the model p-box
        # this could also be solved differently for left-right value pairs, e.g., by min/max or by boolean comparison
        qois_system_da.data.sort(axis=qois_system_da.dims.index('pbox_edges'))

        # sort the values along the old space_samples dimension to construct CDFs
        qois_system_da.data.sort(axis=qois_system_da.dims.index('space_samples'))

        # add leading -inf for our ECDF representation
        # dropping the space samples dimension by indexing to create the inf_da is short (but maybe not the fastest)
        inf_da = xr.ones_like(qois_system_da.loc[{'space_samples': 0}]) * np.NINF
        qois_system_da = xr.concat([inf_da, qois_system_da], 'space_samples')

        # rename the space_samples to aleatory_samples and automatically transpose the aleatory_samples dim to the end,
        # since we pooled the data from several space samples to a single space sample with aleatory samples
        qois_system_da = qois_system_da.rename({'space_samples': 'aleatory_samples'})

        # create probs
        qois_system_da.attrs['probs'] = \
            np.linspace(0, 1, qois_system_da.shape[qois_system_da.dims.index('aleatory_samples')])
        qois_system_da.attrs['probs_da'] = xr.DataArray(data=qois_system_da.probs, dims='aleatory_samples')

        return qois_system_da
