"""
This module is responsible for the integration / aggregation of simulation errors and uncertainties.

It includes one class for the error integration. See details in its own documentation.

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
from src.blocks.Metric import area_metric_dict


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class ErrorIntegration:
    """
    This class is responsible for the integration / aggregation of simulation errors and uncertainties.

    It includes a main method called "error_integration". See more details in its own documentation.
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
        self.cfgei = self.config[domain]['error_integration']
        self.cfgme = self.config['validation']['metric']
        self.cfgdm = self.config[domain]['decision_making']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.error_validation_rep_da = xr.DataArray(None)
        self.numerical_rep_da = xr.DataArray(None)
        self.kpi_system_estimated_validation_da = xr.DataArray(None)
        self.kpi_system_estimated_da = xr.DataArray(None)

    def error_integration(self, kpi_model_da, error_validation_da, numerical_uncertainty_da):
        """
        This function aggregates errors from validation and verification to the model predictions in the application.

        It offers two techniques:
        - bias correction: use the errors to correct the nominal model predictions,
        - uncertainty expansion: add conservatism by expanding uncertainties.
        In the deterministic case, we offer both options; in the non-deterministic case, just the uncertainty expansion.

        :param kpi_model_da: array of nominal model responses
        :param error_validation_da: model-form errors or uncertainties from the validation domain
        :param numerical_uncertainty_da: numerical uncertainties from the verification domain
        :return: array of aggregated model predictions as estimator of the actual system values
        :rtype: xr.DataArray
        """

        # distinguish the non-deterministic and deterministic case
        if 'pbox_edges' in kpi_model_da.dims or 'repetitions' in kpi_model_da.dims:
            # -- non-deterministic case

            if self.cfgei['method'] != 'uncertainty_expansion':
                raise ValueError("we only support a uncertainty expansion for non-deterministic simulations.")

            # get the correct index of the aleatory samples or repetitions
            if 'aleatory_samples' in kpi_model_da.dims:
                idx = kpi_model_da.dims.index('aleatory_samples')
                dim_name = 'aleatory_samples'
            elif 'repetitions' in kpi_model_da.dims:
                idx = kpi_model_da.dims.index('repetitions')
                dim_name = 'repetitions'
            else:
                raise ValueError("should be unreachable")

            # repeat the error array times the number of aleatory samples as all of them must be shifted
            self.error_validation_rep_da = xr.concat(
                [error_validation_da] * kpi_model_da.shape[idx], dim_name)  # type: xr.DataArray

            # -- UNCERTAINTY EXPANSION ---------------------------------------------------------------------------------

            if 'pbox_edges' in kpi_model_da.dims:
                # shift the left pbox edge to the left
                kpi_system_estimated_validation_left_da =\
                    kpi_model_da.loc[{'pbox_edges': 'left'}] - self.error_validation_rep_da.loc[{'interval': 'left'}]

                # shift the right pbox edge to the right
                kpi_system_estimated_validation_right_da = \
                    kpi_model_da.loc[{'pbox_edges': 'right'}] + self.error_validation_rep_da.loc[{'interval': 'right'}]

            else:
                # shift the cdf to the left
                kpi_system_estimated_validation_left_da = \
                    kpi_model_da - self.error_validation_rep_da.loc[{'interval': 'left'}]

                # shift the cdf to the right
                kpi_system_estimated_validation_right_da = \
                    kpi_model_da + self.error_validation_rep_da.loc[{'interval': 'right'}]

            if numerical_uncertainty_da.dims != ():
                # -- if numerical uncertainties exist, do the same shifts also for them
                self.numerical_rep_da = xr.concat(
                    [numerical_uncertainty_da] * kpi_model_da.shape[idx], dim_name)  # type: xr.DataArray

                kpi_system_estimated_left_da = kpi_system_estimated_validation_left_da - self.numerical_rep_da
                kpi_system_estimated_right_da = kpi_system_estimated_validation_right_da + self.numerical_rep_da
            else:
                kpi_system_estimated_left_da = kpi_system_estimated_validation_left_da
                kpi_system_estimated_right_da = kpi_system_estimated_validation_right_da

            # combine both edges in one pbox structure
            self.kpi_system_estimated_validation_da = xr.concat(
                [kpi_system_estimated_left_da, kpi_system_estimated_right_da], 'pbox_edges')  # type: xr.DataArray
            self.kpi_system_estimated_da = xr.concat(
                [kpi_system_estimated_left_da, kpi_system_estimated_right_da], 'pbox_edges')  # type: xr.DataArray

            # set the coordinate labels (necessary if pbox_edges not yet in dims before)
            self.kpi_system_estimated_validation_da = self.kpi_system_estimated_validation_da.assign_coords(
                {'pbox_edges': ['left', 'right']})
            self.kpi_system_estimated_da = self.kpi_system_estimated_da.assign_coords(
                {'pbox_edges': ['left', 'right']})

            # create the desired order of the dimensions again
            self.kpi_system_estimated_validation_da = self.kpi_system_estimated_validation_da.transpose(
                'qois', 'space_samples', 'pbox_edges', ...)
            self.kpi_system_estimated_da = self.kpi_system_estimated_da.transpose(
                'qois', 'space_samples', 'pbox_edges', ...)

            # add probs to attrs dict
            self.kpi_system_estimated_validation_da.attrs['probs'] = kpi_model_da.probs
            self.kpi_system_estimated_da.attrs['probs'] = kpi_model_da.probs

        else:
            # -- deterministic case
            # error = model - system -> (estimated) system = model - error

            if 'interval' in error_validation_da.dims:
                # -- deterministic metric with prediction intervals

                if self.cfgei['method'] in {'bias_correction', 'uncertainty_expansion'}:
                    # we interpret the 'uncertainty_expansion' as an extension (correction) of the 'bias_correction'

                    if self.cfgme['metric'] in area_metric_dict:
                        # shift the model value with the left error (positive area, its upper PI bound) to the left
                        kpi_system_estimated_left_da = kpi_model_da - error_validation_da.loc[{'interval': 'left'}]

                        # shift the model value with the right error (positive area, its upper PI bound) to the right
                        kpi_system_estimated_right_da = kpi_model_da + error_validation_da.loc[{'interval': 'right'}]

                    else:
                        # subtract the upper PI bound of the error to get the lower system value
                        kpi_system_estimated_left_da = kpi_model_da - error_validation_da.loc[{'interval': 'right'}]

                        # subtract the lower PI bound of the error to get the upper system value
                        kpi_system_estimated_right_da = kpi_model_da - error_validation_da.loc[{'interval': 'left'}]

                    if self.cfgme['metric'] == 'transformed_deviation':
                        # for this validation metric, we need to compensate the transformation in the error integration

                        # combine all thresholds in one data array
                        thresh_array = np.concatenate((np.array(self.cfgdm['qois_lower_threshold_list'])[:, None],
                                                       np.array(self.cfgdm['qois_upper_threshold_list'])[:, None]),
                                                      axis=1)
                        thresh_da = xr.DataArray(thresh_array, dims=('qois', 'threshold'),
                                                 coords={'qois': self.cfgdm['qois_name_list'],
                                                         'threshold': ['lower', 'upper']})

                        # determine the distances to the closest regulation threshold, respectively
                        lower_dist_da = np.abs(kpi_model_da - thresh_da.loc[{'threshold': 'lower'}])
                        upper_dist_da = np.abs(kpi_model_da - thresh_da.loc[{'threshold': 'upper'}])
                        min_dist_da = np.minimum(lower_dist_da, upper_dist_da)

                        # compensate the denominator from the validation metric by multiplying with it now
                        kpi_system_estimated_left_da = kpi_system_estimated_left_da * min_dist_da
                        kpi_system_estimated_right_da = kpi_system_estimated_right_da * min_dist_da

                    if self.cfgei['method'] == 'uncertainty_expansion':
                        # we interpret the uncertainty expansion of the deterministic simulation here,
                        # as ensuring that the model values are always included in the estimation bounds.
                        # if that is not the case (both bounds smaller or greater than the model),
                        # we correct the bounds so that the model value is included (represents one bound)

                        # if the right bound of the system value is smaller than the model, both bounds are smaller
                        both_smaller_mask_da = kpi_system_estimated_right_da < kpi_model_da

                        # in true cases (both smaller), correct the right bound to the (greater) model value
                        kpi_system_estimated_right_da.data[both_smaller_mask_da.data] = \
                            kpi_model_da.data[both_smaller_mask_da.data]

                        # if the left bound of the system value is greater than the model, both bounds are greater
                        both_greater_mask_da = kpi_system_estimated_left_da > kpi_model_da

                        # in true cases (both greater), correct the left bound to the (smaller) model value
                        kpi_system_estimated_left_da.data[both_greater_mask_da.data] = \
                            kpi_model_da.data[both_greater_mask_da.data]

                    # combine both interval values in one data array
                    self.kpi_system_estimated_da = xr.concat(
                        [kpi_system_estimated_left_da, kpi_system_estimated_right_da], 'interval')  # type: xr.DataArray

                    # switch the coordinate labels, since they are coming from the inverse error labels
                    self.kpi_system_estimated_da = self.kpi_system_estimated_da.assign_coords(
                        {'interval': ['left', 'right']})

                    # create the desired order of the dimensions again
                    self.kpi_system_estimated_da = self.kpi_system_estimated_da.transpose(..., 'interval')

                else:
                    raise ValueError("The selected error integration method is not available.")

            else:
                # -- deterministic case without prediction intervals

                if self.cfgei['method'] == 'bias_correction':

                    # directly subtract the value
                    kpi_system_estimated_da = kpi_model_da - error_validation_da

                    if self.cfgme['metric'] == 'transformed_deviation':
                        # for this validation metric, we need to compensate the transformation in the error integration

                        # combine all thresholds in one data array
                        thresh_array = np.concatenate((np.array(self.cfgdm['qois_lower_threshold_list'])[:, None],
                                                       np.array(self.cfgdm['qois_upper_threshold_list'])[:, None]),
                                                      axis=1)
                        thresh_da = xr.DataArray(thresh_array, dims=('qois', 'threshold'),
                                                 coords={'qois': self.cfgdm['qois_name_list'],
                                                         'threshold': ['lower', 'upper']})

                        # determine the distances to the closest regulation threshold, respectively
                        lower_dist_da = np.abs(kpi_model_da - thresh_da.loc[{'threshold': 'lower'}])
                        upper_dist_da = np.abs(kpi_model_da - thresh_da.loc[{'threshold': 'upper'}])
                        min_dist_da = np.minimum(lower_dist_da, upper_dist_da)

                        # compensate the denominator from the validation metric by multiplying with it now
                        kpi_system_estimated_da = kpi_system_estimated_da * min_dist_da

                    self.kpi_system_estimated_da = kpi_system_estimated_da

                elif self.cfgei['method'] == 'uncertainty_expansion':
                    raise ValueError("uncertainty expansion with deterministic metric without PIs is not possible.")

                else:
                    raise ValueError("the selected error integration method is not available.")

        return self.kpi_system_estimated_da
