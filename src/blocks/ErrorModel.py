"""
This module is responsible for training and inference of error models.

It includes one class for the error models. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import numpy as np
import xarray as xr
from scipy.stats import norm
from PyIPM import IPM

# -- custom imports --
from src.variants.metamodels import regression
from src.variants.metamodels import kriging
# from src.variants.metamodels import neural_networks


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class ErrorModel:
    """
    This class is responsible for training and inference of error models.

    It includes two main methods called "train_models" and "infer_models" that offer different meta-modeling techniques.

    Most techniques are included from own external modules.
    In case of IPMs, two thin wrapper functions are required: train_ipm_model and infer_ipm_model.

    See method details in their own documentations.
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
        self.cfgem = self.config[domain]['error_model']

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.model_list = []

        if self.cfgem['method'] == 'linear_regression':
            self.design_array_list = []
            self.mse_list = []
            self.t_value_list = []

        self.interval_flag = False

        self.error_da = xr.DataArray(None)
        self.error_full_da = xr.DataArray(None)
        self.qoi_name_list = []

    def train_models(self, space_scenarios_da, metric_da):
        """
        This function trains an error model of validation metric results across the scenario space.

        :param xr.DataArray space_scenarios_da: data array of scenarios
        :param xr.DataArray metric_da: data array of validation metric results
        :return:
        """

        # store some information for model inference later
        if 'interval' in metric_da.dims:
            self.interval_flag = True
        self.qoi_name_list = metric_da.qois.values.tolist()

        # skip the error modeling, if the user has not selected it
        if self.cfgem['method'] == "none":
            return

        # take only the parameters that the user selected for extrapolation (reduce DoF and model complexity)
        idx_dict = dict.fromkeys(space_scenarios_da.dims, 0)
        idx_dict['space_samples'] = slice(None)
        idx_dict['parameters'] = self.cfgem['extrapolation_parameters']
        scenarios_extrapolation_da = space_scenarios_da.loc[idx_dict]

        # loop through the QoIs and possibly the left and right interval boundaries
        self.model_list, self.design_array_list, self.mse_list, self.t_value_list = [], [], [], []
        idx_space = metric_da.dims.index('space_samples')
        shape_wo_space = metric_da.shape[:idx_space] + metric_da.shape[idx_space + 1:]

        for idx in np.ndindex(shape_wo_space):
            idx_tuple = idx[:idx_space] + (slice(None),) + idx[idx_space:]
            idx_dict = dict(zip(metric_da.dims, idx_tuple))

            # extract the metric for all space_samples, but just one QoI and metric boundary
            metric_vector = metric_da[idx_dict].data

            # distinguish between modeling methods
            if self.cfgem['method'] == "linear_regression":

                # train a linear regression model
                model = regression.train_linear_regression_model(scenarios_extrapolation_da.data, metric_vector)

                if self.cfgem['prediction_interval']:
                    # prepare the calculation of external prediction intervals for inference
                    design_array, mse, t_value = regression.prepare_lr_prediction_interval(
                        scenarios_extrapolation_da.data, metric_vector, model, self.cfgem['alpha'])

                    # append results to lists
                    self.design_array_list.append(design_array)
                    self.mse_list.append(mse)
                    self.t_value_list.append(t_value)

            elif self.cfgem['method'] == "polynomial_regression":

                # train a polynomial regression model
                model = regression.train_polynomial_regression_model(scenarios_extrapolation_da.data, metric_vector,
                                                                     self.cfgem['degree'])

                if self.cfgem['prediction_interval']:
                    # prepare the calculation of external prediction intervals for inference
                    design_array, mse, t_value = regression.prepare_pr_prediction_interval(
                        scenarios_extrapolation_da.data, metric_vector, model, self.cfgem['degree'],
                        self.cfgem['alpha'])

                    # append results to lists
                    self.design_array_list.append(design_array)
                    self.mse_list.append(mse)
                    self.t_value_list.append(t_value)

            elif self.cfgem['method'] == "kriging":

                # train a Gaussian Process (kriging) model (offers confidence intervals)
                model = kriging.train_gp_model(scenarios_extrapolation_da.data, metric_vector)

            elif self.cfgem['method'] == "MLP":

                # train a multi-layer perceptron (MLP) model
                # model = neural_networks.train_mlp_model(scenarios_extrapolation_da.data, metric_vector)
                pass

            elif self.cfgem['method'] == "IPM":

                # train an interval predictor model (directly includes upper and lower bounds)
                model = self.train_ipm_model(scenarios_extrapolation_da.data, metric_vector, self.cfgem['degree'])

            else:
                raise ValueError("this error modeling technique is not available")

            # append results to lists
            self.model_list.append(model)

        return

    def infer_models(self, scenarios_da):
        """
        This function uses the trained error model to infer errors in the application domain for new scenarios.

        :param scenarios_da: data array of scenarios
        :return:
        """

        # skip the error modeling, if the user has not selected it
        if self.cfgem['method'] == "none":
            # return a data array of zeros, so that nothing will be added in the error integration
            error_array = np.zeros(shape=(
                len(self.qoi_name_list), scenarios_da.shape[scenarios_da.dims.index('space_samples')], 2))
            self.error_da = xr.DataArray(error_array, dims=('qois', 'space_samples', 'interval'),
                                         coords={'qois': self.qoi_name_list, 'interval': ['left', 'right']})
            return self.error_da

        # take only the parameters that the user selected for extrapolation (reduce DoF and model complexity)
        idx_dict = dict.fromkeys(scenarios_da.dims, 0)
        idx_dict['space_samples'] = slice(None)
        idx_dict['parameters'] = self.cfgem['extrapolation_parameters']
        scenarios_extrapolation_da = scenarios_da.loc[idx_dict]

        # -- create arrays for the infered errors / uncertainties
        idx_space = scenarios_da.dims.index('space_samples')
        if self.interval_flag:
            # if model-form uncertainty described by an interval:
            # pairs of models in the list correspond to the same QoI -> (len(self.model_list) // 2
            # the first element of the pair corresponds to the left interval boundary, the second to the right -> last 2
            # independent of whether prediction intervals were selected, we can intervals here anyway
            error_array = np.ndarray(shape=(len(self.model_list) // 2, scenarios_da.shape[idx_space], 2))
            dims = ('qois', 'space_samples', 'interval')
            self.error_da = xr.DataArray(error_array, dims=dims)
            self.error_da['interval'] = ['left', 'right']

            if self.cfgem['prediction_interval']:
                # create further array to store the full information of the prediction interval combinations
                error_full_array = np.ndarray(shape=(len(self.model_list) // 2, scenarios_da.shape[idx_space], 2, 3))
                full_dims = ('qois', 'space_samples', 'interval', 'prediction_interval')
                self.error_full_da = xr.DataArray(error_full_array, dims=full_dims)
                self.error_full_da['interval'] = ['left', 'right']
                self.error_full_da['prediction_interval'] = ['lower', 'regression', 'upper']
            else:
                self.error_full_da = self.error_da

        else:
            if self.cfgem['prediction_interval']:
                # if model-form error, each error model in the list corresponds to a new QoI -> len(self.model_list)
                # considering prediction uncertainty of the error model itself, we still get an interval -> last "2"
                error_array = np.ndarray(shape=(len(self.model_list), scenarios_da.shape[idx_space], 2))
                dims = ('qois', 'space_samples', 'interval')
                self.error_da = xr.DataArray(error_array, dims=dims)
                self.error_da['interval'] = ['left', 'right']

                # create further array to store the full information of the prediction interval combinations
                error_full_array = np.ndarray(shape=(len(self.model_list), scenarios_da.shape[idx_space], 3))
                full_dims = ('qois', 'space_samples', 'prediction_interval')
                self.error_full_da = xr.DataArray(error_full_array, dims=full_dims)
                self.error_full_da['prediction_interval'] = ['lower', 'regression', 'upper']

            else:
                # if not considering prediction uncertainty, we get no interval
                error_array = np.ndarray(shape=(len(self.model_list), scenarios_da.shape[idx_space]))
                dims = ('qois', 'space_samples')
                self.error_da = xr.DataArray(error_array, dims=dims)
                self.error_full_da = self.error_da

        # loop through all error models
        for i in range(len(self.model_list)):

            if self.cfgem['method'] in {'linear_regression', 'polynomial_regression', 'kriging'}:

                if self.cfgem['method'] == "linear_regression":

                    # use the linear error model for inference
                    error_estimate = regression.infer_linear_regression_model(scenarios_extrapolation_da.data,
                                                                              self.model_list[i])

                    if self.cfgem['prediction_interval']:
                        # calculate an external prediction interval for the linear regression model
                        prediction_interval = regression.calc_prediction_interval(
                            scenarios_extrapolation_da.data, self.design_array_list[i], self.mse_list[i],
                            self.t_value_list[i])

                elif self.cfgem['method'] == "polynomial_regression":

                    # use the polynomial error model for inference
                    error_estimate = regression.infer_polynomial_regression_model(
                        scenarios_extrapolation_da.data, self.model_list[i], self.cfgem['degree'])

                    if self.cfgem['prediction_interval']:
                        # calculate an external prediction interval for the polynomial regression model
                        prediction_interval = regression.calc_prediction_interval(
                            scenarios_extrapolation_da.data, self.design_array_list[i], self.mse_list[i],
                            self.t_value_list[i])

                else:

                    # use the GP error model for inference
                    error_estimate, std = kriging.infer_gp_model(scenarios_extrapolation_da.data, self.model_list[i])

                    if self.cfgem['prediction_interval']:
                        # the probability that the data lies within a z-sigma band is: p = 2 * norm.cdf(z) - 1
                        # 2 * norm.cdf(1) - 1 = 0.68; 2 * norm.cdf(2) - 1 = 0.95; 2 * norm.cdf(3) - 1 = 0.997
                        # and vice versa, the width of the band for a given p is: z = norm.ppf((p + 1) / 2)
                        # norm.ppf((0.68 + 1) / 2) = 1; norm.ppf((0.95 + 1) / 2) = 2; norm.ppf((0.997 + 1) / 2) = 3
                        # https://de.wikipedia.org/wiki/Normalverteilung
                        p = 1 - self.cfgem['alpha']
                        z = norm.ppf((p + 1) / 2)

                        # multiple z with the standard deviation from the GP to get the confidence interval
                        prediction_interval = z * std

                if self.interval_flag:
                    if self.cfgem['prediction_interval']:
                        # assign the inferred errors to the full data array
                        self.error_full_da[{'qois': i // 2, 'interval': i % 2, 'prediction_interval': 0}] = \
                            error_estimate - prediction_interval
                        self.error_full_da[{'qois': i // 2, 'interval': i % 2, 'prediction_interval': 1}] = \
                            error_estimate
                        self.error_full_da[{'qois': i // 2, 'interval': i % 2, 'prediction_interval': 2}] = \
                            error_estimate + prediction_interval

                        # add the relative prediction interval value to the positive area metric estimate
                        # discard subtracting the relative PI value from the positive area metric estimate,
                        # since it will anyway be included in the uncertainty expansion of the positive worst case value
                        error_estimate = error_estimate + prediction_interval

                    # assign the inferred errors to the data array
                    self.error_da[{'qois': i // 2, 'interval': i % 2}] = error_estimate

                    # if not self.cfgem['prediction_interval']:
                    #     self.error_full_da = self.error_da

                else:
                    if self.cfgem['prediction_interval']:
                        # subtract the prediction uncertainty from the left error estimate and assign to the data array
                        self.error_da.loc[{'qois': i, 'interval': 'left'}] = error_estimate - prediction_interval

                        # add the prediction uncertainty to the right error estimate and assign to the data array
                        self.error_da.loc[{'qois': i, 'interval': 'right'}] = error_estimate + prediction_interval

                        # assign the inferred errors to the full data array
                        self.error_full_da.loc[{'qois': i, 'prediction_interval': 'lower'}] = \
                            error_estimate - prediction_interval
                        self.error_full_da.loc[{'qois': i, 'prediction_interval': 'regression'}] = \
                            error_estimate
                        self.error_full_da.loc[{'qois': i, 'prediction_interval': 'upper'}] = \
                            error_estimate + prediction_interval

                    else:
                        # directly assign the error estimate to the data array
                        self.error_da[{'qois': i}] = error_estimate
                        # self.error_full_da = self.error_da

            elif self.cfgem['method'] == "MLP":

                # use the MLP model for inferece
                # error_estimate = neural_networks.infer_mlp_model(scenarios_extrapolation_da.data, self.model_list[i])

                if self.interval_flag:
                    # assign the inferred errors to the data array
                    self.error_da[{'qois': i // 2, 'interval': i % 2}] = error_estimate
                else:
                    # directly assign the error estimate to the data array
                    self.error_da[{'qois': i}] = error_estimate

                # self.error_full_da = self.error_da

            elif self.cfgem['method'] == "IPM":

                # use the IPM error model for inference
                upper_bound, lower_bound = self.infer_ipm_model(scenarios_extrapolation_da.data, self.model_list[i])

                if self.interval_flag:
                    # the upper bound is the analogon to "error_estimate + prediction_interval"
                    self.error_da[{'qois': i // 2, 'interval': i % 2}] = upper_bound

                    # assign the inferred errors to the full data array
                    self.error_full_da[{'qois': i, 'interval': i % 2, 'prediction_interval': 0}] = lower_bound
                    self.error_full_da[{'qois': i, 'interval': i % 2, 'prediction_interval': 2}] = upper_bound

                else:
                    # the lower bound is the analogon to "error_estimate - prediction_interval"
                    self.error_da.loc[{'qois': i, 'interval': 'left'}] = lower_bound

                    # the upper bound is the analogon to "error_estimate + prediction_interval"
                    self.error_da.loc[{'qois': i, 'interval': 'right'}] = upper_bound

                    # assign the inferred errors to the full data array
                    self.error_full_da.loc[{'qois': i, 'prediction_interval': 'lower'}] = lower_bound
                    self.error_full_da.loc[{'qois': i, 'prediction_interval': 'upper'}] = upper_bound

            else:
                raise ValueError("this error modeling technique is not available")

        self.error_da['qois'] = self.qoi_name_list
        self.error_full_da['qois'] = self.qoi_name_list

        return self.error_da

    @staticmethod
    def train_ipm_model(x, y, degree):
        """
        This function trains an Interval Predictor Model (IPM).

        It uses the PyIPM package. The theory can be found, e.g., in [1] and [2].

        Literature:
        [1] E Patelli, M Broggi, S Tolo, J Sadeghi, Cossan Software: A Multidisciplinary And Collaborative Software For
        Uncertainty Quantification, UNCECOMP 2017, At Rhodes Island, Greece, 2nd ECCOMAS Thematic Conference on
        Uncertainty Quantification in Computational Sciences and Engineering, June 2017.
        [2] L. G. Crespo, S. P. Kenny, D. P. Giesy, Y, R. B. Norman and S. R. Blattnig, „Application of
        Interval Predictor Models to Space Radiation Shielding,“ in 18th AIAA Non-Deterministic
        Approaches Conference, AIAA SciTech Forum, 2016.

        :param np.ndarray x: inputs
        :param np.ndarray y: outputs
        :param int degree: degree of the polynomial
        :return: trained interval predictor model
        """

        model = IPM(polynomial_degree=degree)
        model.fit(x, y)

        return model

    @staticmethod
    def infer_ipm_model(x, model):
        """
        This function performs inference using an Interval Predictor Model (IPM).

        It uses the PyIPM package.

        :param np.ndarray x: inputs
        :param model: IPM model
        :return tuple upper_bound, lower_bound: upper and lower interval boundaries
        """

        # an interval predictor model (IPM) predicts an upper and lower interval boundary
        upper_bound, lower_bound = model.predict(x)

        # model_reliability = model.get_model_reliability()

        return upper_bound, lower_bound
