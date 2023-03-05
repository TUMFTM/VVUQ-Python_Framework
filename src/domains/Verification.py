"""
This module is responsible for the model verification process.

It includes one class for the model verification. See details in its own documentation.

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
from src.blocks.Scenarios import Scenarios
from src.blocks.Simulator import Simulator
from src.blocks.Assessment import Assessment


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Verification:

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """

        self.config = config
        domain = 'verification'

        # -- PARSE CONFIG ----------------------------------------------------------------------------------------------
        # create dict pointers to the config subdicts as short-cuts
        self.cfgdi = self.config[domain]['discretization']

        # -- INSTANTIATE VVUQ OBJECTS ----------------------------------------------------------------------------------
        self.scenarios = Scenarios(config, domain, instance='simulator')
        self.simulator = Simulator(config, domain, instance='simulator')
        self.assessment = Assessment(config, domain, instance='simulator')

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.discretization_error_da = xr.DataArray(None)
        self.numerical_uncertainty_da = xr.DataArray(None)

    def process(self):
        """
        This method runs through each step of the model verification process.

        It contains the following steps:
        1.1) Generation of verification scenarios for the simulation model.
        1.2) Execution of the generated scenarios.
        1.3) Assessment of the model responses in the executed scenarios.
        2) Richardson extrapolation to determine the discretization error.
        3) Application of the Grid Convergence Index to convert the error to a numerical uncertainty.

        The interfaces are based on xarrays, recognizable by the "_da"-endings of the variables.

        :return: array of numerical uncertainties
        :rtype: xr.DataArray
        """

        if self.cfgdi['discretization_method'] == 'Richardson':

            # generate the scenario array for the simulation
            scenarios_model_da, space_scenarios_model_da = self.scenarios.generate_scenarios()

            # run the simulations
            (quantities_ts_model_da, scenarios_model_da) =\
                self.simulator.run_simulation_process(scenarios_model_da, space_scenarios_model_da)

            # post-processing to assess the responses
            (qois_kpi_model_da, scenarios_model_da) = self.assessment.run_process(quantities_ts_model_da,
                                                                                  scenarios_model_da)

            # Richardson Extrapolation to determine the discretization error
            self.discretization_error_da = self.richardson_extrapolation(scenarios_model_da, qois_kpi_model_da)

            if self.cfgdi['discretization_uncertainty'] == 'GCI':
                # Grid Convergence Index to convert the error to a numerical uncertainty
                self.numerical_uncertainty_da = self.gci_uncertainty(self.discretization_error_da)

            else:
                self.numerical_uncertainty_da = self.discretization_error_da

        return self.numerical_uncertainty_da

    @staticmethod
    def richardson_extrapolation(scenarios_model_da, qois_kpi_model_da):
        """
        This function calculates the discretization error of a simulation model via Richardson Extrapolation.

        It requires simulations with three step sizes of equal ratio.

        The theory can be found, e.g., in [1, Sec. 2.1.1] and [2, Ch. 8.4.2.1].

        Literature:
        [1] S. Sankararaman and S. Mahadevan, „Integration of model verification, validation, and
        calibration for uncertainty quantification in engineering systems,“ Reliability Engineering
        & System Safety, vol. 138, pp. 194–209, 2015.
        [2] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
        Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.

        :param xr.DataArray scenarios_model_da: array with input scenarios
        :param xr.DataArray qois_kpi_model_da: array with output kpis
        :return: array of numerical discretization errors
        :rtype: xr.DataArray
        """

        step_sizes = scenarios_model_da.loc[{'parameters': '$stepsize'}].data

        # check if we have three step sizes
        if step_sizes.shape != (3,):
            raise IndexError("three step sizes required for Richardson extrapolation.")

        idx = np.argsort(step_sizes)

        h_fine = step_sizes[idx[0]]
        h_medium = step_sizes[idx[1]]
        h_coarse = step_sizes[idx[2]]

        kpi_fine_da = qois_kpi_model_da[{'space_samples': idx[0]}]
        kpi_medium_da = qois_kpi_model_da[{'space_samples': idx[1]}]
        kpi_coarse_da = qois_kpi_model_da[{'space_samples': idx[2]}]

        # calculate the grid refinement factor r as ratio between the step sizes
        r = h_coarse / h_medium
        r2 = h_medium / h_fine

        # check if both ratios of step sizes are unequal (floating point comparison)
        if abs(r - r2) > 1e-6:
            raise ValueError("Richardson extrapolation with different grid refinement factor currenty not implemented.")

        # calculate the observed order of accuracy
        p_da = np.log(np.abs((kpi_coarse_da - kpi_medium_da) / (kpi_medium_da - kpi_fine_da))) / np.log(r)

        # estimate the exact solution
        kpi_exact_da = kpi_fine_da + ((kpi_fine_da - kpi_medium_da) / (r ** p_da - 1))

        # if the medium and fine results are equal, "p" will be nan due to a division by zero, and thus also kpi_exact
        # correct kpi_exact to the equal result
        kpi_exact_da[np.isnan(kpi_exact_da)] = kpi_fine_da[np.isnan(kpi_exact_da)]

        # calculate the discretization error
        discretization_error_da = kpi_coarse_da - kpi_exact_da

        return discretization_error_da

    def gci_uncertainty(self, discretization_error_da):
        """
        This functions calculates the numerical discretization uncertainty based on the Grid Convergence Index (GCI).

        The theory can be found in [1, Ch. 8.6].

        Literature:
        [1] W. L. Oberkampf and C. J. Roy, Verification and Validation in Scientific Computing,
        Cambridge, Cambridge University Press, 2010, ISBN: 9780511760396.

        :param: xr.DataArray: array of discretization errors
        :return: array of numerical discretization uncertainties
        :rtype: xr.DataArray
        """

        numerical_uncertainty_da = discretization_error_da * self.cfgdi['GCI_safety_factor']

        return numerical_uncertainty_da
