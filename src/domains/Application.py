"""
This module is responsible for the model application process.

It includes one class for the model application. See details in its own documentation.

Contact person: Stefan Riedmaier
Creation date: 20.04.2020
Python version: 3.8
"""

# -- IMPORTS -----------------------------------------------------------------------------------------------------------
# -- built-in imports --

# -- third-party imports --
import xarray as xr

# -- custom imports --
from src.blocks.Scenarios import Scenarios
from src.blocks.Simulator import Simulator
from src.blocks.Assessment import Assessment
from src.blocks.ErrorIntegration import ErrorIntegration
from src.blocks.DecisionMaking import DecisionMaking


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Application:
    """
    This class is responsible for the model application process.

    It includes a main method called "process". See details in its own documentation.
    """

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """
        self.config = config
        domain = 'application'

        # -- INSTANTIATE VVUQ OBJECTS ----------------------------------------------------------------------------------
        self.scenarios_model = Scenarios(config, domain, instance='simulator')
        self.simulator = Simulator(config, domain, instance='simulator')
        self.assessment_model = Assessment(config, domain, instance='simulator')
        self.error_integration = ErrorIntegration(config, domain)
        self.decision_making_model = DecisionMaking(config, domain)

        if self.config['cross_domain']['experiment']['application']:
            self.scenarios_system = Scenarios(config, domain, instance='experiment')
            self.experiment = Simulator(config, domain, instance='experiment')
            self.assessment_system = Assessment(config, domain, instance='experiment')
            self.decision_making_system = DecisionMaking(config, domain)

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.qois_kpi_system_da = xr.DataArray(None)
        self.qois_kpi_model_da = xr.DataArray(None)
        self.error_validation_da = xr.DataArray(None)
        self.qois_kpi_system_estimated_da = xr.DataArray(None)
        self.decision_system_estimated_da = xr.DataArray(None)
        self.decision_system_da = xr.DataArray(None)
        self.decision_model_da = xr.DataArray(None)

    def process(self, error_model_validation, numerical_uncertainty):
        """
        This method runs through each step of the model application process.

        It contains the following steps:
        1.1) Generation of application scenarios for the simulation model.
        1.2) Execution of the generated scenarios.
        1.3) Assessment of the model responses in the executed scenarios.
        2) Inference of the error models to predict the simulation errors in the application scenarios.
        3) Integration of the inferred errors to the nominal model responses in the application scenarios.
        4) Decision making by comparison with pass/fail criteria from the application.

        In case the VVUQ methodology itself shall be validated, it is important to have the Ground Truth (GT) values
        from the system also in the application domain. Then, the process continues:
        5.1) Generation of application scenarios for the system.
        5.2) Execution of the generated scenarios.
        5.3) Assessment of the system responses in the executed scenarios.
        6) Comparison of nominal model responses with the pass/fail criteria.
        7) Comparison of the system responses (GT) with the pass/fail criteria.

        The interfaces are based on xarrays, recognizable by the "_da"-endings of the variables.

        :return: array of estimated system reponses via the VVUQ methodology
        :rtype: xr.DataArray
        """

        # generate the scenario array for the simulation
        scenarios_model_da, space_scenarios_model_da = self.scenarios_model.generate_scenarios()

        # run the simulations
        (quantities_ts_model_da, scenarios_model_da) =\
            self.simulator.run_simulation_process(scenarios_model_da, space_scenarios_model_da)

        # post-processing to assess the responses
        (self.qois_kpi_model_da, scenarios_model_da) = self.assessment_model.run_process(quantities_ts_model_da,
                                                                                         scenarios_model_da)

        # inference with the trained error models
        self.error_validation_da = error_model_validation.infer_models(space_scenarios_model_da)

        # estimate the system response
        self.qois_kpi_system_estimated_da = self.error_integration.error_integration(
            self.qois_kpi_model_da, self.error_validation_da, numerical_uncertainty)

        # compare estimated system responses with pass/fail criteria
        self.decision_system_estimated_da, _ = self.decision_making_model.check_regulation(
            self.qois_kpi_system_estimated_da)

        # compare model responses before uncertainty expansion with pass/fail criteria
        self.decision_model_da, _ = self.decision_making_model.check_regulation(
            self.qois_kpi_model_da)

        if self.config['cross_domain']['experiment']['application']:
            # generate the scenario array for the experiment
            scenarios_system_da, space_scenarios_system_da =\
                self.scenarios_system.generate_scenarios(scenarios_model_da, space_scenarios_model_da)

            # run the experiments
            (responses_system_da, scenarios_system_da) =\
                self.experiment.run_simulation_process(scenarios_system_da, space_scenarios_system_da)

            # post-processing to assess the responses
            self.qois_kpi_system_da, scenarios_system_da = self.assessment_system.run_process(responses_system_da,
                                                                                              scenarios_system_da)

            # calculate validation metrics at application points
            # self.metric_da = Metric.metric.calculate_metric(self.kpi_model_da, kpi_system_da)

            # compare the actual system responses (GT) with pass/fail criteria
            self.decision_system_da, _ = self.decision_making_system.check_regulation(self.qois_kpi_system_da)

        return self.decision_system_estimated_da
