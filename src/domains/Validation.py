"""
This module is responsible for the model validation process.

It includes one class for the model validation. See details in its own documentation.

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
from src.blocks.Metric import Metric
from src.blocks.DecisionMaking import DecisionMaking
from src.blocks.ErrorModel import ErrorModel


# -- CLASSES -----------------------------------------------------------------------------------------------------------
class Validation:
    """
    This class is responsible for the model validation process.

    It includes a main method called "process". See details in its own documentation.
    """

    def __init__(self, config):
        """
        This method initializes a new class instance.

        :param dict config: configuration dictionary
        """

        domain = 'validation'

        # -- INSTANTIATE VVUQ OBJECTS ----------------------------------------------------------------------------------
        self.scenarios_system = Scenarios(config, domain, instance='experiment')
        self.experiment = Simulator(config, domain, instance='experiment')
        self.assessment_system = Assessment(config, domain, instance='experiment')

        self.scenarios_model = Scenarios(config, domain, instance='simulator')
        self.simulator = Simulator(config, domain, instance='simulator')
        self.assessment_model = Assessment(config, domain, instance='simulator')

        self.metric = Metric(config, domain)
        self.decision_making = DecisionMaking(config, domain)
        self.error_model = ErrorModel(config, domain)

        self.decision_making_model = DecisionMaking(config, 'application')
        self.decision_making_system = DecisionMaking(config, 'application')

        # -- INSTANTIATE FURTHER INSTANCE ATTRIBUTES -------------------------------------------------------------------
        self.metric_da = xr.DataArray(None)
        self.qois_kpi_system_da = xr.DataArray(None)
        self.qois_kpi_model_da = xr.DataArray(None)
        self.decision_da = xr.DataArray(None)
        self.decision_model_da = xr.DataArray(None)
        self.decision_system_da = xr.DataArray(None)

    def process(self):
        """
        This method runs through each step of the model validation process.

        It contains the following steps:
        1.1) Generation of validation scenarios for the system.
        1.2) Execution of the generated scenarios.
        1.3) Assessment of the system responses in the executed scenarios.
        2.1) Generation of validation scenarios for the simulation model.
        2.2) Execution of the generated scenarios.
        2.3) Assessment of the model responses in the executed scenarios.
        3) Calculation of a validation metric to compare the system and model responses.
        4) Comparison of the validation results against model accuracy requirements (tolerances).
        5) Learning of an error model to aggregate the validation results.

        The interfaces are based on xarrays, recognizable by the "_da"-endings of the variables.

        :return ErrorModel self.error_model: instance of the ErrorModel class with a trained error model
        """

        # generate the scenario array for the experiment
        scenarios_system_da, space_scenarios_system_da = self.scenarios_system.generate_scenarios()

        # run the reference simulation
        (quantities_ts_system_da, scenarios_system_da) =\
            self.experiment.run_simulation_process(scenarios_system_da, space_scenarios_system_da)

        # post-processing to assess the responses
        (self.qois_kpi_system_da, scenarios_system_da) = self.assessment_system.run_process(quantities_ts_system_da,
                                                                                            scenarios_system_da)

        # generate the scenario array for the simulation
        scenarios_model_da, space_scenarios_model_da =\
            self.scenarios_model.generate_scenarios(scenarios_system_da, space_scenarios_system_da)

        # run the simulations
        (quantities_ts_model_da, scenarios_model_da) =\
            self.simulator.run_simulation_process(scenarios_model_da, space_scenarios_model_da)

        # post-processing to assess the responses
        (self.qois_kpi_model_da, scenarios_model_da) = self.assessment_model.run_process(quantities_ts_model_da,
                                                                                         scenarios_model_da)

        # calculate validation metrics
        self.metric_da = self.metric.calculate_metric(self.qois_kpi_model_da, self.qois_kpi_system_da)
        
        # compare validation metrics with model accuracy requirements
        self.decision_da, _ = self.decision_making.check_tolerances(self.metric_da, self.qois_kpi_model_da,
                                                                    self.qois_kpi_system_da)

        # learn an error model of the validation metrics
        self.error_model.train_models(space_scenarios_model_da, self.metric_da)

        # also compare assessment results of the model against the safety regulation
        self.decision_model_da, _ = self.decision_making_model.check_regulation(self.qois_kpi_model_da)

        # also compare assessment results of the system against the safety regulation
        self.decision_system_da, _ = self.decision_making_system.check_regulation(self.qois_kpi_system_da)

        return self.error_model
